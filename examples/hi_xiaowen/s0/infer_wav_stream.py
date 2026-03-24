#!/usr/bin/env python3

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import infer_wav as iw
from torch2lite import fbank_pybind
from torch2lite.ctc_decoder_pybind import StreamingCTCDecoder as CppStreamingCTCDecoder
from torch2lite.tflite_quant_utils import dequantize_from_detail, quantize_to_detail


def parse_args():
    parser = argparse.ArgumentParser(
        description="对单个 wav 做流式唤醒推理，按 chunk 模拟在线输入。"
    )
    parser.add_argument("--wav", required=True, help="输入 wav 路径")
    parser.add_argument(
        "--model",
        default=iw.DEFAULT_MODEL_ALIAS,
        help="模型别名，例如 s3 / s1 / v2 / top20 / distill199",
    )
    parser.add_argument("--checkpoint", default="", help="显式指定 checkpoint/.zip 路径")
    parser.add_argument("--model_dir", default="", help="显式指定实验目录，例如 exp/xxx")
    parser.add_argument(
        "--checkpoint_name",
        default="",
        help="配合 --model_dir 使用，例如 399.pt / avg_30.pt / final.pt",
    )
    parser.add_argument("--config", default="", help="显式指定 config.yaml")
    parser.add_argument("--dict_dir", default="", help="显式指定 dict 目录")
    parser.add_argument("--stats_dir", default="", help="显式指定 stats 所在目录")
    parser.add_argument(
        "--keywords",
        default=iw.DEFAULT_KEYWORDS,
        help="关键词列表，逗号分隔，例如 嗨小问,你好问问",
    )
    parser.add_argument(
        "--threshold_map",
        default="",
        help='手动阈值覆盖，例如 "嗨小问=0.272,你好问问=0.016"',
    )
    parser.add_argument(
        "--target_fa_per_hour",
        type=float,
        default=1.0,
        help="从 stats 文件挑选阈值时优先满足的 FA/h 上限",
    )
    parser.add_argument(
        "--pick_mode",
        choices=["legacy", "recall", "robust"],
        default="legacy",
        help="阈值挑选策略",
    )
    parser.add_argument("--frr_eps", type=float, default=0.001, help="robust 模式下 FRR 容差")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id，默认 -1 表示 CPU")
    parser.add_argument("--disable_threshold", action="store_true", help="只输出原始候选结果，不做最终阈值判定")
    parser.add_argument("--chunk_ms", type=float, default=300.0, help="每次送入流式推理的音频 chunk 时长")
    parser.add_argument("--score_beam_size", type=int, default=3, help="逐帧 token 初筛 beam size")
    parser.add_argument("--path_beam_size", type=int, default=20, help="prefix beam size")
    parser.add_argument("--min_frames", type=int, default=5, help="关键词最短帧数")
    parser.add_argument("--max_frames", type=int, default=250, help="关键词最长帧数")
    parser.add_argument("--interval_frames", type=int, default=50, help="两次连续触发的最小间隔帧数")
    parser.add_argument("--indent", type=int, default=2, help="JSON 输出缩进空格数")
    parser.add_argument("--use_cpp_decoder", action="store_true", help="使用 C++ pybind 后处理（beam search + keyword detection）")
    return parser.parse_args()


def streaming_ctc_prefix_beam_search_step(
    frame_index: int,
    probs: torch.Tensor,
    cur_hyps,
    keywords_idxset,
    score_beam_size: int,
):
    next_hyps = defaultdict(lambda: (0.0, 0.0, []))

    top_k_probs, top_k_index = probs.topk(score_beam_size)
    filter_probs = []
    filter_index = []
    for prob, idx in zip(top_k_probs.tolist(), top_k_index.tolist()):
        if keywords_idxset is not None:
            if prob > 0.05 and idx in keywords_idxset:
                filter_probs.append(prob)
                filter_index.append(idx)
        else:
            if prob > 0.05:
                filter_probs.append(prob)
                filter_index.append(idx)

    if not filter_index:
        return cur_hyps

    for token_id in filter_index:
        token_prob = probs[token_id].item()
        for prefix, (pb, pnb, cur_nodes) in cur_hyps:
            last = prefix[-1] if prefix else None
            if token_id == 0:
                n_pb, n_pnb, nodes = next_hyps[prefix]
                n_pb = n_pb + pb * token_prob + pnb * token_prob
                next_hyps[prefix] = (n_pb, n_pnb, cur_nodes.copy())
            elif token_id == last:
                if not math.isclose(pnb, 0.0, abs_tol=1e-6):
                    n_pb, n_pnb, nodes = next_hyps[prefix]
                    n_pnb = n_pnb + pnb * token_prob
                    nodes = cur_nodes.copy()
                    if token_prob > nodes[-1]["prob"]:
                        nodes[-1]["prob"] = token_prob
                        nodes[-1]["frame"] = frame_index
                    next_hyps[prefix] = (n_pb, n_pnb, nodes)

                if not math.isclose(pb, 0.0, abs_tol=1e-6):
                    n_prefix = prefix + (token_id,)
                    n_pb, n_pnb, nodes = next_hyps[n_prefix]
                    n_pnb = n_pnb + pb * token_prob
                    nodes = cur_nodes.copy()
                    nodes.append(dict(token=token_id, frame=frame_index, prob=token_prob))
                    next_hyps[n_prefix] = (n_pb, n_pnb, nodes)
            else:
                n_prefix = prefix + (token_id,)
                n_pb, n_pnb, nodes = next_hyps[n_prefix]
                if nodes:
                    if token_prob > nodes[-1]["prob"]:
                        nodes.pop()
                        nodes.append(dict(token=token_id, frame=frame_index, prob=token_prob))
                else:
                    nodes = cur_nodes.copy()
                    nodes.append(dict(token=token_id, frame=frame_index, prob=token_prob))
                n_pnb = n_pnb + pb * token_prob + pnb * token_prob
                next_hyps[n_prefix] = (n_pb, n_pnb, nodes)

    return sorted(next_hyps.items(), key=lambda x: (x[1][0] + x[1][1]), reverse=True)


class StreamingKeywordSpotter:
    def __init__(
        self,
        configs: Dict,
        model,
        device: torch.device,
        model_type: str,
        keywords: List[str],
        keywords_token,
        keywords_idxset,
        threshold_map: Dict[str, Optional[float]],
        score_beam_size: int,
        path_beam_size: int,
        min_frames: int,
        max_frames: int,
        interval_frames: int,
        disable_threshold: bool,
        use_cpp_decoder: bool = False,
    ):
        dataset_conf = configs["dataset_conf"]
        fbank_conf = dataset_conf["fbank_conf"]
        self.model = model
        self.device = device
        self.model_type = model_type
        self.sample_rate = int(dataset_conf.get("resample_conf", {}).get("resample_rate", 16000))
        self.frame_length = float(fbank_conf.get("frame_length", 25))
        self.frame_shift = float(fbank_conf.get("frame_shift", 10))
        self.num_mel_bins = int(fbank_conf.get("num_mel_bins", 80))
        self.downsampling = int(dataset_conf.get("frame_skip", 1))
        self.context_expansion = bool(dataset_conf.get("context_expansion", False))
        self.left_context = int(dataset_conf.get("context_expansion_conf", {}).get("left", 0))
        self.right_context = int(dataset_conf.get("context_expansion_conf", {}).get("right", 0))
        self.resolution = self.frame_shift / 1000.0
        self.fbank_stream_extractor = fbank_pybind.StreamingFbankExtractor(
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=self.sample_rate,
        )

        self.keywords = keywords
        self.keywords_token = keywords_token
        self.keywords_idxset = keywords_idxset
        self.threshold_map = threshold_map
        self.score_beam_size = score_beam_size
        self.path_beam_size = path_beam_size
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.interval_frames = interval_frames
        self.disable_threshold = disable_threshold

        self.feature_remained = None
        self.feats_ctx_offset = 0
        self.in_cache = torch.zeros(0, 0, 0, dtype=torch.float32, device=device)
        self.tflite_feat_input_detail = None
        self.tflite_cache_input_detail = None
        self.tflite_logits_output_detail = None
        self.tflite_cache_output_detail = None
        self.tflite_chunk_frames = None
        if model_type == "tflite":
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            if len(input_details) != 2 or len(output_details) != 2:
                raise ValueError(
                    f"流式 .tflite 期望 2 输入 2 输出（feats/cache -> logits/cache），收到 {len(input_details)} 输入 {len(output_details)} 输出。"
                )
            self.tflite_feat_input_detail = next((item for item in input_details if len(item["shape"]) == 3), None)
            self.tflite_cache_input_detail = next((item for item in input_details if len(item["shape"]) == 4), None)
            self.tflite_logits_output_detail = next((item for item in output_details if len(item["shape"]) == 3), None)
            self.tflite_cache_output_detail = next((item for item in output_details if len(item["shape"]) == 4), None)
            if any(item is None for item in [
                self.tflite_feat_input_detail,
                self.tflite_cache_input_detail,
                self.tflite_logits_output_detail,
                self.tflite_cache_output_detail,
            ]):
                raise ValueError("无法从 .tflite 模型中识别 streaming feats/cache 输入输出。")
            self.model.allocate_tensors()
            feat_shape = tuple(int(dim) for dim in self.tflite_feat_input_detail["shape"])
            cache_shape = tuple(int(dim) for dim in self.tflite_cache_input_detail["shape"])
            self.tflite_chunk_frames = int(feat_shape[1])
            self.in_cache = np.zeros(cache_shape, dtype=np.float32)
        self.cur_hyps = [(tuple(), (1.0, 0.0, []))]
        self.total_frames = 0
        self.last_active_pos = -1
        self.best_decode = {
            "candidate_keyword": None,
            "candidate_score": None,
            "start_frame": None,
            "end_frame": None,
        }

        self.use_cpp_decoder = use_cpp_decoder
        if use_cpp_decoder:
            self._cpp_decoder = CppStreamingCTCDecoder(
                score_beam_size=score_beam_size,
                path_beam_size=path_beam_size,
                min_frames=min_frames,
                max_frames=max_frames,
                interval_frames=interval_frames,
            )
            self._cpp_decoder.set_keywords(keywords_token, keywords_idxset)
            self._cpp_decoder.set_thresholds(threshold_map)

    def _update_best_decode(self, keyword, score, start_frame, end_frame):
        current = self.best_decode["candidate_score"]
        if current is None or score > current:
            self.best_decode = {
                "candidate_keyword": keyword,
                "candidate_score": score,
                "start_frame": start_frame,
                "end_frame": end_frame,
            }

    def accept_wave_chunk(self, chunk_wave: np.ndarray) -> Optional[torch.Tensor]:
        chunk_pcm = chunk_wave.astype(np.int16, copy=False)
        if chunk_pcm.size == 0:
            return None

        chunk_tensor = torch.from_numpy(chunk_pcm)
        feats = self.fbank_stream_extractor.accept_int16(
            chunk_tensor, num_samples=int(chunk_tensor.numel())
        )
        feat_len = int(feats.size(0))
        if feat_len == 0:
            return None

        # Kaldi reference path for future validation/debugging:
        # import torchaudio.compliance.kaldi as kaldi
        # ref_waveform = chunk_tensor.to(torch.float32).unsqueeze(0)
        # ref_feats = kaldi.fbank(
        #     ref_waveform,
        #     num_mel_bins=self.num_mel_bins,
        #     frame_length=self.frame_length,
        #     frame_shift=self.frame_shift,
        #     dither=0.0,
        #     energy_floor=0.0,
        #     sample_frequency=self.sample_rate,
        # )

        if self.context_expansion:
            if self.feature_remained is None:
                feats_pad = torch.nn.functional.pad(
                    feats.T, (self.left_context, 0), mode="replicate"
                ).T
            else:
                feats_pad = torch.cat((self.feature_remained, feats))

            ctx_frm = feats_pad.shape[0] - (self.right_context + self.right_context)
            if ctx_frm <= 0:
                self.feature_remained = feats_pad.clone()
                return None

            ctx_win = self.left_context + self.right_context + 1
            ctx_dim = feats.shape[1] * ctx_win
            feats_ctx = torch.zeros(ctx_frm, ctx_dim, dtype=torch.float32)
            for index in range(ctx_frm):
                feats_ctx[index] = torch.cat(tuple(feats_pad[index:index + ctx_win])).unsqueeze(0)

            remained = self.left_context + self.right_context
            self.feature_remained = feats_pad[-remained:].clone() if remained > 0 else None
            feats = feats_ctx

        if self.downsampling > 1:
            last_remainder = 0 if self.feats_ctx_offset == 0 else self.downsampling - self.feats_ctx_offset
            remainder = (feats.size(0) + last_remainder) % self.downsampling
            feats = feats[self.feats_ctx_offset::self.downsampling, :]
            self.feats_ctx_offset = remainder if remainder == 0 else self.downsampling - remainder

        return feats

    def _forward_model(self, feats: torch.Tensor) -> torch.Tensor:
        if self.model_type == "tflite":
            if self.tflite_chunk_frames != 1:
                raise ValueError(
                    f"当前 streaming .tflite 推理仅支持 chunk_frames=1，收到 {self.tflite_chunk_frames}"
                )
            probs_list = []
            for frame_index in range(feats.size(0)):
                frame = feats[frame_index:frame_index + 1].unsqueeze(0).cpu().numpy().astype(np.float32)
                self.model.set_tensor(
                    self.tflite_feat_input_detail["index"],
                    quantize_to_detail(frame, self.tflite_feat_input_detail),
                )
                self.model.set_tensor(
                    self.tflite_cache_input_detail["index"],
                    quantize_to_detail(self.in_cache, self.tflite_cache_input_detail),
                )
                self.model.invoke()
                logits = dequantize_from_detail(
                    self.model.get_tensor(self.tflite_logits_output_detail["index"]),
                    self.tflite_logits_output_detail,
                )
                self.in_cache = dequantize_from_detail(
                    self.model.get_tensor(self.tflite_cache_output_detail["index"]),
                    self.tflite_cache_output_detail,
                )
                probs_list.append(torch.from_numpy(logits).softmax(2)[0])
            return torch.cat(probs_list, dim=0).cpu()

        feats = feats.unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.model_type == "jit":
                logits, self.in_cache = self.model(feats, self.in_cache)
            else:
                logits, self.in_cache = self.model(feats, self.in_cache)
        return logits.softmax(2)[0].cpu()

    def _execute_detection(self, frame_index: int) -> Optional[Dict[str, object]]:
        hyps = [(item[0], item[1][0] + item[1][1], item[1][2]) for item in self.cur_hyps]
        for prefix_ids, _, prefix_nodes in hyps:
            for word in self.keywords:
                label = self.keywords_token[word]["token_id"]
                offset = iw.is_sublist(prefix_ids, label)
                if offset == -1:
                    continue

                score = 1.0
                start_frame = prefix_nodes[offset]["frame"]
                end_frame = prefix_nodes[offset + len(label) - 1]["frame"]
                for index in range(offset, offset + len(label)):
                    score *= prefix_nodes[index]["prob"]
                score = math.sqrt(score)
                duration = end_frame - start_frame
                self._update_best_decode(word, score, start_frame, end_frame)

                threshold = self.threshold_map.get(word)
                passed_threshold = self.disable_threshold or (
                    threshold is not None and score >= threshold
                )
                enough_interval = self.last_active_pos == -1 or (
                    end_frame - self.last_active_pos >= self.interval_frames
                )
                if (
                    passed_threshold
                    and self.min_frames <= duration <= self.max_frames
                    and enough_interval
                ):
                    self.last_active_pos = end_frame
                    return {
                        "candidate_keyword": word,
                        "candidate_score": score,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                    }
                return None
        return None

    def _normalize_cpp_result(self, cpp_result: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
        if cpp_result is None:
            return None
        return {
            "candidate_keyword": cpp_result["keyword"],
            "candidate_score": cpp_result["candidate_score"],
            "start_frame": cpp_result["start_frame"],
            "end_frame": cpp_result["end_frame"],
        }

    def _step_decoder(self, absolute_frame: int, prob: torch.Tensor) -> Optional[Dict[str, object]]:
        if self.use_cpp_decoder:
            self._cpp_decoder.advance_frame(absolute_frame, prob)
            return self._normalize_cpp_result(
                self._cpp_decoder.execute_detection(self.disable_threshold)
            )

        self.cur_hyps = streaming_ctc_prefix_beam_search_step(
            absolute_frame,
            prob,
            self.cur_hyps,
            self.keywords_idxset,
            self.score_beam_size,
        )[:self.path_beam_size]
        return self._execute_detection(absolute_frame)

    def _get_first_hyp_start_frame(self) -> int:
        if self.use_cpp_decoder:
            return self._cpp_decoder.get_first_hyp_start_frame()
        if self.cur_hyps and self.cur_hyps[0][0]:
            return int(self.cur_hyps[0][1][2][0]["frame"])
        return -1

    def forward_chunk(
        self,
        chunk_wave: np.ndarray,
        stop_on_activation: bool = True,
        reset_on_activation: bool = True,
    ) -> Optional[Dict[str, object]]:
        feats = self.accept_wave_chunk(chunk_wave)
        if feats is None or feats.size(0) < 1:
            return None

        probs = self._forward_model(feats)
        last_activation = None

        for local_frame_index, prob in enumerate(probs):
            absolute_frame = self.total_frames + local_frame_index * self.downsampling
            activation = self._step_decoder(absolute_frame, prob)
            if activation is None:
                continue
            last_activation = activation
            if reset_on_activation:
                self.reset_decode_state()
            if stop_on_activation:
                break

        self.total_frames += len(probs) * self.downsampling
        start = self._get_first_hyp_start_frame()
        if start >= 0 and (self.total_frames - start) > self.max_frames:
            self.reset_decode_state()

        return last_activation

    def reset_decode_state(self):
        if self.use_cpp_decoder:
            self._cpp_decoder.reset_beam_search()
            return
        self.cur_hyps = [(tuple(), (1.0, 0.0, []))]

    def get_best_decode_result(self) -> Dict[str, object]:
        if self.use_cpp_decoder:
            return self._cpp_decoder.get_best_decode_result()
        return dict(self.best_decode)


def compute_streaming_lookahead_sec(configs: Dict) -> float:
    dataset_conf = configs.get("dataset_conf", {})
    fbank_conf = dataset_conf.get("fbank_conf", {})
    frame_shift_ms = float(fbank_conf.get("frame_shift", 10))
    right_context = int(dataset_conf.get("context_expansion_conf", {}).get("right", 0))
    frame_skip = int(dataset_conf.get("frame_skip", 1))
    backbone_conf = configs.get("model", {}).get("backbone", {})
    num_layers = int(backbone_conf.get("num_layers", 0))
    right_order = int(backbone_conf.get("right_order", 0))
    right_stride = int(backbone_conf.get("right_stride", 1))

    frontend_lookahead_ms = right_context * frame_shift_ms
    backbone_lookahead_ms = num_layers * right_order * right_stride * frame_shift_ms * frame_skip
    return (frontend_lookahead_ms + backbone_lookahead_ms) / 1000.0


def collect_streaming_probs(streamer: StreamingKeywordSpotter, pcm: np.ndarray, chunk_ms: float) -> torch.Tensor:
    chunk_samples = max(1, int(chunk_ms / 1000.0 * streamer.sample_rate))
    probs_list = []
    for start in range(0, pcm.shape[0], chunk_samples):
        feats = streamer.accept_wave_chunk(pcm[start:start + chunk_samples])
        if feats is None or feats.size(0) < 1:
            continue
        probs_list.append(streamer._forward_model(feats))
    if not probs_list:
        return torch.zeros((0, 0), dtype=torch.float32)
    return torch.cat(probs_list, dim=0)


def collect_streaming_best_decode(
    streamer: StreamingKeywordSpotter, pcm: np.ndarray, chunk_ms: float
) -> Dict[str, object]:
    chunk_samples = max(1, int(chunk_ms / 1000.0 * streamer.sample_rate))
    for start in range(0, pcm.shape[0], chunk_samples):
        streamer.forward_chunk(
            pcm[start:start + chunk_samples],
            stop_on_activation=False,
            reset_on_activation=False,
        )
    return streamer.get_best_decode_result()


def main():
    args = parse_args()
    wav_path = iw.to_abs_path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(f"找不到 wav: {wav_path}")

    keywords = iw.parse_keywords_arg(args.keywords)
    if not keywords:
        raise ValueError("至少需要一个关键词")

    model_info = iw.resolve_model_paths(args)
    configs = iw.load_config(model_info["config"])
    model, device, model_type = iw.load_model(model_info["checkpoint"], configs, args.gpu)
    threshold_map = iw.load_threshold_map(args, model_info, keywords)
    keywords_token, keywords_idxset = iw.build_keyword_token_info(keywords, model_info["dict_dir"])

    streamer = StreamingKeywordSpotter(
        configs=configs,
        model=model,
        device=device,
        model_type=model_type,
        keywords=keywords,
        keywords_token=keywords_token,
        keywords_idxset=keywords_idxset,
        threshold_map=threshold_map,
        score_beam_size=args.score_beam_size,
        path_beam_size=args.path_beam_size,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        interval_frames=args.interval_frames,
        disable_threshold=args.disable_threshold,
        use_cpp_decoder=args.use_cpp_decoder,
    )

    waveform = iw.load_wav_and_resample(wav_path, streamer.sample_rate)
    waveform = waveform.squeeze(0).numpy()
    pcm = np.clip(np.round(waveform * (1 << 15)), -32768, 32767).astype(np.int16)
    chunk_samples = max(1, int(args.chunk_ms / 1000.0 * streamer.sample_rate))

    activated_decode = None
    for start in range(0, pcm.shape[0], chunk_samples):
        activated_decode = streamer.forward_chunk(pcm[start:start + chunk_samples])
        if activated_decode is not None:
            break

    decode_result = activated_decode if activated_decode is not None else streamer.get_best_decode_result()
    time_resolution_sec = iw.get_time_resolution_sec(configs)
    result = iw.format_result(
        wav_path=wav_path,
        model_info=model_info,
        threshold_map=threshold_map,
        decode_result=decode_result,
        time_resolution_sec=time_resolution_sec,
        disable_threshold=args.disable_threshold,
    )
    if activated_decode is not None:
        result["triggered"] = True
        result["keyword"] = activated_decode["candidate_keyword"]
        result["wake_time_sec"] = activated_decode["end_frame"] * time_resolution_sec

    result["mode"] = "streaming"
    result["chunk_ms"] = args.chunk_ms
    result["streaming_lookahead_sec"] = compute_streaming_lookahead_sec(configs)
    result["model_type"] = model_type
    result["use_cpp_decoder"] = args.use_cpp_decoder
    print(json.dumps(result, ensure_ascii=False, indent=args.indent))


if __name__ == "__main__":
    main()
