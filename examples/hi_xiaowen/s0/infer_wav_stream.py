#!/usr/bin/env python3

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

import infer_wav as iw
from torch2lite import fbank_pybind


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
    ):
        if model_type == "tflite":
            raise ValueError("当前流式推理仅支持 .pt / .zip，暂不支持 .tflite。")

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

        self.wave_remained = np.array([], dtype=np.int16)
        self.feature_remained = None
        self.feats_ctx_offset = 0
        self.in_cache = torch.zeros(0, 0, 0, dtype=torch.float32, device=device)
        self.cur_hyps = [(tuple(), (1.0, 0.0, []))]
        self.total_frames = 0
        self.last_active_pos = -1
        self.best_decode = {
            "candidate_keyword": None,
            "candidate_score": None,
            "start_frame": None,
            "end_frame": None,
        }

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
        frame_length_samples = int(self.frame_length / 1000.0 * self.sample_rate)
        frame_shift_samples = int(self.frame_shift / 1000.0 * self.sample_rate)
        min_samples = frame_length_samples + self.right_context * frame_shift_samples

        wave = np.concatenate([self.wave_remained, chunk_wave.astype(np.int16, copy=False)])
        if wave.size < min_samples:
            self.wave_remained = wave
            return None

        wave_tensor = torch.from_numpy(wave.astype(np.float32)).unsqueeze(0)
        feats = fbank_pybind.fbank(
            wave_tensor,
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=self.sample_rate,
        )
        feat_len = int(feats.size(0))
        if feat_len == 0:
            self.wave_remained = wave
            return None

        self.wave_remained = wave[feat_len * frame_shift_samples:]

        if self.context_expansion:
            if feat_len <= self.right_context:
                self.wave_remained = wave
                return None
            if self.feature_remained is None:
                feats_pad = torch.nn.functional.pad(
                    feats.T, (self.left_context, 0), mode="replicate"
                ).T
            else:
                feats_pad = torch.cat((self.feature_remained, feats))

            ctx_frm = feats_pad.shape[0] - (self.right_context + self.right_context)
            ctx_win = self.left_context + self.right_context + 1
            ctx_dim = feats.shape[1] * ctx_win
            feats_ctx = torch.zeros(ctx_frm, ctx_dim, dtype=torch.float32)
            for index in range(ctx_frm):
                feats_ctx[index] = torch.cat(tuple(feats_pad[index:index + ctx_win])).unsqueeze(0)

            self.feature_remained = feats[-(self.left_context + self.right_context):]
            feats = feats_ctx

        if self.downsampling > 1:
            last_remainder = 0 if self.feats_ctx_offset == 0 else self.downsampling - self.feats_ctx_offset
            remainder = (feats.size(0) + last_remainder) % self.downsampling
            feats = feats[self.feats_ctx_offset::self.downsampling, :]
            self.feats_ctx_offset = remainder if remainder == 0 else self.downsampling - remainder

        return feats

    def _forward_model(self, feats: torch.Tensor) -> torch.Tensor:
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

    def forward_chunk(self, chunk_wave: np.ndarray) -> Optional[Dict[str, object]]:
        feats = self.accept_wave_chunk(chunk_wave)
        if feats is None or feats.size(0) < 1:
            return None

        probs = self._forward_model(feats)
        activated_result = None
        for local_frame_index, prob in enumerate(probs):
            absolute_frame = self.total_frames + local_frame_index * self.downsampling
            self.cur_hyps = streaming_ctc_prefix_beam_search_step(
                absolute_frame,
                prob,
                self.cur_hyps,
                self.keywords_idxset,
                self.score_beam_size,
            )[:self.path_beam_size]
            activated_result = self._execute_detection(absolute_frame)
            if activated_result is not None:
                self.reset_decode_state()
                break

        self.total_frames += len(probs) * self.downsampling
        if self.cur_hyps and self.cur_hyps[0][0]:
            keyword_may_start = int(self.cur_hyps[0][1][2][0]["frame"])
            if (self.total_frames - keyword_may_start) > self.max_frames:
                self.reset_decode_state()
        return activated_result

    def reset_decode_state(self):
        self.cur_hyps = [(tuple(), (1.0, 0.0, []))]

    def get_best_decode_result(self) -> Dict[str, object]:
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
    print(json.dumps(result, ensure_ascii=False, indent=args.indent))


if __name__ == "__main__":
    main()
