#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio

import diagnose_wav as dw
import infer_wav as iw


warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _to_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu()


def _to_nested_list(tensor: torch.Tensor) -> List:
    return _to_cpu_float(tensor).tolist()


def _tensor_stats(tensor: torch.Tensor) -> Dict[str, object]:
    item = _to_cpu_float(tensor)
    flat = item.reshape(-1)
    if flat.numel() == 0:
        return {
            "shape": list(item.shape),
            "numel": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "nonzero_ratio": None,
        }

    quantiles = torch.quantile(flat, torch.tensor([0.01, 0.50, 0.99]))
    nonzero_ratio = float((flat != 0).float().mean())
    return {
        "shape": list(item.shape),
        "numel": int(flat.numel()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "p01": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "nonzero_ratio": nonzero_ratio,
    }


def _build_stage_summary(
    key: str,
    title: str,
    tensor: torch.Tensor,
    note: str,
) -> Dict[str, object]:
    stats = _tensor_stats(tensor)
    return {
        "key": key,
        "title": title,
        "shape": stats["shape"],
        "stats": stats,
        "note": note,
    }


def _build_matrix_payload(
    key: str,
    title: str,
    tensor: torch.Tensor,
    note: str,
    time_axis_sec: Optional[List[float]] = None,
    y_axis_values: Optional[List[float]] = None,
    y_axis_name: str = "dim",
) -> Dict[str, object]:
    matrix = _to_cpu_float(tensor)
    return {
        "key": key,
        "title": title,
        "shape": list(matrix.shape),
        "stats": _tensor_stats(matrix),
        "note": note,
        "matrix": matrix.tolist(),
        "time_axis_sec": time_axis_sec,
        "y_axis_values": y_axis_values,
        "y_axis_name": y_axis_name,
    }


def _extract_waveform_payload(waveform: torch.Tensor, sample_rate: int) -> Dict[str, object]:
    samples = _to_cpu_float(waveform[0])
    num_samples = int(samples.numel())
    times = [float(index / sample_rate) for index in range(num_samples)]
    return {
        "shape": list(waveform.shape),
        "sample_rate": int(sample_rate),
        "duration_sec": float(num_samples / sample_rate) if sample_rate > 0 else 0.0,
        "stats": _tensor_stats(samples),
        "times": times,
        "amplitude": samples.tolist(),
    }


def _compute_analysis_spectra(
    waveform: torch.Tensor,
    sample_rate: int,
    dataset_conf: Dict,
) -> Dict[str, object]:
    samples = _to_cpu_float(waveform[0])
    fbank_conf = dataset_conf.get("fbank_conf", {})
    frame_length_ms = float(fbank_conf.get("frame_length", 25.0))
    frame_shift_ms = float(fbank_conf.get("frame_shift", 10.0))
    num_mel_bins = int(fbank_conf.get("num_mel_bins", 80))
    win_length = max(1, int(sample_rate * frame_length_ms / 1000.0))
    hop_length = max(1, int(sample_rate * frame_shift_ms / 1000.0))
    n_fft = _next_power_of_two(win_length)
    window = torch.hann_window(win_length)

    stft = torch.stft(
        samples,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        return_complex=True,
    )
    magnitude = stft.abs()
    power = magnitude.pow(2)
    stft_db = 10.0 * torch.log10(power.clamp_min(1e-10))

    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=int(power.size(0)),
        f_min=0.0,
        f_max=float(sample_rate) / 2.0,
        n_mels=num_mel_bins,
        sample_rate=sample_rate,
        norm=None,
        mel_scale="htk",
    )
    mel_energy = torch.matmul(power.transpose(0, 1), mel_fb).transpose(0, 1)
    log_mel = torch.log(mel_energy.clamp_min(1e-10))

    frame_times = [float(index * hop_length / sample_rate) for index in range(int(power.size(1)))]
    freqs = [float(index * sample_rate / n_fft) for index in range(int(power.size(0)))]
    mel_bins = list(range(num_mel_bins))

    return {
        "frame_length_ms": frame_length_ms,
        "frame_shift_ms": frame_shift_ms,
        "n_fft": int(n_fft),
        "win_length": int(win_length),
        "hop_length": int(hop_length),
        "frame_times_sec": frame_times,
        "freq_hz": freqs,
        "mel_bins": mel_bins,
        "stft_db": _build_matrix_payload(
            key="stft_db",
            title="STFT 幅度谱(dB)",
            tensor=stft_db.transpose(0, 1),
            note="解释性视图：用当前帧长/帧移计算得到的时频能量。",
            time_axis_sec=frame_times,
            y_axis_values=freqs,
            y_axis_name="freq_hz",
        ),
        "mel_energy": _build_matrix_payload(
            key="mel_energy",
            title="Mel 能量",
            tensor=mel_energy.transpose(0, 1),
            note="把线性频率能量投影到 80 维 mel 滤波器组。",
            time_axis_sec=frame_times,
            y_axis_values=mel_bins,
            y_axis_name="mel_bin",
        ),
        "log_mel": _build_matrix_payload(
            key="log_mel",
            title="log-Mel",
            tensor=log_mel.transpose(0, 1),
            note="对 mel 能量取对数后的分析视图。",
            time_axis_sec=frame_times,
            y_axis_values=mel_bins,
            y_axis_name="mel_bin",
        ),
    }


def _context_windows(num_frames: int, left: int, right: int) -> List[List[int]]:
    valid_frames = max(0, num_frames - right)
    windows: List[List[int]] = []
    for frame_index in range(valid_frames):
        window = []
        for lag in range(-left, right + 1):
            source_index = frame_index + lag
            if source_index < 0:
                source_index = 0
            elif source_index >= num_frames:
                source_index = num_frames - 1
            window.append(int(source_index))
        windows.append(window)
    return windows


def _apply_context_expansion_with_details(
    feats: torch.Tensor,
    dataset_conf: Dict,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if not dataset_conf.get("context_expansion", False):
        return feats, {
            "enabled": False,
            "left": 0,
            "right": 0,
            "input_shape": list(feats.shape),
            "output_shape": list(feats.shape),
            "windows": [[int(index)] for index in range(int(feats.size(0)))],
        }

    left = int(dataset_conf.get("context_expansion_conf", {}).get("left", 0))
    right = int(dataset_conf.get("context_expansion_conf", {}).get("right", 0))
    expanded = iw.apply_context_expansion(feats, dataset_conf)
    windows = _context_windows(int(feats.size(0)), left, right)
    return expanded, {
        "enabled": True,
        "left": left,
        "right": right,
        "input_shape": list(feats.shape),
        "output_shape": list(expanded.shape),
        "windows": windows,
    }


def _apply_frame_skip_with_details(
    feats: torch.Tensor,
    dataset_conf: Dict,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    frame_skip = int(dataset_conf.get("frame_skip", 1))
    skipped = iw.apply_frame_skip(feats, dataset_conf)
    kept_indices = list(range(0, int(feats.size(0)), max(frame_skip, 1)))
    dropped_indices = [index for index in range(int(feats.size(0))) if index not in set(kept_indices)]
    return skipped, {
        "frame_skip": frame_skip,
        "input_shape": list(feats.shape),
        "output_shape": list(skipped.shape),
        "kept_input_indices": kept_indices,
        "dropped_input_indices": dropped_indices,
    }


def _manual_forward_layers(
    model,
    feats: torch.Tensor,
    device: torch.device,
    is_jit: bool,
) -> Tuple[torch.Tensor, List[Dict[str, object]]]:
    feats = feats.to(device)
    layers: List[Dict[str, object]] = []

    if is_jit or not hasattr(model, "backbone") or not hasattr(model.backbone, "fsmn"):
        with torch.no_grad():
            empty_cache = torch.zeros(0, 0, 0, dtype=torch.float32, device=device)
            if is_jit:
                logits, _ = model(feats, empty_cache)
            else:
                logits, _ = model(feats)
        probs = logits.softmax(dim=-1)
        layers.append(
            {
                "key": "logits",
                "title": "Logits",
                "note": "当前模型格式不支持分层抓取，只保留最终输出。",
                "tensor": _to_cpu_float(logits[0]),
            }
        )
        layers.append(
            {
                "key": "probs",
                "title": "Softmax 概率",
                "note": "对 logits 在 token 维做 softmax。",
                "tensor": _to_cpu_float(probs[0]),
            }
        )
        return _to_cpu_float(logits[0]), layers

    with torch.no_grad():
        current = feats
        layers.append(
            {
                "key": "model_input",
                "title": "模型输入",
                "note": "进入 KWSModel.forward 的 batch 特征。",
                "tensor": _to_cpu_float(current[0]),
            }
        )

        if model.global_cmvn is not None:
            current = model.global_cmvn(current)
            layers.append(
                {
                    "key": "cmvn",
                    "title": "Global CMVN",
                    "note": "按全局均值/方差做归一化。",
                    "tensor": _to_cpu_float(current[0]),
                }
            )

        current = model.preprocessing(current)
        layers.append(
            {
                "key": "preprocess",
                "title": "Preprocessing",
                "note": "当前配置是 none，因此输出与 CMVN 后一致。",
                "tensor": _to_cpu_float(current[0]),
            }
        )

        backbone = model.backbone
        x1, _ = backbone.in_linear1(current)
        layers.append(
            {
                "key": "in_linear1",
                "title": "Input Affine 1",
                "note": "400 -> input_affine_dim。",
                "tensor": _to_cpu_float(x1[0]),
            }
        )

        x2, _ = backbone.in_linear2(x1)
        layers.append(
            {
                "key": "in_linear2",
                "title": "Input Affine 2",
                "note": "input_affine_dim -> linear_dim。",
                "tensor": _to_cpu_float(x2[0]),
            }
        )

        x3, _ = backbone.relu(x2)
        layers.append(
            {
                "key": "input_relu",
                "title": "Input ReLU",
                "note": "进入 FSMN 堆叠前的激活输出。",
                "tensor": _to_cpu_float(x3[0]),
            }
        )

        x = x3
        caches = [
            torch.zeros(0, 0, 0, 0, dtype=torch.float32, device=device)
            for _ in range(len(backbone.fsmn))
        ]
        for layer_index, module in enumerate(backbone.fsmn):
            proj, _ = module[0]((x, caches[layer_index]))
            layers.append(
                {
                    "key": f"fsmn_{layer_index + 1}_proj",
                    "title": f"FSMN Layer {layer_index + 1} Proj",
                    "note": "linear_dim -> proj_dim。",
                    "tensor": _to_cpu_float(proj[0]),
                }
            )
            memory, caches[layer_index] = module[1]((proj, caches[layer_index]))
            layers.append(
                {
                    "key": f"fsmn_{layer_index + 1}_memory",
                    "title": f"FSMN Layer {layer_index + 1} Memory",
                    "note": "左右时序记忆卷积叠加后的输出。",
                    "tensor": _to_cpu_float(memory[0]),
                }
            )
            affine, _ = module[2]((memory, caches[layer_index]))
            layers.append(
                {
                    "key": f"fsmn_{layer_index + 1}_affine",
                    "title": f"FSMN Layer {layer_index + 1} Affine",
                    "note": "proj_dim -> linear_dim。",
                    "tensor": _to_cpu_float(affine[0]),
                }
            )
            relu, _ = module[3]((affine, caches[layer_index]))
            layers.append(
                {
                    "key": f"fsmn_{layer_index + 1}_relu",
                    "title": f"FSMN Layer {layer_index + 1} ReLU",
                    "note": "该层 block 的最终输出。",
                    "tensor": _to_cpu_float(relu[0]),
                }
            )
            x = relu

        if backbone.out_linear1 is not None:
            head1, _ = backbone.out_linear1((x, caches[-1] if caches else torch.zeros(0, 0, 0, 0, device=device)))
            layers.append(
                {
                    "key": "out_linear1",
                    "title": "Output Affine 1",
                    "note": "linear_dim -> output_affine_dim。",
                    "tensor": _to_cpu_float(head1[0]),
                }
            )
            head2, _ = backbone.out_linear2((head1, caches[-1] if caches else torch.zeros(0, 0, 0, 0, device=device)))
        else:
            head2, _ = backbone.out_linear2((x, caches[-1] if caches else torch.zeros(0, 0, 0, 0, device=device)))

        logits = head2
        probs = logits.softmax(dim=-1)
        layers.append(
            {
                "key": "logits",
                "title": "Logits",
                "note": "20 维 token 未归一化输出。",
                "tensor": _to_cpu_float(logits[0]),
            }
        )
        layers.append(
            {
                "key": "probs",
                "title": "Softmax 概率",
                "note": "逐帧 token 后验分布。",
                "tensor": _to_cpu_float(probs[0]),
            }
        )
        return _to_cpu_float(logits[0]), layers


def _format_trace_hyp(
    prefix: Tuple[int, ...],
    score: float,
    nodes: List[Dict],
    id2tok: Dict[int, str],
) -> Dict[str, object]:
    token_ids = [int(item) for item in prefix]
    text, tokens = dw.ids_to_text(token_ids, id2tok)
    return {
        "token_ids": token_ids,
        "tokens": tokens,
        "text": text,
        "score": float(score),
        "frames": [int(node["frame"]) for node in nodes],
    }


def _ctc_prefix_beam_search_with_trace(
    probs: torch.Tensor,
    id2tok: Dict[int, str],
    time_resolution_sec: float,
    keywords_tokenset: Optional[set],
    score_beam_size: int,
    path_beam_size: int,
    prob_threshold: float,
) -> Dict[str, object]:
    cur_hyps = [(tuple(), (1.0, 0.0, []))]
    trace: List[Dict[str, object]] = []

    for frame_index in range(int(probs.size(0))):
        frame_probs = probs[frame_index]
        next_hyps = defaultdict(lambda: (0.0, 0.0, []))
        top_k_probs, top_k_index = frame_probs.topk(score_beam_size)

        filtered = []
        for prob, token_id in zip(top_k_probs.tolist(), top_k_index.tolist()):
            if keywords_tokenset is not None:
                if prob > prob_threshold and token_id in keywords_tokenset:
                    filtered.append((int(token_id), float(prob)))
            elif prob > prob_threshold:
                filtered.append((int(token_id), float(prob)))

        if not filtered:
            trace.append(
                {
                    "frame_index": frame_index,
                    "time_sec": float(frame_index * time_resolution_sec),
                    "top_tokens": [],
                    "beam_after": [_format_trace_hyp(item[0], item[1][0] + item[1][1], item[1][2], id2tok) for item in cur_hyps[:path_beam_size]],
                }
            )
            continue

        for token_id, prob in filtered:
            for prefix, (pb, pnb, cur_nodes) in cur_hyps:
                last = prefix[-1] if prefix else None
                if token_id == 0:
                    n_pb, n_pnb, nodes = next_hyps[prefix]
                    n_pb = n_pb + pb * prob + pnb * prob
                    next_hyps[prefix] = (n_pb, n_pnb, cur_nodes.copy())
                    continue

                if token_id == last:
                    if not math.isclose(pnb, 0.0, abs_tol=1e-6):
                        n_pb, n_pnb, nodes = next_hyps[prefix]
                        n_pnb = n_pnb + pnb * prob
                        nodes = cur_nodes.copy()
                        if nodes and prob > nodes[-1]["prob"]:
                            nodes[-1]["prob"] = prob
                            nodes[-1]["frame"] = frame_index
                        next_hyps[prefix] = (n_pb, n_pnb, nodes)
                    if not math.isclose(pb, 0.0, abs_tol=1e-6):
                        n_prefix = prefix + (token_id,)
                        n_pb, n_pnb, nodes = next_hyps[n_prefix]
                        n_pnb = n_pnb + pb * prob
                        nodes = cur_nodes.copy()
                        nodes.append({"token": token_id, "frame": frame_index, "prob": prob})
                        next_hyps[n_prefix] = (n_pb, n_pnb, nodes)
                    continue

                n_prefix = prefix + (token_id,)
                n_pb, n_pnb, nodes = next_hyps[n_prefix]
                nodes = cur_nodes.copy()
                nodes.append({"token": token_id, "frame": frame_index, "prob": prob})
                n_pnb = n_pnb + pb * prob + pnb * prob
                next_hyps[n_prefix] = (n_pb, n_pnb, nodes)

        sorted_hyps = sorted(
            next_hyps.items(),
            key=lambda item: item[1][0] + item[1][1],
            reverse=True,
        )
        cur_hyps = sorted_hyps[:path_beam_size]
        trace.append(
            {
                "frame_index": frame_index,
                "time_sec": float(frame_index * time_resolution_sec),
                "top_tokens": [
                    {
                        "token_id": token_id,
                        "token": id2tok.get(token_id, f"<unk:{token_id}>"),
                        "prob": prob,
                    }
                    for token_id, prob in filtered
                ],
                "beam_after": [
                    _format_trace_hyp(prefix, scores[0] + scores[1], nodes, id2tok)
                    for prefix, scores, nodes in [
                        (item[0], item[1], item[1][2]) for item in cur_hyps
                    ]
                ],
            }
        )

    final_hyps = [
        _format_trace_hyp(prefix, scores[0] + scores[1], scores[2], id2tok)
        for prefix, scores in cur_hyps
    ]
    return {
        "frames": trace,
        "final_hyps": final_hyps,
        "score_beam_size": score_beam_size,
        "path_beam_size": path_beam_size,
        "prob_threshold": prob_threshold,
    }


def _build_keyword_token_strength(
    probs: torch.Tensor,
    resources: Dict,
    id2tok: Dict[int, str],
    frame_energy: torch.Tensor,
) -> List[Dict[str, object]]:
    rows = []
    threshold_map = resources["threshold_map"]
    keywords = resources["keywords"]
    keywords_token = resources["keywords_token"]

    keyword_hyps = dw.decode_keyword_beam_hyps(
        probs,
        keywords,
        keywords_token,
        resources["keywords_idxset"],
        id2tok,
        threshold_map,
    )
    keyword_diag = dw.summarize_keyword_diagnostics(
        keyword_hyps,
        keywords,
        keywords_token,
        id2tok,
        threshold_map,
    )
    diag_by_keyword = {item["keyword"]: item for item in keyword_diag}

    for keyword in keywords:
        diag = diag_by_keyword.get(keyword, {})
        matched_tokens = diag.get("matched_tokens", [])
        matched_frames = diag.get("matched_frames", [])
        token_probs = diag.get("token_probs", [])
        token_rows = []
        for token_index, token_id in enumerate(keywords_token[keyword]["token_id"]):
            token_probs_all = probs[:, token_id]
            peak_prob, peak_frame = torch.max(token_probs_all, dim=0)
            matched_frame = matched_frames[token_index] if token_index < len(matched_frames) else None
            matched_prob = token_probs[token_index] if token_index < len(token_probs) else None
            local_energy = None
            if matched_frame is not None:
                left = max(0, matched_frame - 1)
                right = min(int(frame_energy.numel()), matched_frame + 2)
                local_energy = float(frame_energy[left:right].mean())
            token_rows.append(
                {
                    "token_id": int(token_id),
                    "token": id2tok.get(int(token_id), f"<unk:{token_id}>"),
                    "peak_prob": float(peak_prob),
                    "peak_frame": int(peak_frame),
                    "peak_time_sec": float(int(peak_frame) * resources["time_resolution_sec"]),
                    "mean_prob": float(token_probs_all.mean()),
                    "matched_frame": int(matched_frame) if matched_frame is not None else None,
                    "matched_prob": float(matched_prob) if matched_prob is not None else None,
                    "matched_token": matched_tokens[token_index] if token_index < len(matched_tokens) else None,
                    "frame_energy": float(frame_energy[int(peak_frame)]),
                    "local_energy": local_energy,
                }
            )
        rows.append(
            {
                "keyword": keyword,
                "candidate_score": diag.get("candidate_score"),
                "threshold": diag.get("threshold"),
                "status": diag.get("status"),
                "token_rows": token_rows,
            }
        )
    return rows


def inspect_one_wav_pipeline(
    wav_path: Path,
    resources: Dict,
    id2tok: Dict[int, str],
    beam_size: int = 5,
) -> Dict[str, object]:
    dataset_conf = resources["configs"].get("dataset_conf", {})
    sample_rate = int(dataset_conf.get("resample_conf", {}).get("resample_rate", 16000))

    raw_waveform, raw_sr = torchaudio.load(str(wav_path))
    if raw_waveform.size(0) > 1:
        raw_waveform = raw_waveform.mean(dim=0, keepdim=True)
    waveform = iw.load_wav_and_resample(wav_path, sample_rate)

    spectra = _compute_analysis_spectra(waveform, sample_rate, dataset_conf)
    actual_fbank = iw.extract_fbank_features(waveform, dataset_conf)
    context_feats, context_details = _apply_context_expansion_with_details(actual_fbank, dataset_conf)
    skip_feats, skip_details = _apply_frame_skip_with_details(context_feats, dataset_conf)
    batched_feats = skip_feats.unsqueeze(0)

    logits, layer_tensors = _manual_forward_layers(
        resources["model"],
        batched_feats,
        resources["device"],
        resources["is_jit"],
    )
    probs = logits.softmax(dim=-1)
    frame_energy = actual_fbank.mean(dim=-1)

    time_resolution_sec = resources["time_resolution_sec"]
    model_layers = [
        _build_matrix_payload(
            key=item["key"],
            title=item["title"],
            tensor=item["tensor"],
            note=item["note"],
            time_axis_sec=[float(index * time_resolution_sec) for index in range(int(item["tensor"].size(0)))],
            y_axis_values=list(range(int(item["tensor"].size(1)))),
        )
        for item in layer_tensors
        if item["tensor"].dim() == 2
    ]

    flow_stages = [
        _build_stage_summary(
            "raw_wav",
            "原始 Wav",
            raw_waveform,
            "录到的原始波形；若输入是双声道，会先均值成单声道。",
        ),
        _build_stage_summary(
            "resampled_wav",
            "重采样后 Wav",
            waveform,
            f"推理使用 {sample_rate} Hz 单声道波形。",
        ),
        _build_stage_summary(
            "stft_db",
            "STFT 幅度谱",
            torch.tensor(spectra["stft_db"]["matrix"]),
            "解释性时频图，帮助看爆破音、拖尾和能量集中频带。",
        ),
        _build_stage_summary(
            "mel_energy",
            "Mel 能量",
            torch.tensor(spectra["mel_energy"]["matrix"]),
            "线性频率投到 mel 滤波器组后的能量。",
        ),
        _build_stage_summary(
            "log_mel",
            "log-Mel",
            torch.tensor(spectra["log_mel"]["matrix"]),
            "mel 能量取 log 后的分析视图。",
        ),
        _build_stage_summary(
            "fbank",
            "Kaldi Fbank",
            actual_fbank,
            "真实推理路径使用的 `kaldi.fbank` 输出。",
        ),
        _build_stage_summary(
            "context",
            "Context Expansion",
            context_feats,
            f"左右拼接：left={context_details['left']}, right={context_details['right']}。",
        ),
        _build_stage_summary(
            "frame_skip",
            "Frame Skip",
            skip_feats,
            f"按 `frame_skip={skip_details['frame_skip']}` 取帧。",
        ),
    ]
    for layer in model_layers:
        flow_stages.append(
            {
                "key": layer["key"],
                "title": layer["title"],
                "shape": layer["shape"],
                "stats": layer["stats"],
                "note": layer["note"],
            }
        )

    plain_beam_trace = _ctc_prefix_beam_search_with_trace(
        probs,
        id2tok,
        time_resolution_sec,
        keywords_tokenset=None,
        score_beam_size=max(beam_size, 5),
        path_beam_size=max(beam_size, 8),
        prob_threshold=0.0,
    )
    keyword_beam_trace = _ctc_prefix_beam_search_with_trace(
        probs,
        id2tok,
        time_resolution_sec,
        keywords_tokenset=resources["keywords_idxset"],
        score_beam_size=max(beam_size, 5),
        path_beam_size=max(beam_size, 8),
        prob_threshold=0.0,
    )

    cmvn_module = getattr(resources["model"], "global_cmvn", None)
    cmvn_debug = {
        "enabled": cmvn_module is not None,
        "mean_preview": _to_nested_list(cmvn_module.mean[:16]) if cmvn_module is not None else [],
        "istd_preview": _to_nested_list(cmvn_module.istd[:16]) if cmvn_module is not None else [],
    }

    return {
        "wav": str(wav_path),
        "audio": {
            "raw_sample_rate": int(raw_sr),
            "resampled_sample_rate": int(sample_rate),
            "raw_waveform": _extract_waveform_payload(raw_waveform, int(raw_sr)),
            "resampled_waveform": _extract_waveform_payload(waveform, int(sample_rate)),
        },
        "spectra": {
            "frame_length_ms": spectra["frame_length_ms"],
            "frame_shift_ms": spectra["frame_shift_ms"],
            "n_fft": spectra["n_fft"],
            "win_length": spectra["win_length"],
            "hop_length": spectra["hop_length"],
            "stft_db": spectra["stft_db"],
            "mel_energy": spectra["mel_energy"],
            "log_mel": spectra["log_mel"],
            "fbank": _build_matrix_payload(
                key="fbank",
                title="Kaldi Fbank",
                tensor=actual_fbank,
                note="真实喂给模型前的 80 维 Fbank 特征。",
                time_axis_sec=[float(index * spectra["hop_length"] / sample_rate) for index in range(int(actual_fbank.size(0)))],
                y_axis_values=list(range(int(actual_fbank.size(1)))),
                y_axis_name="mel_bin",
            ),
        },
        "context_debug": context_details,
        "frame_skip_debug": skip_details,
        "cmvn_debug": cmvn_debug,
        "model_layers": model_layers,
        "flow_stages": flow_stages,
        "token_strength": _build_keyword_token_strength(probs, resources, id2tok, frame_energy),
        "frame_energy": {
            "values": _to_nested_list(frame_energy),
            "times_sec": [float(index * spectra["hop_length"] / sample_rate) for index in range(int(frame_energy.numel()))],
            "stats": _tensor_stats(frame_energy),
        },
        "beam_trace": {
            "plain": plain_beam_trace,
            "keyword": keyword_beam_trace,
        },
    }
