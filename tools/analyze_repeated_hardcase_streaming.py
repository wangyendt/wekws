#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-repeated-hardcase")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
S0_DIR = REPO_ROOT / "examples" / "hi_xiaowen" / "s0"
if str(S0_DIR) not in sys.path:
    sys.path.insert(0, str(S0_DIR))

import diagnose_wav as dw
import infer_wav as iw
import infer_wav_stream as iws


warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


DEFAULT_DICT_DIR = S0_DIR / "dict_top20_xlxl"
DEFAULT_KEYWORDS = "小雷小雷,小雷快拍"
DEFAULT_TEACHER_EXP_DIR = S0_DIR / "exp" / "fsmn_ctc_xlxl_top20_weight_surgery"
DEFAULT_TEACHER_TEST_ID = "159"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "把单条 hard case 音频重复拼接成更长音频，然后同时检查 teacher/student "
            "在非流式和流式模式下的行为。"
        )
    )
    parser.add_argument("--wav", required=True, help="原始 hard case wav")
    parser.add_argument("--student-exp-dir", required=True, help="学生实验目录，例如 exp/fsmn_ctc_xlxl_distill_199k")
    parser.add_argument("--student-test-id", required=True, help="学生评测 id，例如 119 / 229")
    parser.add_argument("--teacher-exp-dir", default=str(DEFAULT_TEACHER_EXP_DIR), help="老师实验目录")
    parser.add_argument("--teacher-test-id", default=DEFAULT_TEACHER_TEST_ID, help="老师评测 id，默认 159")
    parser.add_argument("--dict-dir", default=str(DEFAULT_DICT_DIR), help="词表目录")
    parser.add_argument("--keywords", default=DEFAULT_KEYWORDS, help="模型关键词列表")
    parser.add_argument("--target-keyword", default="小雷小雷", help="重点关注的目标关键词")
    parser.add_argument("--repeat", type=int, default=3, help="总重复次数；3 表示原音频重复成 3 段")
    parser.add_argument("--gap-ms", type=float, default=0.0, help="每两段之间插入多少毫秒静音，默认 0")
    parser.add_argument("--chunk-ms", type=float, default=300.0, help="流式模拟时的送入 chunk 时长")
    parser.add_argument("--beam-size", type=int, default=8, help="非流式 diagnose beam top-k")
    parser.add_argument("--frame-topk", type=int, default=8, help="非流式 diagnose 平均帧 top-k")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id，传 -1 表示 CPU")
    parser.add_argument("--use-cpp-decoder", action="store_true", help="流式模式使用 C++ pybind decoder")
    parser.add_argument("--use-c-decoder", action="store_true", help="流式模式使用 C 风格 pybind decoder")
    parser.add_argument("--dump-dir", default="", help="输出目录；默认 student_exp_dir/analysis/repeated_hardcase_<wavstem>_xN")
    parser.add_argument("--indent", type=int, default=2, help="JSON 输出缩进空格数")
    return parser.parse_args()


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [
        Path.cwd() / path,
        REPO_ROOT / path,
        S0_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def normalize_test_id(test_id: str) -> str:
    return test_id if test_id.startswith("test_") else f"test_{test_id}"


def infer_checkpoint_name(test_id: str) -> str:
    stem = test_id
    if stem.startswith("test_"):
        stem = stem[len("test_") :]
    return stem if stem.endswith(".pt") else f"{stem}.pt"


def build_model_info(exp_dir: Path, test_id: str, dict_dir: Path, alias: str) -> Dict[str, Optional[Path]]:
    checkpoint = exp_dir / infer_checkpoint_name(test_id)
    config = exp_dir / "config.yaml"
    stats_dir = exp_dir / normalize_test_id(test_id)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not config.exists():
        raise FileNotFoundError(f"config not found: {config}")
    if not stats_dir.exists():
        raise FileNotFoundError(f"stats dir not found: {stats_dir}")
    return {
        "alias": alias,
        "checkpoint": checkpoint,
        "config": config,
        "dict_dir": dict_dir,
        "stats_dir": stats_dir,
    }


def build_threshold_args() -> SimpleNamespace:
    return SimpleNamespace(
        threshold_map="",
        target_fa_per_hour=1.0,
        pick_mode="legacy",
        frr_eps=0.001,
        stats_dir="",
    )


def count_token_occurrences(text: Optional[str], token: str) -> int:
    if not text:
        return 0
    return sum(1 for item in str(text).split() if item == token)


def load_tensor(path: Path) -> np.ndarray:
    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, torch.Tensor):
        return tensor.numpy()
    return np.asarray(tensor)


def token_label(token_id: int, id2tok: Dict[int, str]) -> str:
    if token_id == 0:
        return "<blk>"
    return id2tok.get(token_id, f"<unk:{token_id}>")


def build_plot_token_ids(
    target_keyword: str,
    keywords_token: Dict[str, Dict[str, tuple]],
    id2tok: Dict[int, str],
) -> List[int]:
    token_ids: List[int] = [0]
    seen = {0}

    filler_id = None
    for token_id, token in id2tok.items():
        if token == "<filler>":
            filler_id = int(token_id)
            break
    if filler_id is not None and filler_id not in seen:
        token_ids.append(filler_id)
        seen.add(filler_id)

    if target_keyword not in keywords_token:
        raise KeyError(f"target keyword not found in keywords_token: {target_keyword}")
    for token_id in keywords_token[target_keyword]["token_id"]:
        token_id = int(token_id)
        if token_id in seen:
            continue
        token_ids.append(token_id)
        seen.add(token_id)
    return token_ids


def build_time_axis(num_frames: int, time_resolution_sec: float) -> np.ndarray:
    return np.arange(num_frames, dtype=np.float32) * float(time_resolution_sec)


def compute_output_time_resolution_sec(configs: Dict) -> float:
    return float(iw.get_time_resolution_sec(configs) * iw.get_frame_skip(configs))


def build_segment_boundaries(repeat_meta: Dict[str, object]) -> List[float]:
    repeat = int(repeat_meta["repeat"])
    segment_span_sec = float(repeat_meta["segment_span_sec"])
    return [segment_span_sec * index for index in range(1, repeat)]


def compute_segment_token_peaks(
    probs: np.ndarray,
    token_ids: List[int],
    id2tok: Dict[int, str],
    repeat_meta: Dict[str, object],
    time_resolution_sec: float,
) -> Dict[str, List[Dict[str, float]]]:
    summary: Dict[str, List[Dict[str, float]]] = {}
    if probs.ndim != 2 or probs.shape[0] == 0:
        return summary

    repeat = int(repeat_meta["repeat"])
    segment_span_sec = float(repeat_meta["segment_span_sec"])
    segment_frames = max(1, int(round(segment_span_sec / time_resolution_sec)))

    for token_id in token_ids:
        if token_id >= probs.shape[1]:
            continue
        token_values = probs[:, token_id]
        rows: List[Dict[str, float]] = []
        for segment_index in range(repeat):
            start_frame = segment_index * segment_frames
            end_frame = probs.shape[0] if segment_index == repeat - 1 else min(probs.shape[0], (segment_index + 1) * segment_frames)
            if start_frame >= end_frame:
                continue
            seg_values = token_values[start_frame:end_frame]
            peak_offset = int(np.argmax(seg_values))
            peak_frame = start_frame + peak_offset
            rows.append(
                {
                    "segment_index": segment_index,
                    "peak_prob": float(seg_values[peak_offset]),
                    "peak_frame": int(peak_frame),
                    "peak_time_sec": float(peak_frame * time_resolution_sec),
                }
            )
        summary[token_label(token_id, id2tok)] = rows
    return summary


def _draw_segment_boundaries(ax, repeat_meta: Dict[str, object]):
    for boundary_sec in build_segment_boundaries(repeat_meta):
        ax.axvline(boundary_sec, color="#666666", linestyle="--", linewidth=1.0, alpha=0.7)


def _draw_activation_spans(ax, activations: List[Dict[str, object]], color: str = "#2ca02c"):
    for item in activations:
        start_sec = float(item["start_time_sec"])
        end_sec = float(item["end_time_sec"])
        ax.axvspan(start_sec, end_sec, color=color, alpha=0.16)
        ax.axvline(end_sec, color=color, linewidth=1.0, alpha=0.65)


def _draw_nonstreaming_candidate_span(
    ax,
    keyword_diagnostic: Optional[Dict[str, object]],
    output_time_resolution_sec: float,
):
    if not keyword_diagnostic:
        return
    start_frame = keyword_diagnostic.get("candidate_start_frame")
    end_frame = keyword_diagnostic.get("candidate_end_frame")
    if start_frame is None or end_frame is None:
        return
    ax.axvspan(
        float(start_frame) * output_time_resolution_sec,
        float(end_frame) * output_time_resolution_sec,
        color="#ff7f0e",
        alpha=0.10,
    )


def plot_model_overview(
    output_path: Path,
    model_label: str,
    token_ids: List[int],
    id2tok: Dict[int, str],
    repeat_meta: Dict[str, object],
    output_time_resolution_sec: float,
    nonstreaming_logits: np.ndarray,
    nonstreaming_probs: np.ndarray,
    streaming_probs: np.ndarray,
    keyword_diagnostic: Optional[Dict[str, object]],
    activations: List[Dict[str, object]],
):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
    panels = [
        (axes[0], "Non-streaming Logits", nonstreaming_logits, False),
        (axes[1], "Non-streaming Softmax Prob", nonstreaming_probs, True),
        (axes[2], "Streaming Softmax Prob", streaming_probs, True),
    ]

    for ax, title, values, is_prob in panels:
        if values.ndim != 2 or values.shape[0] == 0:
            ax.set_title(f"{title} (empty)")
            ax.grid(True, alpha=0.25)
            continue
        time_axis = build_time_axis(values.shape[0], output_time_resolution_sec)
        for token_id in token_ids:
            if token_id >= values.shape[1]:
                continue
            ax.plot(
                time_axis,
                values[:, token_id],
                label=token_label(token_id, id2tok),
                linewidth=1.5,
            )
        _draw_segment_boundaries(ax, repeat_meta)
        if is_prob:
            ax.set_ylim(-0.02, 1.02)
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)

    _draw_nonstreaming_candidate_span(axes[0], keyword_diagnostic, output_time_resolution_sec)
    _draw_nonstreaming_candidate_span(axes[1], keyword_diagnostic, output_time_resolution_sec)
    _draw_activation_spans(axes[2], activations)

    axes[0].set_ylabel("logit")
    axes[1].set_ylabel("prob")
    axes[2].set_ylabel("prob")
    fig.suptitle(f"{model_label}: repeated hard case token overview", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_teacher_student_filler_compare(
    output_path: Path,
    repeat_meta: Dict[str, object],
    output_time_resolution_sec: float,
    student_id2tok: Dict[int, str],
    teacher_id2tok: Dict[int, str],
    student_nonstreaming_probs: np.ndarray,
    student_streaming_probs: np.ndarray,
    teacher_nonstreaming_probs: np.ndarray,
    teacher_streaming_probs: np.ndarray,
    student_activations: List[Dict[str, object]],
    teacher_activations: List[Dict[str, object]],
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False, sharey=True)
    panels = [
        (axes[0, 0], "Student Non-streaming Prob", student_nonstreaming_probs, student_id2tok, None),
        (axes[0, 1], "Student Streaming Prob", student_streaming_probs, student_id2tok, student_activations),
        (axes[1, 0], "Teacher Non-streaming Prob", teacher_nonstreaming_probs, teacher_id2tok, None),
        (axes[1, 1], "Teacher Streaming Prob", teacher_streaming_probs, teacher_id2tok, teacher_activations),
    ]

    focus_tokens = ["<filler>", "小", "雷"]
    for ax, title, values, id2tok, activations in panels:
        if values.ndim != 2 or values.shape[0] == 0:
            ax.set_title(f"{title} (empty)")
            ax.grid(True, alpha=0.25)
            continue
        time_axis = build_time_axis(values.shape[0], output_time_resolution_sec)
        for focus_token in focus_tokens:
            token_id = None
            for idx, tok in id2tok.items():
                if tok == focus_token:
                    token_id = int(idx)
                    break
            if token_id is None or token_id >= values.shape[1]:
                continue
            ax.plot(time_axis, values[:, token_id], label=focus_token, linewidth=1.7)
        _draw_segment_boundaries(ax, repeat_meta)
        if activations is not None:
            _draw_activation_spans(ax, activations)
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("prob")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("Teacher vs Student filler / keyword-token response", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_repeat_wav(
    wav_path: Path,
    repeat: int,
    gap_ms: float,
    output_path: Path,
) -> Dict[str, object]:
    waveform, sample_rate = torchaudio.load(str(wav_path))
    if repeat < 1:
        raise ValueError("--repeat must be >= 1")
    gap_samples = max(0, int(round(gap_ms / 1000.0 * sample_rate)))
    gap = torch.zeros((waveform.size(0), gap_samples), dtype=waveform.dtype)
    pieces: List[torch.Tensor] = []
    for index in range(repeat):
        pieces.append(waveform)
        if gap_samples > 0 and index != repeat - 1:
            pieces.append(gap)
    merged = torch.cat(pieces, dim=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), merged, sample_rate, encoding="PCM_S", bits_per_sample=16)
    orig_duration_sec = float(waveform.size(1) / sample_rate)
    total_duration_sec = float(merged.size(1) / sample_rate)
    segment_span_sec = orig_duration_sec + float(gap_samples / sample_rate)
    return {
        "sample_rate": int(sample_rate),
        "orig_num_samples": int(waveform.size(1)),
        "orig_duration_sec": orig_duration_sec,
        "repeat": int(repeat),
        "gap_ms": float(gap_ms),
        "gap_samples": int(gap_samples),
        "total_num_samples": int(merged.size(1)),
        "total_duration_sec": total_duration_sec,
        "segment_span_sec": segment_span_sec,
    }


def make_resources(
    model_info: Dict[str, Optional[Path]],
    keywords: List[str],
    gpu: int,
):
    configs = iw.load_config(model_info["config"])
    model, device, model_type = iw.load_model(model_info["checkpoint"], configs, gpu)
    threshold_map = iw.load_threshold_map(build_threshold_args(), model_info, keywords)
    time_resolution_sec = iw.get_time_resolution_sec(configs)
    keywords_token, keywords_idxset = iw.build_keyword_token_info(keywords, model_info["dict_dir"])
    id2tok = dw.load_id2token(model_info["dict_dir"])
    return {
        "model_info": model_info,
        "configs": configs,
        "model": model,
        "device": device,
        "model_type": model_type,
        "threshold_map": threshold_map,
        "time_resolution_sec": time_resolution_sec,
        "keywords": keywords,
        "keywords_token": keywords_token,
        "keywords_idxset": keywords_idxset,
        "id2tok": id2tok,
    }


def run_nonstreaming_analysis(
    wav_path: Path,
    resources: Dict[str, object],
    dump_dir: Path,
    beam_size: int,
    frame_topk: int,
    target_keyword: str,
) -> Dict[str, object]:
    dump_dir.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        beam_size=beam_size,
        frame_topk=frame_topk,
        dump_dir=dump_dir,
    )
    report = dw.diagnose_one_wav(
        wav_path,
        args,
        {
            "keywords": resources["keywords"],
            "model_info": resources["model_info"],
            "configs": resources["configs"],
            "model": resources["model"],
            "device": resources["device"],
            "is_jit": resources["model_type"] == "jit",
            "threshold_map": resources["threshold_map"],
            "time_resolution_sec": resources["time_resolution_sec"],
            "keywords_token": resources["keywords_token"],
            "keywords_idxset": resources["keywords_idxset"],
            "dump_dir": dump_dir,
        },
        resources["id2tok"],
    )
    greedy_text = report.get("greedy_decode", {}).get("text")
    beam_text = report.get("beam_topk", [{}])[0].get("text") if report.get("beam_topk") else None
    keyword_diag = None
    for item in report.get("keyword_diagnostics", []):
        if "".join(str(item.get("keyword", "")).split()) == "".join(target_keyword.split()):
            keyword_diag = item
            break
    return {
        "infer_result": report.get("infer_result"),
        "decode_assessment": report.get("decode_assessment"),
        "keyword_diagnostic": keyword_diag,
        "greedy_text": greedy_text,
        "greedy_filler_count": count_token_occurrences(greedy_text, "<filler>"),
        "beam_top1_text": beam_text,
        "beam_top1_filler_count": count_token_occurrences(beam_text, "<filler>"),
        "artifact_dir": report.get("artifact_dir"),
    }


def collect_streaming_activations(
    streamer: iws.StreamingKeywordSpotter,
    pcm: np.ndarray,
    chunk_ms: float,
    time_resolution_sec: float,
    segment_span_sec: float,
) -> Dict[str, object]:
    chunk_samples = max(1, int(chunk_ms / 1000.0 * streamer.sample_rate))
    activations: List[Dict[str, object]] = []
    probs_chunks: List[torch.Tensor] = []

    for start in range(0, pcm.shape[0], chunk_samples):
        feats = streamer.accept_wave_chunk(pcm[start:start + chunk_samples])
        if feats is None or feats.size(0) < 1:
            continue

        probs = streamer._forward_model(feats)
        probs_chunks.append(probs.detach().cpu())
        for local_frame_index, prob in enumerate(probs):
            absolute_frame = streamer.total_frames + local_frame_index * streamer.downsampling
            activation = streamer._step_decoder(absolute_frame, prob)
            if activation is None:
                continue
            start_time_sec = float(activation["start_frame"] * time_resolution_sec)
            end_time_sec = float(activation["end_frame"] * time_resolution_sec)
            segment_index = int(end_time_sec / segment_span_sec) if segment_span_sec > 0 else 0
            activations.append(
                {
                    "keyword": activation["candidate_keyword"],
                    "score": float(activation["candidate_score"]),
                    "start_frame": int(activation["start_frame"]),
                    "end_frame": int(activation["end_frame"]),
                    "start_time_sec": start_time_sec,
                    "end_time_sec": end_time_sec,
                    "segment_index": segment_index,
                }
            )
            streamer.reset_decode_state()

        streamer.total_frames += len(probs) * streamer.downsampling
        if streamer._native_decoder is None:
            first_hyp_start = streamer._get_first_hyp_start_frame()
            if first_hyp_start >= 0 and (streamer.total_frames - first_hyp_start) > streamer.max_frames:
                streamer.reset_decode_state()

    if probs_chunks:
        all_probs = torch.cat(probs_chunks, dim=0)
    else:
        all_probs = torch.zeros((0, 0), dtype=torch.float32)

    return {
        "activations": activations,
        "probs": all_probs,
    }


def run_streaming_analysis(
    wav_path: Path,
    resources: Dict[str, object],
    dump_dir: Path,
    chunk_ms: float,
    use_cpp_decoder: bool,
    use_c_decoder: bool,
    repeat_meta: Dict[str, object],
) -> Dict[str, object]:
    dump_dir.mkdir(parents=True, exist_ok=True)
    streamer = iws.StreamingKeywordSpotter(
        configs=resources["configs"],
        model=resources["model"],
        device=resources["device"],
        model_type=resources["model_type"],
        keywords=resources["keywords"],
        keywords_token=resources["keywords_token"],
        keywords_idxset=resources["keywords_idxset"],
        threshold_map=resources["threshold_map"],
        score_beam_size=3,
        path_beam_size=20,
        min_frames=5,
        max_frames=250,
        interval_frames=50,
        disable_threshold=False,
        use_cpp_decoder=use_cpp_decoder,
        use_c_decoder=use_c_decoder,
        dump_dir=dump_dir,
        align_center_context=False,
    )

    waveform = iw.load_wav_and_resample(wav_path, streamer.sample_rate)
    waveform = waveform.squeeze(0).numpy()
    pcm = np.clip(np.round(waveform * (1 << 15)), -32768, 32767).astype(np.int16)

    streaming_payload = collect_streaming_activations(
        streamer=streamer,
        pcm=pcm,
        chunk_ms=chunk_ms,
        time_resolution_sec=resources["time_resolution_sec"],
        segment_span_sec=float(repeat_meta["segment_span_sec"]),
    )
    activations = streaming_payload["activations"]
    streaming_probs = streaming_payload["probs"]
    best_decode = streamer.get_best_decode_result()
    artifact_dir = streamer.dump_artifacts()
    streaming_probs_path = None
    if artifact_dir is not None:
        streaming_probs_path = str(Path(artifact_dir) / "streaming_probs.pt")
        torch.save(streaming_probs, streaming_probs_path)

    per_segment_counts: Dict[str, int] = {}
    for item in activations:
        key = str(item["segment_index"])
        per_segment_counts[key] = per_segment_counts.get(key, 0) + 1

    return {
        "chunk_ms": float(chunk_ms),
        "lookahead_sec": float(iws.compute_streaming_lookahead_sec(resources["configs"])),
        "num_activations": len(activations),
        "activations": activations,
        "activations_per_segment": per_segment_counts,
        "best_decode_result": best_decode,
        "artifact_dir": artifact_dir,
        "streaming_probs_path": streaming_probs_path,
        "use_cpp_decoder": bool(use_cpp_decoder),
        "use_c_decoder": bool(use_c_decoder),
    }


def main():
    args = parse_args()

    wav_path = resolve_input_path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(f"wav not found: {wav_path}")

    student_exp_dir = resolve_input_path(args.student_exp_dir)
    teacher_exp_dir = resolve_input_path(args.teacher_exp_dir)
    dict_dir = resolve_input_path(args.dict_dir)
    if not dict_dir.exists():
        raise FileNotFoundError(f"dict dir not found: {dict_dir}")

    keywords = iw.parse_keywords_arg(args.keywords)
    if not keywords:
        raise ValueError("keywords must not be empty")

    if args.dump_dir:
        dump_dir = resolve_input_path(args.dump_dir)
    else:
        dump_dir = student_exp_dir / "analysis" / f"repeated_hardcase_{wav_path.stem}_x{args.repeat}"
    dump_dir.mkdir(parents=True, exist_ok=True)

    repeated_wav_path = dump_dir / f"{wav_path.stem}_x{args.repeat}.wav"
    repeat_meta = build_repeat_wav(
        wav_path=wav_path,
        repeat=args.repeat,
        gap_ms=args.gap_ms,
        output_path=repeated_wav_path,
    )

    student_info = build_model_info(
        exp_dir=student_exp_dir,
        test_id=args.student_test_id,
        dict_dir=dict_dir,
        alias="student",
    )
    teacher_info = build_model_info(
        exp_dir=teacher_exp_dir,
        test_id=args.teacher_test_id,
        dict_dir=dict_dir,
        alias="teacher",
    )

    model_entries = [
        ("student", student_info),
        ("teacher", teacher_info),
    ]
    summary_models: Dict[str, Dict[str, object]] = {}
    plot_dir = dump_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_inputs: Dict[str, Dict[str, object]] = {}

    for label, model_info in model_entries:
        resources = make_resources(model_info, keywords, args.gpu)
        model_dump_dir = dump_dir / label
        nonstreaming = run_nonstreaming_analysis(
            wav_path=repeated_wav_path,
            resources=resources,
            dump_dir=model_dump_dir / "nonstreaming_diag",
            beam_size=args.beam_size,
            frame_topk=args.frame_topk,
            target_keyword=args.target_keyword,
        )
        streaming = run_streaming_analysis(
            wav_path=repeated_wav_path,
            resources=resources,
            dump_dir=model_dump_dir / "streaming_diag",
            chunk_ms=args.chunk_ms,
            use_cpp_decoder=args.use_cpp_decoder,
            use_c_decoder=args.use_c_decoder,
            repeat_meta=repeat_meta,
        )
        artifact_dir = Path(nonstreaming["artifact_dir"])
        nonstreaming_logits = load_tensor(artifact_dir / "logits.pt")
        nonstreaming_probs = load_tensor(artifact_dir / "probs.pt")
        streaming_probs = (
            load_tensor(Path(streaming["streaming_probs_path"]))
            if streaming.get("streaming_probs_path")
            else np.zeros((0, 0), dtype=np.float32)
        )
        token_ids = build_plot_token_ids(args.target_keyword, resources["keywords_token"], resources["id2tok"])
        output_time_resolution_sec = compute_output_time_resolution_sec(resources["configs"])
        nonstreaming_peaks = compute_segment_token_peaks(
            probs=nonstreaming_probs,
            token_ids=token_ids,
            id2tok=resources["id2tok"],
            repeat_meta=repeat_meta,
            time_resolution_sec=output_time_resolution_sec,
        )
        streaming_peaks = compute_segment_token_peaks(
            probs=streaming_probs,
            token_ids=token_ids,
            id2tok=resources["id2tok"],
            repeat_meta=repeat_meta,
            time_resolution_sec=output_time_resolution_sec,
        )
        overview_plot_path = plot_dir / f"{label}_token_overview.png"
        plot_model_overview(
            output_path=overview_plot_path,
            model_label=label,
            token_ids=token_ids,
            id2tok=resources["id2tok"],
            repeat_meta=repeat_meta,
            output_time_resolution_sec=output_time_resolution_sec,
            nonstreaming_logits=nonstreaming_logits,
            nonstreaming_probs=nonstreaming_probs,
            streaming_probs=streaming_probs,
            keyword_diagnostic=nonstreaming.get("keyword_diagnostic"),
            activations=streaming["activations"],
        )
        summary_models[label] = {
            "checkpoint": str(model_info["checkpoint"]),
            "stats_dir": str(model_info["stats_dir"]),
            "threshold_map": resources["threshold_map"],
            "nonstreaming": nonstreaming,
            "streaming": streaming,
            "plots": {
                "token_overview": str(overview_plot_path),
            },
            "selected_tokens": [token_label(token_id, resources["id2tok"]) for token_id in token_ids],
            "nonstreaming_segment_prob_peaks": nonstreaming_peaks,
            "streaming_segment_prob_peaks": streaming_peaks,
        }
        plot_inputs[label] = {
            "id2tok": resources["id2tok"],
            "nonstreaming_probs": nonstreaming_probs,
            "streaming_probs": streaming_probs,
            "activations": streaming["activations"],
            "output_time_resolution_sec": output_time_resolution_sec,
        }
        del resources

    compare_plot_path = plot_dir / "student_teacher_filler_compare.png"
    plot_teacher_student_filler_compare(
        output_path=compare_plot_path,
        repeat_meta=repeat_meta,
        output_time_resolution_sec=float(plot_inputs["student"]["output_time_resolution_sec"]),
        student_id2tok=plot_inputs["student"]["id2tok"],
        teacher_id2tok=plot_inputs["teacher"]["id2tok"],
        student_nonstreaming_probs=plot_inputs["student"]["nonstreaming_probs"],
        student_streaming_probs=plot_inputs["student"]["streaming_probs"],
        teacher_nonstreaming_probs=plot_inputs["teacher"]["nonstreaming_probs"],
        teacher_streaming_probs=plot_inputs["teacher"]["streaming_probs"],
        student_activations=plot_inputs["student"]["activations"],
        teacher_activations=plot_inputs["teacher"]["activations"],
    )

    summary = {
        "source_wav": str(wav_path),
        "repeated_wav": str(repeated_wav_path),
        "target_keyword": args.target_keyword,
        "repeat_meta": repeat_meta,
        "plots": {
            "student_teacher_filler_compare": str(compare_plot_path),
        },
        "models": summary_models,
    }

    summary_path = dump_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=args.indent),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=args.indent))


if __name__ == "__main__":
    main()
