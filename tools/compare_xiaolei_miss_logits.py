#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-compare-xiaolei")

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


DEFAULT_DATA_LIST = S0_DIR / "data_xlxl_0327_ctc_v1_clean" / "test" / "data.list"
DEFAULT_DICT_DIR = S0_DIR / "dict_top20_xlxl"
DEFAULT_KEYWORDS = "小雷小雷,小雷快拍"
DEFAULT_TEACHER_EXP_DIR = S0_DIR / "exp" / "fsmn_ctc_xlxl_top20_weight_surgery"
DEFAULT_TEACHER_TEST_ID = "159"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "对比小雷测试集中学生漏检样本在学生模型和 159 老师模型上的 logit/prob 曲线，"
            "帮助判断 blank 压制、前缀缺失还是整体路径塌缩。"
        )
    )
    parser.add_argument("--student-exp-dir", required=True, help="学生实验目录，如 exp/fsmn_ctc_xlxl_distill_199k_replay_from79")
    parser.add_argument("--student-test-id", required=True, help="学生评测 id，如 119 / 229 / test_229")
    parser.add_argument("--teacher-exp-dir", default=str(DEFAULT_TEACHER_EXP_DIR), help="老师实验目录")
    parser.add_argument("--teacher-test-id", default=DEFAULT_TEACHER_TEST_ID, help="老师评测 id，默认 159")
    parser.add_argument("--keyword", default="小雷小雷", help="目标关键词，默认小雷小雷")
    parser.add_argument("--keywords", default=DEFAULT_KEYWORDS, help="模型加载时使用的关键词列表")
    parser.add_argument("--data-list", default=str(DEFAULT_DATA_LIST), help="评测数据 data.list")
    parser.add_argument("--dict-dir", default=str(DEFAULT_DICT_DIR), help="词表目录")
    parser.add_argument(
        "--teacher-filter",
        choices=["all", "detected_target", "miss_only"],
        default="detected_target",
        help="只保留老师命中 / 老师也漏 / 全部学生漏检样本",
    )
    parser.add_argument("--max-samples", type=int, default=10, help="最多处理多少条学生漏检样本")
    parser.add_argument("--sample-key", default="", help="只对比指定 key")
    parser.add_argument("--extra-tokens", default="", help="额外绘制的 token，逗号分隔，例如 快,拍")
    parser.add_argument("--student-checkpoint", default="", help="显式指定学生 checkpoint")
    parser.add_argument("--student-config", default="", help="显式指定学生 config.yaml")
    parser.add_argument("--student-stats-dir", default="", help="显式指定学生 test 目录")
    parser.add_argument("--teacher-checkpoint", default="", help="显式指定老师 checkpoint")
    parser.add_argument("--teacher-config", default="", help="显式指定老师 config.yaml")
    parser.add_argument("--teacher-stats-dir", default="", help="显式指定老师 test 目录")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id，传 -1 表示 CPU")
    parser.add_argument("--beam-size", type=int, default=8, help="diagnose_wav beam top-k")
    parser.add_argument("--frame-topk", type=int, default=8, help="diagnose_wav mean frame top-k")
    parser.add_argument("--dump-dir", default="", help="输出目录；默认 student_exp_dir/analysis/logit_compare_<keyword>_<test>")
    return parser.parse_args()


def compact_text(text: str) -> str:
    return "".join(str(text).split())


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


def parse_score_file(score_path: Path) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    for raw_line in score_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        key = parts[0]
        status = parts[1] if len(parts) > 1 else ""
        item = {"status": status, "keyword": None, "score": None}
        if status == "detected" and len(parts) >= 4:
            item["keyword"] = parts[2]
            try:
                item["score"] = float(parts[3])
            except ValueError:
                item["score"] = None
        results[key] = item
    return results


def classify_outcome(score_item: Optional[Dict], target_keyword: str) -> str:
    if not score_item:
        return "missing_in_score"
    status = score_item.get("status")
    if status == "rejected":
        return "rejected"
    if status == "detected":
        pred_keyword = compact_text(score_item.get("keyword") or "")
        if pred_keyword == compact_text(target_keyword):
            return "detected_target"
        return f"detected_other:{pred_keyword or 'unknown'}"
    return status or "unknown"


def load_target_records(data_list: Path, target_keyword: str) -> List[Dict]:
    target_compact = compact_text(target_keyword)
    records: List[Dict] = []
    with data_list.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if compact_text(item.get("txt", "")) != target_compact:
                continue
            records.append(item)
    return records


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff-]+", "_", text)
    cleaned = cleaned.strip("_")
    return cleaned or "sample"


def artifact_dir_for_wav(dump_dir: Path, wav_path: Path) -> Path:
    digest = hashlib.sha1(str(wav_path).encode("utf-8")).hexdigest()[:10]
    return dump_dir / f"{wav_path.stem}_{digest}"


def build_resources(
    checkpoint: Path,
    config: Path,
    dict_dir: Path,
    stats_dir: Path,
    keywords: str,
    gpu: int,
    dump_dir: Path,
):
    model_args = SimpleNamespace(
        model="s3",
        checkpoint=str(checkpoint),
        model_dir="",
        checkpoint_name="",
        config=str(config),
        dict_dir=str(dict_dir),
        stats_dir=str(stats_dir),
        keywords=keywords,
        gpu=gpu,
        indent=2,
    )
    infer_args = dw.build_model_args(model_args)
    parsed_keywords = iw.parse_keywords_arg(keywords)
    model_info = iw.resolve_model_paths(infer_args)
    configs = iw.load_config(model_info["config"])
    model, device, is_jit = iw.load_model(model_info["checkpoint"], configs, gpu)
    threshold_map = iw.load_threshold_map(infer_args, model_info, parsed_keywords)
    time_resolution_sec = iw.get_time_resolution_sec(configs)
    frame_skip = iw.get_frame_skip(configs)
    keywords_token, keywords_idxset = iw.build_keyword_token_info(parsed_keywords, model_info["dict_dir"])
    id2tok = dw.load_id2token(model_info["dict_dir"])
    dump_dir.mkdir(parents=True, exist_ok=True)
    resources = {
        "keywords": parsed_keywords,
        "model_info": model_info,
        "configs": configs,
        "model": model,
        "device": device,
        "is_jit": is_jit,
        "threshold_map": threshold_map,
        "time_resolution_sec": time_resolution_sec,
        "frame_skip": frame_skip,
        "output_time_resolution_sec": time_resolution_sec * frame_skip,
        "keywords_token": keywords_token,
        "keywords_idxset": keywords_idxset,
        "dump_dir": dump_dir,
    }
    return resources, id2tok


def load_or_run_report(
    wav_path: Path,
    cache_dir: Path,
    diagnose_args,
    resources: Dict,
    id2tok: Dict[int, str],
) -> Tuple[Dict, Path]:
    artifact_dir = artifact_dir_for_wav(cache_dir, wav_path)
    report_path = artifact_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = dw.diagnose_one_wav(wav_path, diagnose_args, resources, id2tok)
    report["artifact_dir"] = str(artifact_dir)
    return report, artifact_dir


def load_tensor(path: Path) -> np.ndarray:
    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, torch.Tensor):
        return tensor.numpy()
    return np.asarray(tensor)


def build_token_lookup(id2tok: Dict[int, str]) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for idx, tok in id2tok.items():
        lookup.setdefault(tok, int(idx))
    lookup.setdefault("<blk>", 0)
    return lookup


def resolve_token_ids(
    keyword: str,
    keywords_token: Dict[str, Dict[str, Tuple[int, ...]]],
    id2tok: Dict[int, str],
    extra_tokens: str,
) -> List[int]:
    token_ids: List[int] = [0]
    seen = {0}
    lookup = build_token_lookup(id2tok)
    filler_id = lookup.get("<filler>")
    if filler_id is not None and filler_id not in seen:
        token_ids.append(int(filler_id))
        seen.add(int(filler_id))
    if keyword not in keywords_token:
        raise KeyError(f"keyword not found in keywords_token: {keyword}")
    for token_id in keywords_token[keyword]["token_id"]:
        token_id = int(token_id)
        if token_id in seen:
            continue
        token_ids.append(token_id)
        seen.add(token_id)

    for token in [item.strip() for item in extra_tokens.split(",") if item.strip()]:
        if token not in lookup:
            raise KeyError(f"extra token not found in dict: {token}")
        token_id = int(lookup[token])
        if token_id in seen:
            continue
        token_ids.append(token_id)
        seen.add(token_id)
    return token_ids


def token_label(token_id: int, id2tok: Dict[int, str]) -> str:
    if token_id == 0:
        return "<blk>"
    return id2tok.get(token_id, f"<unk:{token_id}>")


def find_keyword_diag(report: Dict, keyword: str) -> Optional[Dict]:
    target = compact_text(keyword)
    for item in report.get("keyword_diagnostics", []):
        if compact_text(item.get("keyword", "")) == target:
            return item
    return None


def top_beam_text(report: Dict) -> str:
    if report.get("beam_topk"):
        return report["beam_topk"][0].get("text") or "<empty>"
    return "<none>"


def compare_sample_row(
    record: Dict,
    student_status: str,
    teacher_status: str,
    student_report: Dict,
    teacher_report: Dict,
) -> Dict:
    return {
        "key": record["key"],
        "wav": record["wav"],
        "txt": record["txt"],
        "student_status": student_status,
        "teacher_status": teacher_status,
        "student_greedy": student_report.get("greedy_decode", {}).get("text"),
        "student_beam_top1": top_beam_text(student_report),
        "teacher_greedy": teacher_report.get("greedy_decode", {}).get("text"),
        "teacher_beam_top1": top_beam_text(teacher_report),
        "student_decode_assessment": student_report.get("decode_assessment", {}).get("status"),
        "teacher_decode_assessment": teacher_report.get("decode_assessment", {}).get("status"),
        "student_keyword_diag": (find_keyword_diag(student_report, record["txt"]) or {}).get("status"),
        "teacher_keyword_diag": (find_keyword_diag(teacher_report, record["txt"]) or {}).get("status"),
        "student_artifact_dir": student_report.get("artifact_dir"),
        "teacher_artifact_dir": teacher_report.get("artifact_dir"),
    }


def build_time_axis(num_frames: int, time_resolution_sec: float) -> np.ndarray:
    return np.arange(num_frames, dtype=np.float32) * float(time_resolution_sec)


def build_keyword_prefixes(
    keyword: str,
    keywords_token: Dict[str, Dict[str, Tuple[int, ...]]],
    id2tok: Dict[int, str],
) -> List[Dict]:
    token_ids = [int(x) for x in keywords_token[keyword]["token_id"]]
    prefixes: List[Dict] = []
    for end in range(1, len(token_ids) + 1):
        prefix_ids = token_ids[:end]
        prefixes.append(
            {
                "label": "".join(token_label(token_id, id2tok) for token_id in prefix_ids),
                "token_ids": prefix_ids,
            }
        )
    return prefixes


def compute_ctc_label_logprob_curve(
    probs: np.ndarray,
    label_ids: List[int],
    blank_id: int = 0,
    eps: float = 1e-12,
) -> np.ndarray:
    if probs.ndim != 2:
        raise ValueError(f"Expected probs shape (T, V), got {probs.shape}")
    if not label_ids:
        raise ValueError("label_ids must not be empty")

    num_frames = int(probs.shape[0])
    ext_labels: List[int] = [blank_id]
    for token_id in label_ids:
        ext_labels.append(int(token_id))
        ext_labels.append(blank_id)
    num_states = len(ext_labels)

    safe_probs = np.clip(probs.astype(np.float64), eps, 1.0)
    log_probs = np.log(safe_probs)
    alpha = np.full((num_frames, num_states), -np.inf, dtype=np.float64)

    alpha[0, 0] = log_probs[0, blank_id]
    if num_states > 1:
        alpha[0, 1] = log_probs[0, ext_labels[1]]

    for t in range(1, num_frames):
        for s in range(num_states):
            cur_log = log_probs[t, ext_labels[s]]
            prev_total = alpha[t - 1, s]
            if s - 1 >= 0:
                prev_total = np.logaddexp(prev_total, alpha[t - 1, s - 1])
            if (
                s - 2 >= 0
                and ext_labels[s] != blank_id
                and ext_labels[s] != ext_labels[s - 2]
            ):
                prev_total = np.logaddexp(prev_total, alpha[t - 1, s - 2])
            alpha[t, s] = cur_log + prev_total

    if num_states == 1:
        return alpha[:, 0]
    return np.logaddexp(alpha[:, num_states - 1], alpha[:, num_states - 2])


def build_prefix_curve_bundle(
    probs: np.ndarray,
    prefixes: List[Dict],
) -> List[Dict]:
    bundle: List[Dict] = []
    for item in prefixes:
        logprob_curve = compute_ctc_label_logprob_curve(probs, item["token_ids"], blank_id=0)
        peak_frame = int(np.argmax(logprob_curve))
        bundle.append(
            {
                "label": item["label"],
                "token_ids": list(item["token_ids"]),
                "logprob_curve": logprob_curve,
                "peak_logprob": float(logprob_curve[peak_frame]),
                "peak_frame": peak_frame,
            }
        )
    return bundle


def plot_pair_curves(
    sample_dir: Path,
    sample_row: Dict,
    token_ids: List[int],
    student_id2tok: Dict[int, str],
    teacher_id2tok: Dict[int, str],
    student_output_time_resolution_sec: float,
    teacher_output_time_resolution_sec: float,
    student_logits: np.ndarray,
    student_probs: np.ndarray,
    teacher_logits: np.ndarray,
    teacher_probs: np.ndarray,
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
    panels = [
        (axes[0, 0], "Student Logits", student_logits, student_output_time_resolution_sec, student_id2tok),
        (axes[0, 1], "Teacher Logits", teacher_logits, teacher_output_time_resolution_sec, teacher_id2tok),
        (axes[1, 0], "Student Softmax Prob", student_probs, student_output_time_resolution_sec, student_id2tok),
        (axes[1, 1], "Teacher Softmax Prob", teacher_probs, teacher_output_time_resolution_sec, teacher_id2tok),
    ]

    for ax, title, values, time_resolution_sec, id2tok in panels:
        time_axis = build_time_axis(values.shape[0], time_resolution_sec)
        for token_id in token_ids:
            if token_id >= values.shape[1]:
                continue
            ax.plot(time_axis, values[:, token_id], label=token_label(token_id, id2tok), linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.25)
        if "Prob" in title:
            ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="best", fontsize=9)

    axes[0, 0].set_ylabel("logit")
    axes[0, 1].set_ylabel("logit")
    axes[1, 0].set_ylabel("prob")
    axes[1, 1].set_ylabel("prob")

    fig.suptitle(
        f"{sample_row['key']} | {sample_row['txt']}\n"
        f"student={sample_row['student_status']} greedy={sample_row['student_greedy']} | "
        f"teacher={sample_row['teacher_status']} greedy={sample_row['teacher_greedy']}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(sample_dir / "student_teacher_curves.png", dpi=160)
    plt.close(fig)


def plot_prefix_curves(
    sample_dir: Path,
    sample_row: Dict,
    student_output_time_resolution_sec: float,
    teacher_output_time_resolution_sec: float,
    student_prefix_bundle: List[Dict],
    teacher_prefix_bundle: List[Dict],
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    panels = [
        (axes[0], "Student Prefix Curves", student_output_time_resolution_sec, student_prefix_bundle),
        (axes[1], "Teacher Prefix Curves", teacher_output_time_resolution_sec, teacher_prefix_bundle),
    ]

    for ax, title, time_resolution_sec, bundle in panels:
        for item in bundle:
            time_axis = build_time_axis(len(item["logprob_curve"]), time_resolution_sec)
            ax.plot(time_axis, item["logprob_curve"] / np.log(10.0), label=item["label"], linewidth=1.6)
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("CTC prefix log10 prob")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"{sample_row['key']} | prefix support over time\n"
        f"student={sample_row['student_greedy']} | teacher={sample_row['teacher_greedy']}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(sample_dir / "student_teacher_prefix_curves.png", dpi=160)
    plt.close(fig)


def peak_summary(values: np.ndarray, token_ids: List[int], id2tok: Dict[int, str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for token_id in token_ids:
        if token_id >= values.shape[1]:
            continue
        token_values = values[:, token_id]
        peak_idx = int(np.argmax(token_values))
        summary[token_label(token_id, id2tok)] = {
            "peak_value": float(token_values[peak_idx]),
            "peak_frame": peak_idx,
        }
    return summary


def prefix_peak_summary(
    prefix_bundle: List[Dict],
    output_time_resolution_sec: float,
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for item in prefix_bundle:
        summary[item["label"]] = {
            "peak_logprob": float(item["peak_logprob"]),
            "peak_log10prob": float(item["peak_logprob"] / np.log(10.0)),
            "peak_frame": int(item["peak_frame"]),
            "peak_time_sec": float(item["peak_frame"] * output_time_resolution_sec),
        }
    return summary


def main():
    args = parse_args()

    student_exp_dir = resolve_input_path(args.student_exp_dir)
    teacher_exp_dir = resolve_input_path(args.teacher_exp_dir)
    data_list = resolve_input_path(args.data_list)
    dict_dir = resolve_input_path(args.dict_dir)

    student_stats_dir = (
        resolve_input_path(args.student_stats_dir)
        if args.student_stats_dir
        else student_exp_dir / normalize_test_id(args.student_test_id)
    )
    teacher_stats_dir = (
        resolve_input_path(args.teacher_stats_dir)
        if args.teacher_stats_dir
        else teacher_exp_dir / normalize_test_id(args.teacher_test_id)
    )
    student_checkpoint = (
        resolve_input_path(args.student_checkpoint)
        if args.student_checkpoint
        else student_exp_dir / infer_checkpoint_name(args.student_test_id)
    )
    teacher_checkpoint = (
        resolve_input_path(args.teacher_checkpoint)
        if args.teacher_checkpoint
        else teacher_exp_dir / infer_checkpoint_name(args.teacher_test_id)
    )
    student_config = (
        resolve_input_path(args.student_config)
        if args.student_config
        else student_exp_dir / "config.yaml"
    )
    teacher_config = (
        resolve_input_path(args.teacher_config)
        if args.teacher_config
        else teacher_exp_dir / "config.yaml"
    )

    dump_dir = (
        resolve_input_path(args.dump_dir)
        if args.dump_dir
        else student_exp_dir / "analysis" / f"logit_compare_{compact_text(args.keyword)}_{student_stats_dir.name}"
    )
    plot_root = dump_dir / "plots"
    student_cache_dir = dump_dir / "student_diag_cache"
    teacher_cache_dir = dump_dir / "teacher_diag_cache"

    required_paths = [
        ("data.list", data_list),
        ("student score.txt", student_stats_dir / "score.txt"),
        ("teacher score.txt", teacher_stats_dir / "score.txt"),
        ("student checkpoint", student_checkpoint),
        ("teacher checkpoint", teacher_checkpoint),
        ("student config", student_config),
        ("teacher config", teacher_config),
        ("dict_dir", dict_dir / "dict.txt"),
    ]
    for label, path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    target_records = load_target_records(data_list, args.keyword)
    student_score_map = parse_score_file(student_stats_dir / "score.txt")
    teacher_score_map = parse_score_file(teacher_stats_dir / "score.txt")

    selected_records: List[Tuple[Dict, str, str]] = []
    for record in target_records:
        if args.sample_key and record["key"] != args.sample_key:
            continue
        student_status = classify_outcome(student_score_map.get(record["key"]), args.keyword)
        if student_status == "detected_target":
            continue
        teacher_status = classify_outcome(teacher_score_map.get(record["key"]), args.keyword)
        if args.teacher_filter == "detected_target" and teacher_status != "detected_target":
            continue
        if args.teacher_filter == "miss_only" and teacher_status == "detected_target":
            continue
        selected_records.append((record, student_status, teacher_status))

    if args.max_samples > 0:
        selected_records = selected_records[: args.max_samples]
    if not selected_records:
        raise RuntimeError("没有选中任何样本；请检查 teacher-filter / sample-key / max-samples 设置。")

    student_resources, student_id2tok = build_resources(
        checkpoint=student_checkpoint,
        config=student_config,
        dict_dir=dict_dir,
        stats_dir=student_stats_dir,
        keywords=args.keywords,
        gpu=args.gpu,
        dump_dir=student_cache_dir,
    )
    teacher_resources, teacher_id2tok = build_resources(
        checkpoint=teacher_checkpoint,
        config=teacher_config,
        dict_dir=dict_dir,
        stats_dir=teacher_stats_dir,
        keywords=args.keywords,
        gpu=args.gpu,
        dump_dir=teacher_cache_dir,
    )
    token_ids = resolve_token_ids(
        keyword=args.keyword,
        keywords_token=student_resources["keywords_token"],
        id2tok=student_id2tok,
        extra_tokens=args.extra_tokens,
    )
    prefixes = build_keyword_prefixes(
        keyword=args.keyword,
        keywords_token=student_resources["keywords_token"],
        id2tok=student_id2tok,
    )
    diagnose_args = SimpleNamespace(beam_size=args.beam_size, frame_topk=args.frame_topk)

    dump_dir.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict] = []
    for index, (record, student_status, teacher_status) in enumerate(selected_records, start=1):
        wav_path = Path(record["wav"]).expanduser().resolve()
        student_report, student_artifact_dir = load_or_run_report(
            wav_path, student_cache_dir, diagnose_args, student_resources, student_id2tok
        )
        teacher_report, teacher_artifact_dir = load_or_run_report(
            wav_path, teacher_cache_dir, diagnose_args, teacher_resources, teacher_id2tok
        )

        student_logits = load_tensor(student_artifact_dir / "logits.pt")
        student_probs = load_tensor(student_artifact_dir / "probs.pt")
        teacher_logits = load_tensor(teacher_artifact_dir / "logits.pt")
        teacher_probs = load_tensor(teacher_artifact_dir / "probs.pt")
        student_prefix_bundle = build_prefix_curve_bundle(student_probs, prefixes)
        teacher_prefix_bundle = build_prefix_curve_bundle(teacher_probs, prefixes)

        sample_row = compare_sample_row(record, student_status, teacher_status, student_report, teacher_report)
        sample_row["sample_index"] = index
        sample_dir = plot_root / f"{index:03d}_{safe_name(record['key'])}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        plot_pair_curves(
            sample_dir=sample_dir,
            sample_row=sample_row,
            token_ids=token_ids,
            student_id2tok=student_id2tok,
            teacher_id2tok=teacher_id2tok,
            student_output_time_resolution_sec=float(student_resources["output_time_resolution_sec"]),
            teacher_output_time_resolution_sec=float(teacher_resources["output_time_resolution_sec"]),
            student_logits=student_logits,
            student_probs=student_probs,
            teacher_logits=teacher_logits,
            teacher_probs=teacher_probs,
        )
        plot_prefix_curves(
            sample_dir=sample_dir,
            sample_row=sample_row,
            student_output_time_resolution_sec=float(student_resources["output_time_resolution_sec"]),
            teacher_output_time_resolution_sec=float(teacher_resources["output_time_resolution_sec"]),
            student_prefix_bundle=student_prefix_bundle,
            teacher_prefix_bundle=teacher_prefix_bundle,
        )

        sample_row["selected_tokens"] = [token_label(token_id, student_id2tok) for token_id in token_ids]
        sample_row["selected_prefixes"] = [item["label"] for item in prefixes]
        sample_row["student_logit_peaks"] = peak_summary(student_logits, token_ids, student_id2tok)
        sample_row["teacher_logit_peaks"] = peak_summary(teacher_logits, token_ids, teacher_id2tok)
        sample_row["student_prob_peaks"] = peak_summary(student_probs, token_ids, student_id2tok)
        sample_row["teacher_prob_peaks"] = peak_summary(teacher_probs, token_ids, teacher_id2tok)
        sample_row["student_prefix_peaks"] = prefix_peak_summary(
            student_prefix_bundle, float(student_resources["output_time_resolution_sec"])
        )
        sample_row["teacher_prefix_peaks"] = prefix_peak_summary(
            teacher_prefix_bundle, float(teacher_resources["output_time_resolution_sec"])
        )
        sample_row["student_output_time_resolution_sec"] = float(student_resources["output_time_resolution_sec"])
        sample_row["teacher_output_time_resolution_sec"] = float(teacher_resources["output_time_resolution_sec"])
        sample_row["plot_path"] = str(sample_dir / "student_teacher_curves.png")
        sample_row["prefix_plot_path"] = str(sample_dir / "student_teacher_prefix_curves.png")
        manifest_rows.append(sample_row)

    summary = {
        "student_exp_dir": str(student_exp_dir),
        "student_test_dir": str(student_stats_dir),
        "teacher_exp_dir": str(teacher_exp_dir),
        "teacher_test_dir": str(teacher_stats_dir),
        "keyword": args.keyword,
        "teacher_filter": args.teacher_filter,
        "selected_tokens": [token_label(token_id, student_id2tok) for token_id in token_ids],
        "selected_prefixes": [item["label"] for item in prefixes],
        "student_output_time_resolution_sec": float(student_resources["output_time_resolution_sec"]),
        "teacher_output_time_resolution_sec": float(teacher_resources["output_time_resolution_sec"]),
        "total_target_positives": len(target_records),
        "selected_samples": len(manifest_rows),
        "student_miss_count_before_filter": sum(
            1 for record in target_records if classify_outcome(student_score_map.get(record["key"]), args.keyword) != "detected_target"
        ),
        "teacher_hit_count_in_selection": sum(1 for row in manifest_rows if row["teacher_status"] == "detected_target"),
    }

    summary_path = dump_dir / "summary.json"
    manifest_path = dump_dir / "manifest.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with manifest_path.open("w", encoding="utf-8") as fout:
        for row in manifest_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 72)
    print(f"student_exp_dir: {student_exp_dir}")
    print(f"teacher_exp_dir: {teacher_exp_dir}")
    print(f"keyword:         {args.keyword}")
    print(f"teacher_filter:  {args.teacher_filter}")
    print(f"selected_tokens: {', '.join(summary['selected_tokens'])}")
    print(f"selected_samples:{len(manifest_rows)}")
    print("=" * 72)
    print(f"summary_json:    {summary_path}")
    print(f"manifest_jsonl:  {manifest_path}")
    print(f"plots_dir:       {plot_root}")


if __name__ == "__main__":
    main()
