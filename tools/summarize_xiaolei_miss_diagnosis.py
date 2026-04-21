#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
S0_CANDIDATE = REPO_ROOT / "examples" / "hi_xiaowen" / "s0"
S0_DIR = S0_CANDIDATE if (S0_CANDIDATE / "diagnose_wav.py").exists() else SCRIPT_PATH.parent.parent
if str(S0_DIR) not in sys.path:
    sys.path.insert(0, str(S0_DIR))

import diagnose_wav as dw
import infer_wav as iw


DEFAULT_DATA_LIST = S0_DIR / "data_xlxl_0327_ctc_v1_clean" / "test" / "data.list"
DEFAULT_DICT_DIR = S0_DIR / "dict_top20_xlxl"
DEFAULT_KEYWORDS = "小雷小雷,小雷快拍"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "批量统计小雷漏检模式：先从 score.txt 找出漏检正样本，再批量跑 diagnose_wav，"
            "汇总 greedy/beam/blank-vs-first-token 的模式。"
        )
    )
    parser.add_argument("--exp-dir", required=True, help="实验目录，如 exp/fsmn_ctc_xlxl_distill_199k_replay_from79")
    parser.add_argument("--test-id", required=True, help="评测 id，如 119 / 229 / test_229")
    parser.add_argument("--keyword", default="小雷小雷", help="目标关键词，默认小雷小雷")
    parser.add_argument("--keywords", default=DEFAULT_KEYWORDS, help="诊断时加载的关键词列表")
    parser.add_argument("--data-list", default=str(DEFAULT_DATA_LIST), help="评测数据 data.list")
    parser.add_argument("--checkpoint", default="", help="显式指定 checkpoint；默认按 test-id 推断")
    parser.add_argument("--config", default="", help="显式指定 config.yaml；默认 exp-dir/config.yaml")
    parser.add_argument("--dict-dir", default=str(DEFAULT_DICT_DIR), help="词表目录")
    parser.add_argument("--stats-dir", default="", help="显式指定评测目录；默认按 test-id 推断")
    parser.add_argument("--dump-dir", default="", help="诊断报告输出目录；默认 exp-dir/analysis/miss_diag_<keyword>_<test>")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id，传 -1 表示 CPU")
    parser.add_argument("--beam-size", type=int, default=8, help="beam top-k")
    parser.add_argument("--frame-topk", type=int, default=8, help="平均帧 top-k token")
    parser.add_argument("--max-misses", type=int, default=0, help="只分析前 N 条漏检，0 表示全部")
    return parser.parse_args()


def compact_text(text: str) -> str:
    return "".join(str(text).split())


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


def classify_miss(record: Dict, score_item: Optional[Dict], target_keyword: str) -> str:
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


def artifact_dir_for_wav(dump_dir: Path, wav_path: Path) -> Path:
    digest = hashlib.sha1(str(wav_path).encode("utf-8")).hexdigest()[:10]
    return dump_dir / f"{wav_path.stem}_{digest}"


def load_or_run_report(
    wav_path: Path,
    dump_dir: Path,
    diagnose_args,
    resources: Dict,
    id2tok: Dict[int, str],
) -> Dict:
    report_path = artifact_dir_for_wav(dump_dir, wav_path) / "report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    return dw.diagnose_one_wav(wav_path, diagnose_args, resources, id2tok)


def build_resources(args, exp_dir: Path, checkpoint: Path, config: Path, dict_dir: Path, stats_dir: Path, dump_dir: Path):
    model_args = SimpleNamespace(
        model="s3",
        checkpoint=str(checkpoint),
        model_dir="",
        checkpoint_name="",
        config=str(config),
        dict_dir=str(dict_dir),
        stats_dir=str(stats_dir),
        keywords=args.keywords,
        gpu=args.gpu,
        indent=2,
    )
    infer_args = dw.build_model_args(model_args)
    keywords = iw.parse_keywords_arg(args.keywords)
    model_info = iw.resolve_model_paths(infer_args)
    configs = iw.load_config(model_info["config"])
    model, device, is_jit = iw.load_model(model_info["checkpoint"], configs, args.gpu)
    threshold_map = iw.load_threshold_map(infer_args, model_info, keywords)
    time_resolution_sec = iw.get_time_resolution_sec(configs)
    keywords_token, keywords_idxset = iw.build_keyword_token_info(keywords, model_info["dict_dir"])
    id2tok = dw.load_id2token(model_info["dict_dir"])
    dump_dir.mkdir(parents=True, exist_ok=True)
    resources = {
        "keywords": keywords,
        "model_info": model_info,
        "configs": configs,
        "model": model,
        "device": device,
        "is_jit": is_jit,
        "threshold_map": threshold_map,
        "time_resolution_sec": time_resolution_sec,
        "keywords_token": keywords_token,
        "keywords_idxset": keywords_idxset,
        "dump_dir": dump_dir,
    }
    return resources, id2tok


def find_keyword_diag(report: Dict, keyword: str) -> Optional[Dict]:
    target = compact_text(keyword)
    for item in report.get("keyword_diagnostics", []):
        if compact_text(item.get("keyword", "")) == target:
            return item
    return None


def trace_stats_by_token(report: Dict) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for item in report.get("token_traces", {}).get("traces", []):
        probs = item.get("probs", []) or []
        mean_prob = float(sum(probs) / len(probs)) if probs else 0.0
        stats[item.get("token", "")] = {
            "peak_prob": float(item.get("peak_prob") or 0.0),
            "peak_frame": int(item.get("peak_frame") or -1),
            "mean_prob": mean_prob,
        }
    return stats


def summarize_reports(keyword: str, miss_items: List[Dict], reports: List[Dict]) -> Dict:
    miss_reason_counter = Counter(item["miss_reason"] for item in miss_items)
    greedy_counter = Counter()
    beam_counter = Counter()
    diag_status_counter = Counter()
    assessment_counter = Counter()

    exact_greedy_lei_xiao_lei = 0
    exact_beam_lei_xiao_lei = 0
    prefix_missing = 0
    blank_peak_gt_first = 0
    blank_mean_gt_first = 0
    first_peak_after_second = 0
    likely_blank_swallow_prefix = 0

    first_token_peak_values: List[float] = []
    second_token_peak_values: List[float] = []
    blank_peak_values: List[float] = []

    manifest_rows: List[Dict] = []

    for miss_item, report in zip(miss_items, reports):
        greedy_text = report.get("greedy_decode", {}).get("text") or "<empty>"
        beam_top1_text = "<none>"
        if report.get("beam_topk"):
            beam_top1_text = report["beam_topk"][0].get("text") or "<empty>"

        greedy_counter[greedy_text] += 1
        beam_counter[beam_top1_text] += 1

        keyword_diag = find_keyword_diag(report, keyword)
        diag_status = keyword_diag.get("status") if keyword_diag else "missing_keyword_diag"
        diag_status_counter[diag_status] += 1

        assessment = report.get("decode_assessment", {}).get("status") or "missing_decode_assessment"
        assessment_counter[assessment] += 1

        keyword_tokens = keyword_diag.get("keyword_tokens", []) if keyword_diag else []
        first_token = keyword_tokens[0] if len(keyword_tokens) >= 1 else None
        second_token = keyword_tokens[1] if len(keyword_tokens) >= 2 else None
        trace_stats = trace_stats_by_token(report)
        blank_stats = trace_stats.get("<blk>", {"peak_prob": 0.0, "peak_frame": -1, "mean_prob": 0.0})
        first_stats = trace_stats.get(first_token or "", {"peak_prob": 0.0, "peak_frame": -1, "mean_prob": 0.0})
        second_stats = trace_stats.get(second_token or "", {"peak_prob": 0.0, "peak_frame": -1, "mean_prob": 0.0})

        blank_peak_values.append(blank_stats["peak_prob"])
        first_token_peak_values.append(first_stats["peak_prob"])
        second_token_peak_values.append(second_stats["peak_prob"])

        is_exact_greedy_lei_xiao_lei = greedy_text == "雷 小 雷"
        is_exact_beam_lei_xiao_lei = beam_top1_text == "雷 小 雷"
        is_prefix_missing = bool(first_token) and not greedy_text.startswith(first_token)
        is_blank_peak_gt_first = blank_stats["peak_prob"] > first_stats["peak_prob"]
        is_blank_mean_gt_first = blank_stats["mean_prob"] > first_stats["mean_prob"]
        is_first_peak_after_second = (
            first_stats["peak_frame"] >= 0
            and second_stats["peak_frame"] >= 0
            and first_stats["peak_frame"] > second_stats["peak_frame"]
        )
        is_likely_blank_swallow_prefix = (
            diag_status == "no_complete_match"
            and is_prefix_missing
            and is_blank_peak_gt_first
            and is_first_peak_after_second
        )

        exact_greedy_lei_xiao_lei += int(is_exact_greedy_lei_xiao_lei)
        exact_beam_lei_xiao_lei += int(is_exact_beam_lei_xiao_lei)
        prefix_missing += int(is_prefix_missing)
        blank_peak_gt_first += int(is_blank_peak_gt_first)
        blank_mean_gt_first += int(is_blank_mean_gt_first)
        first_peak_after_second += int(is_first_peak_after_second)
        likely_blank_swallow_prefix += int(is_likely_blank_swallow_prefix)

        manifest_rows.append(
            {
                "key": miss_item["key"],
                "wav": miss_item["wav"],
                "miss_reason": miss_item["miss_reason"],
                "greedy_text": greedy_text,
                "beam_top1_text": beam_top1_text,
                "decode_assessment": assessment,
                "keyword_diag_status": diag_status,
                "first_token": first_token,
                "second_token": second_token,
                "blank_peak_prob": blank_stats["peak_prob"],
                "first_token_peak_prob": first_stats["peak_prob"],
                "second_token_peak_prob": second_stats["peak_prob"],
                "blank_mean_prob": blank_stats["mean_prob"],
                "first_token_mean_prob": first_stats["mean_prob"],
                "second_token_mean_prob": second_stats["mean_prob"],
                "blank_peak_frame": blank_stats["peak_frame"],
                "first_token_peak_frame": first_stats["peak_frame"],
                "second_token_peak_frame": second_stats["peak_frame"],
                "exact_greedy_lei_xiao_lei": is_exact_greedy_lei_xiao_lei,
                "exact_beam_lei_xiao_lei": is_exact_beam_lei_xiao_lei,
                "prefix_missing": is_prefix_missing,
                "blank_peak_gt_first": is_blank_peak_gt_first,
                "blank_mean_gt_first": is_blank_mean_gt_first,
                "first_peak_after_second": is_first_peak_after_second,
                "likely_blank_swallow_prefix": is_likely_blank_swallow_prefix,
                "artifact_dir": report.get("artifact_dir"),
            }
        )

    summary = {
        "keyword": keyword,
        "total_misses": len(miss_items),
        "miss_reason_counts": dict(miss_reason_counter),
        "top_greedy_patterns": greedy_counter.most_common(10),
        "top_beam_top1_patterns": beam_counter.most_common(10),
        "keyword_diag_status_counts": dict(diag_status_counter),
        "decode_assessment_counts": dict(assessment_counter),
        "exact_greedy_雷小雷": exact_greedy_lei_xiao_lei,
        "exact_beam_top1_雷小雷": exact_beam_lei_xiao_lei,
        "greedy_missing_first_token": prefix_missing,
        "blank_peak_gt_first_token_peak": blank_peak_gt_first,
        "blank_mean_gt_first_token_mean": blank_mean_gt_first,
        "first_token_peak_after_second_token_peak": first_peak_after_second,
        "likely_blank_swallow_prefix": likely_blank_swallow_prefix,
        "avg_blank_peak_prob": (sum(blank_peak_values) / len(blank_peak_values)) if blank_peak_values else 0.0,
        "avg_first_token_peak_prob": (sum(first_token_peak_values) / len(first_token_peak_values)) if first_token_peak_values else 0.0,
        "avg_second_token_peak_prob": (sum(second_token_peak_values) / len(second_token_peak_values)) if second_token_peak_values else 0.0,
    }
    return {"summary": summary, "manifest_rows": manifest_rows}


def print_summary(exp_dir: Path, stats_dir: Path, summary: Dict):
    print("=" * 64)
    print(f"exp_dir:   {exp_dir}")
    print(f"stats_dir: {stats_dir}")
    print(f"keyword:   {summary['keyword']}")
    print(f"misses:    {summary['total_misses']}")
    print("=" * 64)
    print("miss_reason_counts:")
    for key, value in summary["miss_reason_counts"].items():
        print(f"  {key}: {value}")
    print("top_greedy_patterns:")
    for text, value in summary["top_greedy_patterns"]:
        print(f"  {value:3d}  {text}")
    print("top_beam_top1_patterns:")
    for text, value in summary["top_beam_top1_patterns"]:
        print(f"  {value:3d}  {text}")
    print("keyword_diag_status_counts:")
    for key, value in summary["keyword_diag_status_counts"].items():
        print(f"  {key}: {value}")
    print("decode_assessment_counts:")
    for key, value in summary["decode_assessment_counts"].items():
        print(f"  {key}: {value}")
    print("key counters:")
    print(f"  exact_greedy_雷小雷:                 {summary['exact_greedy_雷小雷']}")
    print(f"  exact_beam_top1_雷小雷:              {summary['exact_beam_top1_雷小雷']}")
    print(f"  greedy_missing_first_token:         {summary['greedy_missing_first_token']}")
    print(f"  blank_peak_gt_first_token_peak:     {summary['blank_peak_gt_first_token_peak']}")
    print(f"  blank_mean_gt_first_token_mean:     {summary['blank_mean_gt_first_token_mean']}")
    print(f"  first_token_peak_after_second_peak: {summary['first_token_peak_after_second_token_peak']}")
    print(f"  likely_blank_swallow_prefix:        {summary['likely_blank_swallow_prefix']}")
    print("avg peak probs:")
    print(f"  avg_blank_peak_prob:       {summary['avg_blank_peak_prob']:.4f}")
    print(f"  avg_first_token_peak_prob: {summary['avg_first_token_peak_prob']:.4f}")
    print(f"  avg_second_token_peak_prob:{summary['avg_second_token_peak_prob']:.4f}")


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir).expanduser().resolve()
    data_list = Path(args.data_list).expanduser().resolve()
    dict_dir = Path(args.dict_dir).expanduser().resolve()
    stats_dir = Path(args.stats_dir).expanduser().resolve() if args.stats_dir else exp_dir / normalize_test_id(args.test_id)
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else exp_dir / infer_checkpoint_name(args.test_id)
    config = Path(args.config).expanduser().resolve() if args.config else exp_dir / "config.yaml"
    dump_dir = (
        Path(args.dump_dir).expanduser().resolve()
        if args.dump_dir
        else exp_dir / "analysis" / f"miss_diag_{compact_text(args.keyword)}_{stats_dir.name}"
    )

    score_path = stats_dir / "score.txt"
    if not score_path.exists():
        raise FileNotFoundError(f"score.txt not found: {score_path}")
    if not data_list.exists():
        raise FileNotFoundError(f"data.list not found: {data_list}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not config.exists():
        raise FileNotFoundError(f"config not found: {config}")

    target_records = load_target_records(data_list, args.keyword)
    score_map = parse_score_file(score_path)

    miss_items: List[Dict] = []
    hit_count = 0
    for record in target_records:
        score_item = score_map.get(record["key"])
        miss_reason = classify_miss(record, score_item, args.keyword)
        if miss_reason == "detected_target":
            hit_count += 1
            continue
        miss_items.append(
            {
                "key": record["key"],
                "wav": record["wav"],
                "txt": record["txt"],
                "miss_reason": miss_reason,
            }
        )

    if args.max_misses > 0:
        miss_items = miss_items[: args.max_misses]

    resources, id2tok = build_resources(args, exp_dir, checkpoint, config, dict_dir, stats_dir, dump_dir)
    diagnose_args = SimpleNamespace(beam_size=args.beam_size, frame_topk=args.frame_topk)

    reports: List[Dict] = []
    for item in miss_items:
        wav_path = Path(item["wav"]).expanduser().resolve()
        report = load_or_run_report(wav_path, dump_dir, diagnose_args, resources, id2tok)
        reports.append(report)

    bundle = summarize_reports(args.keyword, miss_items, reports)
    summary = bundle["summary"]
    summary["total_target_positives"] = len(target_records)
    summary["detected_target"] = hit_count
    summary["misses_after_limit"] = len(miss_items)

    dump_dir.mkdir(parents=True, exist_ok=True)
    summary_path = dump_dir / "summary.json"
    manifest_path = dump_dir / "miss_manifest.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with manifest_path.open("w", encoding="utf-8") as fout:
        for row in bundle["manifest_rows"]:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print_summary(exp_dir, stats_dir, summary)
    print(f"summary_json:  {summary_path}")
    print(f"manifest_jsonl:{manifest_path}")


if __name__ == "__main__":
    main()
