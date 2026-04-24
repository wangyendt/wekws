#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
S0_DIR = REPO_ROOT / "examples" / "hi_xiaowen" / "s0"
if str(S0_DIR) not in sys.path:
    sys.path.insert(0, str(S0_DIR))

import diagnose_wav as dw
import infer_wav as iw


DEFAULT_DICT_DIR = S0_DIR / "dict_top20_xlxl"
DEFAULT_KEYWORDS = "小雷小雷,小雷快拍"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "对已挖出的 hard positives 再做一次 student 诊断，筛出“前缀起步失败 / 雷小雷塌缩”子集。"
        )
    )
    parser.add_argument("--rows-jsonl", required=True, help="hard_positive_rows.jsonl")
    parser.add_argument("--checkpoint", required=True, help="学生模型 checkpoint，例如 exp/.../119.pt")
    parser.add_argument("--config", default="", help="config.yaml，默认与 checkpoint 同实验目录")
    parser.add_argument("--dict-dir", default=str(DEFAULT_DICT_DIR), help="词表目录")
    parser.add_argument("--stats-dir", required=True, help="学生模型已有评测目录，用于复用 threshold_map")
    parser.add_argument("--keyword", default="小雷小雷", help="目标关键词，默认小雷小雷")
    parser.add_argument("--keywords", default=DEFAULT_KEYWORDS, help="完整关键词列表")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id，传 -1 表示 CPU")
    parser.add_argument("--beam-size", type=int, default=8, help="beam size")
    parser.add_argument("--frame-topk", type=int, default=8, help="frame top-k")
    parser.add_argument(
        "--select-mode",
        default="any_prefix_signal",
        choices=["any_prefix_signal", "greedy_not_start"],
        help="子集选择规则：默认任一前缀异常；也可只保留 greedy 不以第一个 token 开头的样本",
    )
    parser.add_argument("--weak-first-prob-threshold", type=float, default=1e-3, help="认为第一个 token 极弱的阈值")
    parser.add_argument("--max-rows", type=int, default=0, help="只处理前 N 条，0 表示全部")
    parser.add_argument("--output-rows-jsonl", required=True, help="输出筛出的 prefix-start 子集")
    parser.add_argument("--output-summary", required=True, help="输出 summary.json")
    parser.add_argument("--output-manifest", required=True, help="输出 manifest.jsonl")
    return parser.parse_args()


def compact_text(text: str) -> str:
    return "".join(str(text).split())


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_keyword_diag(report: Dict, keyword: str) -> Optional[Dict]:
    target = compact_text(keyword)
    for item in report.get("keyword_diagnostics", []):
        if compact_text(item.get("keyword", "")) == target:
            return item
    return None


def build_resources(args):
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config = Path(args.config).expanduser().resolve() if args.config else checkpoint.parent / "config.yaml"
    dict_dir = Path(args.dict_dir).expanduser().resolve()
    stats_dir = Path(args.stats_dir).expanduser().resolve()

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
        "dump_dir": None,
    }
    return resources, id2tok


def top_counter_rows(counter: Counter, topn: int = 20) -> List[Dict]:
    return [{"label": label, "count": count} for label, count in counter.most_common(topn)]


def main():
    args = parse_args()
    rows_path = Path(args.rows_jsonl).expanduser().resolve()
    output_rows = Path(args.output_rows_jsonl).expanduser().resolve()
    output_summary = Path(args.output_summary).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser().resolve()

    rows = load_jsonl(rows_path)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    resources, id2tok = build_resources(args)

    selected_rows: List[Dict] = []
    manifest_rows: List[Dict] = []
    reason_counter: Counter = Counter()
    greedy_counter: Counter = Counter()
    beam_counter: Counter = Counter()
    weakest_token_counter: Counter = Counter()

    for row in rows:
        wav_path = Path(row["wav"]).expanduser().resolve()
        report = dw.diagnose_one_wav(wav_path, args, resources, id2tok)
        greedy_text = report.get("greedy_decode", {}).get("text") or "<empty>"
        beam_top1_text = report.get("beam_topk", [{}])[0].get("text", "<none>") if report.get("beam_topk") else "<none>"
        keyword_diag = find_keyword_diag(report, args.keyword) or {}
        keyword_tokens = keyword_diag.get("keyword_tokens") or ["小", "雷", "小", "雷"]
        first_token = keyword_tokens[0]
        weakest_token = keyword_diag.get("weakest_token") or ""
        weakest_prob = float(keyword_diag.get("weakest_token_prob") or 0.0)

        reasons: List[str] = []
        if not greedy_text.startswith(first_token):
            reasons.append("greedy_not_start_with_first")
        if beam_top1_text != "<none>" and not beam_top1_text.startswith(first_token):
            reasons.append("beam_not_start_with_first")
        if weakest_token == first_token and weakest_prob <= args.weak_first_prob_threshold:
            reasons.append("first_token_is_weakest")

        if args.select_mode == "greedy_not_start":
            is_selected = "greedy_not_start_with_first" in reasons
        else:
            is_selected = bool(reasons)
        if is_selected:
            selected_rows.append(row)
            reason_counter.update(reasons)

        greedy_counter[greedy_text] += 1
        beam_counter[beam_top1_text] += 1
        weakest_token_counter[weakest_token or "<none>"] += 1

        manifest_rows.append(
            {
                "key": row.get("key"),
                "wav": row.get("wav"),
                "txt": row.get("txt"),
                "selected": is_selected,
                "reasons": reasons,
                "greedy_text": greedy_text,
                "beam_top1_text": beam_top1_text,
                "weakest_token": weakest_token,
                "weakest_token_prob": weakest_prob,
            }
        )

    summary = {
        "rows_jsonl": str(rows_path),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "stats_dir": str(Path(args.stats_dir).expanduser().resolve()),
        "keyword": args.keyword,
        "select_mode": args.select_mode,
        "total_rows": len(rows),
        "selected_rows": len(selected_rows),
        "selected_ratio": (len(selected_rows) / len(rows)) if rows else 0.0,
        "weak_first_prob_threshold": args.weak_first_prob_threshold,
        "reason_counter": dict(reason_counter),
        "top_greedy_texts": top_counter_rows(greedy_counter),
        "top_beam_top1_texts": top_counter_rows(beam_counter),
        "weakest_token_counter": dict(weakest_token_counter),
    }

    write_jsonl(output_rows, selected_rows)
    write_jsonl(output_manifest, manifest_rows)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
