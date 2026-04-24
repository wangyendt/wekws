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
S0_CANDIDATE = REPO_ROOT / "examples" / "hi_xiaowen" / "s0"
S0_DIR = S0_CANDIDATE if (S0_CANDIDATE / "diagnose_wav.py").exists() else REPO_ROOT
if str(S0_DIR) not in sys.path:
    sys.path.insert(0, str(S0_DIR))

import diagnose_wav as dw
import infer_wav as iw


DEFAULT_DICT_DIR = S0_DIR / "dict_top20_xlxl"
DEFAULT_KEYWORDS = "小雷小雷,小雷快拍"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "用老师模型诊断 hard positives，导出 token peak sidecar，"
            "供蒸馏训练中的局部 token CE 使用。"
        )
    )
    parser.add_argument("--rows-jsonl", required=True, help="输入 hard positive rows jsonl")
    parser.add_argument("--checkpoint", required=True, help="老师模型 checkpoint")
    parser.add_argument("--config", default="", help="老师 config.yaml，默认与 checkpoint 同目录")
    parser.add_argument("--dict-dir", default=str(DEFAULT_DICT_DIR), help="词表目录")
    parser.add_argument("--stats-dir", required=True, help="老师已有评测目录，用于加载 threshold")
    parser.add_argument("--keyword", default="小雷小雷", help="目标关键词")
    parser.add_argument("--keywords", default=DEFAULT_KEYWORDS, help="完整关键词列表")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id，-1 表示 CPU")
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--frame-topk", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0, help="只处理前 N 条，0 表示全部")
    parser.add_argument("--min-token-prob", type=float, default=0.0,
                        help="只保留 teacher token prob >= 该值的峰值")
    parser.add_argument("--output-json", required=True, help="输出 key->targets JSON")
    parser.add_argument("--output-summary", required=True, help="输出 summary JSON")
    parser.add_argument("--output-manifest", required=True, help="输出逐条诊断 manifest JSONL")
    return parser.parse_args()


def compact_text(text: str) -> str:
    return "".join(str(text).split())


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
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


def token_to_id_map(id2tok: Dict[int, str]) -> Dict[str, int]:
    return {token: int(idx) for idx, token in id2tok.items()}


def build_targets(keyword_diag: Dict, tok2id: Dict[str, int],
                  min_token_prob: float) -> List[Dict]:
    frames = keyword_diag.get("matched_frames") or []
    tokens = keyword_diag.get("matched_tokens") or []
    probs = keyword_diag.get("token_probs") or []
    targets: List[Dict] = []
    for index, (frame, token) in enumerate(zip(frames, tokens)):
        if token not in tok2id:
            continue
        prob = float(probs[index]) if index < len(probs) else None
        if prob is not None and prob < min_token_prob:
            continue
        try:
            frame_int = int(frame)
        except (TypeError, ValueError):
            continue
        if frame_int < 0:
            continue
        item = {
            "frame": frame_int,
            "token": token,
            "token_id": tok2id[token],
        }
        if prob is not None:
            item["prob"] = prob
        targets.append(item)
    return targets


def main():
    args = parse_args()
    rows_path = Path(args.rows_jsonl).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_summary = Path(args.output_summary).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser().resolve()

    rows = load_jsonl(rows_path)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    resources, id2tok = build_resources(args)
    tok2id = token_to_id_map(id2tok)

    target_map: Dict[str, Dict] = {}
    manifest_rows: List[Dict] = []
    status_counter: Counter = Counter()
    target_count_counter: Counter = Counter()
    token_counter: Counter = Counter()

    for index, row in enumerate(rows, start=1):
        key = str(row.get("key", ""))
        wav_path = Path(row["wav"]).expanduser().resolve()
        report = dw.diagnose_one_wav(wav_path, args, resources, id2tok)
        keyword_diag = find_keyword_diag(report, args.keyword) or {}
        status = keyword_diag.get("status", "missing")
        targets = build_targets(keyword_diag, tok2id, args.min_token_prob)

        if key and targets:
            target_map[key] = {
                "key": key,
                "wav": row.get("wav"),
                "txt": row.get("txt"),
                "targets": targets,
                "teacher_status": status,
            }

        status_counter[status] += 1
        target_count_counter[len(targets)] += 1
        token_counter.update([item["token"] for item in targets])
        manifest_rows.append(
            {
                "index": index,
                "key": key,
                "wav": row.get("wav"),
                "txt": row.get("txt"),
                "teacher_status": status,
                "targets": targets,
                "greedy_text": report.get("greedy_decode", {}).get("text"),
                "beam_top1_text": (
                    report.get("beam_topk", [{}])[0].get("text")
                    if report.get("beam_topk") else None
                ),
            }
        )

    summary = {
        "rows_jsonl": str(rows_path),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "stats_dir": str(Path(args.stats_dir).expanduser().resolve()),
        "keyword": args.keyword,
        "total_rows": len(rows),
        "keys_with_targets": len(target_map),
        "min_token_prob": args.min_token_prob,
        "status_counter": dict(status_counter),
        "target_count_counter": {str(k): v for k, v in target_count_counter.items()},
        "token_counter": dict(token_counter),
        "output_json": str(output_json),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(target_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_manifest, manifest_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
