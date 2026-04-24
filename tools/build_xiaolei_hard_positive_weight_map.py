#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "基于 teacher-score 和 student-score 挖出 "
            "teacher-hit / student-miss 的 hard positives，"
            "并输出 key->weight 的 sample-weight 映射。"
        )
    )
    parser.add_argument("--source-data-dir", required=True,
                        help="原始数据目录，例如 data_xlxl_0327_ctc_v1_clean")
    parser.add_argument("--teacher-score", required=True,
                        help="老师模型 score.txt，例如 exp/.../train_159/score.txt")
    parser.add_argument("--student-score", required=True,
                        help="学生模型 score.txt，例如 exp/.../train_119/score.txt")
    parser.add_argument("--keyword", default="小雷小雷",
                        help="目标关键词，默认小雷小雷")
    parser.add_argument("--hard-weight", type=float, default=5.0,
                        help="hard positive 权重，默认 5.0")
    parser.add_argument("--default-weight", type=float, default=1.0,
                        help="普通样本默认权重，默认 1.0")
    parser.add_argument("--output-json", required=True,
                        help="输出 key->weight 的 JSON 文件")
    parser.add_argument("--output-summary", default=None,
                        help="可选 summary.json 输出路径")
    parser.add_argument("--output-hard-rows", default=None,
                        help="可选 hard positive jsonl 输出路径")
    return parser.parse_args()


def compact_text(text: str) -> str:
    return "".join(str(text).split())


def parse_score_file(path: Path) -> Dict[str, Dict]:
    rows: Dict[str, Dict] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
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
        rows[key] = item
    return rows


def is_detected_target(score_item: Dict, keyword: str) -> bool:
    if not score_item:
        return False
    if score_item.get("status") != "detected":
        return False
    return compact_text(score_item.get("keyword") or "") == compact_text(keyword)


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
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    source_data_dir = Path(args.source_data_dir).expanduser().resolve()
    teacher_score = Path(args.teacher_score).expanduser().resolve()
    student_score = Path(args.student_score).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_summary = (Path(args.output_summary).expanduser().resolve()
                      if args.output_summary else None)
    output_hard_rows = (Path(args.output_hard_rows).expanduser().resolve()
                        if args.output_hard_rows else None)

    if args.hard_weight <= 0:
        raise ValueError(f"--hard-weight must be > 0, got {args.hard_weight}")
    if args.default_weight <= 0:
        raise ValueError(
            f"--default-weight must be > 0, got {args.default_weight}")

    train_path = source_data_dir / "train" / "data.list"
    for path in [train_path, teacher_score, student_score]:
        if not path.exists():
            raise FileNotFoundError(f"not found: {path}")

    teacher_rows = parse_score_file(teacher_score)
    student_rows = parse_score_file(student_score)
    train_rows = load_jsonl(train_path)

    keyword_compact = compact_text(args.keyword)
    hard_positive_rows: List[Dict] = []
    weight_map: Dict[str, float] = {}
    train_positive_rows = 0
    score_missing_counter = Counter()

    for row in train_rows:
        if compact_text(row.get("txt", "")) != keyword_compact:
            continue
        train_positive_rows += 1
        key = row.get("key")
        teacher_item = teacher_rows.get(key)
        student_item = student_rows.get(key)
        if teacher_item is None:
            score_missing_counter["teacher_missing"] += 1
            continue
        if student_item is None:
            score_missing_counter["student_missing"] += 1
            continue
        if (is_detected_target(teacher_item, args.keyword)
                and not is_detected_target(student_item, args.keyword)):
            hard_positive_rows.append(row)
            weight_map[str(key)] = float(args.hard_weight)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(weight_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "source_data_dir": str(source_data_dir),
        "teacher_score": str(teacher_score),
        "student_score": str(student_score),
        "keyword": args.keyword,
        "default_weight": args.default_weight,
        "hard_weight": args.hard_weight,
        "train_total_rows": len(train_rows),
        "train_positive_rows_for_keyword": train_positive_rows,
        "hard_positive_count": len(hard_positive_rows),
        "weighted_key_count": len(weight_map),
        "score_missing_counter": dict(score_missing_counter),
        "output_json": str(output_json),
    }

    if output_summary is not None:
        output_summary.parent.mkdir(parents=True, exist_ok=True)
        output_summary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if output_hard_rows is not None:
        output_hard_rows.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_hard_rows, hard_positive_rows)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
