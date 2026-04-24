#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "基于 teacher-score 和 student-score 从 train split 挖出 "
            "teacher-hit / student-miss 的 hard positives，并把它们重复写入新的训练集。"
        )
    )
    parser.add_argument("--source-data-dir", required=True, help="原始数据目录，例如 data_xlxl_0327_ctc_v1_clean")
    parser.add_argument("--teacher-score", required=True, help="老师模型 score.txt，例如 exp/.../train_159/score.txt")
    parser.add_argument("--student-score", required=True, help="学生模型 score.txt，例如 exp/.../train_119/score.txt")
    parser.add_argument("--keyword", default="小雷小雷", help="目标关键词，默认小雷小雷")
    parser.add_argument("--replay-factor", type=int, default=3, help="hard positive 额外重复次数，默认 3")
    parser.add_argument("--output-dir", required=True, help="输出增强后的数据目录")
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
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.replay_factor < 1:
        raise ValueError(f"--replay-factor must be >= 1, got {args.replay_factor}")

    train_path = source_data_dir / "train" / "data.list"
    dev_path = source_data_dir / "dev" / "data.list"
    test_path = source_data_dir / "test" / "data.list"
    for path in [train_path, dev_path, test_path, teacher_score, student_score]:
        if not path.exists():
            raise FileNotFoundError(f"not found: {path}")

    teacher_rows = parse_score_file(teacher_score)
    student_rows = parse_score_file(student_score)
    train_rows = load_jsonl(train_path)

    keyword_compact = compact_text(args.keyword)
    hard_positive_rows: List[Dict] = []
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
        if is_detected_target(teacher_item, args.keyword) and not is_detected_target(student_item, args.keyword):
            hard_positive_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "dev", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    augmented_train_rows = list(train_rows)
    for row in hard_positive_rows:
        augmented_train_rows.extend([row] * args.replay_factor)

    write_jsonl(output_dir / "train" / "data.list", augmented_train_rows)
    write_jsonl(output_dir / "dev" / "data.list", load_jsonl(dev_path))
    write_jsonl(output_dir / "test" / "data.list", load_jsonl(test_path))

    for name in ["global_cmvn.kaldi", "summary.json"]:
        src = source_data_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    metadata = {
        "source_data_dir": str(source_data_dir),
        "teacher_score": str(teacher_score),
        "student_score": str(student_score),
        "keyword": args.keyword,
        "replay_factor": args.replay_factor,
        "train_total_rows": len(train_rows),
        "train_positive_rows_for_keyword": train_positive_rows,
        "hard_positive_count": len(hard_positive_rows),
        "augmented_train_total_rows": len(augmented_train_rows),
        "added_rows": len(hard_positive_rows) * args.replay_factor,
        "score_missing_counter": dict(score_missing_counter),
    }
    (output_dir / "hard_positive_replay_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "hard_positive_rows.jsonl", hard_positive_rows)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
