#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(
        description="把给定 rows.jsonl 中的样本额外重复写入 train split，生成新的 replay 数据目录。"
    )
    parser.add_argument("--source-data-dir", required=True, help="原始数据目录，例如 data_xlxl_0327_ctc_v1_clean")
    parser.add_argument("--rows-jsonl", required=True, help="待 replay 的样本行，例如 prefix-start 子集")
    parser.add_argument("--replay-factor", type=int, default=10, help="额外重复次数，默认 10")
    parser.add_argument("--output-dir", required=True, help="输出数据目录")
    return parser.parse_args()


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
    if args.replay_factor < 1:
        raise ValueError(f"--replay-factor must be >= 1, got {args.replay_factor}")

    source_data_dir = Path(args.source_data_dir).expanduser().resolve()
    rows_jsonl = Path(args.rows_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    train_path = source_data_dir / "train" / "data.list"
    dev_path = source_data_dir / "dev" / "data.list"
    test_path = source_data_dir / "test" / "data.list"
    for path in [train_path, dev_path, test_path, rows_jsonl]:
        if not path.exists():
            raise FileNotFoundError(f"not found: {path}")

    train_rows = load_jsonl(train_path)
    selected_rows = load_jsonl(rows_jsonl)
    augmented_train_rows = list(train_rows)
    for row in selected_rows:
        augmented_train_rows.extend([row] * args.replay_factor)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "dev", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "train" / "data.list", augmented_train_rows)
    write_jsonl(output_dir / "dev" / "data.list", load_jsonl(dev_path))
    write_jsonl(output_dir / "test" / "data.list", load_jsonl(test_path))

    for name in ["global_cmvn.kaldi", "summary.json"]:
        src = source_data_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    metadata = {
        "source_data_dir": str(source_data_dir),
        "rows_jsonl": str(rows_jsonl),
        "replay_factor": args.replay_factor,
        "selected_rows": len(selected_rows),
        "train_total_rows": len(train_rows),
        "augmented_train_total_rows": len(augmented_train_rows),
        "added_rows": len(selected_rows) * args.replay_factor,
    }
    (output_dir / "selected_replay_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "selected_rows.jsonl", selected_rows)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
