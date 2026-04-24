#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio


warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "把选中的小雷小雷 hard positive 拼成 repeated sequence 训练样本，"
            "同时把标签扩展成多遍关键词，用于训练连续流式多次触发能力。"
        )
    )
    parser.add_argument("--source-data-dir", required=True, help="基础数据目录，dev/test 原样复制")
    parser.add_argument("--rows-jsonl", required=True, help="待拼接的样本行，例如 prefix-start 子集")
    parser.add_argument("--output-dir", required=True, help="输出数据目录")
    parser.add_argument("--repeat", type=int, default=3, help="每条 wav 拼接次数，默认 3")
    parser.add_argument("--gap-ms", type=float, default=0.0, help="拼接间隔静音长度，默认 0")
    parser.add_argument("--add-factor", type=int, default=1, help="每条选中样本额外生成几条 repeated row，默认 1")
    parser.add_argument("--max-rows", type=int, default=0, help="最多使用多少条选中样本，0 表示全部")
    parser.add_argument("--key-prefix", default="repeatseq", help="新样本 key 前缀")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]):
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def repeated_label(txt: str, repeat: int) -> str:
    tokens = str(txt).split()
    if not tokens:
        raise ValueError("empty txt cannot be repeated")
    return " ".join(tokens * repeat)


def stable_id(row: Dict, repeat: int, gap_ms: float, factor_index: int) -> str:
    raw = f"{row.get('key','')}|{row.get('wav','')}|{repeat}|{gap_ms}|{factor_index}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def build_repeated_wav(row: Dict, repeat: int, gap_ms: float, output_path: Path) -> float:
    wav_path = Path(row["wav"]).expanduser()
    waveform, sample_rate = torchaudio.load(str(wav_path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
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
    return float(merged.size(1) / sample_rate)


def main():
    args = parse_args()
    if args.repeat < 2:
        raise ValueError(f"--repeat must be >= 2, got {args.repeat}")
    if args.add_factor < 1:
        raise ValueError(f"--add-factor must be >= 1, got {args.add_factor}")

    source_data_dir = Path(args.source_data_dir).expanduser().resolve()
    rows_jsonl = Path(args.rows_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    train_path = source_data_dir / "train" / "data.list"
    dev_path = source_data_dir / "dev" / "data.list"
    test_path = source_data_dir / "test" / "data.list"
    for path in [train_path, dev_path, test_path, rows_jsonl]:
        if not path.exists():
            raise FileNotFoundError(f"not found: {path}")

    base_train_rows = load_jsonl(train_path)
    selected_rows = load_jsonl(rows_jsonl)
    if args.max_rows > 0:
        selected_rows = selected_rows[: args.max_rows]

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "dev", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / "repeated_wavs"

    repeated_rows: List[Dict] = []
    for row in selected_rows:
        for factor_index in range(args.add_factor):
            item_id = stable_id(row, args.repeat, args.gap_ms, factor_index)
            key = f"{args.key_prefix}_x{args.repeat}_{item_id}_{row['key']}"
            wav_path = wav_dir / f"{key}.wav"
            duration = build_repeated_wav(row, args.repeat, args.gap_ms, wav_path)
            new_row = dict(row)
            new_row.update(
                {
                    "key": key,
                    "wav": str(wav_path),
                    "txt": repeated_label(row["txt"], args.repeat),
                    "duration": duration,
                    "source_key": row.get("key"),
                    "source_wav": row.get("wav"),
                    "repeat": args.repeat,
                    "gap_ms": args.gap_ms,
                }
            )
            repeated_rows.append(new_row)

    write_jsonl(output_dir / "train" / "data.list", base_train_rows + repeated_rows)
    write_jsonl(output_dir / "dev" / "data.list", load_jsonl(dev_path))
    write_jsonl(output_dir / "test" / "data.list", load_jsonl(test_path))
    write_jsonl(output_dir / "selected_rows.jsonl", selected_rows)
    write_jsonl(output_dir / "repeated_rows.jsonl", repeated_rows)

    for name in ["global_cmvn.kaldi", "summary.json"]:
        src = source_data_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    metadata = {
        "source_data_dir": str(source_data_dir),
        "rows_jsonl": str(rows_jsonl),
        "output_dir": str(output_dir),
        "repeat": args.repeat,
        "gap_ms": args.gap_ms,
        "add_factor": args.add_factor,
        "selected_rows": len(selected_rows),
        "base_train_total_rows": len(base_train_rows),
        "added_rows": len(repeated_rows),
        "augmented_train_total_rows": len(base_train_rows) + len(repeated_rows),
        "label_example": repeated_rows[0]["txt"] if repeated_rows else "",
    }
    (output_dir / "repeated_sequence_replay_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
