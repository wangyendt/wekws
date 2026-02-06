#!/usr/bin/env python3
"""
Build a reduced dict from model_vocab_freq_asr_sorted.txt.

Default behavior:
- Select tokens with cum_freq_pct <= 80.0, and include the first token
  that crosses the threshold (to match "80.027036" example).
- Force include special tokens: <blk> and <filler>
- Reindex tokens: <eps> -> -1, <blk> -> 0, <filler> -> 1, others -> 2..N-1

Outputs a new dict directory with dict.txt and words.txt.
"""

import argparse
import os
from typing import List, Tuple


SPECIAL_TOKENS = {"<blk>", "<filler>", "<eps>", "sil"}


def parse_sorted(path: str) -> List[Tuple[int, int, str, float]]:
    rows: List[Tuple[int, int, str, float]] = []
    with open(path, "r", encoding="utf8") as f:
        _header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            rank = int(parts[0])
            tok_id = int(parts[1])
            tok = parts[2]
            cum_pct = float(parts[5])
            rows.append((rank, tok_id, tok, cum_pct))
    return rows


def select_by_cum_pct(rows: List[Tuple[int, int, str, float]],
                      cum_pct: float,
                      include_cross: bool) -> List[str]:
    selected: List[str] = []
    crossed = False
    for _, _, tok, pct in rows:
        if pct <= cum_pct:
            selected.append(tok)
            continue
        if include_cross and not crossed:
            selected.append(tok)
            crossed = True
        break
    return selected


def select_by_num_keywords(rows: List[Tuple[int, int, str, float]],
                           num_keywords: int) -> List[str]:
    if num_keywords < 2:
        raise SystemExit("num_keywords must be >= 2 (for <blk>/<filler>)")
    candidates = [tok for _, _, tok, _ in rows if tok not in SPECIAL_TOKENS]
    need = num_keywords - 2
    selected = candidates[:need]
    if len(selected) < need:
        raise SystemExit(
            f"not enough tokens: need {need}, got {len(selected)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reduced dict by cum_freq_pct or num_keywords.")
    parser.add_argument(
        "--sorted_file",
        default="examples/hi_xiaowen/s0/dict/model_vocab_freq_asr_sorted.txt",
        help="Path to model_vocab_freq_asr_sorted.txt",
    )
    parser.add_argument(
        "--cum_pct",
        type=float,
        default=80.0,
        help="Cumulative frequency percent threshold (if num_keywords unset)",
    )
    parser.add_argument(
        "--include_cross",
        action="store_true",
        default=True,
        help="Include first row that crosses threshold",
    )
    parser.add_argument(
        "--num_keywords",
        type=int,
        default=None,
        help="Target vocab size (includes <blk>/<filler>)",
    )
    parser.add_argument(
        "--output_dir",
        default="examples/hi_xiaowen/s0/dict_top440",
        help="Output dict directory",
    )
    args = parser.parse_args()

    sorted_path = args.sorted_file
    if not os.path.isabs(sorted_path):
        sorted_path = os.path.abspath(sorted_path)
    if not os.path.exists(sorted_path):
        raise SystemExit(f"sorted file not found: {sorted_path}")

    rows = parse_sorted(sorted_path)
    if not rows:
        raise SystemExit("sorted file empty or invalid")

    if args.num_keywords is not None:
        selected = select_by_num_keywords(rows, args.num_keywords)
    else:
        selected = select_by_cum_pct(rows, args.cum_pct, args.include_cross)

    # Force include special tokens
    for s in ("<blk>", "<filler>"):
        if s not in selected:
            selected.insert(0, s)

    # Remove duplicates while keeping order
    seen = set()
    filtered: List[str] = []
    for tok in selected:
        if tok in seen:
            continue
        seen.add(tok)
        filtered.append(tok)
    selected = filtered

    os.makedirs(args.output_dir, exist_ok=True)
    dict_path = os.path.join(args.output_dir, "dict.txt")
    words_path = os.path.join(args.output_dir, "words.txt")

    # Write dict with reindexed ids
    with open(dict_path, "w", encoding="utf8") as f:
        f.write("sil 0\n")
        f.write("<eps> -1\n")
        f.write("<blk> 0\n")
        f.write("<filler> 1\n")
        next_id = 2
        for tok in selected:
            if tok in SPECIAL_TOKENS:
                continue
            f.write(f"{tok} {next_id}\n")
            next_id += 1

    # Write words.txt
    with open(words_path, "w", encoding="utf8") as f:
        f.write("<SILENCE>\n")
        f.write("<EPS>\n")
        f.write("<BLK>\n")

    vocab_size = next_id  # ids 0..next_id-1
    print("Done.")
    print(f"- dict:  {dict_path}")
    print(f"- words: {words_path}")
    print(f"- vocab_size: {vocab_size}")


if __name__ == "__main__":
    main()
