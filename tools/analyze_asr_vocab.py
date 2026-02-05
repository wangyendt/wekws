#!/usr/bin/env python3
"""
Analyze ASR transcription token frequency and model vocab coverage.

Outputs:
1) dict_token_freq_asr.txt     - frequency of each dict token in ASR text
2) model_vocab_2599.txt        - token list for ids [0..output_dim-1]
3) model_vocab_freq_asr.txt    - frequency of model vocab tokens in ASR text

Default paths target examples/hi_xiaowen/s0.
"""

import argparse
import glob
import json
import os
from collections import Counter
from typing import Dict, List, Tuple


def load_id2token(path: str) -> Dict[int, str]:
    id2tok: Dict[int, str] = {}
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            tok = " ".join(parts[:-1])
            try:
                idx = int(parts[-1])
            except ValueError:
                continue
            id2tok[idx] = tok
    return id2tok


def iter_data_list_tokens(paths: List[str]) -> Counter:
    counter = Counter()
    for path in paths:
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                txt = obj.get("txt", "")
                if not txt:
                    continue
                # data.list uses space-separated tokens
                tokens = [t for t in txt.split(" ") if t]
                counter.update(tokens)
    return counter


def write_token_freq(
    out_path: str,
    tokens: List[str],
    freq: Counter,
    total_tokens: int,
    include_id: bool = False,
    id_map: Dict[str, int] = None,
) -> None:
    with open(out_path, "w", encoding="utf8") as f:
        if include_id:
            f.write("id\ttoken\tfreq\tfreq_pct\n")
        else:
            f.write("token\tfreq\tfreq_pct\n")
        for tok in tokens:
            count = freq.get(tok, 0)
            pct = (count / total_tokens * 100.0) if total_tokens > 0 else 0.0
            if include_id:
                tok_id = id_map.get(tok, -1) if id_map else -1
                f.write(f"{tok_id}\t{tok}\t{count}\t{pct:.6f}\n")
            else:
                f.write(f"{tok}\t{count}\t{pct:.6f}\n")


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    repo_root = _repo_root()
    alt = os.path.join(repo_root, path)
    if os.path.exists(alt):
        return alt
    return path


def _expand_data_list_patterns(pattern: str) -> List[str]:
    """Expand brace pattern or directory into data.list files."""
    pattern = pattern.strip()
    if not pattern:
        return []

    # If it's a directory, collect all data.list under it
    if os.path.isdir(pattern):
        paths: List[str] = []
        for root, _, files in os.walk(pattern):
            for name in files:
                if name == "data.list":
                    paths.append(os.path.join(root, name))
        return paths

    # Support simple brace expansion like {train,dev,test}
    if "{" in pattern and "}" in pattern:
        prefix, rest = pattern.split("{", 1)
        mid, suffix = rest.split("}", 1)
        parts = [p.strip() for p in mid.split(",") if p.strip()]
        expanded: List[str] = []
        for p in parts:
            expanded.append(f"{prefix}{p}{suffix}")
        paths: List[str] = []
        for p in expanded:
            paths.extend(glob.glob(p, recursive=True))
        return paths

    return glob.glob(pattern, recursive=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze ASR token frequency and model vocab coverage."
    )
    parser.add_argument(
        "--dict",
        default="dict/dict.txt",
        help="Path to dict.txt (default relative to examples/hi_xiaowen/s0)",
    )
    parser.add_argument(
        "--data_lists",
        default="data",
        help="Glob or directory for data.list files (default relative to examples/hi_xiaowen/s0)",
    )
    parser.add_argument(
        "--output_dir",
        default="dict",
        help="Output directory (default relative to examples/hi_xiaowen/s0)",
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=None,
        help="Model output dimension (vocab size). If not set, try --config.",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional model config.yaml to infer output_dim",
    )
    parser.add_argument(
        "--vocab_file",
        default="",
        help="Optional vocab file (token id mapping) to build model vocab list",
    )
    args = parser.parse_args()

    dict_path = _resolve_path(args.dict)
    output_dir = _resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Expand data list globs with braces
    data_lists_input = _resolve_path(args.data_lists)
    data_list_paths = _expand_data_list_patterns(data_lists_input)
    if not data_list_paths:
        # If nothing found, try relative to repo root for convenience
        alt = _resolve_path(os.path.join("examples/hi_xiaowen/s0", args.data_lists))
        if alt != data_lists_input:
            data_list_paths = _expand_data_list_patterns(alt)
    if not data_list_paths:
        raise SystemExit(f"No data.list files found: {args.data_lists}")

    dict_id2tok = load_id2token(dict_path)
    if not dict_id2tok:
        raise SystemExit(f"Empty or invalid dict: {dict_path}")

    # Collect ASR token frequency
    asr_freq = iter_data_list_tokens(data_list_paths)
    total_asr_tokens = sum(asr_freq.values())

    # 1) Frequency for each dict token
    dict_tokens = [tok for _, tok in sorted(dict_id2tok.items(), key=lambda x: x[0])]
    dict_freq_path = os.path.join(output_dir, "dict_token_freq_asr.txt")
    write_token_freq(dict_freq_path, dict_tokens, asr_freq, total_asr_tokens)

    output_dim = args.output_dim
    if output_dim is None and args.config:
        try:
            import yaml  # lazy import
            with open(args.config, "r", encoding="utf8") as f:
                cfg = yaml.safe_load(f)
            output_dim = int(cfg["model"]["output_dim"])
        except Exception:
            output_dim = None
    if output_dim is None:
        output_dim = 2599

    # 2) Model vocab list for ids <= output_dim
    # Choose vocab source for model mapping
    model_id2tok = dict_id2tok
    if args.vocab_file:
        vocab_path = _resolve_path(args.vocab_file)
        model_id2tok = load_id2token(vocab_path)
        if not model_id2tok:
            raise SystemExit(f"Empty or invalid vocab file: {vocab_path}")

    model_vocab_path = os.path.join(output_dir, f"model_vocab_{output_dim}.txt")
    model_tokens: List[str] = []
    with open(model_vocab_path, "w", encoding="utf8") as f:
        f.write("id\ttoken\n")
        valid_ids = [i for i in sorted(model_id2tok.keys()) if i <= output_dim]
        for idx in valid_ids:
            tok = model_id2tok.get(idx, f"<unk:{idx}>")
            model_tokens.append(tok)
            f.write(f"{idx}\t{tok}\n")

    # 3) Frequency for model vocab tokens in ASR
    model_freq_path = os.path.join(output_dir, "model_vocab_freq_asr.txt")
    write_token_freq(
        model_freq_path,
        model_tokens,
        asr_freq,
        total_asr_tokens,
        include_id=True,
        id_map=model_id2tok,
    )

    # 4) Model vocab sorted by frequency with cumulative percentage
    sorted_path = os.path.join(output_dir, "model_vocab_freq_asr_sorted.txt")
    pairs: List[Tuple[int, str, int]] = []
    for idx in valid_ids:
        tok = model_id2tok.get(idx, f"<unk:{idx}>")
        pairs.append((idx, tok, asr_freq.get(tok, 0)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    cum = 0
    with open(sorted_path, "w", encoding="utf8") as f:
        f.write("rank\tid\ttoken\tfreq\tfreq_pct\tcum_freq_pct\n")
        for rank, (idx, tok, cnt) in enumerate(pairs, start=1):
            cum += cnt
            pct = (cnt / total_asr_tokens * 100.0) if total_asr_tokens > 0 else 0.0
            cum_pct = (cum / total_asr_tokens * 100.0) if total_asr_tokens > 0 else 0.0
            f.write(f"{rank}\t{idx}\t{tok}\t{cnt}\t{pct:.6f}\t{cum_pct:.6f}\n")

    print("Done.")
    print(f"- Dict freq:   {dict_freq_path}")
    print(f"- Model vocab: {model_vocab_path}")
    print(f"- Model freq:  {model_freq_path}")
    print(f"- Sorted freq: {sorted_path}")


if __name__ == "__main__":
    main()
