#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

SPECIAL_TOKENS = {'sil', '<eps>', '<blk>', '<filler>'}
SPECIAL_TEXT_MAP = {'<FILLER>': '<filler>', '<filler>': '<filler>'}


def parse_args():
    parser = argparse.ArgumentParser(description='Build sorted vocab file from data.list and existing dict.')
    parser.add_argument('--dict', required=True, help='base dict.txt path')
    parser.add_argument('--output', required=True, help='output sorted vocab file')
    parser.add_argument('--data-dir', default='', help='dataset root containing train/dev/test/data.list')
    parser.add_argument('--splits', default='train,dev,test', help='comma-separated splits under --data-dir')
    parser.add_argument('--input-list', action='append', default=[], help='explicit data.list/jsonl paths, can be repeated')
    return parser.parse_args()


def load_dict(dict_path: Path):
    tok2id = {}
    with dict_path.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            tok = parts[0]
            try:
                idx = int(parts[-1])
            except ValueError:
                continue
            if idx >= 0:
                tok2id[tok] = idx
    return tok2id


def resolve_inputs(args):
    inputs = [Path(p) for p in args.input_list]
    if args.data_dir:
        root = Path(args.data_dir)
        for split in [x.strip() for x in args.splits.split(',') if x.strip()]:
            inputs.append(root / split / 'data.list')
    uniq = []
    seen = set()
    for p in inputs:
        rp = str(p)
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    if not uniq:
        raise SystemExit('no input data.list specified')
    for p in uniq:
        if not p.is_file():
            raise SystemExit(f'missing input list: {p}')
    return uniq


def text_to_tokens(text: str):
    text = str(text).strip()
    if not text:
        return []
    if text in SPECIAL_TEXT_MAP:
        return [SPECIAL_TEXT_MAP[text]]
    if ' ' in text:
        return [tok for tok in text.split() if tok]
    return [text]


def main():
    args = parse_args()
    dict_path = Path(args.dict)
    if not dict_path.is_file():
        raise SystemExit(f'missing dict: {dict_path}')
    tok2id = load_dict(dict_path)
    input_lists = resolve_inputs(args)

    counts = Counter()
    unknown = Counter()
    for input_path in input_lists:
        with input_path.open('r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                for tok in text_to_tokens(obj.get('txt', '')):
                    if tok in tok2id:
                        counts[tok] += 1
                    else:
                        unknown[tok] += 1

    rows = []
    total = 0
    for tok, idx in tok2id.items():
        if tok in SPECIAL_TOKENS:
            continue
        freq = counts[tok]
        rows.append((idx, tok, freq))
        total += freq
    if total <= 0:
        raise SystemExit('token frequency total is zero')

    rows.sort(key=lambda x: (-x[2], x[0]))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cum = 0
    with out_path.open('w', encoding='utf-8') as f:
        f.write('rank\tid\ttoken\tfreq\tfreq_pct\tcum_freq_pct\n')
        for rank, (idx, tok, freq) in enumerate(rows, 1):
            cum += freq
            freq_pct = freq * 100.0 / total
            cum_pct = cum * 100.0 / total
            f.write(f'{rank}\t{idx}\t{tok}\t{freq}\t{freq_pct:.6f}\t{cum_pct:.6f}\n')

    print(out_path)
    print('inputs', [str(p) for p in input_lists])
    print('total_token_count', total)
    print('top20', rows[:20])
    if unknown:
        print('unknown_tokens', unknown.most_common(20))


if __name__ == '__main__':
    main()
