#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Export xiaolei review package with copied wavs and manifest.')
    parser.add_argument('--data-dir', required=True, help='prepared dataset root, e.g. data_xlxl_0327_ctc_v1')
    parser.add_argument('--splits', default='dev,test', help='comma-separated splits to export')
    parser.add_argument('--output-dir', required=True, help='export directory')
    parser.add_argument('--limit-per-split', type=int, default=0, help='0 means all')
    return parser.parse_args()


def load_items(path, limit=0):
    items = []
    with Path(path).open('r', encoding='utf-8') as fin:
        for line in fin:
            if not line.strip():
                continue
            items.append(json.loads(line))
            if limit > 0 and len(items) >= limit:
                break
    return items


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_root = output_dir / 'audio'
    audio_root.mkdir(parents=True, exist_ok=True)

    manifest_jsonl = output_dir / 'manifest.jsonl'
    manifest_csv = output_dir / 'manifest.csv'

    rows = []
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    for split in splits:
        split_list = data_dir / split / 'data.list'
        items = load_items(split_list, args.limit_per_split)
        split_audio_dir = audio_root / split
        split_audio_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            src = Path(item['wav'])
            dst = split_audio_dir / f"{item['key']}{src.suffix}"
            shutil.copy2(src, dst)
            row = {
                'split': split,
                'key': item['key'],
                'txt': item['txt'],
                'duration': item['duration'],
                'src_wav': str(src),
                'dst_wav': str(dst.relative_to(output_dir)),
            }
            rows.append(row)

    with manifest_jsonl.open('w', encoding='utf-8') as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + '\n')

    with manifest_csv.open('w', encoding='utf-8', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=['split', 'key', 'txt', 'duration', 'src_wav', 'dst_wav'])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'data_dir': str(data_dir),
        'output_dir': str(output_dir),
        'splits': splits,
        'count': len(rows),
    }
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
