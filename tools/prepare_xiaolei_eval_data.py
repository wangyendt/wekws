#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

LABEL_MAP = {
    '<XIAOLEI_XIAOLEI>': '小 雷 小 雷',
    '<XIAOLEI_KUAIPAI>': '小 雷 快 拍',
    '<FILLER>': '<FILLER>',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Convert external xiaolei eval list into standard data.list layout.')
    parser.add_argument('--input-list', required=True, help='jsonl list with key/wav/txt/duration')
    parser.add_argument('--output-dir', required=True, help='output dataset root')
    parser.add_argument('--split', default='test', help='split name under output dir')
    parser.add_argument('--max-duration', type=float, default=0.0, help='skip utterances whose duration is greater than this value, 0 means keep all')
    parser.add_argument('--keep-source-manifest', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    input_list = Path(args.input_list)
    if not input_list.is_file():
        raise FileNotFoundError(f'missing input list: {input_list}')

    split_dir = Path(args.output_dir) / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    wav_scp = split_dir / 'wav.scp'
    text_file = split_dir / 'text'
    dur_file = split_dir / 'wav.dur'
    data_list = split_dir / 'data.list'
    manifest_file = split_dir / 'manifest.jsonl'
    summary_file = Path(args.output_dir) / 'summary.json'

    count = 0
    duration_sum = 0.0
    label_counter = Counter()
    dropped_too_long = 0

    with input_list.open('r', encoding='utf-8') as fin,             wav_scp.open('w', encoding='utf-8') as wav_out,             text_file.open('w', encoding='utf-8') as text_out,             dur_file.open('w', encoding='utf-8') as dur_out,             data_list.open('w', encoding='utf-8') as list_out:
        manifest_out = manifest_file.open('w', encoding='utf-8') if args.keep_source_manifest else None
        try:
            for raw in fin:
                if not raw.strip():
                    continue
                obj = json.loads(raw)
                key = str(obj['key'])
                wav = str(obj['wav'])
                txt = LABEL_MAP.get(str(obj.get('txt', '')).strip(), str(obj.get('txt', '')).strip())
                duration = float(obj.get('duration', 0.0))
                if args.max_duration > 0 and duration > args.max_duration:
                    dropped_too_long += 1
                    continue
                item = {
                    'key': key,
                    'wav': wav,
                    'txt': txt,
                    'duration': duration,
                }
                wav_out.write(f"{key} {wav}\n")
                text_out.write(f"{key} {txt}\n")
                dur_out.write(f"{key} {duration}\n")
                list_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                if manifest_out is not None:
                    manifest_out.write(json.dumps(obj, ensure_ascii=False) + '\n')
                count += 1
                duration_sum += duration
                label_counter[txt] += 1
        finally:
            if manifest_out is not None:
                manifest_out.close()

    summary = {
        'input_list': str(input_list),
        'output_dir': str(Path(args.output_dir)),
        'split': args.split,
        'max_duration': args.max_duration,
        'count': count,
        'dropped_too_long': dropped_too_long,
        'duration_hours': round(duration_sum / 3600.0, 4),
        'labels': dict(sorted(label_counter.items())),
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
