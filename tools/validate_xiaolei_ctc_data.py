#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

TARGET_TEXTS = {'小 雷 小 雷', '小 雷 快 拍'}
SPLITS = ('train', 'dev', 'test')


def parse_args():
    parser = argparse.ArgumentParser(description='Validate prepared xiaolei CTC dataset.')
    parser.add_argument('--data-dir', required=True, help='dataset root created by prepare_xiaolei_ctc_data.py')
    return parser.parse_args()


def read_nonempty_lines(path):
    with path.open('r', encoding='utf-8') as fin:
        return [line.rstrip('\n') for line in fin if line.strip()]


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    errors = []
    all_keys = {}
    summary = {'splits': {}}

    for split in SPLITS:
        split_dir = data_dir / split
        required = {
            'wav.scp': split_dir / 'wav.scp',
            'text': split_dir / 'text',
            'wav.dur': split_dir / 'wav.dur',
            'data.list': split_dir / 'data.list',
        }
        for name, path in required.items():
            if not path.is_file():
                errors.append(f'{split}: missing {name} at {path}')
        if errors and any(msg.startswith(f'{split}: missing') for msg in errors):
            continue

        wav_lines = read_nonempty_lines(required['wav.scp'])
        text_lines = read_nonempty_lines(required['text'])
        dur_lines = read_nonempty_lines(required['wav.dur'])
        list_lines = read_nonempty_lines(required['data.list'])
        counts = {len(wav_lines), len(text_lines), len(dur_lines), len(list_lines)}
        if len(counts) != 1:
            errors.append(f'{split}: file line counts do not match: wav={len(wav_lines)}, text={len(text_lines)}, dur={len(dur_lines)}, list={len(list_lines)}')
            continue

        label_counter = Counter()
        duration_hours = 0.0
        pos_count = 0
        neg_count = 0
        split_keys = set()

        for line in list_lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f'{split}: invalid json line in data.list')
                continue
            for field in ('key', 'txt', 'duration', 'wav'):
                if field not in obj:
                    errors.append(f'{split}: missing field {field} in data.list entry')
                    continue
            key = obj['key']
            txt = obj['txt']
            wav = Path(obj['wav'])
            try:
                duration = float(obj['duration'])
            except (TypeError, ValueError):
                errors.append(f'{split}: bad duration for {key}')
                continue
            if key in split_keys:
                errors.append(f'{split}: duplicate key inside split: {key}')
            split_keys.add(key)
            if key in all_keys:
                errors.append(f'{split}: key duplicated across splits: {key} (already in {all_keys[key]})')
            else:
                all_keys[key] = split
            if not wav.is_file():
                errors.append(f'{split}: missing wav for {key}: {wav}')
            if duration <= 0:
                errors.append(f'{split}: non-positive duration for {key}: {duration}')
            label_counter[txt] += 1
            duration_hours += duration / 3600.0
            if txt in TARGET_TEXTS:
                pos_count += 1
            else:
                neg_count += 1
                compact = ''.join(str(txt).split())
                if '小雷小雷' in compact or '小雷快拍' in compact:
                    errors.append(f'{split}: negative sample still contains target keyword text: {key} -> {txt}')

        summary['splits'][split] = {
            'count': len(list_lines),
            'duration_hours': round(duration_hours, 4),
            'positives': pos_count,
            'negatives': neg_count,
            'labels': dict(sorted(label_counter.items())),
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        print('\nVALIDATION_ERRORS:')
        for err in errors:
            print(err)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
