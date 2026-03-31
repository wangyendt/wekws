#!/usr/bin/env python3
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

POS_TEXT_MAP = {
    '<XIAOLEI_XIAOLEI>': '小 雷 小 雷',
    '<XIAOLEI_KUAIPAI>': '小 雷 快 拍',
}
NEG_TEXT_MAP = {
    '<FILLER>': '<filler>',
    '<SILENCE>': '<SILENCE>',
    '<RAYNEO>': '<filler>',
}
TARGET_TEXTS = set(POS_TEXT_MAP.values())
TARGET_JOINED = {text.replace(' ', '') for text in TARGET_TEXTS}
SPLITS = ('train', 'dev', 'test')


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare xiaolei CTC dataset from jsonl manifests.')
    parser.add_argument('--xlxl-list', required=True, help='jsonl list for <XIAOLEI_XIAOLEI>')
    parser.add_argument('--kuaipai-list', required=True, help='jsonl list for <XIAOLEI_KUAIPAI>')
    parser.add_argument('--filler-list', required=True, help='jsonl list for <FILLER> negatives')
    parser.add_argument('--extra-neg-list', action='append', default=[], help='additional manifests to treat as negative pool')
    parser.add_argument('--output-dir', required=True, help='output dataset directory')
    parser.add_argument('--dev-ratio', type=float, default=0.05)
    parser.add_argument('--test-ratio', type=float, default=0.05)
    parser.add_argument('--neg-pos-ratio', type=float, default=3.0, help='target negative/positive ratio per split')
    parser.add_argument('--max-train-negatives', type=int, default=0, help='0 means no explicit cap')
    parser.add_argument('--max-dev-negatives', type=int, default=0, help='0 means no explicit cap')
    parser.add_argument('--max-test-negatives', type=int, default=0, help='0 means no explicit cap')
    parser.add_argument('--min-duration', type=float, default=0.3)
    parser.add_argument('--max-duration', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--keep-source-manifest', action='store_true', help='write manifest.jsonl with source metadata')
    return parser.parse_args()


def normalize_text(text):
    if text is None:
        return ''
    text = str(text).strip()
    if not text:
        return ''
    if text in POS_TEXT_MAP:
        return POS_TEXT_MAP[text]
    if text in NEG_TEXT_MAP:
        return NEG_TEXT_MAP[text]
    return ' '.join(text.split())


def contains_target(text):
    compact = ''.join(str(text).split())
    return any(keyword in compact for keyword in TARGET_JOINED)


def infer_group(source, key, wav):
    if source == 'xlxl':
        parts = key.split('--')
        if len(parts) >= 2:
            return '--'.join(parts[:2])
    if source == 'kuaipai':
        parts = key.split('_')
        if len(parts) >= 7:
            return '_'.join(parts[:7])
    stem = Path(wav).stem
    if '--' in stem:
        parts = stem.split('--')
        if len(parts) >= 2:
            return '--'.join(parts[:2])
    return key


def dedupe_key(key, seen, source):
    if key not in seen:
        seen.add(key)
        return key
    base = f'{source}__{key}'
    candidate = base
    index = 2
    while candidate in seen:
        candidate = f'{base}__{index}'
        index += 1
    seen.add(candidate)
    return candidate


def load_manifest(path, source, role, seen_keys, min_duration, max_duration, dropped):
    items = []
    path = Path(path)
    with path.open('r', encoding='utf-8') as fin:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                dropped[f'{source}:json_error'] += 1
                continue
            key = str(obj.get('key', '')).strip()
            wav = str(obj.get('wav', '')).strip()
            text = normalize_text(obj.get('txt', ''))
            try:
                duration = float(obj.get('duration', 0.0))
            except (TypeError, ValueError):
                dropped[f'{source}:bad_duration'] += 1
                continue
            if not key or not wav:
                dropped[f'{source}:missing_key_or_wav'] += 1
                continue
            if duration < min_duration:
                dropped[f'{source}:too_short'] += 1
                continue
            if max_duration > 0 and duration > max_duration:
                dropped[f'{source}:too_long'] += 1
                continue
            if not Path(wav).is_file():
                dropped[f'{source}:missing_wav'] += 1
                continue
            if role == 'pos' and text not in TARGET_TEXTS:
                dropped[f'{source}:unexpected_pos_text'] += 1
                continue
            if role == 'neg' and contains_target(text):
                dropped[f'{source}:target_in_negative'] += 1
                continue
            final_key = dedupe_key(key, seen_keys, source)
            items.append({
                'key': final_key,
                'txt': text,
                'duration': round(duration, 6),
                'wav': wav,
                'source': source,
                'role': role,
                'group': infer_group(source, key, wav) if role == 'pos' else final_key,
                'original_key': key,
                'source_path': str(path),
            })
    return items


def split_grouped(items, dev_ratio, test_ratio, seed):
    groups = defaultdict(list)
    for item in items:
        groups[item['group']].append(item)
    group_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    total_groups = len(group_ids)
    test_groups = int(round(total_groups * test_ratio))
    dev_groups = int(round(total_groups * dev_ratio))
    if total_groups >= 3 and test_ratio > 0 and test_groups == 0:
        test_groups = 1
    if total_groups >= 3 and dev_ratio > 0 and dev_groups == 0:
        dev_groups = 1
    if dev_groups + test_groups >= total_groups:
        overflow = dev_groups + test_groups - total_groups + 1
        while overflow > 0 and (dev_groups > 0 or test_groups > 0):
            if dev_groups >= test_groups and dev_groups > 0:
                dev_groups -= 1
            elif test_groups > 0:
                test_groups -= 1
            overflow -= 1
    test_ids = set(group_ids[:test_groups])
    dev_ids = set(group_ids[test_groups:test_groups + dev_groups])
    split_items = {split: [] for split in SPLITS}
    for gid in group_ids:
        if gid in test_ids:
            split = 'test'
        elif gid in dev_ids:
            split = 'dev'
        else:
            split = 'train'
        split_items[split].extend(groups[gid])
    return split_items


def sample_negatives(items, split_pos, neg_pos_ratio, caps, seed):
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    split_items = {split: [] for split in SPLITS}
    start = 0
    for idx, split in enumerate(SPLITS):
        pos_count = len(split_pos[split])
        target = int(round(pos_count * neg_pos_ratio))
        if caps.get(split, 0) > 0:
            target = min(target, caps[split])
        remaining_splits = len(SPLITS) - idx
        remaining_items = len(shuffled) - start
        if remaining_splits > 1:
            target = min(target, remaining_items)
        else:
            target = min(target, remaining_items)
        split_items[split] = shuffled[start:start + target]
        start += target
    return split_items, len(shuffled) - start


def write_split(split_dir, items, keep_source_manifest):
    split_dir.mkdir(parents=True, exist_ok=True)
    wav_scp = split_dir / 'wav.scp'
    text_file = split_dir / 'text'
    dur_file = split_dir / 'wav.dur'
    data_list = split_dir / 'data.list'
    manifest_file = split_dir / 'manifest.jsonl'

    with wav_scp.open('w', encoding='utf-8') as wav_out, \
            text_file.open('w', encoding='utf-8') as text_out, \
            dur_file.open('w', encoding='utf-8') as dur_out, \
            data_list.open('w', encoding='utf-8') as list_out:
        for item in items:
            wav_out.write(f"{item['key']} {item['wav']}\n")
            text_out.write(f"{item['key']} {item['txt']}\n")
            dur_out.write(f"{item['key']} {item['duration']}\n")
            list_out.write(json.dumps({
                'key': item['key'],
                'txt': item['txt'],
                'duration': item['duration'],
                'wav': item['wav'],
            }, ensure_ascii=False) + '\n')

    if keep_source_manifest:
        with manifest_file.open('w', encoding='utf-8') as fout:
            for item in items:
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')


def summarize_split(items):
    label_counter = Counter(item['txt'] for item in items)
    source_counter = Counter(item['source'] for item in items)
    duration_hours = round(sum(item['duration'] for item in items) / 3600.0, 4)
    return {
        'count': len(items),
        'duration_hours': duration_hours,
        'labels': dict(sorted(label_counter.items())),
        'sources': dict(sorted(source_counter.items())),
    }


def main():
    args = parse_args()
    if args.dev_ratio < 0 or args.test_ratio < 0 or args.dev_ratio + args.test_ratio >= 1:
        raise ValueError('dev_ratio/test_ratio must be >=0 and sum to < 1')
    if args.neg_pos_ratio <= 0:
        raise ValueError('neg_pos_ratio must be > 0')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dropped = Counter()
    seen_keys = set()

    xlxl_items = load_manifest(args.xlxl_list, 'xlxl', 'pos', seen_keys,
                               args.min_duration, args.max_duration, dropped)
    kuaipai_items = load_manifest(args.kuaipai_list, 'kuaipai', 'pos', seen_keys,
                                  args.min_duration, args.max_duration, dropped)
    filler_items = load_manifest(args.filler_list, 'filler_0327', 'neg', seen_keys,
                                 args.min_duration, args.max_duration, dropped)

    extra_neg_items = []
    for index, path in enumerate(args.extra_neg_list, 1):
        source = f'extra_neg_{index}'
        extra_neg_items.extend(load_manifest(path, source, 'neg', seen_keys,
                                             args.min_duration, args.max_duration, dropped))

    pos_split = {split: [] for split in SPLITS}
    for idx, source_items in enumerate((xlxl_items, kuaipai_items)):
        split_items = split_grouped(source_items, args.dev_ratio, args.test_ratio,
                                    args.seed + idx)
        for split in SPLITS:
            pos_split[split].extend(split_items[split])

    neg_pool = filler_items + extra_neg_items
    neg_caps = {
        'train': args.max_train_negatives,
        'dev': args.max_dev_negatives,
        'test': args.max_test_negatives,
    }
    neg_split, unused_negatives = sample_negatives(neg_pool, pos_split,
                                                   args.neg_pos_ratio,
                                                   neg_caps, args.seed + 100)

    summary = {
        'config': {
            'xlxl_list': args.xlxl_list,
            'kuaipai_list': args.kuaipai_list,
            'filler_list': args.filler_list,
            'extra_neg_list': args.extra_neg_list,
            'output_dir': args.output_dir,
            'dev_ratio': args.dev_ratio,
            'test_ratio': args.test_ratio,
            'neg_pos_ratio': args.neg_pos_ratio,
            'max_train_negatives': args.max_train_negatives,
            'max_dev_negatives': args.max_dev_negatives,
            'max_test_negatives': args.max_test_negatives,
            'min_duration': args.min_duration,
            'max_duration': args.max_duration,
            'seed': args.seed,
        },
        'input_counts': {
            'xlxl_items': len(xlxl_items),
            'kuaipai_items': len(kuaipai_items),
            'filler_items': len(filler_items),
            'extra_neg_items': len(extra_neg_items),
            'neg_pool_total': len(neg_pool),
            'unused_negatives': unused_negatives,
            'dropped': dict(sorted(dropped.items())),
        },
        'splits': {},
    }

    for split in SPLITS:
        items = pos_split[split] + neg_split[split]
        write_split(output_dir / split, items, args.keep_source_manifest)
        summary['splits'][split] = {
            'positives': summarize_split(pos_split[split]),
            'negatives': summarize_split(neg_split[split]),
            'merged': summarize_split(items),
        }

    with (output_dir / 'summary.json').open('w', encoding='utf-8') as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
