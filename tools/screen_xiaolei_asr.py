#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

TARGETS = {
    '小 雷 小 雷': '小雷小雷',
    '小 雷 快 拍': '小雷快拍',
}
TARGET_COMPACT = set(TARGETS.values())
PARTIAL_HINTS = ('小雷', '快拍')


def parse_args():
    parser = argparse.ArgumentParser(description='ASR-screen xiaolei data.list and split into clean/ambiguous/reject buckets.')
    parser.add_argument('--input-list', required=True, help='input data.list/jsonl')
    parser.add_argument('--output-dir', required=True, help='output directory for ASR results and split manifests')
    parser.add_argument('--model', default='paraformer-zh')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--hotword', default='小雷小雷 小雷快拍')
    parser.add_argument('--batch-size-s', type=int, default=60)
    parser.add_argument('--limit', type=int, default=0, help='0 means all')
    parser.add_argument('--sentence-timestamp', action='store_true', help='request sentence_info when supported')
    parser.add_argument('--disable-update', action='store_true', default=True)
    parser.add_argument('--progress-every', type=int, default=100)
    return parser.parse_args()


def compact_text(text):
    return ''.join(str(text).split())


def has_partial_hint(text_compact):
    return any(hint in text_compact for hint in PARTIAL_HINTS)


def classify(expected_text, asr_text):
    expected_compact = compact_text(expected_text)
    asr_compact = compact_text(asr_text)
    is_pos = expected_text in TARGETS

    if is_pos:
        if asr_compact == expected_compact:
            return 'clean_exact'
        if expected_compact and expected_compact in asr_compact:
            return 'ambiguous_extra_context'
        if asr_compact in TARGET_COMPACT and asr_compact != expected_compact:
            return 'ambiguous_other_target'
        if any(target in asr_compact for target in TARGET_COMPACT):
            return 'ambiguous_target_overlap'
        if has_partial_hint(asr_compact):
            return 'ambiguous_partial_keyword'
        return 'reject_positive_miss'

    if asr_compact in TARGET_COMPACT:
        return 'reject_negative_exact_target'
    if any(target in asr_compact for target in TARGET_COMPACT):
        return 'ambiguous_negative_target_overlap'
    if has_partial_hint(asr_compact):
        return 'ambiguous_negative_partial_keyword'
    return 'clean_negative'


def bucket_from_status(status):
    if status.startswith('clean_'):
        return 'clean'
    if status.startswith('reject_'):
        return 'reject'
    return 'ambiguous'


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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(args.input_list, args.limit)

    from funasr import AutoModel
    model = AutoModel(
        model=args.model,
        device=args.device,
        disable_update=args.disable_update,
    )

    result_path = output_dir / 'asr_results.jsonl'
    bucket_paths = {
        'clean': output_dir / 'clean.data.list',
        'ambiguous': output_dir / 'ambiguous.data.list',
        'reject': output_dir / 'reject.data.list',
    }
    summary = {
        'config': {
            'input_list': args.input_list,
            'output_dir': args.output_dir,
            'model': args.model,
            'device': args.device,
            'hotword': args.hotword,
            'batch_size_s': args.batch_size_s,
            'limit': args.limit,
            'sentence_timestamp': args.sentence_timestamp,
        },
        'counts': {},
        'status_counts': {},
        'expected_label_counts': {},
        'error_count': 0,
    }

    status_counter = Counter()
    label_counter = Counter()
    bucket_counter = Counter()

    result_f = result_path.open('w', encoding='utf-8')
    bucket_files = {name: path.open('w', encoding='utf-8') for name, path in bucket_paths.items()}
    try:
        for idx, item in enumerate(items, 1):
            kwargs = {
                'input': item['wav'],
                'batch_size_s': args.batch_size_s,
                'hotword': args.hotword,
            }
            if args.sentence_timestamp:
                kwargs['sentence_timestamp'] = True

            result0 = {}
            asr_text = ''
            asr_error = None
            try:
                result = model.generate(**kwargs)
                result0 = result[0] if isinstance(result, list) and result else {}
                asr_text = result0.get('text', '')
                status = classify(item.get('txt', ''), asr_text)
            except Exception as exc:
                asr_error = f'{type(exc).__name__}: {exc}'
                status = 'reject_load_error'
                summary['error_count'] += 1

            bucket = bucket_from_status(status)
            enriched = {
                **item,
                'asr_text': asr_text,
                'asr_text_compact': compact_text(asr_text),
                'expected_text_compact': compact_text(item.get('txt', '')),
                'screen_status': status,
                'screen_bucket': bucket,
                'timestamp': result0.get('timestamp', []),
            }
            if asr_error is not None:
                enriched['asr_error'] = asr_error
            if 'sentence_info' in result0:
                enriched['sentence_info'] = result0['sentence_info']
            result_f.write(json.dumps(enriched, ensure_ascii=False) + '\n')
            bucket_files[bucket].write(json.dumps(item, ensure_ascii=False) + '\n')
            status_counter[status] += 1
            label_counter[item.get('txt', '')] += 1
            bucket_counter[bucket] += 1
            if idx % args.progress_every == 0:
                print(f'processed={idx} clean={bucket_counter["clean"]} ambiguous={bucket_counter["ambiguous"]} reject={bucket_counter["reject"]} errors={summary["error_count"]}')
    finally:
        result_f.close()
        for f in bucket_files.values():
            f.close()

    summary['counts'] = dict(sorted(bucket_counter.items()))
    summary['status_counts'] = dict(sorted(status_counter.items()))
    summary['expected_label_counts'] = dict(sorted(label_counter.items()))
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
