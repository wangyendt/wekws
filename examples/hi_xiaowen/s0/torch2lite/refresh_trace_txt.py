#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


DIRECT_FILE_QPARAM_KEYS = {
    'feat_input_quant': ('input0_feat',),
    'normal_feat_quant': ('input0_feat',),
    'debug_feat_quant': ('input0_feat',),
    'cache_input_quant': ('input1_cache',),
    'normal_cache_quant': ('input1_cache',),
    'debug_cache_quant': ('input1_cache',),
    'tflite_cache_raw': ('output1_cache', 'input1_cache'),
    'normal_cache_raw': ('output1_cache', 'input1_cache'),
    'debug_cache_raw': ('output1_cache', 'input1_cache'),
    'tflite_logits_raw': ('output0_logits',),
    'normal_logits_raw': ('output0_logits',),
    'debug_logits_raw': ('output0_logits',),
}


DETAIL_SOURCE_BY_STEM = {
    'feat_input_quant': ('input', 3),
    'normal_feat_quant': ('input', 3),
    'debug_feat_quant': ('input', 3),
    'cache_input_quant': ('input', 4),
    'normal_cache_quant': ('input', 4),
    'debug_cache_quant': ('input', 4),
    'tflite_cache_raw': ('output', 4),
    'normal_cache_raw': ('output', 4),
    'debug_cache_raw': ('output', 4),
    'tflite_logits_raw': ('output', 3),
    'normal_logits_raw': ('output', 3),
    'debug_logits_raw': ('output', 3),
}


def qparams_from_detail_entry(detail: dict | None):
    if not detail:
        return None, None
    qp = detail.get('quantization_parameters', {}) or {}
    scales = qp.get('scales') or []
    zero_points = qp.get('zero_points') or []
    if len(scales) == 1 and len(zero_points) == 1 and float(scales[0]) != 0.0:
        return float(scales[0]), int(zero_points[0])
    quant = detail.get('quantization') or []
    if len(quant) == 2 and float(quant[0]) != 0.0:
        return float(quant[0]), int(quant[1])
    return None, None


def qparams_from_detail_lists(meta: dict | None, stem: str):
    if not meta:
        return None, None
    source = DETAIL_SOURCE_BY_STEM.get(stem)
    if not source:
        return None, None
    io_kind, rank = source
    candidates = []
    for key in (('normal_inputs', 'normal_outputs') if io_kind == 'input' else ('normal_outputs', 'tflite_outputs', 'debug_outputs')):
        pass
    if io_kind == 'input':
        pools = [meta.get('normal_inputs') or [], meta.get('tflite_inputs') or [], meta.get('debug_inputs') or []]
    else:
        pools = [meta.get('normal_outputs') or [], meta.get('tflite_outputs') or [], meta.get('debug_outputs') or []]
    for pool in pools:
        for detail in pool:
            shape = detail.get('shape') or []
            if len(shape) == rank:
                q = qparams_from_detail_entry(detail)
                if q[0] is not None:
                    return q
    return None, None



def array_to_string(arr: np.ndarray) -> str:
    return np.array2string(arr, threshold=arr.size, max_line_width=1_000_000)


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def find_root_metadata(start: Path) -> tuple[Path | None, dict | None]:
    for parent in [start, *start.parents]:
        meta = parent / 'metadata.json'
        if meta.is_file():
            return meta, load_json(meta)
    return None, None


def qparams_from_metadata(meta: dict | None, stem: str):
    if not meta:
        return None, None
    qparams = meta.get('qparams', {})
    for key in DIRECT_FILE_QPARAM_KEYS.get(stem, ()):
        item = qparams.get(key)
        if item and item.get('scale') is not None:
            return item.get('scale'), item.get('zero_point')
    return qparams_from_detail_lists(meta, stem)


def qparams_from_tensor_manifest(npy_path: Path):
    step_dir = npy_path.parent.parent if npy_path.parent.name == 'tensors' else None
    if step_dir is None:
        return None, None
    manifest_path = step_dir / 'tensor_manifest.json'
    if not manifest_path.is_file():
        return None, None
    entries = load_json(manifest_path)
    txt_path = npy_path.with_suffix('.txt').resolve()
    for entry in entries:
        artifact = entry.get('artifact')
        if not artifact:
            continue
        if Path(artifact).resolve() == txt_path:
            qp = entry.get('quantization_parameters', {})
            scales = qp.get('scales') or []
            zero_points = qp.get('zero_points') or []
            if len(scales) == 1 and len(zero_points) == 1:
                return float(scales[0]), int(zero_points[0])
            quant = entry.get('quantization') or []
            if len(quant) == 2 and quant[0] not in (0, 0.0):
                return float(quant[0]), int(quant[1])
    return None, None


def write_array_txt(txt_path: Path, arr: np.ndarray, scale=None, zero_point=None):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open('w', encoding='utf-8') as f:
        f.write(f'shape={list(arr.shape)}\n')
        f.write(f'dtype={arr.dtype}\n')
        if scale is not None:
            f.write(f'scale={scale}\n')
            f.write(f'zero_point={zero_point}\n')
            f.write('raw=')
            f.write(array_to_string(arr))
            f.write('\n')
            dequant = (arr.astype(np.float32) - float(zero_point)) * float(scale)
            f.write('dequant=')
            f.write(array_to_string(dequant))
            f.write('\n')
        else:
            f.write('values=')
            f.write(array_to_string(arr))
            f.write('\n')


def refresh_trace_dir(root: Path):
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    meta_path, meta = find_root_metadata(root)
    refreshed = 0
    for npy_path in root.rglob('*.npy'):
        arr = np.load(npy_path, allow_pickle=False)
        stem = npy_path.stem
        scale, zero_point = qparams_from_metadata(meta, stem)
        if scale is None:
            scale, zero_point = qparams_from_tensor_manifest(npy_path)
        write_array_txt(npy_path.with_suffix('.txt'), arr, scale=scale, zero_point=zero_point)
        refreshed += 1
    for p in list(root.rglob('*.json')) + list(root.rglob('*.jsonl')):
        p.with_suffix('.txt').write_text(p.read_text(encoding='utf-8'), encoding='utf-8')
    return {
        'root': str(root),
        'metadata': str(meta_path) if meta_path else None,
        'refreshed_array_txt': refreshed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Rewrite trace txt files into readable format using trace metadata/tensor manifests.')
    parser.add_argument('roots', nargs='+', help='trace directories to refresh')
    return parser.parse_args()


def main():
    args = parse_args()
    for root in args.roots:
        print(json.dumps(refresh_trace_dir(Path(root)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
