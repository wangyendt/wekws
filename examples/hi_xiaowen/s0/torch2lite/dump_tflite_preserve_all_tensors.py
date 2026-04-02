#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
S0_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[4]
if str(S0_DIR) not in sys.path:
    sys.path.insert(0, str(S0_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import infer_wav as iw
from tflite_quant_utils import dequantize_from_detail, get_tensor_quant_params, quantize_to_detail


def sanitize_name(name: str) -> str:
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', name)
    value = value.strip('._')
    return value or 'tensor'


def tensor_stats(array) -> dict:
    arr = np.asarray(array)
    flat = arr.reshape(-1)
    if flat.size == 0:
        return {'shape': list(arr.shape), 'numel': 0, 'min': None, 'max': None, 'mean': None, 'std': None}
    if np.issubdtype(arr.dtype, np.integer):
        work = arr.astype(np.int64)
    else:
        work = arr.astype(np.float32)
    return {
        'shape': list(arr.shape),
        'numel': int(flat.size),
        'min': float(work.min()),
        'max': float(work.max()),
        'mean': float(work.mean()),
        'std': float(work.std()),
    }


def diff_stats(lhs, rhs) -> dict:
    a = np.asarray(lhs, dtype=np.float32)
    b = np.asarray(rhs, dtype=np.float32)
    diff = a - b
    abs_diff = np.abs(diff)
    return {
        'max_abs': float(abs_diff.max()) if abs_diff.size else 0.0,
        'mean_abs': float(abs_diff.mean()) if abs_diff.size else 0.0,
        'rmse': float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0,
    }


def _write_array_txt(txt_path: Path, array) -> None:
    arr = np.asarray(array)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open('w', encoding='utf-8') as f:
        f.write(f"shape={list(arr.shape)}\n")
        f.write(f"dtype={arr.dtype}\n")
        f.write(np.array2string(arr, threshold=arr.size, max_line_width=1_000_000))
        f.write('\n')


def save_array(base_dir: Path, rel_name: str, array) -> str:
    out_path = base_dir / rel_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    np.save(out_path, arr)
    _write_array_txt(out_path.with_suffix('.txt'), arr)
    return str(out_path.with_suffix('.txt'))


def pick_io_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    feat_input = next((item for item in input_details if len(item['shape']) == 3), None)
    cache_input = next((item for item in input_details if len(item['shape']) == 4), None)
    logits_output = next((item for item in output_details if len(item['shape']) == 3), None)
    cache_output = next((item for item in output_details if len(item['shape']) == 4), None)
    if feat_input is None or cache_input is None or logits_output is None or cache_output is None:
        raise RuntimeError('Failed to identify feature/cache inputs or logits/cache outputs from interpreter.')
    return feat_input, cache_input, logits_output, cache_output


def chunk_with_padding(feats: np.ndarray, start: int, chunk_frames: int) -> np.ndarray:
    chunk = feats[:, start:start + chunk_frames, :]
    if chunk.shape[1] == chunk_frames:
        return chunk.astype(np.float32, copy=False)
    padded = np.zeros((1, chunk_frames, feats.shape[2]), dtype=np.float32)
    padded[:, :chunk.shape[1], :] = chunk
    return padded


def detail_to_json(detail: dict) -> dict:
    quant_params = detail.get('quantization_parameters', {})
    return {
        'name': detail.get('name'),
        'index': int(detail.get('index')),
        'shape': np.asarray(detail.get('shape')).astype(int).tolist(),
        'shape_signature': np.asarray(detail.get('shape_signature', detail.get('shape'))).astype(int).tolist(),
        'dtype': str(np.dtype(detail.get('dtype'))),
        'quantization': list(detail.get('quantization', (0.0, 0))),
        'quantization_parameters': {
            'scales': np.asarray(quant_params.get('scales', [])).astype(np.float64).tolist(),
            'zero_points': np.asarray(quant_params.get('zero_points', [])).astype(np.int64).tolist(),
            'quantized_dimension': int(quant_params.get('quantized_dimension', 0)),
        },
    }


def dump_debug_tensors(interpreter, base_dir: Path):
    tensor_entries = []
    for detail in interpreter.get_tensor_details():
        entry = detail_to_json(detail)
        name = sanitize_name(str(entry['name']))
        rel = Path('tensors') / f"{int(entry['index']):03d}_{name}.npy"
        try:
            value = interpreter.get_tensor(detail['index'])
            entry['readable'] = True
            entry['artifact'] = save_array(base_dir, str(rel), value)
            entry['stats'] = tensor_stats(value)
        except Exception as exc:
            entry['readable'] = False
            entry['artifact'] = None
            entry['stats'] = None
            entry['error'] = str(exc)
        tensor_entries.append(entry)
    return tensor_entries


def parse_args():
    parser = argparse.ArgumentParser(description='Dump all readable TFLite tensors with experimental_preserve_all_tensors=True and compare final outputs with normal TFLite.')
    parser.add_argument('--model', required=True, help='tflite model path')
    parser.add_argument('--wav', required=True, help='wav path used to build features')
    parser.add_argument('--config', required=True, help='config yaml used to build features')
    parser.add_argument('--chunk_frames', type=int, default=1, help='streaming chunk size')
    parser.add_argument('--max_steps', type=int, default=0, help='max number of steps to dump; <=0 means all')
    parser.add_argument('--dump_dir', required=True, help='output directory')
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = str(Path(args.model).expanduser().resolve())
    wav_path = iw.to_abs_path(args.wav)
    config_path = Path(args.config).expanduser().resolve()
    dump_dir = Path(args.dump_dir).expanduser().resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)

    configs = iw.load_config(config_path)
    feats = iw.build_input_features(wav_path, configs).cpu().numpy().astype(np.float32)
    save_array(dump_dir, 'full_feats.npy', feats)

    normal = tf.lite.Interpreter(model_path=model_path)
    debug = tf.lite.Interpreter(model_path=model_path, experimental_preserve_all_tensors=True)
    normal.allocate_tensors()
    debug.allocate_tensors()

    n_feat_in, n_cache_in, n_logits_out, n_cache_out = pick_io_details(normal)
    d_feat_in, d_cache_in, d_logits_out, d_cache_out = pick_io_details(debug)

    if tuple(np.asarray(n_feat_in['shape']).tolist()) != tuple(np.asarray(d_feat_in['shape']).tolist()):
        raise RuntimeError('normal/debug feature input shapes differ unexpectedly')

    cache_shape = tuple(int(x) for x in n_cache_in['shape'])
    normal_cache = np.zeros(cache_shape, dtype=np.float32)
    debug_cache = np.zeros(cache_shape, dtype=np.float32)

    chunk_frames = int(args.chunk_frames)
    num_steps_total = int((feats.shape[1] + chunk_frames - 1) // chunk_frames)
    num_steps_dump = num_steps_total if args.max_steps <= 0 else min(num_steps_total, int(args.max_steps))

    metadata = {
        'model': model_path,
        'wav': str(wav_path),
        'config': str(config_path),
        'full_feats_shape': list(feats.shape),
        'num_steps_total': num_steps_total,
        'num_steps_dumped': num_steps_dump,
        'normal_inputs': [detail_to_json(x) for x in normal.get_input_details()],
        'normal_outputs': [detail_to_json(x) for x in normal.get_output_details()],
        'debug_tensor_count': len(debug.get_tensor_details()),
    }

    summaries = []
    for step_idx in range(num_steps_dump):
        start = step_idx * chunk_frames
        end = min(start + chunk_frames, feats.shape[1])
        feat_chunk = chunk_with_padding(feats, start, chunk_frames)

        n_feat_quant = quantize_to_detail(feat_chunk, n_feat_in)
        n_cache_quant = quantize_to_detail(normal_cache, n_cache_in)
        d_feat_quant = quantize_to_detail(feat_chunk, d_feat_in)
        d_cache_quant = quantize_to_detail(debug_cache, d_cache_in)

        normal.set_tensor(n_feat_in['index'], n_feat_quant)
        normal.set_tensor(n_cache_in['index'], n_cache_quant)
        normal.invoke()
        normal_logits_raw = normal.get_tensor(n_logits_out['index'])
        normal_cache_raw = normal.get_tensor(n_cache_out['index'])
        normal_logits = dequantize_from_detail(normal_logits_raw, n_logits_out)
        normal_cache_next = dequantize_from_detail(normal_cache_raw, n_cache_out)

        debug.set_tensor(d_feat_in['index'], d_feat_quant)
        debug.set_tensor(d_cache_in['index'], d_cache_quant)
        debug.invoke()
        debug_logits_raw = debug.get_tensor(d_logits_out['index'])
        debug_cache_raw = debug.get_tensor(d_cache_out['index'])
        debug_logits = dequantize_from_detail(debug_logits_raw, d_logits_out)
        debug_cache_next = dequantize_from_detail(debug_cache_raw, d_cache_out)

        step_dir = dump_dir / 'steps' / f'step_{step_idx:04d}'
        step_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {
            'feat_chunk_fp32': save_array(dump_dir, f'steps/step_{step_idx:04d}/feat_chunk_fp32.npy', feat_chunk),
            'normal_feat_quant': save_array(dump_dir, f'steps/step_{step_idx:04d}/normal_feat_quant.npy', n_feat_quant),
            'debug_feat_quant': save_array(dump_dir, f'steps/step_{step_idx:04d}/debug_feat_quant.npy', d_feat_quant),
            'normal_cache_quant': save_array(dump_dir, f'steps/step_{step_idx:04d}/normal_cache_quant.npy', n_cache_quant),
            'debug_cache_quant': save_array(dump_dir, f'steps/step_{step_idx:04d}/debug_cache_quant.npy', d_cache_quant),
            'normal_logits_raw': save_array(dump_dir, f'steps/step_{step_idx:04d}/normal_logits_raw.npy', normal_logits_raw),
            'normal_logits_dequant': save_array(dump_dir, f'steps/step_{step_idx:04d}/normal_logits_dequant.npy', normal_logits),
            'debug_logits_raw': save_array(dump_dir, f'steps/step_{step_idx:04d}/debug_logits_raw.npy', debug_logits_raw),
            'debug_logits_dequant': save_array(dump_dir, f'steps/step_{step_idx:04d}/debug_logits_dequant.npy', debug_logits),
            'normal_cache_dequant': save_array(dump_dir, f'steps/step_{step_idx:04d}/normal_cache_dequant.npy', normal_cache_next),
            'debug_cache_dequant': save_array(dump_dir, f'steps/step_{step_idx:04d}/debug_cache_dequant.npy', debug_cache_next),
        }
        tensors = dump_debug_tensors(debug, step_dir)
        step_manifest_path = step_dir / 'tensor_manifest.json'
        tensor_manifest_text = json.dumps(tensors, ensure_ascii=False, indent=2)
        step_manifest_path.write_text(tensor_manifest_text, encoding='utf-8')
        step_manifest_path.with_suffix('.txt').write_text(tensor_manifest_text, encoding='utf-8')

        summary = {
            'step': step_idx,
            'frame_range': [int(start), int(end)],
            'valid_frames': int(end - start),
            'artifacts': artifacts,
            'tensor_manifest': str(step_manifest_path),
            'tensor_count': len(tensors),
            'readable_tensor_count': int(sum(1 for x in tensors if x.get('readable'))),
            'logits_debug_vs_normal': diff_stats(debug_logits, normal_logits),
            'cache_debug_vs_normal': diff_stats(debug_cache_next, normal_cache_next),
            'argmax_match': bool(np.array_equal(np.argmax(debug_logits, axis=-1), np.argmax(normal_logits, axis=-1))),
        }
        summaries.append(summary)

        print(
            f"step={step_idx:04d} frames=[{start},{end}) "
            f"tensor_readable={summary['readable_tensor_count']}/{summary['tensor_count']} "
            f"logits_max_abs={summary['logits_debug_vs_normal']['max_abs']:.8f} "
            f"cache_max_abs={summary['cache_debug_vs_normal']['max_abs']:.8f} "
            f"argmax_match={int(summary['argmax_match'])}"
        )

        normal_cache = normal_cache_next.astype(np.float32)
        debug_cache = debug_cache_next.astype(np.float32)

    metadata_path = dump_dir / 'metadata.json'
    summaries_path = dump_dir / 'step_summaries.jsonl'
    metadata['artifacts'] = {
        'full_feats_npy': str(dump_dir / 'full_feats.npy'),
        'metadata_json': str(metadata_path),
        'step_summaries_jsonl': str(summaries_path),
    }
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2)
    metadata_path.write_text(metadata_text, encoding='utf-8')
    metadata_path.with_suffix('.txt').write_text(metadata_text, encoding='utf-8')
    with summaries_path.open('w', encoding='utf-8') as f:
        for item in summaries:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    summaries_path.with_suffix('.txt').write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in summaries), encoding='utf-8')

    print(f'dump_dir: {dump_dir}')
    print(f'metadata_json: {metadata_path}')
    print(f'step_summaries_jsonl: {summaries_path}')


if __name__ == '__main__':
    main()
