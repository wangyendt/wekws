#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
S0_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[4]
if str(S0_DIR) not in sys.path:
    sys.path.insert(0, str(S0_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import infer_wav as iw
from export_streaming_litert_tflite import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    infer_cache_shape,
    load_streaming_wrapper,
    resolve_streaming_output_path,
)
from tflite_quant_utils import (
    dequantize_from_detail,
    get_tensor_quant_params,
    load_tflite_interpreter,
    quantize_to_detail,
)


def _to_jsonable_detail(detail: dict) -> dict:
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


def _pick_io_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 2 or len(output_details) != 2:
        raise RuntimeError(
            f'Expected 2 inputs and 2 outputs for streaming model, got {len(input_details)} inputs and {len(output_details)} outputs.'
        )

    feat_input = next((item for item in input_details if len(item['shape']) == 3), None)
    cache_input = next((item for item in input_details if len(item['shape']) == 4), None)
    logits_output = next((item for item in output_details if len(item['shape']) == 3), None)
    cache_output = next((item for item in output_details if len(item['shape']) == 4), None)
    if feat_input is None or cache_input is None or logits_output is None or cache_output is None:
        raise RuntimeError('Failed to identify feature/cache inputs or logits/cache outputs from TFLite interpreter.')
    return feat_input, cache_input, logits_output, cache_output


def _tensor_stats(array) -> dict:
    arr = np.asarray(array, dtype=np.float32)
    flat = arr.reshape(-1)
    if flat.size == 0:
        return {
            'shape': list(arr.shape),
            'numel': 0,
            'min': None,
            'max': None,
            'mean': None,
            'std': None,
        }
    return {
        'shape': list(arr.shape),
        'numel': int(flat.size),
        'min': float(flat.min()),
        'max': float(flat.max()),
        'mean': float(flat.mean()),
        'std': float(flat.std()),
    }


def _diff_stats(lhs, rhs) -> dict:
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


def _save_array(base_dir: Path, rel_name: str, array) -> str:
    out_path = base_dir / rel_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    np.save(out_path, arr)
    _write_array_txt(out_path.with_suffix('.txt'), arr)
    return str(out_path.with_suffix('.txt'))


def _chunk_with_padding(feats: torch.Tensor, start: int, chunk_frames: int) -> torch.Tensor:
    chunk = feats[:, start:start + chunk_frames, :]
    if chunk.size(1) == chunk_frames:
        return chunk
    padded = torch.zeros((1, chunk_frames, feats.size(2)), dtype=feats.dtype)
    padded[:, :chunk.size(1), :] = chunk
    return padded


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dump full streaming model trace for one wav, including features, quantized inputs, caches, logits, and per-step diffs.'
    )
    parser.add_argument('--wav', required=True, help='input wav path')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG), help='model config yaml')
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT), help='checkpoint path')
    parser.add_argument('--tflite', default='', help='tflite path, default derived from checkpoint')
    parser.add_argument('--chunk_frames', type=int, default=1, help='streaming feature frames per step')
    parser.add_argument('--max_steps', type=int, default=0, help='max streaming steps to dump; <=0 means all')
    parser.add_argument('--dump_dir', required=True, help='output directory for trace files')
    return parser.parse_args()


def main():
    args = parse_args()

    wav_path = iw.to_abs_path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(f'wav not found: {wav_path}')

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    tflite_path = resolve_streaming_output_path(checkpoint_path, args.tflite, quant_mode='int8')
    if not tflite_path.exists():
        raise FileNotFoundError(f'tflite model not found: {tflite_path}')

    dump_dir = Path(args.dump_dir).expanduser().resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)
    steps_dir = dump_dir / 'steps'
    steps_dir.mkdir(parents=True, exist_ok=True)

    configs, wrapper = load_streaming_wrapper(config_path, checkpoint_path)
    feats = iw.build_input_features(wav_path, configs).cpu()
    cache_shape = infer_cache_shape(configs)
    chunk_frames = int(args.chunk_frames)

    interpreter = load_tflite_interpreter(tflite_path)
    feat_input, cache_input, logits_output, cache_output = _pick_io_details(interpreter)
    interpreter.allocate_tensors()

    feat_input_scale, feat_input_zero_point = get_tensor_quant_params(feat_input)
    cache_input_scale, cache_input_zero_point = get_tensor_quant_params(cache_input)
    logits_output_scale, logits_output_zero_point = get_tensor_quant_params(logits_output)
    cache_output_scale, cache_output_zero_point = get_tensor_quant_params(cache_output)

    metadata = {
        'wav': str(wav_path),
        'config': str(config_path),
        'checkpoint': str(checkpoint_path),
        'tflite': str(tflite_path),
        'chunk_frames': chunk_frames,
        'cache_shape': list(cache_shape),
        'full_feats_shape': list(feats.shape),
        'num_total_frames': int(feats.size(1)),
        'num_steps_total': int((feats.size(1) + chunk_frames - 1) // chunk_frames),
        'num_steps_dumped': None,
        'configs': configs,
        'tflite_inputs': [_to_jsonable_detail(item) for item in interpreter.get_input_details()],
        'tflite_outputs': [_to_jsonable_detail(item) for item in interpreter.get_output_details()],
        'qparams': {
            'input0_feat': {'scale': feat_input_scale, 'zero_point': feat_input_zero_point},
            'input1_cache': {'scale': cache_input_scale, 'zero_point': cache_input_zero_point},
            'output0_logits': {'scale': logits_output_scale, 'zero_point': logits_output_zero_point},
            'output1_cache': {'scale': cache_output_scale, 'zero_point': cache_output_zero_point},
        },
        'artifacts': {},
    }

    metadata['artifacts']['full_feats_npy'] = _save_array(dump_dir, 'full_feats.npy', feats.numpy())

    torch_cache = torch.zeros(cache_shape, dtype=torch.float32)
    tflite_cache_fp32 = np.zeros(cache_shape, dtype=np.float32)
    step_summaries = []

    num_steps_total = metadata['num_steps_total']
    num_steps_dump = num_steps_total if args.max_steps <= 0 else min(num_steps_total, int(args.max_steps))
    metadata['num_steps_dumped'] = num_steps_dump

    for step_idx in range(num_steps_dump):
        start = step_idx * chunk_frames
        end = min(start + chunk_frames, feats.size(1))
        valid_frames = int(end - start)
        feat_chunk = _chunk_with_padding(feats, start, chunk_frames)
        feat_chunk_np = feat_chunk.numpy().astype(np.float32)
        cache_in_torch_np = torch_cache.numpy().astype(np.float32)
        cache_in_tflite_np = tflite_cache_fp32.astype(np.float32, copy=True)

        with torch.no_grad():
            torch_logits, torch_cache_next = wrapper(feat_chunk, torch_cache)
        torch_logits_np = torch_logits.cpu().numpy().astype(np.float32)
        torch_cache_next_np = torch_cache_next.cpu().numpy().astype(np.float32)

        feat_in_quant = quantize_to_detail(feat_chunk_np, feat_input)
        feat_in_dequant = dequantize_from_detail(feat_in_quant, feat_input)
        cache_in_quant = quantize_to_detail(cache_in_tflite_np, cache_input)
        cache_in_dequant = dequantize_from_detail(cache_in_quant, cache_input)

        interpreter.set_tensor(feat_input['index'], feat_in_quant)
        interpreter.set_tensor(cache_input['index'], cache_in_quant)
        interpreter.invoke()

        logits_out_raw = interpreter.get_tensor(logits_output['index'])
        cache_out_raw = interpreter.get_tensor(cache_output['index'])
        logits_out_dequant = dequantize_from_detail(logits_out_raw, logits_output)
        cache_out_dequant = dequantize_from_detail(cache_out_raw, cache_output)

        step_dir_rel = Path('steps') / f'step_{step_idx:04d}'
        step_summary = {
            'step': step_idx,
            'frame_range': [int(start), int(end)],
            'valid_frames': valid_frames,
            'artifacts': {
                'feat_chunk_fp32': _save_array(dump_dir, str(step_dir_rel / 'feat_chunk_fp32.npy'), feat_chunk_np),
                'feat_input_quant': _save_array(dump_dir, str(step_dir_rel / 'feat_input_quant.npy'), feat_in_quant),
                'feat_input_dequant': _save_array(dump_dir, str(step_dir_rel / 'feat_input_dequant.npy'), feat_in_dequant),
                'cache_in_torch_fp32': _save_array(dump_dir, str(step_dir_rel / 'cache_in_torch_fp32.npy'), cache_in_torch_np),
                'cache_in_tflite_fp32': _save_array(dump_dir, str(step_dir_rel / 'cache_in_tflite_fp32.npy'), cache_in_tflite_np),
                'cache_input_quant': _save_array(dump_dir, str(step_dir_rel / 'cache_input_quant.npy'), cache_in_quant),
                'cache_input_dequant': _save_array(dump_dir, str(step_dir_rel / 'cache_input_dequant.npy'), cache_in_dequant),
                'torch_logits_fp32': _save_array(dump_dir, str(step_dir_rel / 'torch_logits_fp32.npy'), torch_logits_np),
                'tflite_logits_raw': _save_array(dump_dir, str(step_dir_rel / 'tflite_logits_raw.npy'), logits_out_raw),
                'tflite_logits_dequant': _save_array(dump_dir, str(step_dir_rel / 'tflite_logits_dequant.npy'), logits_out_dequant),
                'torch_cache_out_fp32': _save_array(dump_dir, str(step_dir_rel / 'torch_cache_out_fp32.npy'), torch_cache_next_np),
                'tflite_cache_raw': _save_array(dump_dir, str(step_dir_rel / 'tflite_cache_raw.npy'), cache_out_raw),
                'tflite_cache_dequant': _save_array(dump_dir, str(step_dir_rel / 'tflite_cache_dequant.npy'), cache_out_dequant),
            },
            'stats': {
                'feat_chunk_fp32': _tensor_stats(feat_chunk_np),
                'feat_input_dequant_vs_fp32': _diff_stats(feat_in_dequant, feat_chunk_np),
                'cache_in_dequant_vs_fp32': _diff_stats(cache_in_dequant, cache_in_tflite_np),
                'logits_tflite_vs_torch': _diff_stats(logits_out_dequant, torch_logits_np),
                'cache_out_tflite_vs_torch': _diff_stats(cache_out_dequant, torch_cache_next_np),
            },
            'argmax_match': bool(np.array_equal(np.argmax(logits_out_dequant, axis=-1), np.argmax(torch_logits_np, axis=-1))),
        }
        step_summaries.append(step_summary)

        torch_cache = torch_cache_next.detach().cpu()
        tflite_cache_fp32 = cache_out_dequant.astype(np.float32)

        print(
            f"step={step_idx:04d} frames=[{start},{end}) valid={valid_frames} "
            f"feat_q_max_abs={step_summary['stats']['feat_input_dequant_vs_fp32']['max_abs']:.8f} "
            f"logits_max_abs={step_summary['stats']['logits_tflite_vs_torch']['max_abs']:.8f} "
            f"cache_max_abs={step_summary['stats']['cache_out_tflite_vs_torch']['max_abs']:.8f} "
            f"argmax_match={int(step_summary['argmax_match'])}"
        )

    metadata_path = dump_dir / 'metadata.json'
    steps_path = dump_dir / 'step_summaries.jsonl'
    metadata['artifacts']['metadata_json'] = str(metadata_path)
    metadata['artifacts']['step_summaries_jsonl'] = str(steps_path)
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2)
    metadata_path.write_text(metadata_text, encoding='utf-8')
    metadata_path.with_suffix('.txt').write_text(metadata_text, encoding='utf-8')
    with steps_path.open('w', encoding='utf-8') as f:
        for item in step_summaries:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    steps_path.with_suffix('.txt').write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in step_summaries), encoding='utf-8')

    print(f'dump_dir: {dump_dir}')
    print(f'metadata_json: {metadata_path}')
    print(f'step_summaries_jsonl: {steps_path}')


if __name__ == '__main__':
    main()
