# Copyright 2026 Wayne
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Export KWS model to ExecuTorch (.pte) with PT2E pipeline.

Supports:
  - fp32: torch.export -> ExecuTorch
  - int8: PT2E post-training quantization -> torch.export -> ExecuTorch

Backends:
  - xnnpack: default CPU backend
  - portable: no backend partitioner
  - hifi4: Cadence AOT pipeline (HiFi4 target toolchain/runtime required)
"""

from __future__ import print_function

import argparse
import copy
import logging
import os
import random
import tempfile
import time
from typing import List, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

from wekws.dataset.init_dataset import init_dataset
from wekws.model.kws_model import init_model
from wekws.utils.checkpoint import load_checkpoint
from wenet.text.char_tokenizer import CharTokenizer


class KwsLogitsWrapper(torch.nn.Module):
    """Export wrapper: keep only logits output for downstream KWS scoring."""
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(feats)
        return logits


def get_args():
    parser = argparse.ArgumentParser(
        description='Export KWS to ExecuTorch (.pte) with PT2E')
    parser.add_argument('--config', required=True, help='model config yaml')
    parser.add_argument('--checkpoint', required=True, help='source checkpoint')
    parser.add_argument('--output_model',
                        default=None,
                        help='output .pte path (default: <ckpt_dir>/<ckpt>_executorch_<quant>.pte)')
    parser.add_argument('--quant_type',
                        choices=['fp32', 'int8'],
                        default='fp32',
                        help='export type')
    parser.add_argument('--backend',
                        choices=['xnnpack', 'portable', 'hifi4', 'cadence',
                                 'cadence_hifi4'],
                        default='xnnpack',
                        help='target backend (cadence/cadence_hifi4 are aliases of hifi4)')
    parser.add_argument('--dict', default='dict_top20', help='dict dir')
    parser.add_argument('--calib_data',
                        default='data/train/data.list',
                        help='calibration data list (for int8)')
    parser.add_argument('--num_calib',
                        type=int,
                        default=200,
                        help='number of calibration samples (for int8)')
    parser.add_argument('--batch_size', type=int, default=1, help='calib batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='calib workers')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--export_seq_len',
                        type=int,
                        default=100,
                        help='sequence length used in example inputs')
    parser.add_argument('--calib_log_interval',
                        type=int,
                        default=10,
                        help='log every N calibration batches (for int8)')
    parser.add_argument('--disable_xnnpack',
                        action='store_true',
                        default=False,
                        help='deprecated: same as --backend portable when backend is xnnpack')
    parser.add_argument('--hifi4_opt_level',
                        type=int,
                        default=1,
                        help='Cadence/HiFi4 frontend optimization level')
    parser.add_argument('--hifi4_quantizer',
                        choices=['wakeword', 'default'],
                        default='wakeword',
                        help='Cadence/HiFi4 int8 quantizer profile')
    parser.add_argument('--require_full_int8',
                        action='store_true',
                        default=False,
                        help='best-effort check: fail export when float compute ops are detected')
    return parser.parse_args()


def _resolve_backend(args) -> str:
    backend = args.backend
    if backend in ('cadence', 'cadence_hifi4'):
        backend = 'hifi4'

    if args.disable_xnnpack:
        if backend != 'xnnpack':
            raise ValueError('--disable_xnnpack cannot be used with --backend {}'
                             .format(args.backend))
        logging.warning('--disable_xnnpack is deprecated, use --backend portable.')
        backend = 'portable'
    return backend


def _normalize_seq_len(feats: torch.Tensor, target_len: int) -> torch.Tensor:
    # feats: [B, T, F]
    cur = feats.size(1)
    if cur == target_len:
        return feats
    if cur > target_len:
        return feats[:, :target_len, :]
    pad = torch.zeros(feats.size(0),
                      target_len - cur,
                      feats.size(2),
                      dtype=feats.dtype)
    return torch.cat([feats, pad], dim=1)


def subsample_data_list(data_list_file: str,
                        num_samples: int,
                        seed: int = 42) -> Tuple[str, bool, int]:
    with open(data_list_file, 'r', encoding='utf8') as f:
        lines = [line for line in f if line.strip()]
    total = len(lines)
    if num_samples >= total:
        return data_list_file, False, total

    random.seed(seed)
    sampled = random.sample(lines, num_samples)
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='_et_calib.list', delete=False, dir='.')
    tmp.writelines(sampled)
    tmp.close()
    return tmp.name, True, total


def create_calib_dataloader(configs,
                            calib_data_file: str,
                            dict_dir: str,
                            batch_size: int = 1,
                            num_workers: int = 4):
    calib_conf = copy.deepcopy(configs['dataset_conf'])
    calib_conf['filter_conf']['max_length'] = 102400
    calib_conf['filter_conf']['min_length'] = 0
    calib_conf['filter_conf']['token_max_length'] = 10240
    calib_conf['filter_conf']['token_min_length'] = 1
    calib_conf['filter_conf']['min_output_input_ratio'] = 1e-6
    calib_conf['filter_conf']['max_output_input_ratio'] = 1
    calib_conf['speed_perturb'] = False
    calib_conf['spec_aug'] = False
    calib_conf['shuffle'] = False
    feats_type = calib_conf.get('feats_type', 'fbank')
    calib_conf[f'{feats_type}_conf']['dither'] = 0.0
    calib_conf['batch_conf']['batch_size'] = batch_size

    dict_file = os.path.join(dict_dir, 'dict.txt')
    words_file = os.path.join(dict_dir, 'words.txt')
    tokenizer = CharTokenizer(dict_file,
                              words_file,
                              unk='<filler>',
                              split_with_space=True)

    dataset = init_dataset(data_list_file=calib_data_file,
                           conf=calib_conf,
                           tokenizer=tokenizer,
                           split='test')
    prefetch = 2 if num_workers > 0 else None
    return DataLoader(dataset,
                      batch_size=None,
                      pin_memory=False,
                      num_workers=num_workers,
                      prefetch_factor=prefetch)


def _lower_exported_to_executorch(exported_program, backend: str):
    from executorch.exir import to_edge_transform_and_lower

    if backend == 'xnnpack':
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import \
            XnnpackPartitioner
        # Compat across ExecuTorch versions:
        # newer builds use kwarg `partitioner`, some older ones use `partitioners`.
        partitioner_list = [XnnpackPartitioner()]
        try:
            edge_mgr = to_edge_transform_and_lower(
                exported_program, partitioner=partitioner_list)
        except TypeError as e:
            if 'partitioner' not in str(e):
                raise
            edge_mgr = to_edge_transform_and_lower(
                exported_program, partitioners=partitioner_list)
    elif backend == 'portable':
        edge_mgr = to_edge_transform_and_lower(exported_program)
    else:
        raise ValueError('Unsupported backend for generic lowering: {}'.format(backend))
    return edge_mgr.to_executorch()


def _save_executorch_program(exec_prog, output_model: str):
    with open(output_model, 'wb') as f:
        exec_prog.write_to_file(f)


def _export_int8_pt2e_xnnpack(wrapper: torch.nn.Module,
                              example_inputs,
                              configs,
                              args) -> torch.export.ExportedProgram:
    from torchao.quantization.pt2e.quantize_pt2e import \
        prepare_pt2e, convert_pt2e
    from torchao.quantization.pt2e import move_exported_model_to_eval
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import \
        XNNPACKQuantizer, get_symmetric_quantization_config
    logging.info('Using torchao PT2E quantization API.')

    model_graph = torch.export.export_for_training(
        wrapper, example_inputs).module()

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=False))
    prepared = prepare_pt2e(model_graph, quantizer)
    move_exported_model_to_eval(prepared)

    calib_list = args.calib_data
    tmp_created = False
    if args.num_calib > 0:
        calib_list, tmp_created, total = subsample_data_list(
            args.calib_data, args.num_calib, args.seed)
        logging.info('Calibration samples: %d (from total %d)',
                     min(args.num_calib, total), total)

    try:
        calib_loader = create_calib_dataloader(
            configs=configs,
            calib_data_file=calib_list,
            dict_dir=args.dict,
            batch_size=args.batch_size,
            num_workers=args.num_workers)

        calib_start = time.time()
        calib_batches = 0
        calib_samples = 0
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(calib_loader):
                feats = batch_dict['feats'].cpu()
                feats = _normalize_seq_len(feats, args.export_seq_len)
                prepared(feats)
                calib_batches += 1
                calib_samples += int(feats.size(0))
                if args.calib_log_interval > 0 and \
                        (batch_idx + 1) % args.calib_log_interval == 0:
                    logging.info('Calibration progress: %d batches, %d samples',
                                 batch_idx + 1, calib_samples)
        logging.info('Calibration done: %d batches, %d samples, %.1fs',
                     calib_batches, calib_samples, time.time() - calib_start)
    finally:
        if tmp_created and os.path.exists(calib_list):
            os.remove(calib_list)

    quantized_model = convert_pt2e(prepared)
    move_exported_model_to_eval(quantized_model)
    return torch.export.export(quantized_model, example_inputs)


def _build_calibration_tuples(configs, args) -> List[Tuple[torch.Tensor]]:
    calib_list = args.calib_data
    tmp_created = False
    if args.num_calib > 0:
        calib_list, tmp_created, total = subsample_data_list(
            args.calib_data, args.num_calib, args.seed)
        logging.info('Calibration samples: %d (from total %d)',
                     min(args.num_calib, total), total)

    tuples: List[Tuple[torch.Tensor]] = []
    try:
        calib_loader = create_calib_dataloader(
            configs=configs,
            calib_data_file=calib_list,
            dict_dir=args.dict,
            batch_size=args.batch_size,
            num_workers=args.num_workers)
        calib_start = time.time()
        calib_batches = 0
        calib_samples = 0
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(calib_loader):
                feats = batch_dict['feats'].cpu()
                feats = _normalize_seq_len(feats, args.export_seq_len)
                tuples.append((feats,))
                calib_batches += 1
                calib_samples += int(feats.size(0))
                if args.calib_log_interval > 0 and \
                        (batch_idx + 1) % args.calib_log_interval == 0:
                    logging.info('Calibration tuple progress: %d batches, %d samples',
                                 batch_idx + 1, calib_samples)
        logging.info('Calibration tuples ready: %d batches, %d samples, %.1fs',
                     calib_batches, calib_samples, time.time() - calib_start)
    finally:
        if tmp_created and os.path.exists(calib_list):
            os.remove(calib_list)
    return tuples


def _export_hifi4_to_executorch(wrapper: torch.nn.Module,
                                example_inputs,
                                configs,
                                args):
    from executorch.backends.cadence.aot.compiler import \
        export_to_edge, get_cadence_passes, quantize_and_export_to_edge

    if args.quant_type == 'fp32':
        edge_mgr = export_to_edge(wrapper, example_inputs)
        exported_for_check = edge_mgr.exported_program()
    else:
        from executorch.backends.cadence.aot.quantizer.quantizer import \
            CadenceDefaultQuantizer, CadenceWakeWordQuantizer
        if args.hifi4_quantizer == 'wakeword':
            quantizer = CadenceWakeWordQuantizer()
        else:
            quantizer = CadenceDefaultQuantizer()
        calib_tuples = _build_calibration_tuples(configs, args)
        edge_mgr = quantize_and_export_to_edge(
            wrapper,
            example_inputs,
            quantizer=quantizer,
            calibration_data=calib_tuples)
        exported_for_check = edge_mgr.exported_program()

    cadence_passes = list(get_cadence_passes(args.hifi4_opt_level))
    cadence_mgr = edge_mgr.transform(cadence_passes)
    return cadence_mgr.to_executorch(), exported_for_check


def _find_float_compute_ops(exported_program) -> Tuple[List[str], int]:
    float_dtypes = {
        torch.float16, torch.float32, torch.float64, torch.bfloat16
    }
    hits: List[str] = []
    checked_nodes = 0

    for node in exported_program.graph_module.graph.nodes:
        if node.op != 'call_function':
            continue
        target_str = str(node.target)
        if 'quantize' in target_str or 'dequantize' in target_str:
            continue

        val = node.meta.get('val', None)
        vals = val if isinstance(val, (list, tuple)) else [val]
        has_meta = False
        is_float = False
        for v in vals:
            if isinstance(v, torch.Tensor):
                has_meta = True
                if v.dtype in float_dtypes:
                    is_float = True
        if has_meta:
            checked_nodes += 1
            if is_float:
                hits.append(target_str)
    return hits, checked_nodes


def _maybe_require_full_int8(exported_program, backend: str, args):
    if args.quant_type != 'int8' or not args.require_full_int8:
        return
    if exported_program is None:
        raise RuntimeError(
            'Cannot run --require_full_int8 check for backend {} in current pipeline.'
            .format(backend))

    float_ops, checked_nodes = _find_float_compute_ops(exported_program)
    if checked_nodes == 0:
        raise RuntimeError(
            'No dtype metadata found for int8 strict check; export aborted. '
            'Try backend=xnnpack or disable --require_full_int8.')
    if float_ops:
        preview = ', '.join(float_ops[:12])
        raise RuntimeError(
            'Full-int8 check failed: detected float compute ops (showing up to 12): {}'
            .format(preview))
    logging.info('Full-int8 check passed (best effort): no float compute ops detected.')


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    backend = _resolve_backend(args)

    import executorch  # noqa: F401

    if not os.path.exists(args.config):
        raise FileNotFoundError('config not found: {}'.format(args.config))
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError('checkpoint not found: {}'.format(args.checkpoint))

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    model = init_model(configs['model'])
    load_checkpoint(model, args.checkpoint)
    model.eval()
    model.cpu()
    wrapper = KwsLogitsWrapper(model).eval().cpu()

    input_dim = int(configs['model']['input_dim'])
    example_input = torch.randn(1, args.export_seq_len, input_dim, dtype=torch.float32)
    example_inputs = (example_input,)

    checkpoint_dir = os.path.dirname(args.checkpoint)
    checkpoint_basename = os.path.basename(args.checkpoint).rsplit('.', 1)[0]
    output_model = args.output_model
    if not output_model:
        backend_suffix = ''
        if backend != 'xnnpack':
            backend_suffix = '_{}'.format(backend)
        output_model = os.path.join(
            checkpoint_dir,
            '{}_executorch{}_{:s}.pte'.format(
                checkpoint_basename, backend_suffix, args.quant_type))
    os.makedirs(os.path.dirname(os.path.abspath(output_model)) or '.',
                exist_ok=True)

    logging.info('Export type: %s', args.quant_type)
    logging.info('Backend: %s', backend)
    logging.info('Output model: %s', output_model)

    if backend == 'hifi4':
        exec_prog, exported_for_check = _export_hifi4_to_executorch(
            wrapper, example_inputs, configs, args)
        _maybe_require_full_int8(exported_for_check, backend, args)
    else:
        if args.quant_type == 'fp32':
            exported = torch.export.export(wrapper, example_inputs)
        else:
            exported = _export_int8_pt2e_xnnpack(wrapper, example_inputs, configs, args)
        _maybe_require_full_int8(exported, backend, args)
        exec_prog = _lower_exported_to_executorch(exported, backend=backend)

    _save_executorch_program(exec_prog, output_model)
    logging.info('Done. ExecuTorch model saved: %s', output_model)


if __name__ == '__main__':
    main()
