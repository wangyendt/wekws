# Copyright 2026 Wayne
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Post-Training Quantization (PTQ) for KWS models.

Supports:
  - INT8:  PyTorch static quantization (qnnpack) -> TorchScript (.zip)
  - INT16: Simulated quantization (weight quantize-dequantize to 16-bit,
           stored as float32 .pt for accuracy evaluation)

Usage:
    python wekws/bin/ptq_quantize.py \
        --config exp/.../config.yaml \
        --checkpoint exp/.../229.pt \
        --calib_data data/train/data.list \
        --num_calib 200 \
        --quant_type int8 \
        --dict dict_top20
"""

from __future__ import print_function

import argparse
import copy
import logging
import os
import random
import sys
import tempfile

import torch
import yaml
from torch.utils.data import DataLoader

from wekws.dataset.init_dataset import init_dataset
from wekws.model.kws_model import init_model

try:
    from wenet.text.char_tokenizer import CharTokenizer
except ImportError:
    CharTokenizer = None


def get_args():
    parser = argparse.ArgumentParser(
        description='Post-Training Quantization for KWS models')
    parser.add_argument('--config', required=True,
                        help='model config yaml file')
    parser.add_argument('--checkpoint', required=True,
                        help='source model checkpoint (.pt)')
    parser.add_argument('--calib_data', default='data/train/data.list',
                        help='calibration data file (data.list format)')
    parser.add_argument('--num_calib', type=int, default=200,
                        help='number of calibration samples to use')
    parser.add_argument('--quant_type', choices=['int8', 'int16'],
                        default='int8',
                        help='quantization type: int8 or int16')
    parser.add_argument('--output_dir', default=None,
                        help='output directory (default: same as checkpoint)')
    parser.add_argument('--dict', default='dict_top20',
                        help='dict directory for tokenizer')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for calibration')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for data subsampling')
    args = parser.parse_args()
    return args


def subsample_data_list(data_list_file, num_samples, seed=42):
    """Subsample N lines from a data.list (JSON Lines) file.

    Returns:
        path: path to the subsampled file (or original if num_samples >= total)
        is_tmp: True if a temporary file was created (caller should clean up)
        total: total number of lines in the original file
    """
    with open(data_list_file, 'r', encoding='utf8') as f:
        lines = [line for line in f if line.strip()]

    total = len(lines)
    if num_samples >= total:
        logging.info('Calibration: using all %d samples (requested %d)',
                     total, num_samples)
        return data_list_file, False, total

    random.seed(seed)
    sampled = random.sample(lines, num_samples)
    logging.info('Calibration: subsampled %d / %d samples (seed=%d)',
                 num_samples, total, seed)

    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='_calib.list', delete=False, dir='.')
    tmp.writelines(sampled)
    tmp.close()
    return tmp.name, True, total


def get_model_size_bytes(path):
    """Get file size in bytes."""
    return os.path.getsize(path)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_size(size_bytes):
    """Format byte size to human-readable string."""
    if size_bytes < 1024:
        return '{} B'.format(size_bytes)
    elif size_bytes < 1024 * 1024:
        return '{:.1f} KB'.format(size_bytes / 1024)
    else:
        return '{:.2f} MB'.format(size_bytes / (1024 * 1024))


def create_calib_dataloader(configs, calib_data_file, dict_dir,
                            batch_size=1, num_workers=4):
    """Create a DataLoader for calibration."""
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
    calib_conf['{}_conf'.format(feats_type)]['dither'] = 0.0
    calib_conf['batch_conf']['batch_size'] = batch_size

    tokenizer = None
    if CharTokenizer is not None:
        dict_file = os.path.join(dict_dir, 'dict.txt')
        words_file = os.path.join(dict_dir, 'words.txt')
        if os.path.exists(dict_file) and os.path.exists(words_file):
            tokenizer = CharTokenizer(dict_file, words_file,
                                      unk='<filler>',
                                      split_with_space=True)

    dataset = init_dataset(data_list_file=calib_data_file,
                           conf=calib_conf,
                           tokenizer=tokenizer,
                           split='test')
    prefetch = 2 if num_workers > 0 else None
    loader = DataLoader(dataset,
                        batch_size=None,
                        pin_memory=False,
                        num_workers=num_workers,
                        prefetch_factor=prefetch)
    return loader


def quantize_int8(model, calib_loader, output_path):
    """INT8 static quantization via PyTorch (qnnpack backend).

    Args:
        model: FP32 model (already in eval mode)
        calib_loader: calibration DataLoader
        output_path: path to save the TorchScript INT8 model (.zip)

    Returns:
        output_path: path to the saved model
    """
    torch.backends.quantized.engine = 'qnnpack'

    print('=' * 60)
    print('INT8 Static Quantization')
    print('=' * 60)

    # Step 1: Fuse modules (safe -- skip submodules that lack fuse_modules)
    print('\n[1/4] Fusing modules ...')
    if hasattr(model, 'fuse_modules'):
        try:
            model.fuse_modules()
        except AttributeError as e:
            # Some submodules (e.g. NoSubsampling) may not have fuse_modules;
            # fall back to fusing only the backbone.
            print('  Warning: full fuse_modules failed ({}), '
                  'fusing backbone only ...'.format(e))
            if hasattr(model, 'backbone') and hasattr(model.backbone,
                                                      'fuse_modules'):
                model.backbone.fuse_modules()

    # Step 2: Set qconfig
    print('[2/4] Setting qconfig (qnnpack) ...')
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')

    # Step 3: Prepare and calibrate
    print('[3/4] Preparing observers and calibrating ...')
    model_prepared = torch.quantization.prepare(model)

    num_batches = 0
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(calib_loader):
            feats = batch_dict['feats']
            # Model forward returns (logits, out_cache) - call normally
            _ = model_prepared(feats)
            num_batches += 1
            if batch_idx % 50 == 0:
                print('  Calibration: {} batches processed'.format(batch_idx))
                sys.stdout.flush()
    print('  Calibration done: {} batches total'.format(num_batches))

    # Step 4: Convert to quantized model
    print('[4/4] Converting to INT8 ...')
    model_int8 = torch.quantization.convert(model_prepared)

    # Save as TorchScript via trace (FSMN uses Tuple-typed forward inputs
    # inside nn.Sequential which is incompatible with torch.jit.script)
    print('\nSaving TorchScript INT8 model to: {}'.format(output_path))
    input_dim = model_int8.idim if hasattr(model_int8, 'idim') else 400
    example_input = torch.randn(1, 100, input_dim)
    example_cache = torch.zeros(0, 0, 0)
    with torch.no_grad():
        traced_model = torch.jit.trace(model_int8,
                                       (example_input, example_cache))
    traced_model.save(output_path)

    return output_path


def quantize_int16_simulated(model, output_path):
    """Simulated INT16 quantization (weight quantize-dequantize).

    Quantizes each weight tensor to 16-bit integer precision and
    dequantizes back to float32. The resulting model can be loaded
    and evaluated with the standard checkpoint loading pipeline.

    Args:
        model: FP32 model (already in eval mode)
        output_path: path to save the simulated INT16 model (.pt)

    Returns:
        output_path: path to the saved model
    """
    print('=' * 60)
    print('INT16 Simulated Quantization')
    print('=' * 60)

    num_quantized = 0
    total_params = 0
    max_quant_error = 0.0

    with torch.no_grad():
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.dim() < 1:
                continue  # skip scalar params

            max_val = param.data.abs().max().item()
            if max_val == 0:
                continue

            # Symmetric quantization to INT16 range [-32767, 32767]
            scale = max_val / 32767.0
            param_quantized = torch.round(param.data / scale)
            param_quantized = torch.clamp(param_quantized, -32767, 32767)
            param_dequantized = param_quantized * scale

            # Track quantization error
            error = (param.data - param_dequantized).abs().max().item()
            max_quant_error = max(max_quant_error, error)

            param.data.copy_(param_dequantized)
            num_quantized += 1

    print('\n  Quantized {} parameter tensors'.format(num_quantized))
    print('  Total parameters: {:,}'.format(total_params))
    print('  Max quantization error: {:.8f}'.format(max_quant_error))
    fp32_kb = total_params * 4 / 1024
    int16_kb = total_params * 2 / 1024
    ratio_pct = int16_kb / fp32_kb * 100 if fp32_kb > 0 else 0
    print('  Theoretical size: FP32={:.1f} KB -> INT16={:.1f} KB ({:.0f}%)'
          .format(fp32_kb, int16_kb, ratio_pct))

    # Save as regular state_dict checkpoint
    print('\nSaving simulated INT16 model to: {}'.format(output_path))
    state_dict = model.state_dict()
    torch.save(state_dict, output_path)

    return output_path


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    print('=' * 60)
    print('PTQ Quantization')
    print('=' * 60)
    print('  Config:      {}'.format(args.config))
    print('  Checkpoint:  {}'.format(args.checkpoint))
    print('  Quant type:  {}'.format(args.quant_type))
    print('  Calib data:  {}'.format(args.calib_data))
    print('  Num calib:   {}'.format(args.num_calib))
    print('  Dict:        {}'.format(args.dict))
    print('  Seed:        {}'.format(args.seed))
    print('=' * 60)
    print()

    # Check input files
    if not os.path.isfile(args.config):
        print('Error: config file not found: {}'.format(args.config))
        sys.exit(1)
    if not os.path.isfile(args.checkpoint):
        print('Error: checkpoint not found: {}'.format(args.checkpoint))
        sys.exit(1)
    if not os.path.isfile(args.calib_data):
        print('Error: calibration data not found: {}'.format(args.calib_data))
        sys.exit(1)

    # Output directory
    if args.output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Output path
    basename = os.path.splitext(os.path.basename(args.checkpoint))[0]
    if args.quant_type == 'int8':
        output_path = os.path.join(output_dir, '{}_int8.zip'.format(basename))
    else:
        output_path = os.path.join(output_dir,
                                   '{}_int16_sim.pt'.format(basename))

    # Load config
    with open(args.config, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Subsample calibration data
    calib_file, is_tmp, total_lines = subsample_data_list(
        args.calib_data, args.num_calib, seed=args.seed)

    try:
        # Create calibration dataloader
        calib_loader = create_calib_dataloader(
            configs, calib_file, args.dict,
            batch_size=args.batch_size,
            num_workers=args.num_workers)

        # Load model (always to CPU -- quantization is CPU-only)
        print('Loading model ...')
        model = init_model(configs['model'])
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.cpu().eval()

        total_p, trainable_p = count_parameters(model)
        print('  Parameters: {:,} total, {:,} trainable'.format(
            total_p, trainable_p))

        # Record original checkpoint size
        orig_size = get_model_size_bytes(args.checkpoint)
        print('  Original checkpoint size: {}'.format(format_size(orig_size)))
        print()

        # Quantize
        if args.quant_type == 'int8':
            quantize_int8(model, calib_loader, output_path)
        else:
            quantize_int16_simulated(model, output_path)

        # Report output size
        output_size = get_model_size_bytes(output_path)
        ratio = output_size / orig_size * 100 if orig_size > 0 else 0

        print()
        print('=' * 60)
        print('Summary')
        print('=' * 60)
        print('  Quant type:       {}'.format(args.quant_type))
        print('  Input checkpoint: {} ({})'.format(
            args.checkpoint, format_size(orig_size)))
        print('  Output model:     {} ({})'.format(
            output_path, format_size(output_size)))
        print('  Size ratio:       {:.1f}%'.format(ratio))
        print('  Calib samples:    {} / {}'.format(
            min(args.num_calib, total_lines), total_lines))
        print('=' * 60)

    finally:
        # Clean up temp file
        if is_tmp and os.path.exists(calib_file):
            os.unlink(calib_file)
            logging.info('Cleaned up temporary calibration file: %s',
                         calib_file)


if __name__ == '__main__':
    main()
