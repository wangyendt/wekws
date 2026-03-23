#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch

from export_streaming_litert_tflite import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    infer_cache_shape,
    load_streaming_wrapper,
    resolve_streaming_output_path,
)
from tflite_quant_utils import (
    dequantize_from_detail,
    load_tflite_interpreter,
    quantize_to_detail,
)


def load_interpreter(model_path: Path):
    return load_tflite_interpreter(model_path)


def pick_io_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 2 or len(output_details) != 2:
        raise RuntimeError(
            f"Expected 2 inputs and 2 outputs for streaming model, got {len(input_details)} inputs and {len(output_details)} outputs."
        )

    feat_input = next((item for item in input_details if len(item["shape"]) == 3), None)
    cache_input = next((item for item in input_details if len(item["shape"]) == 4), None)
    logits_output = next((item for item in output_details if len(item["shape"]) == 3), None)
    cache_output = next((item for item in output_details if len(item["shape"]) == 4), None)
    if feat_input is None or cache_input is None or logits_output is None or cache_output is None:
        raise RuntimeError("Failed to identify feature/cache inputs or logits/cache outputs from TFLite interpreter.")
    return feat_input, cache_input, logits_output, cache_output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare streaming-step PyTorch outputs against exported streaming TFLite outputs."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="model config yaml")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="checkpoint path")
    parser.add_argument(
        "--tflite",
        default="",
        help="tflite path, default: <checkpoint_dir>/<checkpoint_stem>_stream_litert_fp32.tflite",
    )
    parser.add_argument("--num_steps", type=int, default=100, help="number of streaming steps to compare")
    parser.add_argument("--chunk_frames", type=int, default=1, help="feature frames per streaming step")
    parser.add_argument("--seed", type=int, default=20260320, help="random seed")
    parser.add_argument(
        "--value_scale",
        type=float,
        default=1.0,
        help="stddev scale for the random normal feature input",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    tflite_path = resolve_streaming_output_path(checkpoint_path, args.tflite)

    if not tflite_path.exists():
        raise FileNotFoundError(
            f"tflite model not found: {tflite_path}\n"
            "Export it first with torch2lite/export_streaming_litert_tflite.py."
        )

    configs, wrapper = load_streaming_wrapper(config_path, checkpoint_path)
    input_dim = int(configs["model"]["input_dim"])
    cache_shape = infer_cache_shape(configs)

    interpreter = load_interpreter(tflite_path)
    feat_input, cache_input, logits_output, cache_output = pick_io_details(interpreter)
    interpreter.allocate_tensors()

    feat_shape = tuple(int(dim) for dim in feat_input["shape"])
    if feat_shape[1] != args.chunk_frames:
        raise ValueError(
            f"TFLite chunk_frames={feat_shape[1]} but compare script received --chunk_frames={args.chunk_frames}"
        )
    if tuple(int(dim) for dim in cache_input["shape"]) != cache_shape:
        raise ValueError(
            f"TFLite cache shape {tuple(int(dim) for dim in cache_input['shape'])} != expected {cache_shape}"
        )

    rng = np.random.default_rng(args.seed)
    torch_cache = torch.zeros(cache_shape, dtype=torch.float32)
    tflite_cache = np.zeros(cache_shape, dtype=np.float32)

    logits_max_abs_values = []
    logits_mean_abs_values = []
    logits_rmse_values = []
    cache_max_abs_values = []
    cache_mean_abs_values = []
    argmax_match_values = []

    print(f"config: {config_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"tflite: {tflite_path}")
    print(f"feat_shape: {feat_shape}")
    print(f"cache_shape: {cache_shape}")
    print(f"num_steps: {args.num_steps}")
    print(f"seed: {args.seed}")

    for step_idx in range(args.num_steps):
        feats = rng.normal(
            loc=0.0,
            scale=args.value_scale,
            size=(1, args.chunk_frames, input_dim),
        ).astype(np.float32)

        with torch.no_grad():
            torch_logits, torch_cache = wrapper(torch.from_numpy(feats), torch_cache)
            torch_logits = torch_logits.cpu().numpy()
            torch_cache_np = torch_cache.cpu().numpy()

        interpreter.set_tensor(
            feat_input["index"],
            quantize_to_detail(feats, feat_input),
        )
        interpreter.set_tensor(
            cache_input["index"],
            quantize_to_detail(tflite_cache, cache_input),
        )
        interpreter.invoke()
        tflite_logits = dequantize_from_detail(
            interpreter.get_tensor(logits_output["index"]),
            logits_output,
        )
        tflite_cache = dequantize_from_detail(
            interpreter.get_tensor(cache_output["index"]),
            cache_output,
        )

        logits_diff = tflite_logits.astype(np.float64) - torch_logits.astype(np.float64)
        logits_abs = np.abs(logits_diff)
        cache_diff = tflite_cache.astype(np.float64) - torch_cache_np.astype(np.float64)
        cache_abs = np.abs(cache_diff)

        logits_max_abs_values.append(float(logits_abs.max()))
        logits_mean_abs_values.append(float(logits_abs.mean()))
        logits_rmse_values.append(float(np.sqrt(np.mean(logits_diff ** 2))))
        cache_max_abs_values.append(float(cache_abs.max()))
        cache_mean_abs_values.append(float(cache_abs.mean()))
        argmax_match_values.append(float(np.mean(np.argmax(tflite_logits, axis=-1) == np.argmax(torch_logits, axis=-1))))

        print(
            f"step={step_idx:03d} "
            f"logits_max_abs={logits_max_abs_values[-1]:.8f} "
            f"logits_mean_abs={logits_mean_abs_values[-1]:.8f} "
            f"logits_rmse={logits_rmse_values[-1]:.8f} "
            f"cache_max_abs={cache_max_abs_values[-1]:.8f} "
            f"cache_mean_abs={cache_mean_abs_values[-1]:.8f} "
            f"argmax_match={argmax_match_values[-1]:.6f}"
        )

    print("summary:")
    print(f"  sample_max_of_logits_max_abs={max(logits_max_abs_values):.8f}")
    print(f"  sample_mean_of_logits_mean_abs={float(np.mean(logits_mean_abs_values)):.8f}")
    print(f"  sample_mean_of_logits_rmse={float(np.mean(logits_rmse_values)):.8f}")
    print(f"  sample_max_of_cache_max_abs={max(cache_max_abs_values):.8f}")
    print(f"  sample_mean_of_cache_mean_abs={float(np.mean(cache_mean_abs_values)):.8f}")
    print(f"  sample_mean_argmax_match={float(np.mean(argmax_match_values)):.6f}")


if __name__ == "__main__":
    main()
