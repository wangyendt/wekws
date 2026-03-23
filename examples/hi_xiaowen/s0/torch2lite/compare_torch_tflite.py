#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch

from export_litert_tflite import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    load_logits_wrapper,
    resolve_output_path,
)
from tflite_quant_utils import (
    dequantize_from_detail,
    load_tflite_interpreter,
    quantize_to_detail,
)


def load_interpreter(model_path: Path):
    return load_tflite_interpreter(model_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch KWS logits against exported TFLite logits with random tensors."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="model config yaml")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="checkpoint path")
    parser.add_argument(
        "--tflite",
        default="",
        help="tflite path, default: <checkpoint_dir>/<checkpoint_stem>_litert_fp32.tflite",
    )
    parser.add_argument("--num_samples", type=int, default=20, help="number of random tests")
    parser.add_argument("--seed", type=int, default=20260320, help="random seed")
    parser.add_argument(
        "--seq_len",
        type=int,
        default=100,
        help="sequence length used when the tflite input shape is dynamic",
    )
    parser.add_argument(
        "--value_scale",
        type=float,
        default=1.0,
        help="stddev scale for the random normal input",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    tflite_path = resolve_output_path(checkpoint_path, args.tflite)

    if not tflite_path.exists():
        raise FileNotFoundError(
            f"tflite model not found: {tflite_path}\n"
            "Export it first with torch2lite/export_litert_tflite.py."
        )

    _, wrapper = load_logits_wrapper(config_path, checkpoint_path)
    interpreter = load_interpreter(tflite_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1 or len(output_details) != 1:
        raise RuntimeError(
            f"Expected single-input single-output model, got {len(input_details)} inputs and "
            f"{len(output_details)} outputs."
        )

    input_detail = input_details[0]
    output_detail = output_details[0]
    input_shape = [int(dim) for dim in input_detail["shape"]]
    shape_signature = input_detail.get("shape_signature", input_shape)
    input_shape = [
        args.seq_len if int(shape_signature[index]) <= 0 else int(input_shape[index])
        for index in range(len(input_shape))
    ]
    interpreter.resize_tensor_input(input_detail["index"], input_shape, strict=False)
    interpreter.allocate_tensors()

    rng = np.random.default_rng(args.seed)
    max_abs_values = []
    mean_abs_values = []
    rmse_values = []
    p99_abs_values = []
    argmax_match_values = []

    print(f"config: {config_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"tflite: {tflite_path}")
    print(f"input_shape: {tuple(input_shape)}")
    print(f"num_samples: {args.num_samples}")
    print(f"seed: {args.seed}")

    for sample_idx in range(args.num_samples):
        sample = rng.normal(loc=0.0, scale=args.value_scale, size=input_shape).astype(np.float32)
        torch_input = torch.from_numpy(sample)

        with torch.no_grad():
            torch_logits = wrapper(torch_input).cpu().numpy()

        interpreter.set_tensor(
            input_detail["index"],
            quantize_to_detail(sample, input_detail),
        )
        interpreter.invoke()
        tflite_logits = dequantize_from_detail(
            interpreter.get_tensor(output_detail["index"]),
            output_detail,
        )

        diff = tflite_logits.astype(np.float64) - torch_logits.astype(np.float64)
        abs_diff = np.abs(diff)

        max_abs_values.append(float(abs_diff.max()))
        mean_abs_values.append(float(abs_diff.mean()))
        rmse_values.append(float(np.sqrt(np.mean(diff**2))))
        p99_abs_values.append(float(np.percentile(abs_diff, 99)))
        argmax_match_values.append(float(np.mean(np.argmax(tflite_logits, axis=-1) == np.argmax(torch_logits, axis=-1))))

        print(
            f"sample={sample_idx:02d} "
            f"max_abs={max_abs_values[-1]:.8f} "
            f"mean_abs={mean_abs_values[-1]:.8f} "
            f"rmse={rmse_values[-1]:.8f} "
            f"p99_abs={p99_abs_values[-1]:.8f} "
            f"argmax_match={argmax_match_values[-1]:.6f}"
        )

    max_abs_arr = np.asarray(max_abs_values, dtype=np.float64)
    mean_abs_arr = np.asarray(mean_abs_values, dtype=np.float64)
    rmse_arr = np.asarray(rmse_values, dtype=np.float64)
    p99_abs_arr = np.asarray(p99_abs_values, dtype=np.float64)
    argmax_match_arr = np.asarray(argmax_match_values, dtype=np.float64)

    print("summary:")
    print(f"  sample_max_of_max_abs={float(max_abs_arr.max()):.8f}")
    print(f"  sample_mean_of_mean_abs={float(mean_abs_arr.mean()):.8f}")
    print(f"  sample_mean_of_rmse={float(rmse_arr.mean()):.8f}")
    print(f"  sample_mean_of_p99_abs={float(p99_abs_arr.mean()):.8f}")
    print(f"  sample_mean_argmax_match={float(argmax_match_arr.mean()):.6f}")


if __name__ == "__main__":
    main()
