#!/usr/bin/env python3
"""
TFLite Native INT8 PTQ pipeline: PyTorch → ONNX → onnx2tf → TFLite INT8

量化模式: 权重 INT8，输入/输出保持 float32（cache 天然 FP32，避免 LiteRT PT2E 的
cache 量化误差累积问题）。

依赖: pip install onnx onnxruntime onnx2tf
"""

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
S0_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[4]
DEFAULT_CHECKPOINT = S0_DIR / "exp" / "fsmn_ctc_distill_s3_a48_p24_l3_merged" / "399.pt"
DEFAULT_CONFIG = S0_DIR / "exp" / "fsmn_ctc_distill_s3_a48_p24_l3_merged" / "config.yaml"
DEFAULT_CALIB_DATA = S0_DIR / "data" / "train" / "data.list"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(S0_DIR) not in sys.path:
    sys.path.insert(0, str(S0_DIR))

import infer_wav as iw

from export_streaming_litert_tflite import (
    infer_cache_shape,
    load_calibration_items,
    load_streaming_wrapper,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export streaming KWS model to INT8 TFLite via ONNX + onnx2tf."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="model config yaml")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="checkpoint path")
    parser.add_argument(
        "--output",
        default="",
        help="output .tflite path; default: <checkpoint_dir>/<stem>_stream_native_int8.tflite",
    )
    parser.add_argument(
        "--num_calib",
        type=int,
        default=200,
        help="number of utterances for calibration representative dataset",
    )
    parser.add_argument(
        "--calib_data",
        default=str(DEFAULT_CALIB_DATA),
        help="calibration data.list path",
    )
    parser.add_argument("--seed", type=int, default=20260323, help="random seed for calibration subsampling")
    parser.add_argument("--dump_meta", action="store_true", help="dump export metadata JSON")
    parser.add_argument(
        "--onnx_only",
        action="store_true",
        help="only export ONNX, skip TFLite quantization",
    )
    parser.add_argument(
        "--skip_onnx",
        action="store_true",
        help="skip ONNX export, use existing .onnx file for quantization",
    )
    parser.add_argument(
        "--tmp_dir",
        default="",
        help="temp directory for onnx2tf; default: <output_dir>/.onnx2tf_tmp",
    )
    return parser.parse_args()


def resolve_onnx_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.stem}_stream_step.onnx")


def resolve_output_path(checkpoint: Path, output: str, num_calib: int) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    return checkpoint.with_name(f"{checkpoint.stem}_stream_native_int8_calib{num_calib}.tflite")


def export_onnx(
    wrapper: torch.nn.Module,
    cache_shape: tuple,
    onnx_path: Path,
    input_dim: int,
    opset: int = 13,
) -> None:
    """Export streaming step model to ONNX with fixed shapes."""
    print(f"Exporting ONNX: {onnx_path}")
    sample_feats = torch.randn(1, 1, input_dim, dtype=torch.float32)
    sample_cache = torch.zeros(cache_shape, dtype=torch.float32)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (sample_feats, sample_cache),
        str(onnx_path),
        input_names=["feats", "cache"],
        output_names=["logits", "out_cache"],
        opset_version=opset,
        verbose=False,
        do_constant_folding=True,
    )
    print(f"Exported ONNX: {onnx_path} ({onnx_path.stat().st_size} bytes)")


def verify_onnx(
    onnx_path: Path,
    wrapper: torch.nn.Module,
    cache_shape: tuple,
    input_dim: int,
    num_steps: int = 5,
) -> None:
    """Verify ONNX model matches PyTorch using onnxruntime."""
    import onnxruntime as ort

    print(f"Verifying ONNX with onnxruntime ({num_steps} steps)...")
    sess = ort.InferenceSession(str(onnx_path))
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    print(f"  ONNX inputs:  {input_names}")
    print(f"  ONNX outputs: {output_names}")

    torch_cache = torch.zeros(cache_shape, dtype=torch.float32)
    onnx_cache = np.zeros(cache_shape, dtype=np.float32)

    for step in range(num_steps):
        feats = np.random.randn(1, 1, input_dim).astype(np.float32)
        feats_t = torch.from_numpy(feats)

        with torch.no_grad():
            torch_logits, torch_cache = wrapper(feats_t, torch_cache)
            torch_logits = torch_logits.cpu().numpy()
            torch_cache_np = torch_cache.cpu().numpy()

        onnx_outputs = sess.run(None, {"feats": feats, "cache": onnx_cache})
        onnx_logits = onnx_outputs[0]
        onnx_cache = onnx_outputs[1]

        logits_diff = np.abs(onnx_logits.astype(np.float64) - torch_logits.astype(np.float64))
        cache_diff = np.abs(onnx_cache.astype(np.float64) - torch_cache_np.astype(np.float64))
        print(
            f"  step {step}: logits_max_diff={logits_diff.max():.2e}, "
            f"cache_max_diff={cache_diff.max():.2e}"
        )


def prepare_calibration_data(
    wrapper: torch.nn.Module,
    configs: dict,
    cache_shape: tuple,
    calib_data_path: Path,
    num_calib: int,
    seed: int,
    tmp_dir: Path,
    input_dim: int,
) -> tuple[Path, Path]:
    """Prepare .npy calibration files for onnx2tf.

    Returns paths to feats.npy and cache.npy.
    """
    calib_items = load_calibration_items(calib_data_path, num_calib, seed)
    print(f"Preparing calibration data: {len(calib_items)} utterances")

    feats_list = []
    cache_list = []

    with torch.no_grad():
        for item_index, item in enumerate(calib_items, start=1):
            wav_key = item.get("wav")
            if not wav_key:
                raise KeyError(f"Calibration item missing 'wav' field: {item}")
            wav_path = iw.to_abs_path(wav_key)
            feats = iw.build_input_features(wav_path, configs).cpu()
            cache = torch.zeros(cache_shape, dtype=torch.float32)

            for start in range(feats.size(1)):
                frame = feats[:, start:start + 1, :]
                _, cache = wrapper(frame, cache)
                feats_list.append(frame.numpy().astype(np.float32))
                cache_list.append(cache.numpy().astype(np.float32))

            if item_index % 50 == 0 or item_index == len(calib_items):
                print(f"  collected {len(feats_list)} steps from {item_index}/{len(calib_items)} utterances")

    feats_array = np.concatenate(feats_list, axis=0)
    cache_array = np.concatenate(cache_list, axis=0)
    print(f"  calibration shapes: feats={feats_array.shape}, cache={cache_array.shape}")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    feats_path = tmp_dir / "feats.npy"
    cache_path = tmp_dir / "cache.npy"
    np.save(str(feats_path), feats_array)
    np.save(str(cache_path), cache_array)
    print(f"  saved: {feats_path}, {cache_path}")
    return feats_path, cache_path


def convert_onnx_to_int8_tflite(
    onnx_path: Path,
    feats_npy: Path,
    cache_npy: Path,
    output_path: Path,
    tmp_dir: Path,
) -> None:
    """Convert ONNX to INT8 TFLite using onnx2tf."""
    import onnx2tf

    print(f"Converting ONNX → TFLite INT8 via onnx2tf...")
    print(f"  onnx: {onnx_path}")
    print(f"  output: {output_path}")
    print(f"  tmp_dir: {tmp_dir}")

    work_dir = tmp_dir / "onnx2tf_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(work_dir),
        output_signaturedefs=True,
        output_integer_quantized_tflite=True,
        custom_input_op_name_np_data_path=[
            ["feats", str(feats_npy), 0.0, 1.0],
            ["cache", str(cache_npy), 0.0, 1.0],
        ],
        output_weights=False,
        copy_onnx_input_output_names_to_tflite=True,
        keep_shape_absolutely_input_names=["feats", "cache"],
        non_verbose=True,
        disable_model_save=False,
    )

    integer_quant_path = work_dir / "399_stream_step_integer_quant.tflite"
    if not integer_quant_path.exists():
        # Fallback: try the default name from older onnx2tf versions
        integer_quant_path = work_dir / "model_integer_quant.tflite"
    if not integer_quant_path.exists():
        raise FileNotFoundError(
            f"onnx2tf did not produce model_integer_quant.tflite in {work_dir}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(integer_quant_path), str(output_path))
    print(f"Exported INT8 TFLite: {output_path} ({output_path.stat().st_size} bytes)")

    # Cleanup work dir
    shutil.rmtree(work_dir, ignore_errors=True)


def main():
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    onnx_path = resolve_onnx_path(checkpoint_path)
    output_path = resolve_output_path(checkpoint_path, args.output, args.num_calib)
    tmp_dir = (
        Path(args.tmp_dir).expanduser().resolve()
        if args.tmp_dir
        else (output_path.parent / ".onnx2tf_tmp").resolve()
    )
    calib_data_path = Path(args.calib_data).expanduser().resolve()

    for p, label in [(config_path, "config"), (checkpoint_path, "checkpoint")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    configs, wrapper = load_streaming_wrapper(config_path, checkpoint_path)
    input_dim = int(configs["model"]["input_dim"])
    cache_shape = infer_cache_shape(configs)

    print(f"config: {config_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"input_dim: {input_dim}")
    print(f"cache_shape: {cache_shape}")
    print(f"onnx_path: {onnx_path}")
    print(f"output_path: {output_path}")
    print(f"tmp_dir: {tmp_dir}")

    # Step 1: ONNX export
    if not args.skip_onnx:
        export_onnx(wrapper, cache_shape, onnx_path, input_dim)
        verify_onnx(onnx_path, wrapper, cache_shape, input_dim)
    else:
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")
        print(f"Using existing ONNX: {onnx_path}")

    if args.onnx_only:
        print("Done (--onnx_only).")
        return

    # Step 2: Prepare calibration data
    feats_npy, cache_npy = prepare_calibration_data(
        wrapper=wrapper,
        configs=configs,
        cache_shape=cache_shape,
        calib_data_path=calib_data_path,
        num_calib=args.num_calib,
        seed=args.seed,
        tmp_dir=tmp_dir,
        input_dim=input_dim,
    )

    # Step 3: ONNX → TFLite INT8 via onnx2tf
    tmp_dir.mkdir(parents=True, exist_ok=True)
    convert_onnx_to_int8_tflite(
        onnx_path=onnx_path,
        feats_npy=feats_npy,
        cache_npy=cache_npy,
        output_path=output_path,
        tmp_dir=tmp_dir,
    )

    # Cleanup calibration .npy files
    for p in [feats_npy, cache_npy]:
        p.unlink(missing_ok=True)

    # Dump meta
    if args.dump_meta:
        meta = {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "onnx": str(onnx_path),
            "output": str(output_path),
            "pipeline": "pytorch_onnx_onnx2tf",
            "quant_mode": "int8_weights_float32_io",
            "input_dim": input_dim,
            "cache_shape": list(cache_shape),
            "calib_data": str(calib_data_path),
            "num_calib": args.num_calib,
            "seed": args.seed,
        }
        meta_path = output_path.with_suffix(output_path.suffix + ".json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf8")
        print(f"export_meta: {meta_path}")


if __name__ == "__main__":
    main()
