#!/usr/bin/env python3
"""Compute analytical MACs/FLOPs for Wekws FSMN CTC models.

Notes:
- This script computes analytical MACs/FLOPs from model config.
- FLOPs are reported as 2 * MACs (multiply + add).
- It focuses on major ops (Linear + depthwise Conv in FSMN blocks).
- Elementwise ops (ReLU/residual add/CMVN) are not counted.
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Any

import yaml


def human(n: float) -> str:
    abs_n = abs(n)
    if abs_n >= 1e12:
        return f"{n / 1e12:.3f} T"
    if abs_n >= 1e9:
        return f"{n / 1e9:.3f} G"
    if abs_n >= 1e6:
        return f"{n / 1e6:.3f} M"
    if abs_n >= 1e3:
        return f"{n / 1e3:.3f} K"
    return f"{n:.3f}"


def load_config_from_checkpoint(ckpt: Path) -> Dict[str, Any]:
    cfg = ckpt.parent / "config.yaml"
    if not cfg.is_file():
        raise FileNotFoundError(f"config.yaml not found next to checkpoint: {cfg}")
    with cfg.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fsmn_macs_per_model_frame(model_conf: Dict[str, Any]) -> int:
    backbone = model_conf["backbone"]
    if backbone.get("type") != "fsmn":
        raise ValueError("Only fsmn backbone is supported by this script.")

    input_dim = int(model_conf["input_dim"])
    output_dim = int(model_conf["output_dim"])

    input_affine_dim = int(backbone["input_affine_dim"])
    num_layers = int(backbone["num_layers"])
    linear_dim = int(backbone["linear_dim"])
    proj_dim = int(backbone["proj_dim"])
    left_order = int(backbone["left_order"])
    right_order = int(backbone["right_order"])
    output_affine_dim = int(backbone["output_affine_dim"])

    macs = input_dim * input_affine_dim + input_affine_dim * linear_dim
    per_layer = (
        linear_dim * proj_dim
        + proj_dim * left_order
        + proj_dim * right_order
        + proj_dim * linear_dim
    )
    macs += num_layers * per_layer
    macs += linear_dim * output_affine_dim + output_affine_dim * output_dim
    return macs


def model_frames_for_seconds(
    seconds: float,
    frame_shift_ms: float,
    skip_frame: int,
    right_context: int,
) -> int:
    raw_frames = max(int(math.floor(seconds * 1000.0 / frame_shift_ms)), 0)
    after_context = max(raw_frames - right_context, 0)
    return int(math.ceil(after_context / skip_frame)) if after_context > 0 else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute analytical FSMN MACs/FLOPs")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint paths, e.g. exp/a/79.pt exp/b/229.pt",
    )
    parser.add_argument(
        "--skip_frame",
        type=int,
        default=3,
        help="Override skip_frame. If <=0, use config dataset_conf.frame_skip.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=1.0,
        help="Report total MACs/FLOPs for this many seconds of audio.",
    )
    args = parser.parse_args()

    for ckpt_str in args.checkpoints:
        ckpt = Path(ckpt_str)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        cfg = load_config_from_checkpoint(ckpt)
        model_conf = cfg["model"]
        data_conf = cfg.get("dataset_conf", {})
        ctx_conf = data_conf.get("context_expansion_conf", {})

        frame_shift_ms = float(data_conf.get("fbank_conf", {}).get("frame_shift", 10))
        right_context = int(ctx_conf.get("right", 0)) if data_conf.get("context_expansion", False) else 0
        cfg_skip = int(data_conf.get("frame_skip", 1))
        used_skip = args.skip_frame if args.skip_frame > 0 else cfg_skip

        macs_per_frame = fsmn_macs_per_model_frame(model_conf)
        flops_per_frame = 2 * macs_per_frame

        frames = model_frames_for_seconds(args.seconds, frame_shift_ms, used_skip, right_context)
        total_macs = macs_per_frame * frames
        total_flops = flops_per_frame * frames

        print("=" * 88)
        print(f"checkpoint         : {ckpt}")
        print(f"config             : {ckpt.parent / 'config.yaml'}")
        print(f"model              : {model_conf['backbone']['type']}")
        print(f"output_dim         : {model_conf['output_dim']}")
        print(f"frame_shift_ms     : {frame_shift_ms}")
        print(f"right_context      : {right_context}")
        print(f"skip_frame(cfg/use): {cfg_skip} / {used_skip}")
        print(f"seconds            : {args.seconds}")
        print(f"model_frames       : {frames}")
        print("-" * 88)
        print(f"MACs / model-frame : {macs_per_frame} ({human(macs_per_frame)})")
        print(f"FLOPs/ model-frame : {flops_per_frame} ({human(flops_per_frame)})")
        print(f"MACs / {args.seconds:g}s       : {total_macs} ({human(total_macs)})")
        print(f"FLOPs/ {args.seconds:g}s       : {total_flops} ({human(total_flops)})")


if __name__ == "__main__":
    main()
