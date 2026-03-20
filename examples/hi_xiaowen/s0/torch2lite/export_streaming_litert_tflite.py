#!/usr/bin/env python3

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
S0_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[4]
DEFAULT_CHECKPOINT = S0_DIR / "exp" / "fsmn_ctc_distill_s3_a48_p24_l3_merged" / "399.pt"
DEFAULT_CONFIG = S0_DIR / "exp" / "fsmn_ctc_distill_s3_a48_p24_l3_merged" / "config.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wekws.model.kws_model import init_model
from wekws.utils.checkpoint import load_checkpoint

from export_litert_tflite import fold_cmvn_into_first_affine, load_configs


class KwsStreamingWrapper(torch.nn.Module):
    """Export logits + FSMN cache for streaming inference."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, feats: torch.Tensor, in_cache: torch.Tensor):
        logits, out_cache = self.model(feats, in_cache)
        return logits, out_cache


def load_streaming_wrapper(config_path: Path, checkpoint_path: Path):
    configs = load_configs(config_path)
    model = init_model(configs["model"])
    load_checkpoint(model, str(checkpoint_path))
    model.eval()
    model.cpu()
    export_model = fold_cmvn_into_first_affine(model)
    return configs, KwsStreamingWrapper(export_model).eval().cpu()


def resolve_streaming_output_path(checkpoint: Path, output: str) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    return checkpoint.with_name(f"{checkpoint.stem}_stream_litert_fp32.tflite")


def infer_cache_shape(configs: dict) -> tuple[int, int, int, int]:
    backbone = configs["model"]["backbone"]
    proj_dim = int(backbone["proj_dim"])
    num_layers = int(backbone["num_layers"])
    left_order = int(backbone["left_order"])
    left_stride = int(backbone.get("left_stride", 1))
    right_order = int(backbone["right_order"])
    right_stride = int(backbone.get("right_stride", 1))
    padding = (left_order - 1) * left_stride + right_order * right_stride
    return (1, proj_dim, padding, num_layers)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export streaming hi_xiaowen KWS checkpoint to TFLite via litert-torch."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="model config yaml")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="checkpoint path")
    parser.add_argument(
        "--output",
        default="",
        help="output .tflite path, default: <checkpoint_dir>/<checkpoint_stem>_stream_litert_fp32.tflite",
    )
    parser.add_argument(
        "--chunk_frames",
        type=int,
        default=1,
        help="streaming feature frames per invocation; default 1 gives the safest online step model",
    )
    parser.add_argument(
        "--dump_meta",
        action="store_true",
        help="also dump a small json sidecar with export metadata",
    )
    parser.add_argument(
        "--tmp_dir",
        default="",
        help="temporary directory used by litert/tensorflow during export; default: <output_dir>/.litert_tmp",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import litert_torch
    except ImportError as exc:
        raise RuntimeError(
            "litert_torch is not installed in the current python environment."
        ) from exc

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = resolve_streaming_output_path(checkpoint_path, args.output)
    tmp_dir = (
        Path(args.tmp_dir).expanduser().resolve()
        if args.tmp_dir
        else (output_path.parent / ".litert_tmp").resolve()
    )

    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    configs, wrapper = load_streaming_wrapper(config_path, checkpoint_path)
    input_dim = int(configs["model"]["input_dim"])
    cache_shape = infer_cache_shape(configs)

    sample_feats = torch.randn(1, args.chunk_frames, input_dim, dtype=torch.float32)
    sample_cache = torch.zeros(cache_shape, dtype=torch.float32)
    sample_args = (sample_feats, sample_cache)

    print(f"config: {config_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"output: {output_path}")
    print(f"tmp_dir: {tmp_dir}")
    print(f"sample_feats_shape: {tuple(sample_feats.shape)}")
    print(f"sample_cache_shape: {tuple(sample_cache.shape)}")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)

    edge_model = litert_torch.convert(wrapper, sample_args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(output_path))

    print(f"exported_tflite: {output_path}")

    if args.dump_meta:
        meta = {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "output": str(output_path),
            "chunk_frames": int(args.chunk_frames),
            "input_dim": input_dim,
            "sample_feats_shape": list(sample_feats.shape),
            "sample_cache_shape": list(sample_cache.shape),
        }
        meta_path = output_path.with_suffix(output_path.suffix + ".json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf8")
        print(f"export_meta: {meta_path}")


if __name__ == "__main__":
    main()
