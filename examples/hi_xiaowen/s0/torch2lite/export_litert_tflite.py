#!/usr/bin/env python3

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import litert_torch
import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
S0_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[4]
DEFAULT_CHECKPOINT = S0_DIR / "exp" / "fsmn_ctc_distill_mini_align_20_test2" / "229.pt"
DEFAULT_CONFIG = S0_DIR / "exp" / "fsmn_ctc_distill_mini_align_20_test2" / "config.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wekws.model.kws_model import init_model
from wekws.utils.checkpoint import load_checkpoint


class KwsLogitsWrapper(torch.nn.Module):
    """Keep only logits for export."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(feats)
        return logits


def fold_cmvn_into_first_affine(model: torch.nn.Module) -> torch.nn.Module:
    export_model = copy.deepcopy(model).cpu().eval()
    global_cmvn = export_model.global_cmvn
    if global_cmvn is None:
        return export_model
    if export_model.preprocessing.__class__.__name__ != "NoSubsampling":
        raise ValueError("当前仅支持在 preprocessing=none 时将 CMVN 折叠进第一层。")
    if not hasattr(export_model.backbone, "in_linear1"):
        raise ValueError("当前模型 backbone 没有 in_linear1，无法折叠 CMVN。")

    affine = export_model.backbone.in_linear1.linear
    weight = affine.weight.detach().clone()
    bias = affine.bias.detach().clone()
    mean = global_cmvn.mean.detach().clone().to(weight.dtype)
    if global_cmvn.norm_var:
        scale = global_cmvn.istd.detach().clone().to(weight.dtype)
    else:
        scale = torch.ones_like(mean)

    folded_weight = weight * scale.unsqueeze(0)
    folded_bias = bias - torch.matmul(folded_weight, mean)

    affine.weight.data.copy_(folded_weight)
    affine.bias.data.copy_(folded_bias)
    export_model.global_cmvn = None
    return export_model


def load_configs(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf8") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    cmvn = configs.get("model", {}).get("cmvn", {})
    cmvn_file = cmvn.get("cmvn_file")
    if cmvn_file:
        cmvn_path = Path(cmvn_file)
        if not cmvn_path.is_absolute():
            configs["model"]["cmvn"]["cmvn_file"] = str((S0_DIR / cmvn_path).resolve())
    return configs


def load_logits_wrapper(config_path: Path, checkpoint_path: Path) -> tuple[dict, torch.nn.Module]:
    configs = load_configs(config_path)
    model = init_model(configs["model"])
    load_checkpoint(model, str(checkpoint_path))
    model.eval()
    model.cpu()
    export_model = fold_cmvn_into_first_affine(model)
    return configs, KwsLogitsWrapper(export_model).eval().cpu()


def resolve_output_path(checkpoint: Path, output: str) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    return checkpoint.with_name(f"{checkpoint.stem}_litert_fp32.tflite")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export hi_xiaowen KWS checkpoint to TFLite via litert-torch."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="model config yaml")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="checkpoint path")
    parser.add_argument(
        "--output",
        default="",
        help="output .tflite path, default: <checkpoint_dir>/<checkpoint_stem>_litert_fp32.tflite",
    )
    parser.add_argument(
        "--export_seq_len",
        type=int,
        default=100,
        help="sequence length used for example input during export",
    )
    parser.add_argument(
        "--dynamic_time_dim",
        action="store_true",
        help="experimental: export T as dynamic dimension; current litert-torch may still fail on this model",
    )
    parser.add_argument(
        "--dynamic_time_min",
        type=int,
        default=1,
        help="minimum allowed T when --dynamic_time_dim is enabled",
    )
    parser.add_argument(
        "--dynamic_time_max",
        type=int,
        default=2048,
        help="maximum allowed T when --dynamic_time_dim is enabled",
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

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = resolve_output_path(checkpoint_path, args.output)
    tmp_dir = (
        Path(args.tmp_dir).expanduser().resolve()
        if args.tmp_dir
        else (output_path.parent / ".litert_tmp").resolve()
    )

    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    configs, wrapper = load_logits_wrapper(config_path, checkpoint_path)

    input_dim = int(configs["model"]["input_dim"])
    sample_input = torch.randn(1, args.export_seq_len, input_dim, dtype=torch.float32)
    sample_args = (sample_input,)
    dynamic_shapes = None
    if args.dynamic_time_dim:
        time_dim = torch.export.Dim(
            "num_frames",
            min=args.dynamic_time_min,
            max=args.dynamic_time_max,
        )
        dynamic_shapes = ({1: time_dim},)

    print(f"config: {config_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"output: {output_path}")
    print(f"tmp_dir: {tmp_dir}")
    print(f"sample_input_shape: {tuple(sample_input.shape)}")
    print(f"dynamic_shapes: {dynamic_shapes}")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)

    edge_model = litert_torch.convert(wrapper, sample_args, dynamic_shapes=dynamic_shapes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(output_path))

    print(f"exported_tflite: {output_path}")

    if args.dump_meta:
        meta = {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "output": str(output_path),
            "export_seq_len": int(args.export_seq_len),
            "input_dim": input_dim,
            "sample_input_shape": list(sample_input.shape),
            "dynamic_time_dim": bool(args.dynamic_time_dim),
            "dynamic_time_min": int(args.dynamic_time_min),
            "dynamic_time_max": int(args.dynamic_time_max),
        }
        meta_path = output_path.with_suffix(output_path.suffix + ".json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf8")
        print(f"export_meta: {meta_path}")


if __name__ == "__main__":
    main()
