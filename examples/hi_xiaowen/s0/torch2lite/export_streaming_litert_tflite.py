#!/usr/bin/env python3

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

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


def resolve_streaming_output_path(checkpoint: Path, output: str, quant_mode: str = "fp32") -> Path:
    if output:
        return Path(output).expanduser().resolve()
    suffix = "fp32" if quant_mode == "fp32" else "int8"
    return checkpoint.with_name(f"{checkpoint.stem}_stream_litert_{suffix}.tflite")


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


def resolve_data_list_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (S0_DIR / path).resolve()


def pad_chunk_feats(feats: torch.Tensor, chunk_frames: int) -> torch.Tensor:
    if feats.size(1) == chunk_frames:
        return feats
    padded = torch.zeros((1, chunk_frames, feats.size(2)), dtype=feats.dtype)
    padded[:, : feats.size(1), :] = feats
    return padded


def load_calibration_items(calib_data_path: Path, num_calib: int, seed: int) -> list[dict]:
    with open(calib_data_path, "r", encoding="utf8") as fin:
        items = [json.loads(line) for line in fin if line.strip()]
    if not items:
        raise ValueError(f"Calibration data is empty: {calib_data_path}")
    if num_calib <= 0 or num_calib >= len(items):
        return items
    rng = random.Random(seed)
    return rng.sample(items, num_calib)


def calibrate_streaming_model(
    prepared_model,
    configs: dict,
    cache_shape: tuple[int, int, int, int],
    calib_data_path: Path,
    num_calib: int,
    seed: int,
    chunk_frames: int,
    progress_every: int,
):
    calib_items = load_calibration_items(calib_data_path, num_calib, seed)
    print(f"calib_data: {calib_data_path}")
    print(f"calib_items: {len(calib_items)}")

    with torch.no_grad():
        for item_index, item in enumerate(calib_items, start=1):
            wav_key = item.get("wav")
            if not wav_key:
                raise KeyError(f"Calibration item missing 'wav' field: {item}")
            wav_path = iw.to_abs_path(wav_key)
            feats = iw.build_input_features(wav_path, configs).cpu()
            cache = torch.zeros(cache_shape, dtype=torch.float32)
            for start in range(0, feats.size(1), chunk_frames):
                chunk = feats[:, start:start + chunk_frames, :]
                chunk = pad_chunk_feats(chunk, chunk_frames)
                _, cache = prepared_model(chunk, cache)
            if (
                progress_every > 0
                and item_index % progress_every == 0
            ) or item_index == len(calib_items):
                print(f"  calibrated {item_index}/{len(calib_items)} utterances")


def get_pt2e_prepare_and_convert():
    try:
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    except ImportError:
        from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
    return prepare_pt2e, convert_pt2e


def maybe_eval(module):
    try:
        module.eval()
    except (AttributeError, NotImplementedError):
        pass
    return module


def build_pt2e_quantizer():
    from litert_torch.quantize.pt2e_quantizer import (
        PT2EQuantizer,
        get_symmetric_quantization_config,
    )

    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=False,
        )
    )
    return quantizer


def build_edge_model(litert_torch, wrapper, sample_args, configs, cache_shape, args):
    if args.quant_mode == "fp32":
        return litert_torch.convert(wrapper, sample_args)

    prepare_pt2e, convert_pt2e = get_pt2e_prepare_and_convert()
    from litert_torch.quantize.quant_config import QuantConfig

    pt2e_quantizer = build_pt2e_quantizer()
    pt2e_model = torch.export.export(wrapper.eval(), sample_args).module()
    pt2e_model = maybe_eval(pt2e_model)
    pt2e_model = prepare_pt2e(pt2e_model, pt2e_quantizer)
    pt2e_model = maybe_eval(pt2e_model)
    calibrate_streaming_model(
        prepared_model=pt2e_model,
        configs=configs,
        cache_shape=cache_shape,
        calib_data_path=resolve_data_list_path(args.calib_data),
        num_calib=args.num_calib,
        seed=args.seed,
        chunk_frames=args.chunk_frames,
        progress_every=args.progress_every,
    )
    pt2e_model = convert_pt2e(pt2e_model, fold_quantize=False)
    pt2e_model = maybe_eval(pt2e_model)
    return litert_torch.convert(
        pt2e_model,
        sample_args,
        quant_config=QuantConfig(pt2e_quantizer=pt2e_quantizer),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export streaming hi_xiaowen KWS checkpoint to TFLite via litert-torch."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="model config yaml")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="checkpoint path")
    parser.add_argument(
        "--output",
        default="",
        help="output .tflite path, default: <checkpoint_dir>/<checkpoint_stem>_stream_litert_<fp32|int8>.tflite",
    )
    parser.add_argument(
        "--quant_mode",
        choices=["fp32", "int8_pt2e"],
        default="fp32",
        help="fp32 keeps the current path; int8_pt2e uses PT2E quantization before LiteRT export",
    )
    parser.add_argument(
        "--chunk_frames",
        type=int,
        default=1,
        help="streaming feature frames per invocation; default 1 gives the safest online step model",
    )
    parser.add_argument(
        "--calib_data",
        default=str(DEFAULT_CALIB_DATA),
        help="calibration data.list used only by --quant_mode int8_pt2e",
    )
    parser.add_argument(
        "--num_calib",
        type=int,
        default=200,
        help="number of utterances used for PT2E calibration; <=0 means use all",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=5,
        help="print calibration progress every N utterances; <=0 disables periodic progress",
    )
    parser.add_argument("--seed", type=int, default=20260323, help="random seed for calibration subsampling")
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
    output_path = resolve_streaming_output_path(checkpoint_path, args.output, args.quant_mode)
    tmp_dir = (
        Path(args.tmp_dir).expanduser().resolve()
        if args.tmp_dir
        else (output_path.parent / ".litert_tmp").resolve()
    )

    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if args.chunk_frames < 1:
        raise ValueError("--chunk_frames must be >= 1")

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
    print(f"quant_mode: {args.quant_mode}")
    print(f"sample_feats_shape: {tuple(sample_feats.shape)}")
    print(f"sample_cache_shape: {tuple(sample_cache.shape)}")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)

    edge_model = build_edge_model(
        litert_torch=litert_torch,
        wrapper=wrapper,
        sample_args=sample_args,
        configs=configs,
        cache_shape=cache_shape,
        args=args,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(output_path))

    print(f"exported_tflite: {output_path}")

    if args.dump_meta:
        meta = {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "output": str(output_path),
            "quant_mode": args.quant_mode,
            "chunk_frames": int(args.chunk_frames),
            "input_dim": input_dim,
            "sample_feats_shape": list(sample_feats.shape),
            "sample_cache_shape": list(sample_cache.shape),
            "calib_data": str(resolve_data_list_path(args.calib_data)),
            "num_calib": int(args.num_calib),
            "progress_every": int(args.progress_every),
            "seed": int(args.seed),
        }
        meta_path = output_path.with_suffix(output_path.suffix + ".json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf8")
        print(f"export_meta: {meta_path}")


if __name__ == "__main__":
    main()
