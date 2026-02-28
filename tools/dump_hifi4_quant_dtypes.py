#!/usr/bin/env python3
# Copyright 2026 Wayne
#
# Dump per-layer dtype report for Cadence/HiFi4 PT2E quantization.

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from executorch.backends.cadence.aot.compiler import quantize_and_export_to_edge
from executorch.backends.cadence.aot.quantizer.quantizer import (  # noqa: E402
    CadenceDefaultQuantizer,
    CadenceWakeWordQuantizer,
)
from wekws.bin.export_executorch_pt2e import (  # noqa: E402
    KwsLogitsWrapper,
    _build_calibration_tuples,
)
from wekws.model.kws_model import init_model  # noqa: E402
from wekws.utils.checkpoint import load_checkpoint  # noqa: E402


@dataclass
class LayerParam:
    name: str
    role: str
    dtype: str
    shape: Tuple[int, ...]
    numel: int


def get_args():
    parser = argparse.ArgumentParser(
        description="Dump per-layer hifi4 quantization dtype report to JSON.")
    parser.add_argument("--config", required=True, help="model config yaml")
    parser.add_argument("--checkpoint", required=True, help="source checkpoint")
    parser.add_argument("--dict", default="dict_top20", help="dict dir for calibration")
    parser.add_argument("--output_json", required=True, help="output json path")
    parser.add_argument("--workdir",
                        default=".",
                        help="working directory for relative dataset/cmvn paths")
    parser.add_argument("--calib_data",
                        default="data/train/data.list",
                        help="calibration data list")
    parser.add_argument("--num_calib", type=int, default=32, help="calibration samples")
    parser.add_argument("--batch_size", type=int, default=1, help="calib batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="calib workers")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--export_seq_len", type=int, default=100, help="export seq len")
    parser.add_argument("--hifi4_quantizer",
                        choices=["wakeword", "default"],
                        default="wakeword",
                        help="Cadence quantizer profile")
    return parser.parse_args()


def _to_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _shape_of(x: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(i) for i in x.shape)


def _collect_original_params(wrapper: torch.nn.Module) -> List[LayerParam]:
    out: List[LayerParam] = []
    for name, param in wrapper.named_parameters():
        if not (name.endswith("weight") or name.endswith("bias")):
            continue
        role = "weight" if name.endswith("weight") else "bias"
        out.append(
            LayerParam(name=name,
                       role=role,
                       dtype=_to_dtype_name(param.dtype),
                       shape=_shape_of(param),
                       numel=int(param.numel())))
    return out


def _collect_quant_placeholders(ep) -> List[Dict[str, Any]]:
    sd = ep.state_dict() if callable(ep.state_dict) else ep.state_dict
    out: List[Dict[str, Any]] = []
    for node in ep.graph.nodes:
        if node.op != "placeholder":
            continue
        node_name = str(node.target)
        vmeta = node.meta.get("val", None)
        v_dtype = _to_dtype_name(vmeta.dtype) if torch.is_tensor(vmeta) else "unknown"
        v_shape = _shape_of(vmeta) if torch.is_tensor(vmeta) else ()
        source_key = node_name[2:] if node_name.startswith("b_") else node_name

        t = sd.get(source_key, None)
        s_dtype = _to_dtype_name(t.dtype) if torch.is_tensor(t) else None
        s_shape = _shape_of(t) if torch.is_tensor(t) else None
        s_numel = int(t.numel()) if torch.is_tensor(t) else None

        if node_name == "feats":
            kind = "input"
        elif "global_cmvn" in node_name:
            kind = "cmvn"
        elif "_frozen_param" in node_name:
            kind = "quant_param"
        else:
            kind = "other_placeholder"

        out.append({
            "node_name": node_name,
            "source_key": source_key,
            "kind": kind,
            "node_meta_dtype": v_dtype,
            "node_meta_shape": list(v_shape),
            "state_dict_dtype": s_dtype,
            "state_dict_shape": list(s_shape) if s_shape is not None else None,
            "state_dict_numel": s_numel,
        })
    return out


def _collect_graph_op_stats(ep) -> Dict[str, Any]:
    op_counter = Counter()
    target_counter = Counter()
    float_compute_targets = Counter()
    float_dtypes = {torch.float16, torch.float32, torch.float64, torch.bfloat16}

    for node in ep.graph.nodes:
        op_counter[node.op] += 1
        if node.op == "call_function":
            target = str(node.target)
            target_counter[target] += 1
            if "quantize" in target or "dequantize" in target:
                continue
            val = node.meta.get("val", None)
            vals = val if isinstance(val, (list, tuple)) else [val]
            has_float = False
            for item in vals:
                if torch.is_tensor(item) and item.dtype in float_dtypes:
                    has_float = True
                    break
            if has_float:
                float_compute_targets[target] += 1

    return {
        "op_histogram": dict(sorted(op_counter.items())),
        "call_function_target_histogram": dict(sorted(target_counter.items())),
        "float_compute_target_histogram": dict(sorted(float_compute_targets.items())),
    }


def _map_layers(original: List[LayerParam],
                quant_placeholders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    quant_params = [p for p in quant_placeholders if p["kind"] == "quant_param"]

    def frozen_idx(name: str) -> int:
        m = re.search(r"_frozen_param(\d+)$", name)
        return int(m.group(1)) if m else 10**9

    quant_params.sort(key=lambda x: frozen_idx(x["node_name"]))

    mapped: List[Dict[str, Any]] = []
    qidx = 0
    for fp in original:
        one = {
            "layer_name": fp.name,
            "role": fp.role,
            "fp_dtype": fp.dtype,
            "fp_shape": list(fp.shape),
            "fp_numel": fp.numel,
            "quant_node": None,
            "quant_dtype": None,
            "quant_shape": None,
            "quant_numel": None,
            "status": "unmapped",
        }
        if qidx < len(quant_params):
            qp = quant_params[qidx]
            qshape = tuple(qp["node_meta_shape"])
            if qshape == fp.shape:
                one["quant_node"] = qp["node_name"]
                one["quant_dtype"] = qp["node_meta_dtype"]
                one["quant_shape"] = qp["node_meta_shape"]
                one["quant_numel"] = qp["state_dict_numel"]
                one["status"] = "mapped_by_order"
                qidx += 1
        mapped.append(one)

    for i in range(qidx, len(quant_params)):
        qp = quant_params[i]
        mapped.append({
            "layer_name": None,
            "role": None,
            "fp_dtype": None,
            "fp_shape": None,
            "fp_numel": None,
            "quant_node": qp["node_name"],
            "quant_dtype": qp["node_meta_dtype"],
            "quant_shape": qp["node_meta_shape"],
            "quant_numel": qp["state_dict_numel"],
            "status": "extra_quant_param",
        })
    return mapped


def _summary_dtypes(entries: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, int]]:
    out = defaultdict(lambda: {"tensors": 0, "numel": 0})
    for e in entries:
        dtype = e.get(key)
        numel = e.get("quant_numel")
        if dtype is None:
            continue
        out[dtype]["tensors"] += 1
        out[dtype]["numel"] += int(numel) if numel is not None else 0
    return dict(sorted(out.items()))


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    os.chdir(args.workdir)
    if not os.path.exists(args.config):
        raise FileNotFoundError("config not found: {}".format(args.config))
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError("checkpoint not found: {}".format(args.checkpoint))

    with open(args.config, "r", encoding="utf8") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    model = init_model(configs["model"])
    load_checkpoint(model, args.checkpoint)
    model.eval().cpu()
    wrapper = KwsLogitsWrapper(model).eval().cpu()
    original = _collect_original_params(wrapper)

    input_dim = int(configs["model"]["input_dim"])
    example_inputs = (
        torch.randn(1, args.export_seq_len, input_dim, dtype=torch.float32),
    )

    calib_args = SimpleNamespace(
        calib_data=args.calib_data,
        num_calib=args.num_calib,
        seed=args.seed,
        dict=args.dict,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        export_seq_len=args.export_seq_len,
        calib_log_interval=0,
    )
    calib_tuples = _build_calibration_tuples(configs, calib_args)
    quantizer = CadenceWakeWordQuantizer() \
        if args.hifi4_quantizer == "wakeword" else CadenceDefaultQuantizer()
    edge_mgr = quantize_and_export_to_edge(wrapper,
                                           example_inputs,
                                           quantizer=quantizer,
                                           calibration_data=calib_tuples)
    ep = edge_mgr.exported_program()

    quant_placeholders = _collect_quant_placeholders(ep)
    graph_stats = _collect_graph_op_stats(ep)
    layer_mapping = _map_layers(original, quant_placeholders)

    original_dtype_summary = defaultdict(lambda: {"tensors": 0, "numel": 0})
    for p in original:
        original_dtype_summary[p.dtype]["tensors"] += 1
        original_dtype_summary[p.dtype]["numel"] += p.numel

    mapped_count = sum(1 for x in layer_mapping if x["status"] == "mapped_by_order")
    unmapped_count = sum(1 for x in layer_mapping if x["status"] == "unmapped")
    report = {
        "meta": {
            "checkpoint": os.path.abspath(args.checkpoint),
            "config": os.path.abspath(args.config),
            "dict_dir": args.dict,
            "workdir": os.path.abspath(args.workdir),
            "quant_backend": "hifi4(cadence)",
            "quantizer": args.hifi4_quantizer,
            "num_calib": args.num_calib,
            "calib_data": args.calib_data,
            "export_seq_len": args.export_seq_len,
        },
        "original_model": {
            "num_weight_bias_tensors": len(original),
            "dtype_summary": dict(sorted(original_dtype_summary.items())),
            "layers": [{
                "layer_name": p.name,
                "role": p.role,
                "dtype": p.dtype,
                "shape": list(p.shape),
                "numel": p.numel,
            } for p in original],
        },
        "quantized_graph": {
            "placeholders": quant_placeholders,
            "graph_stats": graph_stats,
        },
        "layer_mapping": layer_mapping,
        "layer_mapping_summary": {
            "mapped_count": mapped_count,
            "unmapped_count": unmapped_count,
            "quant_dtype_summary": _summary_dtypes(layer_mapping, "quant_dtype"),
        },
    }

    output_dir = os.path.dirname(os.path.abspath(args.output_json))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logging.info("Saved dtype report: %s", args.output_json)
    logging.info("Layer mapping: mapped=%d, unmapped=%d",
                 mapped_count, unmapped_count)


if __name__ == "__main__":
    main()
