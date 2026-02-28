#!/usr/bin/env python3
# Copyright 2026 Wayne
#
# Print model parameter dtype distribution for float/quantized models.

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from executorch.exir import schema  # noqa: E402
from executorch.exir._serialize._program import deserialize_pte_binary  # noqa: E402
from wekws.model.kws_model import init_model  # noqa: E402
from wekws.utils.checkpoint import load_checkpoint  # noqa: E402


def _detect_project_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(start_dir)
        cur = parent


PROJECT_ROOT = _detect_project_root(THIS_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print dtype distribution for .pt/.zip/.jit/.pte models.")
    parser.add_argument("--model", required=True, help="model path")
    parser.add_argument("--config",
                        default="",
                        help="config yaml for float checkpoint (.pt)")
    parser.add_argument("--show_details",
                        action="store_true",
                        default=False,
                        help="print per-tensor details")
    parser.add_argument("--include_buffers",
                        action="store_true",
                        default=False,
                        help="for float checkpoints loaded via config, include buffers")
    parser.add_argument("--used_only",
                        action="store_true",
                        default=False,
                        help="for .pte, only count tensors referenced by graph")
    parser.add_argument("--workdir",
                        default=".",
                        help="working directory for relative paths in config/checkpoint")
    return parser.parse_args()


def _dtype_name_torch(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_name_scalar_type(st: schema.ScalarType) -> str:
    mapping = {
        "BYTE": "uint8",
        "CHAR": "int8",
        "SHORT": "int16",
        "INT": "int32",
        "LONG": "int64",
        "HALF": "float16",
        "FLOAT": "float32",
        "DOUBLE": "float64",
        "BOOL": "bool",
        "BFLOAT16": "bfloat16",
        "UINT16": "uint16",
    }
    return mapping.get(st.name, st.name.lower())


def _numel_from_shape(shape: Sequence[int]) -> int:
    if len(shape) == 0:
        return 1
    out = 1
    for x in shape:
        out *= int(x)
    return int(out)


def _print_stats(title: str,
                 items: List[Tuple[str, str, Tuple[int, ...], int]],
                 show_details: bool = False):
    print("=" * 80)
    print(title)
    print("=" * 80)
    if len(items) == 0:
        print("No tensors found.")
        return

    total_tensors = len(items)
    total_numel = sum(i[3] for i in items)
    by_dtype: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tensors": 0, "numel": 0})
    for _, dtype, _, numel in items:
        by_dtype[dtype]["tensors"] += 1
        by_dtype[dtype]["numel"] += int(numel)

    print(f"Total tensors: {total_tensors}")
    print(f"Total numel:   {total_numel}")
    print("-" * 80)
    print(f"{'dtype':12s} {'tensors':>8s} {'numel':>15s} {'numel_ratio':>12s}")
    print("-" * 80)
    for dtype, stat in sorted(by_dtype.items(), key=lambda kv: (-kv[1]["numel"], kv[0])):
        ratio = 100.0 * stat["numel"] / max(total_numel, 1)
        print(f"{dtype:12s} {stat['tensors']:8d} {stat['numel']:15d} {ratio:11.2f}%")

    if show_details:
        print("-" * 80)
        print("Details: name | dtype | shape | numel")
        print("-" * 80)
        for name, dtype, shape, numel in items:
            print(f"{name:60s} | {dtype:10s} | {shape!s:20s} | {numel}")


def _collect_from_model_with_config(model_path: str,
                                    config_path: str,
                                    include_buffers: bool = False):
    with open(config_path, "r", encoding="utf8") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    model = init_model(configs["model"])
    load_checkpoint(model, model_path)
    model.eval().cpu()

    items: List[Tuple[str, str, Tuple[int, ...], int]] = []
    for name, p in model.named_parameters():
        items.append((name, _dtype_name_torch(p.dtype), tuple(int(x) for x in p.shape), int(p.numel())))
    if include_buffers:
        for name, b in model.named_buffers():
            items.append((f"[buffer] {name}",
                          _dtype_name_torch(b.dtype),
                          tuple(int(x) for x in b.shape),
                          int(b.numel())))
    return items


def _find_tensor_dict(obj) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(obj, dict):
        if len(obj) > 0 and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
        for key in ("state_dict", "model_state_dict", "model"):
            maybe = obj.get(key)
            if isinstance(maybe, dict) and len(maybe) > 0 and \
                    all(isinstance(v, torch.Tensor) for v in maybe.values()):
                return maybe
    return None


def _collect_from_pt_or_torchscript(model_path: str):
    items: List[Tuple[str, str, Tuple[int, ...], int]] = []

    # 1) TorchScript
    if model_path.endswith(".zip") or model_path.endswith(".jit"):
        try:
            m = torch.jit.load(model_path, map_location="cpu")
            sd = m.state_dict()
            for name, t in sd.items():
                if torch.is_tensor(t):
                    items.append((name,
                                  _dtype_name_torch(t.dtype),
                                  tuple(int(x) for x in t.shape),
                                  int(t.numel())))
            if len(items) > 0:
                return items
        except Exception:
            pass

    # 2) Generic torch.load
    obj = torch.load(model_path, map_location="cpu")
    if isinstance(obj, torch.nn.Module):
        for name, p in obj.named_parameters():
            items.append((name,
                          _dtype_name_torch(p.dtype),
                          tuple(int(x) for x in p.shape),
                          int(p.numel())))
        if len(items) > 0:
            return items

    sd = _find_tensor_dict(obj)
    if sd is None:
        raise RuntimeError(
            "Unsupported .pt/.zip content: cannot locate parameter tensors. "
            "Try providing --config for float checkpoints.")
    for name, t in sd.items():
        items.append((name,
                      _dtype_name_torch(t.dtype),
                      tuple(int(x) for x in t.shape),
                      int(t.numel())))
    return items


def _collect_used_evalue_indices(ep: schema.ExecutionPlan) -> set:
    used = set(ep.inputs) | set(ep.outputs)
    for chain in ep.chains:
        used.update(chain.inputs)
        used.update(chain.outputs)
        for ins in chain.instructions:
            ia = ins.instr_args
            if hasattr(ia, "args"):
                used.update(getattr(ia, "args"))
            if hasattr(ia, "move_from"):
                used.add(getattr(ia, "move_from"))
            if hasattr(ia, "move_to"):
                used.add(getattr(ia, "move_to"))
            if hasattr(ia, "cond_value_index"):
                used.add(getattr(ia, "cond_value_index"))
    return used


def _collect_from_pte(model_path: str, used_only: bool = False):
    with open(model_path, "rb") as f:
        program = deserialize_pte_binary(f.read())
    ep = program.execution_plan[0]
    used_indices = _collect_used_evalue_indices(ep) if used_only else None

    # Parameter-like tensors in .pte are treated as constant tensors in value table:
    # allocation_info is None and data_buffer_idx > 0.
    items: List[Tuple[str, str, Tuple[int, ...], int]] = []
    all_tensors: List[Tuple[str, str, Tuple[int, ...], int]] = []

    for idx, ev in enumerate(ep.values):
        v = ev.val
        if not isinstance(v, schema.Tensor):
            continue
        if used_only and idx not in used_indices:
            continue
        dtype = _dtype_name_scalar_type(v.scalar_type)
        shape = tuple(int(x) for x in v.sizes)
        numel = _numel_from_shape(shape)
        name = f"value[{idx}]"
        all_tensors.append((name, dtype, shape, numel))
        is_constant_tensor = (v.allocation_info is None and int(v.data_buffer_idx) > 0)
        if is_constant_tensor:
            items.append((name, dtype, shape, numel))

    delegate_ids = [d.id for d in ep.delegates]
    return items, all_tensors, delegate_ids


def main():
    args = parse_args()
    cwd_before = os.getcwd()

    workdir_arg = args.workdir
    if os.path.isabs(workdir_arg):
        resolved_workdir = workdir_arg
    else:
        # Try several bases for convenience.
        cand_cwd = os.path.abspath(os.path.join(cwd_before, workdir_arg))
        cand_project = os.path.abspath(os.path.join(PROJECT_ROOT, workdir_arg))
        cand_repo = os.path.abspath(os.path.join(REPO_ROOT, workdir_arg))
        candidates = []
        for p in (cand_cwd, cand_project, cand_repo):
            if os.path.isdir(p) and p not in candidates:
                candidates.append(p)
        if not candidates:
            raise FileNotFoundError(
                f"workdir not found: {workdir_arg} "
                f"(tried: {cand_cwd}, {cand_project}, {cand_repo})")

        # Heuristic: for multi-level paths like examples/hi_xiaowen/s0, prefer
        # project-root resolution to avoid accidental nesting under current cwd.
        has_sep = (os.sep in workdir_arg) or ("/" in workdir_arg)
        if has_sep and cand_project in candidates:
            resolved_workdir = cand_project
        else:
            resolved_workdir = candidates[0]

    os.chdir(resolved_workdir)
    model_path = args.model
    if not os.path.exists(model_path):
        # Fallback: also try resolving from original cwd if model path was given
        # relative to where the command was launched.
        alt_model = os.path.abspath(os.path.join(cwd_before, model_path))
        if os.path.exists(alt_model):
            model_path = alt_model
        else:
            raise FileNotFoundError(f"model not found: {model_path}")

    config_path = args.config
    if config_path and not os.path.exists(config_path):
        # Keep behavior consistent with model path resolution.
        alt_config = os.path.abspath(os.path.join(cwd_before, config_path))
        if os.path.exists(alt_config):
            config_path = alt_config
        else:
            raise FileNotFoundError(f"config not found: {config_path}")

    ext = os.path.splitext(model_path)[1].lower()
    print(f"Model: {os.path.abspath(model_path)}")
    print(f"Type:  {ext if ext else '<no extension>'}")

    if ext == ".pte":
        const_items, all_items, delegate_ids = _collect_from_pte(
            model_path, used_only=args.used_only)
        print(f"Delegates: {delegate_ids if len(delegate_ids) > 0 else 'none'}")
        _print_stats("PTE Constant Tensors (parameter-like)", const_items, args.show_details)
        _print_stats("PTE All Tensor Values", all_items, False)
        if len(delegate_ids) > 0:
            print("-" * 80)
            print("Note: delegate backends (e.g., xnnpack) may pack weights into delegate blobs;")
            print("those packed weights might not appear as explicit constant tensors above.")
        return

    # .pt / .zip / .jit
    if config_path:
        items = _collect_from_model_with_config(model_path,
                                                config_path,
                                                include_buffers=args.include_buffers)
        _print_stats("Model Parameters (loaded with config)", items, args.show_details)
    else:
        items = _collect_from_pt_or_torchscript(model_path)
        _print_stats("Tensor Entries (auto-loaded)", items, args.show_details)


if __name__ == "__main__":
    main()
