#!/usr/bin/env python3
"""Build a single-audio TFLite alignment dump package.

The package is intended for endpoint/runtime alignment. It follows the same
streaming path used by infer_wav_stream.py:

  audio -> StreamingFbankExtractor -> context expansion -> frame_skip
        -> streaming TFLite(feats, cache) -> logits/probs -> CTC decoder

For every model frame it dumps the input feature/cache, logits, cache output,
all TFLite operator output tensors readable through
experimental_preserve_all_tensors, and Python/C decoder traces.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import flatbuffers  # noqa: F401  # imported so flatbuffer runtime is explicit in dependencies
import numpy as np
import tensorflow as tf
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
S0_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[4]
if str(S0_DIR) not in sys.path:
    sys.path.insert(0, str(S0_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import infer_wav as iw
import infer_wav_stream as iws
from tensorflow.lite.python import schema_py_generated as schema
from torch2lite import fbank_pybind
from torch2lite.ctc_decoder_c_pybind import StreamingCTCDecoderC
from torch2lite.tflite_quant_utils import dequantize_from_detail, quantize_to_detail


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="input wav/mp3 path")
    parser.add_argument("--model", required=True, help="streaming TFLite model")
    parser.add_argument("--config", required=True, help="config.yaml")
    parser.add_argument("--dict_dir", required=True, help="dict directory")
    parser.add_argument("--stats_dir", required=True, help="stats directory for thresholds")
    parser.add_argument("--keywords", default="小雷小雷,小雷快拍")
    parser.add_argument("--output_dir", required=True, help="alignment package directory")
    parser.add_argument("--chunk_ms", type=float, default=300.0)
    parser.add_argument("--target_fa_per_hour", type=float, default=1.0)
    parser.add_argument("--pick_mode", choices=["legacy", "recall", "robust"], default="legacy")
    parser.add_argument("--frr_eps", type=float, default=0.001)
    parser.add_argument("--score_beam_size", type=int, default=3)
    parser.add_argument("--path_beam_size", type=int, default=20)
    parser.add_argument("--min_frames", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=250)
    parser.add_argument("--interval_frames", type=int, default=50)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=0, help="<=0 means all model frames")
    parser.add_argument("--txt_max_elements", type=int, default=200000)
    parser.add_argument("--disable_threshold", action="store_true")
    parser.add_argument("--align_center_context", action="store_true")
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    value = value.strip("._")
    return value or "tensor"


def rel(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def tensor_stats(array) -> dict:
    arr = np.asarray(array)
    flat = arr.reshape(-1)
    if flat.size == 0:
        return {"shape": list(arr.shape), "dtype": str(arr.dtype), "numel": 0}
    work = flat.astype(np.float64)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "numel": int(flat.size),
        "min": float(work.min()),
        "max": float(work.max()),
        "mean": float(work.mean()),
        "std": float(work.std()),
    }


def save_array(base_dir: Path, rel_name: str, array, txt_max_elements: int) -> dict:
    arr = np.asarray(array)
    stem = base_dir / rel_name
    stem.parent.mkdir(parents=True, exist_ok=True)

    npy_path = stem.with_suffix(".npy")
    bin_path = stem.with_suffix(".bin")
    meta_path = stem.with_suffix(".json")
    txt_path = stem.with_suffix(".txt")

    np.save(npy_path, arr)
    np.ascontiguousarray(arr).tofile(bin_path)
    meta = {
        "npy": rel(npy_path, base_dir),
        "bin": rel(bin_path, base_dir),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "stats": tensor_stats(arr),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if arr.size <= txt_max_elements:
        with txt_path.open("w", encoding="utf-8") as fout:
            fout.write(f"shape={list(arr.shape)}\n")
            fout.write(f"dtype={arr.dtype}\n")
            fout.write(np.array2string(arr, threshold=arr.size, max_line_width=1_000_000))
            fout.write("\n")
        meta["txt"] = rel(txt_path, base_dir)
    else:
        meta["txt"] = None
        meta["txt_note"] = f"skip txt because numel={arr.size} > {txt_max_elements}"

    meta["json"] = rel(meta_path, base_dir)
    return meta


def read_id_to_token(dict_dir: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with (dict_dir / "dict.txt").open("r", encoding="utf-8") as fin:
        for line in fin:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            token, raw_idx = parts
            try:
                idx = int(raw_idx)
            except ValueError:
                continue
            mapping[idx] = token
    mapping.setdefault(0, "<blk>")
    return mapping


def tensor_detail_to_json(detail: dict) -> dict:
    quant_params = detail.get("quantization_parameters", {})
    return {
        "index": int(detail["index"]),
        "name": str(detail.get("name", "")),
        "shape": np.asarray(detail.get("shape", [])).astype(int).tolist(),
        "shape_signature": np.asarray(
            detail.get("shape_signature", detail.get("shape", []))
        ).astype(int).tolist(),
        "dtype": str(np.dtype(detail["dtype"])),
        "quantization": [float(detail.get("quantization", (0.0, 0))[0]), int(detail.get("quantization", (0.0, 0))[1])],
        "quantization_parameters": {
            "scales": np.asarray(quant_params.get("scales", [])).astype(np.float64).tolist(),
            "zero_points": np.asarray(quant_params.get("zero_points", [])).astype(np.int64).tolist(),
            "quantized_dimension": int(quant_params.get("quantized_dimension", 0)),
        },
    }


def builtin_name_map() -> Dict[int, str]:
    result = {}
    for name in dir(schema.BuiltinOperator):
        if name.startswith("_"):
            continue
        value = getattr(schema.BuiltinOperator, name)
        if isinstance(value, int):
            result[value] = name
    return result


def parse_tflite_graph(model_path: Path) -> Tuple[List[dict], List[int]]:
    names = builtin_name_map()
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(model_path.read_bytes(), 0))
    if len(model.subgraphs) != 1:
        raise RuntimeError(f"expected one subgraph, got {len(model.subgraphs)}")
    subgraph = model.subgraphs[0]

    operators = []
    output_tensors: List[int] = []
    for op_index, op in enumerate(subgraph.operators):
        opcode = model.operatorCodes[op.opcodeIndex]
        builtin = int(opcode.builtinCode)
        outputs = [int(x) for x in op.outputs]
        output_tensors.extend(outputs)
        operators.append(
            {
                "op_index": op_index,
                "builtin_code": builtin,
                "builtin_name": names.get(builtin, f"BUILTIN_{builtin}"),
                "inputs": [int(x) for x in op.inputs],
                "outputs": outputs,
            }
        )
    return operators, output_tensors


def pick_io_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    feat_input = next((item for item in input_details if len(item["shape"]) == 3), None)
    cache_input = next((item for item in input_details if len(item["shape"]) == 4), None)
    logits_output = next((item for item in output_details if len(item["shape"]) == 3), None)
    cache_output = next((item for item in output_details if len(item["shape"]) == 4), None)
    if feat_input is None or cache_input is None or logits_output is None or cache_output is None:
        raise RuntimeError("failed to identify TFLite streaming IO tensors")
    return feat_input, cache_input, logits_output, cache_output


def build_threshold_args(args):
    class ThresholdArgs:
        pass

    obj = ThresholdArgs()
    obj.threshold_map = ""
    obj.target_fa_per_hour = args.target_fa_per_hour
    obj.pick_mode = args.pick_mode
    obj.frr_eps = args.frr_eps
    return obj


def collect_streaming_features(
    pcm: np.ndarray,
    configs: Dict,
    chunk_ms: float,
    align_center_context: bool,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    dataset_conf = configs["dataset_conf"]
    fbank_conf = dataset_conf["fbank_conf"]
    sample_rate = int(dataset_conf.get("resample_conf", {}).get("resample_rate", 16000))
    frame_length = float(fbank_conf.get("frame_length", 25))
    frame_shift = float(fbank_conf.get("frame_shift", 10))
    num_mel_bins = int(fbank_conf.get("num_mel_bins", 80))
    downsampling = int(dataset_conf.get("frame_skip", 1))
    context_expansion = bool(dataset_conf.get("context_expansion", False))
    left_context = int(dataset_conf.get("context_expansion_conf", {}).get("left", 0))
    right_context = int(dataset_conf.get("context_expansion_conf", {}).get("right", 0))

    extractor = fbank_pybind.StreamingFbankExtractor(
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=sample_rate,
    )

    chunk_samples = max(1, int(chunk_ms / 1000.0 * sample_rate))
    feature_remained = None
    feats_ctx_offset = (left_context % downsampling) if (align_center_context and downsampling > 1) else 0
    raw_chunks: List[np.ndarray] = []
    accepted_chunks: List[np.ndarray] = []
    chunk_rows: List[dict] = []

    for chunk_index, start in enumerate(range(0, pcm.shape[0], chunk_samples)):
        chunk_pcm = pcm[start : start + chunk_samples].astype(np.int16, copy=False)
        if chunk_pcm.size == 0:
            continue
        feats = extractor.accept_int16(torch.from_numpy(chunk_pcm), num_samples=int(chunk_pcm.size))
        raw_feat_count = int(feats.size(0))
        if raw_feat_count > 0:
            raw_chunks.append(feats.detach().cpu().numpy().astype(np.float32, copy=True))

        accepted = torch.zeros((0, int(num_mel_bins * (left_context + right_context + 1))), dtype=torch.float32)
        if raw_feat_count > 0:
            work = feats
            if context_expansion:
                if feature_remained is None:
                    feats_pad = torch.nn.functional.pad(work.T, (left_context, 0), mode="replicate").T
                else:
                    feats_pad = torch.cat((feature_remained, work))

                ctx_frm = feats_pad.shape[0] - (right_context + right_context)
                if ctx_frm <= 0:
                    feature_remained = feats_pad.clone()
                    work = None
                else:
                    ctx_win = left_context + right_context + 1
                    ctx_dim = work.shape[1] * ctx_win
                    feats_ctx = torch.zeros(ctx_frm, ctx_dim, dtype=torch.float32)
                    for index in range(ctx_frm):
                        feats_ctx[index] = torch.cat(tuple(feats_pad[index : index + ctx_win])).unsqueeze(0)
                    remained = left_context + right_context
                    feature_remained = feats_pad[-remained:].clone() if remained > 0 else None
                    work = feats_ctx

            if work is not None:
                if downsampling > 1:
                    last_remainder = 0 if feats_ctx_offset == 0 else downsampling - feats_ctx_offset
                    remainder = (work.size(0) + last_remainder) % downsampling
                    work = work[feats_ctx_offset::downsampling, :]
                    feats_ctx_offset = remainder if remainder == 0 else downsampling - remainder
                accepted = work
                if accepted.size(0) > 0:
                    accepted_chunks.append(accepted.detach().cpu().numpy().astype(np.float32, copy=True))

        chunk_rows.append(
            {
                "chunk_index": chunk_index,
                "sample_range": [int(start), int(min(start + chunk_samples, pcm.shape[0]))],
                "raw_fbank_frames": raw_feat_count,
                "accepted_model_frames": int(accepted.size(0)),
            }
        )

    raw_fbank = (
        np.concatenate(raw_chunks, axis=0).astype(np.float32, copy=False)
        if raw_chunks
        else np.zeros((0, num_mel_bins), dtype=np.float32)
    )
    model_feats = (
        np.concatenate(accepted_chunks, axis=0).astype(np.float32, copy=False)
        if accepted_chunks
        else np.zeros((0, num_mel_bins * (left_context + right_context + 1)), dtype=np.float32)
    )
    return raw_fbank, model_feats, chunk_rows


def top_tokens(prob: np.ndarray, id_to_token: Dict[int, str], topk: int) -> List[dict]:
    indices = np.argsort(prob)[::-1][:topk]
    return [
        {
            "token_id": int(idx),
            "token": id_to_token.get(int(idx), f"<id:{int(idx)}>"),
            "prob": float(prob[idx]),
        }
        for idx in indices
    ]


def format_nodes(nodes, id_to_token: Dict[int, str]) -> List[dict]:
    return [
        {
            "token_id": int(node["token"] if isinstance(node, dict) else node.token),
            "token": id_to_token.get(int(node["token"] if isinstance(node, dict) else node.token), ""),
            "frame": int(node["frame"] if isinstance(node, dict) else node.frame),
            "prob": float(node["prob"] if isinstance(node, dict) else node.prob),
        }
        for node in nodes
    ]


def format_python_hyps(cur_hyps, id_to_token: Dict[int, str], limit: int) -> List[dict]:
    rows = []
    for prefix, (pb, pnb, nodes) in cur_hyps[:limit]:
        rows.append(
            {
                "prefix_ids": [int(x) for x in prefix],
                "prefix_tokens": [id_to_token.get(int(x), f"<id:{int(x)}>") for x in prefix],
                "pb": float(pb),
                "pnb": float(pnb),
                "score": float(pb + pnb),
                "nodes": format_nodes(nodes, id_to_token),
            }
        )
    return rows


def format_c_hyps(hyps, id_to_token: Dict[int, str], limit: int) -> List[dict]:
    rows = []
    for hyp in list(hyps)[:limit]:
        prefix = [int(x) for x in hyp.prefix]
        rows.append(
            {
                "prefix_ids": prefix,
                "prefix_tokens": [id_to_token.get(int(x), f"<id:{int(x)}>") for x in prefix],
                "pb": float(hyp.pb),
                "pnb": float(hyp.pnb),
                "score": float(hyp.pb + hyp.pnb),
                "nodes": format_nodes(hyp.nodes, id_to_token),
            }
        )
    return rows


def python_detect(
    cur_hyps,
    keywords: Sequence[str],
    keywords_token,
    threshold_map: Dict[str, Optional[float]],
    disable_threshold: bool,
    min_frames: int,
    max_frames: int,
    interval_frames: int,
    last_active_pos: int,
):
    for prefix_ids, (_, _, prefix_nodes) in cur_hyps:
        for word in keywords:
            label = keywords_token[word]["token_id"]
            offset = iw.is_sublist(prefix_ids, label)
            if offset == -1:
                continue
            score = 1.0
            start_frame = int(prefix_nodes[offset]["frame"])
            end_frame = int(prefix_nodes[offset + len(label) - 1]["frame"])
            for index in range(offset, offset + len(label)):
                score *= float(prefix_nodes[index]["prob"])
            score = math.sqrt(score)
            threshold = threshold_map.get(word)
            duration = end_frame - start_frame
            passed_threshold = disable_threshold or (threshold is not None and score >= threshold)
            enough_interval = last_active_pos == -1 or (end_frame - last_active_pos >= interval_frames)
            detected = passed_threshold and min_frames <= duration <= max_frames and enough_interval
            return {
                "candidate_keyword": word,
                "candidate_score": score,
                "threshold": threshold,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_frames": int(duration),
                "passed_threshold": bool(passed_threshold),
                "detected": bool(detected),
            }
    return None


def normalize_c_detection(result):
    if result is None:
        return None
    return {
        "candidate_keyword": result.get("keyword"),
        "candidate_score": result.get("candidate_score"),
        "start_frame": result.get("start_frame"),
        "end_frame": result.get("end_frame"),
    }


def update_best(best: dict, candidate: Optional[dict]) -> dict:
    if not candidate or candidate.get("candidate_keyword") is None:
        return best
    score = candidate.get("candidate_score")
    if score is None:
        return best
    if best.get("candidate_score") is None or float(score) > float(best["candidate_score"]):
        return {
            "candidate_keyword": candidate.get("candidate_keyword"),
            "candidate_score": float(score),
            "start_frame": candidate.get("start_frame"),
            "end_frame": candidate.get("end_frame"),
        }
    return best


def model_info_for_thresholds(model_path: Path, config_path: Path, dict_dir: Path, stats_dir: Path):
    return {
        "alias": "xlxl_alignment",
        "checkpoint": model_path,
        "config": config_path,
        "dict_dir": dict_dir,
        "stats_dir": stats_dir,
    }


def dump_tflite_step_tensors(
    interpreter,
    tensor_detail_by_index: Dict[int, dict],
    operators: List[dict],
    step_dir: Path,
    package_dir: Path,
    txt_max_elements: int,
) -> List[dict]:
    entries = []
    for op in operators:
        for output_pos, tensor_index in enumerate(op["outputs"]):
            detail = tensor_detail_by_index.get(tensor_index)
            entry = {
                "op_index": op["op_index"],
                "op_name": op["builtin_name"],
                "output_position": output_pos,
                "tensor_index": tensor_index,
                "tensor_detail": tensor_detail_to_json(detail) if detail is not None else None,
            }
            if detail is None:
                entry["readable"] = False
                entry["error"] = "tensor detail not found"
                entries.append(entry)
                continue
            try:
                raw = interpreter.get_tensor(tensor_index)
                base_name = (
                    f"tensors/op_{op['op_index']:03d}_{sanitize_name(op['builtin_name'])}"
                    f"_out{output_pos}_t{tensor_index:03d}_{sanitize_name(detail['name'])}"
                )
                entry["readable"] = True
                entry["raw"] = save_array(package_dir, f"{rel(step_dir, package_dir)}/{base_name}_raw", raw, txt_max_elements)
                entry["raw_stats"] = tensor_stats(raw)
                entry["dequant"] = save_array(
                    package_dir,
                    f"{rel(step_dir, package_dir)}/{base_name}_dequant",
                    dequantize_from_detail(raw, detail),
                    txt_max_elements,
                )
            except Exception as exc:  # TFLite may keep some tensors unreadable.
                entry["readable"] = False
                entry["error"] = str(exc)
            entries.append(entry)
    return entries


def main():
    args = parse_args()
    package_dir = Path(args.output_dir).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    dict_dir = Path(args.dict_dir).expanduser().resolve()
    stats_dir = Path(args.stats_dir).expanduser().resolve()

    package_dir.mkdir(parents=True, exist_ok=True)
    copied_audio = package_dir / "audio" / f"input{audio_path.suffix.lower()}"
    copied_model = package_dir / "model" / model_path.name
    copied_config = package_dir / "model" / "config.yaml"
    copied_dict = package_dir / "model" / "dict"
    copied_stats = package_dir / "model" / "stats"
    copied_audio.parent.mkdir(parents=True, exist_ok=True)
    copied_model.parent.mkdir(parents=True, exist_ok=True)
    if audio_path != copied_audio:
        shutil.copy2(audio_path, copied_audio)
    shutil.copy2(model_path, copied_model)
    shutil.copy2(config_path, copied_config)
    if copied_dict.exists():
        shutil.rmtree(copied_dict)
    shutil.copytree(dict_dir, copied_dict)
    if copied_stats.exists():
        shutil.rmtree(copied_stats)
    shutil.copytree(stats_dir, copied_stats)

    configs = iw.load_config(config_path)
    dataset_conf = configs["dataset_conf"]
    sample_rate = int(dataset_conf.get("resample_conf", {}).get("resample_rate", 16000))
    waveform = iw.load_wav_and_resample(audio_path, sample_rate)
    waveform_mono = waveform.squeeze(0).contiguous()
    pcm = np.clip(np.round(waveform_mono.numpy() * (1 << 15)), -32768, 32767).astype(np.int16)

    save_array(package_dir, "audio/decoded_waveform_float32", waveform_mono.numpy().astype(np.float32), args.txt_max_elements)
    save_array(package_dir, "audio/decoded_pcm_int16", pcm, args.txt_max_elements)

    offline_raw_fbank = iw.extract_fbank_features(waveform, dataset_conf).detach().cpu()
    offline_model_feats = iw.apply_frame_skip(
        iw.apply_context_expansion(offline_raw_fbank, dataset_conf),
        dataset_conf,
    ).numpy().astype(np.float32)
    streaming_raw_fbank, model_feats, chunk_rows = collect_streaming_features(
        pcm,
        configs,
        args.chunk_ms,
        args.align_center_context,
    )

    save_array(package_dir, "features/offline_fbank_80dim_float32", offline_raw_fbank.numpy().astype(np.float32), args.txt_max_elements)
    save_array(package_dir, "features/offline_model_input_400dim_float32", offline_model_feats, args.txt_max_elements)
    save_array(package_dir, "features/streaming_fbank_80dim_float32", streaming_raw_fbank, args.txt_max_elements)
    save_array(package_dir, "features/model_input_400dim_float32", model_feats, args.txt_max_elements)
    write_json(package_dir / "features/chunk_feature_summary.json", chunk_rows)

    keywords = iw.parse_keywords_arg(args.keywords)
    threshold_map = iw.load_threshold_map(
        build_threshold_args(args),
        model_info_for_thresholds(model_path, config_path, dict_dir, stats_dir),
        keywords,
    )
    keywords_token, keywords_idxset = iw.build_keyword_token_info(keywords, dict_dir)
    id_to_token = read_id_to_token(dict_dir)

    operators, _ = parse_tflite_graph(model_path)
    write_json(package_dir / "model/operator_manifest.json", operators)

    normal = tf.lite.Interpreter(model_path=str(model_path))
    debug = tf.lite.Interpreter(model_path=str(model_path), experimental_preserve_all_tensors=True)
    normal.allocate_tensors()
    debug.allocate_tensors()
    n_feat, n_cache, n_logits, n_out_cache = pick_io_details(normal)
    d_feat, d_cache, d_logits, d_out_cache = pick_io_details(debug)

    tensor_details = [tensor_detail_to_json(item) for item in debug.get_tensor_details()]
    write_json(package_dir / "model/tensor_details.json", tensor_details)
    tensor_detail_by_index = {int(item["index"]): item for item in debug.get_tensor_details()}

    cache_shape = tuple(int(x) for x in n_cache["shape"])
    normal_cache = np.zeros(cache_shape, dtype=np.float32)
    debug_cache = np.zeros(cache_shape, dtype=np.float32)

    max_steps = model_feats.shape[0] if args.max_steps <= 0 else min(model_feats.shape[0], args.max_steps)
    logits_raw_rows = []
    logits_dequant_rows = []
    probs_rows = []
    out_cache_raw_rows = []
    out_cache_dequant_rows = []
    step_summaries = []

    py_cur_hyps = [(tuple(), (1.0, 0.0, []))]
    py_last_active_pos = -1
    py_best = {
        "candidate_keyword": None,
        "candidate_score": None,
        "start_frame": None,
        "end_frame": None,
    }
    py_first_activation = None
    c_first_activation = None
    c_decoder = StreamingCTCDecoderC(
        score_beam_size=args.score_beam_size,
        path_beam_size=args.path_beam_size,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        interval_frames=args.interval_frames,
        frame_step=iw.get_frame_skip(configs),
    )
    c_decoder.set_keywords(keywords_token, keywords_idxset)
    c_decoder.set_thresholds(threshold_map)

    decoder_rows = []
    frame_skip = iw.get_frame_skip(configs)
    time_resolution_sec = iw.get_time_resolution_sec(configs)

    for model_frame in range(max_steps):
        absolute_frame = model_frame * frame_skip
        feat = model_feats[model_frame : model_frame + 1].reshape(1, 1, -1).astype(np.float32)

        n_feat_q = quantize_to_detail(feat, n_feat)
        n_cache_q = quantize_to_detail(normal_cache, n_cache)
        d_feat_q = quantize_to_detail(feat, d_feat)
        d_cache_q = quantize_to_detail(debug_cache, d_cache)

        normal.set_tensor(n_feat["index"], n_feat_q)
        normal.set_tensor(n_cache["index"], n_cache_q)
        normal.invoke()
        normal_logits_raw = normal.get_tensor(n_logits["index"])
        normal_cache_raw = normal.get_tensor(n_out_cache["index"])
        normal_logits = dequantize_from_detail(normal_logits_raw, n_logits)
        normal_cache_next = dequantize_from_detail(normal_cache_raw, n_out_cache)

        debug.set_tensor(d_feat["index"], d_feat_q)
        debug.set_tensor(d_cache["index"], d_cache_q)
        debug.invoke()
        debug_logits_raw = debug.get_tensor(d_logits["index"])
        debug_cache_raw = debug.get_tensor(d_out_cache["index"])
        debug_logits = dequantize_from_detail(debug_logits_raw, d_logits)
        debug_cache_next = dequantize_from_detail(debug_cache_raw, d_out_cache)
        prob = torch.from_numpy(debug_logits).softmax(2)[0, 0].cpu()
        prob_np = prob.numpy().astype(np.float32)

        step_dir = package_dir / "steps" / f"step_{model_frame:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        step_artifacts = {
            "feat_fp32": save_array(package_dir, f"steps/step_{model_frame:04d}/inputs/feat_fp32", feat, args.txt_max_elements),
            "feat_quant_int8": save_array(package_dir, f"steps/step_{model_frame:04d}/inputs/feat_quant_int8", d_feat_q, args.txt_max_elements),
            "cache_in_fp32": save_array(package_dir, f"steps/step_{model_frame:04d}/inputs/cache_in_fp32", debug_cache, args.txt_max_elements),
            "cache_in_quant_int8": save_array(package_dir, f"steps/step_{model_frame:04d}/inputs/cache_in_quant_int8", d_cache_q, args.txt_max_elements),
            "logits_raw_int8": save_array(package_dir, f"steps/step_{model_frame:04d}/outputs/logits_raw_int8", debug_logits_raw, args.txt_max_elements),
            "logits_dequant_float32": save_array(package_dir, f"steps/step_{model_frame:04d}/outputs/logits_dequant_float32", debug_logits, args.txt_max_elements),
            "probs_float32": save_array(package_dir, f"steps/step_{model_frame:04d}/outputs/probs_float32", prob_np.reshape(1, 1, -1), args.txt_max_elements),
            "out_cache_raw_int8": save_array(package_dir, f"steps/step_{model_frame:04d}/outputs/out_cache_raw_int8", debug_cache_raw, args.txt_max_elements),
            "out_cache_dequant_float32": save_array(package_dir, f"steps/step_{model_frame:04d}/outputs/out_cache_dequant_float32", debug_cache_next, args.txt_max_elements),
        }

        tensor_entries = dump_tflite_step_tensors(
            debug,
            tensor_detail_by_index,
            operators,
            step_dir,
            package_dir,
            args.txt_max_elements,
        )
        write_json(step_dir / "tensor_manifest.json", tensor_entries)

        py_cur_hyps = iws.streaming_ctc_prefix_beam_search_step(
            absolute_frame,
            prob,
            py_cur_hyps,
            keywords_idxset,
            args.score_beam_size,
        )[: args.path_beam_size]
        py_detection = python_detect(
            py_cur_hyps,
            keywords,
            keywords_token,
            threshold_map,
            args.disable_threshold,
            args.min_frames,
            args.max_frames,
            args.interval_frames,
            py_last_active_pos,
        )
        py_best = update_best(py_best, py_detection)
        if py_detection and py_detection.get("detected"):
            if py_first_activation is None:
                py_first_activation = dict(py_detection)
            py_last_active_pos = int(py_detection["end_frame"])

        c_detection = normalize_c_detection(c_decoder.step_and_detect_next(prob, args.disable_threshold))
        if c_detection is not None and c_first_activation is None:
            c_first_activation = dict(c_detection)
        c_best = c_decoder.get_best_decode_result()
        decoder_row = {
            "model_frame": model_frame,
            "absolute_frame": absolute_frame,
            "time_sec": absolute_frame * time_resolution_sec,
            "top_tokens": top_tokens(prob_np, id_to_token, args.topk),
            "python_detection": py_detection,
            "python_best_so_far": py_best,
            "python_beam_top": format_python_hyps(py_cur_hyps, id_to_token, min(args.path_beam_size, 8)),
            "c_detection": c_detection,
            "c_best_so_far": c_best,
            "c_beam_top": format_c_hyps(c_decoder.get_hypotheses(), id_to_token, min(args.path_beam_size, 8)),
        }
        decoder_rows.append(decoder_row)

        logits_raw_rows.append(debug_logits_raw.reshape(-1).astype(np.int8))
        logits_dequant_rows.append(debug_logits.reshape(-1).astype(np.float32))
        probs_rows.append(prob_np.reshape(-1))
        out_cache_raw_rows.append(debug_cache_raw.astype(np.int8, copy=True))
        out_cache_dequant_rows.append(debug_cache_next.astype(np.float32, copy=True))

        diff_logits = np.abs(debug_logits.astype(np.float32) - normal_logits.astype(np.float32))
        diff_cache = np.abs(debug_cache_next.astype(np.float32) - normal_cache_next.astype(np.float32))
        step_summaries.append(
            {
                "model_frame": model_frame,
                "absolute_frame": absolute_frame,
                "time_sec": absolute_frame * time_resolution_sec,
                "artifacts": step_artifacts,
                "tensor_manifest": rel(step_dir / "tensor_manifest.json", package_dir),
                "readable_op_output_tensors": int(sum(1 for item in tensor_entries if item.get("readable"))),
                "op_output_tensor_count": len(tensor_entries),
                "debug_vs_normal_logits_max_abs": float(diff_logits.max()) if diff_logits.size else 0.0,
                "debug_vs_normal_cache_max_abs": float(diff_cache.max()) if diff_cache.size else 0.0,
            }
        )

        normal_cache = normal_cache_next.astype(np.float32)
        debug_cache = debug_cache_next.astype(np.float32)

    network_dir = package_dir / "network"
    network_dir.mkdir(parents=True, exist_ok=True)
    save_array(package_dir, "network/logits_raw_int8_by_frame", np.stack(logits_raw_rows, axis=0), args.txt_max_elements)
    save_array(package_dir, "network/logits_dequant_float32_by_frame", np.stack(logits_dequant_rows, axis=0), args.txt_max_elements)
    save_array(package_dir, "network/probs_float32_by_frame", np.stack(probs_rows, axis=0), args.txt_max_elements)
    save_array(package_dir, "network/out_cache_raw_int8_by_frame", np.stack(out_cache_raw_rows, axis=0), args.txt_max_elements)
    save_array(package_dir, "network/out_cache_dequant_float32_by_frame", np.stack(out_cache_dequant_rows, axis=0), args.txt_max_elements)
    write_jsonl(package_dir / "steps/step_summaries.jsonl", step_summaries)

    write_jsonl(package_dir / "decoder/decoder_trace.jsonl", decoder_rows)
    c_best_final = c_decoder.get_best_decode_result()
    py_result = iw.format_result(
        wav_path=audio_path,
        model_info=model_info_for_thresholds(model_path, config_path, dict_dir, stats_dir),
        threshold_map=threshold_map,
        decode_result=py_best,
        time_resolution_sec=time_resolution_sec,
        disable_threshold=args.disable_threshold,
    )
    c_result = iw.format_result(
        wav_path=audio_path,
        model_info=model_info_for_thresholds(model_path, config_path, dict_dir, stats_dir),
        threshold_map=threshold_map,
        decode_result=c_best_final,
        time_resolution_sec=time_resolution_sec,
        disable_threshold=args.disable_threshold,
    )
    final_result = {
        "audio": str(audio_path),
        "copied_audio": rel(copied_audio, package_dir),
        "model": str(model_path),
        "copied_model": rel(copied_model, package_dir),
        "sample_rate": sample_rate,
        "num_samples": int(pcm.size),
        "duration_sec": float(pcm.size / sample_rate),
        "chunk_ms": args.chunk_ms,
        "raw_fbank_shape": list(streaming_raw_fbank.shape),
        "model_input_shape": list(model_feats.shape),
        "dumped_model_frames": int(max_steps),
        "threshold_map": threshold_map,
        "keywords_token": {
            keyword: list(map(int, keywords_token[keyword]["token_id"]))
            for keyword in keywords
        },
        "python_decoder_result": py_result,
        "c_decoder_result": c_result,
        "python_first_activation": py_first_activation,
        "c_first_activation": c_first_activation,
    }
    write_json(package_dir / "decoder/final_result.json", final_result)

    manifest = {
        "package_dir": str(package_dir),
        "created_for": "xiaolei no-float-island TFLite endpoint alignment",
        "audio": rel(copied_audio, package_dir),
        "model": rel(copied_model, package_dir),
        "config": rel(copied_config, package_dir),
        "dict_dir": rel(copied_dict, package_dir),
        "stats_dir": rel(copied_stats, package_dir),
        "main_files": {
            "readme": "README.md",
            "final_result": "decoder/final_result.json",
            "decoder_trace": "decoder/decoder_trace.jsonl",
            "operator_manifest": "model/operator_manifest.json",
            "tensor_details": "model/tensor_details.json",
            "step_summaries": "steps/step_summaries.jsonl",
            "features": "features/",
            "network": "network/",
            "per_step_tensors": "steps/step_*/",
        },
    }
    write_json(package_dir / "manifest.json", manifest)

    readme = f"""# 小雷小雷 TFLite 端侧对齐 Dump 包

## 样本与结论

- 输入音频：`{rel(copied_audio, package_dir)}`
- 模型：`{rel(copied_model, package_dir)}`
- 关键词：`{", ".join(keywords)}`
- 评测口径 best decode：`{c_result.get("keyword")}`，triggered=`{c_result.get("triggered")}`，score=`{c_result.get("score")}`，wake_time_sec=`{c_result.get("wake_time_sec")}`
- C decoder 首次触发：`{c_first_activation}`

## 目录说明

- `audio/`：输入 mp3、解码后的 `decoded_waveform_float32.*` 和 `decoded_pcm_int16.*`。
- `model/`：本次对齐使用的 tflite、config、dict、stats，以及 `operator_manifest.json`、`tensor_details.json`。
- `features/offline_fbank_80dim_float32.*`：整段音频离线 fbank，主要用于参考。
- `features/streaming_fbank_80dim_float32.*`：按 `chunk_ms={args.chunk_ms}` 走 StreamingFbankExtractor 得到的 80 维 fbank。
- `features/model_input_400dim_float32.*`：实际送进 TFLite 的特征，已经完成 left/right context 拼接和 frame_skip。
- `steps/step_XXXX/inputs/`：第 X 个模型帧的 feature/cache 输入，包含 fp32 和 int8 量化值。
- `steps/step_XXXX/outputs/`：第 X 个模型帧的 logits/probs/cache 输出。
- `steps/step_XXXX/tensors/`：第 X 个模型帧每个 TFLite op 输出 tensor，文件名包含 op 序号、op 名和 tensor index。
- `steps/step_XXXX/tensor_manifest.json`：该 step 内每个 op 输出 tensor 的 shape、dtype、量化参数、文件路径和统计值。
- `network/`：按帧汇总后的 logits、probs、out_cache。
- `decoder/decoder_trace.jsonl`：逐帧 beam search trace，包含 top token、Python decoder beam、C pybind decoder beam 和触发结果。
- `decoder/final_result.json`：最终唤醒结果、阈值、关键词 token id、样本时长等摘要。

## 对齐建议

1. 先对齐 `audio/decoded_pcm_int16.bin`。
2. 再对齐 `features/streaming_fbank_80dim_float32.bin` 和 `features/model_input_400dim_float32.bin`。
3. 按 `steps/step_XXXX/inputs/feat_quant_int8.bin`、`cache_in_quant_int8.bin` 对齐模型输入。
4. 按 `steps/step_XXXX/tensor_manifest.json` 中的 op 顺序对齐每层输出，优先看 `_raw.bin` 的 int8 值。
5. 最后对齐 `network/logits_raw_int8_by_frame.bin`、`network/probs_float32_by_frame.bin` 和 `decoder/decoder_trace.jsonl`。

所有 `.bin` 都是 numpy C-order 原始内存；shape/dtype 见同名 `.json`。
"""
    (package_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(final_result, ensure_ascii=False, indent=2))
    print(f"package_dir={package_dir}")


if __name__ == "__main__":
    main()
