#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio

import infer_wav as iw
from wekws.model.loss import ctc_prefix_beam_search


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_KEYWORDS = "嗨小问,你好问问"
DEFAULT_RECORDS_JSON = SCRIPT_DIR / "record_label_webui" / "runtime" / "records.json"

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


def parse_args():
    parser = argparse.ArgumentParser(
        description="诊断单条或最近几条录音的唤醒推理结果，帮助判断是阈值问题还是解码/发音问题。"
    )
    parser.add_argument("--wav", nargs="*", default=[], help="一个或多个 wav 路径")
    parser.add_argument("--model", default="s3", help="模型别名，例如 s3 / top20 / v2")
    parser.add_argument("--checkpoint", default="", help="显式指定 checkpoint/.zip")
    parser.add_argument("--model_dir", default="", help="显式指定实验目录")
    parser.add_argument("--checkpoint_name", default="", help="配合 --model_dir 使用")
    parser.add_argument("--config", default="", help="显式指定 config.yaml")
    parser.add_argument("--dict_dir", default="", help="显式指定 dict 目录")
    parser.add_argument("--stats_dir", default="", help="显式指定 stats 目录")
    parser.add_argument("--keywords", default=DEFAULT_KEYWORDS, help="关键词列表，逗号分隔")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id，传 -1 表示 CPU")
    parser.add_argument("--last_n", type=int, default=5, help="当未显式传 --wav 时，分析最近 N 条 WebUI 录音")
    parser.add_argument("--records_json", default=str(DEFAULT_RECORDS_JSON), help="WebUI records.json 路径")
    parser.add_argument("--beam_size", type=int, default=5, help="输出前 N 个 beam 假设")
    parser.add_argument("--frame_topk", type=int, default=5, help="输出整段平均 top-k token")
    parser.add_argument("--dump_dir", default="", help="可选：把每条 wav 的 report/logits/probs 落盘到这个目录")
    parser.add_argument("--indent", type=int, default=2, help="JSON 输出缩进")
    return parser.parse_args()


def build_model_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        checkpoint=args.checkpoint,
        model_dir=args.model_dir,
        checkpoint_name=args.checkpoint_name,
        config=args.config if hasattr(args, "config") else "",
        dict_dir=args.dict_dir if hasattr(args, "dict_dir") else "",
        stats_dir=args.stats_dir if hasattr(args, "stats_dir") else "",
        keywords=args.keywords,
        threshold_map="",
        target_fa_per_hour=1.0,
        pick_mode="legacy",
        frr_eps=0.001,
        gpu=args.gpu,
        disable_threshold=False,
        indent=args.indent,
    )


def load_recent_webui_wavs(records_json: Path, last_n: int) -> List[Path]:
    if not records_json.exists():
        raise FileNotFoundError(f"找不到 records.json: {records_json}")

    records = json.loads(records_json.read_text(encoding="utf8"))
    if not isinstance(records, list):
        raise ValueError(f"records.json 不是 list: {records_json}")

    selected = records[-last_n:]
    wavs: List[Path] = []
    app_dir = records_json.parent.parent
    for record in selected:
        rel = record.get("wav_relpath")
        if not rel:
            continue
        wav_path = (app_dir / rel).resolve()
        wavs.append(wav_path)
    return wavs


def load_id2token(dict_dir: Path) -> Dict[int, str]:
    id2tok: Dict[int, str] = {}
    dict_path = dict_dir / "dict.txt"
    with open(dict_path, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            tok = " ".join(parts[:-1])
            try:
                idx = int(parts[-1])
            except ValueError:
                continue
            id2tok[idx] = tok
    return id2tok


def ctc_greedy_decode(token_ids: List[int], blank_id: int = 0) -> List[int]:
    out: List[int] = []
    prev: Optional[int] = None
    for token_id in token_ids:
        if token_id == prev:
            continue
        prev = token_id
        if token_id == blank_id:
            continue
        out.append(token_id)
    return out


def ids_to_text(ids: List[int], id2tok: Dict[int, str]) -> Tuple[str, List[str]]:
    toks = [id2tok.get(i, f"<unk:{i}>") for i in ids]
    return (" ".join(toks)).strip(), toks


def run_model_logits(model, feats: torch.Tensor, device: torch.device, is_jit: bool) -> torch.Tensor:
    feats = feats.to(device)
    with torch.no_grad():
        if is_jit:
            empty_cache = torch.zeros(0, 0, 0, dtype=torch.float32, device=device)
            logits, _ = model(feats, empty_cache)
        else:
            logits, _ = model(feats)
    return logits.detach().cpu()[0]


def get_audio_stats(wav_path: Path) -> Dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = torchaudio.info(str(wav_path))
    wav, sr = torchaudio.load(str(wav_path))
    peak = float(wav.abs().max()) if wav.numel() else 0.0
    rms = float(wav.pow(2).mean().sqrt()) if wav.numel() else 0.0
    duration_sec = wav.shape[1] / sr if sr > 0 else 0.0
    return {
        "sample_rate": info.sample_rate,
        "channels": info.num_channels,
        "bits_per_sample": getattr(info, "bits_per_sample", None),
        "encoding": str(getattr(info, "encoding", "unknown")),
        "num_frames": info.num_frames,
        "duration_sec": round(duration_sec, 3),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
    }


def summarize_frame_topk(probs: torch.Tensor, id2tok: Dict[int, str], topk: int) -> List[Dict]:
    mean_probs = probs.mean(dim=0)
    values, indices = torch.topk(mean_probs, k=min(topk, mean_probs.numel()))
    summary = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        summary.append(
            {
                "token_id": int(idx),
                "token": id2tok.get(int(idx), f"<unk:{idx}>"),
                "mean_prob": float(score),
            }
        )
    return summary


def summarize_token_traces(
    probs: torch.Tensor,
    id2tok: Dict[int, str],
    token_ids: List[int],
    time_resolution_sec: float,
) -> Dict:
    frame_time_sec = [round(index * time_resolution_sec, 3) for index in range(int(probs.size(0)))]
    traces = []
    for token_id in token_ids:
        token_probs = probs[:, token_id]
        peak_prob, peak_frame = torch.max(token_probs, dim=0)
        traces.append(
            {
                "token_id": int(token_id),
                "token": id2tok.get(int(token_id), f"<unk:{token_id}>"),
                "peak_prob": float(peak_prob),
                "peak_frame": int(peak_frame),
                "peak_time_sec": float(int(peak_frame) * time_resolution_sec),
                "probs": [float(value) for value in token_probs.tolist()],
            }
        )
    return {
        "frame_time_sec": frame_time_sec,
        "traces": traces,
    }


def build_visual_token_ids(
    keywords: List[str],
    keywords_token: Dict[str, Dict[str, Tuple[int, ...]]],
) -> List[int]:
    ordered_ids: List[int] = [0, 1]
    seen = {0, 1}
    for keyword in keywords:
        for token_id in keywords_token[keyword]["token_id"]:
            if token_id in seen:
                continue
            ordered_ids.append(int(token_id))
            seen.add(int(token_id))
    return ordered_ids


def summarize_beam_hyps(
    probs: torch.Tensor,
    id2tok: Dict[int, str],
    beam_size: int,
) -> List[Dict]:
    hyps = ctc_prefix_beam_search(
        probs,
        int(probs.size(0)),
        keywords_tokenset=None,
        score_beam_size=max(beam_size, 5),
        path_beam_size=max(beam_size, 20),
        prob_threshold=0.0,
    )
    results = []
    for prefix_ids, path_score, nodes in hyps[:beam_size]:
        token_ids = list(prefix_ids)
        text, toks = ids_to_text(token_ids, id2tok)
        results.append(
            {
                "token_ids": token_ids,
                "tokens": toks,
                "text": text,
                "path_score": float(path_score),
                "frames": [int(node["frame"]) for node in nodes],
            }
        )
    return results


def summarize_keyword_beam_hyps(
    probs: torch.Tensor,
    keywords: List[str],
    keywords_token: Dict[str, Dict[str, Tuple[int, ...]]],
    keywords_idxset,
    id2tok: Dict[int, str],
    beam_size: int,
) -> List[Dict]:
    hyps = ctc_prefix_beam_search(
        probs,
        int(probs.size(0)),
        keywords_tokenset=keywords_idxset,
        score_beam_size=max(beam_size, 5),
        path_beam_size=max(beam_size, 20),
        prob_threshold=0.0,
    )
    results = []
    for prefix_ids, path_score, nodes in hyps[:beam_size]:
        token_ids = list(prefix_ids)
        text, toks = ids_to_text(token_ids, id2tok)
        matched_keyword = None
        matched_range = None
        candidate_score = None
        token_probs = []
        for keyword in keywords:
            label = keywords_token[keyword]["token_id"]
            offset = iw.is_sublist(prefix_ids, label)
            if offset == -1:
                continue
            matched_keyword = keyword
            matched_range = [offset, offset + len(label) - 1]
            score = 1.0
            for idx in range(offset, offset + len(label)):
                prob = float(nodes[idx]["prob"])
                token_probs.append(prob)
                score *= prob
            candidate_score = score ** 0.5
            break
        results.append(
            {
                "token_ids": token_ids,
                "tokens": toks,
                "text": text,
                "path_score": float(path_score),
                "matched_keyword": matched_keyword,
                "matched_range": matched_range,
                "candidate_score": candidate_score,
                "token_probs": token_probs,
                "frames": [int(node["frame"]) for node in nodes],
            }
        )
    return results


def maybe_dump_tensor_artifacts(
    wav_path: Path,
    result: Dict,
    logits: torch.Tensor,
    probs: torch.Tensor,
    dump_dir: Optional[Path],
) -> Optional[str]:
    if dump_dir is None:
        return None

    digest = hashlib.sha1(str(wav_path).encode("utf8")).hexdigest()[:10]
    item_dir = dump_dir / f"{wav_path.stem}_{digest}"
    item_dir.mkdir(parents=True, exist_ok=True)
    torch.save(logits, item_dir / "logits.pt")
    torch.save(probs, item_dir / "probs.pt")
    (item_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf8",
    )
    return str(item_dir)


def diagnose_one_wav(wav_path: Path, args, resources: Dict, id2tok: Dict[int, str]) -> Dict:
    if not wav_path.exists():
        return {"wav": str(wav_path), "error": "file_not_found"}

    audio_stats = get_audio_stats(wav_path)
    feats = iw.build_input_features(wav_path, resources["configs"])
    logits = run_model_logits(resources["model"], feats, resources["device"], resources["is_jit"])
    probs = logits.softmax(dim=-1)

    raw_decode = iw.decode_keyword_hit_with_token_info(
        probs=probs,
        keywords=resources["keywords"],
        keywords_token=resources["keywords_token"],
        keywords_idxset=resources["keywords_idxset"],
    )
    final_result = iw.format_result(
        wav_path=wav_path,
        model_info=resources["model_info"],
        threshold_map=resources["threshold_map"],
        decode_result=raw_decode,
        time_resolution_sec=resources["time_resolution_sec"],
        disable_threshold=False,
    )

    frame_ids = probs.argmax(dim=-1).tolist()
    greedy_ids = ctc_greedy_decode(frame_ids, blank_id=0)
    greedy_text, greedy_tokens = ids_to_text(greedy_ids, id2tok)
    beam_hyps = summarize_beam_hyps(probs, id2tok, args.beam_size)
    keyword_beam_hyps = summarize_keyword_beam_hyps(
        probs,
        resources["keywords"],
        resources["keywords_token"],
        resources["keywords_idxset"],
        id2tok,
        args.beam_size,
    )
    frame_topk = summarize_frame_topk(probs, id2tok, args.frame_topk)
    visual_token_ids = build_visual_token_ids(resources["keywords"], resources["keywords_token"])
    token_traces = summarize_token_traces(
        probs,
        id2tok,
        visual_token_ids,
        resources["time_resolution_sec"],
    )

    result = {
        "wav": str(wav_path),
        "audio_stats": audio_stats,
        "model_alias": resources["model_info"]["alias"],
        "checkpoint": str(resources["model_info"]["checkpoint"]),
        "time_resolution_sec": resources["time_resolution_sec"],
        "threshold_map": resources["threshold_map"],
        "infer_result": final_result,
        "greedy_decode": {
            "token_ids": greedy_ids,
            "tokens": greedy_tokens,
            "text": greedy_text,
        },
        "beam_topk": beam_hyps,
        "keyword_beam_topk": keyword_beam_hyps,
        "mean_frame_topk_tokens": frame_topk,
        "token_traces": token_traces,
    }
    artifact_dir = maybe_dump_tensor_artifacts(
        wav_path,
        result,
        logits,
        probs,
        resources["dump_dir"],
    )
    if artifact_dir is not None:
        result["artifact_dir"] = artifact_dir
    return result


def main():
    args = parse_args()

    wav_paths: List[Path] = [Path(item).expanduser().resolve() for item in args.wav]
    if not wav_paths:
        records_json = Path(args.records_json).expanduser().resolve()
        wav_paths = load_recent_webui_wavs(records_json, args.last_n)
        if not wav_paths:
            raise ValueError("没有找到可诊断的 wav，请传 --wav 或检查 records.json")

    infer_args = build_model_args(args)
    keywords = iw.parse_keywords_arg(args.keywords)
    model_info = iw.resolve_model_paths(infer_args)
    configs = iw.load_config(model_info["config"])
    model, device, is_jit = iw.load_model(model_info["checkpoint"], configs, args.gpu)
    threshold_map = iw.load_threshold_map(infer_args, model_info, keywords)
    time_resolution_sec = iw.get_time_resolution_sec(configs)
    keywords_token, keywords_idxset = iw.build_keyword_token_info(keywords, model_info["dict_dir"])
    id2tok = load_id2token(model_info["dict_dir"])
    dump_dir = Path(args.dump_dir).expanduser().resolve() if args.dump_dir else None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)

    resources = {
        "keywords": keywords,
        "model_info": model_info,
        "configs": configs,
        "model": model,
        "device": device,
        "is_jit": is_jit,
        "threshold_map": threshold_map,
        "time_resolution_sec": time_resolution_sec,
        "keywords_token": keywords_token,
        "keywords_idxset": keywords_idxset,
        "dump_dir": dump_dir,
    }

    for wav_path in wav_paths:
        result = diagnose_one_wav(wav_path, args, resources, id2tok)
        print(json.dumps(result, ensure_ascii=False, indent=args.indent))


if __name__ == "__main__":
    main()
