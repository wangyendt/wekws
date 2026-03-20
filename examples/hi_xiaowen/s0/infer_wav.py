#!/usr/bin/env python3

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml
from wenet.text.char_tokenizer import CharTokenizer

from torch2lite import fbank_pybind
from wekws.model.kws_model import init_model
from wekws.model.loss import ctc_prefix_beam_search
from wekws.utils.checkpoint import load_checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_KEYWORDS = "嗨小问,你好问问"
DEFAULT_MODEL_ALIAS = "s3"

# 保持 stdout 只输出结果 JSON，避免 torchaudio 警告污染脚本输出。
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


MODEL_ALIASES: Dict[str, Dict[str, object]] = {
    "top20": {
        "checkpoint": "exp/fsmn_ctc_top20_weight_surgery/79.pt",
        "dict_dir": "dict_top20",
        "stats_subdir": "test_79",
    },
    "distill199": {
        "checkpoint": "exp/fsmn_ctc_distill_mini_align_20_test2/229.pt",
        "dict_dir": "dict_top20",
        "stats_subdir": "test_229",
    },
    "distill199_int8_pt": {
        "checkpoint": "exp/fsmn_ctc_distill_mini_align_20_test2/229_int8.pt",
        "dict_dir": "dict_top20",
        "stats_subdir": "test_229_int8",
    },
    "distill199_int8_zip": {
        "checkpoint": "exp/fsmn_ctc_distill_mini_align_20_test2/229_int8.zip",
        "dict_dir": "dict_top20",
        "stats_subdir": "test_229_int8",
    },
    "v2": {
        "checkpoint": "exp/fsmn_ctc_distill_v2_a64_p32_l2/299.pt",
        "dict_dir": "dict_top20",
        "stats_subdir": "test_299",
    },
    "v2_ft": {
        "checkpoint": "exp/fsmn_ctc_distill_v2_a64_p32_l2_ft_from199_lr1e4/299.pt",
        "dict_dir": "dict_top20",
        "stats_subdir": "test_299",
    },
    "s1": {
        "checkpoint": "exp/fsmn_ctc_distill_s1_a64_p32_l3_merged/399.pt",
        "dict_dir": "dict_top20",
        "stats_subdir": "test_399",
    },
    "s2": {
        "checkpoint": "exp/fsmn_ctc_distill_s2_a56_p28_l3_merged/399.pt",
        "dict_dir": "dict_top20",
        "stats_subdir": "test_399",
    },
    "s3": {
        "checkpoint": "exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/399.pt",
        "dict_dir": "dict_top20",
        "stats_subdir": "test_399",
    },
    "s4": {
        "checkpoint": "exp/fsmn_ctc_distill_s4_a40_p20_l3_merged/399.pt",
        "dict_dir": "dict_top20",
        "stats_subdir": "test_399",
    },
    "s5": {
        "checkpoint": "exp/fsmn_ctc_distill_s5_a32_p16_l3_merged/399.pt",
        "dict_dir": "dict_top20",
        "stats_subdir": "test_399",
    },
    "top440": {
        "checkpoint": "exp/fsmn_ctc_top440_weight_surgery/79.pt",
        "dict_dir": "dict_top440",
        "stats_subdir": "test_79",
    },
}


# 只有在 stats 文件缺失时才使用这些回退阈值。
FALLBACK_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "s1": {"你好问问": 0.000, "嗨小问": 0.338},
    "s2": {"你好问问": 0.000, "嗨小问": 0.417},
    "s3": {"你好问问": 0.016, "嗨小问": 0.272},
    "s4": {"你好问问": 0.000, "嗨小问": 0.423},
    "s5": {"你好问问": 0.000, "嗨小问": 0.520},
}


@dataclass
class StatRow:
    threshold: float
    fa_per_hour: float
    frr: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="对单个 wav 做完整离线唤醒推理，默认以 S3 模型为例。"
    )
    parser.add_argument("--wav", required=True, help="输入 wav 路径")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ALIAS,
        help="模型别名，例如 s3 / s1 / v2 / v2_ft / top20 / distill199",
    )
    parser.add_argument("--checkpoint", default="", help="显式指定 checkpoint/.zip 路径")
    parser.add_argument("--model_dir", default="", help="显式指定实验目录，例如 exp/xxx")
    parser.add_argument(
        "--checkpoint_name",
        default="",
        help="配合 --model_dir 使用，例如 399.pt / avg_30.pt / final.pt",
    )
    parser.add_argument("--config", default="", help="显式指定 config.yaml")
    parser.add_argument("--dict_dir", default="", help="显式指定 dict 目录")
    parser.add_argument("--stats_dir", default="", help="显式指定 stats 所在目录")
    parser.add_argument(
        "--keywords",
        default=DEFAULT_KEYWORDS,
        help="关键词列表，逗号分隔，例如 嗨小问,你好问问",
    )
    parser.add_argument(
        "--threshold_map",
        default="",
        help='手动阈值覆盖，例如 "嗨小问=0.272,你好问问=0.016"',
    )
    parser.add_argument(
        "--target_fa_per_hour",
        type=float,
        default=1.0,
        help="从 stats 文件挑选阈值时优先满足的 FA/h 上限",
    )
    parser.add_argument(
        "--pick_mode",
        choices=["legacy", "recall", "robust"],
        default="legacy",
        help="阈值挑选策略，默认与现有 analyze_exp_test_stats.py 一致",
    )
    parser.add_argument(
        "--frr_eps",
        type=float,
        default=0.001,
        help="robust 模式下 FRR 容差",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU id，默认 -1 表示 CPU；.zip 默认强制走 CPU",
    )
    parser.add_argument(
        "--disable_threshold",
        action="store_true",
        help="只输出原始候选结果，不做最终阈值判定",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON 输出缩进空格数",
    )
    return parser.parse_args()


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def try_fix_mojibake(text: str) -> str:
    if not text or has_cjk(text):
        return text
    try:
        fixed = text.encode("latin1").decode("utf-8")
        if has_cjk(fixed):
            return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text


def parse_keywords_arg(raw_keywords: str) -> List[str]:
    if raw_keywords is None:
        return []
    text = raw_keywords.strip()
    if not text:
        return []
    if "\\u" in text or "\\U" in text or "\\x" in text:
        try:
            text = text.encode("utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            pass
    text = try_fix_mojibake(text)
    return [item.strip() for item in text.replace(" ", "").split(",") if item.strip()]


def parse_threshold_map(raw_text: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if not raw_text.strip():
        return result
    for item in raw_text.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip().replace(" ", "")
        value = value.strip()
        if not key or not value:
            continue
        result[key] = float(value)
    return result


def split_mixed_label(text: str) -> List[str]:
    tokens: List[str] = []
    text = text.strip()
    while text:
        match = re.match(r"[A-Za-z!?,<>_()']+", text)
        token = match.group(0) if match is not None else text[0:1]
        tokens.append(token)
        text = text.replace(token, "", 1).strip(" ")
    return tokens


def space_mixed_label(text: str) -> str:
    return " ".join(split_mixed_label(text))


def to_abs_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def choose_checkpoint_from_model_dir(model_dir: Path, checkpoint_name: str) -> Path:
    if checkpoint_name:
        checkpoint = model_dir / checkpoint_name
        if checkpoint.exists():
            return checkpoint
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint}")

    for candidate in ["avg_30.pt", "final.pt"]:
        checkpoint = model_dir / candidate
        if checkpoint.exists():
            return checkpoint

    numbered = sorted(model_dir.glob("[0-9]*.pt"), key=lambda p: int(p.stem))
    if numbered:
        return numbered[-1]

    raise FileNotFoundError(f"在 {model_dir} 下没有找到可用的 checkpoint")


def infer_default_dict_dir(checkpoint_path: Path) -> str:
    text = str(checkpoint_path).lower()
    if "top440" in text or "mini_440" in text:
        return "dict_top440"
    if "top2598" in text:
        return "dict_top2598"
    if "top2599" in text or "baseline" in text or "2599" in text:
        return "dict"
    return "dict_top20"


def infer_stats_dir(model_dir: Path, checkpoint_path: Path) -> Optional[Path]:
    filename = checkpoint_path.name
    stem = checkpoint_path.stem
    candidates = [model_dir / f"test_{filename}", model_dir / f"test_{stem}"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_model_paths(args) -> Dict[str, Optional[Path]]:
    alias = args.model.strip().lower() if args.model else DEFAULT_MODEL_ALIAS
    alias_info = MODEL_ALIASES.get(alias, {})

    if args.checkpoint:
        checkpoint = to_abs_path(args.checkpoint)
    elif args.model_dir:
        model_dir = to_abs_path(args.model_dir)
        checkpoint = choose_checkpoint_from_model_dir(model_dir, args.checkpoint_name)
    else:
        alias_checkpoint = alias_info.get("checkpoint")
        if not alias_checkpoint:
            raise ValueError(f"未知模型别名: {args.model}")
        checkpoint = to_abs_path(str(alias_checkpoint))

    if not checkpoint.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint}")

    model_dir = checkpoint.parent
    if args.config:
        config = to_abs_path(args.config)
    else:
        config = model_dir / "config.yaml"

    if not config.exists():
        raise FileNotFoundError(f"找不到 config: {config}")

    if args.dict_dir:
        dict_dir = to_abs_path(args.dict_dir)
    else:
        alias_dict = alias_info.get("dict_dir")
        default_dict = str(alias_dict) if alias_dict else infer_default_dict_dir(checkpoint)
        dict_dir = to_abs_path(default_dict)

    if not dict_dir.exists():
        raise FileNotFoundError(f"找不到 dict 目录: {dict_dir}")

    if args.stats_dir:
        stats_dir = to_abs_path(args.stats_dir)
    else:
        stats_dir = None
        alias_stats = alias_info.get("stats_subdir")
        if alias_stats:
            candidate = model_dir / str(alias_stats)
            if candidate.exists():
                stats_dir = candidate
        if stats_dir is None:
            stats_dir = infer_stats_dir(model_dir, checkpoint)

    return {
        "alias": alias,
        "checkpoint": checkpoint,
        "config": config,
        "dict_dir": dict_dir,
        "stats_dir": stats_dir,
    }


def resolve_cmvn_path(configs: Dict) -> None:
    cmvn = configs.get("model", {}).get("cmvn", {})
    cmvn_file = cmvn.get("cmvn_file")
    if not cmvn_file:
        return
    cmvn_path = Path(cmvn_file)
    if cmvn_path.is_absolute():
        return
    configs["model"]["cmvn"]["cmvn_file"] = str((SCRIPT_DIR / cmvn_path).resolve())


def load_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf8") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    resolve_cmvn_path(configs)
    return configs


def load_model(checkpoint_path: Path, configs: Dict, gpu: int):
    suffix = checkpoint_path.suffix.lower()
    if suffix == ".pte":
        raise ValueError("当前脚本暂不支持 ExecuTorch .pte，请先使用 .pt 或 .zip 模型。")
    is_jit = suffix == ".zip"

    if is_jit:
        model = torch.jit.load(str(checkpoint_path))
        device = torch.device("cpu")
    else:
        model = init_model(configs["model"])
        load_checkpoint(model, str(checkpoint_path))
        use_cuda = gpu >= 0 and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(gpu)
            device = torch.device(f"cuda:{gpu}")
        else:
            device = torch.device("cpu")
        model = model.to(device)

    model.eval()
    return model, device, is_jit


def get_time_resolution_sec(configs: Dict) -> float:
    dataset_conf = configs.get("dataset_conf", {})
    fbank_conf = dataset_conf.get("fbank_conf", {})
    frame_shift_ms = float(fbank_conf.get("frame_shift", 10))
    frame_skip = int(dataset_conf.get("frame_skip", 1))
    if frame_skip <= 0:
        frame_skip = 1
    return frame_shift_ms * frame_skip / 1000.0


def load_wav_and_resample(wav_path: Path, target_sr: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(wav_path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    return waveform


def extract_fbank_features(waveform: torch.Tensor, dataset_conf: Dict) -> torch.Tensor:
    fbank_conf = dataset_conf.get("fbank_conf", {})
    sample_rate = int(dataset_conf.get("resample_conf", {}).get("resample_rate", 16000))
    waveform = waveform * (1 << 15)
    feats = fbank_pybind.fbank(
        waveform,
        num_mel_bins=int(fbank_conf.get("num_mel_bins", 80)),
        frame_length=float(fbank_conf.get("frame_length", 25)),
        frame_shift=float(fbank_conf.get("frame_shift", 10)),
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=sample_rate,
    )
    # feats = kaldi.fbank(
    #     waveform,
    #     num_mel_bins=int(fbank_conf.get("num_mel_bins", 80)),
    #     frame_length=float(fbank_conf.get("frame_length", 25)),
    #     frame_shift=float(fbank_conf.get("frame_shift", 10)),
    #     dither=0.0,
    #     energy_floor=0.0,
    #     sample_frequency=sample_rate,
    # )
    return feats


def apply_context_expansion(feats: torch.Tensor, dataset_conf: Dict) -> torch.Tensor:
    if not dataset_conf.get("context_expansion", False):
        return feats

    left = int(dataset_conf.get("context_expansion_conf", {}).get("left", 0))
    right = int(dataset_conf.get("context_expansion_conf", {}).get("right", 0))
    if feats.size(0) <= right:
        raise ValueError("音频太短，无法完成 context expansion")

    num_frames, feat_dim = feats.shape
    ctx_dim = feat_dim * (left + right + 1)
    feats_ctx = torch.zeros(num_frames, ctx_dim, dtype=torch.float32)
    offset = 0
    for lag in range(-left, right + 1):
        feats_ctx[:, offset:offset + feat_dim] = torch.roll(feats, -lag, 0)
        offset += feat_dim

    for frame_index in range(left):
        for copy_index in range(left - frame_index):
            start = copy_index * feat_dim
            end = (copy_index + 1) * feat_dim
            feats_ctx[frame_index, start:end] = feats_ctx[left, :feat_dim]

    return feats_ctx[: num_frames - right]


def apply_frame_skip(feats: torch.Tensor, dataset_conf: Dict) -> torch.Tensor:
    frame_skip = int(dataset_conf.get("frame_skip", 1))
    if frame_skip <= 1:
        return feats
    return feats[::frame_skip, :]


def build_input_features(wav_path: Path, configs: Dict) -> torch.Tensor:
    dataset_conf = configs.get("dataset_conf", {})
    target_sr = int(dataset_conf.get("resample_conf", {}).get("resample_rate", 16000))
    waveform = load_wav_and_resample(wav_path, target_sr)
    feats = extract_fbank_features(waveform, dataset_conf)
    feats = apply_context_expansion(feats, dataset_conf)
    feats = apply_frame_skip(feats, dataset_conf)
    if feats.numel() == 0:
        raise ValueError("特征为空，请检查音频长度和配置")
    return feats.unsqueeze(0)


def run_model_forward(model, feats: torch.Tensor, device: torch.device, is_jit: bool) -> torch.Tensor:
    feats = feats.to(device)
    with torch.no_grad():
        if is_jit:
            empty_cache = torch.zeros(0, 0, 0, dtype=torch.float32, device=device)
            logits, _ = model(feats, empty_cache)
        else:
            logits, _ = model(feats)
    return logits.softmax(2).cpu()[0]


def is_sublist(main_list, check_list):
    if len(main_list) < len(check_list):
        return -1
    if len(main_list) == len(check_list):
        return 0 if main_list == check_list else -1
    for index in range(len(main_list) - len(check_list) + 1):
        if tuple(main_list[index:index + len(check_list)]) == tuple(check_list):
            return index
    return -1


def build_keyword_token_info(keywords: List[str], dict_dir: Path):
    tokenizer = CharTokenizer(
        str(dict_dir / "dict.txt"),
        str(dict_dir / "words.txt"),
        unk="<filler>",
        split_with_space=True,
    )
    keywords_token: Dict[str, Dict[str, Tuple[int, ...]]] = {}
    keywords_idxset = {0}
    for keyword in keywords:
        _, indexes = tokenizer.tokenize(" ".join(list(keyword)))
        indexes = tuple(indexes)
        keywords_token[keyword] = {"token_id": indexes}
        keywords_idxset.update(indexes)
    return keywords_token, keywords_idxset


def decode_keyword_hit_with_token_info(
    probs: torch.Tensor,
    keywords: List[str],
    keywords_token: Dict[str, Dict[str, Tuple[int, ...]]],
    keywords_idxset,
) -> Dict[str, object]:
    utt_len = int(probs.size(0))
    hyps = ctc_prefix_beam_search(probs[:utt_len], utt_len, keywords_idxset)

    hit_keyword = None
    hit_score = None
    start_frame = None
    end_frame = None

    for one_hyp in hyps:
        prefix_ids = one_hyp[0]
        prefix_nodes = one_hyp[2]
        for word in keywords:
            label = keywords_token[word]["token_id"]
            offset = is_sublist(prefix_ids, label)
            if offset == -1:
                continue

            score = 1.0
            start_frame = prefix_nodes[offset]["frame"]
            end_frame = prefix_nodes[offset + len(label) - 1]["frame"]
            for index in range(offset, offset + len(label)):
                score *= prefix_nodes[index]["prob"]

            hit_keyword = word
            hit_score = math.sqrt(score)
            break

        if hit_keyword is not None:
            break

    return {
        "candidate_keyword": hit_keyword,
        "candidate_score": hit_score,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }


def decode_keyword_hit(probs: torch.Tensor, keywords: List[str], dict_dir: Path) -> Dict[str, object]:
    keywords_token, keywords_idxset = build_keyword_token_info(keywords, dict_dir)
    return decode_keyword_hit_with_token_info(
        probs=probs,
        keywords=keywords,
        keywords_token=keywords_token,
        keywords_idxset=keywords_idxset,
    )


def parse_stats_file(stats_file: Path) -> List[StatRow]:
    rows: List[StatRow] = []
    with open(stats_file, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            rows.append(
                StatRow(
                    threshold=float(parts[0]),
                    fa_per_hour=float(parts[1]),
                    frr=float(parts[2]),
                )
            )
    return rows


def pick_best_row(rows: List[StatRow], target_fa: float, pick_mode: str, frr_eps: float) -> Optional[StatRow]:
    if not rows:
        return None
    candidates = [row for row in rows if row.fa_per_hour <= target_fa]

    if pick_mode == "legacy":
        if candidates:
            return min(candidates, key=lambda row: (row.frr, row.fa_per_hour, row.threshold))
        return min(rows, key=lambda row: (row.frr, row.fa_per_hour, row.threshold))

    if pick_mode == "recall":
        if candidates:
            return min(candidates, key=lambda row: (row.frr, target_fa - row.fa_per_hour, row.threshold))
        return min(rows, key=lambda row: (row.frr, row.fa_per_hour, row.threshold))

    if pick_mode == "robust":
        if candidates:
            best_frr = min(row.frr for row in candidates)
            robust_rows = [row for row in candidates if row.frr <= best_frr + frr_eps]
            return max(robust_rows, key=lambda row: (row.threshold, row.fa_per_hour))
        return min(rows, key=lambda row: (row.frr, row.fa_per_hour, row.threshold))

    raise ValueError(f"不支持的 pick_mode: {pick_mode}")


def get_stats_file_for_keyword(stats_dir: Path, keyword: str) -> Path:
    stats_name = f"stats.{space_mixed_label(keyword).replace(' ', '_')}.txt"
    return stats_dir / stats_name


def load_threshold_map(args, model_info: Dict[str, Optional[Path]], keywords: List[str]) -> Dict[str, Optional[float]]:
    override_map = parse_threshold_map(args.threshold_map)
    alias = model_info["alias"] or ""
    fallback_map = FALLBACK_THRESHOLDS.get(alias, {})
    stats_dir = model_info.get("stats_dir")
    threshold_map: Dict[str, Optional[float]] = {}

    for keyword in keywords:
        keyword_key = keyword.replace(" ", "")
        if keyword_key in override_map:
            threshold_map[keyword] = override_map[keyword_key]
            continue

        threshold = None
        if stats_dir is not None:
            stats_file = get_stats_file_for_keyword(stats_dir, keyword)
            if stats_file.exists():
                rows = parse_stats_file(stats_file)
                best_row = pick_best_row(rows, args.target_fa_per_hour, args.pick_mode, args.frr_eps)
                if best_row is not None:
                    threshold = best_row.threshold

        if threshold is None:
            threshold = fallback_map.get(keyword_key)

        threshold_map[keyword] = threshold

    return threshold_map


def format_result(
    wav_path: Path,
    model_info: Dict[str, Optional[Path]],
    threshold_map: Dict[str, Optional[float]],
    decode_result: Dict[str, object],
    time_resolution_sec: float,
    disable_threshold: bool,
) -> Dict[str, object]:
    candidate_keyword = decode_result["candidate_keyword"]
    candidate_score = decode_result["candidate_score"]
    start_frame = decode_result["start_frame"]
    end_frame = decode_result["end_frame"]

    start_time_sec = None if start_frame is None else start_frame * time_resolution_sec
    end_time_sec = None if end_frame is None else end_frame * time_resolution_sec

    triggered = False
    final_keyword = None
    threshold = None
    if candidate_keyword is not None:
        threshold = threshold_map.get(candidate_keyword)
        if disable_threshold:
            triggered = True
            final_keyword = candidate_keyword
        elif threshold is None:
            triggered = False
        else:
            triggered = candidate_score is not None and candidate_score >= threshold
            if triggered:
                final_keyword = candidate_keyword

    return {
        "wav": str(wav_path),
        "model_alias": model_info["alias"],
        "checkpoint": str(model_info["checkpoint"]),
        "config": str(model_info["config"]),
        "dict_dir": str(model_info["dict_dir"]),
        "stats_dir": str(model_info["stats_dir"]) if model_info["stats_dir"] is not None else None,
        "triggered": triggered,
        "keyword": final_keyword,
        "wake_time_sec": end_time_sec if triggered else None,
        "score": candidate_score,
        "threshold": threshold,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_sec": start_time_sec,
        "end_time_sec": end_time_sec,
        "candidate_keyword": candidate_keyword,
        "candidate_score": candidate_score,
        "threshold_map": threshold_map,
        "time_resolution_sec": time_resolution_sec,
    }


def main():
    args = parse_args()
    wav_path = to_abs_path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(f"找不到 wav: {wav_path}")

    keywords = parse_keywords_arg(args.keywords)
    if not keywords:
        raise ValueError("至少需要一个关键词")

    model_info = resolve_model_paths(args)
    configs = load_config(model_info["config"])
    model, device, is_jit = load_model(model_info["checkpoint"], configs, args.gpu)
    feats = build_input_features(wav_path, configs)
    probs = run_model_forward(model, feats, device, is_jit)
    decode_result = decode_keyword_hit(probs, keywords, model_info["dict_dir"])
    threshold_map = load_threshold_map(args, model_info, keywords)
    time_resolution_sec = get_time_resolution_sec(configs)
    result = format_result(
        wav_path=wav_path,
        model_info=model_info,
        threshold_map=threshold_map,
        decode_result=decode_result,
        time_resolution_sec=time_resolution_sec,
        disable_threshold=args.disable_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=args.indent))


if __name__ == "__main__":
    main()
