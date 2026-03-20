#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import sys
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torchaudio


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
REPO_ROOT = PROJECT_DIR.parent.parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import infer_wav as iw  # noqa: E402
import diagnose_wav as dw  # noqa: E402
import trace_wav_pipeline as tw  # noqa: E402


st.set_page_config(
    page_title="唤醒词录音标注台",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


RUNTIME_DIR = APP_DIR / "runtime"
RECORDINGS_DIR = RUNTIME_DIR / "recordings"
RECORDS_JSON = RUNTIME_DIR / "records.json"
LABEL_OPTIONS = ["嗨小问", "你好问问", "非唤醒词"]
DEFAULT_KEYWORDS = "嗨小问,你好问问"
DEFAULT_MODEL_ALIAS = "s3"
DEFAULT_DIAGNOSE_BEAM_SIZE = 5
DEFAULT_DIAGNOSE_FRAME_TOPK = 5
DIAGNOSIS_CACHE_VERSION = "v3_keyword_diagnostics"
PIPELINE_CACHE_VERSION = "v1_full_pipeline_visualizer"
MODEL_SUMMARY = {
    "top20": {
        "title": "Top20 权重手术老师模型",
        "description": "基于 top20 词表做权重手术，保留关键输出层能力，是当前蒸馏路线的老师模型。",
        "param_count": "~392,494",
        "metrics": "你好问问 98.63%，嗨小问 98.88%（test_79）",
    },
    "distill199": {
        "title": "199K 蒸馏学生",
        "description": "mini-align 两阶段蒸馏版本，精度高，但参数量和计算量都高于 v2 / S 系列。",
        "param_count": "~199,760",
        "metrics": "你好问问 98.05%，嗨小问 98.13%（test_229）",
    },
    "distill199_int8_pt": {
        "title": "199K INT8 量化模型（pt）",
        "description": "在 199K 蒸馏模型上做 PTQ 量化，精度轻微下降，体积更小。",
        "param_count": "~199,760（INT8 存储）",
        "metrics": "你好问问 97.97%，嗨小问 97.82%（test_229_int8）",
    },
    "distill199_int8_zip": {
        "title": "199K INT8 量化模型（zip）",
        "description": "与 pt 版本同源的量化导出形式，便于不同部署链路使用。",
        "param_count": "~199,760（INT8 存储）",
        "metrics": "你好问问 97.97%，嗨小问 97.82%（test_229_int8）",
    },
    "v2": {
        "title": "v2 轻量蒸馏学生",
        "description": "a64/p32/l2 的 113K 学生结构，是从大模型向超轻量模型过渡的一版。",
        "param_count": "~113,142",
        "metrics": "你好问问 97.30%，嗨小问 96.42%（test_299）",
    },
    "v2_ft": {
        "title": "v2 从 199.pt 继续微调",
        "description": "从 v2 的中间 checkpoint 继续训练，固定较小学习率，整体没有明显优于从头训练。",
        "param_count": "~113,142",
        "metrics": "你好问问 97.30%，嗨小问 96.04%（test_299）",
    },
    "s1": {
        "title": "S1 merged 结构",
        "description": "S 系列里最大的 merged-head 3 层学生，偏向保精度。",
        "param_count": "~96,836",
        "metrics": "S1-S3 都可用；S1 在嗨小问上更强（test_399）",
    },
    "s2": {
        "title": "S2 merged 结构",
        "description": "在 S1 基础上进一步缩宽度，是 S 系列中间档位。",
        "param_count": "~85,484",
        "metrics": "两个关键词均保持 96%+ 可用区间（test_399）",
    },
    "s3": {
        "title": "S3 merged 结构",
        "description": "当前精度/参数折中最好的版本之一，也是网页默认模型。",
        "param_count": "~74,132",
        "metrics": "你好问问 97.66%，嗨小问 96.75%（test_399）",
    },
    "s4": {
        "title": "S4 merged 结构",
        "description": "继续压缩到 40/20，参数更小，但相比 S3 已有明显退化。",
        "param_count": "~62,780",
        "metrics": "较 S3 开始出现可见精度回退（test_399）",
    },
    "s5": {
        "title": "S5 merged 结构",
        "description": "S 系列最小版本，参数最省，但精度退化也最明显。",
        "param_count": "~51,428",
        "metrics": "嗨小问降到约 93.40%，已偏离最佳折中点（test_399）",
    },
    "top440": {
        "title": "Top440 权重手术版本",
        "description": "保留更多 token 的裁剪版本，适合作为更大词表保留方案参考。",
        "param_count": "未整理",
        "metrics": "当前页面未内置该实验摘要",
    },
}


def choose_chinese_font_name() -> Optional[str]:
    candidate_names = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Micro Hei",
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
    ]
    installed_names = {font.name for font in fm.fontManager.ttflist}
    for name in candidate_names:
        if name in installed_names:
            return name
    return None


CHINESE_FONT_NAME = choose_chinese_font_name()
if CHINESE_FONT_NAME:
    plt.rcParams["font.sans-serif"] = [CHINESE_FONT_NAME, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ensure_runtime_dirs() -> None:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if not RECORDS_JSON.exists():
        RECORDS_JSON.write_text("[]\n", encoding="utf8")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59, 130, 246, 0.10), transparent 24%),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.10), transparent 22%),
                linear-gradient(180deg, #f6f8fc 0%, #eef3fb 42%, #e8eef8 100%);
            color: #0f172a;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f5f8fd 100%);
            border-right: 1px solid rgba(15, 23, 42, 0.08);
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.07);
        }
        .hero-card, .panel-card {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(249, 251, 255, 0.98));
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 22px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(16px);
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
            letter-spacing: 0.01em;
            color: #0f172a;
        }
        .hero-subtitle {
            color: #475569;
            line-height: 1.65;
            margin-bottom: 0.8rem;
        }
        .tag-row {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 0.6rem;
        }
        .tag-pill {
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            font-size: 0.86rem;
            font-weight: 700;
            border: 1px solid rgba(148, 163, 184, 0.20);
            background: rgba(241, 245, 249, 0.95);
            color: #1e293b;
        }
        .pred-good {
            background: rgba(16, 185, 129, 0.14);
            color: #047857;
            border-color: rgba(16, 185, 129, 0.24);
        }
        .pred-bad {
            background: rgba(239, 68, 68, 0.12);
            color: #b91c1c;
            border-color: rgba(239, 68, 68, 0.20);
        }
        .pred-pending {
            background: rgba(245, 158, 11, 0.12);
            color: #b45309;
            border-color: rgba(245, 158, 11, 0.20);
        }
        .small-muted {
            color: #64748b;
            font-size: 0.9rem;
        }
        div[data-testid="stAudioInput"] {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 22px;
            padding: 1rem 1rem 0.4rem 1rem;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
        }
        div[data-testid="stAudioInput"] button {
            min-height: 4.2rem;
            font-size: 1.1rem;
            font-weight: 800;
            border-radius: 18px;
        }
        div[data-testid="stAudioInput"] label {
            font-weight: 700;
            color: #0f172a;
        }
        .flow-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            align-items: stretch;
            margin: 0.4rem 0 0.9rem 0;
        }
        .flow-card {
            min-width: 180px;
            flex: 1 1 220px;
            background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(241,245,249,0.96));
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        }
        .flow-card-title {
            font-size: 0.95rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .flow-card-shape {
            font-family: "SFMono-Regular", Consolas, monospace;
            font-size: 0.82rem;
            color: #0f172a;
            background: rgba(226, 232, 240, 0.72);
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            display: inline-block;
            margin-bottom: 0.45rem;
        }
        .flow-card-note {
            color: #475569;
            font-size: 0.84rem;
            line-height: 1.45;
        }
        .flow-card-stats {
            color: #334155;
            font-size: 0.78rem;
            line-height: 1.5;
            margin-top: 0.45rem;
        }
        .flow-arrow {
            align-self: center;
            font-size: 1.25rem;
            color: #94a3b8;
            font-weight: 800;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_records() -> List[Dict]:
    ensure_runtime_dirs()
    try:
        data = json.loads(RECORDS_JSON.read_text(encoding="utf8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def save_records(records: List[Dict]) -> None:
    ensure_runtime_dirs()
    temp_path = RECORDS_JSON.with_suffix(".tmp")
    temp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf8")
    temp_path.replace(RECORDS_JSON)


def delete_record(records: List[Dict], record_id: str) -> List[Dict]:
    remaining: List[Dict] = []
    for record in records:
        if record["id"] != record_id:
            remaining.append(record)
            continue
        wav_path = get_record_path(record)
        try:
            if wav_path.exists():
                wav_path.unlink()
        except OSError:
            pass
    return remaining


def get_record_path(record: Dict) -> Path:
    rel_path = record.get("wav_relpath", "")
    return (APP_DIR / rel_path).resolve()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_predicted_label(result: Dict) -> str:
    keyword = result.get("keyword")
    if isinstance(keyword, str) and keyword.strip():
        return keyword.replace(" ", "")
    return "非唤醒词"


def build_model_args(
    model_alias: str,
    custom_checkpoint: str,
    custom_model_dir: str,
    checkpoint_name: str,
    gpu: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model_alias,
        checkpoint=custom_checkpoint.strip(),
        model_dir=custom_model_dir.strip(),
        checkpoint_name=checkpoint_name.strip(),
        config="",
        dict_dir="",
        stats_dir="",
        keywords=DEFAULT_KEYWORDS,
        threshold_map="",
        target_fa_per_hour=1.0,
        pick_mode="legacy",
        frr_eps=0.001,
        gpu=gpu,
        disable_threshold=False,
        indent=2,
    )


def build_explicit_model_args(
    model_alias: str,
    checkpoint: str,
    config: str,
    dict_dir: str,
    stats_dir: str,
    gpu: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model_alias,
        checkpoint=checkpoint.strip(),
        model_dir="",
        checkpoint_name="",
        config=config.strip(),
        dict_dir=dict_dir.strip(),
        stats_dir=stats_dir.strip(),
        keywords=DEFAULT_KEYWORDS,
        threshold_map="",
        target_fa_per_hour=1.0,
        pick_mode="legacy",
        frr_eps=0.001,
        gpu=gpu,
        disable_threshold=False,
        indent=2,
    )


@st.cache_resource(show_spinner=False)
def load_infer_resources(
    model_alias: str,
    custom_checkpoint: str,
    custom_model_dir: str,
    checkpoint_name: str,
    gpu: int,
):
    infer_args = build_model_args(model_alias, custom_checkpoint, custom_model_dir, checkpoint_name, gpu)
    keywords = iw.parse_keywords_arg(infer_args.keywords)
    model_info = iw.resolve_model_paths(infer_args)
    configs = iw.load_config(model_info["config"])
    model, device, is_jit = iw.load_model(model_info["checkpoint"], configs, gpu)
    threshold_map = iw.load_threshold_map(infer_args, model_info, keywords)
    time_resolution_sec = iw.get_time_resolution_sec(configs)
    keywords_token, keywords_idxset = iw.build_keyword_token_info(keywords, model_info["dict_dir"])
    id2tok = dw.load_id2token(model_info["dict_dir"])
    return {
        "infer_args": infer_args,
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
        "id2tok": id2tok,
        "dump_dir": None,
    }


@st.cache_resource(show_spinner=False)
def load_explicit_infer_resources(
    model_alias: str,
    checkpoint: str,
    config: str,
    dict_dir: str,
    stats_dir: str,
    gpu: int,
):
    infer_args = build_explicit_model_args(model_alias, checkpoint, config, dict_dir, stats_dir, gpu)
    keywords = iw.parse_keywords_arg(infer_args.keywords)
    model_info = iw.resolve_model_paths(infer_args)
    configs = iw.load_config(model_info["config"])
    model, device, is_jit = iw.load_model(model_info["checkpoint"], configs, gpu)
    threshold_map = iw.load_threshold_map(infer_args, model_info, keywords)
    time_resolution_sec = iw.get_time_resolution_sec(configs)
    keywords_token, keywords_idxset = iw.build_keyword_token_info(keywords, model_info["dict_dir"])
    id2tok = dw.load_id2token(model_info["dict_dir"])
    return {
        "infer_args": infer_args,
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
        "id2tok": id2tok,
        "dump_dir": None,
    }


def infer_audio_file(wav_path: Path, resources: Dict) -> Dict:
    feats = iw.build_input_features(wav_path, resources["configs"])
    probs = iw.run_model_forward(resources["model"], feats, resources["device"], resources["is_jit"])
    decode_result = iw.decode_keyword_hit_with_token_info(
        probs=probs,
        keywords=resources["keywords"],
        keywords_token=resources["keywords_token"],
        keywords_idxset=resources["keywords_idxset"],
    )
    result = iw.format_result(
        wav_path=wav_path,
        model_info=resources["model_info"],
        threshold_map=resources["threshold_map"],
        decode_result=decode_result,
        time_resolution_sec=resources["time_resolution_sec"],
        disable_threshold=False,
    )
    result["predicted_label"] = normalize_predicted_label(result)
    return result


def build_model_signature(resources: Dict) -> str:
    checkpoint = resources["model_info"]["checkpoint"]
    alias = resources["model_info"]["alias"]
    return f"{alias}::{checkpoint}"


@st.cache_data(show_spinner=False)
def compute_record_diagnosis(
    wav_path: str,
    model_alias: str,
    checkpoint: str,
    config: str,
    dict_dir: str,
    stats_dir: str,
    gpu: int,
    beam_size: int,
    frame_topk: int,
    cache_version: str,
) -> Dict:
    resources = load_explicit_infer_resources(
        model_alias=model_alias,
        checkpoint=checkpoint,
        config=config,
        dict_dir=dict_dir,
        stats_dir=stats_dir,
        gpu=gpu,
    )
    args = SimpleNamespace(beam_size=beam_size, frame_topk=frame_topk)
    return dw.diagnose_one_wav(Path(wav_path), args, resources, resources["id2tok"])


def compute_record_diagnosis_uncached(
    wav_path: str,
    model_alias: str,
    checkpoint: str,
    config: str,
    dict_dir: str,
    stats_dir: str,
    gpu: int,
    beam_size: int,
    frame_topk: int,
) -> Dict:
    resources = load_explicit_infer_resources(
        model_alias=model_alias,
        checkpoint=checkpoint,
        config=config,
        dict_dir=dict_dir,
        stats_dir=stats_dir,
        gpu=gpu,
    )
    args = SimpleNamespace(beam_size=beam_size, frame_topk=frame_topk)
    return dw.diagnose_one_wav(Path(wav_path), args, resources, resources["id2tok"])


@st.cache_data(show_spinner=False)
def compute_record_pipeline(
    wav_path: str,
    model_alias: str,
    checkpoint: str,
    config: str,
    dict_dir: str,
    stats_dir: str,
    gpu: int,
    beam_size: int,
    cache_version: str,
) -> Dict:
    resources = load_explicit_infer_resources(
        model_alias=model_alias,
        checkpoint=checkpoint,
        config=config,
        dict_dir=dict_dir,
        stats_dir=stats_dir,
        gpu=gpu,
    )
    return tw.inspect_one_wav_pipeline(
        wav_path=Path(wav_path),
        resources=resources,
        id2tok=resources["id2tok"],
        beam_size=beam_size,
    )


def create_record_from_audio(
    audio_bytes: bytes,
    file_suffix: str,
    resources: Dict,
) -> Dict:
    ensure_runtime_dirs()
    digest = hashlib.sha1(audio_bytes).hexdigest()
    record_id = uuid.uuid4().hex
    file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{digest[:8]}{file_suffix}"
    wav_path = RECORDINGS_DIR / file_name
    wav_path.write_bytes(audio_bytes)

    infer_result = infer_audio_file(wav_path, resources)
    record = {
        "id": record_id,
        "created_at": now_str(),
        "audio_sha1": digest,
        "wav_relpath": str(wav_path.relative_to(APP_DIR)),
        "model_signature": build_model_signature(resources),
        "model_alias": resources["model_info"]["alias"],
        "checkpoint": str(resources["model_info"]["checkpoint"]),
        "predicted_label": infer_result["predicted_label"],
        "human_label": None,
        "is_correct": None,
        "infer_result": infer_result,
    }
    return record


def update_record_label(records: List[Dict], record_id: str, human_label: str) -> List[Dict]:
    for record in records:
        if record["id"] != record_id:
            continue
        record["human_label"] = human_label
        record["is_correct"] = (human_label == record["predicted_label"])
        break
    return records


def build_records_df(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        infer_result = record.get("infer_result", {})
        rows.append(
            {
                "id": record["id"],
                "created_at": record.get("created_at", ""),
                "model": record.get("model_alias", ""),
                "predicted": record.get("predicted_label", ""),
                "actual": record.get("human_label") or "待标注",
                "correct": record.get("is_correct"),
                "score": infer_result.get("score"),
                "wake_time_sec": infer_result.get("wake_time_sec"),
            }
        )
    return pd.DataFrame(rows)


def compute_confusion(records: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    labeled = [record for record in records if record.get("human_label") in LABEL_OPTIONS]
    matrix = pd.DataFrame(0, index=LABEL_OPTIONS, columns=LABEL_OPTIONS)

    for record in labeled:
        actual = record["human_label"]
        predicted = record.get("predicted_label", "非唤醒词")
        if predicted not in LABEL_OPTIONS:
            predicted = "非唤醒词"
        matrix.loc[actual, predicted] += 1

    metric_rows = []
    total_correct = 0
    total_count = int(matrix.values.sum())
    for label in LABEL_OPTIONS:
        tp = int(matrix.loc[label, label])
        fp = int(matrix[label].sum() - tp)
        fn = int(matrix.loc[label].sum() - tp)
        support = int(matrix.loc[label].sum())
        predicted_count = int(matrix[label].sum())
        precision = tp / predicted_count if predicted_count > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        total_correct += tp
        metric_rows.append(
            {
                "类别": label,
                "Precision": precision,
                "Recall": recall,
                "Support": support,
                "Predicted": predicted_count,
            }
        )

    stats = {
        "labeled_count": len(labeled),
        "overall_accuracy": (total_correct / total_count) if total_count > 0 else 0.0,
        "macro_precision": float(pd.DataFrame(metric_rows)["Precision"].mean()) if metric_rows else 0.0,
        "macro_recall": float(pd.DataFrame(metric_rows)["Recall"].mean()) if metric_rows else 0.0,
    }
    metrics_df = pd.DataFrame(metric_rows)
    return matrix, metrics_df, stats


def render_hero(resources: Dict, records: List[Dict], active_records: List[Dict]) -> None:
    total = len(records)
    active_total = len(active_records)
    labeled = sum(1 for item in active_records if item.get("human_label") in LABEL_OPTIONS)
    correct = sum(1 for item in active_records if item.get("is_correct") is True)
    pending = active_total - labeled

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">唤醒词录音标注台</div>
            <div class="hero-subtitle">
                录一段音，立刻走后端推理，然后人工确认真实标签。
                页面会根据你已标注的录音自动计算 precision、recall 和混淆矩阵。
            </div>
            <div class="tag-row">
                <span class="tag-pill">当前模型：{resources["model_info"]["alias"]}</span>
                <span class="tag-pill">Checkpoint：{Path(resources["model_info"]["checkpoint"]).name}</span>
                <span class="tag-pill">总录音：{total}</span>
                <span class="tag-pill">当前视图：{active_total}</span>
                <span class="tag-pill">待标注：{pending}</span>
                <span class="tag-pill">已标注：{labeled}</span>
                <span class="tag-pill">标注正确：{correct}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_bool_text(value: object) -> str:
    if value is True:
        return "是"
    if value is False:
        return "否"
    return "-"


def build_model_profile(resources: Dict) -> Dict[str, str]:
    model_info = resources["model_info"]
    configs = resources["configs"]
    dataset_conf = configs.get("dataset_conf", {})
    model_conf = configs.get("model", {})
    backbone_conf = model_conf.get("backbone", {})
    summary = MODEL_SUMMARY.get(model_info["alias"], {})

    return {
        "别名": model_info["alias"] or "-",
        "说明": summary.get("title", "自定义/未登记模型"),
        "描述": summary.get("description", "当前模型没有预置说明，可结合 checkpoint/config 自行确认。"),
        "参数量": summary.get("param_count", "未整理"),
        "历史评测": summary.get("metrics", "暂无预置结果摘要"),
        "Checkpoint": Path(model_info["checkpoint"]).name,
        "词表目录": Path(model_info["dict_dir"]).name,
        "input_affine_dim": str(backbone_conf.get("input_affine_dim", "-")),
        "proj_dim": str(backbone_conf.get("proj_dim", "-")),
        "num_layers": str(backbone_conf.get("num_layers", "-")),
        "linear_dim": str(backbone_conf.get("linear_dim", "-")),
        "merge_head": format_bool_text(backbone_conf.get("merge_head")),
        "output_dim": str(model_conf.get("output_dim", "-")),
        "input_dim": str(model_conf.get("input_dim", "-")),
        "hidden_dim": str(model_conf.get("hidden_dim", "-")),
        "num_mel_bins": str(dataset_conf.get("fbank_conf", {}).get("num_mel_bins", "-")),
        "frame_shift_ms": str(dataset_conf.get("fbank_conf", {}).get("frame_shift", "-")),
        "frame_skip": str(dataset_conf.get("frame_skip", "-")),
        "sample_rate": str(dataset_conf.get("resample_conf", {}).get("resample_rate", "-")),
    }


def render_model_summary(resources: Dict) -> None:
    profile = build_model_profile(resources)
    threshold_map = resources.get("threshold_map", {})
    threshold_text = " / ".join(
        f"{key}: {value:.3f}" if isinstance(value, (int, float)) else f"{key}: -"
        for key, value in threshold_map.items()
    ) or "-"

    st.markdown("### 当前模型说明")
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="hero-subtitle" style="margin-bottom:0.35rem;">
                <strong>{profile["说明"]}</strong> · {profile["描述"]}
            </div>
            <div class="tag-row">
                <span class="tag-pill">参数量：{profile["参数量"]}</span>
                <span class="tag-pill">历史评测：{profile["历史评测"]}</span>
                <span class="tag-pill">阈值：{threshold_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.1, 1.0], gap="large")
    with left_col:
        st.dataframe(
            pd.DataFrame(
                [
                    {"字段": "模型别名", "值": profile["别名"]},
                    {"字段": "Checkpoint", "值": profile["Checkpoint"]},
                    {"字段": "词表目录", "值": profile["词表目录"]},
                    {"字段": "input_affine_dim", "值": profile["input_affine_dim"]},
                    {"字段": "proj_dim", "值": profile["proj_dim"]},
                    {"字段": "num_layers", "值": profile["num_layers"]},
                    {"字段": "linear_dim", "值": profile["linear_dim"]},
                    {"字段": "merge_head", "值": profile["merge_head"]},
                    {"字段": "output_dim", "值": profile["output_dim"]},
                    {"字段": "hidden_dim", "值": profile["hidden_dim"]},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right_col:
        st.dataframe(
            pd.DataFrame(
                [
                    {"字段": "input_dim", "值": profile["input_dim"]},
                    {"字段": "num_mel_bins", "值": profile["num_mel_bins"]},
                    {"字段": "frame_shift_ms", "值": profile["frame_shift_ms"]},
                    {"字段": "frame_skip", "值": profile["frame_skip"]},
                    {"字段": "sample_rate", "值": profile["sample_rate"]},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def format_diag_hyp_df(hyps: List[Dict], include_match: bool) -> pd.DataFrame:
    rows = []
    for index, hyp in enumerate(hyps, start=1):
        row = {
            "rank": index,
            "text": hyp.get("text", ""),
            "path_score": hyp.get("path_score"),
            "frames": ",".join(str(item) for item in hyp.get("frames", [])),
        }
        if include_match:
            row["matched_keyword"] = hyp.get("matched_keyword")
            row["candidate_score"] = hyp.get("candidate_score")
            row["threshold"] = hyp.get("threshold")
            row["score_margin"] = hyp.get("score_margin")
            row["weakest_token"] = hyp.get("weakest_token")
            row["weakest_token_prob"] = hyp.get("weakest_token_prob")
        rows.append(row)
    return pd.DataFrame(rows)


def format_keyword_diag_df(keyword_diagnostics: List[Dict], time_resolution_sec: float) -> pd.DataFrame:
    rows = []
    status_map = {
        "triggered": "已触发",
        "below_threshold": "已成词但低于阈值",
        "no_complete_match": "未形成完整关键词",
        "matched_without_threshold": "已成词但无阈值",
    }
    for item in keyword_diagnostics:
        start_frame = item.get("candidate_start_frame")
        end_frame = item.get("candidate_end_frame")
        if isinstance(start_frame, int) and isinstance(end_frame, int):
            span_text = f"{start_frame * time_resolution_sec:.2f}-{end_frame * time_resolution_sec:.2f}s"
        else:
            span_text = "-"
        rows.append(
            {
                "keyword": item.get("keyword"),
                "tokens": " ".join(item.get("keyword_tokens", [])),
                "status": status_map.get(item.get("status"), item.get("status")),
                "best_rank": item.get("best_rank"),
                "candidate_score": item.get("candidate_score"),
                "threshold": item.get("threshold"),
                "score_margin": item.get("score_margin"),
                "weakest_token": item.get("weakest_token"),
                "weakest_token_prob": item.get("weakest_token_prob"),
                "candidate_span": span_text,
                "best_text": item.get("best_text"),
            }
        )
    return pd.DataFrame(rows)


def pick_visual_keyword_candidate(diagnosis: Dict) -> Optional[Dict]:
    infer_result = diagnosis.get("infer_result", {})
    keyword_diagnostics = diagnosis.get("keyword_diagnostics", [])
    if infer_result.get("triggered"):
        triggered_keyword = infer_result.get("keyword")
        return next(
            (
                item for item in keyword_diagnostics
                if item.get("keyword") == triggered_keyword and item.get("status") == "triggered"
            ),
            None,
        )

    below_threshold = [item for item in keyword_diagnostics if item.get("status") == "below_threshold"]
    if below_threshold:
        return max(
            below_threshold,
            key=lambda item: item.get("candidate_score") if isinstance(item.get("candidate_score"), (int, float)) else float("-inf"),
        )

    return None


@st.cache_data(show_spinner=False)
def load_waveform_plot_data(wav_path: str, max_points: int = 4000) -> Dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        waveform, sample_rate = torchaudio.load(wav_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    samples = waveform[0]
    total_samples = int(samples.numel())
    if total_samples == 0:
        return {"times": [], "amplitude": [], "sample_rate": sample_rate, "duration_sec": 0.0}

    step = max(1, total_samples // max_points)
    sampled = samples[::step]
    times = [(index * step) / sample_rate for index in range(int(sampled.numel()))]
    return {
        "times": [float(value) for value in times],
        "amplitude": [float(value) for value in sampled.tolist()],
        "sample_rate": int(sample_rate),
        "duration_sec": float(total_samples / sample_rate),
    }


def render_waveform_keyword_plot(record: Dict, diagnosis: Dict) -> None:
    wav_data = load_waveform_plot_data(str(get_record_path(record)))
    if not wav_data["times"]:
        st.info("这条录音没有可视化波形数据。")
        return

    infer_result = diagnosis.get("infer_result", {})
    time_resolution_sec = float(diagnosis.get("time_resolution_sec") or infer_result.get("time_resolution_sec") or 0.03)
    keyword_beam_topk = diagnosis.get("keyword_beam_topk", [])
    matched_hyp = next((item for item in keyword_beam_topk if item.get("matched_keyword")), None)
    visual_hyp = matched_hyp if matched_hyp is not None else (keyword_beam_topk[0] if keyword_beam_topk else None)
    visual_candidate = pick_visual_keyword_candidate(diagnosis)

    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.plot(wav_data["times"], wav_data["amplitude"], color="#2563eb", linewidth=0.8)
    ax.set_title("波形与关键词候选时间位置")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("振幅")
    ax.grid(alpha=0.18)

    candidate_start_frame = visual_candidate.get("candidate_start_frame") if visual_candidate is not None else None
    candidate_end_frame = visual_candidate.get("candidate_end_frame") if visual_candidate is not None else None
    if isinstance(candidate_start_frame, int) and isinstance(candidate_end_frame, int):
        ax.axvspan(
            candidate_start_frame * time_resolution_sec,
            candidate_end_frame * time_resolution_sec,
            color="#f59e0b",
            alpha=0.16,
            label="关键词候选区间",
        )

    start_time = infer_result.get("start_time_sec")
    end_time = infer_result.get("end_time_sec")
    if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
        ax.axvspan(start_time, end_time, color="#10b981", alpha=0.15, label="最终触发区间")

    if visual_hyp is not None:
        frames = visual_hyp.get("frames", [])
        tokens = visual_hyp.get("tokens", [])
        token_times = [frame * time_resolution_sec for frame in frames]
        y_anchor = max(abs(min(wav_data["amplitude"])), abs(max(wav_data["amplitude"]))) * 0.9
        if y_anchor == 0:
            y_anchor = 0.1
        for token_time, token in zip(token_times, tokens):
            ax.axvline(token_time, color="#f59e0b", alpha=0.55, linewidth=1.2)
            ax.text(token_time, y_anchor, token, rotation=90, va="bottom", ha="center", fontsize=9, color="#92400e")

    if (
        (matched_hyp is not None and matched_hyp.get("matched_keyword"))
        or (isinstance(candidate_start_frame, int) and isinstance(candidate_end_frame, int))
        or (isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)))
    ):
        ax.legend(loc="upper right")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_ctc_token_trace_plot(diagnosis: Dict) -> None:
    token_traces = diagnosis.get("token_traces", {})
    frame_time_sec = token_traces.get("frame_time_sec", [])
    traces = token_traces.get("traces", [])
    if not frame_time_sec or not traces:
        st.info("这条录音没有可视化 token 后验曲线。")
        return

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    color_cycle = ["#64748b", "#94a3b8", "#ef4444", "#0ea5e9", "#22c55e", "#8b5cf6", "#f59e0b"]
    for index, trace in enumerate(traces):
        token = trace.get("token", "")
        label = f"{token} ({trace.get('peak_prob', 0.0):.3f})"
        ax.plot(
            frame_time_sec,
            trace.get("probs", []),
            linewidth=1.35 if token not in {"<blk>", "<filler>"} else 1.0,
            alpha=0.95 if token not in {"<blk>", "<filler>"} else 0.7,
            color=color_cycle[index % len(color_cycle)],
            label=label,
        )

    ax.set_title("CTC 原始输出：关键词相关 token 逐帧后验")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("概率")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_ctc_token_heatmap_plot(diagnosis: Dict) -> None:
    token_traces = diagnosis.get("token_traces", {})
    frame_time_sec = token_traces.get("frame_time_sec", [])
    traces = token_traces.get("traces", [])
    if not frame_time_sec or not traces:
        st.info("这条录音没有可视化 token 热力图。")
        return

    labels = [trace.get("token", "") for trace in traces]
    matrix = [trace.get("probs", []) for trace in traces]
    fig, ax = plt.subplots(figsize=(8.6, max(2.4, 0.5 * len(labels) + 1.2)))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma", origin="lower")
    ax.set_title("CTC 后验热力图")
    ax.set_ylabel("Token")
    ax.set_xlabel("时间 (s)")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    tick_count = min(10, len(frame_time_sec))
    if tick_count > 1:
        tick_positions = [round(index * (len(frame_time_sec) - 1) / (tick_count - 1)) for index in range(tick_count)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{frame_time_sec[pos]:.2f}" for pos in tick_positions])

    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_beam_timeline_plot(diagnosis: Dict) -> None:
    keyword_beam_topk = diagnosis.get("keyword_beam_topk", [])
    time_resolution_sec = float(diagnosis.get("time_resolution_sec") or 0.03)
    if not keyword_beam_topk:
        st.info("没有可视化的 beam 路径。")
        return

    rows = keyword_beam_topk[:5]
    fig_height = max(2.8, 0.8 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(8.6, fig_height))

    y_ticks = []
    y_labels = []
    for rank, hyp in enumerate(rows, start=1):
        y_value = len(rows) - rank + 1
        y_ticks.append(y_value)
        matched_keyword = hyp.get("matched_keyword")
        path_score = hyp.get("path_score")
        candidate_score = hyp.get("candidate_score")
        threshold = hyp.get("threshold")
        label_text = matched_keyword or hyp.get("text", "") or f"rank{rank}"
        score_text = f"path={path_score:.3f}" if isinstance(path_score, (int, float)) else "path=-"
        if isinstance(candidate_score, (int, float)):
            score_text += f", cand={candidate_score:.3f}"
        if isinstance(threshold, (int, float)):
            score_text += f", thr={threshold:.3f}"
        y_labels.append(f"#{rank} {label_text} [{score_text}]")
        frames = hyp.get("frames", [])
        token_times = [frame * time_resolution_sec for frame in frames]
        tokens = hyp.get("tokens", [])
        if matched_keyword and isinstance(candidate_score, (int, float)) and isinstance(threshold, (int, float)):
            color = "#10b981" if candidate_score >= threshold else "#f59e0b"
        else:
            color = "#3b82f6" if not matched_keyword else "#14b8a6"
        ax.plot(token_times, [y_value] * len(token_times), color=color, linewidth=1.2, alpha=0.9)
        ax.scatter(token_times, [y_value] * len(token_times), color=color, s=28)
        for token_time, token in zip(token_times, tokens):
            ax.text(token_time, y_value + 0.08, token, ha="center", va="bottom", fontsize=9)
        if token_times:
            score_annot = f"{path_score:.3f}" if isinstance(path_score, (int, float)) else "-"
            if isinstance(candidate_score, (int, float)):
                score_annot += f" / {candidate_score:.3f}"
            ax.text(token_times[-1] + 0.03, y_value, score_annot, va="center", fontsize=9, color=color)

    ax.set_title("关键词约束 Beam Search 路径时间线")
    ax.set_xlabel("时间 (s)")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(0.5, len(rows) + 0.8)
    ax.grid(alpha=0.18, axis="x")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def format_shape(shape: List[int]) -> str:
    if not shape:
        return "-"
    return " x ".join(str(item) for item in shape)


def render_flow_stage_cards(pipeline: Dict) -> None:
    stages = pipeline.get("flow_stages", [])
    if not stages:
        st.info("没有可展示的全链路阶段信息。")
        return

    fragments = []
    for index, stage in enumerate(stages):
        stats = stage.get("stats", {})
        fragments.append(
            f"""
            <div class="flow-card">
                <div class="flow-card-title">{stage.get('title', '')}</div>
                <div class="flow-card-shape">{format_shape(stage.get('shape', []))}</div>
                <div class="flow-card-note">{stage.get('note', '')}</div>
                <div class="flow-card-stats">
                    mean={stats.get('mean', 0.0):.3f} | std={stats.get('std', 0.0):.3f}<br/>
                    min={stats.get('min', 0.0):.3f} | max={stats.get('max', 0.0):.3f}
                </div>
            </div>
            """
        )
        if index != len(stages) - 1:
            fragments.append('<div class="flow-arrow">→</div>')

    st.markdown(f"<div class='flow-grid'>{''.join(fragments)}</div>", unsafe_allow_html=True)


def payload_to_heatmap_matrix(payload: Dict) -> Optional[np.ndarray]:
    matrix = payload.get("matrix", [])
    if not matrix:
        return None
    array = np.array(matrix, dtype=float)
    if array.ndim != 2:
        return None
    return array.T


def render_matrix_heatmap_plot(payload: Dict, cmap: str = "magma") -> None:
    data = payload_to_heatmap_matrix(payload)
    if data is None:
        st.info("当前阶段没有二维矩阵可视化。")
        return

    fig, ax = plt.subplots(figsize=(8.4, max(3.2, min(6.5, data.shape[0] * 0.03 + 2.2))))
    image = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, origin="lower")
    ax.set_title(payload.get("title", ""))
    ax.set_ylabel(payload.get("y_axis_name", "dim"))
    ax.set_xlabel("时间 / 帧")

    time_axis = payload.get("time_axis_sec", [])
    if time_axis:
        tick_count = min(10, len(time_axis))
        if tick_count > 1:
            tick_positions = [round(index * (len(time_axis) - 1) / (tick_count - 1)) for index in range(tick_count)]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"{time_axis[pos]:.2f}" for pos in tick_positions])

    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_matrix_surface_plot(payload: Dict, cmap: str = "viridis") -> None:
    data = payload_to_heatmap_matrix(payload)
    if data is None:
        st.info("当前阶段没有 3D 曲面可视化。")
        return

    y_step = max(1, data.shape[0] // 48)
    x_step = max(1, data.shape[1] // 72)
    data_sub = data[::y_step, ::x_step]
    x_coords = np.arange(data_sub.shape[1])
    y_coords = np.arange(data_sub.shape[0])
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    fig = plt.figure(figsize=(8.4, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(grid_x, grid_y, data_sub, cmap=cmap, linewidth=0, antialiased=True, alpha=0.96)
    ax.set_title(f"{payload.get('title', '')} 3D 曲面")
    ax.set_xlabel("时间索引")
    ax.set_ylabel(payload.get("y_axis_name", "dim"))
    ax.set_zlabel("值")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_matrix_hist_plot(payload: Dict, color: str = "#2563eb") -> None:
    matrix = payload.get("matrix", [])
    if not matrix:
        st.info("当前阶段没有分布直方图。")
        return
    values = np.array(matrix, dtype=float).reshape(-1)
    if values.size == 0:
        st.info("当前阶段没有分布直方图。")
        return

    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    ax.hist(values, bins=48, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_title(f"{payload.get('title', '')} 数值分布")
    ax.set_xlabel("值")
    ax.set_ylabel("计数")
    ax.grid(alpha=0.18)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_stage_visual_triptych(payload: Dict, cmap: str = "magma") -> None:
    col1, col2 = st.columns([1.2, 1.0], gap="large")
    with col1:
        render_matrix_heatmap_plot(payload, cmap=cmap)
    with col2:
        render_matrix_surface_plot(payload, cmap="viridis")
    render_matrix_hist_plot(payload)

    stats = payload.get("stats", {})
    if stats:
        stats_df = pd.DataFrame(
            [
                {"字段": "shape", "值": format_shape(stats.get("shape", []))},
                {"字段": "min", "值": stats.get("min")},
                {"字段": "p01", "值": stats.get("p01")},
                {"字段": "mean", "值": stats.get("mean")},
                {"字段": "p50", "值": stats.get("p50")},
                {"字段": "p99", "值": stats.get("p99")},
                {"字段": "max", "值": stats.get("max")},
                {"字段": "std", "值": stats.get("std")},
                {"字段": "nonzero_ratio", "值": stats.get("nonzero_ratio")},
            ]
        )
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    if payload.get("note"):
        st.caption(payload["note"])


def render_frame_energy_plot(pipeline: Dict) -> None:
    frame_energy = pipeline.get("frame_energy", {})
    values = frame_energy.get("values", [])
    times_sec = frame_energy.get("times_sec", [])
    if not values or not times_sec:
        st.info("没有可展示的逐帧能量。")
        return

    fig, ax = plt.subplots(figsize=(8.4, 3.1))
    ax.plot(times_sec, values, color="#0f766e", linewidth=1.6)
    ax.fill_between(times_sec, values, color="#2dd4bf", alpha=0.18)
    ax.set_title("逐帧 Fbank 平均能量")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("平均能量")
    ax.grid(alpha=0.18)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_context_skip_panel(record: Dict, pipeline: Dict) -> None:
    context_debug = pipeline.get("context_debug", {})
    frame_skip_debug = pipeline.get("frame_skip_debug", {})
    cmvn_debug = pipeline.get("cmvn_debug", {})
    windows = context_debug.get("windows", [])

    summary_rows = [
        {"字段": "Context 开关", "值": context_debug.get("enabled")},
        {"字段": "left/right", "值": f"{context_debug.get('left', 0)} / {context_debug.get('right', 0)}"},
        {"字段": "Context 输入 shape", "值": format_shape(context_debug.get("input_shape", []))},
        {"字段": "Context 输出 shape", "值": format_shape(context_debug.get("output_shape", []))},
        {"字段": "frame_skip", "值": frame_skip_debug.get("frame_skip")},
        {"字段": "Skip 输入 shape", "值": format_shape(frame_skip_debug.get("input_shape", []))},
        {"字段": "Skip 输出 shape", "值": format_shape(frame_skip_debug.get("output_shape", []))},
        {"字段": "CMVN 开关", "值": cmvn_debug.get("enabled")},
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    if windows:
        default_frame = min(len(windows) - 1, max(0, len(windows) // 2))
        selected_frame = st.slider(
            "选择一个 context 输出帧",
            min_value=0,
            max_value=len(windows) - 1,
            value=default_frame,
            key=f"context_window_{record['id']}",
        )
        selected_window = windows[selected_frame]
        source_df = pd.DataFrame(
            [
                {"拼接槽位": slot_index, "来源原始 Fbank 帧": frame_index}
                for slot_index, frame_index in enumerate(selected_window)
            ]
        )
        col1, col2 = st.columns([0.95, 1.05], gap="large")
        with col1:
            st.markdown("###### Context 窗口来源")
            st.dataframe(source_df, use_container_width=True, hide_index=True)
        with col2:
            fig, ax = plt.subplots(figsize=(6.8, 2.8))
            ax.step(range(len(selected_window)), selected_window, where="mid", color="#7c3aed", linewidth=2.0)
            ax.scatter(range(len(selected_window)), selected_window, color="#a855f7", s=42)
            ax.set_title(f"输出帧 {selected_frame} 的拼接来源")
            ax.set_xlabel("槽位")
            ax.set_ylabel("原始帧索引")
            ax.grid(alpha=0.18)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    kept = frame_skip_debug.get("kept_input_indices", [])
    dropped = frame_skip_debug.get("dropped_input_indices", [])
    if kept:
        fig, ax = plt.subplots(figsize=(8.2, 2.8))
        ax.scatter(kept, [1] * len(kept), color="#10b981", s=36, label="保留帧")
        if dropped:
            ax.scatter(dropped, [0] * len(dropped), color="#ef4444", s=18, alpha=0.7, label="丢弃帧")
        ax.set_title("Frame Skip 取帧示意")
        ax.set_xlabel("Context 输出帧索引")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["drop", "keep"])
        ax.grid(alpha=0.18, axis="x")
        ax.legend(loc="upper right")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    if cmvn_debug.get("enabled"):
        st.caption(
            "CMVN 预览：前 16 维全局均值与逆标准差已加载，可直接用来解释归一化前后分布差异。"
        )
        cmvn_df = pd.DataFrame(
            {
                "dim": list(range(len(cmvn_debug.get("mean_preview", [])))),
                "mean": cmvn_debug.get("mean_preview", []),
                "istd": cmvn_debug.get("istd_preview", []),
            }
        )
        st.dataframe(cmvn_df, use_container_width=True, hide_index=True)


def render_keyword_strength_panel(pipeline: Dict) -> None:
    token_strength = pipeline.get("token_strength", [])
    if not token_strength:
        st.info("没有可展示的关键词 token 强弱分析。")
        return

    for keyword_item in token_strength:
        keyword = keyword_item.get("keyword", "")
        token_rows = keyword_item.get("token_rows", [])
        if not token_rows:
            continue

        st.markdown(f"###### {keyword}")
        summary = [
            {"字段": "状态", "值": keyword_item.get("status")},
            {"字段": "candidate_score", "值": keyword_item.get("candidate_score")},
            {"字段": "threshold", "值": keyword_item.get("threshold")},
        ]
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        df = pd.DataFrame(token_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        weakest = None
        if "matched_prob" in df.columns and df["matched_prob"].notna().any():
            weakest = df[df["matched_prob"].notna()].sort_values("matched_prob").iloc[0]
        elif "peak_prob" in df.columns and not df.empty:
            weakest = df.sort_values("peak_prob").iloc[0]
        if weakest is not None:
            st.caption(
                f"当前最弱 token: `{weakest['token']}`，peak_prob={weakest.get('peak_prob', float('nan')):.3f}，"
                f"matched_prob={weakest.get('matched_prob') if pd.notna(weakest.get('matched_prob')) else 'N/A'}。"
            )

        fig, ax1 = plt.subplots(figsize=(8.4, 3.2))
        x = np.arange(len(df))
        width = 0.35
        peak_prob = df["peak_prob"].fillna(0.0).to_numpy()
        matched_prob = df["matched_prob"].fillna(0.0).to_numpy() if "matched_prob" in df else np.zeros(len(df))
        ax1.bar(x - width / 2, peak_prob, width=width, color="#2563eb", label="peak_prob")
        ax1.bar(x + width / 2, matched_prob, width=width, color="#f59e0b", label="matched_prob")
        ax1.set_ylabel("token 概率")
        ax1.set_ylim(0.0, max(1.0, float(max(peak_prob.max(), matched_prob.max(), 1.0))))
        ax1.set_xticks(x)
        ax1.set_xticklabels(df["token"].tolist())
        ax1.grid(alpha=0.18, axis="y")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        local_energy = df["local_energy"].fillna(df["frame_energy"]).fillna(0.0).to_numpy()
        ax2.plot(x, local_energy, color="#047857", marker="o", linewidth=2.0, label="局部能量")
        ax2.set_ylabel("能量")
        ax1.set_title(f"{keyword} token 强弱对比")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def render_beam_trace_panel(record: Dict, pipeline: Dict) -> None:
    beam_trace = pipeline.get("beam_trace", {})
    trace_mode = st.radio(
        "Beam 视角",
        ["关键词约束", "普通"],
        horizontal=True,
        key=f"beam_trace_mode_{record['id']}",
    )
    trace_key = "keyword" if trace_mode == "关键词约束" else "plain"
    trace = beam_trace.get(trace_key, {})
    frames = trace.get("frames", [])
    if not frames:
        st.info("没有可展示的 beam 轨迹。")
        return

    rank_count = min(5, max((len(frame.get("beam_after", [])) for frame in frames), default=0))
    if rank_count > 0:
        fig, ax = plt.subplots(figsize=(8.4, 3.4))
        frame_times = [frame.get("time_sec", 0.0) for frame in frames]
        for rank in range(rank_count):
            values = []
            for frame in frames:
                beam_after = frame.get("beam_after", [])
                values.append(beam_after[rank].get("score") if rank < len(beam_after) else np.nan)
            ax.plot(frame_times, values, linewidth=1.45, label=f"rank {rank + 1}")
        ax.set_title(f"{trace_mode} Beam 每帧 Top 路径分数")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("score")
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    selected_frame = st.slider(
        "逐帧查看 Beam 展开过程",
        min_value=0,
        max_value=len(frames) - 1,
        value=min(len(frames) - 1, max(0, len(frames) // 2)),
        key=f"beam_trace_frame_{record['id']}_{trace_key}",
    )
    frame_item = frames[selected_frame]
    st.caption(
        f"第 {selected_frame} 帧，time={frame_item.get('time_sec', 0.0):.3f}s。"
        "上表是这一帧参与扩展的 top token，下表是二次 prune 后保留下来的 beam。"
    )

    top_tokens = frame_item.get("top_tokens", [])
    beam_after = frame_item.get("beam_after", [])
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("###### 当前帧 Top Token")
        if top_tokens:
            st.dataframe(pd.DataFrame(top_tokens), use_container_width=True, hide_index=True)
        else:
            st.info("这一帧没有通过阈值过滤的 token。")
    with right:
        st.markdown("###### 当前帧保留的 Beam")
        if beam_after:
            st.dataframe(pd.DataFrame(beam_after), use_container_width=True, hide_index=True)
        else:
            st.info("这一帧没有保留下来的路径。")

    final_hyps = trace.get("final_hyps", [])
    st.markdown("###### 最终 Beam 结果")
    if final_hyps:
        st.dataframe(pd.DataFrame(final_hyps), use_container_width=True, hide_index=True)
    else:
        st.info("没有最终 Beam 结果。")


def render_pipeline_visualization_section(record: Dict, gpu: int) -> None:
    infer_result = record.get("infer_result", {})
    wav_path = str(get_record_path(record))
    model_alias = record.get("model_alias", "")
    checkpoint = record.get("checkpoint", "")
    config = infer_result.get("config", "")
    dict_dir = infer_result.get("dict_dir", "")
    stats_dir = infer_result.get("stats_dir") or ""

    with st.spinner("正在提取 wav -> 特征 -> FSMN -> beam 的全链路中间态..."):
        pipeline = compute_record_pipeline(
            wav_path=wav_path,
            model_alias=model_alias,
            checkpoint=checkpoint,
            config=config,
            dict_dir=dict_dir,
            stats_dir=stats_dir,
            gpu=gpu,
            beam_size=DEFAULT_DIAGNOSE_BEAM_SIZE,
            cache_version=PIPELINE_CACHE_VERSION,
        )

    st.markdown("##### 全链路数据流")
    render_flow_stage_cards(pipeline)

    top_tabs = st.tabs(["输入与频谱", "前处理细节", "模型层", "关键词强弱", "Beam 逐帧"])

    with top_tabs[0]:
        raw_waveform = pipeline.get("audio", {}).get("resampled_waveform", {})
        if raw_waveform.get("times") and raw_waveform.get("amplitude"):
            fig, ax = plt.subplots(figsize=(8.4, 3.0))
            ax.plot(raw_waveform["times"], raw_waveform["amplitude"], color="#2563eb", linewidth=0.75)
            ax.set_title("重采样后波形")
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("振幅")
            ax.grid(alpha=0.18)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        render_frame_energy_plot(pipeline)
        spectral_tabs = st.tabs(["STFT", "Mel", "log-Mel", "实际 Fbank"])
        spectral_payloads = [
            pipeline.get("spectra", {}).get("stft_db", {}),
            pipeline.get("spectra", {}).get("mel_energy", {}),
            pipeline.get("spectra", {}).get("log_mel", {}),
            pipeline.get("spectra", {}).get("fbank", {}),
        ]
        for tab, payload in zip(spectral_tabs, spectral_payloads):
            with tab:
                render_stage_visual_triptych(payload)

    with top_tabs[1]:
        render_context_skip_panel(record, pipeline)
        fbank_payload = pipeline.get("spectra", {}).get("fbank", {})
        if fbank_payload:
            st.markdown("###### 实际 Fbank 可视化")
            render_stage_visual_triptych(fbank_payload)

    with top_tabs[2]:
        model_layers = pipeline.get("model_layers", [])
        if not model_layers:
            st.info("当前模型格式不支持逐层展开。")
        else:
            layer_titles = [f"{item['title']} [{format_shape(item.get('shape', []))}]" for item in model_layers]
            selected_title = st.selectbox(
                "选择一个模型内部阶段",
                options=layer_titles,
                index=min(len(layer_titles) - 1, 0),
                key=f"model_layer_select_{record['id']}",
            )
            selected_index = layer_titles.index(selected_title)
            render_stage_visual_triptych(model_layers[selected_index], cmap="viridis")

    with top_tabs[3]:
        render_keyword_strength_panel(pipeline)

    with top_tabs[4]:
        render_beam_trace_panel(record, pipeline)


def render_diagnosis_section(record: Dict, gpu: int) -> None:
    infer_result = record.get("infer_result", {})
    wav_path = str(get_record_path(record))
    model_alias = record.get("model_alias", "")
    checkpoint = record.get("checkpoint", "")
    config = infer_result.get("config", "")
    dict_dir = infer_result.get("dict_dir", "")
    stats_dir = infer_result.get("stats_dir") or ""

    st.markdown("##### 诊断结果")
    with st.spinner("正在生成 greedy / beam / 关键词约束 beam 诊断..."):
        diagnosis = compute_record_diagnosis(
            wav_path=wav_path,
            model_alias=model_alias,
            checkpoint=checkpoint,
            config=config,
            dict_dir=dict_dir,
            stats_dir=stats_dir,
            gpu=gpu,
            beam_size=DEFAULT_DIAGNOSE_BEAM_SIZE,
            frame_topk=DEFAULT_DIAGNOSE_FRAME_TOPK,
            cache_version=DIAGNOSIS_CACHE_VERSION,
        )
    token_traces = diagnosis.get("token_traces", {})
    if not token_traces.get("frame_time_sec") or not token_traces.get("traces"):
        with st.spinner("检测到旧诊断缓存，正在刷新可视化数据..."):
            diagnosis = compute_record_diagnosis_uncached(
                wav_path=wav_path,
                model_alias=model_alias,
                checkpoint=checkpoint,
                config=config,
                dict_dir=dict_dir,
                stats_dir=stats_dir,
                gpu=gpu,
                beam_size=DEFAULT_DIAGNOSE_BEAM_SIZE,
                frame_topk=DEFAULT_DIAGNOSE_FRAME_TOPK,
            )

    audio_stats = diagnosis.get("audio_stats", {})
    greedy_decode = diagnosis.get("greedy_decode", {})
    beam_topk = diagnosis.get("beam_topk", [])
    keyword_beam_topk = diagnosis.get("keyword_beam_topk", [])
    keyword_diagnostics = diagnosis.get("keyword_diagnostics", [])
    decode_assessment = diagnosis.get("decode_assessment", {})
    mean_frame_topk_tokens = diagnosis.get("mean_frame_topk_tokens", [])
    matched_hyp = next((item for item in keyword_beam_topk if item.get("matched_keyword")), None)
    time_resolution_sec = float(diagnosis.get("time_resolution_sec") or 0.03)
    best_keyword_row = pick_visual_keyword_candidate(diagnosis)

    summary_rows = [
        {"字段": "整体判断", "值": decode_assessment.get("message") or "-"},
        {"字段": "Greedy 文本", "值": greedy_decode.get("text") or "(空)"},
        {"字段": "Beam Top1", "值": beam_topk[0].get("text") if beam_topk else "(空)"},
        {
            "字段": "关键词 Beam 命中",
            "值": matched_hyp.get("matched_keyword") if matched_hyp else "未形成完整关键词",
        },
        {
            "字段": "关键词 Beam 文本",
            "值": matched_hyp.get("text") if matched_hyp else (keyword_beam_topk[0].get("text") if keyword_beam_topk else "(空)"),
        },
    ]
    if best_keyword_row is not None:
        summary_rows.append(
            {
                "字段": "最强候选阈值差",
                "值": (
                    f"{best_keyword_row.get('keyword')}: "
                    f"{best_keyword_row.get('score_margin'):.3f}"
                    if isinstance(best_keyword_row.get("score_margin"), (int, float))
                    else "-"
                ),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.caption("说明：`path_score` 是 beam 内部路径排序分数；`candidate_score` 只对已形成完整关键词的路径计算，并会拿去和阈值比较。")

    if decode_assessment.get("status") == "no_complete_match":
        st.warning("这条录音在关键词约束 beam 下也没有形成完整关键词子串，更像是发音/声学条件偏移，或 beam 没保住关键路径，而不是单纯阈值太高。")
    elif decode_assessment.get("status") == "below_threshold":
        st.warning("这条录音已经形成完整关键词，但 candidate_score 仍低于阈值。优先检查阈值、弱 token 概率和该关键词对应训练样本覆盖。")

    st.markdown("###### 可视化")
    render_waveform_keyword_plot(record, diagnosis)
    render_ctc_token_trace_plot(diagnosis)
    render_ctc_token_heatmap_plot(diagnosis)
    render_beam_timeline_plot(diagnosis)

    col1, col2 = st.columns([1.0, 1.3], gap="large")
    with col1:
        st.markdown("###### 关键词诊断")
        keyword_diag_df = format_keyword_diag_df(keyword_diagnostics, time_resolution_sec)
        st.dataframe(keyword_diag_df, use_container_width=True, hide_index=True)
        st.markdown("###### 音频统计")
        st.dataframe(
            pd.DataFrame([{"字段": key, "值": value} for key, value in audio_stats.items()]),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("###### 平均帧 Top-K Token")
        st.dataframe(pd.DataFrame(mean_frame_topk_tokens), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("###### Greedy Token 序列")
        st.code(" ".join(greedy_decode.get("tokens", [])) or "(空)", language="text")
        st.markdown("###### 关键词约束 Beam Top-K")
        keyword_df = format_diag_hyp_df(keyword_beam_topk, include_match=True)
        if keyword_df.empty:
            st.info("没有可展示的关键词约束 beam 假设。")
        else:
            st.dataframe(keyword_df, use_container_width=True, hide_index=True)

    st.markdown("###### 普通 Beam Top-K")
    beam_df = format_diag_hyp_df(beam_topk, include_match=False)
    if beam_df.empty:
        st.info("没有可展示的 beam 假设。")
    else:
        st.dataframe(beam_df, use_container_width=True, hide_index=True)

    with st.expander("查看完整诊断 JSON", expanded=False):
        st.json(diagnosis)


def render_metrics(records: List[Dict]) -> None:
    matrix, metrics_df, stats = compute_confusion(records)
    labeled_count = stats["labeled_count"]
    wrong_count = sum(1 for item in records if item.get("is_correct") is False)
    correct_count = sum(1 for item in records if item.get("is_correct") is True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("当前视图总音频", len(records))
    col2.metric("已标注数量", labeled_count)
    col3.metric("标注正确数", correct_count)
    col4.metric("标注错误数", wrong_count)

    col5, col6, col7 = st.columns(3)
    col5.metric("Overall Accuracy", f"{stats['overall_accuracy'] * 100:.2f}%")
    col6.metric("Macro Precision", f"{stats['macro_precision'] * 100:.2f}%")
    col7.metric("Macro Recall", f"{stats['macro_recall'] * 100:.2f}%")

    matrix_col, metric_col = st.columns([1.25, 1.0])
    with matrix_col:
        st.markdown("#### 混淆矩阵")
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="crest",
            cbar=False,
            ax=ax,
            annot_kws={"fontsize": 13, "fontweight": "bold"},
        )
        ax.set_xlabel("预测标签")
        ax.set_ylabel("人工标签")
        ax.tick_params(axis="x", rotation=15)
        ax.tick_params(axis="y", rotation=0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with metric_col:
        st.markdown("#### 分类指标")
        if metrics_df.empty:
            st.info("还没有已标注录音，先录一条并打标签。")
        else:
            display_df = metrics_df.copy()
            display_df["Precision"] = display_df["Precision"].map(lambda x: f"{x * 100:.2f}%")
            display_df["Recall"] = display_df["Recall"].map(lambda x: f"{x * 100:.2f}%")
            st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_sidebar(records: List[Dict], active_records: List[Dict]) -> Optional[str]:
    st.sidebar.title("录音历史")
    st.sidebar.caption("点击左侧条目可在主面板中回放并重新标注。")

    selected_id = st.session_state.get("selected_record_id")

    if not records:
        st.sidebar.info("还没有录音记录。")
        return selected_id

    for record in active_records:
        created = record.get("created_at", "")[-8:]
        predicted = record.get("predicted_label", "非唤醒词")
        human = record.get("human_label")
        if human is None:
            status = "待标"
            status_marker = "•"
        elif record.get("is_correct"):
            status = "正确"
            status_marker = "✓"
        else:
            status = "错误"
            status_marker = "×"
        button_text = f"{status_marker} {created} | {predicted} | {status}"
        row_col1, row_col2 = st.sidebar.columns([4.4, 1.0], gap="small")
        if row_col1.button(button_text, key=f"history_{record['id']}", use_container_width=True):
            st.session_state["selected_record_id"] = record["id"]
            selected_id = record["id"]
        if row_col2.button("🗑", key=f"delete_{record['id']}", use_container_width=True):
            updated = delete_record(records, record["id"])
            if st.session_state.get("selected_record_id") == record["id"]:
                st.session_state["selected_record_id"] = None
            save_records(updated)
            st.rerun()

    st.sidebar.markdown("---")
    if selected_id:
        selected = next((item for item in records if item["id"] == selected_id), None)
        if selected:
            st.sidebar.markdown("### 侧边栏试听")
            st.sidebar.audio(str(get_record_path(selected)))
            st.sidebar.caption(f"录制时间：{selected.get('created_at', '')}")
            st.sidebar.caption(f"预测标签：{selected.get('predicted_label', '非唤醒词')}")

    return selected_id


def render_selected_record(record: Optional[Dict], records: List[Dict], gpu: int) -> None:
    st.markdown("#### 当前录音详情")
    if not record:
        st.info("先录一条音，或者从左侧选择一条历史录音。")
        return

    infer_result = record.get("infer_result", {})
    human_label = record.get("human_label")
    current_index = LABEL_OPTIONS.index(human_label) if human_label in LABEL_OPTIONS else LABEL_OPTIONS.index(record["predicted_label"])
    status_class = "pred-pending"
    status_text = "待标注"
    if record.get("is_correct") is True:
        status_class = "pred-good"
        status_text = "标注正确"
    elif record.get("is_correct") is False:
        status_class = "pred-bad"
        status_text = "标注错误"

    st.markdown(
        f"""
        <div class="panel-card">
            <div class="tag-row">
                <span class="tag-pill">{record.get('created_at', '')}</span>
                <span class="tag-pill">{record.get('model_alias', '')}</span>
                <span class="tag-pill {status_class}">{status_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("预测标签", record.get("predicted_label", "非唤醒词"))
    score = infer_result.get("score")
    info_col2.metric("推理分数", f"{score:.3f}" if isinstance(score, (int, float)) else "N/A")
    wake_time = infer_result.get("wake_time_sec")
    info_col3.metric("唤醒时间点", f"{wake_time:.3f}s" if isinstance(wake_time, (int, float)) else "未触发")

    st.audio(str(get_record_path(record)))

    st.markdown("##### 人工标签")
    selected_label = st.radio(
        "请选择真实标签",
        LABEL_OPTIONS,
        index=current_index,
        horizontal=True,
        key=f"label_radio_{record['id']}",
    )
    if st.button("保存标签", key=f"save_label_{record['id']}", type="primary"):
        updated = update_record_label(records, record["id"], selected_label)
        save_records(updated)
        st.success("标签已保存。")
        st.rerun()

    detail_df = pd.DataFrame(
        [
            {"字段": "预测关键词", "值": infer_result.get("keyword")},
            {"字段": "最终阈值", "值": infer_result.get("threshold")},
            {"字段": "开始时间", "值": infer_result.get("start_time_sec")},
            {"字段": "结束时间", "值": infer_result.get("end_time_sec")},
            {"字段": "Checkpoint", "值": record.get("checkpoint")},
        ]
    )
    st.dataframe(detail_df, use_container_width=True, hide_index=True)
    diag_tab, pipeline_tab = st.tabs(["诊断视图", "全链路工作台"])
    with diag_tab:
        render_diagnosis_section(record, gpu)
    with pipeline_tab:
        render_pipeline_visualization_section(record, gpu)


def render_recorder(resources: Dict, records: List[Dict]) -> None:
    st.markdown("#### 录音并推理")
    st.caption("停止录音后会自动保存并推理，不需要再点额外按钮。")

    audio_source = None
    if hasattr(st, "audio_input"):
        audio_source = st.audio_input("点击开始录音", key="audio_input_recorder")
    else:
        st.warning("当前 Streamlit 版本不支持 `audio_input`，自动切换为 wav 文件上传。")
        audio_source = st.file_uploader("上传 wav 音频", type=["wav"], key="audio_input_fallback")

    if audio_source is not None:
        st.audio(audio_source)
        audio_bytes = audio_source.getvalue()
        digest = hashlib.sha1(audio_bytes).hexdigest()
        if st.session_state.get("last_audio_sha1") != digest:
            suffix = Path(getattr(audio_source, "name", "record.wav")).suffix or ".wav"
            with st.spinner("正在保存录音并执行推理..."):
                record = create_record_from_audio(audio_bytes, suffix, resources)
                records.insert(0, record)
                save_records(records)

            st.session_state["selected_record_id"] = record["id"]
            st.session_state["last_audio_sha1"] = digest
            st.success("录音已自动保存并完成推理。")
            st.rerun()


def filter_records(records: List[Dict], scope: str, current_signature: str) -> List[Dict]:
    if scope == "全部记录":
        return records
    return [record for record in records if record.get("model_signature") == current_signature]


def main():
    ensure_runtime_dirs()
    inject_css()

    records = sorted(load_records(), key=lambda item: item.get("created_at", ""), reverse=True)

    with st.sidebar:
        st.markdown("## 模型设置")
        model_alias = st.selectbox(
            "模型别名",
            options=sorted(iw.MODEL_ALIASES.keys()),
            index=sorted(iw.MODEL_ALIASES.keys()).index(DEFAULT_MODEL_ALIAS),
        )
        gpu = st.number_input("GPU ID", min_value=-1, value=0, step=1)
        with st.expander("高级覆盖参数", expanded=False):
            custom_checkpoint = st.text_input("自定义 checkpoint", value="")
            custom_model_dir = st.text_input("自定义 model_dir", value="")
            checkpoint_name = st.text_input("checkpoint_name", value="")
        scope = st.radio("统计范围", ["当前模型", "全部记录"], horizontal=False)

    resources = load_infer_resources(
        model_alias=model_alias,
        custom_checkpoint=custom_checkpoint,
        custom_model_dir=custom_model_dir,
        checkpoint_name=checkpoint_name,
        gpu=int(gpu),
    )

    current_signature = build_model_signature(resources)
    active_records = filter_records(records, scope, current_signature)
    render_hero(resources, records, active_records)
    st.write("")
    render_model_summary(resources)
    st.write("")

    selected_id = render_sidebar(records, active_records)
    if not selected_id and active_records:
        selected_id = active_records[0]["id"]
    selected_record = next((item for item in records if item["id"] == selected_id), None) if selected_id else None

    top_left, top_right = st.columns([1.05, 1.2], gap="large")
    with top_left:
        render_recorder(resources, records)
    with top_right:
        render_selected_record(selected_record, records, int(gpu))

    st.write("")
    st.markdown("### 整体统计")
    render_metrics(active_records)

    st.write("")
    st.markdown("### 当前记录表")
    records_df = build_records_df(active_records)
    if records_df.empty:
        st.info("当前范围下还没有录音。")
    else:
        st.dataframe(records_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
