#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频数据浏览器 WebUI - Stage 1 可视化工具
作者: Wayne
功能: 交互式浏览、搜索和可视化音频数据集
"""

import os
import sys
import sqlite3
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from PIL import Image
from pywayne.tools import read_yaml_config, write_yaml_config

# 设置页面配置
st.set_page_config(
    page_title="音频数据浏览器",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 获取项目路径
# 假设从项目根目录启动：tools/ -> ../examples/hi_xiaowen/s0/
SCRIPT_DIR = Path(__file__).parent.resolve()  # tools/
REPO_ROOT = SCRIPT_DIR.parent  # wekws/
PROJECT_DIR = REPO_ROOT / "examples" / "hi_xiaowen" / "s0"  # 项目目录
DB_PATH = PROJECT_DIR / "data" / "metadata.db"
VISUALIZE_SCRIPT = PROJECT_DIR / "wayne_scripts" / "stage_1_visualize.py"
VISUALIZE_DIR = PROJECT_DIR / "wayne_scripts" / "visualizations"
SCORE_CTC_SCRIPT = REPO_ROOT / "wekws" / "bin" / "score_ctc.py"
MODEL_REGISTRY_YAML = PROJECT_DIR / "wayne_scripts" / "webui_models.yaml"
INFER_DICT_DIR = PROJECT_DIR / "dict_top20"
INFER_TOKEN_FILE = PROJECT_DIR / "mobvoi_kws_transcription" / "tokens.txt"
INFER_LEXICON_FILE = PROJECT_DIR / "mobvoi_kws_transcription" / "lexicon.txt"
INFER_KEYWORDS_ESCAPED = r"\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee"
INFER_WAKEWORDS = {"嗨小问", "你好问问"}
FRAME_SHIFT_SECONDS = 0.01
WAKEWORD_DISPLAY = {
    "嗨小问": "haixiaowen",
    "你好问问": "nihaowenwen",
}


# ==================== 数据库操作 ====================
@st.cache_resource
def get_database_connection():
    """
    获取数据库连接（缓存）
    """
    # 调试信息
    db_exists = DB_PATH.exists()
    
    if not db_exists:
        st.error(f"❌ 数据库不存在")
        st.code(f"期望路径: {DB_PATH}", language="bash")
        
        # 调试信息
        with st.expander("🔍 调试信息"):
            st.write(f"**SCRIPT_DIR**: `{SCRIPT_DIR}`")
            st.write(f"**PROJECT_DIR**: `{PROJECT_DIR}`")
            st.write(f"**DB_PATH**: `{DB_PATH}`")
            st.write(f"**数据库存在**: {db_exists}")
            
            # 检查 data 目录
            data_dir = PROJECT_DIR / "data"
            st.write(f"**data 目录存在**: {data_dir.exists()}")
            if data_dir.exists():
                try:
                    files = list(data_dir.glob("*"))
                    st.write(f"**data 目录内容** ({len(files)} 个文件):")
                    for f in files[:10]:  # 只显示前10个
                        st.write(f"  - {f.name}")
                except Exception as e:
                    st.write(f"无法读取 data 目录: {e}")
        
        st.info("""
请先运行构建数据库脚本：

**方法1（推荐）：通过主脚本**
```bash
cd /path/to/project/examples/hi_xiaowen/s0
bash run_fsmn_ctc.sh 1.5 1.5
```

**方法2：直接运行**
```bash
cd /path/to/project/examples/hi_xiaowen/s0
tools/generate_metadata_db.py
```
        """)
        st.stop()
    
    # 显示数据库信息
    db_size = DB_PATH.stat().st_size / 1024 / 1024  # MB
    st.sidebar.success(f"✅ 数据库已连接 ({db_size:.1f} MB)")
    
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row  # 使结果可以通过列名访问
    return conn


def _normalize_model_path(path_str: str) -> str:
    model_path = Path(path_str).expanduser()
    if not model_path.is_absolute():
        model_path = (PROJECT_DIR / model_path).resolve()
    else:
        model_path = model_path.resolve()
    return str(model_path)


def load_model_registry() -> tuple[Dict[str, Any], Optional[str]]:
    default_registry: Dict[str, Any] = {'models': [], 'last_selected': ''}
    if not MODEL_REGISTRY_YAML.exists():
        write_yaml_config(str(MODEL_REGISTRY_YAML), default_registry, update=False)
        return default_registry, None

    try:
        data = read_yaml_config(str(MODEL_REGISTRY_YAML))
    except Exception as e:
        return default_registry, str(e)

    if not isinstance(data, dict):
        return default_registry, None

    models = []
    for model_path in data.get('models', []):
        if not isinstance(model_path, str):
            continue
        model_path = model_path.strip()
        if not model_path:
            continue
        if model_path not in models:
            models.append(model_path)

    last_selected = data.get('last_selected', '')
    if not isinstance(last_selected, str):
        last_selected = ''
    if last_selected not in models:
        last_selected = ''

    return {'models': models, 'last_selected': last_selected}, None


def save_model_registry(registry: Dict[str, Any]) -> None:
    models = []
    for model_path in registry.get('models', []):
        if not isinstance(model_path, str):
            continue
        model_path = model_path.strip()
        if not model_path:
            continue
        if model_path not in models:
            models.append(model_path)

    last_selected = registry.get('last_selected', '')
    if not isinstance(last_selected, str):
        last_selected = ''
    if last_selected not in models:
        last_selected = ''

    write_yaml_config(
        str(MODEL_REGISTRY_YAML),
        {'models': models, 'last_selected': last_selected},
        update=False
    )


def _resolve_config_from_model(model_path: Path) -> Path:
    return model_path.parent / 'config.yaml'


def _get_time_resolution_from_config(config_path: Path) -> float:
    try:
        config = read_yaml_config(str(config_path))
        dataset_conf = config.get('dataset_conf', {}) if isinstance(config, dict) else {}
        fbank_conf = dataset_conf.get('fbank_conf', {}) if isinstance(dataset_conf, dict) else {}
        frame_shift_ms = float(fbank_conf.get('frame_shift', 10))
        frame_skip = int(dataset_conf.get('frame_skip', 1))
        if frame_skip <= 0:
            frame_skip = 1
        return max(frame_shift_ms * frame_skip / 1000.0, 1e-3)
    except Exception:
        return FRAME_SHIFT_SECONDS


@st.cache_data(show_spinner=False)
def run_single_audio_inference(model_path: str,
                               model_mtime: float,
                               config_mtime: float,
                               utt_id: str,
                               wav_path: str,
                               text_content: str,
                               duration: float) -> Dict[str, Any]:
    del model_mtime, config_mtime
    model_path_obj = Path(model_path)
    config_path = _resolve_config_from_model(model_path_obj)
    time_resolution_sec = _get_time_resolution_from_config(config_path)
    temp_dir = Path(tempfile.mkdtemp(prefix='webui_kws_infer_'))
    test_data_path = temp_dir / 'single_data.list'
    score_file = temp_dir / 'score.txt'
    decode_file = temp_dir / 'decode.jsonl'

    test_obj = {
        'key': utt_id,
        'wav': wav_path,
        'txt': text_content or '',
        'duration': float(duration) if duration is not None else 0.0
    }

    try:
        with open(test_data_path, 'w', encoding='utf8') as fout:
            fout.write(json.dumps(test_obj, ensure_ascii=False) + '\n')

        cmd = [
            sys.executable,
            str(SCORE_CTC_SCRIPT),
            '--config', str(config_path),
            '--test_data', str(test_data_path),
            '--gpu', '-1',
            '--batch_size', '1',
            '--num_workers', '1',
            '--dict', str(INFER_DICT_DIR),
            '--score_file', str(score_file),
            '--keywords', INFER_KEYWORDS_ESCAPED,
            '--token_file', str(INFER_TOKEN_FILE),
            '--lexicon_file', str(INFER_LEXICON_FILE),
            '--checkpoint', str(model_path_obj),
            '--dump_decode_file', str(decode_file)
        ]

        if model_path_obj.suffix == '.zip':
            cmd.append('--jit_model')

        env = os.environ.copy()
        old_pythonpath = env.get('PYTHONPATH', '')
        if old_pythonpath:
            env['PYTHONPATH'] = f"{REPO_ROOT}:{old_pythonpath}"
        else:
            env['PYTHONPATH'] = str(REPO_ROOT)

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=180
        )
        if result.returncode != 0:
            return {
                'ok': False,
                'error': f'推理失败(returncode={result.returncode})',
                'stderr': result.stderr,
                'stdout': result.stdout,
            }

        decode_lines = []
        if decode_file.exists():
            with open(decode_file, 'r', encoding='utf8') as fin:
                decode_lines = [line.strip() for line in fin if line.strip()]

        if not decode_lines:
            return {
                'ok': False,
                'error': '推理完成但未生成解码结果',
                'stderr': result.stderr,
                'stdout': result.stdout,
            }

        decode_obj = json.loads(decode_lines[0])
        kws_result = decode_obj.get('kws_result', {}) if isinstance(decode_obj, dict) else {}

        triggered = bool(kws_result.get('triggered'))
        keyword = kws_result.get('keyword')
        start_frame = kws_result.get('start_frame')
        end_frame = kws_result.get('end_frame')

        start_time_sec = None
        end_time_sec = None
        if start_frame is not None:
            start_time_sec = float(start_frame) * time_resolution_sec
        if end_frame is not None:
            end_time_sec = float(end_frame) * time_resolution_sec

        return {
            'ok': True,
            'triggered': triggered,
            'keyword': keyword,
            'score': kws_result.get('score'),
            'start_frame': start_frame,
            'end_frame': end_frame,
            'time_resolution_sec': time_resolution_sec,
            'start_time_sec': start_time_sec,
            'end_time_sec': end_time_sec,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': '推理超时（超过180秒）'}
    except Exception as e:
        return {'ok': False, 'error': f'推理异常: {e}'}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def query_audio_files(conn, filters=None, search_text="", limit=100, offset=0):
    """
    查询音频文件
    """
    query = "SELECT * FROM audio_metadata WHERE 1=1"
    params = []
    
    # 基本搜索
    if search_text:
        query += " AND (utt_id LIKE ? OR text_content LIKE ?)"
        params.extend([f"%{search_text}%", f"%{search_text}%"])
    
    # 高级筛选
    if filters:
        # 数据集分割
        if filters.get('splits'):
            placeholders = ','.join(['?' for _ in filters['splits']])
            query += f" AND split IN ({placeholders})"
            params.extend(filters['splits'])
        
        # 标签类型
        if filters.get('label_types'):
            placeholders = ','.join(['?' for _ in filters['label_types']])
            query += f" AND label_type IN ({placeholders})"
            params.extend(filters['label_types'])
        
        # 性别
        if filters.get('genders'):
            placeholders = ','.join(['?' for _ in filters['genders']])
            query += f" AND gender IN ({placeholders})"
            params.extend(filters['genders'])
        
        # 年龄范围
        if filters.get('age_min') is not None:
            query += " AND age >= ?"
            params.append(filters['age_min'])
        if filters.get('age_max') is not None:
            query += " AND age <= ?"
            params.append(filters['age_max'])
        
        # 年龄精确值
        if filters.get('age_exact') is not None:
            query += " AND age = ?"
            params.append(filters['age_exact'])
        
        # 距离
        if filters.get('distances'):
            placeholders = ','.join(['?' for _ in filters['distances']])
            query += f" AND distance IN ({placeholders})"
            params.extend(filters['distances'])
        
        # 噪声音量范围
        if filters.get('noise_min') is not None:
            query += " AND CAST(noise_volume AS INTEGER) >= ?"
            params.append(filters['noise_min'])
        if filters.get('noise_max') is not None:
            query += " AND CAST(noise_volume AS INTEGER) <= ?"
            params.append(filters['noise_max'])
        
        # 噪声音量精确值
        if filters.get('noise_exact') is not None:
            query += " AND noise_volume = ?"
            params.append(str(filters['noise_exact']))
        
        # 噪声类型
        if filters.get('noise_types'):
            placeholders = ','.join(['?' for _ in filters['noise_types']])
            query += f" AND noise_type IN ({placeholders})"
            params.extend(filters['noise_types'])
        
        # 角度
        if filters.get('angles'):
            placeholders = ','.join(['?' for _ in filters['angles']])
            query += f" AND angle IN ({placeholders})"
            params.extend(filters['angles'])
    
    # 排序和分页
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor = conn.cursor()
    cursor.execute(query, params)
    
    return cursor.fetchall()


def count_audio_files(conn, filters=None, search_text=""):
    """
    统计符合条件的音频文件数量
    """
    query = "SELECT COUNT(*) FROM audio_metadata WHERE 1=1"
    params = []
    
    # 基本搜索
    if search_text:
        query += " AND (utt_id LIKE ? OR text_content LIKE ?)"
        params.extend([f"%{search_text}%", f"%{search_text}%"])
    
    # 高级筛选（同上）
    if filters:
        if filters.get('splits'):
            placeholders = ','.join(['?' for _ in filters['splits']])
            query += f" AND split IN ({placeholders})"
            params.extend(filters['splits'])
        
        if filters.get('label_types'):
            placeholders = ','.join(['?' for _ in filters['label_types']])
            query += f" AND label_type IN ({placeholders})"
            params.extend(filters['label_types'])
        
        if filters.get('genders'):
            placeholders = ','.join(['?' for _ in filters['genders']])
            query += f" AND gender IN ({placeholders})"
            params.extend(filters['genders'])
        
        if filters.get('age_min') is not None:
            query += " AND age >= ?"
            params.append(filters['age_min'])
        if filters.get('age_max') is not None:
            query += " AND age <= ?"
            params.append(filters['age_max'])
        
        if filters.get('age_exact') is not None:
            query += " AND age = ?"
            params.append(filters['age_exact'])
        
        if filters.get('distances'):
            placeholders = ','.join(['?' for _ in filters['distances']])
            query += f" AND distance IN ({placeholders})"
            params.extend(filters['distances'])
        
        if filters.get('noise_min') is not None:
            query += " AND CAST(noise_volume AS INTEGER) >= ?"
            params.append(filters['noise_min'])
        if filters.get('noise_max') is not None:
            query += " AND CAST(noise_volume AS INTEGER) <= ?"
            params.append(filters['noise_max'])
        
        if filters.get('noise_exact') is not None:
            query += " AND noise_volume = ?"
            params.append(str(filters['noise_exact']))
        
        if filters.get('noise_types'):
            placeholders = ','.join(['?' for _ in filters['noise_types']])
            query += f" AND noise_type IN ({placeholders})"
            params.extend(filters['noise_types'])
        
        if filters.get('angles'):
            placeholders = ','.join(['?' for _ in filters['angles']])
            query += f" AND angle IN ({placeholders})"
            params.extend(filters['angles'])
    
    cursor = conn.cursor()
    cursor.execute(query, params)
    
    return cursor.fetchone()[0]


def get_filter_options(conn):
    """
    获取所有可用的筛选选项
    """
    cursor = conn.cursor()
    
    options = {}
    
    # 数据集分割
    cursor.execute("SELECT DISTINCT split FROM audio_metadata ORDER BY split")
    options['splits'] = [row[0] for row in cursor.fetchall()]
    
    # 标签类型
    cursor.execute("SELECT DISTINCT label_type FROM audio_metadata ORDER BY label_type")
    options['label_types'] = [row[0] for row in cursor.fetchall()]
    
    # 性别
    cursor.execute("SELECT DISTINCT gender FROM audio_metadata WHERE gender IS NOT NULL AND gender != '' ORDER BY gender")
    options['genders'] = [row[0] for row in cursor.fetchall()]
    
    # 距离
    cursor.execute("SELECT DISTINCT distance FROM audio_metadata WHERE distance IS NOT NULL AND distance != '' ORDER BY distance")
    options['distances'] = [row[0] for row in cursor.fetchall()]
    
    # 噪声类型
    cursor.execute("SELECT DISTINCT noise_type FROM audio_metadata WHERE noise_type IS NOT NULL AND noise_type != '' ORDER BY noise_type")
    options['noise_types'] = [row[0] for row in cursor.fetchall()]
    
    # 角度
    cursor.execute("SELECT DISTINCT angle FROM audio_metadata WHERE angle IS NOT NULL AND angle != '' ORDER BY angle")
    options['angles'] = [row[0] for row in cursor.fetchall()]
    
    # 年龄范围
    cursor.execute("SELECT MIN(age), MAX(age) FROM audio_metadata WHERE age IS NOT NULL")
    age_range = cursor.fetchone()
    options['age_range'] = (age_range[0], age_range[1]) if age_range[0] is not None else (0, 100)
    
    # 噪声音量范围
    cursor.execute("SELECT DISTINCT noise_volume FROM audio_metadata WHERE noise_volume IS NOT NULL AND noise_volume != '' ORDER BY CAST(noise_volume AS INTEGER)")
    noise_volumes = [row[0] for row in cursor.fetchall()]
    if noise_volumes:
        try:
            noise_values = [int(v) for v in noise_volumes if v.isdigit()]
            options['noise_range'] = (min(noise_values), max(noise_values)) if noise_values else (0, 50)
        except:
            options['noise_range'] = (0, 50)
    else:
        options['noise_range'] = (0, 50)
    
    return options


# ==================== 可视化 ====================
def generate_visualization(audio_id):
    """
    生成音频特征可视化图片
    """
    output_file = VISUALIZE_DIR / f"{audio_id}_features.png"
    
    # 如果图片已存在，直接返回
    if output_file.exists():
        return output_file
    
    # 运行可视化脚本
    try:
        cmd = [
            sys.executable,
            str(VISUALIZE_SCRIPT),
            audio_id,
            "--no-show"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0 and output_file.exists():
            return output_file
        else:
            st.error(f"生成可视化失败: {result.stderr}")
            return None
    
    except Exception as e:
        st.error(f"生成可视化时出错: {e}")
        return None


# ==================== UI 组件 ====================
def render_sidebar(filter_options):
    """
    渲染侧边栏（筛选器）
    """
    st.sidebar.title("🔍 搜索与筛选")
    
    # 基本搜索
    search_text = st.sidebar.text_input(
        "🔎 搜索音频ID或文本",
        placeholder="输入音频ID或文本内容...",
        help="支持模糊搜索"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 高级筛选")
    
    filters = {}
    
    # 数据集分割
    with st.sidebar.expander("📁 数据集", expanded=True):
        splits = st.multiselect(
            "选择数据集",
            options=filter_options['splits'],
            default=filter_options['splits'],
            help="train/dev/test"
        )
        # 必须至少选择一个数据集
        if splits:
            filters['splits'] = splits
        else:
            st.sidebar.warning("⚠️ 请至少选择一个数据集")
    
    # 标签类型
    with st.sidebar.expander("🏷️ 标签类型", expanded=True):
        label_types = st.multiselect(
            "选择标签类型",
            options=filter_options['label_types'],
            default=filter_options['label_types'],
            help="positive/negative"
        )
        # 必须至少选择一个标签类型
        if label_types:
            filters['label_types'] = label_types
        else:
            st.sidebar.warning("⚠️ 请至少选择一个标签类型")
    
    # 性别
    with st.sidebar.expander("👤 性别", expanded=False):
        genders = st.multiselect(
            "选择性别",
            options=filter_options['genders'],
            help="f=女性, m=男性"
        )
        if genders:
            filters['genders'] = genders
    
    # 年龄
    with st.sidebar.expander("🎂 年龄", expanded=False):
        enable_age_filter = st.checkbox("启用年龄筛选", value=False)
        
        if enable_age_filter:
            age_filter_type = st.radio(
                "年龄筛选方式",
                options=["范围", "精确值"],
                horizontal=True
            )
            
            if age_filter_type == "范围":
                age_range = st.slider(
                    "年龄范围",
                    min_value=int(filter_options['age_range'][0]),
                    max_value=int(filter_options['age_range'][1]),
                    value=(int(filter_options['age_range'][0]), int(filter_options['age_range'][1]))
                )
                filters['age_min'] = age_range[0]
                filters['age_max'] = age_range[1]
            else:
                age_exact = st.number_input(
                    "精确年龄",
                    min_value=int(filter_options['age_range'][0]),
                    max_value=int(filter_options['age_range'][1]),
                    value=None,
                    placeholder="输入年龄..."
                )
                if age_exact is not None:
                    filters['age_exact'] = age_exact
    
    # 距离
    with st.sidebar.expander("📏 距离", expanded=False):
        distances = st.multiselect(
            "选择距离",
            options=filter_options['distances'],
            help="说话人与麦克风的距离"
        )
        if distances:
            filters['distances'] = distances
    
    # 噪声音量
    with st.sidebar.expander("🔊 噪声音量", expanded=False):
        enable_noise_filter = st.checkbox("启用噪声筛选", value=False)
        
        if enable_noise_filter:
            noise_filter_type = st.radio(
                "噪声筛选方式",
                options=["范围", "精确值"],
                horizontal=True
            )
            
            if noise_filter_type == "范围":
                noise_range = st.slider(
                    "噪声音量范围 (dB)",
                    min_value=int(filter_options['noise_range'][0]),
                    max_value=int(filter_options['noise_range'][1]),
                    value=(int(filter_options['noise_range'][0]), int(filter_options['noise_range'][1]))
                )
                filters['noise_min'] = noise_range[0]
                filters['noise_max'] = noise_range[1]
            else:
                noise_exact = st.number_input(
                    "精确噪声音量 (dB)",
                    min_value=int(filter_options['noise_range'][0]),
                    max_value=int(filter_options['noise_range'][1]),
                    value=None,
                    placeholder="输入噪声音量..."
                )
                if noise_exact is not None:
                    filters['noise_exact'] = noise_exact
    
    # 噪声类型
    with st.sidebar.expander("📢 噪声类型", expanded=False):
        noise_types = st.multiselect(
            "选择噪声类型",
            options=filter_options['noise_types'],
            help="背景噪声的类型"
        )
        if noise_types:
            filters['noise_types'] = noise_types
    
    # 角度
    with st.sidebar.expander("🎯 角度", expanded=False):
        angles = st.multiselect(
            "选择角度",
            options=filter_options['angles'],
            help="说话人相对麦克风的角度"
        )
        if angles:
            filters['angles'] = angles
    
    # 重置按钮
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 重置所有筛选", width='stretch'):
        st.rerun()

    # 显示当前筛选摘要
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 当前筛选")
    
    filter_count = 0
    if search_text:
        st.sidebar.text(f"🔎 搜索: {search_text[:20]}...")
        filter_count += 1
    
    if filters:
        if 'splits' in filters:
            st.sidebar.text(f"📁 数据集: {len(filters['splits'])}个")
            filter_count += 1
        if 'label_types' in filters:
            st.sidebar.text(f"🏷️ 标签: {len(filters['label_types'])}个")
            filter_count += 1
        if 'genders' in filters:
            st.sidebar.text(f"👤 性别: {len(filters['genders'])}个")
            filter_count += 1
        if 'distances' in filters:
            st.sidebar.text(f"📏 距离: {len(filters['distances'])}个")
            filter_count += 1
        if 'noise_types' in filters:
            st.sidebar.text(f"📢 噪声: {len(filters['noise_types'])}个")
            filter_count += 1
    
    if filter_count == 0:
        st.sidebar.text("无筛选条件")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 指定模型推理")
    st.sidebar.caption(f"模型列表持久化: `{MODEL_REGISTRY_YAML}`")

    registry, registry_error = load_model_registry()
    if registry_error:
        st.sidebar.warning(f"读取模型列表失败，使用空列表: {registry_error}")
    model_list = registry.get('models', [])

    model_options = [''] + model_list
    default_index = 0
    last_selected = registry.get('last_selected', '')
    if last_selected in model_options:
        default_index = model_options.index(last_selected)

    selected_model = st.sidebar.selectbox(
        "推理模型",
        options=model_options,
        index=default_index,
        format_func=lambda value: "（默认空，不做推理）" if value == '' else value,
        help="每条选中的音频会用该模型推理，并在波形图上标注唤醒时刻"
    )

    new_model_raw = st.sidebar.text_input(
        "新增模型路径",
        placeholder="如: exp/fsmn_ctc_distill_v2_a64_p32_l2/299.pt"
    )
    add_col, remove_col = st.sidebar.columns(2)

    if add_col.button("➕ 添加模型", width='stretch'):
        model_input = new_model_raw.strip()
        if not model_input:
            st.sidebar.error("请先输入模型路径")
        else:
            model_path = _normalize_model_path(model_input)
            if not Path(model_path).exists():
                st.sidebar.error(f"模型不存在: {model_path}")
            elif model_path not in model_list:
                model_list.append(model_path)
                registry['models'] = model_list
                registry['last_selected'] = model_path
                save_model_registry(registry)
                st.rerun()
            else:
                st.sidebar.info("模型已存在于列表")

    if remove_col.button("🗑️ 删除模型", width='stretch', disabled=(selected_model == '')):
        model_list = [path for path in model_list if path != selected_model]
        registry['models'] = model_list
        registry['last_selected'] = ''
        save_model_registry(registry)
        st.rerun()

    if selected_model != registry.get('last_selected', ''):
        registry['last_selected'] = selected_model
        save_model_registry(registry)

    return search_text, filters, selected_model


def render_audio_info(audio_row):
    """
    渲染音频详细信息卡片
    """
    st.subheader("📋 音频详细信息")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**基本信息**")
        st.write(f"**音频ID**: `{audio_row['utt_id']}`")
        st.write(f"**数据集**: {audio_row['split']}")
        st.write(f"**标签类型**: {audio_row['label_type']}")
        st.write(f"**时长**: {audio_row['duration']:.3f} 秒")
        st.write(f"**文本内容**: {audio_row['text_content'] or '无'}")
    
    with col2:
        st.markdown("**录音条件**")
        st.write(f"**距离**: {audio_row['distance'] or '未知'}")
        st.write(f"**角度**: {audio_row['angle'] or '未知'}")
        st.write(f"**噪声音量**: {audio_row['noise_volume'] or '未知'}")
        st.write(f"**噪声类型**: {audio_row['noise_type'] or '未知'}")
    
    with col3:
        st.markdown("**说话人信息**")
        gender = audio_row['gender']
        if gender == 'f':
            gender_text = '女性'
        elif gender == 'm':
            gender_text = '男性'
        elif gender:
            gender_text = gender
        else:
            gender_text = '未知'
        st.write(f"**性别**: {gender_text}")
        
        age = audio_row['age']
        st.write(f"**年龄**: {age if age else '未知'} {'岁' if age else ''}")
        
        speaker_id = audio_row['speaker_id']
        if speaker_id and len(speaker_id) > 16:
            st.write(f"**说话人ID**: `{speaker_id[:16]}...`")
        elif speaker_id:
            st.write(f"**说话人ID**: `{speaker_id}`")
        else:
            st.write(f"**说话人ID**: 未知")
        
        keyword_id = audio_row['keyword_id']
        st.write(f"**关键词ID**: {keyword_id if keyword_id is not None else '未知'}")
    
    # 文件路径
    st.markdown("**文件路径**")
    st.code(audio_row['wav_path'], language='bash')
    
    # 音频播放器
    st.markdown("**🔊 音频播放**")
    wav_path = Path(audio_row['wav_path'])
    if wav_path.exists():
        try:
            st.audio(str(wav_path))
        except Exception as e:
            st.error(f"无法播放音频: {e}")
    else:
        st.warning(f"⚠️ 音频文件不存在: {wav_path}")


def render_visualization(audio_id):
    """
    渲染可视化图片
    """
    st.subheader("🎨 音频特征可视化")
    
    with st.spinner(f"正在生成可视化图表..."):
        viz_file = generate_visualization(audio_id)
    
    if viz_file and viz_file.exists():
        image = Image.open(viz_file)
        st.image(image, width='stretch')
        
        # 下载按钮
        with open(viz_file, 'rb') as f:
            st.download_button(
                label="📥 下载图片",
                data=f,
                file_name=f"{audio_id}_features.png",
                mime="image/png"
            )
    else:
        st.error("❌ 无法生成可视化图表")


def render_waveform_with_detection(audio_row, inference_result):
    st.subheader("📈 音频波形时序图")
    wav_path = Path(audio_row['wav_path'])
    if not wav_path.exists():
        st.warning(f"⚠️ 音频文件不存在: {wav_path}")
        return

    try:
        waveform, sample_rate = torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0).numpy()
    except Exception as e:
        st.error(f"读取音频失败: {e}")
        return

    if waveform.size == 0:
        st.warning("⚠️ 波形为空，无法绘图")
        return

    time_axis = np.arange(waveform.shape[0], dtype=np.float32) / float(sample_rate)
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(time_axis, waveform, linewidth=0.6, color='#1f77b4')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform - {audio_row['utt_id']}")
    ax.grid(alpha=0.3)
    ax.set_xlim([0, float(time_axis[-1]) if len(time_axis) > 0 else 0.0])

    triggered = bool(inference_result.get('triggered', False))
    keyword = inference_result.get('keyword')
    if triggered and isinstance(keyword, str):
        keyword_norm = keyword.replace(' ', '')
        if keyword_norm in INFER_WAKEWORDS:
            display_name = WAKEWORD_DISPLAY.get(keyword_norm, keyword_norm)
            end_time = inference_result.get('end_time_sec')
            color = '#d62728' if keyword_norm == '嗨小问' else '#2ca02c'
            y_max = float(np.max(np.abs(waveform)))
            y_text = y_max * 0.85 if y_max > 1e-6 else 0.1

            if end_time is not None:
                end_time = float(end_time)
                ax.axvline(end_time, color=color, linestyle=':', linewidth=2.0)
                ax.text(
                    end_time,
                    y_text,
                    f"{display_name} end @ {end_time:.2f}s",
                    color=color,
                    rotation=90,
                    verticalalignment='bottom',
                    horizontalalignment='left',
                    fontsize=9,
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'edgecolor': color}
                )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_model_inference(audio_row, selected_model):
    st.subheader("🧠 模型推理结果")
    model_path = Path(selected_model)
    if not model_path.exists():
        st.error(f"模型不存在: {model_path}")
        return

    if model_path.suffix not in {'.pt', '.zip'}:
        st.error("当前仅支持 .pt / .zip 模型推理")
        return

    config_path = _resolve_config_from_model(model_path)
    if not config_path.exists():
        st.error(f"模型配置不存在: {config_path}")
        return

    for file_path in [SCORE_CTC_SCRIPT, INFER_DICT_DIR / 'dict.txt',
                      INFER_TOKEN_FILE, INFER_LEXICON_FILE]:
        if not file_path.exists():
            st.error(f"推理依赖文件不存在: {file_path}")
            return

    model_mtime = model_path.stat().st_mtime
    config_mtime = config_path.stat().st_mtime
    with st.spinner("正在执行单条音频推理..."):
        infer_result = run_single_audio_inference(
            model_path=str(model_path),
            model_mtime=model_mtime,
            config_mtime=config_mtime,
            utt_id=str(audio_row['utt_id']),
            wav_path=str(audio_row['wav_path']),
            text_content=str(audio_row['text_content'] or ''),
            duration=float(audio_row['duration']) if audio_row['duration'] is not None else 0.0,
        )

    if not infer_result.get('ok', False):
        st.error(f"推理失败: {infer_result.get('error', '未知错误')}")
        if infer_result.get('stderr'):
            st.code(infer_result['stderr'], language='bash')
        return

    triggered = bool(infer_result.get('triggered', False))
    keyword = infer_result.get('keyword')
    score = infer_result.get('score')
    end_time = infer_result.get('end_time_sec')

    if triggered:
        st.success(
            f"检测结果: 触发 `{keyword}` (score={score:.3f})"
            if isinstance(score, (int, float)) else
            f"检测结果: 触发 `{keyword}`"
        )
    else:
        st.info("检测结果: 未触发唤醒词")

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("触发状态", "Triggered" if triggered else "Rejected")
    info_col2.metric(
        "置信度",
        f"{score:.3f}" if isinstance(score, (int, float)) else "-"
    )
    info_col3.metric(
        "结束时刻(秒)",
        f"{end_time:.3f}" if isinstance(end_time, (int, float)) else "-"
    )
    resolution = infer_result.get('time_resolution_sec')
    if isinstance(resolution, (int, float)):
        st.caption(f"时间分辨率: {resolution:.3f}s / frame (来自模型 config 的 frame_shift × frame_skip)")
    st.caption("说明：CTC 前缀匹配的起始时刻常偏早，页面默认仅展示结束时刻。")

    render_waveform_with_detection(audio_row, infer_result)


# ==================== 主应用 ====================
def main():
    st.title("🎵 音频数据浏览器")
    st.markdown("**Stage 1 可视化工具** - 交互式浏览、搜索和可视化音频数据集")
    
    # 初始化数据库连接
    conn = get_database_connection()
    
    # 获取筛选选项
    filter_options = get_filter_options(conn)
    
    # 渲染侧边栏
    search_text, filters, selected_model = render_sidebar(filter_options)
    
    # 主内容区
    st.markdown("---")
    
    # 统计信息
    total_count = count_audio_files(conn, filters, search_text)
    
    if total_count > 0:
        st.info(f"📊 找到 **{total_count:,}** 条符合条件的音频记录")
    else:
        st.warning(f"⚠️ 找到 **0** 条符合条件的音频记录")
        
        # 显示当前筛选条件
        with st.expander("🔍 当前筛选条件（点击查看）"):
            if search_text:
                st.write(f"🔎 **搜索文本**: {search_text}")
            
            if filters:
                st.write("**活动的筛选条件**:")
                for key, value in filters.items():
                    key_name = {
                        'splits': '📁 数据集',
                        'label_types': '🏷️ 标签类型',
                        'genders': '👤 性别',
                        'age_min': '🎂 年龄最小',
                        'age_max': '🎂 年龄最大',
                        'age_exact': '🎂 年龄精确',
                        'distances': '📏 距离',
                        'noise_min': '🔊 噪声最小',
                        'noise_max': '🔊 噪声最大',
                        'noise_exact': '🔊 噪声精确',
                        'noise_types': '📢 噪声类型',
                        'angles': '🎯 角度'
                    }.get(key, key)
                    st.write(f"- {key_name}: `{value}`")
            else:
                st.write("**没有应用任何筛选条件**")
            
            st.info("💡 **提示**: 请检查左侧边栏的筛选条件，确保至少选择了一个数据集和一个标签类型。")
    
    # 分页设置
    page_size = st.selectbox("每页显示数量", options=[10, 25, 50, 100], index=1)
    
    # 初始化页码
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    
    total_pages = max((total_count + page_size - 1) // page_size, 1)
    if st.session_state.current_page >= total_pages:
        st.session_state.current_page = total_pages - 1

    # 查询数据
    offset = st.session_state.current_page * page_size
    results = query_audio_files(conn, filters, search_text, limit=page_size, offset=offset)
    
    if not results:
        st.warning("⚠️ 没有找到符合条件的音频")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame([dict(row) for row in results])
    
    # 显示音频列表
    st.subheader(f"📁 音频列表 (第 {st.session_state.current_page + 1} 页)")
    
    # 创建可点击的音频列表
    display_df = df[[
        'utt_id', 'split', 'label_type', 'duration', 
        'text_content', 'gender', 'age', 'distance', 'noise_volume'
    ]].copy()
    
    display_df.columns = [
        '音频ID', '数据集', '标签', '时长(s)', 
        '文本', '性别', '年龄', '距离', '噪声(dB)'
    ]
    
    # 显示表格
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True
    )
    
    # 分页控制
    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
    
    with col1:
        if st.button("⬅️ 上一页", disabled=(st.session_state.current_page == 0)):
            st.session_state.current_page -= 1
            st.rerun()
    
    with col2:
        if st.button("➡️ 下一页", disabled=(st.session_state.current_page >= total_pages - 1)):
            st.session_state.current_page += 1
            st.rerun()
    
    with col3:
        st.markdown(f"<center>第 {st.session_state.current_page + 1} / {total_pages} 页</center>", unsafe_allow_html=True)
    
    with col4:
        if st.button("🔝 回到首页"):
            st.session_state.current_page = 0
            st.rerun()

    jump_hint_col, jump_col1, jump_col2, _ = st.columns([0.8, 0.9, 0.6, 7.7])
    with jump_hint_col:
        st.markdown("**跳页**")
    with jump_col1:
        jump_page = st.number_input(
            "跳转到页码",
            min_value=1,
            max_value=total_pages,
            value=st.session_state.current_page + 1,
            step=1,
            key='jump_to_page_number',
            label_visibility='collapsed'
        )
    with jump_col2:
        if st.button("🔢 跳转", width='stretch'):
            st.session_state.current_page = int(jump_page) - 1
            st.rerun()
    
    # 选择音频查看详情
    st.markdown("---")
    st.subheader("🔍 查看音频详情")
    
    audio_ids = df['utt_id'].tolist()
    selected_audio_id = st.selectbox(
        "选择一个音频查看详情和可视化",
        options=audio_ids,
        format_func=lambda x: f"{x} - {df[df['utt_id']==x]['text_content'].values[0]}"
    )
    
    if selected_audio_id:
        st.markdown("---")
        
        # 获取选中音频的完整信息
        selected_row = df[df['utt_id'] == selected_audio_id].iloc[0]
        
        # 渲染音频信息
        render_audio_info(selected_row)
        
        st.markdown("---")
        
        # 渲染可视化
        render_visualization(selected_audio_id)

        st.markdown("---")

        if selected_model:
            render_model_inference(selected_row, selected_model)
        else:
            st.info("未选择推理模型：仅显示可视化。可在左侧“指定模型推理”中添加并选择模型。")


if __name__ == "__main__":
    main()
