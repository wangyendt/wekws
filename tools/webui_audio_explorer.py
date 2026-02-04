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
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image

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
    
    return search_text, filters


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


# ==================== 主应用 ====================
def main():
    st.title("🎵 音频数据浏览器")
    st.markdown("**Stage 1 可视化工具** - 交互式浏览、搜索和可视化音频数据集")
    
    # 初始化数据库连接
    conn = get_database_connection()
    
    # 获取筛选选项
    filter_options = get_filter_options(conn)
    
    # 渲染侧边栏
    search_text, filters = render_sidebar(filter_options)
    
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
    
    total_pages = (total_count + page_size - 1) // page_size
    
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


if __name__ == "__main__":
    main()
