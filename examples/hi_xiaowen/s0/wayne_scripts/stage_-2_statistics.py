#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计脚本：
1. 统计 wav 文件数量和时长
2. 统计 JSON 文件中的各种分布（train/dev/test）

优化说明：
- 使用多线程加速IO密集型任务（读取WAV文件）
- 自动检测CPU核心数，合理设置线程数
"""

import os
import json
import wave
import sys
from collections import Counter, defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from pywayne.tools import read_yaml_config

# 设置默认编码为 UTF-8，兼容 Mac 和 Linux
if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
    import locale
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

# 全局锁用于线程安全的进度显示
print_lock = Lock()

# 检测是否有GUI环境
HAS_GUI = False
try:
    import matplotlib
    # 检测是否有显示环境
    if os.environ.get('DISPLAY') or sys.platform == 'darwin' or sys.platform == 'win32':
        # 尝试使用默认后端
        try:
            matplotlib.use('TkAgg')  # 尝试使用交互式后端
            import matplotlib.pyplot as plt
            import seaborn as sns
            HAS_GUI = True
            print("检测到GUI环境，将生成可视化图表")
        except:
            # 如果失败，使用非交互式后端保存图片
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            HAS_GUI = True
            print("检测到无显示环境，图表将保存为文件")
    else:
        print("检测到headless环境，跳过图表生成")
except ImportError:
    print("未安装matplotlib/seaborn，跳过图表生成")


def get_wav_duration(wav_path):
    """获取 wav 文件的时长（秒）"""
    try:
        with wave.open(wav_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"警告: 无法读取 {wav_path}: {e}", file=sys.stderr)
        return 0


def format_duration(seconds):
    """格式化时长显示"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.2f}s"


def get_duration_range(duration):
    """将时长分段"""
    if duration < 1:
        return "<1s"
    elif duration < 2:
        return "1-2s"
    elif duration < 3:
        return "2-3s"
    elif duration < 5:
        return "3-5s"
    else:
        return ">=5s"


def process_wav_batch(wav_files, progress_counter, total_count, report_interval=10000):
    """批量处理WAV文件（多线程任务）"""
    durations = {}  # 返回字典：{filename: duration}
    local_count = 0
    
    for wav_file in wav_files:
        duration = get_wav_duration(str(wav_file))
        # 使用文件名（不含扩展名）作为key
        filename = wav_file.stem
        durations[filename] = duration
        local_count += 1
        
        # 减少锁竞争：批量更新进度
        if local_count % 100 == 0:
            with print_lock:
                progress_counter[0] += 100
                if progress_counter[0] % report_interval == 0:
                    print(f"  已处理: {progress_counter[0]}/{total_count}")
    
    # 处理剩余的计数
    remainder = local_count % 100
    if remainder > 0:
        with print_lock:
            progress_counter[0] += remainder
    
    return durations


def statistics_wav_files(wav_dir, max_workers=None):
    """统计 WAV 文件（多线程加速），返回时长缓存字典"""
    print("=" * 80)
    print("【1】WAV 文件统计与缓存构建")
    print("=" * 80)
    
    wav_files = list(Path(wav_dir).glob("*.wav"))
    total_count = len(wav_files)
    
    if total_count == 0:
        print(f"未找到 WAV 文件: {wav_dir}")
        return {}
    
    # 自动设置线程数：CPU核心数的2-4倍（IO密集型任务）
    if max_workers is None:
        max_workers = min(os.cpu_count() * 3, 32)
    
    print(f"\n目录: {wav_dir}")
    print(f"文件总数: {total_count}")
    print(f"线程数: {max_workers}")
    
    # 统计时长并缓存
    print("\n正在统计音频时长并缓存（多线程处理）...")
    start_time = time.time()
    
    # 分批处理
    batch_size = max(1, total_count // max_workers)
    batches = [wav_files[i:i + batch_size] for i in range(0, total_count, batch_size)]
    
    progress_counter = [0]  # 使用列表以便在线程间共享
    duration_cache = {}  # 缓存：{filename: duration}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_wav_batch, batch, progress_counter, total_count)
            for batch in batches
        ]
        
        for future in as_completed(futures):
            batch_durations = future.result()
            duration_cache.update(batch_durations)
    
    elapsed_time = time.time() - start_time
    
    # 统计信息
    durations = list(duration_cache.values())
    total_duration = sum(durations)
    avg_duration = total_duration / total_count if total_count > 0 else 0
    
    # 时长分布
    duration_ranges = Counter([get_duration_range(d) for d in durations])
    
    print(f"\n处理完成，耗时: {elapsed_time:.2f}秒（已缓存{len(duration_cache)}个文件的时长）")
    print(f"总时长: {format_duration(total_duration)} ({total_duration:.2f}s)")
    print(f"平均时长: {avg_duration:.2f}s")
    print(f"最短时长: {min(durations):.2f}s")
    print(f"最长时长: {max(durations):.2f}s")
    
    print("\n时长分布:")
    for duration_range in ["<1s", "1-2s", "2-3s", "3-5s", ">=5s"]:
        count = duration_ranges.get(duration_range, 0)
        percentage = (count / total_count * 100) if total_count > 0 else 0
        print(f"  {duration_range:8s}: {count:6d} ({percentage:5.2f}%)")
    
    return duration_cache


def process_sample_batch_with_cache(samples, duration_cache, progress_counter, total_samples, report_interval=10000):
    """批量处理样本（使用缓存，避免重复读取文件）"""
    durations = []
    valid_count = 0
    local_count = 0
    
    for sample in samples:
        utt_id = sample.get('utt_id', '')
        
        # 从缓存中获取时长
        if utt_id in duration_cache:
            duration = duration_cache[utt_id]
            durations.append(duration)
            valid_count += 1
        
        local_count += 1
        
        # 减少锁竞争：批量更新进度
        if local_count % 100 == 0:
            with print_lock:
                progress_counter[0] += 100
                if progress_counter[0] % report_interval == 0:
                    print(f"  已处理: {progress_counter[0]}/{total_samples}")
    
    # 处理剩余的计数
    remainder = local_count % 100
    if remainder > 0:
        with print_lock:
            progress_counter[0] += remainder
    
    return durations, valid_count


def statistics_json_files(json_dir, duration_cache, max_workers=None):
    """统计 JSON 文件中的分布信息（使用缓存，避免重复读取）"""
    print("\n\n" + "=" * 80)
    print("【2】JSON 文件统计（train/dev/test 分组，使用缓存）")
    print("=" * 80)
    
    # 自动设置线程数
    if max_workers is None:
        max_workers = min(os.cpu_count() * 3, 32)
    
    # 分组文件
    groups = {
        'train': ['p_train.json', 'n_train.json'],
        'dev': ['p_dev.json', 'n_dev.json'],
        'test': ['p_test.json', 'n_test.json']
    }
    
    for group_name, json_files in groups.items():
        print(f"\n{'=' * 80}")
        print(f"【{group_name.upper()}】数据集统计")
        print("=" * 80)
        
        all_samples = []
        
        # 读取所有 JSON 文件
        for json_file in json_files:
            json_path = Path(json_dir) / json_file
            if not json_path.exists():
                print(f"警告: 文件不存在 {json_path}")
                continue
            
            print(f"\n正在读取: {json_file}")
            with open(json_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
                all_samples.extend(samples)
                print(f"  样本数: {len(samples)}")
        
        if not all_samples:
            print(f"未找到 {group_name} 数据")
            continue
        
        total_samples = len(all_samples)
        print(f"\n总样本数: {total_samples}")
        print(f"线程数: {max_workers}")
        
        # 从缓存中提取时长（快速）
        print("\n正在从缓存中提取时长信息（无需重新读取文件）...")
        start_time = time.time()
        
        # 分批处理
        batch_size = max(1, total_samples // max_workers)
        batches = [all_samples[i:i + batch_size] for i in range(0, total_samples, batch_size)]
        
        progress_counter = [0]
        durations = []
        valid_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_sample_batch_with_cache, batch, duration_cache, progress_counter, total_samples)
                for batch in batches
            ]
            
            for future in as_completed(futures):
                batch_durations, batch_valid = future.result()
                durations.extend(batch_durations)
                valid_count += batch_valid
        
        elapsed_time = time.time() - start_time
        
        if durations:
            total_duration = sum(durations)
            avg_duration = total_duration / len(durations)
            
            print(f"\n处理完成，耗时: {elapsed_time:.2f}秒")
            print(f"\n【时长统计】")
            print(f"  有效音频数: {valid_count}/{total_samples}")
            print(f"  总时长: {format_duration(total_duration)} ({total_duration:.2f}s)")
            print(f"  平均时长: {avg_duration:.2f}s")
            print(f"  最短时长: {min(durations):.2f}s")
            print(f"  最长时长: {max(durations):.2f}s")
            
            # 时长分布
            duration_ranges = Counter([get_duration_range(d) for d in durations])
            print("\n  时长分布:")
            for duration_range in ["<1s", "1-2s", "2-3s", "3-5s", ">=5s"]:
                count = duration_ranges.get(duration_range, 0)
                percentage = (count / len(durations) * 100) if len(durations) > 0 else 0
                print(f"    {duration_range:8s}: {count:6d} ({percentage:5.2f}%)")
        
        # 统计各种分布
        distance_dist = Counter([s.get('distance', 'unknown') for s in all_samples])
        angle_dist = Counter([s.get('angle', 'unknown') for s in all_samples])
        noise_volume_dist = Counter([s.get('noise_volume', 'unknown') for s in all_samples])
        gender_dist = Counter([s.get('gender', 'unknown') for s in all_samples])
        age_dist = Counter([s.get('age', 'unknown') for s in all_samples])
        noise_type_dist = Counter([s.get('noise_type', 'unknown') for s in all_samples])
        
        # 统计 speaker 数量
        speaker_ids = set([s.get('speaker_id', '') for s in all_samples])
        speaker_num = len(speaker_ids)
        
        # 打印分布
        print(f"\n【距离分布】")
        for key, count in sorted(distance_dist.items()):
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"  {key:10s}: {count:6d} ({percentage:5.2f}%)")
        
        print(f"\n【角度分布】")
        for key, count in sorted(angle_dist.items()):
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"  {key:10s}: {count:6d} ({percentage:5.2f}%)")
        
        print(f"\n【噪声音量分布】")
        for key, count in sorted(noise_volume_dist.items()):
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"  {key:10s}: {count:6d} ({percentage:5.2f}%)")
        
        print(f"\n【性别分布】")
        gender_map = {'m': '男', 'f': '女', 'unknown': '未知'}
        for key, count in sorted(gender_dist.items()):
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            label = gender_map.get(key, key)
            print(f"  {label:10s}: {count:6d} ({percentage:5.2f}%)")
        
        print(f"\n【年龄分布】")
        for key, count in sorted(age_dist.items()):
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"  {key:10s}: {count:6d} ({percentage:5.2f}%)")
        
        print(f"\n【噪声类型分布】")
        for key, count in sorted(noise_type_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"  {key:15s}: {count:6d} ({percentage:5.2f}%)")
        
        print(f"\n【说话人统计】")
        print(f"  说话人总数: {speaker_num}")
        print(f"  平均每人样本数: {total_samples / speaker_num:.2f}" if speaker_num > 0 else "  平均每人样本数: 0")


def plot_statistics(json_dir, duration_cache, output_dir):
    """生成统计图表（使用缓存数据）"""
    if not HAS_GUI:
        return
    
    print("\n" + "=" * 80)
    print("【3】生成可视化图表（使用缓存数据）")
    print("=" * 80)
    
    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    
    # 读取所有数据
    groups = {
        'train': ['p_train.json', 'n_train.json'],
        'dev': ['p_dev.json', 'n_dev.json'],
        'test': ['p_test.json', 'n_test.json']
    }
    
    all_data = {}
    for group_name, json_files in groups.items():
        samples = []
        for json_file in json_files:
            json_path = Path(json_dir) / json_file
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    samples.extend(json.load(f))
        all_data[group_name] = samples
    
    # 1. 绘制时长分布对比图（使用缓存）
    print("\n生成时长分布图...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Audio Duration Distribution', fontsize=16, fontweight='bold')
    
    for idx, (group_name, samples) in enumerate(all_data.items()):
        durations = []
        for sample in samples[:10000]:  # 限制样本数以加快速度
            utt_id = sample.get('utt_id', '')
            if utt_id in duration_cache:
                durations.append(duration_cache[utt_id])
        
        ax = axes[idx // 2, idx % 2]
        ax.hist(durations, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Duration (seconds)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{group_name.upper()} Set (n={len(durations)})', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 总体时长分布
    ax = axes[1, 1]
    for group_name, samples in all_data.items():
        durations = []
        for sample in samples[:5000]:
            utt_id = sample.get('utt_id', '')
            if utt_id in duration_cache:
                durations.append(duration_cache[utt_id])
        ax.hist(durations, bins=50, alpha=0.5, label=group_name.upper(), edgecolor='black')
    
    ax.set_xlabel('Duration (seconds)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Overall Distribution Comparison', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    duration_plot = output_dir / 'duration_distribution.png'
    plt.savefig(duration_plot, dpi=150, bbox_inches='tight')
    print(f"  已保存: {duration_plot}")
    plt.close()
    
    # 2. 绘制样本数量对比图
    print("\n生成样本数量对比图...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 总样本数
    sample_counts = {name: len(samples) for name, samples in all_data.items()}
    ax = axes[0]
    bars = ax.bar(sample_counts.keys(), sample_counts.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    ax.set_title('Sample Count by Dataset', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    # 在柱子上添加数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 正负样本分布
    ax = axes[1]
    pos_samples = []
    neg_samples = []
    for group_name, json_files in groups.items():
        pos_file = [f for f in json_files if f.startswith('p_')][0]
        neg_file = [f for f in json_files if f.startswith('n_')][0]
        
        pos_path = Path(json_dir) / pos_file
        neg_path = Path(json_dir) / neg_file
        
        if pos_path.exists():
            with open(pos_path, 'r', encoding='utf-8') as f:
                pos_samples.append(len(json.load(f)))
        else:
            pos_samples.append(0)
        
        if neg_path.exists():
            with open(neg_path, 'r', encoding='utf-8') as f:
                neg_samples.append(len(json.load(f)))
        else:
            neg_samples.append(0)
    
    x = list(range(len(groups)))
    width = 0.35
    ax.bar([i - width/2 for i in x], pos_samples, width, label='Positive', color='#95E1D3', edgecolor='black', linewidth=1.5)
    ax.bar([i + width/2 for i in x], neg_samples, width, label='Negative', color='#F38181', edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    ax.set_title('Positive vs Negative Samples', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([name.upper() for name in groups.keys()])
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    count_plot = output_dir / 'sample_count_comparison.png'
    plt.savefig(count_plot, dpi=150, bbox_inches='tight')
    print(f"  已保存: {count_plot}")
    plt.close()
    
    # 3. 绘制各种属性分布图（为所有数据集生成）
    print("\n生成属性分布图（Train/Dev/Test）...")
    
    for group_name, samples in all_data.items():
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Feature Distribution ({group_name.upper()} Set)', fontsize=16, fontweight='bold')
        
        # 距离分布
        ax = axes[0, 0]
        distance_dist = Counter([s.get('distance', 'unknown') for s in samples])
        distance_dist = {k: v for k, v in distance_dist.items() if k != 'unknown'}
        if distance_dist:
            ax.bar(distance_dist.keys(), distance_dist.values(), color='#3498db', edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Distance', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title('Distance Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
        
        # 角度分布
        ax = axes[0, 1]
        angle_dist = Counter([s.get('angle', 'unknown') for s in samples])
        angle_dist = {k: v for k, v in angle_dist.items() if k != 'unknown'}
        if angle_dist:
            ax.bar(sorted(angle_dist.keys()), [angle_dist[k] for k in sorted(angle_dist.keys())], 
                   color='#e74c3c', edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Angle', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title('Angle Distribution', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, axis='y', alpha=0.3)
        
        # 噪声音量分布
        ax = axes[0, 2]
        noise_volume_dist = Counter([s.get('noise_volume', 'unknown') for s in samples])
        noise_volume_dist = {k: v for k, v in noise_volume_dist.items() if k != 'unknown'}
        if noise_volume_dist:
            ax.bar(sorted(noise_volume_dist.keys()), [noise_volume_dist[k] for k in sorted(noise_volume_dist.keys())], 
                   color='#9b59b6', edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Noise Volume', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title('Noise Volume Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
        
        # 性别分布
        ax = axes[1, 0]
        gender_dist = Counter([s.get('gender', 'unknown') for s in samples])
        gender_dist = {k: v for k, v in gender_dist.items() if k != 'unknown'}
        if gender_dist:
            gender_map = {'m': 'Male', 'f': 'Female'}
            labels = [gender_map.get(k, k) for k in gender_dist.keys()]
            colors = ['#3498db' if k == 'm' else '#e74c3c' for k in gender_dist.keys()]
            ax.bar(labels, gender_dist.values(), color=colors, edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Gender', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title('Gender Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
        
        # 噪声类型分布
        ax = axes[1, 1]
        noise_type_dist = Counter([s.get('noise_type', 'unknown') for s in samples])
        noise_type_dist = {k: v for k, v in noise_type_dist.items() if k != 'unknown'}
        if noise_type_dist:
            ax.bar(noise_type_dist.keys(), noise_type_dist.values(), color='#1abc9c', edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Noise Type', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title('Noise Type Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
        
        # 年龄分布 (Top 10)
        ax = axes[1, 2]
        age_dist = Counter([s.get('age', 'unknown') for s in samples])
        age_dist = {k: v for k, v in age_dist.items() if k != 'unknown'}
        if age_dist:
            top_ages = dict(sorted(age_dist.items(), key=lambda x: x[1], reverse=True)[:10])
            ax.bar(top_ages.keys(), top_ages.values(), color='#f39c12', edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Age', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title('Age Distribution (Top 10)', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        feature_plot = output_dir / f'feature_distribution_{group_name}.png'
        plt.savefig(feature_plot, dpi=150, bbox_inches='tight')
        print(f"  已保存: {feature_plot}")
        plt.close()
    
    print(f"\n所有图表已保存到: {output_dir}")
    print("=" * 80)


def main():
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    
    # 从配置文件加载路径
    config_file = script_dir / "config.yaml"
    if config_file.exists():
        config = read_yaml_config(str(config_file))
        
        # 解析路径
        download_dir = Path(config['download_dir'])
        if not download_dir.is_absolute():
            download_dir = (script_dir / download_dir).resolve()
        
        wav_subdir = config.get('wav_dir', 'mobvoi_hotword_dataset')
        json_subdir = config.get('json_dir', 'mobvoi_hotword_dataset_resources')
        
        wav_dir = download_dir / wav_subdir if not Path(wav_subdir).is_absolute() else Path(wav_subdir)
        json_dir = download_dir / json_subdir if not Path(json_subdir).is_absolute() else Path(json_subdir)
        
        output_subdir = config.get('statistics_plots_dir', './statistics_plots')
        output_dir = Path(output_subdir)
        if not output_dir.is_absolute():
            output_dir = (script_dir / output_dir).resolve()
        
        print("✅ 已从 config.yaml 加载路径配置")
    else:
        # 如果配置文件不存在，使用默认相对路径
        print(f"⚠️  配置文件不存在: {config_file}，使用默认路径")
        wav_dir = (script_dir / "../data/mobvoi_hotword_dataset").resolve()
        json_dir = (script_dir / "../data/mobvoi_hotword_dataset_resources").resolve()
        output_dir = (script_dir / "./statistics_plots").resolve()
    
    # 自动检测CPU核心数或从配置文件读取
    cpu_count = os.cpu_count()
    if config_file.exists():
        config_max_workers = config.get('max_workers')
        max_workers = config_max_workers if config_max_workers is not None else min(cpu_count * 3, 32)
    else:
        max_workers = min(cpu_count * 3, 32)  # IO密集型，使用CPU核心数的3倍
    
    print("=" * 80)
    print("数据统计脚本（多线程优化版 - 缓存模式）")
    print("=" * 80)
    print(f"CPU 核心数: {cpu_count}")
    print(f"线程池大小: {max_workers}")
    print(f"WAV 目录: {wav_dir}")
    print(f"JSON 目录: {json_dir}")
    if HAS_GUI:
        print(f"图表输出: {output_dir}")
    print("\n优化说明：")
    print("  - 只读取一次WAV文件，缓存时长信息")
    print("  - 减少进度打印频率，降低锁竞争")
    print("  - 为Train/Dev/Test分别生成feature分布图")
    
    total_start = time.time()
    
    # 统计 WAV 文件并构建缓存
    duration_cache = {}
    if wav_dir.exists():
        duration_cache = statistics_wav_files(wav_dir, max_workers=max_workers)
    else:
        print(f"警告: WAV 目录不存在: {wav_dir}")
    
    # 统计 JSON 文件（使用缓存）
    if json_dir.exists() and duration_cache:
        statistics_json_files(json_dir, duration_cache, max_workers=max_workers)
    else:
        if not json_dir.exists():
            print(f"警告: JSON 目录不存在: {json_dir}")
        if not duration_cache:
            print(f"警告: 缓存为空，跳过JSON统计")
    
    # 生成可视化图表（使用缓存）
    if HAS_GUI and json_dir.exists() and duration_cache:
        plot_statistics(json_dir, duration_cache, output_dir)
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 80)
    print(f"统计完成！总耗时: {total_elapsed:.2f}秒")
    print(f"性能提升：避免了重复读取{len(duration_cache)}个WAV文件")
    print("=" * 80)
    print(f"统计完成！总耗时: {total_elapsed:.2f}秒")
    print("=" * 80)


if __name__ == "__main__":
    main()
