#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 音频特征可视化工具
作者: Wayne
功能: 可视化音频的波形、STFT、Log-Mel、MFCC 和 CMVN 处理后的特征
"""

import os
import sys
import json
import argparse
import platform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torchaudio.transforms as T
import torch

# ==================== 中文字体配置 ====================
def setup_chinese_fonts():
    """
    配置matplotlib和seaborn的中文字体，兼容macOS和Linux
    """
    system = platform.system()
    
    # 设置seaborn样式
    sns.set_style("whitegrid")
    
    if system == 'Darwin':  # macOS
        # macOS 系统字体
        fonts = [
            'Arial Unicode MS',      # macOS 默认中文字体
            'PingFang SC',           # macOS 苹方字体
            'Heiti SC',              # 黑体
            'STHeiti',               # 华文黑体
        ]
    elif system == 'Linux':
        # Linux 常见中文字体
        fonts = [
            'WenQuanYi Micro Hei',   # 文泉驿微米黑
            'WenQuanYi Zen Hei',     # 文泉驿正黑
            'Noto Sans CJK SC',      # 思源黑体
            'Droid Sans Fallback',   # Droid 字体
        ]
    else:  # Windows
        fonts = [
            'SimHei',                # 黑体
            'Microsoft YaHei',       # 微软雅黑
        ]
    
    # 尝试设置字体
    for font in fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            # 测试字体是否可用
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试', fontsize=10)
            plt.close(fig)
            print(f"✅ 使用字体: {font} (系统: {system})")
            break
        except Exception as e:
            continue
    else:
        print(f"⚠️  未找到合适的中文字体，可能显示为方块 (系统: {system})")
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False


# ==================== 数据加载 ====================
def find_audio_file(audio_id, project_dir):
    """
    根据音频ID查找对应的wav文件
    """
    # 可能的路径
    wav_path = os.path.join(project_dir, 'data/mobvoi_hotword_dataset', f'{audio_id}.wav')
    
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"找不到音频文件: {wav_path}")
    
    return wav_path


def load_cmvn_stats(project_dir):
    """
    加载 CMVN 统计量
    """
    cmvn_path = os.path.join(project_dir, 'data/train/global_cmvn')
    
    if not os.path.exists(cmvn_path):
        raise FileNotFoundError(f"找不到 CMVN 文件: {cmvn_path}")
    
    with open(cmvn_path, 'r') as f:
        cmvn_info = json.load(f)
    
    mean_stat = np.array(cmvn_info['mean_stat'])
    var_stat = np.array(cmvn_info['var_stat'])
    frame_num = cmvn_info['frame_num']
    
    # 计算均值和标准差
    mean = mean_stat / frame_num
    std = np.sqrt(var_stat / frame_num - mean ** 2)
    
    return mean, std


def load_audio(wav_path, target_sr=16000):
    """
    使用 torchaudio 加载音频
    """
    waveform, sample_rate = torchaudio.load(wav_path)
    
    # 重采样到目标采样率
    if sample_rate != target_sr:
        resampler = T.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr
    
    # 转换为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    waveform = waveform.squeeze().numpy()
    duration = len(waveform) / sample_rate
    
    print(f"📊 音频信息:")
    print(f"   文件: {os.path.basename(wav_path)}")
    print(f"   采样率: {sample_rate} Hz")
    print(f"   时长: {duration:.3f} 秒")
    print(f"   样本数: {len(waveform)}")
    
    return waveform, sample_rate


# ==================== 特征提取 ====================
def extract_features(waveform, sr):
    """
    提取各种音频特征
    """
    features = {}
    
    # 参数设置
    n_fft = 512
    hop_length = 160  # 10ms at 16kHz
    win_length = 400  # 25ms at 16kHz
    
    # 1. STFT 使用 scipy
    f, t, stft = signal.stft(
        waveform, fs=sr, nperseg=win_length,
        noverlap=win_length-hop_length, nfft=n_fft
    )
    features['stft_mag'] = np.abs(stft)
    features['stft_db'] = 20 * np.log10(features['stft_mag'] + 1e-10) - 20 * np.log10(np.max(features['stft_mag']))
    
    # 2-4. 使用 torchaudio 提取特征
    waveform_torch = torch.from_numpy(waveform).unsqueeze(0).float()
    
    # 2. Mel Spectrogram
    mel_spec_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=80,
        f_min=0.0,
        f_max=sr/2.0
    )
    mel_spec = mel_spec_transform(waveform_torch)
    features['log_mel'] = 10 * torch.log10(mel_spec + 1e-10).squeeze().numpy()
    features['log_mel'] = features['log_mel'] - features['log_mel'].max()
    
    # 3. MFCC
    mfcc_transform = T.MFCC(
        sample_rate=sr,
        n_mfcc=40,
        melkwargs={
            'n_fft': n_fft,
            'hop_length': hop_length,
            'win_length': win_length,
            'n_mels': 80,
        }
    )
    features['mfcc'] = mfcc_transform(waveform_torch).squeeze().numpy()
    
    # 4. FBANK (与训练时一致) - 使用 Kaldi 兼容的方式
    waveform_kaldi = waveform_torch * (1 << 15)  # 归一化到 int16 范围
    fbank = kaldi.fbank(
        waveform_kaldi,
        num_mel_bins=80,
        frame_shift=10,
        frame_length=25,
        dither=1.0,
        sample_frequency=sr
    )
    features['fbank'] = fbank.numpy().T  # (time, freq) -> (freq, time)
    
    return features


def apply_cmvn(fbank, mean, std):
    """
    应用 CMVN 归一化
    """
    # fbank shape: (freq, time)
    fbank_cmvn = (fbank - mean[:, np.newaxis]) / (std[:, np.newaxis] + 1e-8)
    return fbank_cmvn


# ==================== 可视化 ====================
def visualize_features(waveform, sr, features, cmvn_mean, cmvn_std, audio_id, output_dir):
    """
    可视化所有特征
    """
    # 创建大图
    fig = plt.figure(figsize=(20, 12))
    
    # 时间轴
    time = np.arange(len(waveform)) / sr
    
    # ========== 1. 波形图 ==========
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(time, waveform, linewidth=0.5, alpha=0.8)
    ax1.set_title(f'音频波形 - {audio_id}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('时间 (秒)', fontsize=12)
    ax1.set_ylabel('振幅', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, time[-1]])
    
    # ========== 2. STFT 频谱图 ==========
    ax2 = plt.subplot(3, 2, 2)
    stft_time = np.arange(features['stft_db'].shape[1]) * 160 / sr
    stft_freq = np.fft.rfftfreq(512, 1/sr)
    
    im2 = ax2.pcolormesh(
        stft_time, stft_freq, features['stft_db'],
        shading='auto', cmap='viridis'
    )
    ax2.set_title('短时傅里叶变换 (STFT)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('时间 (秒)', fontsize=12)
    ax2.set_ylabel('频率 (Hz)', fontsize=12)
    ax2.set_ylim([0, sr // 2])
    plt.colorbar(im2, ax=ax2, label='幅度 (dB)')
    
    # ========== 3. Log-Mel 频谱图 ==========
    ax3 = plt.subplot(3, 2, 3)
    mel_time = np.arange(features['log_mel'].shape[1]) * 160 / sr
    
    im3 = ax3.pcolormesh(
        mel_time, np.arange(80), features['log_mel'],
        shading='auto', cmap='hot'
    )
    ax3.set_title('Log-Mel 频谱图 (80 bins)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('时间 (秒)', fontsize=12)
    ax3.set_ylabel('Mel 频率 Bin', fontsize=12)
    plt.colorbar(im3, ax=ax3, label='能量 (dB)')
    
    # ========== 4. MFCC ==========
    ax4 = plt.subplot(3, 2, 4)
    mfcc_time = np.arange(features['mfcc'].shape[1]) * 160 / sr
    
    im4 = ax4.pcolormesh(
        mfcc_time, np.arange(40), features['mfcc'],
        shading='auto', cmap='coolwarm'
    )
    ax4.set_title('梅尔频率倒谱系数 (MFCC, 40维)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('时间 (秒)', fontsize=12)
    ax4.set_ylabel('MFCC 系数', fontsize=12)
    plt.colorbar(im4, ax=ax4, label='系数值')
    
    # ========== 5. FBANK (训练使用的特征) ==========
    ax5 = plt.subplot(3, 2, 5)
    fbank_time = np.arange(features['fbank'].shape[1]) * 10 / 1000  # frame_shift=10ms
    
    im5 = ax5.pcolormesh(
        fbank_time, np.arange(80), features['fbank'],
        shading='auto', cmap='jet'
    )
    ax5.set_title('FBANK 特征 (训练使用, 80维)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('时间 (秒)', fontsize=12)
    ax5.set_ylabel('Filter Bank Bin', fontsize=12)
    plt.colorbar(im5, ax=ax5, label='能量')
    
    # ========== 6. CMVN 归一化后的 FBANK ==========
    ax6 = plt.subplot(3, 2, 6)
    fbank_cmvn = apply_cmvn(features['fbank'], cmvn_mean, cmvn_std)
    
    im6 = ax6.pcolormesh(
        fbank_time, np.arange(80), fbank_cmvn,
        shading='auto', cmap='RdBu_r', vmin=-3, vmax=3
    )
    ax6.set_title('CMVN 归一化后的 FBANK (Stage 1 输出)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('时间 (秒)', fontsize=12)
    ax6.set_ylabel('Filter Bank Bin', fontsize=12)
    plt.colorbar(im6, ax=ax6, label='归一化值')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{audio_id}_features.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 图片已保存: {output_path}")
    
    # 显示图片
    plt.show()
    
    return fbank_cmvn


def print_feature_stats(features, fbank_cmvn):
    """
    打印特征统计信息
    """
    print("\n" + "="*60)
    print("特征统计信息")
    print("="*60)
    
    print(f"\n1. STFT:")
    print(f"   形状: {features['stft_mag'].shape} (频率bins × 时间帧)")
    print(f"   幅度范围: [{features['stft_mag'].min():.2f}, {features['stft_mag'].max():.2f}]")
    
    print(f"\n2. Log-Mel 频谱:")
    print(f"   形状: {features['log_mel'].shape} (Mel bins × 时间帧)")
    print(f"   能量范围: [{features['log_mel'].min():.2f}, {features['log_mel'].max():.2f}] dB")
    
    print(f"\n3. MFCC:")
    print(f"   形状: {features['mfcc'].shape} (MFCC维度 × 时间帧)")
    print(f"   系数范围: [{features['mfcc'].min():.2f}, {features['mfcc'].max():.2f}]")
    
    print(f"\n4. FBANK (训练使用):")
    print(f"   形状: {features['fbank'].shape} (频率bins × 时间帧)")
    print(f"   能量范围: [{features['fbank'].min():.2f}, {features['fbank'].max():.2f}]")
    
    print(f"\n5. CMVN 归一化后的 FBANK:")
    print(f"   形状: {fbank_cmvn.shape} (频率bins × 时间帧)")
    print(f"   归一化值范围: [{fbank_cmvn.min():.2f}, {fbank_cmvn.max():.2f}]")
    print(f"   均值: {fbank_cmvn.mean():.4f}")
    print(f"   标准差: {fbank_cmvn.std():.4f}")
    
    print("\n" + "="*60)


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(
        description='Stage 1 音频特征可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python stage_1_visualize.py 6388e4b9fb1e2439281a11cdeea78943
  python stage_1_visualize.py 68c08ef7b1fbf26612271f3f6f7ddc62 --output-dir ./visualizations
  python stage_1_visualize.py ae4a93276151f8da99d7ef4a03a14aa5 --no-show
        """
    )
    
    parser.add_argument(
        'audio_id',
        type=str,
        help='音频文件ID (不带.wav后缀)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出图片的目录 (默认: wayne_scripts/visualizations/)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='不显示图片，只保存'
    )
    
    args = parser.parse_args()
    
    # 配置中文字体
    setup_chinese_fonts()
    
    # 获取项目目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(script_dir, 'visualizations')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Stage 1 音频特征可视化工具")
    print("="*60)
    print(f"音频ID: {args.audio_id}")
    print(f"项目目录: {project_dir}")
    print(f"输出目录: {output_dir}")
    print("="*60 + "\n")
    
    try:
        # 1. 查找音频文件
        print("🔍 查找音频文件...")
        wav_path = find_audio_file(args.audio_id, project_dir)
        print(f"✅ 找到文件: {wav_path}\n")
        
        # 2. 加载 CMVN 统计量
        print("📊 加载 CMVN 统计量...")
        cmvn_mean, cmvn_std = load_cmvn_stats(project_dir)
        print(f"✅ CMVN 均值形状: {cmvn_mean.shape}")
        print(f"✅ CMVN 标准差形状: {cmvn_std.shape}\n")
        
        # 3. 加载音频
        print("🎵 加载音频...")
        waveform, sr = load_audio(wav_path)
        print()
        
        # 4. 提取特征
        print("🔬 提取音频特征...")
        features = extract_features(waveform, sr)
        print("✅ 特征提取完成\n")
        
        # 5. 可视化
        print("🎨 生成可视化图表...")
        if args.no_show:
            plt.ioff()  # 关闭交互模式
        
        fbank_cmvn = visualize_features(
            waveform, sr, features, 
            cmvn_mean, cmvn_std, 
            args.audio_id, output_dir
        )
        
        # 6. 打印统计信息
        print_feature_stats(features, fbank_cmvn)
        
        print("\n✅ 可视化完成！")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
