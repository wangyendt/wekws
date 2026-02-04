#!/usr/bin/env python3
# encoding: utf-8

import sys
import platform
import warnings

# 过滤 torchaudio 相关的所有弃用警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import torchaudio


# 根据操作系统自动选择合适的 backend
def get_audio_backend():
    """
    根据操作系统和可用的 backend 选择最合适的音频后端
    
    Returns:
        str or None: backend 名称，如果返回 None 则使用 torchaudio 默认
    """
    system = platform.system().lower()
    
    # 获取可用的 backends
    try:
        available_backends = torchaudio.list_audio_backends()
    except:
        available_backends = []
    
    if system == 'darwin':  # macOS
        # macOS 通常使用 soundfile，sox 不可用
        if 'soundfile' in available_backends:
            return 'soundfile'
        return None  # 使用默认
    elif system == 'linux':
        # Linux 优先使用 sox，不可用则使用 soundfile
        if 'sox' in available_backends:
            return 'sox'
        elif 'soundfile' in available_backends:
            return 'soundfile'
        return None
    else:  # Windows 等其他系统
        # 使用 soundfile 或默认
        if 'soundfile' in available_backends:
            return 'soundfile'
        return None


# 全局 backend 配置
AUDIO_BACKEND = get_audio_backend()

scp = sys.argv[1]
dur_scp = sys.argv[2]

with open(scp, 'r') as f, open(dur_scp, 'w') as fout:
    cnt = 0
    total_duration = 0
    for l in f:
        items = l.strip().split()
        wav_id = items[0]
        fname = items[1]
        cnt += 1
        
        # 根据系统使用合适的 backend
        if AUDIO_BACKEND:
            waveform, rate = torchaudio.load(fname, backend=AUDIO_BACKEND)
        else:
            waveform, rate = torchaudio.load(fname)
        
        frames = len(waveform[0])
        duration = frames / float(rate)
        total_duration += duration
        fout.write('{} {}\n'.format(wav_id, duration))
    print('process {} utts'.format(cnt))
    print('total {} s'.format(total_duration))
