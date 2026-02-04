#!/usr/bin/env python3
# encoding: utf-8

import sys
import argparse
import json
import codecs
import yaml
import platform

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset, DataLoader


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
if AUDIO_BACKEND:
    print(f"使用音频后端: {AUDIO_BACKEND} (系统: {platform.system()})", file=sys.stderr)
else:
    print(f"使用默认音频后端 (系统: {platform.system()})", file=sys.stderr)


class CollateFunc(object):
    ''' Collate function for AudioDataset
    '''

    def __init__(self, feat_dim, feat_type, resample_rate):
        self.feat_dim = feat_dim
        self.resample_rate = resample_rate
        self.feat_type = feat_type
        pass

    def __call__(self, batch):
        mean_stat = torch.zeros(self.feat_dim)
        var_stat = torch.zeros(self.feat_dim)
        number = 0
        for item in batch:
            value = item[1].strip().split(",")
            assert len(value) == 3 or len(value) == 1
            wav_path = value[0]
            
            # 根据系统使用合适的 backend
            if AUDIO_BACKEND:
                sample_rate = torchaudio.info(wav_path, backend=AUDIO_BACKEND).sample_rate
            else:
                sample_rate = torchaudio.info(wav_path).sample_rate
            
            resample_rate = sample_rate
            # len(value) == 3 means segmented wav.scp,
            # len(value) == 1 means original wav.scp
            if len(value) == 3:
                start_frame = int(float(value[1]) * sample_rate)
                end_frame = int(float(value[2]) * sample_rate)
                if AUDIO_BACKEND:
                    waveform, sample_rate = torchaudio.load(
                        filepath=wav_path,
                        num_frames=end_frame - start_frame,
                        frame_offset=start_frame,
                        backend=AUDIO_BACKEND)
                else:
                    waveform, sample_rate = torchaudio.load(
                        filepath=wav_path,
                        num_frames=end_frame - start_frame,
                        frame_offset=start_frame)
            else:
                if AUDIO_BACKEND:
                    waveform, sample_rate = torchaudio.load(item[1], backend=AUDIO_BACKEND)
                else:
                    waveform, sample_rate = torchaudio.load(item[1])

            waveform = waveform * (1 << 15)
            if self.resample_rate != 0 and self.resample_rate != sample_rate:
                resample_rate = self.resample_rate
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=resample_rate)(waveform)
            if self.feat_type == 'fbank':
                mat = kaldi.fbank(waveform,
                                  num_mel_bins=self.feat_dim,
                                  dither=0.0,
                                  energy_floor=0.0,
                                  sample_frequency=resample_rate)
            elif self.feat_type == 'mfcc':
                mat = kaldi.mfcc(
                    waveform,
                    num_ceps=self.feat_dim,
                    num_mel_bins=self.feat_dim,
                    dither=0.0,
                    energy_floor=0.0,
                    sample_frequency=resample_rate,
                )
            mean_stat += torch.sum(mat, axis=0)
            var_stat += torch.sum(torch.square(mat), axis=0)
            number += mat.shape[0]
        return number, mean_stat, var_stat


class AudioDataset(Dataset):

    def __init__(self, data_file):
        self.items = []
        with codecs.open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                arr = line.strip().split()
                self.items.append((arr[0], arr[1]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract CMVN stats')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for processing')
    parser.add_argument('--train_config',
                        default='',
                        help='training yaml conf')
    parser.add_argument('--in_scp', default=None, help='wav scp file')
    parser.add_argument('--out_cmvn',
                        default='global_cmvn',
                        help='global cmvn file')

    args = parser.parse_args()

    with open(args.train_config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    feat_type = configs['dataset_conf']['feats_type']
    feat_dim = configs['dataset_conf'][f'{feat_type}_conf'][
        'num_mel_bins']
    resample_rate = 0
    if 'resample_conf' in configs['dataset_conf']:
        resample_rate = configs['dataset_conf']['resample_conf'][
            'resample_rate']
        print('using resample and new sample rate is {}'.format(resample_rate))

    collate_func = CollateFunc(feat_dim, feat_type, resample_rate)
    dataset = AudioDataset(args.in_scp)
    batch_size = 20
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             sampler=None,
                             num_workers=args.num_workers,
                             collate_fn=collate_func)

    with torch.no_grad():
        all_number = 0
        all_mean_stat = torch.zeros(feat_dim)
        all_var_stat = torch.zeros(feat_dim)
        wav_number = 0
        for i, batch in enumerate(data_loader):
            number, mean_stat, var_stat = batch
            all_mean_stat += mean_stat
            all_var_stat += var_stat
            all_number += number
            wav_number += batch_size
            if wav_number % 1000 == 0:
                print(f'processed {wav_number} wavs, {all_number} frames',
                      file=sys.stderr,
                      flush=True)

    cmvn_info = {
        'mean_stat': list(all_mean_stat.tolist()),
        'var_stat': list(all_var_stat.tolist()),
        'frame_num': all_number
    }

    with open(args.out_cmvn, 'w') as fout:
        fout.write(json.dumps(cmvn_info))
