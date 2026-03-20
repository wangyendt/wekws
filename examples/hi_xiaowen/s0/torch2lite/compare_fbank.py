#!/usr/bin/env python3

import argparse
import json
from typing import Dict, List

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi

import fbank_pybind


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare torchaudio.compliance.kaldi.fbank against torch2lite pybind "
            "in single-frame and batch modes using random PCM-scale audio."
        )
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--num_samples", type=int, default=32000)
    parser.add_argument("--num_mel_bins", type=int, default=80)
    parser.add_argument("--frame_length", type=float, default=25.0)
    parser.add_argument("--frame_shift", type=float, default=10.0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--single_frame_trials", type=int, default=32)
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def compute_stats(ref: torch.Tensor, out: torch.Tensor) -> Dict[str, object]:
    if ref.shape != out.shape:
        raise ValueError(f"shape mismatch: ref={tuple(ref.shape)} out={tuple(out.shape)}")
    diff = (out - ref).abs().reshape(-1)
    sq = (out - ref).pow(2).reshape(-1)
    if diff.numel() == 0:
        return {
            "shape": list(ref.shape),
            "count": 0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "median_abs": 0.0,
            "p95_abs": 0.0,
            "p99_abs": 0.0,
            "rmse": 0.0,
        }
    return {
        "shape": list(ref.shape),
        "count": int(diff.numel()),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "median_abs": float(torch.quantile(diff, 0.5).item()),
        "p95_abs": float(torch.quantile(diff, 0.95).item()),
        "p99_abs": float(torch.quantile(diff, 0.99).item()),
        "rmse": float(sq.mean().sqrt().item()),
    }


def build_extractor(args):
    return fbank_pybind.FbankExtractor(
        num_mel_bins=args.num_mel_bins,
        frame_length=args.frame_length,
        frame_shift=args.frame_shift,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=args.sample_rate,
        low_freq=20.0,
        high_freq=0.0,
        preemphasis_coefficient=0.97,
        remove_dc_offset=True,
        round_to_power_of_two=True,
        snip_edges=True,
    )


def run_kaldi_batch(waveform_2d: torch.Tensor, args) -> torch.Tensor:
    return kaldi.fbank(
        waveform_2d,
        num_mel_bins=args.num_mel_bins,
        frame_length=args.frame_length,
        frame_shift=args.frame_shift,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=args.sample_rate,
        low_freq=20.0,
        high_freq=0.0,
        preemphasis_coefficient=0.97,
        remove_dc_offset=True,
        round_to_power_of_two=True,
        snip_edges=True,
    )


def sample_frame_indices(num_frames: int, trials: int, rng: np.random.Generator) -> List[int]:
    if num_frames <= 0:
        return []
    count = min(num_frames, max(1, trials))
    if count == num_frames:
        return list(range(num_frames))
    return sorted(rng.choice(num_frames, size=count, replace=False).tolist())


def main():
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)

    audio_int16 = torch.from_numpy(rng.integers(-30000, 30001, size=args.num_samples, dtype=np.int16))
    audio_float = audio_int16.to(torch.float32)
    audio_float_2d = audio_float.unsqueeze(0)

    extractor = build_extractor(args)
    kaldi_batch = run_kaldi_batch(audio_float_2d, args)
    pybind_batch_float = extractor.extract_float(audio_float, num_samples=int(audio_float.numel()))
    pybind_batch_int16 = extractor.extract_int16(audio_int16, num_samples=int(audio_int16.numel()))

    frame_indices = sample_frame_indices(int(kaldi_batch.shape[0]), args.single_frame_trials, rng)
    kaldi_single = []
    pybind_single_float = []
    pybind_single_int16 = []
    for frame_index in frame_indices:
        start = frame_index * extractor.frame_shift
        frame_float = audio_float[start:start + extractor.frame_length].unsqueeze(0)
        frame_int16 = audio_int16[start:start + extractor.frame_length]
        kaldi_single.append(run_kaldi_batch(frame_float, args))
        pybind_single_float.append(extractor.process_frame_float(frame_float))
        pybind_single_int16.append(extractor.process_frame_int16(frame_int16))

    if kaldi_single:
        kaldi_single_cat = torch.cat(kaldi_single, dim=0)
        pybind_single_float_cat = torch.cat(pybind_single_float, dim=0)
        pybind_single_int16_cat = torch.cat(pybind_single_int16, dim=0)
    else:
        empty = torch.empty((0, args.num_mel_bins), dtype=torch.float32)
        kaldi_single_cat = empty
        pybind_single_float_cat = empty
        pybind_single_int16_cat = empty

    result = {
        "config": {
            "sample_rate": args.sample_rate,
            "num_samples": args.num_samples,
            "num_mel_bins": args.num_mel_bins,
            "frame_length_ms": args.frame_length,
            "frame_shift_ms": args.frame_shift,
            "frame_length_samples": extractor.frame_length,
            "frame_shift_samples": extractor.frame_shift,
            "fft_size": extractor.fft_size,
            "single_frame_trials": args.single_frame_trials,
            "random_seed": args.random_seed,
        },
        "sampled_frame_indices": frame_indices,
        "single_frame": {
            "kaldi_vs_pybind_float": compute_stats(kaldi_single_cat, pybind_single_float_cat),
            "kaldi_vs_pybind_int16": compute_stats(kaldi_single_cat, pybind_single_int16_cat),
        },
        "batch": {
            "kaldi_vs_pybind_float": compute_stats(kaldi_batch, pybind_batch_float),
            "kaldi_vs_pybind_int16": compute_stats(kaldi_batch, pybind_batch_int16),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=args.indent))


if __name__ == "__main__":
    main()
