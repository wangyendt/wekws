#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

import fbank_pybind


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare torch2lite fbank pybind output against torchaudio.compliance.kaldi.fbank."
    )
    parser.add_argument("--wav", default="", help="Optional wav path for real-audio comparison.")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--num_samples", type=int, default=16000)
    parser.add_argument("--num_mel_bins", type=int, default=80)
    parser.add_argument("--frame_length", type=float, default=25.0)
    parser.add_argument("--frame_shift", type=float, default=10.0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--random_scale", type=float, default=1000.0)
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def compute_stats(ref: torch.Tensor, out: torch.Tensor) -> Dict[str, object]:
    diff = (out - ref).abs()
    sq = (out - ref).pow(2)
    return {
        "ref_shape": list(ref.shape),
        "out_shape": list(out.shape),
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean()) if diff.numel() else 0.0,
        "rmse": float(sq.mean().sqrt()) if sq.numel() else 0.0,
        "p99_abs": float(torch.quantile(diff.reshape(-1), 0.99)) if diff.numel() else 0.0,
    }


def run_fbank(waveform: torch.Tensor, sample_rate: int, args) -> Dict[str, object]:
    ref = kaldi.fbank(
        waveform,
        num_mel_bins=args.num_mel_bins,
        frame_length=args.frame_length,
        frame_shift=args.frame_shift,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=sample_rate,
    )
    out = fbank_pybind.fbank(
        waveform,
        num_mel_bins=args.num_mel_bins,
        frame_length=args.frame_length,
        frame_shift=args.frame_shift,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=sample_rate,
    )
    stats = compute_stats(ref, out)
    stats["first_frame_ref_head"] = ref[0, :8].tolist() if ref.numel() else []
    stats["first_frame_out_head"] = out[0, :8].tolist() if out.numel() else []
    return stats


def main():
    args = parse_args()
    result: Dict[str, object] = {
        "config": {
            "sample_rate": args.sample_rate,
            "num_samples": args.num_samples,
            "num_mel_bins": args.num_mel_bins,
            "frame_length": args.frame_length,
            "frame_shift": args.frame_shift,
            "random_seed": args.random_seed,
            "random_scale": args.random_scale,
        }
    }

    torch.manual_seed(args.random_seed)
    random_waveform = torch.randn(1, args.num_samples, dtype=torch.float32) * args.random_scale
    result["random"] = run_fbank(random_waveform, args.sample_rate, args)

    if args.wav:
        wav_path = Path(args.wav).expanduser().resolve()
        waveform, sample_rate = torchaudio.load(str(wav_path))
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform * (1 << 15)
        result["wav"] = {
            "path": str(wav_path),
            **run_fbank(waveform, sample_rate, args),
        }

    print(json.dumps(result, ensure_ascii=False, indent=args.indent))


if __name__ == "__main__":
    main()
