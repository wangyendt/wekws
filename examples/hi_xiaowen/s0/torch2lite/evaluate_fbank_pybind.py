#!/usr/bin/env python3

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import torch

import fbank_pybind


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate torch2lite pybind fbank self-consistency across single-frame, "
            "batch, and streaming interfaces for both float and int16 inputs."
        )
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--num_samples", type=int, default=32000)
    parser.add_argument("--num_mel_bins", type=int, default=80)
    parser.add_argument("--frame_length", type=float, default=25.0)
    parser.add_argument("--frame_shift", type=float, default=10.0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--single_frame_trials", type=int, default=32)
    parser.add_argument("--stream_min_update_frames", type=int, default=1)
    parser.add_argument("--stream_max_update_frames", type=int, default=5)
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


def build_extractors(args):
    common_kwargs = dict(
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
    return fbank_pybind.FbankExtractor(**common_kwargs), fbank_pybind.StreamingFbankExtractor(**common_kwargs)


def make_random_waveform(num_samples: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    audio_int16 = torch.from_numpy(rng.integers(-30000, 30001, size=num_samples, dtype=np.int16))
    audio_float = audio_int16.to(torch.float32)
    return audio_int16, audio_float


def sample_frame_indices(num_frames: int, trials: int, rng: np.random.Generator) -> List[int]:
    if num_frames <= 0:
        return []
    count = min(num_frames, max(1, trials))
    if count == num_frames:
        return list(range(num_frames))
    return sorted(rng.choice(num_frames, size=count, replace=False).tolist())


def build_stream_update_sizes(
    total_samples: int,
    frame_shift_samples: int,
    min_update_frames: int,
    max_update_frames: int,
    rng: np.random.Generator,
) -> Tuple[List[int], List[int]]:
    if min_update_frames <= 0 or max_update_frames < min_update_frames:
        raise ValueError("invalid stream update frame range")
    frame_counts: List[int] = []
    sample_counts: List[int] = []
    consumed = 0
    while consumed < total_samples:
        frame_count = int(rng.integers(min_update_frames, max_update_frames + 1))
        sample_count = frame_count * frame_shift_samples
        sample_count = min(sample_count, total_samples - consumed)
        frame_counts.append(frame_count)
        sample_counts.append(sample_count)
        consumed += sample_count
    return frame_counts, sample_counts


def collect_stream_outputs(
    stream_extractor,
    waveform: torch.Tensor,
    sample_counts: List[int],
    mode: str,
) -> torch.Tensor:
    outputs = []
    start = 0
    for sample_count in sample_counts:
        chunk = waveform[start:start + sample_count]
        start += sample_count
        if mode == "float":
            out = stream_extractor.accept_float(chunk, num_samples=int(chunk.numel()))
        elif mode == "int16":
            out = stream_extractor.accept_int16(chunk, num_samples=int(chunk.numel()))
        else:
            raise ValueError(f"unsupported mode: {mode}")
        if out.numel() > 0:
            outputs.append(out)
    if not outputs:
        return torch.empty((0, stream_extractor.num_mel_bins), dtype=torch.float32)
    return torch.cat(outputs, dim=0)


def evaluate_mode(
    mode: str,
    waveform: torch.Tensor,
    batch_features: torch.Tensor,
    stream_features: torch.Tensor,
    extractor,
    frame_indices: List[int],
) -> Dict[str, object]:
    frame_length_samples = extractor.frame_length
    frame_shift_samples = extractor.frame_shift
    single_outputs = []
    batch_selected = []
    stream_selected = []

    for frame_index in frame_indices:
        start = frame_index * frame_shift_samples
        frame = waveform[start:start + frame_length_samples]
        if mode == "float":
            single = extractor.process_frame_float(frame)
        elif mode == "int16":
            single = extractor.process_frame_int16(frame)
        else:
            raise ValueError(f"unsupported mode: {mode}")
        single_outputs.append(single)
        batch_selected.append(batch_features[frame_index:frame_index + 1])
        stream_selected.append(stream_features[frame_index:frame_index + 1])

    if single_outputs:
        single_cat = torch.cat(single_outputs, dim=0)
        batch_cat = torch.cat(batch_selected, dim=0)
        stream_cat = torch.cat(stream_selected, dim=0)
    else:
        shape = (0, extractor.num_mel_bins)
        single_cat = torch.empty(shape, dtype=torch.float32)
        batch_cat = torch.empty(shape, dtype=torch.float32)
        stream_cat = torch.empty(shape, dtype=torch.float32)

    return {
        "num_frames": int(batch_features.shape[0]),
        "sampled_frame_indices": frame_indices,
        "single_vs_batch": compute_stats(batch_cat, single_cat),
        "single_vs_stream": compute_stats(stream_cat, single_cat),
        "batch_vs_stream": compute_stats(batch_features, stream_features),
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)

    batch_extractor, stream_extractor = build_extractors(args)
    audio_int16, audio_float = make_random_waveform(args.num_samples, rng)

    batch_float = batch_extractor.extract_float(audio_float, num_samples=int(audio_float.numel()))
    batch_int16 = batch_extractor.extract_int16(audio_int16, num_samples=int(audio_int16.numel()))

    frame_indices = sample_frame_indices(int(batch_float.shape[0]), args.single_frame_trials, rng)
    stream_frame_counts, stream_sample_counts = build_stream_update_sizes(
        total_samples=int(audio_int16.numel()),
        frame_shift_samples=batch_extractor.frame_shift,
        min_update_frames=args.stream_min_update_frames,
        max_update_frames=args.stream_max_update_frames,
        rng=rng,
    )

    stream_float_extractor = stream_extractor
    stream_float = collect_stream_outputs(stream_float_extractor, audio_float, stream_sample_counts, mode="float")

    _, stream_int16_extractor = build_extractors(args)
    stream_int16 = collect_stream_outputs(stream_int16_extractor, audio_int16, stream_sample_counts, mode="int16")

    result = {
        "config": {
            "sample_rate": args.sample_rate,
            "num_samples": args.num_samples,
            "num_mel_bins": args.num_mel_bins,
            "frame_length_ms": args.frame_length,
            "frame_shift_ms": args.frame_shift,
            "frame_length_samples": batch_extractor.frame_length,
            "frame_shift_samples": batch_extractor.frame_shift,
            "fft_size": batch_extractor.fft_size,
            "single_frame_trials": args.single_frame_trials,
            "random_seed": args.random_seed,
        },
        "stream_updates": {
            "count": len(stream_sample_counts),
            "frame_counts": stream_frame_counts,
            "sample_counts": stream_sample_counts,
        },
        "float": evaluate_mode(
            mode="float",
            waveform=audio_float,
            batch_features=batch_float,
            stream_features=stream_float,
            extractor=batch_extractor,
            frame_indices=frame_indices,
        ),
        "int16": evaluate_mode(
            mode="int16",
            waveform=audio_int16,
            batch_features=batch_int16,
            stream_features=stream_int16,
            extractor=batch_extractor,
            frame_indices=frame_indices,
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=args.indent))


if __name__ == "__main__":
    main()
