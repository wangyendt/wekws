from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXTENSION = None


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    base_dir = Path(__file__).resolve().parent
    build_dir = base_dir / ".fbank_pybind_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    _EXTENSION = load(
        name="hi_xiaowen_fbank_ext",
        sources=[
            str(base_dir / "fbank_pybind.cc"),
            str(base_dir / "fbank.cc"),
        ],
        extra_cflags=[
            "-O3",
            "-std=c++17",
        ],
        with_cuda=False,
        build_directory=str(build_dir),
        verbose=False,
    )
    return _EXTENSION


def fbank(
    waveform: torch.Tensor,
    num_mel_bins: int = 80,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    dither: float = 0.0,
    energy_floor: float = 0.0,
    sample_frequency: float = 16000.0,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
    preemphasis_coefficient: float = 0.97,
    remove_dc_offset: bool = True,
    round_to_power_of_two: bool = True,
    snip_edges: bool = True,
) -> torch.Tensor:
    return _load_extension().fbank(
        waveform,
        num_mel_bins,
        frame_length,
        frame_shift,
        dither,
        energy_floor,
        sample_frequency,
        low_freq,
        high_freq,
        preemphasis_coefficient,
        remove_dc_offset,
        round_to_power_of_two,
        snip_edges,
    )
