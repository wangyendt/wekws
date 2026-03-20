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
            "-DFBANK_USE_PRECOMPUTED_TABLES",
        ],
        with_cuda=False,
        build_directory=str(build_dir),
        verbose=False,
    )
    return _EXTENSION


class FbankExtractor:
    def __init__(
        self,
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
        fft_size: int = 0,
    ) -> None:
        self._extractor = _load_extension().FbankExtractor(
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
            fft_size,
        )

    @property
    def frame_length(self) -> int:
        return self._extractor.frame_length

    @property
    def frame_shift(self) -> int:
        return self._extractor.frame_shift

    @property
    def fft_size(self) -> int:
        return self._extractor.fft_size

    @property
    def num_mel_bins(self) -> int:
        return self._extractor.num_mel_bins

    @property
    def work_buffer_bytes(self) -> int:
        return self._extractor.work_buffer_bytes

    def reset(self) -> None:
        self._extractor.reset()

    def num_frames(self, num_samples: int) -> int:
        return self._extractor.num_frames(num_samples)

    def process_frame_float(self, waveform: torch.Tensor) -> torch.Tensor:
        return self._extractor.process_frame_float(waveform)

    def process_frame_int16(self, waveform: torch.Tensor) -> torch.Tensor:
        return self._extractor.process_frame_int16(waveform)

    def extract_float(
        self,
        waveform: torch.Tensor,
        num_samples: int = -1,
        max_frames: int = -1,
    ) -> torch.Tensor:
        return self._extractor.extract_float(waveform, num_samples, max_frames)

    def extract_int16(
        self,
        waveform: torch.Tensor,
        num_samples: int = -1,
        max_frames: int = -1,
    ) -> torch.Tensor:
        return self._extractor.extract_int16(waveform, num_samples, max_frames)


class StreamingFbankExtractor:
    def __init__(
        self,
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
        fft_size: int = 0,
    ) -> None:
        self._extractor = _load_extension().StreamingFbankExtractor(
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
            fft_size,
        )

    @property
    def pending_samples(self) -> int:
        return self._extractor.pending_samples

    @property
    def frame_length(self) -> int:
        return self._extractor.frame_length

    @property
    def frame_shift(self) -> int:
        return self._extractor.frame_shift

    @property
    def fft_size(self) -> int:
        return self._extractor.fft_size

    @property
    def num_mel_bins(self) -> int:
        return self._extractor.num_mel_bins

    def reset(self) -> None:
        self._extractor.reset()

    def accept_float(
        self,
        waveform: torch.Tensor,
        num_samples: int = -1,
        max_frames: int = -1,
    ) -> torch.Tensor:
        return self._extractor.accept_float(waveform, num_samples, max_frames)

    def accept_int16(
        self,
        waveform: torch.Tensor,
        num_samples: int = -1,
        max_frames: int = -1,
    ) -> torch.Tensor:
        return self._extractor.accept_int16(waveform, num_samples, max_frames)
