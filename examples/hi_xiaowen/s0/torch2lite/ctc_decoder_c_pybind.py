import hashlib
import os
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from torch.utils.cpp_extension import load


_EXTENSION = None


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    base_dir = Path(__file__).resolve().parent
    extra_cflags = [
        "-O3",
        "-std=c++17",
    ]
    env_extra = os.environ.get("HI_XIAOWEN_CTC_DECODER_C_EXTRA_CFLAGS", "").strip()
    if env_extra:
        extra_cflags.extend(shlex.split(env_extra))
    flag_hash = hashlib.sha1("\0".join(extra_cflags).encode("utf-8")).hexdigest()[:10]
    build_dir = base_dir / ".ctc_decoder_c_pybind_build" / flag_hash
    build_dir.mkdir(parents=True, exist_ok=True)

    _EXTENSION = load(
        name=f"hi_xiaowen_ctc_decoder_c_ext_{flag_hash}",
        sources=[
            str(base_dir / "ctc_decoder_c_pybind.cc"),
            str(base_dir / "ctc_decoder_c.cc"),
        ],
        extra_cflags=extra_cflags,
        with_cuda=False,
        build_directory=str(build_dir),
        verbose=False,
    )
    return _EXTENSION


class StreamingCTCDecoderC:
    """C-style streaming CTC prefix beam search decoder for KWS."""

    def __init__(
        self,
        score_beam_size: int = 3,
        path_beam_size: int = 20,
        min_frames: int = 5,
        max_frames: int = 250,
        interval_frames: int = 50,
        frame_step: int = 1,
    ) -> None:
        self._decoder = _load_extension().StreamingCTCDecoderCStyle(
            score_beam_size,
            path_beam_size,
            min_frames,
            max_frames,
            interval_frames,
            frame_step,
        )
        self._keyword_strings: List[str] = []
        self._keyword_to_idx: Dict[str, int] = {}

    def set_keywords(
        self,
        keywords_token: Dict[str, Dict[str, Tuple[int, ...]]],
        keywords_idxset: Set[int],
    ) -> None:
        self._keyword_strings = list(keywords_token.keys())
        self._keyword_to_idx = {kw: idx for idx, kw in enumerate(self._keyword_strings)}

        keywords_tokens = []
        for idx, keyword in enumerate(self._keyword_strings):
            token_ids = list(keywords_token[keyword]["token_id"])
            keywords_tokens.append((idx, token_ids))

        self._decoder.set_keywords(
            keywords_tokens,
            list(keywords_idxset),
            self._keyword_strings,
        )

    def set_thresholds(self, threshold_map: Dict[str, Optional[float]]) -> None:
        idx_map = {}
        for kw, th in threshold_map.items():
            if th is not None and kw in self._keyword_to_idx:
                idx_map[self._keyword_to_idx[kw]] = th
        self._decoder.set_thresholds(idx_map)

    def advance_frame(self, frame_index, probs):
        self._decoder.advance_frame(frame_index, probs)

    def execute_detection(self, disable_threshold: bool = False):
        return self._decoder.execute_detection(disable_threshold)

    def step_and_detect(self, frame_index, probs, disable_threshold: bool = False):
        return self._decoder.step_and_detect(frame_index, probs, disable_threshold)

    def step_and_detect_next(self, probs, disable_threshold: bool = False):
        return self._decoder.step_and_detect_next(probs, disable_threshold)

    def reset(self) -> None:
        self._decoder.reset()

    def reset_beam_search(self) -> None:
        self._decoder.reset_beam_search()

    def get_best_decode_result(self) -> dict:
        if not hasattr(self._decoder, "get_best_decode"):
            raise RuntimeError("extended decoder API is disabled at build time")
        return self._decoder.get_best_decode()

    def get_first_hyp_start_frame(self) -> int:
        if not hasattr(self._decoder, "get_first_hyp_start_frame"):
            raise RuntimeError("extended decoder API is disabled at build time")
        return self._decoder.get_first_hyp_start_frame()

    @property
    def num_hypotheses(self) -> int:
        return self._decoder.num_hypotheses()

    def get_hypotheses(self):
        if not hasattr(self._decoder, "get_hypotheses"):
            raise RuntimeError("debug hypotheses API is disabled at build time")
        return self._decoder.get_hypotheses()
