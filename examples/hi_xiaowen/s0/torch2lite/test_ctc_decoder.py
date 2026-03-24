#!/usr/bin/env python3
"""Comparison test: Python vs C++ streaming CTC decoder.

Usage:
    cd examples/hi_xiaowen/s0
    python torch2lite/test_ctc_decoder.py
"""

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import infer_wav_stream as iws
from torch2lite.ctc_decoder_c_pybind import StreamingCTCDecoderC
from torch2lite.ctc_decoder_pybind import StreamingCTCDecoder


def assert_hypotheses_match(left_hyps, right_hyps, label: str):
    assert len(left_hyps) == len(right_hyps), (
        f"{label}: hypothesis count mismatch {len(left_hyps)} vs {len(right_hyps)}"
    )
    for hyp_index, (left, right) in enumerate(zip(left_hyps, right_hyps)):
        assert list(left.prefix) == list(right.prefix), (
            f"{label}: hyp {hyp_index} prefix mismatch {left.prefix} vs {right.prefix}"
        )
        assert math.isclose(left.pb, right.pb, abs_tol=1e-9), (
            f"{label}: hyp {hyp_index} pb mismatch {left.pb} vs {right.pb}"
        )
        assert math.isclose(left.pnb, right.pnb, abs_tol=1e-9), (
            f"{label}: hyp {hyp_index} pnb mismatch {left.pnb} vs {right.pnb}"
        )
        assert len(left.nodes) == len(right.nodes), (
            f"{label}: hyp {hyp_index} node count mismatch {len(left.nodes)} vs {len(right.nodes)}"
        )
        for node_index, (left_node, right_node) in enumerate(zip(left.nodes, right.nodes)):
            assert left_node.token == right_node.token, (
                f"{label}: hyp {hyp_index} node {node_index} token mismatch {left_node.token} vs {right_node.token}"
            )
            assert left_node.frame == right_node.frame, (
                f"{label}: hyp {hyp_index} node {node_index} frame mismatch {left_node.frame} vs {right_node.frame}"
            )
            assert math.isclose(left_node.prob, right_node.prob, abs_tol=1e-7), (
                f"{label}: hyp {hyp_index} node {node_index} prob mismatch {left_node.prob} vs {right_node.prob}"
            )


def assert_detection_match(left_result, right_result, label: str):
    assert (left_result is None) == (right_result is None), (
        f"{label}: detection presence mismatch {left_result} vs {right_result}"
    )
    if left_result is None:
        return
    assert left_result["keyword"] == right_result["keyword"], (
        f"{label}: keyword mismatch {left_result} vs {right_result}"
    )
    assert math.isclose(left_result["candidate_score"], right_result["candidate_score"], abs_tol=1e-9), (
        f"{label}: score mismatch {left_result} vs {right_result}"
    )
    assert left_result["start_frame"] == right_result["start_frame"], (
        f"{label}: start_frame mismatch {left_result} vs {right_result}"
    )
    assert left_result["end_frame"] == right_result["end_frame"], (
        f"{label}: end_frame mismatch {left_result} vs {right_result}"
    )


def assert_best_decode_match(left_best, right_best, label: str):
    assert left_best["candidate_keyword"] == right_best["candidate_keyword"], (
        f"{label}: best keyword mismatch {left_best} vs {right_best}"
    )
    if left_best["candidate_score"] is None:
        assert right_best["candidate_score"] is None, f"{label}: best score presence mismatch"
        return
    assert math.isclose(left_best["candidate_score"], right_best["candidate_score"], abs_tol=1e-9), (
        f"{label}: best score mismatch {left_best} vs {right_best}"
    )
    assert left_best["start_frame"] == right_best["start_frame"], (
        f"{label}: best start mismatch {left_best} vs {right_best}"
    )
    assert left_best["end_frame"] == right_best["end_frame"], (
        f"{label}: best end mismatch {left_best} vs {right_best}"
    )


def is_sublist(main_list, check_list):
    """Python reference implementation from infer_wav.py."""
    if len(main_list) < len(check_list):
        return -1
    if len(main_list) == len(check_list):
        return 0 if main_list == check_list else -1
    for index in range(len(main_list) - len(check_list) + 1):
        if tuple(main_list[index:index + len(check_list)]) == tuple(check_list):
            return index
    return -1


def streaming_ctc_prefix_beam_search_step_ref(
    frame_index: int,
    probs: torch.Tensor,
    cur_hyps,
    keywords_idxset,
    score_beam_size: int,
):
    """Python reference implementation from infer_wav_stream.py."""
    next_hyps = defaultdict(lambda: (0.0, 0.0, []))

    top_k_probs, top_k_index = probs.topk(score_beam_size)
    filter_probs = []
    filter_index = []
    for prob, idx in zip(top_k_probs.tolist(), top_k_index.tolist()):
        if keywords_idxset is not None:
            if prob > 0.05 and idx in keywords_idxset:
                filter_probs.append(prob)
                filter_index.append(idx)
        else:
            if prob > 0.05:
                filter_probs.append(prob)
                filter_index.append(idx)

    if not filter_index:
        return cur_hyps

    for token_id in filter_index:
        token_prob = probs[token_id].item()
        for prefix, (pb, pnb, cur_nodes) in cur_hyps:
            last = prefix[-1] if prefix else None
            if token_id == 0:
                n_pb, n_pnb, nodes = next_hyps[prefix]
                n_pb = n_pb + pb * token_prob + pnb * token_prob
                next_hyps[prefix] = (n_pb, n_pnb, cur_nodes.copy())
            elif token_id == last:
                if not math.isclose(pnb, 0.0, abs_tol=1e-6):
                    n_pb, n_pnb, nodes = next_hyps[prefix]
                    n_pnb = n_pnb + pnb * token_prob
                    nodes = cur_nodes.copy()
                    if token_prob > nodes[-1]["prob"]:
                        nodes[-1]["prob"] = token_prob
                        nodes[-1]["frame"] = frame_index
                    next_hyps[prefix] = (n_pb, n_pnb, nodes)

                if not math.isclose(pb, 0.0, abs_tol=1e-6):
                    n_prefix = prefix + (token_id,)
                    n_pb, n_pnb, nodes = next_hyps[n_prefix]
                    n_pnb = n_pnb + pb * token_prob
                    nodes = cur_nodes.copy()
                    nodes.append(dict(token=token_id, frame=frame_index, prob=token_prob))
                    next_hyps[n_prefix] = (n_pb, n_pnb, nodes)
            else:
                n_prefix = prefix + (token_id,)
                n_pb, n_pnb, nodes = next_hyps[n_prefix]
                if nodes:
                    if token_prob > nodes[-1]["prob"]:
                        nodes.pop()
                        nodes.append(dict(token=token_id, frame=frame_index, prob=token_prob))
                else:
                    nodes = cur_nodes.copy()
                    nodes.append(dict(token=token_id, frame=frame_index, prob=token_prob))
                n_pnb = n_pnb + pb * token_prob + pnb * token_prob
                next_hyps[n_prefix] = (n_pb, n_pnb, nodes)

    return sorted(next_hyps.items(), key=lambda x: (x[1][0] + x[1][1]), reverse=True)


# ---------------------------------------------------------------------------
# Test 1: is_sublist equivalence (via beam search behavior)
# ---------------------------------------------------------------------------
def test_beam_search_single_frame():
    """Compare single frame beam search: Python vs C++."""
    vocab_size = 50
    keywords_idxset = {0, 5, 10, 15, 20}
    frame_index = 42
    score_beam_size = 3
    path_beam_size = 20

    torch.manual_seed(123)
    probs = torch.softmax(torch.randn(vocab_size), dim=0)

    # Python path
    cur_hyps_py = [(tuple(), (1.0, 0.0, []))]
    result_py = streaming_ctc_prefix_beam_search_step_ref(
        frame_index, probs, cur_hyps_py, keywords_idxset, score_beam_size
    )

    # C++ path
    decoder = StreamingCTCDecoder(
        score_beam_size=score_beam_size,
        path_beam_size=path_beam_size,
    )
    decoder._decoder.set_keywords([], list(keywords_idxset), [])
    decoder.advance_frame(frame_index, probs)
    result_cpp = decoder.get_hypotheses()

    # Compare
    assert len(result_cpp) == len(result_py), (
        f"Hypothesis count mismatch: C++ {len(result_cpp)} vs Python {len(result_py)}"
    )

    for i, ((py_prefix, (py_pb, py_pnb, py_nodes)), cpp_hyp) in enumerate(zip(result_py, result_cpp)):
        # Compare prefix
        assert list(py_prefix) == cpp_hyp.prefix, (
            f"Hyp {i} prefix mismatch: Python {list(py_prefix)} vs C++ {cpp_hyp.prefix}"
        )
        # Compare total probability
        py_total = py_pb + py_pnb
        cpp_total = cpp_hyp.pb + cpp_hyp.pnb
        assert math.isclose(py_total, cpp_total, abs_tol=1e-10), (
            f"Hyp {i} total prob mismatch: Python {py_total} vs C++ {cpp_total}"
        )
        # Compare nodes
        assert len(py_nodes) == len(cpp_hyp.nodes), (
            f"Hyp {i} node count mismatch: Python {len(py_nodes)} vs C++ {len(cpp_hyp.nodes)}"
        )
        for j, (py_node, cpp_node) in enumerate(zip(py_nodes, cpp_hyp.nodes)):
            assert py_node["token"] == cpp_node.token, (
                f"Hyp {i} node {j} token mismatch: {py_node['token']} vs {cpp_node.token}"
            )
            assert py_node["frame"] == cpp_node.frame, (
                f"Hyp {i} node {j} frame mismatch: {py_node['frame']} vs {cpp_node.frame}"
            )
            assert math.isclose(py_node["prob"], cpp_node.prob, abs_tol=1e-7), (
                f"Hyp {i} node {j} prob mismatch: {py_node['prob']} vs {cpp_node.prob}"
            )

    print("  [PASS] test_beam_search_single_frame")


def test_beam_search_multi_frame():
    """Compare multi-frame streaming: Python vs C++."""
    vocab_size = 50
    keywords_idxset = {0, 5, 10, 15, 20}
    score_beam_size = 3
    path_beam_size = 20
    num_frames = 30

    torch.manual_seed(456)

    # Python path
    cur_hyps_py = [(tuple(), (1.0, 0.0, []))]

    # C++ path
    decoder = StreamingCTCDecoder(
        score_beam_size=score_beam_size,
        path_beam_size=path_beam_size,
    )
    decoder._decoder.set_keywords([], list(keywords_idxset), [])

    for t in range(num_frames):
        probs = torch.softmax(torch.randn(vocab_size), dim=0)

        # Python
        cur_hyps_py = streaming_ctc_prefix_beam_search_step_ref(
            t, probs, cur_hyps_py, keywords_idxset, score_beam_size
        )[:path_beam_size]

        # C++
        decoder.advance_frame(t, probs)

        # Compare after each frame
        assert len(decoder.get_hypotheses()) == len(cur_hyps_py), (
            f"Frame {t}: hyp count mismatch: C++ {len(decoder.get_hypotheses())} vs Python {len(cur_hyps_py)}"
        )

        result_cpp = decoder.get_hypotheses()
        for i, ((py_prefix, (py_pb, py_pnb, py_nodes)), cpp_hyp) in enumerate(zip(cur_hyps_py, result_cpp)):
            py_total = py_pb + py_pnb
            cpp_total = cpp_hyp.pb + cpp_hyp.pnb
            assert math.isclose(py_total, cpp_total, abs_tol=1e-9), (
                f"Frame {t} hyp {i}: total prob mismatch: Python {py_total} vs C++ {cpp_total}"
            )
            assert list(py_prefix) == cpp_hyp.prefix, (
                f"Frame {t} hyp {i}: prefix mismatch: Python {list(py_prefix)} vs C++ {cpp_hyp.prefix}"
            )
            assert len(py_nodes) == len(cpp_hyp.nodes), (
                f"Frame {t} hyp {i}: node count mismatch: Python {len(py_nodes)} vs C++ {len(cpp_hyp.nodes)}"
            )

    print("  [PASS] test_beam_search_multi_frame")


def test_beam_search_with_keywords():
    """Test beam search with actual keyword configuration."""
    # Simulate keyword token setup
    keywords_token = {
        "嗨小问": {"token_id": (10, 15, 20)},
        "你好问问": {"token_id": (5, 12, 5, 12)},
    }
    keywords_idxset = {0, 5, 10, 12, 15, 20}
    score_beam_size = 3
    path_beam_size = 20

    torch.manual_seed(789)
    vocab_size = 50
    num_frames = 50

    # Python path
    cur_hyps_py = [(tuple(), (1.0, 0.0, []))]

    # C++ path
    decoder = StreamingCTCDecoder(
        score_beam_size=score_beam_size,
        path_beam_size=path_beam_size,
    )
    decoder.set_keywords(keywords_token, keywords_idxset)

    for t in range(num_frames):
        probs = torch.softmax(torch.randn(vocab_size), dim=0)

        # Python
        cur_hyps_py = streaming_ctc_prefix_beam_search_step_ref(
            t, probs, cur_hyps_py, keywords_idxset, score_beam_size
        )[:path_beam_size]

        # C++
        decoder.advance_frame(t, probs)

        result_cpp = decoder.get_hypotheses()
        for i, ((py_prefix, (py_pb, py_pnb, py_nodes)), cpp_hyp) in enumerate(zip(cur_hyps_py, result_cpp)):
            py_total = py_pb + py_pnb
            cpp_total = cpp_hyp.pb + cpp_hyp.pnb
            assert math.isclose(py_total, cpp_total, abs_tol=1e-9), (
                f"Frame {t} hyp {i}: total prob mismatch: Python {py_total} vs C++ {cpp_total}"
            )
            assert list(py_prefix) == cpp_hyp.prefix, (
                f"Frame {t} hyp {i}: prefix mismatch"
            )

    print("  [PASS] test_beam_search_with_keywords")


def test_detection_logic():
    """Test keyword detection logic."""
    keywords_token = {
        "嗨小问": {"token_id": (10, 15, 20)},
    }
    keywords_idxset = {0, 10, 15, 20}
    threshold_map = {"嗨小问": 0.1}

    decoder = StreamingCTCDecoder(
        score_beam_size=3,
        path_beam_size=20,
        min_frames=2,
        max_frames=100,
        interval_frames=10,
    )
    decoder.set_keywords(keywords_token, keywords_idxset)
    decoder.set_thresholds(threshold_map)

    # Craft probabilities that will produce the keyword sequence
    vocab_size = 50
    torch.manual_seed(0)

    # Feed frames, inject high-probability tokens for keyword
    keyword_tokens = [10, 15, 20]
    for t in range(20):
        probs = torch.softmax(torch.randn(vocab_size), dim=0)
        # Boost keyword token probability
        if t < len(keyword_tokens):
            probs[keyword_tokens[t]] = 0.9
            probs = probs / probs.sum()  # re-normalize
        decoder.advance_frame(t, probs)

    # After feeding, check detection
    result = decoder.execute_detection(disable_threshold=True)
    # We don't necessarily trigger (depends on score), but should not crash
    if result is not None:
        assert "keyword" in result
        assert "candidate_score" in result
        print(f"  [INFO] Detection result: {result}")

    # Test get_best_decode
    best = decoder.get_best_decode_result()
    if best.get("candidate_keyword") is not None:
        print(f"  [INFO] Best decode: keyword={best['candidate_keyword']}, score={best['candidate_score']}")

    print("  [PASS] test_detection_logic")


def test_reset():
    """Test full reset functionality."""
    decoder = StreamingCTCDecoder(score_beam_size=3, path_beam_size=20)
    decoder._decoder.set_keywords([], [0, 5], [])

    vocab_size = 50
    torch.manual_seed(42)
    probs = torch.softmax(torch.randn(vocab_size), dim=0)
    decoder.advance_frame(0, probs)

    assert decoder.num_hypotheses > 0, "Should have hypotheses after advance"

    decoder.reset()
    assert decoder.num_hypotheses == 1, f"Should have 1 hypothesis after reset, got {decoder.num_hypotheses}"

    hyps = decoder.get_hypotheses()
    assert len(hyps[0].prefix) == 0, "Reset hypothesis should have empty prefix"
    assert math.isclose(hyps[0].pb, 1.0), "Reset hypothesis pb should be 1.0"
    assert math.isclose(hyps[0].pnb, 0.0), "Reset hypothesis pnb should be 0.0"
    best = decoder.get_best_decode_result()
    assert best["candidate_keyword"] is None, "Full reset should clear best decode history"

    print("  [PASS] test_reset")


def test_reset_beam_search_preserves_history():
    """Match Python reset_decode_state: only clear beam, keep history."""
    keywords_token = {"嗨小问": {"token_id": (10, 15, 20)}}
    keywords_idxset = {0, 10, 15, 20}
    threshold_map = {"嗨小问": 0.1}

    decoder = StreamingCTCDecoder(
        score_beam_size=3,
        path_beam_size=20,
        min_frames=2,
        max_frames=100,
        interval_frames=10,
    )
    decoder.set_keywords(keywords_token, keywords_idxset)
    decoder.set_thresholds(threshold_map)

    vocab_size = 50
    keyword_tokens = [10, 15, 20]

    def make_probs(token_id: int) -> torch.Tensor:
        probs = torch.full((vocab_size,), 1e-4, dtype=torch.float32)
        probs[0] = 0.02
        probs[token_id] = 0.9
        probs = probs / probs.sum()
        return probs

    for frame, token_id in enumerate(keyword_tokens):
        decoder.advance_frame(frame, make_probs(token_id))
    first_result = decoder.execute_detection(disable_threshold=True)
    assert first_result is not None, "First keyword should trigger detection"

    best_before_reset = decoder.get_best_decode_result()
    decoder.reset_beam_search()
    best_after_reset = decoder.get_best_decode_result()
    assert best_after_reset == best_before_reset, "Beam reset should preserve best decode history"
    assert decoder.num_hypotheses == 1, "Beam reset should restore a single empty hypothesis"

    for frame, token_id in enumerate(keyword_tokens, start=5):
        decoder.advance_frame(frame, make_probs(token_id))
    second_result = decoder.execute_detection(disable_threshold=True)
    assert second_result is None, "Interval suppression should survive beam-only reset"

    print("  [PASS] test_reset_beam_search_preserves_history")


def test_get_first_hyp_start_frame():
    """Test get_first_hyp_start_frame for timeout reset."""
    decoder = StreamingCTCDecoder(score_beam_size=3, path_beam_size=20)
    decoder._decoder.set_keywords([], [0, 5, 10], [])

    vocab_size = 50
    torch.manual_seed(42)

    # No frames processed
    assert decoder.get_first_hyp_start_frame() == -1, "Should be -1 with no frames"

    # Process one frame
    probs = torch.softmax(torch.randn(vocab_size), dim=0)
    probs[5] = 0.8
    probs = probs / probs.sum()
    decoder.advance_frame(10, probs)

    # If a hypothesis now has nodes, start_frame should be the frame of first node
    frame = decoder.get_first_hyp_start_frame()
    print(f"  [INFO] First hyp start frame after advance: {frame}")

    print("  [PASS] test_get_first_hyp_start_frame")


def test_empty_filter():
    """Test that empty filter (no tokens pass threshold) preserves state."""
    decoder = StreamingCTCDecoder(score_beam_size=3, path_beam_size=20)
    decoder._decoder.set_keywords([], [0, 5, 10], [])

    vocab_size = 50

    # All probabilities very low
    probs = torch.ones(vocab_size) * 1e-10
    probs = probs / probs.sum()

    decoder.advance_frame(0, probs)
    assert decoder.num_hypotheses == 1, "Should keep initial hypothesis"

    hyps = decoder.get_hypotheses()
    assert len(hyps[0].prefix) == 0, "Prefix should remain empty"

    print("  [PASS] test_empty_filter")


def test_c_style_matches_cpp_multi_frame():
    """Compare C-style decoder against current C++ decoder frame by frame."""
    keywords_token = {
        "嗨小问": {"token_id": (10, 15, 20)},
        "你好问问": {"token_id": (5, 12, 5, 12)},
    }
    keywords_idxset = {0, 5, 10, 12, 15, 20}
    decoder_cpp = StreamingCTCDecoder(score_beam_size=3, path_beam_size=20, min_frames=2, max_frames=100, interval_frames=10)
    decoder_c = StreamingCTCDecoderC(score_beam_size=3, path_beam_size=20, min_frames=2, max_frames=100, interval_frames=10)
    decoder_cpp.set_keywords(keywords_token, keywords_idxset)
    decoder_c.set_keywords(keywords_token, keywords_idxset)

    vocab_size = 50
    torch.manual_seed(2026)
    for frame_index in range(40):
        probs = torch.softmax(torch.randn(vocab_size), dim=0)
        if frame_index % 9 == 0:
            probs[10] = 0.88
            probs = probs / probs.sum()
        decoder_cpp.advance_frame(frame_index, probs)
        decoder_c.advance_frame(frame_index, probs)
        assert_hypotheses_match(
            decoder_cpp.get_hypotheses(),
            decoder_c.get_hypotheses(),
            f"frame {frame_index}",
        )
        assert_detection_match(
            decoder_cpp.execute_detection(disable_threshold=True),
            decoder_c.execute_detection(disable_threshold=True),
            f"frame {frame_index} detection",
        )

    assert_best_decode_match(
        decoder_cpp.get_best_decode_result(),
        decoder_c.get_best_decode_result(),
        "multi-frame best decode",
    )
    print("  [PASS] test_c_style_matches_cpp_multi_frame")


def test_c_style_matches_cpp_reset_and_detection():
    """Compare C-style decoder against current C++ decoder on trigger and reset semantics."""
    keywords_token = {"嗨小问": {"token_id": (10, 15, 20)}}
    keywords_idxset = {0, 10, 15, 20}
    threshold_map = {"嗨小问": 0.1}

    decoder_cpp = StreamingCTCDecoder(score_beam_size=3, path_beam_size=20, min_frames=2, max_frames=100, interval_frames=10)
    decoder_c = StreamingCTCDecoderC(score_beam_size=3, path_beam_size=20, min_frames=2, max_frames=100, interval_frames=10)
    decoder_cpp.set_keywords(keywords_token, keywords_idxset)
    decoder_c.set_keywords(keywords_token, keywords_idxset)
    decoder_cpp.set_thresholds(threshold_map)
    decoder_c.set_thresholds(threshold_map)

    vocab_size = 50

    def make_probs(token_id: int) -> torch.Tensor:
        probs = torch.full((vocab_size,), 1e-4, dtype=torch.float32)
        probs[0] = 0.02
        probs[token_id] = 0.9
        return probs / probs.sum()

    for frame_index, token_id in enumerate((10, 15, 20)):
        probs = make_probs(token_id)
        cpp_result = decoder_cpp.step_and_detect(frame_index, probs, disable_threshold=True)
        c_result = decoder_c.step_and_detect(frame_index, probs, disable_threshold=True)
        assert_detection_match(cpp_result, c_result, f"trigger frame {frame_index}")
        assert_hypotheses_match(decoder_cpp.get_hypotheses(), decoder_c.get_hypotheses(), f"trigger frame {frame_index} hyps")

    assert_best_decode_match(decoder_cpp.get_best_decode_result(), decoder_c.get_best_decode_result(), "best before reset")
    decoder_cpp.reset_beam_search()
    decoder_c.reset_beam_search()
    assert_hypotheses_match(decoder_cpp.get_hypotheses(), decoder_c.get_hypotheses(), "after reset_beam_search")
    assert_best_decode_match(decoder_cpp.get_best_decode_result(), decoder_c.get_best_decode_result(), "best after beam reset")

    decoder_cpp.reset()
    decoder_c.reset()
    assert_hypotheses_match(decoder_cpp.get_hypotheses(), decoder_c.get_hypotheses(), "after full reset")
    assert_best_decode_match(decoder_cpp.get_best_decode_result(), decoder_c.get_best_decode_result(), "best after full reset")
    print("  [PASS] test_c_style_matches_cpp_reset_and_detection")


def main():
    print("=" * 60)
    print("C++ CTC Decoder Comparison Tests")
    print("=" * 60)

    tests = [
        test_beam_search_single_frame,
        test_beam_search_multi_frame,
        test_beam_search_with_keywords,
        test_detection_logic,
        test_reset,
        test_reset_beam_search_preserves_history,
        test_get_first_hyp_start_frame,
        test_empty_filter,
        test_c_style_matches_cpp_multi_frame,
        test_c_style_matches_cpp_reset_and_detection,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print()
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed > 0:
        sys.exit(1)
    print("All tests passed!")


if __name__ == "__main__":
    main()
