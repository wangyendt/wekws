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
from torch2lite.ctc_decoder_pybind import StreamingCTCDecoder


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
