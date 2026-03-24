#!/usr/bin/env python3
"""Find the first divergence frame between Python and C++ beam search."""

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import infer_wav as iw
import infer_wav_stream as iws
from torch2lite.ctc_decoder_pybind import StreamingCTCDecoder


def py_beam_step(frame_index, probs, cur_hyps, kws_set, score_beam_size):
    next_hyps = defaultdict(lambda: (0.0, 0.0, []))
    top_k_probs, top_k_index = probs.topk(score_beam_size)
    filter_index = []
    for prob, token_id in zip(top_k_probs.tolist(), top_k_index.tolist()):
        if prob > 0.05 and token_id in kws_set:
            filter_index.append(token_id)
    if not filter_index:
        return cur_hyps

    for token_id in filter_index:
        token_prob = probs[token_id].item()
        for prefix, (pb, pnb, cur_nodes) in cur_hyps:
            last = prefix[-1] if prefix else None
            if token_id == 0:
                n_pb, n_pnb, _ = next_hyps[prefix]
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


def main():
    wav_path = Path("speech_charctc_kws_phone-xiaoyun/example/kws_xiaoyunxiaoyun.wav").resolve()
    model_args = type(
        "Args",
        (),
        {
            "model": "s3",
            "checkpoint": "",
            "model_dir": "",
            "checkpoint_name": "",
            "config": "",
            "dict_dir": "",
            "stats_dir": "",
            "threshold_map": "",
            "pick_mode": "legacy",
            "frr_eps": 0.001,
        },
    )()
    model_info = iw.resolve_model_paths(model_args)
    configs = iw.load_config(model_info["config"])
    model, device, model_type = iw.load_model(model_info["checkpoint"], configs, -1)
    keywords = ["嗨小问", "你好问问"]
    keywords_token, keywords_idxset = iw.build_keyword_token_info(keywords, model_info["dict_dir"])
    downsampling = int(configs["dataset_conf"].get("frame_skip", 1))

    streamer = iws.StreamingKeywordSpotter(
        configs=configs,
        model=model,
        device=device,
        model_type=model_type,
        keywords=keywords,
        keywords_token=keywords_token,
        keywords_idxset=keywords_idxset,
        threshold_map={"嗨小问": 0.272, "你好问问": 0.016},
        score_beam_size=3,
        path_beam_size=20,
        min_frames=5,
        max_frames=250,
        interval_frames=50,
        disable_threshold=True,
    )
    waveform = iw.load_wav_and_resample(wav_path, streamer.sample_rate)
    waveform = waveform.squeeze(0).numpy()
    pcm = np.clip(np.round(waveform * (1 << 15)), -32768, 32767).astype(np.int16)
    chunk_samples = max(1, int(300.0 / 1000.0 * streamer.sample_rate))

    all_probs = []
    total_frames = 0
    for start in range(0, pcm.shape[0], chunk_samples):
        feats = streamer.accept_wave_chunk(pcm[start:start + chunk_samples])
        if feats is None or feats.size(0) < 1:
            continue
        probs = streamer._forward_model(feats)
        for local_frame, prob in enumerate(probs):
            all_probs.append((total_frames + local_frame * downsampling, prob))
        total_frames += len(probs) * downsampling

    print(f"total frames: {len(all_probs)}, downsampling: {downsampling}")

    py_hyps = [(tuple(), (1.0, 0.0, []))]
    cpp_decoder = StreamingCTCDecoder(3, 20, 5, 250, 50)
    cpp_decoder.set_keywords(keywords_token, keywords_idxset)

    for absolute_frame, prob in all_probs:
        py_hyps = py_beam_step(absolute_frame, prob, py_hyps, keywords_idxset, 3)[:20]
        cpp_decoder.advance_frame(absolute_frame, prob)
        cpp_hyps = cpp_decoder.get_hypotheses()

        if len(py_hyps) != cpp_decoder.num_hypotheses:
            print(
                f"DIVERGE count frame {absolute_frame}: py={len(py_hyps)} cpp={cpp_decoder.num_hypotheses}"
            )
            return

        for hyp_index, ((py_prefix, (py_pb, py_pnb, py_nodes)), cpp_hyp) in enumerate(
            zip(py_hyps, cpp_hyps)
        ):
            py_total = py_pb + py_pnb
            cpp_total = cpp_hyp.pb + cpp_hyp.pnb
            if list(py_prefix) != cpp_hyp.prefix:
                print(
                    f"DIVERGE prefix frame {absolute_frame} hyp {hyp_index}: py={list(py_prefix)} cpp={cpp_hyp.prefix}"
                )
                return
            if not math.isclose(py_total, cpp_total, abs_tol=1e-10):
                print(
                    f"DIVERGE prob frame {absolute_frame} hyp {hyp_index}: py={py_total:.12f} cpp={cpp_total:.12f}"
                )
                return
            if len(py_nodes) != len(cpp_hyp.nodes):
                print(
                    f"DIVERGE node-count frame {absolute_frame} hyp {hyp_index}: py={len(py_nodes)} cpp={len(cpp_hyp.nodes)}"
                )
                return
            for node_index, (py_node, cpp_node) in enumerate(zip(py_nodes, cpp_hyp.nodes)):
                if (
                    py_node["token"] != cpp_node.token
                    or py_node["frame"] != cpp_node.frame
                    or not math.isclose(py_node["prob"], cpp_node.prob, abs_tol=1e-9)
                ):
                    print(f"DIVERGE node frame {absolute_frame} hyp {hyp_index} node {node_index}")
                    print(f"  py : {py_node}")
                    print(
                        "  cpp: {token=%d, frame=%d, prob=%.10f}"
                        % (cpp_node.token, cpp_node.frame, cpp_node.prob)
                    )
                    return

    print(f"All {len(all_probs)} frames matched perfectly")


if __name__ == "__main__":
    main()
