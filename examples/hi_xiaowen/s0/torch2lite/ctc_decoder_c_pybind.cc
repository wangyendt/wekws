#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ctc_decoder_c.h"

namespace py = pybind11;

namespace {

#if CTC_DECODER_C_ENABLE_DEBUG_HYPOTHESES
struct TokenNode {
    int32_t token;
    int32_t frame;
    float prob;
};

struct Hypothesis {
    std::vector<int32_t> prefix;
    double pb;
    double pnb;
    std::vector<TokenNode> nodes;
};
#endif

class StreamingCTCDecoderCStyle {
public:
    StreamingCTCDecoderCStyle(
        int32_t score_beam_size,
        int32_t path_beam_size,
        int32_t min_frames,
        int32_t max_frames,
        int32_t interval_frames,
        int32_t frame_step) {
        CTCDecoderCConfig config;
        ctc_decoder_c_init_default_config(&config);
        config.score_beam_size = score_beam_size;
        config.path_beam_size = path_beam_size;
        config.min_frames = min_frames;
        config.max_frames = max_frames;
        config.interval_frames = interval_frames;
        config.frame_step = frame_step;
        if (ctc_decoder_c_init(&state_, &config) != 0) {
            throw std::runtime_error("failed to initialize C-style CTC decoder");
        }
    }

    ~StreamingCTCDecoderCStyle() {
        ctc_decoder_c_free(&state_);
    }

    void set_keywords(
        const std::vector<std::pair<int32_t, std::vector<int32_t>>>& keywords_tokens,
        const std::vector<int32_t>& keywords_idxset,
        const std::vector<std::string>& keyword_strings) {
        std::vector<CTCDecoderCKeyword> keywords;
        std::vector<const char*> keyword_string_ptrs;
        keywords.reserve(keywords_tokens.size());
        keyword_string_ptrs.reserve(keyword_strings.size());
        for (const auto& item : keywords_tokens) {
            CTCDecoderCKeyword keyword;
            keyword.word_index = item.first;
            keyword.token_ids = item.second.empty() ? nullptr : item.second.data();
            keyword.token_count = static_cast<int32_t>(item.second.size());
            keywords.push_back(keyword);
        }
        for (const auto& item : keyword_strings) {
            keyword_string_ptrs.push_back(item.c_str());
        }
        if (ctc_decoder_c_set_keywords(
                &state_,
                keywords.empty() ? nullptr : keywords.data(),
                static_cast<int32_t>(keywords.size()),
                keywords_idxset.empty() ? nullptr : keywords_idxset.data(),
                static_cast<int32_t>(keywords_idxset.size()),
                keyword_string_ptrs.empty() ? nullptr : keyword_string_ptrs.data()) != 0) {
            throw std::runtime_error("failed to set C-style decoder keywords");
        }
    }

    void set_thresholds(const std::unordered_map<int32_t, float>& threshold_map) {
        std::vector<int32_t> word_indices;
        std::vector<float> threshold_values;
        word_indices.reserve(threshold_map.size());
        threshold_values.reserve(threshold_map.size());
        for (const auto& item : threshold_map) {
            word_indices.push_back(item.first);
            threshold_values.push_back(item.second);
        }
        if (ctc_decoder_c_set_thresholds(
                &state_,
                word_indices.empty() ? nullptr : word_indices.data(),
                threshold_values.empty() ? nullptr : threshold_values.data(),
                static_cast<int32_t>(word_indices.size())) != 0) {
            throw std::runtime_error("failed to set C-style decoder thresholds");
        }
    }

    void advance_frame(int32_t frame_index, torch::Tensor probs) {
        auto probs_contig = validate_probs(probs);
        if (ctc_decoder_c_advance_frame(
                &state_,
                frame_index,
                probs_contig.data_ptr<float>(),
                static_cast<int32_t>(probs_contig.numel())) != 0) {
            throw std::runtime_error("C-style decoder advance_frame failed");
        }
    }

    py::object execute_detection(bool disable_threshold) {
        return detection_to_py(ctc_decoder_c_execute_detection(&state_, disable_threshold ? 1 : 0));
    }

    py::object step_and_detect(int32_t frame_index, torch::Tensor probs, bool disable_threshold) {
        auto probs_contig = validate_probs(probs);
        return detection_to_py(
            ctc_decoder_c_step_and_detect(
                &state_,
                frame_index,
                probs_contig.data_ptr<float>(),
                static_cast<int32_t>(probs_contig.numel()),
                disable_threshold ? 1 : 0));
    }

    py::object step_and_detect_next(torch::Tensor probs, bool disable_threshold) {
        auto probs_contig = validate_probs(probs);
        return detection_to_py(
            ctc_decoder_c_step_and_detect_next(
                &state_,
                probs_contig.data_ptr<float>(),
                static_cast<int32_t>(probs_contig.numel()),
                disable_threshold ? 1 : 0));
    }

    void reset() {
        ctc_decoder_c_reset(&state_);
    }

    void reset_beam_search() {
        ctc_decoder_c_reset_beam_search(&state_);
    }

#if CTC_DECODER_C_ENABLE_EXTENDED_API
    py::dict get_best_decode() const {
        auto result = ctc_decoder_c_get_best_decode(&state_);
        py::dict dict;
        if (!result.valid) {
            dict["candidate_keyword"] = py::none();
            dict["candidate_score"] = py::none();
            dict["start_frame"] = py::none();
            dict["end_frame"] = py::none();
        } else {
            if (result.keyword) {
                dict["candidate_keyword"] = py::str(result.keyword);
            } else {
                dict["candidate_keyword"] = py::none();
            }
            dict["candidate_score"] = result.score;
            dict["start_frame"] = result.start_frame;
            dict["end_frame"] = result.end_frame;
        }
        return dict;
    }

    int32_t get_first_hyp_start_frame() const {
        return ctc_decoder_c_get_first_hyp_start_frame(&state_);
    }
#endif

    int32_t num_hypotheses() const {
        return ctc_decoder_c_num_hypotheses(&state_);
    }

#if CTC_DECODER_C_ENABLE_DEBUG_HYPOTHESES
    std::vector<Hypothesis> get_hypotheses() const {
        std::vector<Hypothesis> hyps;
        int32_t hyp_count = ctc_decoder_c_num_hypotheses(&state_);
        hyps.reserve(hyp_count);
        for (int32_t index = 0; index < hyp_count; ++index) {
            const CTCDecoderCHypothesis* hyp = ctc_decoder_c_get_hypothesis(&state_, index);
            if (!hyp) {
                throw std::runtime_error("failed to materialize C-style decoder hypothesis");
            }
            Hypothesis public_hyp;
            public_hyp.pb = hyp->pb;
            public_hyp.pnb = hyp->pnb;
            public_hyp.prefix.assign(hyp->prefix, hyp->prefix + hyp->prefix_len);
            public_hyp.nodes.reserve(hyp->node_count);
            for (int32_t node_index = 0; node_index < hyp->node_count; ++node_index) {
                TokenNode node;
                node.token = hyp->nodes[node_index].token;
                node.frame = hyp->nodes[node_index].frame;
                node.prob = hyp->nodes[node_index].prob;
                public_hyp.nodes.push_back(node);
            }
            hyps.push_back(std::move(public_hyp));
        }
        return hyps;
    }
#endif

private:
    static torch::Tensor validate_probs(torch::Tensor probs) {
        TORCH_CHECK(probs.dim() == 1, "probs must be 1D, got shape=", probs.sizes());
        TORCH_CHECK(probs.device().is_cpu(), "probs must be on CPU");
        TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32, got ", probs.scalar_type());
        return probs.contiguous();
    }

    static py::object detection_to_py(const CTCDecoderCDetectionResult& result) {
        if (!result.valid) {
            return py::none();
        }
        py::dict dict;
        if (result.keyword) {
            dict["keyword"] = py::str(result.keyword);
        } else {
            dict["keyword"] = py::none();
        }
        dict["candidate_score"] = result.score;
        dict["start_frame"] = result.start_frame;
        dict["end_frame"] = result.end_frame;
        return dict;
    }

    CTCDecoderCState state_{};
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "C-style streaming CTC prefix beam search decoder for KWS";

#if CTC_DECODER_C_ENABLE_DEBUG_HYPOTHESES
    py::class_<TokenNode>(m, "TokenNode")
        .def_readonly("token", &TokenNode::token)
        .def_readonly("frame", &TokenNode::frame)
        .def_readonly("prob", &TokenNode::prob);

    py::class_<Hypothesis>(m, "Hypothesis")
        .def_readonly("prefix", &Hypothesis::prefix)
        .def_readonly("pb", &Hypothesis::pb)
        .def_readonly("pnb", &Hypothesis::pnb)
        .def_readonly("nodes", &Hypothesis::nodes);
#endif

    py::class_<StreamingCTCDecoderCStyle>(m, "StreamingCTCDecoderCStyle")
        .def(
            py::init<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t>(),
            py::arg("score_beam_size") = 3,
            py::arg("path_beam_size") = 20,
            py::arg("min_frames") = 5,
            py::arg("max_frames") = 250,
            py::arg("interval_frames") = 50,
            py::arg("frame_step") = 1)
        .def("set_keywords", &StreamingCTCDecoderCStyle::set_keywords, py::arg("keywords_tokens"), py::arg("keywords_idxset"), py::arg("keyword_strings"))
        .def("set_thresholds", &StreamingCTCDecoderCStyle::set_thresholds, py::arg("threshold_map"))
        .def("advance_frame", &StreamingCTCDecoderCStyle::advance_frame, py::arg("frame_index"), py::arg("probs"))
        .def("execute_detection", &StreamingCTCDecoderCStyle::execute_detection, py::arg("disable_threshold"))
        .def("step_and_detect", &StreamingCTCDecoderCStyle::step_and_detect, py::arg("frame_index"), py::arg("probs"), py::arg("disable_threshold"))
        .def("step_and_detect_next", &StreamingCTCDecoderCStyle::step_and_detect_next, py::arg("probs"), py::arg("disable_threshold"))
        .def("reset", &StreamingCTCDecoderCStyle::reset)
        .def("reset_beam_search", &StreamingCTCDecoderCStyle::reset_beam_search)
#if CTC_DECODER_C_ENABLE_EXTENDED_API
        .def("get_best_decode", &StreamingCTCDecoderCStyle::get_best_decode)
        .def("get_first_hyp_start_frame", &StreamingCTCDecoderCStyle::get_first_hyp_start_frame)
#endif
        .def("num_hypotheses", &StreamingCTCDecoderCStyle::num_hypotheses)
#if CTC_DECODER_C_ENABLE_DEBUG_HYPOTHESES
        .def("get_hypotheses", &StreamingCTCDecoderCStyle::get_hypotheses)
#endif
        ;
}
