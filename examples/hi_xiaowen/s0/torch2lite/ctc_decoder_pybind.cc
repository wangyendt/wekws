#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "ctc_decoder.h"

namespace py = pybind11;
using namespace ctc_decoder;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Streaming CTC prefix beam search decoder for KWS";

    py::class_<TokenNode>(m, "TokenNode")
        .def_readonly("token", &TokenNode::token)
        .def_readonly("frame", &TokenNode::frame)
        .def_readonly("prob", &TokenNode::prob);

    py::class_<Hypothesis>(m, "Hypothesis")
        .def_readonly("prefix", &Hypothesis::prefix)
        .def_readonly("pb", &Hypothesis::pb)
        .def_readonly("pnb", &Hypothesis::pnb)
        .def_readonly("nodes", &Hypothesis::nodes);

    py::class_<StreamingCTCDecoder>(m, "StreamingCTCDecoder")
        .def(
            py::init<int32_t, int32_t, int32_t, int32_t, int32_t>(),
            py::arg("score_beam_size") = 3,
            py::arg("path_beam_size") = 20,
            py::arg("min_frames") = 5,
            py::arg("max_frames") = 250,
            py::arg("interval_frames") = 50)
        .def(
            "set_keywords",
            [](StreamingCTCDecoder& self,
               const std::vector<std::pair<int32_t, std::vector<int32_t>>>& keywords_tokens,
               const std::vector<int32_t>& keywords_idxset,
               const std::vector<std::string>& keyword_strings) {
                std::unordered_set<int32_t> idxset(keywords_idxset.begin(), keywords_idxset.end());
                self.set_keywords(keywords_tokens, idxset);
                self.set_keyword_strings(keyword_strings);
            },
            py::arg("keywords_tokens"),
            py::arg("keywords_idxset"),
            py::arg("keyword_strings"))
        .def("set_thresholds", &StreamingCTCDecoder::set_thresholds, py::arg("threshold_map"))
        .def("advance_frame", &StreamingCTCDecoder::advance_frame, py::arg("frame_index"), py::arg("probs"))
        .def(
            "execute_detection",
            [](StreamingCTCDecoder& self, bool disable_threshold) -> py::object {
                auto result = self.execute_detection(disable_threshold);
                if (!result.valid) {
                    return py::none();
                }
                py::dict dict;
                dict["keyword"] = result.keyword;
                dict["candidate_score"] = result.score;
                dict["start_frame"] = result.start_frame;
                dict["end_frame"] = result.end_frame;
                return std::move(dict);
            },
            py::arg("disable_threshold"))
        .def(
            "step_and_detect",
            [](StreamingCTCDecoder& self, int32_t frame_index, torch::Tensor probs, bool disable_threshold) -> py::object {
                auto result = self.step_and_detect(frame_index, probs, disable_threshold);
                if (!result.valid) {
                    return py::none();
                }
                py::dict dict;
                dict["keyword"] = result.keyword;
                dict["candidate_score"] = result.score;
                dict["start_frame"] = result.start_frame;
                dict["end_frame"] = result.end_frame;
                return std::move(dict);
            },
            py::arg("frame_index"),
            py::arg("probs"),
            py::arg("disable_threshold"))
        .def("reset", &StreamingCTCDecoder::reset)
        .def("reset_beam_search", &StreamingCTCDecoder::reset_beam_search)
        .def(
            "get_best_decode",
            [](const StreamingCTCDecoder& self) -> py::dict {
                auto result = self.get_best_decode();
                py::dict dict;
                if (!result.valid) {
                    dict["candidate_keyword"] = py::none();
                    dict["candidate_score"] = py::none();
                    dict["start_frame"] = py::none();
                    dict["end_frame"] = py::none();
                } else {
                    dict["candidate_keyword"] = result.keyword;
                    dict["candidate_score"] = result.score;
                    dict["start_frame"] = result.start_frame;
                    dict["end_frame"] = result.end_frame;
                }
                return dict;
            })
        .def("get_first_hyp_start_frame", &StreamingCTCDecoder::get_first_hyp_start_frame)
        .def("num_hypotheses", &StreamingCTCDecoder::num_hypotheses)
        .def("get_hypotheses", &StreamingCTCDecoder::get_hypotheses);
}
