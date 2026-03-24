#include "ctc_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/extension.h>

namespace ctc_decoder {
namespace {

struct VectorHash {
    size_t operator()(const std::vector<int32_t>& value) const {
        size_t seed = value.size();
        for (int32_t item : value) {
            seed ^= static_cast<size_t>(item) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

std::string keyword_or_empty(const std::vector<std::string>& keyword_strings, int32_t word_index) {
    if (word_index < 0 || static_cast<size_t>(word_index) >= keyword_strings.size()) {
        return std::string();
    }
    return keyword_strings[word_index];
}

std::shared_ptr<TokenNode> make_node(int32_t token, int32_t frame, float prob) {
    return std::make_shared<TokenNode>(TokenNode{token, frame, prob});
}

}  // namespace

StreamingCTCDecoder::StreamingCTCDecoder(
    int32_t score_beam_size,
    int32_t path_beam_size,
    int32_t min_frames,
    int32_t max_frames,
    int32_t interval_frames)
    : score_beam_size_(score_beam_size),
      path_beam_size_(path_beam_size),
      min_frames_(min_frames),
      max_frames_(max_frames),
      interval_frames_(interval_frames) {
    init_state();
}

void StreamingCTCDecoder::set_keyword_strings(const std::vector<std::string>& keyword_strings) {
    keyword_strings_ = keyword_strings;
}

void StreamingCTCDecoder::set_keywords(
    const std::vector<std::pair<int32_t, std::vector<int32_t>>>& keywords_tokens,
    const std::unordered_set<int32_t>& keywords_idxset) {
    keywords_tokens_ = keywords_tokens;
    keywords_idxset_ = keywords_idxset;
}

void StreamingCTCDecoder::set_thresholds(const std::unordered_map<int32_t, float>& threshold_map) {
    threshold_map_ = threshold_map;
}

void StreamingCTCDecoder::init_beam_state() {
    cur_hyps_.clear();
    InternalHypothesis hyp;
    hyp.pb = 1.0;
    hyp.pnb = 0.0;
    cur_hyps_.push_back(std::move(hyp));
}

void StreamingCTCDecoder::init_state() {
    init_beam_state();
    last_active_pos_ = -1;
    has_best_decode_ = false;
    best_word_index_ = -1;
    best_score_ = 0.0;
    best_start_frame_ = 0;
    best_end_frame_ = 0;
}

void StreamingCTCDecoder::reset() {
    init_state();
}

void StreamingCTCDecoder::reset_beam_search() {
    init_beam_state();
}

bool StreamingCTCDecoder::float_close_zero(double val, double abs_tol) {
    return std::abs(val) < abs_tol;
}

bool StreamingCTCDecoder::is_sublist(
    const std::vector<int32_t>& main_list,
    const std::vector<int32_t>& check_list,
    int32_t& offset_out) {
    size_t m = main_list.size();
    size_t c = check_list.size();
    if (m < c) {
        return false;
    }
    if (m == c) {
        if (main_list == check_list) {
            offset_out = 0;
            return true;
        }
        return false;
    }
    for (size_t i = 0; i <= m - c; i++) {
        bool match = true;
        for (size_t j = 0; j < c; j++) {
            if (main_list[i + j] != check_list[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            offset_out = static_cast<int32_t>(i);
            return true;
        }
    }
    return false;
}

void StreamingCTCDecoder::advance_beam(
    int32_t frame_index,
    const std::vector<std::pair<float, int32_t>>& filtered_tokens) {
    if (filtered_tokens.empty()) {
        return;
    }

    std::vector<InternalHypothesis> next_hyps;
    next_hyps.reserve(cur_hyps_.size() * filtered_tokens.size());
    std::unordered_map<std::vector<int32_t>, size_t, VectorHash> next_hyp_index;

    auto get_hyp = [&](const std::vector<int32_t>& prefix) -> InternalHypothesis& {
        auto it = next_hyp_index.find(prefix);
        if (it == next_hyp_index.end()) {
            InternalHypothesis hyp;
            hyp.prefix = prefix;
            hyp.pb = 0.0;
            hyp.pnb = 0.0;
            next_hyps.push_back(std::move(hyp));
            size_t index = next_hyps.size() - 1;
            next_hyp_index.emplace(next_hyps[index].prefix, index);
            return next_hyps[index];
        }
        return next_hyps[it->second];
    };

    for (const auto& [token_prob_raw, token_id] : filtered_tokens) {
        double token_prob = static_cast<double>(token_prob_raw);

        for (const auto& hyp : cur_hyps_) {
            int32_t last = hyp.prefix.empty() ? -1 : hyp.prefix.back();

            if (token_id == 0) {
                auto& next_hyp = get_hyp(hyp.prefix);
                next_hyp.nodes = hyp.nodes;
                next_hyp.pb += hyp.pb * token_prob + hyp.pnb * token_prob;
                continue;
            }

            if (token_id == last) {
                if (!float_close_zero(hyp.pnb)) {
                    auto& next_hyp = get_hyp(hyp.prefix);
                    next_hyp.nodes = hyp.nodes;
                    next_hyp.pnb += hyp.pnb * token_prob;
                    if (!next_hyp.nodes.empty() &&
                        token_prob > static_cast<double>(next_hyp.nodes.back()->prob)) {
                        next_hyp.nodes.back()->prob = token_prob_raw;
                        next_hyp.nodes.back()->frame = frame_index;
                    }
                }
                if (!float_close_zero(hyp.pb)) {
                    auto next_prefix = hyp.prefix;
                    next_prefix.push_back(token_id);
                    auto& next_hyp = get_hyp(next_prefix);
                    next_hyp.nodes = hyp.nodes;
                    next_hyp.pnb += hyp.pb * token_prob;
                    next_hyp.nodes.push_back(make_node(token_id, frame_index, token_prob_raw));
                }
                continue;
            }

            auto next_prefix = hyp.prefix;
            next_prefix.push_back(token_id);
            auto& next_hyp = get_hyp(next_prefix);
            if (!next_hyp.nodes.empty()) {
                if (token_prob > static_cast<double>(next_hyp.nodes.back()->prob)) {
                    next_hyp.nodes.pop_back();
                    next_hyp.nodes.push_back(make_node(token_id, frame_index, token_prob_raw));
                }
            } else {
                next_hyp.nodes = hyp.nodes;
                next_hyp.nodes.push_back(make_node(token_id, frame_index, token_prob_raw));
            }
            next_hyp.pnb += hyp.pb * token_prob + hyp.pnb * token_prob;
        }
    }

    std::stable_sort(
        next_hyps.begin(),
        next_hyps.end(),
        [](const InternalHypothesis& lhs, const InternalHypothesis& rhs) {
            return (lhs.pb + lhs.pnb) > (rhs.pb + rhs.pnb);
        });

    if (path_beam_size_ >= 0 && static_cast<size_t>(path_beam_size_) < next_hyps.size()) {
        next_hyps.resize(static_cast<size_t>(path_beam_size_));
    }
    cur_hyps_ = std::move(next_hyps);
}

void StreamingCTCDecoder::advance_frame(int32_t frame_index, torch::Tensor probs) {
    TORCH_CHECK(probs.dim() == 1, "probs must be 1D, got shape=", probs.sizes());
    TORCH_CHECK(probs.device().is_cpu(), "probs must be on CPU");
    TORCH_CHECK(
        probs.scalar_type() == torch::kFloat32,
        "probs must be float32, got ",
        probs.scalar_type());

    auto probs_contig = probs.contiguous();
    int32_t vocab_size = static_cast<int32_t>(probs_contig.numel());
    int32_t topk_count = std::min(score_beam_size_, vocab_size);
    if (topk_count <= 0) {
        return;
    }

    auto topk = probs_contig.topk(topk_count);
    auto top_values = std::get<0>(topk).contiguous();
    auto top_indices = std::get<1>(topk).contiguous();
    const float* value_ptr = top_values.data_ptr<float>();
    const int64_t* index_ptr = top_indices.data_ptr<int64_t>();

    std::vector<std::pair<float, int32_t>> filtered_tokens;
    filtered_tokens.reserve(static_cast<size_t>(topk_count));
    for (int32_t i = 0; i < topk_count; ++i) {
        float prob = value_ptr[i];
        int32_t token_id = static_cast<int32_t>(index_ptr[i]);
        if (prob <= 0.05f) {
            continue;
        }
        if (!keywords_idxset_.empty() && keywords_idxset_.find(token_id) == keywords_idxset_.end()) {
            continue;
        }
        filtered_tokens.emplace_back(prob, token_id);
    }

    advance_beam(frame_index, filtered_tokens);
}

StreamingCTCDecoder::DetectionResult StreamingCTCDecoder::execute_detection(bool disable_threshold) {
    for (const auto& hyp : cur_hyps_) {
        for (const auto& [word_idx, label] : keywords_tokens_) {
            int32_t offset = 0;
            if (!is_sublist(hyp.prefix, label, offset)) {
                continue;
            }

            double score = 1.0;
            int32_t start_frame = hyp.nodes[offset]->frame;
            int32_t end_frame = hyp.nodes[offset + static_cast<int32_t>(label.size()) - 1]->frame;
            for (size_t i = static_cast<size_t>(offset); i < static_cast<size_t>(offset) + label.size(); ++i) {
                score *= static_cast<double>(hyp.nodes[i]->prob);
            }
            score = std::sqrt(score);
            int32_t duration = end_frame - start_frame;

            if (!has_best_decode_ || score > best_score_) {
                best_word_index_ = word_idx;
                best_score_ = score;
                best_start_frame_ = start_frame;
                best_end_frame_ = end_frame;
                has_best_decode_ = true;
            }

            bool passed_threshold = disable_threshold;
            if (!passed_threshold) {
                auto threshold_it = threshold_map_.find(word_idx);
                if (threshold_it != threshold_map_.end() &&
                    score >= static_cast<double>(threshold_it->second)) {
                    passed_threshold = true;
                }
            }

            bool enough_interval =
                (last_active_pos_ == -1) || (end_frame - last_active_pos_ >= interval_frames_);
            if (passed_threshold &&
                min_frames_ <= duration && duration <= max_frames_ &&
                enough_interval) {
                last_active_pos_ = end_frame;
                return DetectionResult{
                    word_idx,
                    keyword_or_empty(keyword_strings_, word_idx),
                    score,
                    start_frame,
                    end_frame,
                    true,
                };
            }

            return DetectionResult{-1, std::string(), 0.0, 0, 0, false};
        }
    }

    return DetectionResult{-1, std::string(), 0.0, 0, 0, false};
}

StreamingCTCDecoder::DetectionResult StreamingCTCDecoder::step_and_detect(
    int32_t frame_index, torch::Tensor probs, bool disable_threshold) {
    advance_frame(frame_index, probs);
    return execute_detection(disable_threshold);
}

StreamingCTCDecoder::BestDecodeResult StreamingCTCDecoder::get_best_decode() const {
    if (!has_best_decode_) {
        return BestDecodeResult{-1, std::string(), 0.0, 0, 0, false};
    }
    return BestDecodeResult{
        best_word_index_,
        keyword_or_empty(keyword_strings_, best_word_index_),
        best_score_,
        best_start_frame_,
        best_end_frame_,
        true,
    };
}

int32_t StreamingCTCDecoder::get_first_hyp_start_frame() const {
    if (cur_hyps_.empty() || cur_hyps_[0].nodes.empty()) {
        return -1;
    }
    return cur_hyps_[0].nodes[0]->frame;
}

int32_t StreamingCTCDecoder::num_hypotheses() const {
    return static_cast<int32_t>(cur_hyps_.size());
}

std::vector<Hypothesis> StreamingCTCDecoder::get_hypotheses() const {
    std::vector<Hypothesis> public_hyps;
    public_hyps.reserve(cur_hyps_.size());
    for (const auto& hyp : cur_hyps_) {
        Hypothesis public_hyp;
        public_hyp.prefix = hyp.prefix;
        public_hyp.pb = hyp.pb;
        public_hyp.pnb = hyp.pnb;
        public_hyp.nodes.reserve(hyp.nodes.size());
        for (const auto& node_ptr : hyp.nodes) {
            public_hyp.nodes.push_back(*node_ptr);
        }
        public_hyps.push_back(std::move(public_hyp));
    }
    return public_hyps;
}

}  // namespace ctc_decoder
