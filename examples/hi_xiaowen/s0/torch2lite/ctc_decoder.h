#ifndef CTC_DECODER_H_
#define CTC_DECODER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <torch/extension.h>

namespace ctc_decoder {

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

class StreamingCTCDecoder {
public:
    StreamingCTCDecoder(
        int32_t score_beam_size = 3,
        int32_t path_beam_size = 20,
        int32_t min_frames = 5,
        int32_t max_frames = 250,
        int32_t interval_frames = 50);

    void set_keywords(
        const std::vector<std::pair<int32_t, std::vector<int32_t>>>& keywords_tokens,
        const std::unordered_set<int32_t>& keywords_idxset);
    void set_keyword_strings(const std::vector<std::string>& keyword_strings);
    void set_thresholds(const std::unordered_map<int32_t, float>& threshold_map);
    void advance_frame(int32_t frame_index, torch::Tensor probs);

    struct DetectionResult {
        int32_t word_index;
        std::string keyword;
        double score;
        int32_t start_frame;
        int32_t end_frame;
        bool valid;
    };
    DetectionResult execute_detection(bool disable_threshold);
    DetectionResult step_and_detect(int32_t frame_index, torch::Tensor probs, bool disable_threshold);

    void reset();
    void reset_beam_search();

    struct BestDecodeResult {
        int32_t word_index;
        std::string keyword;
        double score;
        int32_t start_frame;
        int32_t end_frame;
        bool valid;
    };
    BestDecodeResult get_best_decode() const;

    int32_t get_first_hyp_start_frame() const;
    int32_t num_hypotheses() const;
    std::vector<Hypothesis> get_hypotheses() const;

private:
    struct InternalHypothesis {
        std::vector<int32_t> prefix;
        double pb;
        double pnb;
        std::vector<std::shared_ptr<TokenNode>> nodes;
    };

    void init_state();
    void init_beam_state();
    void advance_beam(
        int32_t frame_index,
        const std::vector<std::pair<float, int32_t>>& filtered_tokens);
    static bool float_close_zero(double val, double abs_tol = 1e-6);
    static bool is_sublist(
        const std::vector<int32_t>& main_list,
        const std::vector<int32_t>& check_list,
        int32_t& offset_out);

    int32_t score_beam_size_;
    int32_t path_beam_size_;
    int32_t min_frames_;
    int32_t max_frames_;
    int32_t interval_frames_;

    std::vector<std::pair<int32_t, std::vector<int32_t>>> keywords_tokens_;
    std::unordered_set<int32_t> keywords_idxset_;
    std::unordered_map<int32_t, float> threshold_map_;
    std::vector<std::string> keyword_strings_;

    std::vector<InternalHypothesis> cur_hyps_;
    int32_t last_active_pos_;

    bool has_best_decode_;
    int32_t best_word_index_;
    double best_score_;
    int32_t best_start_frame_;
    int32_t best_end_frame_;
};

}  // namespace ctc_decoder

#endif  // CTC_DECODER_H_
