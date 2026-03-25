#ifndef CTC_DECODER_C_H_
#define CTC_DECODER_C_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t token;
    int32_t frame;
    float prob;
} CTCDecoderCTokenNode;

typedef struct {
    int32_t* prefix;
    int32_t prefix_len;
    double pb;
    double pnb;
    CTCDecoderCTokenNode* nodes;
    int32_t* node_refs;
    int32_t node_count;
} CTCDecoderCHypothesis;

typedef struct {
    int32_t word_index;
    const int32_t* token_ids;
    int32_t token_count;
} CTCDecoderCKeyword;

typedef struct {
    int32_t score_beam_size;
    int32_t path_beam_size;
    int32_t min_frames;
    int32_t max_frames;
    int32_t interval_frames;
    int32_t max_prefix_len;
    int32_t enable_debug_hypotheses;
} CTCDecoderCConfig;

typedef struct {
    int32_t word_index;
    const char* keyword;
    double score;
    int32_t start_frame;
    int32_t end_frame;
    int32_t valid;
} CTCDecoderCDetectionResult;

typedef CTCDecoderCDetectionResult CTCDecoderCBestDecodeResult;

typedef void* (*CTCDecoderCMallocFn)(void* user_data, size_t size);
typedef void* (*CTCDecoderCCallocFn)(void* user_data, size_t count, size_t size);
typedef void (*CTCDecoderCFreeFn)(void* user_data, void* ptr);

typedef struct {
    CTCDecoderCMallocFn malloc_fn;
    CTCDecoderCCallocFn calloc_fn;
    CTCDecoderCFreeFn free_fn;
    void* user_data;
} CTCDecoderCAllocator;

typedef struct {
    CTCDecoderCConfig config;
    CTCDecoderCAllocator allocator;

    CTCDecoderCHypothesis* cur_hyps;
    CTCDecoderCHypothesis* next_hyps;
    int32_t cur_hyp_count;
    int32_t next_hyp_count;
    int32_t cur_hyp_capacity;
    int32_t next_hyp_capacity;

    int32_t* cur_prefix_storage;
    int32_t* next_prefix_storage;
    CTCDecoderCTokenNode* cur_node_storage;
    int32_t* cur_node_ref_storage;
    int32_t* next_node_ref_storage;

    float* topk_probs;
    int32_t* topk_indices;
    int32_t topk_capacity;
    int32_t* temp_prefix;

    CTCDecoderCTokenNode* node_pool;
    int32_t* node_ref_remap;
    int32_t node_pool_capacity;
    int32_t node_pool_size;

    CTCDecoderCKeyword* keywords;
    int32_t num_keywords;
    int32_t* keyword_token_storage;
    int32_t keyword_token_storage_size;
    int32_t* keywords_idxset;
    int32_t num_keywords_idxset;
    const char** keyword_strings;
    char* keyword_string_storage;
    int32_t keyword_string_stride;

    float* thresholds;
    uint8_t* threshold_valid;

    int32_t last_active_pos;
    int32_t has_best_decode;
    int32_t best_word_index;
    double best_score;
    int32_t best_start_frame;
    int32_t best_end_frame;
} CTCDecoderCState;

void ctc_decoder_c_init_default_config(CTCDecoderCConfig* config);
void ctc_decoder_c_init_default_allocator(CTCDecoderCAllocator* allocator);
int32_t ctc_decoder_c_init(CTCDecoderCState* state, const CTCDecoderCConfig* config);
int32_t ctc_decoder_c_init_with_allocator(
    CTCDecoderCState* state,
    const CTCDecoderCConfig* config,
    const CTCDecoderCAllocator* allocator);
void ctc_decoder_c_free(CTCDecoderCState* state);

int32_t ctc_decoder_c_set_keywords(
    CTCDecoderCState* state,
    const CTCDecoderCKeyword* keywords,
    int32_t num_keywords,
    const int32_t* keywords_idxset,
    int32_t num_keywords_idxset,
    const char* const* keyword_strings);

int32_t ctc_decoder_c_set_thresholds(
    CTCDecoderCState* state,
    const int32_t* word_indices,
    const float* threshold_values,
    int32_t num_thresholds);

int32_t ctc_decoder_c_advance_frame(
    CTCDecoderCState* state,
    int32_t frame_index,
    const float* probs,
    int32_t vocab_size);

CTCDecoderCDetectionResult ctc_decoder_c_execute_detection(
    CTCDecoderCState* state,
    int32_t disable_threshold);

CTCDecoderCDetectionResult ctc_decoder_c_step_and_detect(
    CTCDecoderCState* state,
    int32_t frame_index,
    const float* probs,
    int32_t vocab_size,
    int32_t disable_threshold);

void ctc_decoder_c_reset(CTCDecoderCState* state);
void ctc_decoder_c_reset_beam_search(CTCDecoderCState* state);
CTCDecoderCBestDecodeResult ctc_decoder_c_get_best_decode(const CTCDecoderCState* state);
int32_t ctc_decoder_c_get_first_hyp_start_frame(const CTCDecoderCState* state);
int32_t ctc_decoder_c_num_hypotheses(const CTCDecoderCState* state);
const CTCDecoderCHypothesis* ctc_decoder_c_get_hypothesis(
    const CTCDecoderCState* state,
    int32_t index);

#ifdef __cplusplus
}
#endif

#endif  // CTC_DECODER_C_H_
