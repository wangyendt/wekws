#include "ctc_decoder_c.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifndef CTC_DECODER_C_DEFAULT_MAX_PREFIX_LEN
#define CTC_DECODER_C_DEFAULT_MAX_PREFIX_LEN 64
#endif

#ifndef CTC_DECODER_C_HEAP_ALIGNMENT
#define CTC_DECODER_C_HEAP_ALIGNMENT 16u
#endif

typedef struct {
    void* base_ptr;
} CTCDecoderCAlignedAllocHeader;

static void* ctc_decoder_c_default_malloc(void* user_data, size_t size) {
    (void)user_data;
    return malloc(size);
}

static void* ctc_decoder_c_default_calloc(void* user_data, size_t count, size_t size) {
    (void)user_data;
    return calloc(count, size);
}

static void ctc_decoder_c_default_free(void* user_data, void* ptr) {
    (void)user_data;
    free(ptr);
}

void ctc_decoder_c_init_default_allocator(CTCDecoderCAllocator* allocator) {
    if (!allocator) {
        return;
    }
    allocator->malloc_fn = ctc_decoder_c_default_malloc;
    allocator->calloc_fn = ctc_decoder_c_default_calloc;
    allocator->free_fn = ctc_decoder_c_default_free;
    allocator->user_data = NULL;
}

static void ctc_decoder_c_resolve_allocator(
    const CTCDecoderCAllocator* allocator,
    CTCDecoderCAllocator* resolved_allocator) {
    if (allocator && allocator->malloc_fn && allocator->calloc_fn && allocator->free_fn) {
        resolved_allocator->malloc_fn = allocator->malloc_fn;
        resolved_allocator->calloc_fn = allocator->calloc_fn;
        resolved_allocator->free_fn = allocator->free_fn;
        resolved_allocator->user_data = allocator->user_data;
        return;
    }
    ctc_decoder_c_init_default_allocator(resolved_allocator);
}

static uintptr_t ctc_decoder_c_align_up_uintptr(uintptr_t value, size_t alignment) {
    return (value + (uintptr_t)(alignment - 1u)) & ~(uintptr_t)(alignment - 1u);
}

static int32_t ctc_decoder_c_is_aligned_16(const void* ptr) {
    return (((uintptr_t)ptr) & (uintptr_t)(CTC_DECODER_C_HEAP_ALIGNMENT - 1u)) == 0u;
}

static void* ctc_decoder_c_alloc_zeroed(
    const CTCDecoderCAllocator* allocator,
    size_t count,
    size_t elem_size) {
    size_t payload_size;
    size_t total_size;
    void* raw_ptr;
    uintptr_t aligned_addr;
    CTCDecoderCAlignedAllocHeader* header;
    if (!allocator || !allocator->malloc_fn) {
        return NULL;
    }
    if (elem_size != 0 && count > SIZE_MAX / elem_size) {
        return NULL;
    }
    payload_size = count * elem_size;
    if (payload_size == 0) {
        payload_size = 1;
    }
    if (payload_size > SIZE_MAX - (CTC_DECODER_C_HEAP_ALIGNMENT - 1u) - sizeof(CTCDecoderCAlignedAllocHeader)) {
        return NULL;
    }
    total_size = payload_size + (CTC_DECODER_C_HEAP_ALIGNMENT - 1u) + sizeof(CTCDecoderCAlignedAllocHeader);
    raw_ptr = allocator->malloc_fn(allocator->user_data, total_size);
    if (!raw_ptr) {
        return NULL;
    }
    aligned_addr = ctc_decoder_c_align_up_uintptr(
        (uintptr_t)raw_ptr + sizeof(CTCDecoderCAlignedAllocHeader),
        CTC_DECODER_C_HEAP_ALIGNMENT);
    header = (CTCDecoderCAlignedAllocHeader*)(aligned_addr - sizeof(CTCDecoderCAlignedAllocHeader));
    header->base_ptr = raw_ptr;
    memset((void*)aligned_addr, 0, payload_size);
    return (void*)aligned_addr;
}

static void ctc_decoder_c_release_bytes(const CTCDecoderCAllocator* allocator, void* ptr) {
    if (ptr) {
        CTCDecoderCAlignedAllocHeader* header =
            (CTCDecoderCAlignedAllocHeader*)((uintptr_t)ptr - sizeof(CTCDecoderCAlignedAllocHeader));
        allocator->free_fn(allocator->user_data, header->base_ptr);
    }
}

static void ctc_decoder_c_zero_state(CTCDecoderCState* state) {
    memset(state, 0, sizeof(*state));
    state->last_active_pos = -1;
#if CTC_DECODER_C_ENABLE_EXTENDED_API
    state->best_word_index = -1;
#endif
}

void ctc_decoder_c_init_default_config(CTCDecoderCConfig* config) {
    if (!config) {
        return;
    }
    config->score_beam_size = 3;
    config->path_beam_size = 20;
    config->min_frames = 5;
    config->max_frames = 250;
    config->interval_frames = 50;
    config->frame_step = 1;
    config->max_prefix_len = 0;
}

static int32_t ctc_decoder_c_resolve_max_prefix_len(const CTCDecoderCConfig* config) {
    int32_t max_prefix_len = config->max_prefix_len;
    if (max_prefix_len <= 0) {
        max_prefix_len = CTC_DECODER_C_DEFAULT_MAX_PREFIX_LEN;
    }
    if (max_prefix_len < CTC_DECODER_C_DEFAULT_MAX_PREFIX_LEN) {
        max_prefix_len = CTC_DECODER_C_DEFAULT_MAX_PREFIX_LEN;
    }
    return max_prefix_len;
}

static void ctc_decoder_c_reset_hypothesis(CTCDecoderCHypothesis* hyp) {
    hyp->prefix_len = 0;
    hyp->pb = 0.0;
    hyp->pnb = 0.0;
    hyp->node_count = 0;
}

static void ctc_decoder_c_bind_hypothesis_storage(
    CTCDecoderCHypothesis* hyps,
    int32_t hyp_count,
    CTCDecoderCPrefixToken* prefix_storage,
    CTCDecoderCTokenNode* node_storage,
    CTCDecoderCNodeRef* node_ref_storage,
    int32_t max_prefix_len) {
    int32_t index;
#if !CTC_DECODER_C_ENABLE_DEBUG_HYPOTHESES
    (void)node_storage;
#endif
    for (index = 0; index < hyp_count; ++index) {
        hyps[index].prefix = prefix_storage + ((ptrdiff_t)index * max_prefix_len);
#if CTC_DECODER_C_ENABLE_DEBUG_HYPOTHESES
        hyps[index].nodes = node_storage ? (node_storage + ((ptrdiff_t)index * max_prefix_len)) : NULL;
#endif
        hyps[index].node_refs = node_ref_storage + ((ptrdiff_t)index * max_prefix_len);
        ctc_decoder_c_reset_hypothesis(&hyps[index]);
    }
}

#if CTC_DECODER_C_ENABLE_DEBUG_HYPOTHESES
static void ctc_decoder_c_copy_node_value(
    CTCDecoderCTokenNode* dst,
    const CTCDecoderCTokenNode* src);

static int32_t ctc_decoder_c_init_debug_hypotheses(CTCDecoderCState* state) {
    int32_t index;
    ptrdiff_t node_slots;
    if (state->cur_node_storage) {
        return 0;
    }
    node_slots = (ptrdiff_t)state->cur_hyp_capacity * state->config.max_prefix_len;
    state->cur_node_storage = (CTCDecoderCTokenNode*)ctc_decoder_c_alloc_zeroed(
        &state->allocator,
        (size_t)node_slots,
        sizeof(CTCDecoderCTokenNode));
    if (!state->cur_node_storage) {
        return -1;
    }
    for (index = 0; index < state->cur_hyp_capacity; ++index) {
        state->cur_hyps[index].nodes =
            state->cur_node_storage + ((ptrdiff_t)index * state->config.max_prefix_len);
    }
    return 0;
}

static int32_t ctc_decoder_c_materialize_hypothesis(
    CTCDecoderCState* state,
    CTCDecoderCHypothesis* hyp) {
    int32_t index;
    if (!state->cur_node_storage) {
        return -1;
    }
    for (index = 0; index < hyp->node_count; ++index) {
        int32_t ref = (int32_t)hyp->node_refs[index];
        if (ref >= 0 && ref < state->node_pool_size) {
            ctc_decoder_c_copy_node_value(&hyp->nodes[index], &state->node_pool[ref]);
        }
    }
    return 0;
}
#endif

static void ctc_decoder_c_init_beam_state(CTCDecoderCState* state) {
    int32_t index;
    for (index = 0; index < state->cur_hyp_capacity; ++index) {
        ctc_decoder_c_reset_hypothesis(&state->cur_hyps[index]);
    }
    state->cur_hyp_count = 1;
    state->next_hyp_count = 0;
    state->node_pool_size = 0;
    state->cur_hyps[0].pb = 1.0;
    state->cur_hyps[0].pnb = 0.0;
}

static int32_t ctc_decoder_c_prefix_equal(
    const CTCDecoderCPrefixToken* lhs,
    int32_t lhs_len,
    const CTCDecoderCPrefixToken* rhs,
    int32_t rhs_len) {
    if (lhs_len != rhs_len) {
        return 0;
    }
    if (lhs_len == 0) {
        return 1;
    }
    return memcmp(lhs, rhs, (size_t)lhs_len * sizeof(CTCDecoderCPrefixToken)) == 0;
}

static int32_t ctc_decoder_c_copy_node_refs(
    CTCDecoderCHypothesis* dst,
    const CTCDecoderCHypothesis* src,
    int32_t max_prefix_len) {
    if (src->node_count > max_prefix_len) {
        return -1;
    }
    dst->node_count = src->node_count;
    if (src->node_count > 0) {
        memcpy(dst->node_refs, src->node_refs, (size_t)src->node_count * sizeof(CTCDecoderCNodeRef));
    }
    return 0;
}

static int32_t ctc_decoder_c_copy_hypothesis(
    CTCDecoderCHypothesis* dst,
    const CTCDecoderCHypothesis* src,
    int32_t max_prefix_len) {
    if (src->prefix_len > max_prefix_len || src->node_count > max_prefix_len) {
        return -1;
    }
    dst->prefix_len = src->prefix_len;
    if (src->prefix_len > 0) {
        memcpy(dst->prefix, src->prefix, (size_t)src->prefix_len * sizeof(CTCDecoderCPrefixToken));
    }
    dst->pb = src->pb;
    dst->pnb = src->pnb;
    return ctc_decoder_c_copy_node_refs(dst, src, max_prefix_len);
}

static CTCDecoderCHypothesis* ctc_decoder_c_find_or_add_next_hyp(
    CTCDecoderCState* state,
    const CTCDecoderCPrefixToken* prefix,
    int32_t prefix_len) {
    int32_t index;
    CTCDecoderCHypothesis* hyp;
    for (index = 0; index < state->next_hyp_count; ++index) {
        hyp = &state->next_hyps[index];
        if (ctc_decoder_c_prefix_equal(hyp->prefix, hyp->prefix_len, prefix, prefix_len)) {
            return hyp;
        }
    }
    if (state->next_hyp_count >= state->next_hyp_capacity) {
        return NULL;
    }
    hyp = &state->next_hyps[state->next_hyp_count++];
    ctc_decoder_c_reset_hypothesis(hyp);
    hyp->prefix_len = prefix_len;
    if (prefix_len > 0) {
        memcpy(hyp->prefix, prefix, (size_t)prefix_len * sizeof(CTCDecoderCPrefixToken));
    }
    return hyp;
}

static int32_t ctc_decoder_c_keyword_allowed(const CTCDecoderCState* state, int32_t token_id) {
    int32_t index;
    if (state->num_keywords_idxset <= 0) {
        return 1;
    }
    for (index = 0; index < state->num_keywords_idxset; ++index) {
        if (state->keywords_idxset[index] == token_id) {
            return 1;
        }
    }
    return 0;
}

static int32_t ctc_decoder_c_float_close_zero(double value) {
    return fabs(value) < 1.0e-6;
}

static CTCDecoderCNodeFrame ctc_decoder_c_wrap_frame(int32_t frame) {
    return (CTCDecoderCNodeFrame)((uint32_t)frame & (uint32_t)UINT16_MAX);
}

static int32_t ctc_decoder_c_frame_diff(CTCDecoderCNodeFrame newer, CTCDecoderCNodeFrame older) {
    return (int32_t)(CTCDecoderCNodeFrame)(newer - older);
}

#if CTC_DECODER_C_ENABLE_DEBUG_HYPOTHESES
static void ctc_decoder_c_copy_node_value(
    CTCDecoderCTokenNode* dst,
    const CTCDecoderCTokenNode* src) {
    dst->token = src->token;
    dst->frame = src->frame;
    dst->prob = src->prob;
}

#endif

static void ctc_decoder_c_swap_node_value(
    CTCDecoderCTokenNode* lhs,
    CTCDecoderCTokenNode* rhs) {
    CTCDecoderCTokenId token = lhs->token;
    CTCDecoderCNodeFrame frame = lhs->frame;
    float prob = lhs->prob;
    lhs->token = rhs->token;
    lhs->frame = rhs->frame;
    lhs->prob = rhs->prob;
    rhs->token = token;
    rhs->frame = frame;
    rhs->prob = prob;
}

static void ctc_decoder_c_copy_config(
    CTCDecoderCConfig* dst,
    const CTCDecoderCConfig* src) {
    dst->score_beam_size = src->score_beam_size;
    dst->path_beam_size = src->path_beam_size;
    dst->min_frames = src->min_frames;
    dst->max_frames = src->max_frames;
    dst->interval_frames = src->interval_frames;
    dst->frame_step = src->frame_step;
    dst->max_prefix_len = src->max_prefix_len;
}

static int32_t ctc_decoder_c_select_topk(
    CTCDecoderCState* state,
    const float* probs,
    int32_t vocab_size) {
    int32_t topk_count;
    int32_t index;
    int32_t slot;
    int32_t insert_pos;
    if (state->config.score_beam_size < vocab_size) {
        topk_count = state->config.score_beam_size;
    } else {
        topk_count = vocab_size;
    }
    if (topk_count <= 0) {
        return 0;
    }
    for (slot = 0; slot < topk_count; ++slot) {
        state->topk_probs[slot] = -1.0f;
        state->topk_indices[slot] = -1;
    }
    for (index = 0; index < vocab_size; ++index) {
        float prob = probs[index];
        if (state->topk_indices[topk_count - 1] != -1 && prob <= state->topk_probs[topk_count - 1]) {
            continue;
        }
        insert_pos = topk_count - 1;
        while (insert_pos > 0) {
            if (state->topk_indices[insert_pos - 1] == -1 || prob > state->topk_probs[insert_pos - 1]) {
                state->topk_probs[insert_pos] = state->topk_probs[insert_pos - 1];
                state->topk_indices[insert_pos] = state->topk_indices[insert_pos - 1];
                --insert_pos;
                continue;
            }
            break;
        }
        state->topk_probs[insert_pos] = prob;
        state->topk_indices[insert_pos] = index;
    }
    return topk_count;
}

static int32_t ctc_decoder_c_is_sublist(
    const CTCDecoderCPrefixToken* main_list,
    int32_t main_len,
    const int32_t* check_list,
    int32_t check_len,
    int32_t* offset_out) {
    int32_t index;
    int32_t sub_index;
    if (main_len < check_len) {
        return 0;
    }
    if (main_len == check_len) {
        if (check_len == 0) {
            *offset_out = 0;
            return 1;
        }
        for (sub_index = 0; sub_index < check_len; ++sub_index) {
            if ((int32_t)main_list[sub_index] != check_list[sub_index]) {
                return 0;
            }
        }
        {
            *offset_out = 0;
            return 1;
        }
        return 0;
    }
    for (index = 0; index <= main_len - check_len; ++index) {
        int32_t matched = 1;
        for (sub_index = 0; sub_index < check_len; ++sub_index) {
            if ((int32_t)main_list[index + sub_index] != check_list[sub_index]) {
                matched = 0;
                break;
            }
        }
        if (matched) {
            *offset_out = index;
            return 1;
        }
    }
    return 0;
}

static void ctc_decoder_c_swap_hypotheses(CTCDecoderCHypothesis* lhs, CTCDecoderCHypothesis* rhs) {
    CTCDecoderCPrefixToken* prefix = lhs->prefix;
    int32_t prefix_len = lhs->prefix_len;
    double pb = lhs->pb;
    double pnb = lhs->pnb;
#if CTC_DECODER_C_ENABLE_DEBUG_HYPOTHESES
    CTCDecoderCTokenNode* nodes = lhs->nodes;
#endif
    CTCDecoderCNodeRef* node_refs = lhs->node_refs;
    int32_t node_count = lhs->node_count;

    lhs->prefix = rhs->prefix;
    lhs->prefix_len = rhs->prefix_len;
    lhs->pb = rhs->pb;
    lhs->pnb = rhs->pnb;
#if CTC_DECODER_C_ENABLE_DEBUG_HYPOTHESES
    lhs->nodes = rhs->nodes;
#endif
    lhs->node_refs = rhs->node_refs;
    lhs->node_count = rhs->node_count;

    rhs->prefix = prefix;
    rhs->prefix_len = prefix_len;
    rhs->pb = pb;
    rhs->pnb = pnb;
#if CTC_DECODER_C_ENABLE_DEBUG_HYPOTHESES
    rhs->nodes = nodes;
#endif
    rhs->node_refs = node_refs;
    rhs->node_count = node_count;
}

static void ctc_decoder_c_sort_next_hyps(CTCDecoderCState* state) {
    int32_t index;
    int32_t cursor;
    for (index = 1; index < state->next_hyp_count; ++index) {
        cursor = index;
        while (cursor > 0) {
            double prev_total = state->next_hyps[cursor - 1].pb + state->next_hyps[cursor - 1].pnb;
            double cur_total = state->next_hyps[cursor].pb + state->next_hyps[cursor].pnb;
            if (cur_total > prev_total) {
                ctc_decoder_c_swap_hypotheses(&state->next_hyps[cursor - 1], &state->next_hyps[cursor]);
                --cursor;
                continue;
            }
            break;
        }
    }
}

static int32_t ctc_decoder_c_alloc_node(
    CTCDecoderCState* state,
    int32_t token,
    int32_t frame,
    float prob) {
    int32_t ref;
    if (state->node_pool_size >= state->node_pool_capacity) {
        return -1;
    }
    if (token < 0 || token > (int32_t)UINT16_MAX) {
        return -1;
    }
    ref = state->node_pool_size++;
    state->node_pool[ref].token = (CTCDecoderCTokenId)token;
    state->node_pool[ref].frame = ctc_decoder_c_wrap_frame(frame);
    state->node_pool[ref].prob = prob;
    return ref;
}

static int32_t ctc_decoder_c_compact_node_pool(CTCDecoderCState* state) {
    int32_t index;
    int32_t hyp_index;
    int32_t node_index;
    int32_t target_index;
    int32_t new_size = 0;
    for (index = 0; index < state->node_pool_size; ++index) {
        state->node_ref_remap[index] = CTC_DECODER_C_INVALID_NODE_REF;
    }
    for (hyp_index = 0; hyp_index < state->cur_hyp_count; ++hyp_index) {
        CTCDecoderCHypothesis* hyp = &state->cur_hyps[hyp_index];
        for (node_index = 0; node_index < hyp->node_count; ++node_index) {
            int32_t old_ref = (int32_t)hyp->node_refs[node_index];
            int32_t new_ref;
            if (old_ref < 0 || old_ref >= state->node_pool_size) {
                return -1;
            }
            new_ref = state->node_ref_remap[old_ref];
            if ((CTCDecoderCNodeRef)new_ref == CTC_DECODER_C_INVALID_NODE_REF) {
                if (new_size >= state->node_pool_capacity) {
                    return -1;
                }
                state->node_ref_remap[old_ref] = new_size;
                new_ref = new_size;
                ++new_size;
            }
            hyp->node_refs[node_index] = (CTCDecoderCNodeRef)new_ref;
        }
    }
    for (target_index = 0; target_index < new_size; ++target_index) {
        if (state->node_ref_remap[target_index] == (CTCDecoderCNodeRef)target_index) {
            continue;
        }
        for (index = target_index + 1; index < state->node_pool_size; ++index) {
            if (state->node_ref_remap[index] == (CTCDecoderCNodeRef)target_index) {
                CTCDecoderCNodeRef displaced_target = state->node_ref_remap[target_index];
                ctc_decoder_c_swap_node_value(&state->node_pool[target_index], &state->node_pool[index]);
                state->node_ref_remap[target_index] = (CTCDecoderCNodeRef)target_index;
                state->node_ref_remap[index] = displaced_target;
                break;
            }
        }
        if (index >= state->node_pool_size) {
            return -1;
        }
    }
    state->node_pool_size = new_size;
    return 0;
}

static int32_t ctc_decoder_c_finalize_frame(CTCDecoderCState* state) {
    int32_t keep_count;
    int32_t index;
    ctc_decoder_c_sort_next_hyps(state);
    keep_count = state->next_hyp_count;
    if (keep_count > state->config.path_beam_size) {
        keep_count = state->config.path_beam_size;
    }
    for (index = 0; index < keep_count; ++index) {
        if (ctc_decoder_c_copy_hypothesis(&state->cur_hyps[index], &state->next_hyps[index], state->config.max_prefix_len) != 0) {
            return -1;
        }
    }
    state->cur_hyp_count = keep_count;
    return ctc_decoder_c_compact_node_pool(state);
}

static void ctc_decoder_c_cleanup_keyword_storage(CTCDecoderCState* state) {
    ctc_decoder_c_release_bytes(&state->allocator, state->keywords);
    ctc_decoder_c_release_bytes(&state->allocator, state->keyword_token_storage);
    ctc_decoder_c_release_bytes(&state->allocator, state->keywords_idxset);
    ctc_decoder_c_release_bytes(&state->allocator, (void*)state->keyword_strings);
    ctc_decoder_c_release_bytes(&state->allocator, state->keyword_string_storage);
    ctc_decoder_c_release_bytes(&state->allocator, state->thresholds);
    ctc_decoder_c_release_bytes(&state->allocator, state->threshold_valid);
    state->keywords = NULL;
    state->keyword_token_storage = NULL;
    state->keywords_idxset = NULL;
    state->keyword_strings = NULL;
    state->keyword_string_storage = NULL;
    state->thresholds = NULL;
    state->threshold_valid = NULL;
    state->num_keywords = 0;
    state->num_keywords_idxset = 0;
    state->keyword_token_storage_size = 0;
    state->keyword_string_stride = 0;
}

int32_t ctc_decoder_c_init(CTCDecoderCState* state, const CTCDecoderCConfig* config) {
    return ctc_decoder_c_init_with_allocator(state, config, NULL);
}

int32_t ctc_decoder_c_init_with_allocator(
    CTCDecoderCState* state,
    const CTCDecoderCConfig* config,
    const CTCDecoderCAllocator* allocator) {
    int32_t max_prefix_len;
    ptrdiff_t prefix_slots_cur;
    ptrdiff_t prefix_slots_next;
    ptrdiff_t node_pool_capacity;
    CTCDecoderCAllocator resolved_allocator;
    if (!state || !config) {
        return -1;
    }
    if (!ctc_decoder_c_is_aligned_16(state)) {
        return -1;
    }
    if (config->score_beam_size <= 0 || config->path_beam_size <= 0 || config->frame_step <= 0) {
        return -1;
    }
    if (config->max_prefix_len > CTC_DECODER_C_INVALID_NODE_REF || config->max_frames >= (int32_t)UINT16_MAX || config->interval_frames >= (int32_t)UINT16_MAX) {
        return -1;
    }

    ctc_decoder_c_zero_state(state);
    ctc_decoder_c_resolve_allocator(allocator, &resolved_allocator);
    state->allocator = resolved_allocator;
    ctc_decoder_c_copy_config(&state->config, config);
    state->config.max_prefix_len = ctc_decoder_c_resolve_max_prefix_len(config);
    max_prefix_len = state->config.max_prefix_len;
    state->cur_hyp_capacity = state->config.path_beam_size;
    state->next_hyp_capacity = state->config.path_beam_size * (state->config.score_beam_size + 1);
    node_pool_capacity = (ptrdiff_t)state->next_hyp_capacity * max_prefix_len;
    if (state->next_hyp_capacity < state->config.path_beam_size || node_pool_capacity <= 0) {
        return -1;
    }
    if (node_pool_capacity >= (ptrdiff_t)CTC_DECODER_C_INVALID_NODE_REF) {
        return -1;
    }
    state->node_pool_capacity = (int32_t)node_pool_capacity;

    prefix_slots_cur = (ptrdiff_t)state->cur_hyp_capacity * max_prefix_len;
    prefix_slots_next = (ptrdiff_t)state->next_hyp_capacity * max_prefix_len;

    state->cur_hyps = (CTCDecoderCHypothesis*)ctc_decoder_c_alloc_zeroed(&state->allocator, (size_t)state->cur_hyp_capacity, sizeof(CTCDecoderCHypothesis));
    state->next_hyps = (CTCDecoderCHypothesis*)ctc_decoder_c_alloc_zeroed(&state->allocator, (size_t)state->next_hyp_capacity, sizeof(CTCDecoderCHypothesis));
    state->cur_prefix_storage = (CTCDecoderCPrefixToken*)ctc_decoder_c_alloc_zeroed(&state->allocator, (size_t)prefix_slots_cur, sizeof(CTCDecoderCPrefixToken));
    state->next_prefix_storage = (CTCDecoderCPrefixToken*)ctc_decoder_c_alloc_zeroed(&state->allocator, (size_t)prefix_slots_next, sizeof(CTCDecoderCPrefixToken));
    state->cur_node_ref_storage = (CTCDecoderCNodeRef*)ctc_decoder_c_alloc_zeroed(&state->allocator, (size_t)prefix_slots_cur, sizeof(CTCDecoderCNodeRef));
    state->next_node_ref_storage = (CTCDecoderCNodeRef*)ctc_decoder_c_alloc_zeroed(&state->allocator, (size_t)prefix_slots_next, sizeof(CTCDecoderCNodeRef));
    state->topk_capacity = state->config.score_beam_size;
    state->topk_probs = (float*)ctc_decoder_c_alloc_zeroed(&state->allocator, (size_t)state->topk_capacity, sizeof(float));
    state->topk_indices = (int32_t*)ctc_decoder_c_alloc_zeroed(&state->allocator, (size_t)state->topk_capacity, sizeof(int32_t));
    state->temp_prefix = (CTCDecoderCPrefixToken*)ctc_decoder_c_alloc_zeroed(&state->allocator, (size_t)max_prefix_len, sizeof(CTCDecoderCPrefixToken));
    state->node_pool = (CTCDecoderCTokenNode*)ctc_decoder_c_alloc_zeroed(&state->allocator, (size_t)state->node_pool_capacity, sizeof(CTCDecoderCTokenNode));
    state->node_ref_remap = state->next_node_ref_storage;

    if (!state->cur_hyps || !state->next_hyps || !state->cur_prefix_storage || !state->next_prefix_storage ||
        !state->cur_node_ref_storage || !state->next_node_ref_storage || !state->topk_probs ||
        !state->topk_indices || !state->temp_prefix || !state->node_pool) {
        ctc_decoder_c_free(state);
        return -1;
    }

    ctc_decoder_c_bind_hypothesis_storage(state->cur_hyps, state->cur_hyp_capacity, state->cur_prefix_storage, NULL, state->cur_node_ref_storage, max_prefix_len);
    ctc_decoder_c_bind_hypothesis_storage(state->next_hyps, state->next_hyp_capacity, state->next_prefix_storage, NULL, state->next_node_ref_storage, max_prefix_len);
#if CTC_DECODER_C_ENABLE_DEBUG_HYPOTHESES
    if (ctc_decoder_c_init_debug_hypotheses(state) != 0) {
        ctc_decoder_c_free(state);
        return -1;
    }
#endif
    ctc_decoder_c_init_beam_state(state);
    return 0;
}

void ctc_decoder_c_free(CTCDecoderCState* state) {
    CTCDecoderCAllocator allocator;
    if (!state) {
        return;
    }

    ctc_decoder_c_resolve_allocator(&state->allocator, &allocator);

    ctc_decoder_c_release_bytes(&allocator, state->cur_hyps);
    ctc_decoder_c_release_bytes(&allocator, state->next_hyps);
    ctc_decoder_c_release_bytes(&allocator, state->cur_prefix_storage);
    ctc_decoder_c_release_bytes(&allocator, state->next_prefix_storage);
#if CTC_DECODER_C_ENABLE_DEBUG_HYPOTHESES
    ctc_decoder_c_release_bytes(&allocator, state->cur_node_storage);
#endif
    ctc_decoder_c_release_bytes(&allocator, state->cur_node_ref_storage);
    ctc_decoder_c_release_bytes(&allocator, state->topk_probs);
    ctc_decoder_c_release_bytes(&allocator, state->topk_indices);
    ctc_decoder_c_release_bytes(&allocator, state->temp_prefix);
    ctc_decoder_c_release_bytes(&allocator, state->node_pool);
    ctc_decoder_c_release_bytes(&allocator, state->next_node_ref_storage);
    ctc_decoder_c_release_bytes(&allocator, state->keywords);
    ctc_decoder_c_release_bytes(&allocator, state->keyword_token_storage);
    ctc_decoder_c_release_bytes(&allocator, state->keywords_idxset);
    ctc_decoder_c_release_bytes(&allocator, (void*)state->keyword_strings);
    ctc_decoder_c_release_bytes(&allocator, state->keyword_string_storage);
    ctc_decoder_c_release_bytes(&allocator, state->thresholds);
    ctc_decoder_c_release_bytes(&allocator, state->threshold_valid);
    ctc_decoder_c_zero_state(state);
}

int32_t ctc_decoder_c_set_keywords(
    CTCDecoderCState* state,
    const CTCDecoderCKeyword* keywords,
    int32_t num_keywords,
    const int32_t* keywords_idxset,
    int32_t num_keywords_idxset,
    const char* const* keyword_strings) {
    int32_t total_tokens = 0;
    int32_t index;
    int32_t offset = 0;
    int32_t stride = 1;
    if (!state || num_keywords < 0 || num_keywords_idxset < 0) {
        return -1;
    }

    ctc_decoder_c_cleanup_keyword_storage(state);

    for (index = 0; index < num_keywords; ++index) {
        int32_t token_index;
        if (keywords[index].token_count < 0 || keywords[index].token_count > state->config.max_prefix_len) {
            return -1;
        }
        for (token_index = 0; token_index < keywords[index].token_count; ++token_index) {
            if (keywords[index].token_ids[token_index] < 0 || keywords[index].token_ids[token_index] > (int32_t)UINT16_MAX) {
                return -1;
            }
        }
        total_tokens += keywords[index].token_count;
    }

    if (num_keywords > 0) {
        state->keywords = (CTCDecoderCKeyword*)ctc_decoder_c_alloc_zeroed(&state->allocator, (size_t)num_keywords, sizeof(CTCDecoderCKeyword));
        state->keyword_token_storage = (int32_t*)ctc_decoder_c_alloc_zeroed(&state->allocator, (size_t)total_tokens, sizeof(int32_t));
        state->keyword_strings = (const char**)ctc_decoder_c_alloc_zeroed(&state->allocator, (size_t)num_keywords, sizeof(const char*));
        state->thresholds = (float*)ctc_decoder_c_alloc_zeroed(&state->allocator, (size_t)num_keywords, sizeof(float));
        state->threshold_valid = (uint8_t*)ctc_decoder_c_alloc_zeroed(&state->allocator, (size_t)num_keywords, sizeof(uint8_t));
        if (!state->keywords || !state->keyword_token_storage || !state->keyword_strings || !state->thresholds || !state->threshold_valid) {
            return -1;
        }
        if (keyword_strings) {
            for (index = 0; index < num_keywords; ++index) {
                int32_t len = (int32_t)strlen(keyword_strings[index]) + 1;
                if (len > stride) {
                    stride = len;
                }
            }
            state->keyword_string_stride = stride;
            state->keyword_string_storage = (char*)ctc_decoder_c_alloc_zeroed(&state->allocator, (size_t)num_keywords * stride, sizeof(char));
            if (!state->keyword_string_storage) {
                return -1;
            }
        }
    }

    for (index = 0; index < num_keywords; ++index) {
        state->keywords[index].word_index = keywords[index].word_index;
        state->keywords[index].token_count = keywords[index].token_count;
        state->keywords[index].token_ids = state->keyword_token_storage + offset;
        if (keywords[index].token_count > 0) {
            memcpy(state->keyword_token_storage + offset, keywords[index].token_ids, (size_t)keywords[index].token_count * sizeof(int32_t));
        }
        offset += keywords[index].token_count;
        if (keyword_strings && state->keyword_string_storage) {
            char* dst = state->keyword_string_storage + ((ptrdiff_t)index * state->keyword_string_stride);
            strcpy(dst, keyword_strings[index]);
            state->keyword_strings[index] = dst;
        }
    }

    if (num_keywords_idxset > 0) {
        state->keywords_idxset = (int32_t*)ctc_decoder_c_alloc_zeroed(&state->allocator, (size_t)num_keywords_idxset, sizeof(int32_t));
        if (!state->keywords_idxset) {
            return -1;
        }
        memcpy(state->keywords_idxset, keywords_idxset, (size_t)num_keywords_idxset * sizeof(int32_t));
    }
    state->num_keywords = num_keywords;
    state->num_keywords_idxset = num_keywords_idxset;
    state->keyword_token_storage_size = total_tokens;
    return 0;
}

int32_t ctc_decoder_c_set_thresholds(
    CTCDecoderCState* state,
    const int32_t* word_indices,
    const float* threshold_values,
    int32_t num_thresholds) {
    int32_t index;
    if (!state || num_thresholds < 0) {
        return -1;
    }
    if (!state->thresholds || !state->threshold_valid) {
        return num_thresholds == 0 ? 0 : -1;
    }
    memset(state->threshold_valid, 0, (size_t)state->num_keywords * sizeof(uint8_t));
    for (index = 0; index < num_thresholds; ++index) {
        int32_t word_index = word_indices[index];
        if (word_index < 0 || word_index >= state->num_keywords) {
            return -1;
        }
        state->thresholds[word_index] = threshold_values[index];
        state->threshold_valid[word_index] = 1;
    }
    return 0;
}

int32_t ctc_decoder_c_advance_frame(
    CTCDecoderCState* state,
    int32_t frame_index,
    const float* probs,
    int32_t vocab_size) {
    int32_t topk_count;
    int32_t filtered_count = 0;
    int32_t token_slot;
    int32_t hyp_index;
    if (!state || !probs || vocab_size <= 0) {
        return -1;
    }
    topk_count = ctc_decoder_c_select_topk(state, probs, vocab_size);
    for (token_slot = 0; token_slot < topk_count; ++token_slot) {
        float prob = state->topk_probs[token_slot];
        int32_t token_id = state->topk_indices[token_slot];
        if (token_id < 0) {
            continue;
        }
        if (prob <= 0.05f) {
            continue;
        }
        if (!ctc_decoder_c_keyword_allowed(state, token_id)) {
            continue;
        }
        state->topk_probs[filtered_count] = prob;
        state->topk_indices[filtered_count] = token_id;
        ++filtered_count;
    }
    if (filtered_count == 0) {
        return 0;
    }

    state->next_hyp_count = 0;
    for (token_slot = 0; token_slot < filtered_count; ++token_slot) {
        int32_t token_id = state->topk_indices[token_slot];
        float token_prob = state->topk_probs[token_slot];
        double token_prob_d = (double)token_prob;
        for (hyp_index = 0; hyp_index < state->cur_hyp_count; ++hyp_index) {
            CTCDecoderCHypothesis* hyp = &state->cur_hyps[hyp_index];
            int32_t last = hyp->prefix_len > 0 ? (int32_t)hyp->prefix[hyp->prefix_len - 1] : -1;
            if (token_id == 0) {
                CTCDecoderCHypothesis* next_hyp = ctc_decoder_c_find_or_add_next_hyp(state, hyp->prefix, hyp->prefix_len);
                if (!next_hyp || ctc_decoder_c_copy_node_refs(next_hyp, hyp, state->config.max_prefix_len) != 0) {
                    return -1;
                }
                next_hyp->pb += hyp->pb * token_prob_d + hyp->pnb * token_prob_d;
                continue;
            }

            if (token_id == last) {
                if (!ctc_decoder_c_float_close_zero(hyp->pnb)) {
                    CTCDecoderCHypothesis* same_hyp = ctc_decoder_c_find_or_add_next_hyp(state, hyp->prefix, hyp->prefix_len);
                    if (!same_hyp || ctc_decoder_c_copy_node_refs(same_hyp, hyp, state->config.max_prefix_len) != 0) {
                        return -1;
                    }
                    same_hyp->pnb += hyp->pnb * token_prob_d;
                    if (same_hyp->node_count > 0) {
                        int32_t ref = (int32_t)same_hyp->node_refs[same_hyp->node_count - 1];
                        if (ref < 0 || ref >= state->node_pool_size) {
                            return -1;
                        }
                        if (token_prob > state->node_pool[ref].prob) {
                            state->node_pool[ref].prob = token_prob;
                            state->node_pool[ref].frame = ctc_decoder_c_wrap_frame(frame_index);
                        }
                    }
                }
                if (!ctc_decoder_c_float_close_zero(hyp->pb)) {
                    int32_t new_prefix_len = hyp->prefix_len + 1;
                    int32_t ref;
                    CTCDecoderCHypothesis* extend_hyp;
                    if (new_prefix_len > state->config.max_prefix_len) {
                        return -1;
                    }
                    if (hyp->prefix_len > 0) {
                        memcpy(state->temp_prefix, hyp->prefix, (size_t)hyp->prefix_len * sizeof(CTCDecoderCPrefixToken));
                    }
                    state->temp_prefix[hyp->prefix_len] = (CTCDecoderCPrefixToken)token_id;
                    extend_hyp = ctc_decoder_c_find_or_add_next_hyp(state, state->temp_prefix, new_prefix_len);
                    if (!extend_hyp || ctc_decoder_c_copy_node_refs(extend_hyp, hyp, state->config.max_prefix_len) != 0) {
                        return -1;
                    }
                    extend_hyp->pnb += hyp->pb * token_prob_d;
                    if (extend_hyp->node_count >= state->config.max_prefix_len) {
                        return -1;
                    }
                    ref = ctc_decoder_c_alloc_node(state, token_id, frame_index, token_prob);
                    if (ref < 0) {
                        return -1;
                    }
                    extend_hyp->node_refs[extend_hyp->node_count] = (CTCDecoderCNodeRef)ref;
                    extend_hyp->node_count += 1;
                }
                continue;
            }

            {
                int32_t new_prefix_len = hyp->prefix_len + 1;
                CTCDecoderCHypothesis* next_hyp;
                if (new_prefix_len > state->config.max_prefix_len) {
                    return -1;
                }
                if (hyp->prefix_len > 0) {
                    memcpy(state->temp_prefix, hyp->prefix, (size_t)hyp->prefix_len * sizeof(CTCDecoderCPrefixToken));
                }
                state->temp_prefix[hyp->prefix_len] = (CTCDecoderCPrefixToken)token_id;
                next_hyp = ctc_decoder_c_find_or_add_next_hyp(state, state->temp_prefix, new_prefix_len);
                if (!next_hyp) {
                    return -1;
                }
                if (next_hyp->node_count > 0) {
                    if (token_prob > state->node_pool[(int32_t)next_hyp->node_refs[next_hyp->node_count - 1]].prob) {
                        int32_t ref = ctc_decoder_c_alloc_node(state, token_id, frame_index, token_prob);
                        if (ref < 0) {
                            return -1;
                        }
                        next_hyp->node_count -= 1;
                        next_hyp->node_refs[next_hyp->node_count] = (CTCDecoderCNodeRef)ref;
                        next_hyp->node_count += 1;
                    }
                } else {
                    int32_t ref;
                    if (ctc_decoder_c_copy_node_refs(next_hyp, hyp, state->config.max_prefix_len) != 0) {
                        return -1;
                    }
                    if (next_hyp->node_count >= state->config.max_prefix_len) {
                        return -1;
                    }
                    ref = ctc_decoder_c_alloc_node(state, token_id, frame_index, token_prob);
                    if (ref < 0) {
                        return -1;
                    }
                    next_hyp->node_refs[next_hyp->node_count] = (CTCDecoderCNodeRef)ref;
                    next_hyp->node_count += 1;
                }
                next_hyp->pnb += hyp->pb * token_prob_d + hyp->pnb * token_prob_d;
            }
        }
    }
    return ctc_decoder_c_finalize_frame(state);
}

CTCDecoderCDetectionResult ctc_decoder_c_execute_detection(CTCDecoderCState* state, int32_t disable_threshold) {
    int32_t hyp_index;
    CTCDecoderCDetectionResult result;
    result.word_index = -1;
    result.keyword = NULL;
    result.score = 0.0;
    result.start_frame = 0;
    result.end_frame = 0;
    result.valid = 0;
    if (!state) {
        return result;
    }
    for (hyp_index = 0; hyp_index < state->cur_hyp_count; ++hyp_index) {
        CTCDecoderCHypothesis* hyp = &state->cur_hyps[hyp_index];
        int32_t keyword_index;
        for (keyword_index = 0; keyword_index < state->num_keywords; ++keyword_index) {
            int32_t offset = 0;
            CTCDecoderCKeyword* keyword = &state->keywords[keyword_index];
            int32_t node_index;
            double score = 1.0;
            int32_t start_frame;
            int32_t end_frame;
            int32_t duration;
            int32_t enough_interval;
            int32_t passed_threshold = disable_threshold;
            const char* keyword_string = NULL;
            if (!ctc_decoder_c_is_sublist(hyp->prefix, hyp->prefix_len, keyword->token_ids, keyword->token_count, &offset)) {
                continue;
            }
            if (offset + keyword->token_count > hyp->node_count) {
                return result;
            }
            start_frame = (int32_t)state->node_pool[(int32_t)hyp->node_refs[offset]].frame;
            end_frame = (int32_t)state->node_pool[(int32_t)hyp->node_refs[offset + keyword->token_count - 1]].frame;
            for (node_index = offset; node_index < offset + keyword->token_count; ++node_index) {
                score *= (double)state->node_pool[(int32_t)hyp->node_refs[node_index]].prob;
            }
            score = sqrt(score);
            duration = ctc_decoder_c_frame_diff((CTCDecoderCNodeFrame)end_frame, (CTCDecoderCNodeFrame)start_frame);
#if CTC_DECODER_C_ENABLE_EXTENDED_API
            if (!state->has_best_decode || score > state->best_score) {
                state->best_word_index = keyword->word_index;
                state->best_score = score;
                state->best_start_frame = start_frame;
                state->best_end_frame = end_frame;
                state->has_best_decode = 1;
            }
#endif
            if (!passed_threshold && keyword->word_index >= 0 && keyword->word_index < state->num_keywords && state->threshold_valid[keyword->word_index] && score >= (double)state->thresholds[keyword->word_index]) {
                passed_threshold = 1;
            }
            enough_interval = (state->last_active_pos == -1) || (ctc_decoder_c_frame_diff((CTCDecoderCNodeFrame)end_frame, (CTCDecoderCNodeFrame)state->last_active_pos) >= state->config.interval_frames);
            if (keyword_index < state->num_keywords && state->keyword_strings) {
                keyword_string = state->keyword_strings[keyword_index];
            }
            if (passed_threshold && duration >= state->config.min_frames && duration <= state->config.max_frames && enough_interval) {
                state->last_active_pos = end_frame;
                result.word_index = keyword->word_index;
                result.keyword = keyword_string;
                result.score = score;
                result.start_frame = start_frame;
                result.end_frame = end_frame;
                result.valid = 1;
                return result;
            }
            return result;
        }
    }
    return result;
}

static CTCDecoderCDetectionResult ctc_decoder_c_invalid_detection_result(void) {
    CTCDecoderCDetectionResult result;
    result.word_index = -1;
    result.keyword = NULL;
    result.score = 0.0;
    result.start_frame = 0;
    result.end_frame = 0;
    result.valid = 0;
    return result;
}

static int32_t ctc_decoder_c_peek_first_hyp_start_frame(const CTCDecoderCState* state) {
    if (!state || state->cur_hyp_count <= 0 || state->cur_hyps[0].node_count <= 0) {
        return -1;
    }
    return (int32_t)state->node_pool[(int32_t)state->cur_hyps[0].node_refs[0]].frame;
}

static void ctc_decoder_c_maybe_reset_stale_beam(
    CTCDecoderCState* state,
    int32_t next_frame_index) {
    int32_t start_frame;
    if (!state || state->config.max_frames <= 0) {
        return;
    }
    start_frame = ctc_decoder_c_peek_first_hyp_start_frame(state);
    if (start_frame >= 0 && ctc_decoder_c_frame_diff(ctc_decoder_c_wrap_frame(next_frame_index), (CTCDecoderCNodeFrame)start_frame) > state->config.max_frames) {
        ctc_decoder_c_reset_beam_search(state);
    }
}

CTCDecoderCDetectionResult ctc_decoder_c_step_and_detect(
    CTCDecoderCState* state,
    int32_t frame_index,
    const float* probs,
    int32_t vocab_size,
    int32_t disable_threshold) {
    CTCDecoderCDetectionResult result;
    if (ctc_decoder_c_advance_frame(state, frame_index, probs, vocab_size) != 0) {
        return ctc_decoder_c_invalid_detection_result();
    }
    result = ctc_decoder_c_execute_detection(state, disable_threshold);
    ctc_decoder_c_maybe_reset_stale_beam(state, frame_index + state->config.frame_step);
    return result;
}

CTCDecoderCDetectionResult ctc_decoder_c_step_and_detect_next(
    CTCDecoderCState* state,
    const float* probs,
    int32_t vocab_size,
    int32_t disable_threshold) {
    CTCDecoderCDetectionResult result;
    int32_t frame_index;
    if (!state) {
        return ctc_decoder_c_invalid_detection_result();
    }
    frame_index = state->next_frame_index;
    if (ctc_decoder_c_advance_frame(state, frame_index, probs, vocab_size) != 0) {
        return ctc_decoder_c_invalid_detection_result();
    }
    result = ctc_decoder_c_execute_detection(state, disable_threshold);
    state->next_frame_index += state->config.frame_step;
    ctc_decoder_c_maybe_reset_stale_beam(state, state->next_frame_index);
    return result;
}

void ctc_decoder_c_reset(CTCDecoderCState* state) {
    if (!state) {
        return;
    }
    ctc_decoder_c_init_beam_state(state);
    state->last_active_pos = -1;
    state->next_frame_index = 0;
#if CTC_DECODER_C_ENABLE_EXTENDED_API
    state->has_best_decode = 0;
    state->best_word_index = -1;
    state->best_score = 0.0;
    state->best_start_frame = 0;
    state->best_end_frame = 0;
#endif
}

void ctc_decoder_c_reset_beam_search(CTCDecoderCState* state) {
    if (!state) {
        return;
    }
    ctc_decoder_c_init_beam_state(state);
}

#if CTC_DECODER_C_ENABLE_EXTENDED_API
CTCDecoderCBestDecodeResult ctc_decoder_c_get_best_decode(const CTCDecoderCState* state) {
    CTCDecoderCBestDecodeResult result;
    result.word_index = -1;
    result.keyword = NULL;
    result.score = 0.0;
    result.start_frame = 0;
    result.end_frame = 0;
    result.valid = 0;
    if (!state || !state->has_best_decode) {
        return result;
    }
    result.word_index = state->best_word_index;
    if (state->best_word_index >= 0 && state->best_word_index < state->num_keywords && state->keyword_strings) {
        result.keyword = state->keyword_strings[state->best_word_index];
    }
    result.score = state->best_score;
    result.start_frame = state->best_start_frame;
    result.end_frame = state->best_end_frame;
    result.valid = 1;
    return result;
}

int32_t ctc_decoder_c_get_first_hyp_start_frame(const CTCDecoderCState* state) {
    return ctc_decoder_c_peek_first_hyp_start_frame(state);
}
#endif

int32_t ctc_decoder_c_num_hypotheses(const CTCDecoderCState* state) {
    if (!state) {
        return 0;
    }
    return state->cur_hyp_count;
}

#if CTC_DECODER_C_ENABLE_DEBUG_HYPOTHESES
const CTCDecoderCHypothesis* ctc_decoder_c_get_hypothesis(const CTCDecoderCState* state, int32_t index) {
    if (!state || index < 0 || index >= state->cur_hyp_count) {
        return NULL;
    }
    if (ctc_decoder_c_materialize_hypothesis((CTCDecoderCState*)state, &((CTCDecoderCState*)state)->cur_hyps[index]) != 0) {
        return NULL;
    }
    return &state->cur_hyps[index];
}
#endif
