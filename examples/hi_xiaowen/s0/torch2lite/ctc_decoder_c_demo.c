/**
 * @file ctc_decoder_c_demo.c
 * @brief Online-style pure-C demo for feeding int8 TFLite logits into ctc_decoder_c.
 *
 * This demo mirrors the project timing setup shared by S3 / 199K streaming decode:
 *   sample_rate    = 16000 Hz
 *   audio chunk    = 320 samples (20 ms)
 *   fbank window   = 25 ms = 400 samples
 *   fbank shift    = 10 ms = 160 samples
 *   num_mel_bins   = 80
 *   frame_skip     = 3
 *   score_beam     = 3
 *   path_beam      = 20
 *   min_frames     = 5
 *   max_frames     = 250
 *   interval       = 50
 *   max_prefix_len = 64
 *
 * Important:
 *   ctc_decoder_c consumes per-frame probabilities, not raw logits.
 *   So the int8 TFLite logits must be dequantized and softmaxed first.
 *
 * Build example:
 *   gcc -std=c99 -O2 -Wall -Wextra -pedantic -x c \
 *     ctc_decoder_c_demo.c ctc_decoder_c.cc -lm -o ctc_decoder_c_demo
 */

#include "ctc_decoder_c.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DEMO_SAMPLE_RATE 16000
#define DEMO_AUDIO_CHUNK_SAMPLES 320
#define DEMO_FBANK_FRAME_LENGTH_SAMPLES 400
#define DEMO_FBANK_FRAME_SHIFT_SAMPLES 160
#define DEMO_NUM_MEL_BINS 80
#define DEMO_FRAME_SKIP 3
#define DEMO_VOCAB_SIZE 20
#define DEMO_TOTAL_CHUNKS 5000
#define DEMO_NEG_LOGIT (-32)
#define DEMO_BLANK_LOGIT (18)
#define DEMO_PEAK_LOGIT (60)
#define DEMO_MAX_PREFIX_LEN 64

/* Current online 199K streaming thresholds from test_infer_stream_229_tflite_int8/summary.json. */
#define DEMO_THRESHOLD_HI_XIAO_WEN 0.088f
#define DEMO_THRESHOLD_NI_HAO_WEN_WEN 0.0f

static void* demo_allocator_malloc(void* user_data, size_t size) {
    (void)user_data;
    return malloc(size);
}

static void* demo_allocator_calloc(void* user_data, size_t count, size_t size) {
    (void)user_data;
    return calloc(count, size);
}

static void demo_allocator_free(void* user_data, void* ptr) {
    (void)user_data;
    free(ptr);
}

static void fill_background_q_logits(int8_t* q_logits, int32_t vocab_size) {
    int32_t token;
    for (token = 0; token < vocab_size; ++token) {
        q_logits[token] = DEMO_NEG_LOGIT;
    }
    q_logits[0] = DEMO_BLANK_LOGIT;
}

static void fake_tflite_infer_int8(
    int32_t model_output_index,
    int8_t* q_logits,
    int32_t vocab_size) {
    fill_background_q_logits(q_logits, vocab_size);

    /* Event 1: 嗨小问 -> [16, 11, 3] */
    if (model_output_index == 40) {
        q_logits[0] = 6;
        q_logits[16] = DEMO_PEAK_LOGIT;
        return;
    }
    if (model_output_index == 41) {
        q_logits[0] = 6;
        q_logits[11] = DEMO_PEAK_LOGIT;
        return;
    }
    if (model_output_index == 42) {
        q_logits[0] = 6;
        q_logits[3] = DEMO_PEAK_LOGIT;
        return;
    }

    /* Event 2: 你好问问 -> [6, 9, 3, 3], repeated token separated by blank. */
    if (model_output_index == 2200) {
        q_logits[0] = 6;
        q_logits[6] = DEMO_PEAK_LOGIT;
        return;
    }
    if (model_output_index == 2201) {
        q_logits[0] = 6;
        q_logits[9] = DEMO_PEAK_LOGIT;
        return;
    }
    if (model_output_index == 2202) {
        q_logits[0] = 6;
        q_logits[3] = DEMO_PEAK_LOGIT;
        return;
    }
    if (model_output_index == 2203) {
        q_logits[0] = DEMO_PEAK_LOGIT;
        return;
    }
    if (model_output_index == 2204) {
        q_logits[0] = 6;
        q_logits[3] = DEMO_PEAK_LOGIT;
        return;
    }
    if (model_output_index == 2205) {
        q_logits[0] = DEMO_PEAK_LOGIT;
        return;
    }
}

static int32_t pop_ready_fbank_frames(
    int32_t total_samples_seen,
    int32_t* next_fbank_start_sample) {
    int32_t produced = 0;
    while (*next_fbank_start_sample + DEMO_FBANK_FRAME_LENGTH_SAMPLES <= total_samples_seen) {
        ++produced;
        *next_fbank_start_sample += DEMO_FBANK_FRAME_SHIFT_SAMPLES;
    }
    return produced;
}

static void dequantize_logits(
    const int8_t* quantized_logits,
    int32_t vocab_size,
    float scale,
    int32_t zero_point,
    float* float_logits) {
    int32_t i;
    for (i = 0; i < vocab_size; ++i) {
        float_logits[i] = scale * ((float)quantized_logits[i] - (float)zero_point);
    }
}

static void softmax_logits(const float* logits, int32_t vocab_size, float* probs) {
    int32_t i;
    float max_logit = logits[0];
    float sum = 0.0f;
    for (i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    for (i = 0; i < vocab_size; ++i) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    if (sum <= 0.0f) {
        float uniform = 1.0f / (float)vocab_size;
        for (i = 0; i < vocab_size; ++i) {
            probs[i] = uniform;
        }
        return;
    }
    for (i = 0; i < vocab_size; ++i) {
        probs[i] /= sum;
    }
}

int main(void) {
    int32_t chunk_index;
    int32_t status;
    int32_t samples_seen = 0;
    int32_t total_fbank_frames = 0;
    int32_t next_fbank_start_sample = 0;
    int32_t next_model_source_fbank_index = 0;
    int32_t model_output_index = 0;
    int32_t wake_count = 0;

    static const int32_t keyword_hi_xiao_wen_tokens[] = {16, 11, 3};
    static const int32_t keyword_ni_hao_wen_wen_tokens[] = {6, 9, 3, 3};
    static const int32_t keywords_idxset[] = {0, 3, 6, 9, 11, 16};
    static const char* keyword_strings[] = {"嗨小问", "你好问问"};
    static const int32_t threshold_word_indices[] = {0, 1};
    static const float threshold_values[] = {
        DEMO_THRESHOLD_HI_XIAO_WEN,
        DEMO_THRESHOLD_NI_HAO_WEN_WEN,
    };
    static const CTCDecoderCKeyword keywords[] = {
        {0, keyword_hi_xiao_wen_tokens, 3},
        {1, keyword_ni_hao_wen_wen_tokens, 4},
    };

    /* Example TFLite int8 output quantization parameters.
     * Replace these with the real tensor scale / zero_point from your model.
     * If you deploy 199K, threshold usually should come from its own stats/resource config.
     */
    const float logit_scale = 0.125f;
    const int32_t logit_zero_point = 0;

    CTCDecoderCConfig config;
    CTCDecoderCAllocator allocator;
    CTCDecoderCState decoder;
    int8_t q_logits[DEMO_VOCAB_SIZE];
    float frame_logits[DEMO_VOCAB_SIZE];
    float frame_probs[DEMO_VOCAB_SIZE];

    ctc_decoder_c_init_default_config(&config);
    config.score_beam_size = 3;
    config.path_beam_size = 20;
    config.min_frames = 5;
    config.max_frames = 250;
    config.interval_frames = 50;
    config.max_prefix_len = DEMO_MAX_PREFIX_LEN;

    allocator.malloc_fn = demo_allocator_malloc;
    allocator.calloc_fn = demo_allocator_calloc;
    allocator.free_fn = demo_allocator_free;
    allocator.user_data = NULL;

    status = ctc_decoder_c_init_with_allocator(&decoder, &config, &allocator);
    if (status != 0) {
        fprintf(stderr, "ctc_decoder_c_init_with_allocator failed\n");
        return 1;
    }

    status = ctc_decoder_c_set_keywords(
        &decoder,
        keywords,
        2,
        keywords_idxset,
        (int32_t)(sizeof(keywords_idxset) / sizeof(keywords_idxset[0])),
        keyword_strings);
    if (status != 0) {
        fprintf(stderr, "ctc_decoder_c_set_keywords failed\n");
        ctc_decoder_c_free(&decoder);
        return 1;
    }

    status = ctc_decoder_c_set_thresholds(
        &decoder,
        threshold_word_indices,
        threshold_values,
        2);
    if (status != 0) {
        fprintf(stderr, "ctc_decoder_c_set_thresholds failed\n");
        ctc_decoder_c_free(&decoder);
        return 1;
    }

    printf("demo_config: sr=%d chunk=%d(20ms) frame_len=%d(25ms) frame_shift=%d(10ms) mel=%d frame_skip=%d score_beam=%d path_beam=%d min_frames=%d max_frames=%d interval=%d max_prefix_len=%d\n",
           DEMO_SAMPLE_RATE,
           DEMO_AUDIO_CHUNK_SAMPLES,
           DEMO_FBANK_FRAME_LENGTH_SAMPLES,
           DEMO_FBANK_FRAME_SHIFT_SAMPLES,
           DEMO_NUM_MEL_BINS,
           DEMO_FRAME_SKIP,
           config.score_beam_size,
           config.path_beam_size,
           config.min_frames,
           config.max_frames,
           config.interval_frames,
           decoder.config.max_prefix_len);
    printf("keywords: 嗨小问=[16,11,3] threshold=%.3f; 你好问问=[6,9,3,3] threshold=%.3f\n",
           DEMO_THRESHOLD_HI_XIAO_WEN,
           DEMO_THRESHOLD_NI_HAO_WEN_WEN);

    for (chunk_index = 0; chunk_index < DEMO_TOTAL_CHUNKS; ++chunk_index) {
        int32_t new_fbank_frames;

        samples_seen += DEMO_AUDIO_CHUNK_SAMPLES;
        new_fbank_frames = pop_ready_fbank_frames(samples_seen, &next_fbank_start_sample);
        total_fbank_frames += new_fbank_frames;

        while (next_model_source_fbank_index < total_fbank_frames) {
            int32_t decoder_frame_index = next_model_source_fbank_index;
            CTCDecoderCDetectionResult detection;
            int32_t wake_flag;

            fake_tflite_infer_int8(model_output_index, q_logits, DEMO_VOCAB_SIZE);
            dequantize_logits(q_logits, DEMO_VOCAB_SIZE, logit_scale, logit_zero_point, frame_logits);
            softmax_logits(frame_logits, DEMO_VOCAB_SIZE, frame_probs);

            detection = ctc_decoder_c_step_and_detect(
                &decoder,
                decoder_frame_index,
                frame_probs,
                DEMO_VOCAB_SIZE,
                0);
            wake_flag = detection.valid;

            if (wake_flag) {
                ++wake_count;
                printf(
                    "chunk=%d samples_seen=%d model_output=%d decoder_frame=%d wake_flag=%d keyword=%s score=%.8f start=%d end=%d\n",
                    chunk_index,
                    samples_seen,
                    model_output_index,
                    decoder_frame_index,
                    wake_flag,
                    detection.keyword ? detection.keyword : "",
                    detection.score,
                    detection.start_frame,
                    detection.end_frame);
                ctc_decoder_c_reset_beam_search(&decoder);
            }

            ++model_output_index;
            next_model_source_fbank_index += DEMO_FRAME_SKIP;
        }
    }

    printf("summary: total_chunks=%d total_fbank_frames=%d model_outputs=%d total_wakeups=%d\n",
           DEMO_TOTAL_CHUNKS,
           total_fbank_frames,
           model_output_index,
           wake_count);

    ctc_decoder_c_free(&decoder);
    return 0;
}
