/**
 * @file fbank.c
 * @brief Mel-frequency filter bank implementation for MCU
 */

#include "fbank.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef FBANK_USE_PRECOMPUTED_TABLES
#include "fbank_tables_fixed_config.h"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// Mel scale conversion (HTK formula, same as Kaldi)
float hz_to_mel(float hz) {
    return 1127.0f * logf(1.0f + hz / 700.0f);
}

float mel_to_hz(float mel) {
    return 700.0f * (expf(mel / 1127.0f) - 1.0f);
}

// Initialize Povey window (Kaldi default)
#ifndef FBANK_USE_PRECOMPUTED_TABLES
static void init_povey_window(float *window, int32_t window_size) {
    for (int32_t i = 0; i < window_size; i++) {
        float a = 2.0f * M_PI_F * i / (window_size - 1);
        window[i] = powf(0.5f - 0.5f * cosf(a), 0.85f);
    }
}

// Initialize mel filter banks (sparse storage for memory efficiency)
static int32_t init_mel_banks(FbankExtractor *extractor) {
    const FbankConfig *cfg = &extractor->config;
    int32_t num_bins = cfg->fft_size / 2 + 1;
    float nyquist = cfg->sample_rate / 2.0f;
    
    float low_freq = cfg->low_freq;
    float high_freq = cfg->high_freq;
    if (high_freq <= 0.0f || high_freq > nyquist) {
        high_freq = nyquist;
    }
    
    // Convert to mel scale
    float mel_low = hz_to_mel(low_freq);
    float mel_high = hz_to_mel(high_freq);
    float mel_step = (mel_high - mel_low) / (cfg->num_mel_bins + 1);
    
    // Create mel bin centers
    float mel_centers[FBANK_MAX_MEL_BINS + 2];
    for (int32_t i = 0; i <= cfg->num_mel_bins + 1; i++) {
        mel_centers[i] = mel_low + i * mel_step;
    }
    
    // Build sparse triangular filters
    for (int32_t i = 0; i < cfg->num_mel_bins; i++) {
        float left_mel = mel_centers[i];
        float center_mel = mel_centers[i + 1];
        float right_mel = mel_centers[i + 2];
        
        float left_hz = mel_to_hz(left_mel);
        float center_hz = mel_to_hz(center_mel);
        float right_hz = mel_to_hz(right_mel);
        
        // Convert Hz to FFT bin indices
        int32_t left_bin = (int32_t)(left_hz * num_bins / nyquist);
        int32_t right_bin = (int32_t)(right_hz * num_bins / nyquist) + 1;
        
        // Clamp to valid range
        if (left_bin < 0) left_bin = 0;
        if (right_bin > num_bins) right_bin = num_bins;
        
        extractor->mel_filters[i].start_bin = left_bin;
        extractor->mel_filters[i].end_bin = right_bin;
        
        // Calculate and store only non-zero weights
        int32_t filter_width = right_bin - left_bin;
        if (filter_width > FBANK_MAX_FILTER_WIDTH) {
            // Debug output for troubleshooting
            (void)i;
            (void)filter_width;
            return -1;  // Filter too wide
        }
        
        for (int32_t j = 0; j < filter_width; j++) {
            int32_t fft_bin = left_bin + j;
            float freq = fft_bin * nyquist / (num_bins - 1);
            
            float weight = 0.0f;
            if (freq >= left_hz && freq <= right_hz) {
                if (freq <= center_hz) {
                    weight = (freq - left_hz) / (center_hz - left_hz);
                } else {
                    weight = (right_hz - freq) / (right_hz - center_hz);
                }
            }
            extractor->mel_filters[i].weights[j] = weight;
        }
    }
    
    return 0;
}
#endif

#ifdef FBANK_USE_FAST_LOG_APPROX
// Lightweight ln(x) approximation for x > 0.
// Helps reduce libm footprint on constrained MCU builds.
static inline float fast_log_approx(float x) {
    union {
        float f;
        uint32_t i;
    } v;
    v.f = x;

    int32_t exp2 = (int32_t)((v.i >> 23) & 0xFF) - 127;
    v.i = (v.i & 0x7FFFFF) | 0x3F800000;  // normalize mantissa to [1, 2)
    float m = v.f;
    float y = m - 1.0f;

    // Quadratic fit for log2(m) on [1, 2), then convert to natural log.
    float log2_m = y * (1.3465557f + y * (-0.3606742f));
    return (exp2 + log2_m) * 0.69314718056f;
}
#endif

void fbank_init_default(FbankExtractor *extractor) {
    FbankConfig config = {
        .sample_rate = 16000,
        .frame_length_ms = 25,
        .frame_shift_ms = 10,
        .num_mel_bins = 23,
        .fft_size = 512,
        .dither = 0.0f,
        .preemph_coeff = 0.97f,
        .use_energy = 0,
        .low_freq = 20.0f,
        .high_freq = 0.0f  // Will be set to Nyquist
    };
    fbank_init(extractor, &config);
}

int32_t fbank_get_work_buffer_bytes(const FbankConfig *config) {
    if (!config) return -1;
    if (config->fft_size <= 0 || config->fft_size > FBANK_MAX_FFT_SIZE) return -1;
    if (config->num_mel_bins <= 0 || config->num_mel_bins > FBANK_MAX_MEL_BINS) return -1;

    int32_t frame_length = (config->sample_rate * config->frame_length_ms) / 1000;
    if (frame_length <= 0 || frame_length > FBANK_MAX_FRAME_SIZE) return -1;
    if (frame_length > config->fft_size) return -1;

    // precomputed mode: frame_buffer + fft_real + fft_imag + power_spectrum
    // runtime mode: window + frame_buffer + fft_real + fft_imag + power_spectrum
#ifdef FBANK_USE_PRECOMPUTED_TABLES
    int32_t total_floats = frame_length + config->fft_size * 2 + (config->fft_size / 2 + 1);
#else
    int32_t total_floats = frame_length * 2 + config->fft_size * 2 + (config->fft_size / 2 + 1);
#endif
    return (int32_t)(total_floats * (int32_t)sizeof(float));
}

int32_t fbank_init_with_buffer(FbankExtractor *extractor, const FbankConfig *config, void *work_buffer) {
    if (!extractor || !config || !work_buffer) return -1;
    
    // Validate parameters
    if (config->num_mel_bins > FBANK_MAX_MEL_BINS) return -1;
    if (config->fft_size > FBANK_MAX_FFT_SIZE) return -1;
    
    // Copy configuration
    memcpy(&extractor->config, config, sizeof(FbankConfig));
    
    // Calculate frame parameters
    extractor->frame_length = (config->sample_rate * config->frame_length_ms) / 1000;
    extractor->frame_shift = (config->sample_rate * config->frame_shift_ms) / 1000;
    
    if (extractor->frame_length > FBANK_MAX_FRAME_SIZE) return -1;
    if (extractor->frame_length > config->fft_size) return -1;
    
    // Assign external buffers (compact layout by active config)
    uint8_t* buf_ptr = (uint8_t*)work_buffer;

#ifdef FBANK_USE_PRECOMPUTED_TABLES
    // Window is directly referenced from precomputed Flash table (no RAM copy).
#else
    extractor->window = (const float*)buf_ptr;
    buf_ptr += (size_t)extractor->frame_length * sizeof(float);
#endif

    extractor->frame_buffer = (float*)buf_ptr;
    buf_ptr += (size_t)extractor->frame_length * sizeof(float);
    
    extractor->fft_real = (float*)buf_ptr;
    buf_ptr += (size_t)config->fft_size * sizeof(float);
    
    extractor->fft_imag = (float*)buf_ptr;
    buf_ptr += (size_t)config->fft_size * sizeof(float);
    
    extractor->power_spectrum_buffer = (float*)buf_ptr;
    
    // Initialize window + sparse mel filters
#ifdef FBANK_USE_PRECOMPUTED_TABLES
    // Fixed-table mode: use precomputed constants generated offline.
    if (config->sample_rate != FBANK_TABLE_SAMPLE_RATE ||
        extractor->frame_length != FBANK_TABLE_FRAME_LENGTH ||
        config->num_mel_bins != FBANK_TABLE_NUM_MEL_BINS ||
        config->fft_size != FBANK_TABLE_FFT_SIZE) {
        return -1;
    }
    extractor->window = FBANK_PRECOMPUTED_WINDOW;
    extractor->mel_filters = FBANK_PRECOMPUTED_MEL_FILTERS;
    MicroPrintf("using FBANK_USE_PRECOMPUTED_TABLES");
#else
    init_povey_window((float*)extractor->window, extractor->frame_length);
    if (init_mel_banks(extractor) != 0) {
        return -1;
    }
#endif

#ifdef USE_CMSIS_DSP
    MicroPrintf("before arm_rfft_fast_init_f32");
    arm_rfft_fast_init_f32(&extractor->rfft_instance, (uint16_t)config->fft_size);
    MicroPrintf("after arm_rfft_fast_init_f32");
#endif

    // Reset state
    fbank_reset(extractor);
    
    return 0;
}

int32_t fbank_init(FbankExtractor *extractor, const FbankConfig *config) {
    // This version uses internal static buffers (not recommended for embedded)
    // Buffer layout:
    //   precomputed mode: frame_buffer + fft_real + fft_imag + power_spectrum
    //   runtime mode: window + frame_buffer + fft_real + fft_imag + power_spectrum
    static float static_work_buffer[FBANK_MAX_WORK_BUFFER_FLOATS];
    return fbank_init_with_buffer(extractor, config, static_work_buffer);
}

void fbank_reset(FbankExtractor *extractor) {
    // Currently no state to reset (pre-emphasis is per-frame)
    (void)extractor;
}

int32_t fbank_num_frames(const FbankExtractor *extractor, int32_t num_samples) {
    if (num_samples < extractor->frame_length) return 0;
    return 1 + (num_samples - extractor->frame_length) / extractor->frame_shift;
}

// Radix-2 FFT (Cooley-Tukey algorithm)
void fft_radix2(float *real, float *imag, int32_t n) {
    // Bit reversal
    int32_t j = 0;
    for (int32_t i = 0; i < n - 1; i++) {
        if (i < j) {
            float temp = real[i];
            real[i] = real[j];
            real[j] = temp;
            
            temp = imag[i];
            imag[i] = imag[j];
            imag[j] = temp;
        }
        
        int32_t k = n / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }
    
    // FFT computation
    for (int32_t len = 2; len <= n; len *= 2) {
        float angle = -2.0f * M_PI_F / len;
        float wlen_real = cosf(angle);
        float wlen_imag = sinf(angle);
        
        for (int32_t i = 0; i < n; i += len) {
            float w_real = 1.0f;
            float w_imag = 0.0f;
            
            for (int32_t j = 0; j < len / 2; j++) {
                int32_t u_idx = i + j;
                int32_t v_idx = i + j + len / 2;
                
                float u_real = real[u_idx];
                float u_imag = imag[u_idx];
                float v_real = real[v_idx];
                float v_imag = imag[v_idx];
                
                float t_real = w_real * v_real - w_imag * v_imag;
                float t_imag = w_real * v_imag + w_imag * v_real;
                
                real[u_idx] = u_real + t_real;
                imag[u_idx] = u_imag + t_imag;
                real[v_idx] = u_real - t_real;
                imag[v_idx] = u_imag - t_imag;
                
                float w_temp = w_real;
                w_real = w_real * wlen_real - w_imag * wlen_imag;
                w_imag = w_temp * wlen_imag + w_imag * wlen_real;
            }
        }
    }
}

int32_t fbank_process_frame(FbankExtractor *extractor,
                            const float *frame,
                            int32_t frame_len,
                            float *fbank_out) {
    if (!extractor || !frame || !fbank_out) return -1;
    if (frame_len != extractor->frame_length) return -1;
    
    const FbankConfig *cfg = &extractor->config;
    
    // Apply pre-emphasis (use first sample as state, matching Kaldi behavior)
    extractor->frame_buffer[0] = frame[0] * (1.0f - cfg->preemph_coeff);
    for (int32_t i = 1; i < frame_len; i++) {
        extractor->frame_buffer[i] = frame[i] - cfg->preemph_coeff * frame[i-1];
    }
    
    // Apply window
    for (int32_t i = 0; i < frame_len; i++) {
        extractor->frame_buffer[i] *= extractor->window[i];
    }
    memset(extractor->fft_real, 0, cfg->fft_size * sizeof(float));
    memcpy(extractor->fft_real, extractor->frame_buffer, frame_len * sizeof(float));

    int32_t num_bins = cfg->fft_size / 2 + 1;
    float* power_spectrum = extractor->power_spectrum_buffer;

#ifdef USE_CMSIS_DSP
    // arm_rfft_fast_f32 reads fft_real (input) and writes interleaved complex
    // output to fft_imag: [Re[0], Re[N/2], Re[1], Im[1], Re[2], Im[2], ...]
    MicroPrintf("before arm_rfft_fast_f32");    
    arm_rfft_fast_f32(&extractor->rfft_instance,
                      extractor->fft_real, extractor->fft_imag, 0);
    power_spectrum[0]          = extractor->fft_imag[0] * extractor->fft_imag[0];
    power_spectrum[num_bins-1] = extractor->fft_imag[1] * extractor->fft_imag[1];
    arm_cmplx_mag_squared_f32(&extractor->fft_imag[2],
                               &power_spectrum[1],
                               (uint32_t)(num_bins - 2));
    MicroPrintf("after arm_rfft_fast_f32");
#else
    MicroPrintf("before fft_radix2");
    memset(extractor->fft_imag, 0, cfg->fft_size * sizeof(float));
    fft_radix2(extractor->fft_real, extractor->fft_imag, cfg->fft_size);
    for (int32_t i = 0; i < num_bins; i++) {
        power_spectrum[i] = extractor->fft_real[i] * extractor->fft_real[i] +
                           extractor->fft_imag[i] * extractor->fft_imag[i];
    }
    MicroPrintf("before fft_radix2");
#endif
    
    // Apply sparse mel filter banks and compute log energy
    for (int32_t i = 0; i < cfg->num_mel_bins; i++) {
        float energy = 0.0f;
        const SparseMelFilter* filter = &extractor->mel_filters[i];
        
        // Only iterate over non-zero weights
        int32_t filter_width = filter->end_bin - filter->start_bin;
        for (int32_t j = 0; j < filter_width; j++) {
            int32_t fft_bin = filter->start_bin + j;
            energy += filter->weights[j] * power_spectrum[fft_bin];
        }
        
        // Compute log energy (with floor to match Kaldi: exp(-15.942) ≈ 1.19e-07)
        const float kaldi_energy_floor = 1.1925e-07f;
        if (energy < kaldi_energy_floor) energy = kaldi_energy_floor;
#ifdef FBANK_USE_FAST_LOG_APPROX
        fbank_out[i] = fast_log_approx(energy);
#else
        fbank_out[i] = logf(energy);
#endif
    }
    
    return 0;
}

int32_t fbank_extract_float(FbankExtractor *extractor,
                             const float *samples,
                             int32_t num_samples,
                             float *features,
                             int32_t max_frames) {
    if (!extractor || !samples || !features) return -1;
    
    int32_t num_frames = fbank_num_frames(extractor, num_samples);
    if (num_frames > max_frames) num_frames = max_frames;
    
    // No need to reset - each frame is processed independently
    for (int32_t i = 0; i < num_frames; i++) {
        int32_t offset = i * extractor->frame_shift;
        float *feat_out = features + i * extractor->config.num_mel_bins;
        
        if (fbank_process_frame(extractor, samples + offset, 
                               extractor->frame_length, feat_out) != 0) {
            return -1;
        }
    }
    
    return num_frames;
}

int32_t fbank_extract_int16(FbankExtractor *extractor,
                            const int16_t *samples,
                            int32_t num_samples,
                            float *features,
                            int32_t max_frames) {
    if (!extractor || !samples || !features) return -1;
    
    int32_t num_frames = fbank_num_frames(extractor, num_samples);
    if (num_frames > max_frames) num_frames = max_frames;
    
    for (int32_t i = 0; i < num_frames; i++) {
        int32_t offset = i * extractor->frame_shift;
        float *feat_out = features + i * extractor->config.num_mel_bins;

#ifdef FBANK_USE_PRECOMPUTED_TABLES
        // Inline int16 path: pre-emphasis + windowing fused, no conversion buffer needed.
        extractor->frame_buffer[0] =
            ((float)samples[offset]) * (1.0f - extractor->config.preemph_coeff);
        for (int32_t j = 1; j < extractor->frame_length; j++) {
            float cur = (float)samples[offset + j];
            float prev = (float)samples[offset + j - 1];
            extractor->frame_buffer[j] = cur - extractor->config.preemph_coeff * prev;
        }
        for (int32_t j = 0; j < extractor->frame_length; j++) {
            extractor->frame_buffer[j] *= extractor->window[j];
        }

        memset(extractor->fft_real, 0, extractor->config.fft_size * sizeof(float));
        memcpy(extractor->fft_real, extractor->frame_buffer,
               (size_t)extractor->frame_length * sizeof(float));

        int32_t num_bins = extractor->config.fft_size / 2 + 1;
        float* power_spectrum = extractor->power_spectrum_buffer;

#ifdef USE_CMSIS_DSP
        arm_rfft_fast_f32(&extractor->rfft_instance,
                          extractor->fft_real, extractor->fft_imag, 0);
        power_spectrum[0]          = extractor->fft_imag[0] * extractor->fft_imag[0];
        power_spectrum[num_bins-1] = extractor->fft_imag[1] * extractor->fft_imag[1];
        arm_cmplx_mag_squared_f32(&extractor->fft_imag[2],
                                  &power_spectrum[1],
                                  (uint32_t)(num_bins - 2));
#else
        memset(extractor->fft_imag, 0, extractor->config.fft_size * sizeof(float));
        fft_radix2(extractor->fft_real, extractor->fft_imag, extractor->config.fft_size);
        for (int32_t j = 0; j < num_bins; j++) {
            power_spectrum[j] = extractor->fft_real[j] * extractor->fft_real[j] +
                                extractor->fft_imag[j] * extractor->fft_imag[j];
        }
#endif

        for (int32_t mel_i = 0; mel_i < extractor->config.num_mel_bins; mel_i++) {
            float energy = 0.0f;
            const SparseMelFilter* filter = &extractor->mel_filters[mel_i];
            int32_t filter_width = filter->end_bin - filter->start_bin;
            for (int32_t j = 0; j < filter_width; j++) {
                int32_t fft_bin = filter->start_bin + j;
                energy += filter->weights[j] * power_spectrum[fft_bin];
            }
            const float kaldi_energy_floor = 1.1925e-07f;
            if (energy < kaldi_energy_floor) energy = kaldi_energy_floor;
#ifdef FBANK_USE_FAST_LOG_APPROX
            // MicroPrintf("using FBANK_USE_FAST_LOG_APPROX");
            feat_out[mel_i] = fast_log_approx(energy);
#else
            feat_out[mel_i] = logf(energy);
#endif
        }
#else
        // Original path: convert int16->float via conversion_buffer, then delegate to fbank_process_frame.
        for (int32_t j = 0; j < extractor->frame_length; j++) {
            extractor->conversion_buffer[j] = (float)samples[offset + j];
        }
        if (fbank_process_frame(extractor, extractor->conversion_buffer,
                                extractor->frame_length, feat_out) != 0) {
            return -1;
        }
#endif

    }
    
    return num_frames;
}