/**
 * @file fbank.h
 * @brief Mel-frequency filter bank (fbank) feature extraction for MCU
 * 
 * Lightweight implementation compatible with Kaldi fbank.
 * Optimized for embedded systems with limited resources.
 */

#ifndef FBANK_H_
#define FBANK_H_

#include <stdint.h>
#include <math.h>

#include "tensorflow/lite/micro/micro_log.h"

#ifdef USE_CMSIS_DSP
#include "arm_math.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Configuration constants
#define FBANK_MAX_MEL_BINS 23
#define FBANK_MAX_FFT_SIZE 1024  // Support longer frames (40ms @ 16kHz = 640 samples)
#define FBANK_MAX_FRAME_SIZE 1024  // Support up to 64ms @ 16kHz
#define FBANK_MAX_FILTER_WIDTH 120  // Maximum number of non-zero weights per Mel bin (increased for 1024 FFT)

#ifdef FBANK_USE_PRECOMPUTED_TABLES
#define FBANK_MAX_WORK_BUFFER_FLOATS (FBANK_MAX_FRAME_SIZE + FBANK_MAX_FFT_SIZE * 2 + FBANK_MAX_FFT_SIZE / 2 + 1)
#else
#define FBANK_MAX_WORK_BUFFER_FLOATS (FBANK_MAX_FRAME_SIZE * 2 + FBANK_MAX_FFT_SIZE * 2 + FBANK_MAX_FFT_SIZE / 2 + 1)
#endif
#define FBANK_MAX_WORK_BUFFER_BYTES (FBANK_MAX_WORK_BUFFER_FLOATS * (int32_t)sizeof(float))

// Optional compile-time optimizations for MCU Flash footprint.
// FBANK_USE_PRECOMPUTED_TABLES:
//   0 = runtime compute mel/window (default)
//   1 = use precomputed sparse mel + window tables (from generated header)

// FBANK_USE_FAST_LOG_APPROX:
//   0 = use logf (default, better numerical parity)
//   1 = use lightweight log approximation (smaller libm dependency footprint)

/**
 * @brief Sparse Mel filter bank representation
 */
typedef struct {
    int32_t start_bin;  // Starting FFT bin index
    int32_t end_bin;    // Ending FFT bin index (exclusive)
    float weights[FBANK_MAX_FILTER_WIDTH];  // Non-zero weights
} SparseMelFilter;

/**
 * @brief Fbank configuration structure
 */
typedef struct {
    int32_t sample_rate;       // Sample rate in Hz (e.g., 16000)
    int32_t frame_length_ms;   // Frame length in milliseconds (e.g., 25)
    int32_t frame_shift_ms;    // Frame shift in milliseconds (e.g., 10)
    int32_t num_mel_bins;      // Number of mel filter banks (e.g., 23 or 40)
    int32_t fft_size;          // FFT size (power of 2, e.g., 512)
    float dither;              // Dithering constant (0.0 for no dither)
    float preemph_coeff;       // Pre-emphasis coefficient (0.97)
    int32_t use_energy;        // Whether to use energy (0 or 1)
    float low_freq;            // Low frequency cutoff (Hz, default 20)
    float high_freq;           // High frequency cutoff (Hz, 0=Nyquist)
} FbankConfig;

/**
 * @brief Fbank extractor state structure
 */
typedef struct {
    FbankConfig config;
    
    // Sparse Mel filter bank (saves ~44KB compared to dense matrix)
#ifdef FBANK_USE_PRECOMPUTED_TABLES
    const SparseMelFilter* mel_filters;  // Point to Flash table, no RAM copy
#else
    SparseMelFilter mel_filters[FBANK_MAX_MEL_BINS];
#endif
    
    // Working buffers (pointers to external memory / Flash table)
    const float* window;   // Hamming/Povey window (Flash or RAM)
    float* frame_buffer;   // Windowed frame
    float* fft_real;       // FFT real part
    float* fft_imag;       // FFT imaginary part
    float* power_spectrum_buffer;  // Power spectrum buffer (avoid stack overflow)
#ifndef FBANK_USE_PRECOMPUTED_TABLES
    float* conversion_buffer;  // int16->float conversion buffer for fbank_extract_int16
#endif
    
    // Frame processing state
    int32_t frame_length;      // Frame length in samples
    int32_t frame_shift;       // Frame shift in samples

#ifdef USE_CMSIS_DSP
    arm_rfft_fast_instance_f32 rfft_instance;
#endif
} FbankExtractor;

/**
 * @brief Initialize fbank extractor with default configuration
 * 
 * @param extractor Pointer to FbankExtractor structure
 */
void fbank_init_default(FbankExtractor *extractor);

/**
 * @brief Initialize fbank extractor with custom configuration and external buffers
 * 
 * @param extractor Pointer to FbankExtractor structure
 * @param config Pointer to FbankConfig structure
 * @param work_buffer External buffer for working memory (size from fbank_get_work_buffer_bytes)
 * @return 0 on success, -1 on error
 */
int32_t fbank_init_with_buffer(FbankExtractor *extractor, const FbankConfig *config, void *work_buffer);

/**
 * @brief Get required external work buffer size in bytes for the given config
 *
 * @param config Pointer to FbankConfig structure
 * @return Required bytes, or -1 on invalid config
 */
int32_t fbank_get_work_buffer_bytes(const FbankConfig *config);

/**
 * @brief Initialize fbank extractor with custom configuration
 * 
 * @param extractor Pointer to FbankExtractor structure
 * @param config Pointer to FbankConfig structure
 * @return 0 on success, -1 on error
 */
int32_t fbank_init(FbankExtractor *extractor, const FbankConfig *config);

/**
 * @brief Extract fbank features from audio samples
 * 
 * @param extractor Pointer to initialized FbankExtractor
 * @param samples Input audio samples (int16_t or float)
 * @param num_samples Number of input samples
 * @param features Output feature matrix [num_frames x num_mel_bins]
 * @param max_frames Maximum number of frames to extract
 * @return Number of frames extracted, or -1 on error
 */
int32_t fbank_extract_int16(FbankExtractor *extractor,
                            const int16_t *samples,
                            int32_t num_samples,
                            float *features,
                            int32_t max_frames);

int32_t fbank_extract_float(FbankExtractor *extractor,
                             const float *samples,
                             int32_t num_samples,
                             float *features,
                             int32_t max_frames);

/**
 * @brief Process a single frame
 * 
 * @param extractor Pointer to initialized FbankExtractor
 * @param frame Input frame samples
 * @param frame_len Length of frame
 * @param fbank_out Output fbank features [num_mel_bins]
 * @return 0 on success, -1 on error
 */
int32_t fbank_process_frame(FbankExtractor *extractor,
                            const float *frame,
                            int32_t frame_len,
                            float *fbank_out);

/**
 * @brief Reset extractor state (e.g., pre-emphasis)
 * 
 * @param extractor Pointer to FbankExtractor
 */
void fbank_reset(FbankExtractor *extractor);

/**
 * @brief Calculate number of frames for given number of samples
 * 
 * @param extractor Pointer to FbankExtractor
 * @param num_samples Number of input samples
 * @return Number of frames
 */
int32_t fbank_num_frames(const FbankExtractor *extractor, int32_t num_samples);

// Utility functions (can be internal)

/**
 * @brief Convert Hz to Mel scale
 */
float hz_to_mel(float hz);

/**
 * @brief Convert Mel to Hz scale
 */
float mel_to_hz(float mel);

/**
 * @brief Compute FFT (in-place, power-of-2 size only)
 */
void fft_radix2(float *real, float *imag, int32_t n);

#ifdef __cplusplus
}
#endif

#endif // FBANK_H_