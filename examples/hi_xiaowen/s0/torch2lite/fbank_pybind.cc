#include <algorithm>
#include <cstdint>
#include <vector>

#include <torch/extension.h>

#include "fbank.h"

namespace py = pybind11;

namespace {

torch::Tensor ensure_float_waveform_1d(torch::Tensor waveform) {
    TORCH_CHECK(waveform.device().is_cpu(), "waveform must be on CPU");
    TORCH_CHECK(
        waveform.scalar_type() == torch::kFloat32 || waveform.scalar_type() == torch::kFloat64,
        "waveform must be float32/float64");

    if (waveform.dim() == 2) {
        TORCH_CHECK(waveform.size(0) == 1, "only mono waveform is supported, got shape=", waveform.sizes());
        waveform = waveform.squeeze(0);
    }
    TORCH_CHECK(waveform.dim() == 1, "waveform must be 1D or (1, N), got shape=", waveform.sizes());
    return waveform.contiguous().to(torch::kFloat32);
}

torch::Tensor ensure_int16_waveform_1d(torch::Tensor waveform) {
    TORCH_CHECK(waveform.device().is_cpu(), "waveform must be on CPU");
    TORCH_CHECK(waveform.scalar_type() == torch::kInt16, "waveform must be int16");

    if (waveform.dim() == 2) {
        TORCH_CHECK(waveform.size(0) == 1, "only mono waveform is supported, got shape=", waveform.sizes());
        waveform = waveform.squeeze(0);
    }
    TORCH_CHECK(waveform.dim() == 1, "waveform must be 1D or (1, N), got shape=", waveform.sizes());
    return waveform.contiguous();
}

FbankConfig build_config(
    int64_t num_mel_bins,
    double frame_length,
    double frame_shift,
    double dither,
    double sample_frequency,
    double low_freq,
    double high_freq,
    double preemphasis_coefficient,
    bool remove_dc_offset,
    bool round_to_power_of_two,
    bool snip_edges,
    int64_t fft_size) {
    FbankConfig config{};
    config.sample_rate = static_cast<int32_t>(sample_frequency);
    config.frame_length_ms = static_cast<int32_t>(frame_length);
    config.frame_shift_ms = static_cast<int32_t>(frame_shift);
    config.num_mel_bins = static_cast<int32_t>(num_mel_bins);
    TORCH_CHECK(fft_size >= 0, "fft_size must be >= 0, got ", fft_size);
    TORCH_CHECK(fft_size <= static_cast<int64_t>(INT32_MAX), "fft_size is too large");
    config.fft_size = static_cast<int32_t>(fft_size);
    config.dither = static_cast<float>(dither);
    config.preemph_coeff = static_cast<float>(preemphasis_coefficient);
    config.use_energy = 0;
    config.low_freq = static_cast<float>(low_freq);
    config.high_freq = static_cast<float>(high_freq);
    config.remove_dc_offset = remove_dc_offset ? 1 : 0;
    config.round_to_power_of_two = round_to_power_of_two ? 1 : 0;
    config.snip_edges = snip_edges ? 1 : 0;
    return config;
}

int32_t resolve_num_samples(const torch::Tensor& waveform_1d, int64_t num_samples) {
    int64_t total = waveform_1d.numel();
    if (num_samples < 0) {
        num_samples = total;
    }
    TORCH_CHECK(num_samples >= 0, "num_samples must be >= 0, got ", num_samples);
    TORCH_CHECK(num_samples <= total,
                "num_samples exceeds waveform length: ",
                num_samples,
                " > ",
                total);
    TORCH_CHECK(num_samples <= static_cast<int64_t>(INT32_MAX), "num_samples is too large");
    return static_cast<int32_t>(num_samples);
}

int32_t resolve_max_frames(int32_t num_frames, int64_t max_frames) {
    if (max_frames < 0) {
        return num_frames;
    }
    TORCH_CHECK(max_frames <= static_cast<int64_t>(INT32_MAX), "max_frames is too large");
    return std::min(num_frames, static_cast<int32_t>(max_frames));
}

torch::Tensor empty_features(int64_t num_mel_bins) {
    return torch::empty({0, static_cast<long>(num_mel_bins)}, torch::TensorOptions().dtype(torch::kFloat32));
}

class PyFbankExtractor {
public:
    PyFbankExtractor(
        int64_t num_mel_bins = 80,
        double frame_length = 25.0,
        double frame_shift = 10.0,
        double dither = 0.0,
        double energy_floor = 0.0,
        double sample_frequency = 16000.0,
        double low_freq = 20.0,
        double high_freq = 0.0,
        double preemphasis_coefficient = 0.97,
        bool remove_dc_offset = true,
        bool round_to_power_of_two = true,
        bool snip_edges = true,
        int64_t fft_size = 0)
        : config_(build_config(
              num_mel_bins,
              frame_length,
              frame_shift,
              dither,
              sample_frequency,
              low_freq,
              high_freq,
              preemphasis_coefficient,
              remove_dc_offset,
              round_to_power_of_two,
              snip_edges,
              fft_size)) {
        (void)energy_floor;
        work_buffer_bytes_ = fbank_get_work_buffer_bytes(&config_);
        TORCH_CHECK(work_buffer_bytes_ > 0, "fbank_get_work_buffer_bytes failed");
        work_buffer_.resize(static_cast<size_t>(work_buffer_bytes_));
        int32_t init_status = fbank_init_with_buffer(&extractor_, &config_, work_buffer_.data());
        TORCH_CHECK(init_status == 0, "fbank_init_with_buffer failed");
    }

    void reset() {
        fbank_reset(&extractor_);
    }

    int64_t num_frames(int64_t num_samples) const {
        TORCH_CHECK(num_samples >= 0, "num_samples must be >= 0, got ", num_samples);
        TORCH_CHECK(num_samples <= static_cast<int64_t>(INT32_MAX), "num_samples is too large");
        return static_cast<int64_t>(fbank_num_frames(&extractor_, static_cast<int32_t>(num_samples)));
    }

    torch::Tensor process_frame_float(torch::Tensor waveform) {
        auto waveform_1d = ensure_float_waveform_1d(waveform);
        TORCH_CHECK(waveform_1d.numel() == extractor_.frame_length,
                    "frame must contain exactly ", extractor_.frame_length, " samples, got ", waveform_1d.numel());
        auto features = torch::empty({1, static_cast<long>(extractor_.config.num_mel_bins)},
                                     torch::TensorOptions().dtype(torch::kFloat32));
        int32_t status = fbank_process_frame(
            &extractor_, waveform_1d.data_ptr<float>(), extractor_.frame_length, features.data_ptr<float>());
        TORCH_CHECK(status == 0, "fbank_process_frame failed");
        return features;
    }

    torch::Tensor process_frame_int16(torch::Tensor waveform) {
        auto waveform_1d = ensure_int16_waveform_1d(waveform);
        TORCH_CHECK(waveform_1d.numel() == extractor_.frame_length,
                    "frame must contain exactly ", extractor_.frame_length, " samples, got ", waveform_1d.numel());
        auto features = torch::empty({1, static_cast<long>(extractor_.config.num_mel_bins)},
                                     torch::TensorOptions().dtype(torch::kFloat32));
        int32_t status = fbank_process_frame_int16(
            &extractor_, waveform_1d.data_ptr<int16_t>(), extractor_.frame_length, features.data_ptr<float>());
        TORCH_CHECK(status == 0, "fbank_process_frame_int16 failed");
        return features;
    }

    torch::Tensor extract_float(torch::Tensor waveform, int64_t num_samples = -1, int64_t max_frames = -1) {
        auto waveform_1d = ensure_float_waveform_1d(waveform);
        int32_t samples_to_process = resolve_num_samples(waveform_1d, num_samples);
        int32_t num_frames_out = fbank_num_frames(&extractor_, samples_to_process);
        int32_t frames_to_extract = resolve_max_frames(num_frames_out, max_frames);
        if (frames_to_extract <= 0) {
            return empty_features(extractor_.config.num_mel_bins);
        }

        auto features = torch::empty(
            {static_cast<long>(frames_to_extract), static_cast<long>(extractor_.config.num_mel_bins)},
            torch::TensorOptions().dtype(torch::kFloat32));
        int32_t extracted = fbank_extract_float(
            &extractor_,
            waveform_1d.data_ptr<float>(),
            samples_to_process,
            features.data_ptr<float>(),
            frames_to_extract);
        TORCH_CHECK(extracted >= 0, "fbank_extract_float failed");
        if (extracted == frames_to_extract) {
            return features;
        }
        return features.narrow(0, 0, extracted).contiguous();
    }

    torch::Tensor extract_int16(torch::Tensor waveform, int64_t num_samples = -1, int64_t max_frames = -1) {
        auto waveform_1d = ensure_int16_waveform_1d(waveform);
        int32_t samples_to_process = resolve_num_samples(waveform_1d, num_samples);
        int32_t num_frames_out = fbank_num_frames(&extractor_, samples_to_process);
        int32_t frames_to_extract = resolve_max_frames(num_frames_out, max_frames);
        if (frames_to_extract <= 0) {
            return empty_features(extractor_.config.num_mel_bins);
        }

        auto features = torch::empty(
            {static_cast<long>(frames_to_extract), static_cast<long>(extractor_.config.num_mel_bins)},
            torch::TensorOptions().dtype(torch::kFloat32));
        int32_t extracted = fbank_extract_int16(
            &extractor_,
            waveform_1d.data_ptr<int16_t>(),
            samples_to_process,
            features.data_ptr<float>(),
            frames_to_extract);
        TORCH_CHECK(extracted >= 0, "fbank_extract_int16 failed");
        if (extracted == frames_to_extract) {
            return features;
        }
        return features.narrow(0, 0, extracted).contiguous();
    }

    int64_t frame_length() const { return static_cast<int64_t>(extractor_.frame_length); }
    int64_t frame_shift() const { return static_cast<int64_t>(extractor_.frame_shift); }
    int64_t fft_size() const { return static_cast<int64_t>(extractor_.config.fft_size); }
    int64_t num_mel_bins() const { return static_cast<int64_t>(extractor_.config.num_mel_bins); }
    int64_t work_buffer_bytes() const { return static_cast<int64_t>(work_buffer_bytes_); }

private:
    FbankConfig config_{};
    FbankExtractor extractor_{};
    std::vector<uint8_t> work_buffer_;
    int32_t work_buffer_bytes_ = 0;
};

class PyStreamingFbankExtractor {
public:
    PyStreamingFbankExtractor(
        int64_t num_mel_bins = 80,
        double frame_length = 25.0,
        double frame_shift = 10.0,
        double dither = 0.0,
        double energy_floor = 0.0,
        double sample_frequency = 16000.0,
        double low_freq = 20.0,
        double high_freq = 0.0,
        double preemphasis_coefficient = 0.97,
        bool remove_dc_offset = true,
        bool round_to_power_of_two = true,
        bool snip_edges = true,
        int64_t fft_size = 0)
        : config_(build_config(
              num_mel_bins,
              frame_length,
              frame_shift,
              dither,
              sample_frequency,
              low_freq,
              high_freq,
              preemphasis_coefficient,
              remove_dc_offset,
              round_to_power_of_two,
              snip_edges,
              fft_size)) {
        (void)energy_floor;
        int32_t init_status = fbank_stream_init(&state_, &config_);
        TORCH_CHECK(init_status == 0, "fbank_stream_init failed");
    }

    ~PyStreamingFbankExtractor() {
        fbank_stream_free(&state_);
    }

    PyStreamingFbankExtractor(const PyStreamingFbankExtractor&) = delete;
    PyStreamingFbankExtractor& operator=(const PyStreamingFbankExtractor&) = delete;

    void reset() {
        fbank_stream_reset(&state_);
    }

    int64_t pending_samples() const {
        return static_cast<int64_t>(fbank_stream_pending_samples(&state_));
    }

    torch::Tensor accept_float(torch::Tensor waveform, int64_t num_samples = -1, int64_t max_frames = -1) {
        auto waveform_1d = ensure_float_waveform_1d(waveform);
        int32_t samples_to_process = resolve_num_samples(waveform_1d, num_samples);
        int32_t max_possible = fbank_num_frames(&state_.extractor, state_.buffered_samples + samples_to_process);
        int32_t frames_to_extract = resolve_max_frames(max_possible, max_frames);
        if (frames_to_extract <= 0) {
            int32_t emitted = fbank_stream_accept_float(
                &state_, waveform_1d.data_ptr<float>(), samples_to_process, nullptr, 0);
            TORCH_CHECK(emitted == 0, "fbank_stream_accept_float failed");
            return empty_features(state_.extractor.config.num_mel_bins);
        }

        auto features = torch::empty(
            {static_cast<long>(frames_to_extract), static_cast<long>(state_.extractor.config.num_mel_bins)},
            torch::TensorOptions().dtype(torch::kFloat32));
        int32_t emitted = fbank_stream_accept_float(
            &state_,
            waveform_1d.data_ptr<float>(),
            samples_to_process,
            features.data_ptr<float>(),
            frames_to_extract);
        TORCH_CHECK(emitted >= 0, "fbank_stream_accept_float failed");
        if (emitted == frames_to_extract) {
            return features;
        }
        return features.narrow(0, 0, emitted).contiguous();
    }

    torch::Tensor accept_int16(torch::Tensor waveform, int64_t num_samples = -1, int64_t max_frames = -1) {
        auto waveform_1d = ensure_int16_waveform_1d(waveform);
        int32_t samples_to_process = resolve_num_samples(waveform_1d, num_samples);
        int32_t max_possible = fbank_num_frames(&state_.extractor, state_.buffered_samples + samples_to_process);
        int32_t frames_to_extract = resolve_max_frames(max_possible, max_frames);
        if (frames_to_extract <= 0) {
            int32_t emitted = fbank_stream_accept_int16(
                &state_, waveform_1d.data_ptr<int16_t>(), samples_to_process, nullptr, 0);
            TORCH_CHECK(emitted == 0, "fbank_stream_accept_int16 failed");
            return empty_features(state_.extractor.config.num_mel_bins);
        }

        auto features = torch::empty(
            {static_cast<long>(frames_to_extract), static_cast<long>(state_.extractor.config.num_mel_bins)},
            torch::TensorOptions().dtype(torch::kFloat32));
        int32_t emitted = fbank_stream_accept_int16(
            &state_,
            waveform_1d.data_ptr<int16_t>(),
            samples_to_process,
            features.data_ptr<float>(),
            frames_to_extract);
        TORCH_CHECK(emitted >= 0, "fbank_stream_accept_int16 failed");
        if (emitted == frames_to_extract) {
            return features;
        }
        return features.narrow(0, 0, emitted).contiguous();
    }

    int64_t frame_length() const { return static_cast<int64_t>(state_.extractor.frame_length); }
    int64_t frame_shift() const { return static_cast<int64_t>(state_.extractor.frame_shift); }
    int64_t fft_size() const { return static_cast<int64_t>(state_.extractor.config.fft_size); }
    int64_t num_mel_bins() const { return static_cast<int64_t>(state_.extractor.config.num_mel_bins); }

private:
    FbankConfig config_{};
    FbankStreamingState state_{};
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Kaldi-aligned fbank wrapper for hi_xiaowen";

    py::class_<PyFbankExtractor>(m, "FbankExtractor")
        .def(
            py::init<int64_t, double, double, double, double, double, double, double, double, bool, bool, bool, int64_t>(),
            py::arg("num_mel_bins") = 80,
            py::arg("frame_length") = 25.0,
            py::arg("frame_shift") = 10.0,
            py::arg("dither") = 0.0,
            py::arg("energy_floor") = 0.0,
            py::arg("sample_frequency") = 16000.0,
            py::arg("low_freq") = 20.0,
            py::arg("high_freq") = 0.0,
            py::arg("preemphasis_coefficient") = 0.97,
            py::arg("remove_dc_offset") = true,
            py::arg("round_to_power_of_two") = true,
            py::arg("snip_edges") = true,
            py::arg("fft_size") = 0)
        .def("reset", &PyFbankExtractor::reset)
        .def("num_frames", &PyFbankExtractor::num_frames, py::arg("num_samples"))
        .def("process_frame_float", &PyFbankExtractor::process_frame_float, py::arg("waveform"))
        .def("process_frame_int16", &PyFbankExtractor::process_frame_int16, py::arg("waveform"))
        .def(
            "extract_float",
            &PyFbankExtractor::extract_float,
            py::arg("waveform"),
            py::arg("num_samples") = -1,
            py::arg("max_frames") = -1)
        .def(
            "extract_int16",
            &PyFbankExtractor::extract_int16,
            py::arg("waveform"),
            py::arg("num_samples") = -1,
            py::arg("max_frames") = -1)
        .def_property_readonly("frame_length", &PyFbankExtractor::frame_length)
        .def_property_readonly("frame_shift", &PyFbankExtractor::frame_shift)
        .def_property_readonly("fft_size", &PyFbankExtractor::fft_size)
        .def_property_readonly("num_mel_bins", &PyFbankExtractor::num_mel_bins)
        .def_property_readonly("work_buffer_bytes", &PyFbankExtractor::work_buffer_bytes);

    py::class_<PyStreamingFbankExtractor>(m, "StreamingFbankExtractor")
        .def(
            py::init<int64_t, double, double, double, double, double, double, double, double, bool, bool, bool, int64_t>(),
            py::arg("num_mel_bins") = 80,
            py::arg("frame_length") = 25.0,
            py::arg("frame_shift") = 10.0,
            py::arg("dither") = 0.0,
            py::arg("energy_floor") = 0.0,
            py::arg("sample_frequency") = 16000.0,
            py::arg("low_freq") = 20.0,
            py::arg("high_freq") = 0.0,
            py::arg("preemphasis_coefficient") = 0.97,
            py::arg("remove_dc_offset") = true,
            py::arg("round_to_power_of_two") = true,
            py::arg("snip_edges") = true,
            py::arg("fft_size") = 0)
        .def("reset", &PyStreamingFbankExtractor::reset)
        .def_property_readonly("pending_samples", &PyStreamingFbankExtractor::pending_samples)
        .def_property_readonly("frame_length", &PyStreamingFbankExtractor::frame_length)
        .def_property_readonly("frame_shift", &PyStreamingFbankExtractor::frame_shift)
        .def_property_readonly("fft_size", &PyStreamingFbankExtractor::fft_size)
        .def_property_readonly("num_mel_bins", &PyStreamingFbankExtractor::num_mel_bins)
        .def(
            "accept_float",
            &PyStreamingFbankExtractor::accept_float,
            py::arg("waveform"),
            py::arg("num_samples") = -1,
            py::arg("max_frames") = -1)
        .def(
            "accept_int16",
            &PyStreamingFbankExtractor::accept_int16,
            py::arg("waveform"),
            py::arg("num_samples") = -1,
            py::arg("max_frames") = -1);
}
