#include <stdexcept>
#include <string>

#include <torch/extension.h>

#include "fbank.h"

namespace {

torch::Tensor ensure_waveform_1d(torch::Tensor waveform) {
    TORCH_CHECK(waveform.device().is_cpu(), "waveform must be on CPU");
    TORCH_CHECK(waveform.scalar_type() == torch::kFloat32 || waveform.scalar_type() == torch::kFloat64,
                "waveform must be float32/float64");

    if (waveform.dim() == 2) {
        TORCH_CHECK(waveform.size(0) == 1, "only mono waveform is supported, got shape=", waveform.sizes());
        waveform = waveform.squeeze(0);
    }
    TORCH_CHECK(waveform.dim() == 1, "waveform must be 1D or (1, N), got shape=", waveform.sizes());
    return waveform.contiguous().to(torch::kFloat32);
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
    bool snip_edges) {
    FbankConfig config{};
    config.sample_rate = static_cast<int32_t>(sample_frequency);
    config.frame_length_ms = static_cast<int32_t>(frame_length);
    config.frame_shift_ms = static_cast<int32_t>(frame_shift);
    config.num_mel_bins = static_cast<int32_t>(num_mel_bins);
    config.fft_size = 0;
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

}  // namespace

torch::Tensor fbank_bind(
    torch::Tensor waveform,
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
    bool snip_edges = true) {
    (void)energy_floor;
    TORCH_CHECK(dither == 0.0, "current fbank pybind wrapper only supports dither=0.0");

    auto waveform_1d = ensure_waveform_1d(waveform);
    auto config = build_config(
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
        snip_edges);

    FbankExtractor extractor{};
    int32_t init_status = fbank_init(&extractor, &config);
    TORCH_CHECK(init_status == 0, "fbank_init failed");

    int32_t num_frames = fbank_num_frames(&extractor, static_cast<int32_t>(waveform_1d.numel()));
    if (num_frames <= 0) {
        return torch::empty({0, static_cast<long>(num_mel_bins)}, torch::TensorOptions().dtype(torch::kFloat32));
    }

    auto features = torch::empty(
        {static_cast<long>(num_frames), static_cast<long>(num_mel_bins)},
        torch::TensorOptions().dtype(torch::kFloat32));

    int32_t extract_status = fbank_extract_float(
        &extractor,
        waveform_1d.data_ptr<float>(),
        static_cast<int32_t>(waveform_1d.numel()),
        features.data_ptr<float>(),
        num_frames);
    TORCH_CHECK(extract_status >= 0, "fbank_extract_float failed");
    return features;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Kaldi-aligned fbank wrapper for hi_xiaowen";
    m.def(
        "fbank",
        &fbank_bind,
        pybind11::arg("waveform"),
        pybind11::arg("num_mel_bins") = 80,
        pybind11::arg("frame_length") = 25.0,
        pybind11::arg("frame_shift") = 10.0,
        pybind11::arg("dither") = 0.0,
        pybind11::arg("energy_floor") = 0.0,
        pybind11::arg("sample_frequency") = 16000.0,
        pybind11::arg("low_freq") = 20.0,
        pybind11::arg("high_freq") = 0.0,
        pybind11::arg("preemphasis_coefficient") = 0.97,
        pybind11::arg("remove_dc_offset") = true,
        pybind11::arg("round_to_power_of_two") = true,
        pybind11::arg("snip_edges") = true);
}
