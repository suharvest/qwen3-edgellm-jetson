// SPDX-License-Identifier: MIT
//
// MelExtractor — C++ port of `transformers.WhisperFeatureExtractor` for the
// Qwen3-ASR streaming worker.  Loads the upstream `mel_filters` matrix and
// scalar settings dumped by `scripts/extract_whisper_feature_extractor.py`
// and applies the same STFT / mel / log / normalize pipeline that the Python
// feature extractor applies, so the worker can accept raw PCM and produce
// model-input mels that match Python output within ~1e-3 abs diff.
//
// See deploy/audio_preprocessing/README.md for the recipe and bit-exact
// tolerance gate (validated by tests/test_mel_extractor.cpp).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class MelExtractor
{
public:
    //! Settings parsed from whisper_feature_extractor.json.
    struct Settings
    {
        int32_t sampling_rate{16000};
        int32_t n_fft{400};
        int32_t hop_length{160};
        int32_t n_mels{128};
        double mel_floor{1e-10};
        // Pipeline flags — fixed by the upstream HF Whisper recipe, but kept
        // explicit so future tweaks (e.g. a model that disables drop_last) can
        // be honoured without code changes.
        bool stft_center{true};
        bool drop_last_frame{true};
        bool clamp_to_max_minus_8{true};
        bool post_normalize_add4_div4{true};
    };

    //! Construct from upstream-dumped artifacts.
    //!   settings_json_path : whisper_feature_extractor.json
    //!   mel_filters_bin_path : mel_filters.bin (float32 [n_freq, n_mels],
    //!                          row-major, n_freq = n_fft/2 + 1)
    MelExtractor(std::string const& settings_json_path,
                 std::string const& mel_filters_bin_path);
    ~MelExtractor();

    MelExtractor(MelExtractor const&) = delete;
    MelExtractor& operator=(MelExtractor const&) = delete;

    //! Compute log-mel spectrogram for a 16 kHz mono float32 PCM buffer.
    //! Output layout: row-major [n_mels, n_frames] float32. n_frames depends
    //! on `pcm.size()` per the upstream recipe (drop_last_frame=true).
    std::vector<float> compute(std::vector<float> const& pcm,
                               int32_t* out_n_frames = nullptr) const;

    int32_t n_mels() const { return settings_.n_mels; }
    int32_t hop_length() const { return settings_.hop_length; }
    int32_t n_fft() const { return settings_.n_fft; }
    int32_t sampling_rate() const { return settings_.sampling_rate; }
    Settings const& settings() const { return settings_; }

private:
    Settings settings_{};
    // Hann window, fp64, length n_fft (periodic — matches `window_function`).
    std::vector<double> hann_;
    // Mel filterbank, fp32, row-major [n_freq, n_mels]. Same orientation as
    // upstream `mel_filters` so the projection step mirrors `mel_filters.T @ S`.
    std::vector<float> mel_filters_;
    int32_t n_freq_{0};
    // PIMPL holder for KissFFT state (avoids leaking C headers into the API).
    struct FftState;
    std::unique_ptr<FftState> fft_;
};
