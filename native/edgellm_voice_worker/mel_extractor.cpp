// SPDX-License-Identifier: MIT
#include "mel_extractor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

// KissFFT is compiled with `-Dkiss_fft_scalar=double` so the FFT internals
// use fp64 — matching the Python pipeline which promotes to float64 before
// calling np.fft.rfft.
extern "C" {
#include "kissfft/kiss_fftr.h"
}

namespace {
constexpr double kPi = 3.14159265358979323846;
} // anonymous namespace

struct MelExtractor::FftState
{
    kiss_fftr_cfg cfg{nullptr};
    ~FftState()
    {
        if (cfg)
        {
            kiss_fftr_free(cfg);
            cfg = nullptr;
        }
    }
};

static void readBinaryFile(std::string const& path, std::vector<char>& out)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
    {
        throw std::runtime_error("MelExtractor: cannot open " + path);
    }
    std::streamsize const n = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(n));
    if (n > 0 && !f.read(out.data(), n))
    {
        throw std::runtime_error("MelExtractor: short read from " + path);
    }
}

MelExtractor::MelExtractor(std::string const& settings_json_path,
                           std::string const& mel_filters_bin_path)
{
    // --- Load settings ---
    std::ifstream js(settings_json_path);
    if (!js)
    {
        throw std::runtime_error("MelExtractor: cannot open " + settings_json_path);
    }
    nlohmann::json doc;
    js >> doc;
    settings_.sampling_rate = doc.value("sampling_rate", 16000);
    settings_.n_fft = doc.value("n_fft", 400);
    settings_.hop_length = doc.value("hop_length", 160);
    settings_.n_mels = doc.value("n_mels", 128);
    settings_.mel_floor = doc.value("mel_floor", 1e-10);
    settings_.stft_center = doc.value("stft_center", true);
    settings_.drop_last_frame = doc.value("drop_last_frame", true);
    settings_.clamp_to_max_minus_8 = doc.value("clamp_to_max_minus_8", true);
    // post_normalize policy fixed to (x+4)/4 in upstream; treat as bool flag.
    settings_.post_normalize_add4_div4 = true;
    n_freq_ = settings_.n_fft / 2 + 1;

    // Sanity: settings.mel_filters.shape should be [n_freq, n_mels].
    if (doc.contains("mel_filters") && doc["mel_filters"].contains("shape"))
    {
        auto const shape = doc["mel_filters"]["shape"];
        if (shape.is_array() && shape.size() == 2)
        {
            int const sf0 = shape[0].get<int>();
            int const sf1 = shape[1].get<int>();
            if (sf0 != n_freq_ || sf1 != settings_.n_mels)
            {
                char msg[160];
                std::snprintf(msg, sizeof(msg),
                    "MelExtractor: mel_filters shape [%d,%d] != expected [%d,%d]",
                    sf0, sf1, n_freq_, settings_.n_mels);
                throw std::runtime_error(msg);
            }
        }
    }

    // --- Load mel_filters.bin (float32 [n_freq, n_mels]) ---
    std::vector<char> raw;
    readBinaryFile(mel_filters_bin_path, raw);
    size_t const expected = static_cast<size_t>(n_freq_) * static_cast<size_t>(settings_.n_mels) * sizeof(float);
    if (raw.size() != expected)
    {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "MelExtractor: mel_filters.bin size %zu != expected %zu (n_freq=%d n_mels=%d)",
            raw.size(), expected, n_freq_, settings_.n_mels);
        throw std::runtime_error(msg);
    }
    mel_filters_.resize(static_cast<size_t>(n_freq_) * settings_.n_mels);
    std::memcpy(mel_filters_.data(), raw.data(), expected);

    // --- Precompute periodic Hann window (matches `window_function("hann")`
    // from transformers.audio_utils): w[k] = 0.5 * (1 - cos(2π k / N)),
    // k = 0..N-1.  Note: NOT the symmetric Hann (which would use N-1 in the
    // denominator).
    hann_.resize(settings_.n_fft);
    for (int32_t k = 0; k < settings_.n_fft; ++k)
    {
        hann_[k] = 0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(k) / static_cast<double>(settings_.n_fft)));
    }

    // --- KissFFT real-valued forward plan, n_fft samples → n_fft/2+1 bins. ---
    fft_ = std::make_unique<FftState>();
    fft_->cfg = kiss_fftr_alloc(settings_.n_fft, 0, nullptr, nullptr);
    if (!fft_->cfg)
    {
        throw std::runtime_error("MelExtractor: kiss_fftr_alloc failed");
    }
}

MelExtractor::~MelExtractor() = default;

std::vector<float> MelExtractor::compute(std::vector<float> const& pcm_in,
                                         int32_t* out_n_frames) const
{
    int32_t const n_fft = settings_.n_fft;
    int32_t const hop = settings_.hop_length;
    int32_t const n_mels = settings_.n_mels;
    int32_t const n_freq = n_freq_;
    int32_t const pad = n_fft / 2;

    // 1) Center-pad with reflect (mirrors np.pad(..., mode="reflect")).
    //    reflect mode does NOT repeat the edge sample, e.g.
    //      np.pad([1,2,3,4,5], (2,2), "reflect") -> [3,2,1,2,3,4,5,4,3]
    //    We need at least 2 samples for reflect to be well-defined; mirror
    //    the upstream lib behavior: silently allow shorter inputs (the
    //    feature extractor does too — they go through right-zero padding to
    //    `max_length` first, but our worker-streaming path uses padding=False
    //    so reflect with insufficient samples is undefined. Pad the PCM up
    //    to n_fft with zeros first when necessary.
    std::vector<double> waveform;
    int32_t const n_in = static_cast<int32_t>(pcm_in.size());
    if (n_in < 2)
    {
        // Degenerate input — produce zero frames.
        if (out_n_frames) *out_n_frames = 0;
        return {};
    }

    if (settings_.stft_center)
    {
        int32_t const total = n_in + 2 * pad;
        waveform.resize(total);
        // Left pad: indices i in [0, pad) → waveform[i] = pcm[pad - i]
        for (int32_t i = 0; i < pad; ++i)
        {
            int32_t const src = pad - i;
            int32_t const clamped = std::min(src, n_in - 1);
            waveform[i] = static_cast<double>(pcm_in[clamped]);
        }
        // Middle.
        for (int32_t i = 0; i < n_in; ++i)
        {
            waveform[pad + i] = static_cast<double>(pcm_in[i]);
        }
        // Right pad: indices i in [0, pad) → waveform[pad + n_in + i] = pcm[n_in - 2 - i]
        for (int32_t i = 0; i < pad; ++i)
        {
            int32_t const src = n_in - 2 - i;
            int32_t const clamped = std::max(0, src);
            waveform[pad + n_in + i] = static_cast<double>(pcm_in[clamped]);
        }
    }
    else
    {
        waveform.resize(n_in);
        for (int32_t i = 0; i < n_in; ++i)
        {
            waveform[i] = static_cast<double>(pcm_in[i]);
        }
    }

    // 2) Frame.  num_frames = 1 + floor((waveform.size - n_fft) / hop).
    int32_t const total_len = static_cast<int32_t>(waveform.size());
    if (total_len < n_fft)
    {
        if (out_n_frames) *out_n_frames = 0;
        return {};
    }
    int32_t const num_frames_full = 1 + (total_len - n_fft) / hop;
    int32_t const num_frames_kept = settings_.drop_last_frame
                                        ? std::max(0, num_frames_full - 1)
                                        : num_frames_full;
    if (num_frames_kept <= 0)
    {
        if (out_n_frames) *out_n_frames = 0;
        return {};
    }

    // 3) STFT per frame → power spectrum [n_freq] (fp64).
    //    Accumulate into power_spec[n_freq * num_frames_full] for the full
    //    set, then drop the last frame as a slicing step.
    std::vector<double> power_spec(static_cast<size_t>(n_freq) * num_frames_full);
    std::vector<double> buffer(n_fft);
    std::vector<kiss_fft_cpx> spectrum(n_freq);

    for (int32_t frame = 0; frame < num_frames_full; ++frame)
    {
        double const* src = waveform.data() + static_cast<size_t>(frame) * hop;
        for (int32_t k = 0; k < n_fft; ++k)
        {
            buffer[k] = src[k] * hann_[k];
        }
        kiss_fftr(fft_->cfg, reinterpret_cast<kiss_fft_scalar const*>(buffer.data()), spectrum.data());
        double* dst = power_spec.data() + static_cast<size_t>(frame) * n_freq;
        for (int32_t k = 0; k < n_freq; ++k)
        {
            double const re = spectrum[k].r;
            double const im = spectrum[k].i;
            dst[k] = re * re + im * im;
        }
    }

    // 4) Mel projection (fp64).  power_mel[m, t] = sum_k mel_filters[k, m] *
    //    power_spec[k, t], with mel_floor lower clamp.  We compute over the
    //    full num_frames_full grid and discard the last frame at the slicing
    //    step (drop_last_frame=true).
    int32_t const num_frames_proj = num_frames_full; // project all; slice later
    std::vector<double> mel_spec(static_cast<size_t>(n_mels) * num_frames_proj);
    for (int32_t t = 0; t < num_frames_proj; ++t)
    {
        double const* power_t = power_spec.data() + static_cast<size_t>(t) * n_freq;
        for (int32_t m = 0; m < n_mels; ++m)
        {
            double acc = 0.0;
            // mel_filters is [n_freq, n_mels] row-major: weight for (k, m)
            // is at index k * n_mels + m.
            for (int32_t k = 0; k < n_freq; ++k)
            {
                acc += static_cast<double>(mel_filters_[static_cast<size_t>(k) * n_mels + m]) * power_t[k];
            }
            mel_spec[static_cast<size_t>(m) * num_frames_proj + t]
                = std::max(settings_.mel_floor, acc);
        }
    }

    // 5) log10, then drop last frame (slicing).
    std::vector<float> log_mel(static_cast<size_t>(n_mels) * num_frames_kept);
    for (int32_t m = 0; m < n_mels; ++m)
    {
        for (int32_t t = 0; t < num_frames_kept; ++t)
        {
            double const v = mel_spec[static_cast<size_t>(m) * num_frames_proj + t];
            log_mel[static_cast<size_t>(m) * num_frames_kept + t] = static_cast<float>(std::log10(v));
        }
    }

    // 6) clamp(max - 8) then (x + 4) / 4.
    if (settings_.clamp_to_max_minus_8)
    {
        float max_val = log_mel[0];
        for (float v : log_mel) max_val = std::max(max_val, v);
        float const floor_val = max_val - 8.0f;
        for (float& v : log_mel) v = std::max(v, floor_val);
    }
    if (settings_.post_normalize_add4_div4)
    {
        for (float& v : log_mel) v = (v + 4.0f) / 4.0f;
    }

    if (out_n_frames) *out_n_frames = num_frames_kept;
    return log_mel;
}
