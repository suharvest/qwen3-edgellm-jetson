# Audio preprocessing assets — WhisperFeatureExtractor mirror

These two files reproduce the exact mel-filterbank and STFT settings used by
the upstream `transformers.WhisperFeatureExtractor` configured for
`Qwen/Qwen3-ASR-0.6B`. The C++ MelExtractor in the streaming worker loads
both at startup and applies the same operations the Python feature extractor
applies — see M4 in `docs/plans/qwen3-asr-streaming-design-2026-05-13.md`.

| file | description |
| ---- | ----------- |
| `whisper_feature_extractor.json` | All scalar settings (n_fft, hop_length, n_mels, mel_floor, window, padding/log/normalize policy, source model id). Authoritative. |
| `mel_filters.bin`                | Raw `float32` mel filter bank, shape `[n_freq, n_mels] = [201, 128]`, row-major. Matches `WhisperFeatureExtractor.mel_filters` byte-for-byte (after `astype(float32)`). |
| `mel_filters.sha256`             | SHA-256 of `mel_filters.bin`. Cross-check on deploy. |

## How to regenerate

```bash
# On a host with the qwen-asr Python venv (e.g. wsl2-local):
source /home/harve/qwen3-asr-vllm-env/bin/activate
python scripts/extract_whisper_feature_extractor.py \
    --model Qwen/Qwen3-ASR-0.6B \
    --out deploy/audio_preprocessing/
```

If `--model` is changed (e.g. to a forked ASR model with retrained mel
settings), update this README to note the new source. **The shipped C++ worker
assumes the JSON values above are exactly the values used at training time.**
Do NOT regenerate the filterbank from a formula and hope it matches — the
upstream HF implementation uses `librosa`-style slaney-normalised filters and
fp64 internals; bit-exact reproduction in a different code path is fragile.

## Why ship binary weights instead of code?

We tried hand-rolling mel filterbanks (see `scripts/test_streaming_worker.py`
`build_mel_filterbank()` for an example). The hand-rolled filters drift from
upstream by ~1e-3 in places due to subtle differences in slaney vs HTK
normalization and rounding. That drift compounds across ~5 s of audio and
visibly degrades ASR LCS-similarity. Loading the actual filter bank as static
data eliminates that risk entirely.

## STFT recipe (mirrors `transformers.audio_utils.spectrogram` with Whisper params)

For 16 kHz mono float32 PCM:

1. **Center-pad** input with `n_fft // 2 = 200` samples reflected on both
   sides (`np.pad(..., mode="reflect")`).
2. Promote PCM to `float64`. Build a periodic Hann window of length `n_fft`:
   `0.5 * (1 - cos(2π * k / n_fft))` for `k=0..n_fft-1`.
3. Frame: `num_frames = 1 + floor((len(padded) - n_fft) / hop_length)`. For
   frame `t`, take samples `[t*hop_length : t*hop_length + n_fft]`, multiply
   by the window, then run `rfft` of length `n_fft`. Yields `n_fft//2 + 1`
   complex bins.
4. Power spectrum: `|X|^2` (float64). Transpose to `(n_freq, n_frames)`.
5. **Mel projection**: `power_mel = max(mel_floor, mel_filters.T @ power)`.
   Note: `mel_filters` is `[n_freq, n_mels]`, so `mel_filters.T` is
   `[n_mels, n_freq]`.
6. `log_mel = log10(power_mel)`.
7. Drop the **last frame**: `log_mel = log_mel[:, :-1]`. (Whisper artefact —
   the rfft has one extra frame at the right edge that the model never sees.)
8. Clamp dynamic range: `log_mel = max(log_mel, log_mel.max() - 8.0)`.
9. Final normalize: `log_mel = (log_mel + 4.0) / 4.0`.
10. Cast to `float32` for downstream consumption.

The C++ port (`native/edgellm_voice_worker/mel_extractor.cpp`) implements
exactly this pipeline. The golden-mel test
(`native/edgellm_voice_worker/tests/test_mel_extractor.cpp`) gates against
≤ 1e-3 max abs diff on a corpus of synthetic + real-speech inputs.
