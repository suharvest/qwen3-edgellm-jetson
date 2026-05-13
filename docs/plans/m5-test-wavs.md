# M5 streaming verification test WAVs

Inventory of WAVs used by `scripts/verify_reproduction_streaming.sh` and by
the M3-step-5 acceptance test (`scripts/test_streaming_worker.py`,
scenario D hard-gate).

All paths below are within this repo. The same files are mounted into
`jetson_voice_slim` on `orin-nx` under
`/opt/qwen3-edgellm-jetson/docs/audio-evidence/`.

## Reproduction prompts (short, streaming E2E)

The three prompts hard-coded in `scripts/verify_reproduction.sh` map to
three pre-rendered TTS outputs in `docs/audio-evidence/`. They were
synthesized from the highperf TTS profile on `orin-nx` on 2026-05-11
(part of the original reproduction loopback evidence — see
`docs/audio-evidence/`).

| ID | Path | Duration | Ground-truth prompt | SHA-256 |
|----|------|---------:|---------------------|---------|
| p1 | `docs/audio-evidence/nx-loopback-pass-p1-2026-05-11.wav` | 2.64 s | 今天天气真好。 | `0014dc311695b9a6e521c6cb36dc82c07ad9f524841454fb353fa7ea42a861b9` |
| p2 | `docs/audio-evidence/nx-loopback-pass-p2-2026-05-11.wav` | 2.64 s | 人工智能改变了世界。 | `3c799b59cf7e59176827a4c277621405defc7500fdb7f30b48f31f68253fea75` |
| p3 | `docs/audio-evidence/nx-loopback-pass-p3-2026-05-11.wav` | 3.20 s | 一二三四五六七八九十。 | `40609737c5c1d290f6bfe8e45ccebdb316a44c61606d286754b62b229bf753eb` |

All three are 24 kHz mono int16. The streaming driver resamples to 16 kHz
in-process (`scripts/test_streaming_worker.py::resample_to_16k`) before
feeding the worker (mel mode) or before base64-encoding raw PCM (PCM
mode).

## Long Chinese WAV (M3 scenario D hard-gate)

For tightening the M3-step-5 scenario D gate from soft (LCS ≥ 0.90,
historically FAIL because `/tmp/spike_input.wav` had a 3.5 s low-energy
tail) to hard (LCS ≥ 0.95), we curated a clean 12.9 s Chinese utterance
from the `seeed-perf` long-form Chinese corpus.

| ID | Path | Duration | Ground-truth transcript | SHA-256 |
|----|------|---------:|------------------------|---------|
| zh_long_04 | `docs/audio-evidence/zh-long-04-2026-05-13.wav` | 12.90 s | 科学家们可以得出结论，暗物质对其他暗物质的影响方式与普通物质相同。 | `a800cf0d9bb6fb951b3e9fe525a5eba9176a00c886f817a255c8b4027a8fb644` |

Format: 16 kHz mono int16 — no resampling needed in the driver.

Sourced from `/tmp/seeed-perf/corpus/long/zh_long_04.wav` on `orin-nx`
(see `/tmp/seeed-perf/corpus/manifest.json` for the original record).
Selected over candidates `zh_long_01..05` because it has the simplest
phonology (no quote marks, no parenthetical English, no numeric
sequences) and falls comfortably within auto-segmentation territory
(2 segments of ~6.5 s each with 1 s carryover).

The shipped engine's `max_input_len` (= 128 mel tokens, ~6.5 s of audio)
forces auto-segmentation on any utterance ≥ ~7 s, so this WAV exercises
exactly the path scenario D was designed to verify.

### Why other candidates were rejected

- `bench/wavs/S3.wav` (9.76 s) and `bench/wavs/S7.wav` (10.16 s):
  no associated ground-truth transcript in the bench harness.
- `qwen3_verify_wavs/long.wav` (8.64 s) / `/tmp/spike_input.wav` :
  5 s of Chinese speech + 3.5 s of acoustic tail. The tail triggers
  language-detector hallucination after carryover trim (the M3 soft-gate
  failure mode documented in `docs/plans/m3-acceptance-results.md`).
- `seeed-perf` `zh_long_02.wav` (13.86 s) and `zh_long_03.wav` (15.42 s):
  contain numeric strings ("15米", "2011年8月", "Oravec,2002") that
  introduce mid-utterance language switches and digit-vs-character
  ambiguity, which depress LCS on this model.
