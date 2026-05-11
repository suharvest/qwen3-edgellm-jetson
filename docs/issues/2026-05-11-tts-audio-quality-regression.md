# Qwen3 highperf TTS audio quality regression (2026-05-11)

**Severity**: high — TTS pipeline runs structurally to completion but the
emitted audio is **gibberish** (ASR maps it to one repeated character).
Every "smoke success" in the 2026-05-11 reproduction reports passed
structural checks (RTF, audio_complete, chunk_count) but **none were
auditioned**. Once an ASR round-trip was added the regression surfaced.

This document is the hand-off package for the dev agent. All Pipeline /
loopback infrastructure is already in place (slim docker image runs both
ASR and TTS on `orin-nx`), so the dev agent can iterate against a live
server.

## Symptom

| Source audio | ASR transcription |
|---|---|
| `bench/wavs/S1.wav` (real Chinese speech, 2.8s) | `今天天气真好，我们出去玩吧。` (correct) |
| `/tts` on prompt `今天我们继续验证低延迟流式生成的效果。` (4.32s, 24kHz mono) | `嗯嗯嗯嗯嗯嗯嗯嗯嗨嗯嗯嗯嗯。` |
| `/tts/stream` on same prompt (4.8s) — saved in `docs/audio-evidence/nx-highperf-2026-05-11.wav` | `是是是是是…(150个)。` |
| `/tts/clone/stream` with real speaker embedding (4.72s) — `docs/audio-evidence/nx-voice-clone-2026-05-11.wav` | (not yet ASR-tested, expected similar) |

The WAVs have the correct duration, sample rate, channel count, and
header. The PCM body contains real audio energy (not silence) but is
linguistically incoherent — it sounds like one monosyllable repeated.

## What works

1. **Engine deserialization** — `trtexec --loadEngine` PASSES for all 5
   primary engines (asr thinker, asr audio_encoder, talker_w8a16_outputk,
   code_predictor cp, code2wav_stateful) against host TensorRT 10.3.0.30.
2. **Worker handshake** — `qwen3_tts_worker` boots clean, emits
   `{"event":"ready","init_ms":12200}`. CUDA graphs capture, stateful
   Code2Wav allocates its buffers. No initialization error.
3. **Streaming protocol** — 7 chunks (`first=7, chunk=10, max=10`),
   `audio_complete=true`, `is_final=true` on last chunk. RTF 0.749 –
   0.76. Perf matches expectations.
4. **ASR end-to-end** — `qwen3_asr_worker` against the published
   `asr_thinker_full_fp8embed/llm.engine` + `asr_audio_encoder/...`
   correctly transcribes real human speech (S1.wav → expected text).
5. **Loopback wiring** — slim docker image
   (`jetson-voice-qwen3:slim`, 991 MB) exposes both `/asr` and `/tts*`,
   /health goes 200 in ~38 s, all routes accept the documented payloads.
6. **Voice clone protocol** — `speaker_embedding_b64` is consumed by
   the worker without crashing; structural smoke metrics are normal.

## What's broken

The audio coming out of the TTS pipeline is **content-wrong**. Either:

- the Talker is producing the same codec token over and over, or
- the CodePredictor is collapsing its 15 RVQ heads to one fixed code, or
- the Code2Wav engine is decoding fine but the upstream codes are bad.

ASR confirms the audio is roughly Chinese-sounding monosyllable repetition,
which is the smell of "talker generating one token / collapsed sampling /
broken embedding lookup".

## Suspects (ordered by likelihood)

### 1. FP8 text-embedding scale layout vs kernel expectation

The fork commit `c248f73` (`fix(qwen3-tts): pass FP8 dequant scales to
text embedding lookup`) added name-based lookup of
`text_embedding_scale` in `Qwen3OmniTTSRuntime::loadTalkerWeights` and
forwarded the scale tensor to `embeddingLookup` at two call sites
(`initializeTTSEmbeddings`, `prepareTalkerInput`).

The kernel at `cpp/kernels/embeddingKernels/embeddingKernels.cu:491+`
asserts the scale tensor shape `[vocabSize, hiddenSize / blockSize]`
and dtype FP32 — both pass at runtime (no assert). But "passes shape
check" ≠ "applies correctly". Possible failure modes:

- Scale is `[151936, 16]` in row-major but kernel expects `[16, 151936]`
  / strided differently → quietly dequantizes with wrong scale.
- Scale was exported per-row-per-block but kernel assumes per-row-only,
  or vice versa.
- The actual quantize script
  (`qwen3-edgellm-jetson/scripts/quantize_embedding_safetensors_fp8.py`)
  uses `BLOCK_SIZE = 128` along the hidden dim with FP8_E4M3_MAX = 448.
  Kernel expects the same blocking → matches on paper but easy to drift.

**Test idea**: feed the talker with a non-quantized FP16 text embedding
(re-export `text_embedding.safetensors` without quantization) and see
if audio coherency comes back. If yes, FP8 dequant is the bug.

### 2. Engine plan vs runtime plugin drift

The published `orin-nx-highperf-2026-05-11` engines were baked by a
specific git revision of the EdgeLLM fork. The `qwen3-tts-highperf-runtime-w8a16`
runtime tip is now at `5cc6060` (FP8 fix + SM87 + CuTe DSL gemm +
EMBEDDED_TARGET defaults). If the engines were baked against an earlier
revision with a slightly different plugin op signature, the runtime
could load them, run them, and produce arithmetically wrong outputs
because of subtle layout/scale mismatches in custom-op weights.

`trtexec` only validates structural loading, not numerics. The warning
`Using an engine plan file across different models of devices is not
recommended and is likely to affect performance or even cause errors`
shows up consistently — worth taking seriously, not as a generic notice.

**Test idea**: cook a fresh talker engine from the same ONNX export on
the current runtime branch, compare with the HF engine bit-for-bit, and
re-smoke with the freshly baked one.

### 3. CuTe DSL gemm wrong arithmetic

`talkerMLPKernels.cu` has no fallback path — if `ENABLE_CUTE_DSL=gemm`
isn't set the function `LOG_ERROR`s and returns without writing the
output buffer (so MLP output is whatever was in the buffer, effectively
zero). Recent commits make `gemm` the fork default and verified
`runBiasSiLU` symbol is present in `libNvInfer_edgellm_plugin.so`. But
the prebuilt `cutedsl_aarch64_sm_87_cuda12.tar.gz` artifact may have
been generated against a different SM/CUDA tuple than this NX (CUDA
12.6, sm_87). The kernel signatures match but the cubin internals could
be wrong.

**Test idea**: run a unit micro-bench of `CuteDslGemmRunner::runBiasSiLU`
with known inputs/outputs and compare vs a reference (e.g., CPU/eigen).

### 4. Sampling collapse (least likely but cheap to check)

If `talker_temperature` / `talker_top_k` / `talker_top_p` somehow
default to extreme values (e.g., temp=0 with broken tie-breaking), the
talker would sample the same token every step. Worker defaults in
`trt_edge_llm_tts.py`:

- `_DEFAULT_TEMPERATURE = 0.9`
- `_DEFAULT_TOP_K = 50`
- `_DEFAULT_TOP_P = 1.0`
- `_DEFAULT_REPETITION_PENALTY = 1.05`

Reasonable on paper. But worth dumping the worker's request JSON at
runtime to confirm what actually reaches the C++ side.

**Test idea**: add `--debug` to the worker invocation and inspect logged
token sequence — if every step picks the same token id, sampling is
collapsed. If token ids are varied but audio is still bad, problem is
downstream (Code2Wav or codes-to-audio path).

## What's been ruled out

- ❌ engine plan version mismatch (TRT 10.3 vs 10.4) — false alarm, see
  the CORRECTION block in `qwen3-orin-nx-clean-room-2026-05-11.md`.
- ❌ FMHA SM87 missing — fixed by `eacfefc` (default arch list now
  includes 87), worker no longer fails MHA dispatch.
- ❌ CuTe DSL gemm not compiled — verified compiled, `runBiasSiLU`
  symbol present in plugin, no runtime LOG_ERROR.
- ❌ artifact path mismatch — `/opt/models/qwen3-edgellm` resolves
  correctly via `${QWEN3_ARTIFACT_ROOT}` substitution.
- ❌ ASR worker — proven correct on real speech.

## Reproduce

The full loopback runs on `orin-nx` (Tailscale `100.82.225.102`,
HTTP port `18092` exposed by `jetson_voice_slim` container).

```bash
NX=http://100.82.225.102:18092

# Sanity check the ASR is correct on a real WAV (positive control):
curl -s -X POST $NX/asr -F file=@bench/wavs/S1.wav | jq .
# {"text":"今天天气真好，我们出去玩吧。", ...}

# Reproduce the TTS quality bug (negative control):
curl -s -X POST $NX/tts -H 'content-type: application/json' \
  -d '{"text":"今天我们继续验证低延迟流式生成的效果。"}' \
  -o tts.wav
curl -s -X POST $NX/asr -F file=@tts.wav | jq .
# {"text":"嗯嗯嗯嗯…", ...}  (or similar single-char repetition)
```

The slim container is started via `/tmp/run_slim_v2.sh` on `orin-nx`;
the EdgeLLM build dir, the qwen3-edgellm-jetson clone, and the artifact
root are bind-mounted from host paths so a code-only iteration loop is
short (~30 s rebuild + restart).

Worker direct probe (skip Python layer):

```bash
$WORKER --talkerEngineDir=$ROOT/tts/talker \
        --qwen3TtsTalkerBackend=qwen3_tts_explicit_kv \
        --qwen3TtsTalkerEngine=$ROOT/engines/orin-nx/highperf/talker_w8a16_outputk/talker_decode_w8a16_outputk.engine \
        --codePredictorEngineDir=$ROOT/engines/orin-nx/highperf/code_predictor/cp_dir \
        --codePredictorBackend=qwen3_tts_native \
        --code2wavEngineDir=$ROOT/engines/orin-nx/highperf/code2wav_stateful \
        --tokenizerDir=$ROOT/tts/tokenizer \
        --debug \
  <<< '{"id":"t1","text":"测试","stream":true,"stream_only":true,
        "first_chunk_frames":7,"chunk_frames":10,"max_chunk_frames":10}'
```

`--debug` should dump per-step codec tokens — confirm whether the
talker is sampling a varied or constant sequence.

## Artifacts on disk

| Path (on `orin-nx`) | Meaning |
|---|---|
| `~/qwen3-models/` | `orin-nx-highperf-2026-05-11` artifacts (HF) |
| `~/project/repro-qwen3/TensorRT-Edge-LLM/build/examples/omni/qwen3_tts_worker` | md5 `dfe80e62…` |
| `~/project/repro-qwen3/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so` | md5 `079f27b1…` |
| `~/project/repro-qwen3/jetson-voice/build/edgellm_voice_worker/workers/qwen3_asr_worker` | md5 `e2b897e4…` |
| `~/project/qwen3-edgellm-jetson/docs/audio-evidence/nx-highperf-2026-05-11.wav` | the §1 smoke output that ASR maps to `是是是…` |
| `~/project/qwen3-edgellm-jetson/docs/audio-evidence/voice-clone-reference-S1-2026-05-11.wav` | S1.wav reference (real speech) |
| `~/project/qwen3-edgellm-jetson/docs/audio-evidence/nx-voice-clone-2026-05-11.wav` | §5 clone output (not yet ASR-checked) |

## Commits relevant to this regression

EdgeLLM fork `suharvest/TensorRT-Edge-LLM` branch
`qwen3-tts-highperf-runtime-w8a16`:

- `c248f73` — FP8 dequant scale fix (primary suspect)
- `eacfefc` — SM87 in default arch list
- `190d977` — default ENABLE_CUTE_DSL=gemm
- `de3939c` — cache vs normal var fix for ENABLE_CUTE_DSL
- `5cc6060` — default EMBEDDED_TARGET + CUTE_DSL_ARTIFACT_TAG

If the dev agent wants to **revert just the FP8 scale change** to test
suspect #1, `c248f73` is the single diff to drop. Without it the worker
will crash with `scales must be provided for FP8 embedding table` — but
that crash itself becomes useful evidence (proves the FP8 path is the
one being exercised at the failure point).

## Open questions for the dev agent

1. Were the `orin-nx-highperf-2026-05-11` engines cooked against the FP8
   text-embedding format, and if so, was the scale tensor laid out
   `[vocab, hidden/128]` row-major matching what
   `quantize_embedding_safetensors_fp8.py` writes?
2. Should `embeddingLookup` apply the scale **before** the gather, or
   **after**? Current kernel path (`embeddingKernels.cu:491+`) does it
   after the FP8→FP16 conversion. Is that the layout the talker
   prefill is expecting?
3. Is there a known-good runtime commit on this fork (predating the
   2026-05-11 cleanup) that previously produced coherent audio against
   the same HF artifact set? If so, what changed between that and now?
4. Should we re-export the text embedding from
   `Qwen3-TTS/text_embedding.safetensors` raw (no quantization) and
   verify the talker path still works in FP16? That'd isolate the bug.
