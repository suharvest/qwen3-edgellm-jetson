# Qwen3-ASR Thinker Engine — Build Recipe (Production)

This is the **canonical** recipe for building the Qwen3-ASR thinker TensorRT
engine on Jetson Orin (NX / Nano, SM 87). It supersedes any earlier
`build_qwen3_asr_thinker_mlp_int8.py` experiment, which performed an unrelated
MLP-INT8 quantization and does **not** produce a runtime-compatible engine.

> Source authority: `tensorrt-edge-llm/docs/source/user_guide/features/fp8-embedding.md`.

## TL;DR — Two-Step Pipeline

```
1) tensorrt-edgellm-export-llm --fp8_embedding
     → produces:  model.onnx + onnx_model.data + embedding.safetensors
                  (embedding tensor is torch.float8_e4m3fn, with embedding_scale)

2) ./build/examples/llm/llm_build \
     --onnxDir   <dir from step 1> \
     --engineDir <out>             \
     --maxBatchSize 1              \
     --maxInputLen 256             \
     --maxKVCacheCapacity 512
   (FP8 embedding is auto-detected from safetensors metadata;
    the thinker decoder itself stays FP16 — embedding-only FP8.)
```

If you already have an ONNX export with an **FP16** `embedding.safetensors`,
you can quantize it in-place instead of re-exporting:

```
uv run python scripts/quantize_embedding_safetensors_fp8.py \
  thinker/embedding.safetensors            \
  thinker/embedding_fp8.safetensors
mv thinker/embedding_fp8.safetensors thinker/embedding.safetensors
```

The script writes both `embedding` (FP8 E4M3, vocab×hidden) and
`embedding_scale` (FP32, vocab × hidden/128) — exactly what the EdgeLLM
runtime expects.

## Why `--fp8_embedding`?

Qwen3-ASR thinker has a 151,936 × 1024 token-embedding table. FP16 → 297 MB,
FP8 → 153 MB. That's a 144 MB shrink on a model whose total ONNX weights are
~1.2 GB. On a memory-constrained device (Orin NX 16 GB shared, half of which
goes to runtime activations/KV) every MB matters.

The thinker itself is **not** quantized — only the embedding lookup table is
FP8. Decoder layers stay FP16 to preserve generation quality.

## What `llm_build` auto-detects

`llm_build` reads `<onnxDir>/embedding.safetensors` and decides:

| safetensors keys                      | embedding dtype       | runtime behavior |
|---------------------------------------|-----------------------|------------------|
| `embedding`                            | float16 / float32     | FP16 embedding lookup (fallback) |
| `embedding` + `embedding_scale`        | torch.float8_e4m3fn   | **FP8 embedding lookup (target)** |

If `embedding_scale` is missing, the runtime cannot dequantize the FP8
table — make sure both keys exist after step 1.

## `max_input_len` / `max_kv_cache_capacity` choice

These are **build-time hard caps** on the prefill input length and the KV
cache sequence length. Choose them by P95 expected utterance length:

| `max_input_len` | utterance budget (≈13 audio tok/s) | engine + activations |
|-----------------|------------------------------------|----------------------|
| 128 (legacy)    | ~5.5 s                              | smallest |
| **256 (current)** | **~15 s**                          | **+~50 MB KV** |
| 512             | ~33 s                               | +100 MB KV |
| 1024            | ~75 s                               | +200 MB KV |

For the production worker we use **256 / 512** (input / kv), which covers the
M5 scenario D 12.9 s utterance plus prompt overhead with margin.

## Memory Requirements

- **Build-time (Orin NX, max=256)**: TensorRT autotuning peaks around 6-8 GB
  RAM working set. On a fully loaded NX (production runtime resident) the
  build *will* swap heavily; expect 30-60+ minutes wall clock.
- **Runtime (resident)**: ~750 MB thinker engine + ~50 MB KV scratch.

## Common Pitfalls

1. **Do not use `build_qwen3_asr_thinker_mlp_int8.py`.** That script's
   docstring explicitly says it is "a feasibility builder, not production
   recipe". It runs a per-MLP INT8 calibration pass that is incompatible
   with the EdgeLLM `AttentionPlugin` format negotiation (manifests as
   plugin "Error 9" at runtime).

2. **`llm_build` binary must exist.** It's only present after a successful
   `cmake --build` of `TensorRT-Edge-LLM` on the target. Look under
   `<repo>/build/examples/llm/llm_build`.

3. **Plugin ABI must match.** The build-time `libNvInfer_edgellm_plugin.so`
   loaded at engine-build time must be byte-compatible with the one loaded
   by the production worker at runtime. We rely on the same source-built
   plugin (`<repo>/build/libNvInfer_edgellm_plugin.so`) for both.

4. **Memory pressure → glacial build.** If the host has < 4 GB free RAM at
   build time, TRT autotuning will thrash on swap and may stall for hours.
   Stop the production stack (or temporarily lower `concurrency`) before
   building.

5. **DLA fallback warning is benign.** TRT prints
   `DLA requests all profiles have same min, max, and opt value. All dla
   layers are falling back to GPU` — this is expected; we don't use DLA.

## Cross-Device Engine Portability

NX and Nano are both SM 87 (Ampere, 8.7). Engines built on either device
should load on the other **provided** that:

- TensorRT minor versions match (we currently use 10.3 on both),
- the EdgeLLM plugin shared object is built from the same commit,
- `max_batch_size`, `max_input_len`, `max_kv_cache_capacity` cover the
  caller's runtime shapes.

For reproducibility we still rebuild per-device into the corresponding
`engines/orin-{nx,nano}/highperf-v2/` directory.

## Worker-Side Constants to Keep In Sync

When you change `--maxInputLen`, also update
`native/edgellm_voice_worker/qwen3_asr_worker.cpp`:

```cpp
constexpr int32_t kEngineMaxInputLen = 256;        // must match build flag
constexpr double  kSingleChunkHardLimitSec = 15.0; // audio cap derived from above
```

The relation is:
`hard_limit_sec ≈ (max_input_len − safety_margin − prompt_overhead − bos_eos) / audio_tokens_per_sec`.

For max=256, that yields ≈16.4 s; we set 15.0 s for headroom.

## Verifying an Engine After Build

```bash
ls -la <engineDir>/                       # llm.engine + config.json
jq '{max_input_len, max_kv_cache_capacity, max_batch_size}' <engineDir>/config.json
md5sum <engineDir>/llm.engine
```

Then drive a one-shot ASR request against the new engine path with the
existing `test_streaming_worker.py` style harness. A sane Chinese
transcription is the smoke-test pass criterion.

---

History note: see commit messages on `streaming-asr/m2-worker-capacity` for
the build event log and M5 LCS measurements that motivated the
`max_input_len 128 → 256` upgrade.
