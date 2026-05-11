# TensorRT-Edge-LLM upstream PR split for Qwen3-TTS

Date: 2026-05-05

Local EdgeLLM worktree:

```text
.worktrees/tensorrt-edge-llm-official-qwen3-tts
```

Fork branch:

```text
https://github.com/suharvest/TensorRT-Edge-LLM/tree/official-qwen3-tts-upstream-runtime
```

Base:

```text
NVIDIA/TensorRT-Edge-LLM origin/main at bbbab9a
```

## Current commit stack

```text
8bee99e Fix Qwen3-TTS LLM export checkpoint loading
891e84b Harden ONNX parsing and safetensors shard loading
fca4359 Respect explicit CUDA architectures
3f880f8 Fix CuTe DSL CUDA 12.6 static linking
73afadd Align Qwen3-TTS runtime semantics
8a0df08 Add native Qwen3-TTS CodePredictor engine path
631e7f8 Address Qwen3-TTS upstream review findings
```

Review fixes applied after the second-opinion pass:

```text
631e7f8 Address Qwen3-TTS upstream review findings
```

Before opening upstream PRs, fold these fixes back into the appropriate PR commits:

- PR 2: parser lifetime fix in `cpp/builder/builderUtils.cpp`
- PR 3: all Qwen3-TTS runtime/builder fixes in `cpp/builder/llmBuilder.cpp` and `cpp/runtime/qwen3OmniTTSRuntime.{h,cpp}`

## Recommended upstream PRs

### PR 1: Jetson build compatibility

Commits:

```text
fca4359 Respect explicit CUDA architectures
3f880f8 Fix CuTe DSL CUDA 12.6 static linking
```

Purpose:

- Preserve explicit `CMAKE_CUDA_ARCHITECTURES`, including Jetson Orin `87`.
- Fix CuTe DSL static-library propagation for CUDA 12.6/JetPack 6.2 builds.
- Add CUDA 12.6 compatibility around `cudaLibrary_t`.

Why separate:

- This is generic build infrastructure work.
- It is independent of Qwen3-TTS runtime semantics.
- It is likely the easiest upstream review.

Validation:

- Orin native clean build passed for `qwen3_tts_inference`.
- Orin native clean build passed for `NvInfer_edgellm_plugin`.
- Rechecked after review fixes: no code changes in PR 1 files from this pass.

### PR 2: Export and builder robustness

Commits:

```text
8bee99e Fix Qwen3-TTS LLM export checkpoint loading
891e84b Harden ONNX parsing and safetensors shard loading
```

Purpose:

- Load Qwen3-TTS checkpoint weights through the HF module graph used for export.
- Avoid leaving nested CodePredictor weights randomly initialized.
- Support sharded `model.safetensors.index.json` checkpoints.
- Harden ONNX fallback parsing so fallback uses a fresh network.
- Validate builder workspace environment parsing more strictly.

Why separate:

- This is export/build correctness and does not depend on the runtime Qwen3-TTS changes.
- It can be reviewed and tested with Python/export workflows.

Validation:

- `git diff --check origin/main..HEAD` passed.
- Review fix applied: fallback parser now releases the parser bound to the old network before replacing `network`.
- Local lightweight test run is blocked by missing `modelopt` in the current macOS environment:

```text
ModuleNotFoundError: No module named 'modelopt'
```

Expected validation environment:

```text
WSL/x86 export env with nvidia-modelopt installed
```

### PR 3: Qwen3-TTS runtime correctness

Commits:

```text
73afadd Align Qwen3-TTS runtime semantics
8a0df08 Add native Qwen3-TTS CodePredictor engine path
```

Purpose:

- Align Qwen3-TTS runtime behavior for prefill, language IDs, residual addends, and active RVQ groups.
- Add first-class native Qwen3-TTS CodePredictor ONNX/export/build/runtime path.
- Load `qwen3_tts_cp.engine` when present instead of using the generic CodePredictor path.

Why separate:

- This is the largest semantic change.
- It is the core correctness fix for Qwen3-TTS, but it should not block generic build/export PRs.

Validation on Orin:

- Generic official CodePredictor path generated audio, but semantic ASR check failed:

```text
ASR text: 那不是你的。
match: false
```

- Native Qwen3-TTS CodePredictor path was active:

```text
Qwen3-TTS CodePredictor enabled: .../qwen3_tts_cp.engine
Skipping generic CodePredictor CUDA graph capture; Qwen3-TTS CodePredictor engine is enabled.
```

- Native path generated correct audio:

```text
ASR text: 你好。
match: true
```

- Product-side one-shot official backend also passed with the clean official binary and plugin:

```text
ASR text: 你好。
match: true
```

Post-review fixes applied:

- `buildQwen3TTSCodePredictorEngine()` now calls the current `parseOnnxModel(builder.get(), network, path)` signature.
- `Qwen3TTSCodePredictorEngine` GPU buffers use RAII holders, so constructor exceptions do not leak successful prior `cudaMalloc` allocations.
- `kQwen3TTSActiveCodePredictorGroups=13` is only used when the native `qwen3_tts_cp.engine` is active; generic CP fallback preserves the upstream 15 residual groups.
- CPU and native CP samplers now apply `topP` when `0 < topP < 1`.
- raw text/token/logit diagnostic messages were downgraded from `LOG_INFO` to `LOG_DEBUG`.

Post-review Orin validation:

```text
Build:
cmake --build /tmp/tensorrt-edge-llm-upstream-runtime-0505/build_upstream_runtime_nativecp_sm87_gemm \
  --target qwen3_tts_inference llm_build -j1

Result:
qwen3_tts_inference: built
llm_build: built
```

Semantic validation result:

```text
Correct fixed runtime path:
  talker:         /home/harvest/qwen3-tts-edgellm-runtime/engines/talker
  code_predictor: /home/harvest/qwen3-tts-edgellm-runtime/engines/code_predictor
  code2wav:       /home/harvest/qwen3-tts-edgellm-runtime/engines/code2wav
  tokenizer:      /home/harvest/qwen3-tts-trt-edge-llm-export

Input: 你好。
Generated audio: /tmp/qwen3tts-review-regression/out-fixed/audio/audio_req0.wav
Local copy: /Users/harvest/project/jetson-voice/qwen3tts-review-audio-fixed-0505.wav
ASR text: 你好。
match: true
```

Important negative validation:

```text
Wrong/old runtime path:
  talker: /home/harvest/qwen3-tts-trt-edge-llm-export/engines/talker

Input: 你好。
Generated audio: /tmp/qwen3tts-review-regression/out/audio/audio_req0.wav
Local copy: /Users/harvest/project/jetson-voice/qwen3tts-review-audio-0505.wav
ASR text: Current
match: false
```

That bad audio was manually confirmed by listening. It was caused by validating against the old export/runtime Talker engine, not by the review fixes above. The product-side default path in `app/backends/trt_edge_llm_ipc.py` already prefers `~/qwen3-tts-edgellm-runtime` when present, which is the correct path.

## What should not go to EdgeLLM upstream in this round

Product-side streaming should stay in `jetson-voice`, not in the EdgeLLM PRs.

Keep product-side:

- resident `qwen3_tts_worker`
- JSONL worker protocol
- frame/chunk streaming
- `continuous_playback` and `low_latency` profiles
- FastAPI/V2V orchestration

Reason:

- These are product/service behavior, not generic EdgeLLM inference framework fixes.
- A callback-style streaming API was deliberately removed from the EdgeLLM runtime patch to keep the upstream PR focused.

## Product-side local changes

Uncommitted local `jetson-voice` changes from this pass:

```text
app/backends/trt_edge_llm_tts.py
app/backends/trt_edge_llm_ipc.py
app/tests/test_trt_edge_llm_ipc_paths.py
app/tests/test_trt_edge_llm_tts.py
```

Purpose:

- Make the product one-shot official backend pass `--codePredictorEngineDir`.
- Add a regression test for that CLI argument.
- Prefer `~/qwen3-tts-edgellm-runtime` for product-side TTS engine directories when it exists, avoiding accidental use of the old broken Talker engine export.
- Add path-selection regression coverage for the fixed runtime directory and explicit CP override.

Local product tests:

```text
uv run pytest app/tests/test_trt_edge_llm_tts.py app/tests/test_trt_edge_llm_ipc_paths.py
4 passed
```

These changes belong to the product repo, not EdgeLLM upstream.

## Suggested review order

1. Review and open PR 1 first.
2. Review PR 2 after checking WSL/export tests with `nvidia-modelopt`.
3. Review PR 3 last, because it is largest and depends on the correctness argument from Orin semantic validation.

## Independent review findings (2026-05-05, GPT-5 second-opinion pass)

Source: codex:codex-rescue read-only review of the full 6-commit stack at
`.worktrees/tensorrt-edge-llm-official-qwen3-tts` vs `origin/main`.

### PR 1 — Jetson build compatibility: READY

No blocking findings.

- `CMakeLists.txt:65` keeps the `80;86;89` default when `CMAKE_CUDA_ARCHITECTURES` is unset and `AARCH64_BUILD` is not defined → existing CI unchanged.
- `cmake/CuteDsl.cmake:710` change is guarded by `CUDA < 12.8`.
- `cpp/kernels/talkerMLPKernels/cuteDslGemmRunner.h:25` `cudaLibrary_t` shim is double-guarded by the project compat macro and `CUDA_VERSION < 12800`.

### PR 2 — Export and builder robustness: NEEDS-FIX

**Major — ONNX fallback parser/network lifetime**: `cpp/builder/builderUtils.cpp:284,314`. The fallback path resets `network` while the parser created against the old network is still alive, risking a dangling reference. Fix order: `parser.reset()` first, then `network.reset(...)`, then create the fallback parser.

Confirmed clean:
- safetensors shard handling at `tensorrt_edgellm/llm_models/model_utils.py:553` checks for missing index, empty `weight_map`, and missing shard files.
- Qwen3-TTS checkpoint loading is gated on `_is_qwen3_tts_model` — no silent behavior change for other model families.

### PR 3 — Qwen3-TTS runtime correctness: NEEDS-FIX (has a blocker)

**Blocker — compile break**: `cpp/builder/llmBuilder.cpp:272`. `buildQwen3TTSCodePredictorEngine()` still calls the old `parseOnnxModel(network.get(), path)` signature, but the header now declares `parseOnnxModel(builder, unique_ptr<network>&, path)`. Won't link on any platform. Fix: `parseOnnxModel(builder.get(), network, onnxPath.string())`.

**Major — RAII gap on cudaMalloc**: `cpp/runtime/qwen3OmniTTSRuntime.cpp:385,405`. `Qwen3TTSCodePredictorEngine` constructor allocates raw `cudaMalloc` pointers; an exception between allocations leaks. Wrap in RAII buffer holders before assigning to members.

**Major — generic CP fallback silently changes behavior for non-Qwen3-TTS users**: `cpp/runtime/qwen3OmniTTSRuntime.h:50` + `cpp/runtime/qwen3OmniTTSRuntime.cpp:2072,2158`. `kActiveCodePredictorGroups = 13` is applied even when `qwen3_tts_cp.engine` is absent, so the generic CP path is no longer bit-identical to upstream (was 15 heads). Gate the 13-group behavior on a model-type flag; preserve 15 in the generic fallback.

**Major — top-p dropped in CPU sampler**: `cpp/runtime/qwen3OmniTTSRuntime.cpp:161,1818,2052`. `talkerTopP` still flows through `SamplingParams`, but `sampleLogitsCPU` ignores it. Any user with non-trivial top-p gets incorrect distributions. Either implement top-p in the CPU sampler or fall back to the GPU sampler when top-p is set.

**Minor — log noise**: `cpp/runtime/qwen3OmniTTSRuntime.cpp:1772,1802`. Raw text / token ids / logits at `LOG_INFO`. Downgrade to `LOG_DEBUG` or gate on `--debug`.

### Cross-cutting

- Test coverage gap (`tests/python-unittests/test_qwen3_tts_export.py:72`): no coverage for native CP ONNX export, C++ builder compile, runtime fallback when `qwen3_tts_cp.engine` is absent, top-p correctness, or residual active-group behavior.
- License headers OK on inspected new/modified files.
- No FastAPI / IPC / streaming / JSONL product code leaked into the upstream diff — the strip is clean.

### Top 3 most likely reasons NVIDIA reviewers will reject

1. Compile break in `LLMBuilder::buildQwen3TTSCodePredictorEngine()` — stale `parseOnnxModel` signature. Won't link.
2. Generic CP fallback silently switches non-Qwen3-TTS users from 15 to 13 active groups with no opt-out.
3. CPU sampler regression: `topP` is silently ignored, changing output for any user who sets it.

### Issues that hurt non-Jetson / non-Qwen3-TTS upstream users (priority)

- `parseOnnxModel` compile break (platform-independent, blocks all builds).
- ONNX fallback parser/network lifetime hazard (every builder user hitting the fallback).
- Generic CP 13-group regression (every user without `qwen3_tts_cp.engine`).
- Top-p sampler drop (every user setting top-p).

### Recommendation

- PR 1: open as-is.
- PR 2: fix the parser/network lifetime ordering before opening.
- PR 3: do **not** open until the blocker (compile break), the two majors (RAII, generic fallback gating), and the top-p regression are fixed. Add at least one test that exercises the generic CP fallback path so the 15→13 regression cannot reappear.

## Self-review after fixes (2026-05-05)

Reviewer stance: treat this as not ready unless it builds on Orin and produces semantically correct TTS from the fixed runtime artifacts.

### Findings

No remaining blocker found in the files touched by this fix pass.

Resolved findings:

- `cpp/builder/builderUtils.cpp`: parser bound to the old network is destroyed before `network.reset(...)`.
- `cpp/builder/llmBuilder.cpp`: native Qwen3-TTS CP builder now uses the new `parseOnnxModel` API.
- `cpp/runtime/qwen3OmniTTSRuntime.cpp`: native CP temporary device allocations are RAII-managed.
- `cpp/runtime/qwen3OmniTTSRuntime.cpp/.h`: 13 active residual groups are gated to native Qwen3-TTS CP; generic CP fallback keeps 15 groups.
- `cpp/runtime/qwen3OmniTTSRuntime.cpp`: top-p is implemented in both CPU and native-CP host-side samplers.
- `cpp/runtime/qwen3OmniTTSRuntime.cpp`: verbose debug logs are `LOG_DEBUG`.

Residual risks:

- There is still no automated C++ unit/integration test for native CP runtime semantics, top-p sampling, or generic fallback 15-group preservation. Current confidence comes from Orin build plus end-to-end TTS/ASR validation.
- The correct TTS result depends on using the fixed runtime engine export at `~/qwen3-tts-edgellm-runtime`. The older `/home/harvest/qwen3-tts-trt-edge-llm-export/engines/talker` can still generate audio but the audio is semantically wrong. Do not use the old Talker engine as a validation artifact.
- The macOS Python unit test for EdgeLLM export is still blocked by missing `modelopt`; WSL export validation remains required before opening PR 2/3 upstream.

### Commands run

```text
git -C .worktrees/tensorrt-edge-llm-official-qwen3-tts diff --check
PASS

cmake --build /tmp/tensorrt-edge-llm-upstream-runtime-0505/build_upstream_runtime_nativecp_sm87_gemm --target qwen3_tts_inference llm_build -j1
PASS

uv run pytest app/tests/test_trt_edge_llm_tts.py app/tests/test_trt_edge_llm_ipc_paths.py
4 passed
```

### Upstream readiness after fixes

- PR 1: READY.
- PR 2: READY for WSL/export validation; parser lifetime blocker is fixed locally.
- PR 3: READY for another reviewer to inspect locally; do not open upstream until the fix commit is folded into the PR stack and WSL/native CP export validation is rerun.

## TTS quality investigation update (2026-05-06)

Current conclusion: the remaining audio quality issue is not caused by 13 vs 15 residual groups, tokenizer IDs, prefill layout, language token mapping, or the single-head CP engine alone.

Evidence:

- Current official Talker engine reports `hidden_states` as FP16, while the old product `TRTTalkerEngine` reports `last_hidden` as 4-byte FP32 and FP32 KV.
- For `语音合成的稳定性。`, old product and official runtime agree on the prefill primary codec token `1995`; prefill logits are close.
- Divergence starts after frame-0 residual feedback into Talker. Old product frame-0 RVQ starts `[1995,2031,851,...]`; official runtime frame-0 RVQ starts `[1995,1184,282,...]`.
- Temporarily pointing official runtime at the old product `cp_unified_bf16.engine` still produced ASR `语言和声的稳定性。`, so the CP engine file itself is not sufficient to explain the issue.
- Product-side sentence splitting mitigates long-form accumulation, but it cannot fix this first-frame residual divergence.

Patch in progress:

- `LLMEngineRunner` now exposes engine tensor dtype.
- `Qwen3OmniTTSRuntime` no longer assumes Talker `hidden_states` is FP16; it allocates/copies Talker hidden buffers using the engine output dtype and converts native-CP inputs via `copyTensorToHostFloat`.
- Export-side Talker wrapper now returns `hidden_states.to(torch.float32)` so newly exported Talker ONNX can request FP32 hidden output.
- Orin build passes with the current FP16 hidden engine and remains backward compatible, but quality is unchanged until a Talker engine with FP32 hidden/KV behavior is built and validated.

Next validation needed:

- Build a new Talker engine whose `hidden_states` output is FP32, preferably also checking whether KV precision remains FP16 or can/should match the old FP32 behavior.
- Re-run `语音合成的稳定性。` and compare frame-0 RVQ codes; target is old-product-like first residual code `2031` rather than current official `1184`.

### Sampling addendum (2026-05-06)

The first-frame divergence above was partly misleading because the official runtime and old product were not using equivalent sampling defaults. The official example default was `talker_top_k=50, talker_top_p=1.0`, while the EdgeLLM TTS accuracy dataset path uses `talker_top_k=40, talker_top_p=0.8`.

Validation on Orin Nano/NX with the existing official FP16 Talker engine:

- Explicit `top_k=40, top_p=0.8`, `QWEN3_TTS_SEED=1234`, text `语音合成的稳定性。`:
  - ASR: `语言合成的稳定性。`
  - First frames: `[1995,1889,1114,...]`, `[215,1931,1985,...]`, `[1521,490,1264,...]`
  - Primary history: `[1995,215,1521,1095,333,948,172,1749,1746,1050,640,1507,229,2005,2035,2035,781,91,605,261,1390,64,1739,2150]`
- After changing the example defaults to `top_k=40, top_p=0.8`, the same request with no explicit sampling fields produced the same ASR and code history.
- Audio artifact pulled locally: `qwen3tts-official-default-sampling-fixed-0506.wav`.

Updated conclusion: the immediate audible failure was primarily caused by loose example/product sampling defaults, not the 13-vs-15 residual group choice. FP16 Talker hidden/KV may still be a quality risk versus the older FP32 product engine, but it is no longer the first fix to pursue for the current bad-audio report.

Patch applied:

- EdgeLLM `examples/omni/qwen3_tts_inference.cpp`: default `talker_top_k=40`, `talker_top_p=0.8`.
- EdgeLLM `docs/source/user_guide/examples/tts.md`: sample input and parameter table updated to the same defaults.
- Product `app/backends/trt_edge_llm_tts.py` and `native/edgellm_voice_worker/qwen3_tts_worker.cpp`: default sampling aligned to `40/0.8`.

Verification:

```text
cmake --build /tmp/tensorrt-edge-llm-upstream-runtime-0505/build_upstream_runtime_nativecp_sm87_gemm --target qwen3_tts_inference -j1
PASS

uv run pytest app/tests/test_trt_edge_llm_tts.py app/tests/test_trt_edge_llm_ipc_paths.py
PASS (7 tests)
```

## TTS precision investigation correction (2026-05-06 late)

The sampling addendum above is incomplete and should not be used as the final root-cause conclusion.

New evidence:

- Product reference engine `/home/harvest/voice_test/models/qwen3-tts/engines/talker_decode_bf16.engine` was inspected directly with TensorRT runtime metadata:
  - `inputs_embeds`: FP32
  - all `past_key_*` / `past_value_*`: FP32
  - `logits`: FP32
  - `last_hidden`: FP32
  - profiles: prefill profile `[1,1..200,1024]`, decode profile `[1,1,1024]`
- Current official EdgeLLM plugin Talker engines were inspected directly:
  - FP16 engine: `inputs_embeds`: FP16, `past_key_values_*`: FP16, `hidden_states`: FP32, `logits`: FP32
  - hidden-FP32 test engine: still `inputs_embeds`: FP16 and `past_key_values_*`: FP16
  - KV-BF16 test engine: `inputs_embeds`: FP16 and `past_key_values_*`: BF16
- For the same text `语音合成的稳定性。`, same historical CP BF16 engine, and same sampling:
  - product reference primary codes: `[1995,215,1521,690,690,333,1900,1121,376,622,622,1746,1391,589,828,1507,790,386,606,613,429,662,299,91,551,720,1624]`
  - official FP16 plugin path: starts `[1995,215,294,...]`
  - official BF16-KV plugin path: starts `[1995,215,212,...]`
  - official full-BF16 plugin path (`inputs_embeds=BF16`, KV=BF16, hidden/logits=FP32): starts `[1995,215,294,...]`
- Vocoder/code2wav was excluded: running ORT vocoder on official RVQ codes produced audio matching official code2wav output within int16 noise.

Updated conclusion:

- The remaining quality issue is not fixed by top-k/top-p defaults, CP engine replacement, 13-vs-15 residual groups, or vocoder choice.
- Divergence starts after the first residual feedback into Talker, exactly where the loop state crosses `CP residual -> inputs_embeds -> Talker decode`.
- The known-good product engine keeps the Talker loop boundary and KV cache in FP32. Official EdgeLLM's AttentionPlugin path currently supports only FP16/BF16 Q/K/V and KV cache, so `EDGELLM_LLM_BUILDER_PRECISION=fp32` does not produce an FP32-KV Talker engine.
- A true BF16 plugin-boundary engine was tested and did not recover reference codes, so BF16 is not the current primary fix path.

Implication for upstream PR design:

- A small upstream PR that only changes runtime sampling/defaults is not sufficient for Qwen3-TTS quality parity.
- To make the official backend quality-correct, we need one of:
  - a generic precision mode/export path that can preserve FP32 Talker inputs and KV cache for precision-sensitive autoregressive audio models, or
  - an alternate Qwen3-TTS Talker runtime path compatible with the product/reference TensorRT engine layout (`past_key_i` / `past_value_i`, no EdgeLLM AttentionPlugin bindings).
- Extending AttentionPlugin to true FP32 KV is likely large and may not be the best upstreamable first patch, because current plugin kernels explicitly support FP16/BF16 QKV only.

Current next step:

- Prototype the least invasive path that lets official `qwen3_tts_inference` consume a reference-quality FP32-boundary Talker engine, then decide whether it is upstreamable or should remain product-side.
