# Qwen3 Orin Nano Official Smoke — 2026-05-11

Status: PIPELINE OK, quality unaudited (CuTe DSL Talker MLP fallback issue).

This is the §4 official/minimal profile validation per
`docs/reproduction-remaining-work-2026-05-11.md`. Smoke uses the
`official-qwen3-tts-upstream-runtime` EdgeLLM branch + the
`orin-nano-official-2026-05-10` HF artifact set.

## Environment

- Host: Jetson Orin Nano 8GB (actually Orin NX Super per fleet description), JetPack 6
- Host TensorRT: 10.3.0.30
- EdgeLLM fork: `suharvest/TensorRT-Edge-LLM`, branch
  `official-qwen3-tts-upstream-runtime` at commit `631e7f8`
- Binary built on orin-nx (SM87 cross-build) and transferred via `fleet transfer`:
  - `qwen3_tts_inference` md5 `4d11f22e4fdefc917c08e14fabd750aa`
  - `libNvInfer_edgellm_plugin.so` md5 `4619fac869d35f9cec6fe8482c1458b1`

## Artifact set

`orin-nano-official-2026-05-10` at `/home/harvest/qwen3-models-official/`
(29 files, all SHA-256 verified via the sidecar).

Manifest entries had two gaps caught by the smoke and fixed during this run:

1. `code_predictor/{lm_heads,codec_embeddings,small_to_mtp_projection}.safetensors`
   and `talker/{embedding,text_embedding,text_projection}.safetensors` were on
   HF but not listed in `required_files` (commit `0f9f985`).
2. `code2wav/config.json` (1292 B) was on HF but not listed (commit `ff21fea`).
   Without it the runtime emitted "Will output RVQ codes only" — no WAV.

After both fixes, the smoke generated audio end-to-end.

## Smoke result

Input: `{"requests":[{"messages":[{"role":"user","content":"今天天气真好。"}]}]}`

```
[03:44:xx] First codec token (from prefill): 1445 (eos=2150)
[03:44:xx] Clamped maxAudioLength from 4096 to 50 (prefill=9, KV capacity=512)
[03:44:xx] Generated 15 audio frames (exit: EOS, last_code=2150)
[03:44:xx] Done: 1/1 requests succeeded
```

Output:

| File | Size | Meaning |
|---|---|---|
| `audio_req0.wav` | 57644 B | 24kHz mono PCM, 28800 samples = 1.2 s audio |
| `rvq_req0.safetensors` | 1114 B | RVQ codebook ids before Code2Wav |
| `result.json` | 488 B | request metadata |

WAV pulled to `docs/audio-evidence/nano-official-2026-05-11.wav` for
listening comparison. Not yet auditioned.

## Known issue: Talker MLP zero output without CuTe DSL GEMM

Inference logs both at startup and at first token:

```
[ERROR] [talkerMLPKernels.cu:341:invokeTalkerMLP] CuTe DSL GEMM not
compiled. Rebuild with -DENABLE_CUTE_DSL=gemm (or ALL).
```

Reading `cpp/kernels/talkerMLPKernels/talkerMLPKernels.cu`:

```cpp
#ifdef CUTE_DSL_GEMM_ENABLED
    // FC1 -> SiLU -> FC2 …
#else
    LOG_ERROR("CuTe DSL GEMM not compiled. Rebuild with -DENABLE_CUTE_DSL=gemm (or ALL).");
    return;
#endif
```

The function returns without writing the output buffer. The caller
proceeds anyway, so the Talker prefill consumes whatever was in the
buffer at allocation (effectively zero). On the highperf fork branch
we worked around this by defaulting `ENABLE_CUTE_DSL=gemm` and shipping
prebuilt artifacts (commits `190d977` / `de3939c` on
`qwen3-tts-highperf-runtime-w8a16`). The official upstream branch keeps
the default OFF, so a fresh build on the official line silently produces
broken Talker MLP output.

This is a candidate upstream issue: `talkerMLPKernels.cu` should either
(a) have a non-CuTe-DSL fallback path, or (b) refuse to compile rather
than silently no-op at runtime.

## Other observed warnings (not fatal)

- `Using an engine plan file across different models of devices` — the
  HF engines were cooked on a different SM than Orin Nano sees at
  runtime. Loads fine on TRT 10.3, no functional impact observed.
- `Clamped maxAudioLength from 4096 to 50 (prefill=9, KV capacity=512)`
  — the Talker engine was built with KV capacity 512; ample for this
  prompt but a hard cap for long generations.

## Reproduce

```bash
# Source: https://github.com/suharvest/qwen3-edgellm-jetson
cd ~/project/qwen3-edgellm-jetson
HF_ENDPOINT=https://hf-mirror.com python3 scripts/deploy_qwen3_artifacts.py \
  --set orin-nano-official-2026-05-10 \
  --root ~/qwen3-models-official \
  --verify-sha256

# EdgeLLM official branch build (already documented in reproduce-from-zero.md;
# requires -DCMAKE_CUDA_ARCHITECTURES=87 explicit on this branch).

# Smoke:
INFER=~/project/TensorRT-Edge-LLM/build/examples/omni/qwen3_tts_inference
ROOT=~/qwen3-models-official
$INFER \
  --inputFile=<input.json> \
  --talkerEngineDir="$ROOT/engines/orin-nano/official/talker" \
  --codePredictorEngineDir="$ROOT/engines/orin-nano/official/code_predictor" \
  --code2wavEngineDir="$ROOT/engines/orin-nano/official/code2wav" \
  --tokenizerDir="$ROOT/tts/tokenizer" \
  --outputFile=result.json --outputAudioDir=.
```

Input file template:

```json
{
  "batch_size": 1,
  "apply_chat_template": true,
  "add_generation_prompt": true,
  "requests": [
    {"messages": [{"role":"user","content":"今天天气真好。"}]}
  ]
}
```

## Files

- Audio: `docs/audio-evidence/nano-official-2026-05-11.wav`
- Smoke input: `/tmp/nano_official_input.json` (on orin-nano)
- Full smoke log: `/tmp/nano_official_smoke.log` (on orin-nano)
- Output dir: `/tmp/nano_official_smoke_out/` (on orin-nano)
