# Qwen3 highperf TTS+ASR loopback — 3/3 exact match (2026-05-11)

Status: **RESOLVED**. After restoring the proven OLD W8A16 kernels and
plugin .cpp from the original source tree, the slim docker container
TTS → ASR loopback produces exact text matches on 3 distinct Chinese
prompts.

## Result

| Prompt | TTS HTTP code | WAV size | duration | ASR transcription | Match |
|---|---|---|---|---|---|
| 今天天气真好。 | 200 | 127 KB | 2.64 s | 今天天气真好。 | ✅ exact |
| 人工智能改变了世界。 | 200 | 127 KB | 2.64 s | 人工智能改变了世界。 | ✅ exact |
| 一二三四五六七八九十。 | 200 | 154 KB | 3.20 s | 一二三四五六七八九十。 | ✅ exact |

Audio evidence (Mac):
- `docs/audio-evidence/nx-loopback-pass-p1-2026-05-11.wav`
- `docs/audio-evidence/nx-loopback-pass-p2-2026-05-11.wav`
- `docs/audio-evidence/nx-loopback-pass-p3-2026-05-11.wav`

## Fix chain that got us here

| Commit (EdgeLLM fork) | Title | What it did |
|---|---|---|
| `c248f73` | pass FP8 dequant scales to text embedding lookup | unblocked worker init when text_embedding.safetensors is FP8 + scale |
| `eacfefc` | include SM87 in default CUDA arch list | unblocked ASR thinker MHA dispatch |
| `190d977` / `de3939c` / `5cc6060` | ENABLE_CUTE_DSL=gemm + EMBEDDED_TARGET defaults | TalkerMLP path actually computes (was LOG_ERROR no-op) |
| `c807d10` | Revert my prefill-layout misfix | back to 9f248ed semantics + speaker-clone path |
| `6239d5f` (user) | support clone speaker slot in highperf runtime | speaker_id default-row path validated |
| **`8a26eba`** | **restore OLD W8A16 kernels** | put back `w8a16_m1_output_k_kernel`, `_hmma_m16n16k16`, `_small_m_tiled`, `_per_output_output_k_reference` |
| **`7ab7f1c`** | **restore matching plugin .cpp/.h** | plugin dispatcher signature matched to OLD kernel surface |

Plugin / worker md5 in the passing build:
- `libNvInfer_edgellm_plugin.so` = `8d634188d8914db604ae59ccc65cc107`
- `qwen3_tts_worker` = `66aad5ad4ecb0e58bd2b94ca8ff740dd`

W8A16 kernel symbol set in the new plugin matches the OLD baseline:
```
w8a16_hmma_m16n16k16_kernel
w8a16_m1_output_k_kernel
w8a16_per_output_output_k_reference_kernel
w8a16_per_output_reference_kernel
w8a16_small_m_tiled_kernel
```

## What the original regression was

The current branch's `w8A16Linear.cu` and the matching plugin
`w8A16LinearPlugin.cpp` had been refactored to a NEW kernel set
(`_per_output_tiled`, `_per_output_tiled_pair_k`) that produced
numerically different W8A16 outputs vs the validated baseline. The HF
artifacts were exported / validated against the OLD kernel set, so the
runtime+plugin built from current source produced first-codec-token
1574 (vs OLD 1995), the Talker prefill argmax drifted (1093 vs 1995),
and the resulting audio decoded by Code2Wav was monosyllable-collapsed
(嗯嗯/是是).

The OLD `.cu` and `.cpp` survived locally on `orin-nano` at
`/home/harvest/project/tensorrt-edge-llm-hlm-current/` (path baked into
the OLD plugin's string table). Restored verbatim onto this branch.

## Reproduce

The slim docker image is live on `orin-nx` (`100.82.225.102:18092`):

```bash
NX=http://100.82.225.102:18092

# Positive control:
curl -s -X POST $NX/asr -F file=@bench/wavs/S1.wav | jq .
# {"text": "今天天气真好，我们出去玩吧。", ...}

# TTS → ASR loopback (returns exact match for the prompt):
curl -s -X POST $NX/tts \
  -H 'content-type: application/json' \
  -d '{"text":"今天天气真好。"}' -o tts.wav
curl -s -X POST $NX/asr -F file=@tts.wav | jq .
```

## Follow-ups

- Audio quality issue `docs/issues/2026-05-11-tts-audio-quality-regression.md`
  is closed by `7ab7f1c`.
- Voice clone path: the runtime accepts `speaker_embedding_b64`. Quality
  on that path against the current frozen artifacts has not been
  ASR-audited yet — separate gate.
- Reconcile NEW W8A16 kernel set with the model: either keep OLD set as
  the canonical fork tip, or re-export engines to match the NEW kernel
  output. Currently OLD set is the only known-good combination.
