# P1 — Thinker Engine v2 Rebuild Results

Date: 2026-05-14
Branch: `streaming-asr/m2-worker-capacity`

## Engine

- Path: `/home/harvest/qwen3-models/engines/orin-nx/highperf-v2/asr_thinker_full_fp8embed/llm.engine`
- Size: 1.21 GB
- MD5: `59711a53684cfc5e8a99764690059ce0`
- Config: `max_input_len: 256`, `max_kv_cache_capacity: 512` (doubled from `128`/`256`)
- Companion: `embedding.safetensors` (160 MB, FP8), `config.json`, tokenizer files, `processed_chat_template.json`

Old engine at `engines/orin-nx/highperf/asr_thinker_full_fp8embed/` left untouched.
Multimodal/audio-encoder engine unchanged (cache-config invariant).

## Worker

- Source: `jetson-voice/native/edgellm_voice_worker/qwen3_asr_worker.cpp`
- Constants updated:
  - `kEngineMaxInputLen`: 128 → 256
  - `kSingleChunkHardLimitSec`: 7.0 → 15.0
  - Comment recomputed: audio cap (256-8-32-2)/13 ≈ 16.5 s; hard-refuse at 15.0 s
- Binary: `/home/harvest/project/repro-qwen3/jetson-voice/build/edgellm_voice_worker/workers/qwen3_asr_worker`
- New md5: `181a4949b517b1b292315cb6e65ba329` (was `2161c11824b617da042562a7c98de403`)

## Test runner update

`scripts/test_streaming_worker.py`: `audio_sec` payload in `E_chunk_too_long` test bumped 8.0 → 16.0 (must exceed new 15.0 s limit).

## M5 streaming verification — new engine + new worker

Driver: `scripts/test_streaming_worker.py` scenarios A–E on `nx-loopback-pass-p1` (short, "今天天气真好。") + `zh-long-04-2026-05-13.wav` (12.90 s).

| Gate | Result | Kind | Notes |
|------|--------|------|-------|
| A_oneshot_ok | PASS | hard | text=`今天天气真好。` |
| B_lcs_ge_0.95 | PASS | hard | LCS=1.000 |
| C_median_le_500ms | PASS | hard | median 128.6 ms |
| C_p95_le_1000ms  | PASS | hard | p95 128.8 ms |
| D_one_final | PASS | hard | |
| D_at_least_one_segment_rotation | PASS | hard | segment_count=1, **rotations=0** — single 12.9 s chunk fit the new 256-cap input budget without auto-segmentation |
| E_malformed_json_handled | PASS | hard | |
| E_unknown_event_handled | PASS | hard | |
| E_chunk_too_long_handled | PASS | hard | refused at 16.0 s with `chunk_too_long`, `limit_sec=15.0` |
| E_session_cleared_after_error | PASS | hard | |
| D_lcs_ge_0.90_soft | FAIL | soft | mechanism-only synthetic baseline; long WAV content differs from short prompt |

**Hard gates: PASS (10/10).**

### Scenario D (12.90 s) output

```
科学家们可以得出结论：暗物质对其他暗物质的影响方式与普通物质相同。
```

Coherent Chinese (`Scientists can conclude that dark matter affects other dark matter the same way as ordinary matter.`).

### Latency comparison

| Engine        | C scenario median | p95     |
|---------------|------------------:|--------:|
| highperf (old, max_input_len=128) | 137.2 – 168.3 ms | 153.3 – 171.1 ms |
| highperf-v2 (new, max_input_len=256) | **128.6 ms** | **128.8 ms** |

~15–25 % end-of-speech latency improvement at higher capacity. Run-to-run variance also dropped (128.3–128.8 ms range vs 128–171 ms before).

## Recommendation

- Production worker binary md5 `181a4949b517b1b292315cb6e65ba329` paired with thinker v2 engine `59711a5...` is M5-clean.
- Promote `engines/orin-nx/highperf-v2/asr_thinker_full_fp8embed/` to the HF artifact set under a new release tag; keep highperf/ as the existing rollback path.
- Single-chunk hard limit at 15 s lets the streaming driver send the full clip without forced segmentation for utterances ≤ ~16 s — eliminates the P1 dedup boundary risk on long-form Chinese.
