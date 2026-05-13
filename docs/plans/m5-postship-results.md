== M5 streaming verification ==
Worker:  /opt/jv-workers/qwen3_asr_worker
Plugin:  /opt/edgellm-bin/libNvInfer_edgellm_plugin.so
Engine:  /opt/models/qwen3-edgellm/engines/orin-nx/highperf/asr_thinker_full_fp8embed
MM eng:  /opt/models/qwen3-edgellm/engines/orin-nx/highperf/asr_audio_encoder
PCM:     disabled (pass --with-pcm to enable scenario F)

# M5 — End-to-End Streaming Verification Results

Worker: `/opt/jv-workers/qwen3_asr_worker`
Gates: LCS ≥ 0.95, median ≤ 500.0 ms, p95 ≤ 1000.0 ms

## Aggregate

**PASS** — 3 prompts evaluated

## Per-prompt summary

| Prompt | Ground truth | A baseline | B (mel) text | B LCS | F (pcm) text | F LCS | median ms | p95 ms | Verdict |
|--------|--------------|-----------|--------------|------:|--------------|------:|----------:|-------:|---------|
| p1 | `今天天气真好。` | `今天天气真好。` | `今天天气真好。` | 1.000 | `—` | — | 152.5 | 153.3 | PASS |
| p2 | `人工智能改变了世界。` | `人工智能改变了世界。` | `人工智能改变了世界。` | 1.000 | `—` | — | 137.2 | 157.6 | PASS |
| p3 | `一二三四五六七八九十。` | `一二三四五六七八九十。` | `一二三四五六七八九十。` | 1.000 | `—` | — | 168.3 | 171.1 | PASS |

## Per-prompt gates

### p1

- A_ok: **PASS**
- B_lcs_ge_0.95: **PASS**
- C_median_le_500ms: **PASS**
- C_p95_le_1000ms: **PASS**

### p2

- A_ok: **PASS**
- B_lcs_ge_0.95: **PASS**
- C_median_le_500ms: **PASS**
- C_p95_le_1000ms: **PASS**

### p3

- A_ok: **PASS**
- B_lcs_ge_0.95: **PASS**
- C_median_le_500ms: **PASS**
- C_p95_le_1000ms: **PASS**

EXIT=0
