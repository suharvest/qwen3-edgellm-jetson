#!/usr/bin/env bash
# M5 — End-to-end streaming verification on reproduction prompts.
#
# Wraps scripts/streaming_e2e_verify.py with the canonical prompt set
# from docs/plans/m5-test-wavs.md.
#
# Designed to run INSIDE jetson_voice_slim on orin-nx, i.e. with the
# repo mounted at /opt/qwen3-edgellm-jetson and the worker binary at
# /opt/jv-workers/qwen3_asr_worker. Override --repo-root / --worker
# when running elsewhere.
#
# For each reproduction prompt:
#   1. One-shot baseline via worker scenario A (matches HTTP /asr).
#   2. Streaming via scenario B (mel_path chunks, 500 ms cumulative).
#   3. Streaming via scenario F (pcm_b64 chunks, 500 ms cumulative —
#      requires worker built with C++ MelExtractor).
# Hard gates: B/F LCS vs baseline >= 0.95, median end-of-speech latency
# <= 500 ms, p95 <= 1000 ms.
#
# Exits 0 only on full pass.
set -uo pipefail

REPO_ROOT="/opt/qwen3-edgellm-jetson"
WORKER="/opt/jv-workers/qwen3_asr_worker"
PLUGIN="/opt/edgellm-bin/libNvInfer_edgellm_plugin.so"
ENGINE_DIR="/opt/models/qwen3-edgellm/engines/orin-nx/highperf/asr_thinker_full_fp8embed"
MULTIMODAL_ENGINE_DIR="/opt/models/qwen3-edgellm/engines/orin-nx/highperf/asr_audio_encoder"
MEL_SETTINGS=""
MEL_FILTERS=""
RESULTS_MD=""
RESULTS_JSON=""
LATENCY_RUNS=5
LCS_GATE=0.95
MEDIAN_GATE_MS=500
P95_GATE_MS=1000
MAX_GEN=200

while [ $# -gt 0 ]; do
  case "$1" in
    --repo-root)               REPO_ROOT="$2"; shift 2 ;;
    --worker)                  WORKER="$2"; shift 2 ;;
    --plugin)                  PLUGIN="$2"; shift 2 ;;
    --engine-dir)              ENGINE_DIR="$2"; shift 2 ;;
    --multimodal-engine-dir)   MULTIMODAL_ENGINE_DIR="$2"; shift 2 ;;
    --mel-settings)            MEL_SETTINGS="$2"; shift 2 ;;
    --mel-filters)             MEL_FILTERS="$2"; shift 2 ;;
    --results-md)              RESULTS_MD="$2"; shift 2 ;;
    --results-json)            RESULTS_JSON="$2"; shift 2 ;;
    --latency-runs)            LATENCY_RUNS="$2"; shift 2 ;;
    --lcs-gate)                LCS_GATE="$2"; shift 2 ;;
    --median-gate-ms)          MEDIAN_GATE_MS="$2"; shift 2 ;;
    --p95-gate-ms)             P95_GATE_MS="$2"; shift 2 ;;
    --max-gen)                 MAX_GEN="$2"; shift 2 ;;
    --with-pcm)
      # Convenience: default the mel settings to the bundled deploy/ files.
      MEL_SETTINGS="${REPO_ROOT}/deploy/audio_preprocessing/whisper_feature_extractor.json"
      MEL_FILTERS="${REPO_ROOT}/deploy/audio_preprocessing/mel_filters.bin"
      shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Reproduction prompts mirror scripts/verify_reproduction.sh: 3 fixed
# Chinese phrases, each backed by a pre-rendered TTS WAV checked into
# docs/audio-evidence/ (see docs/plans/m5-test-wavs.md).
P1_WAV="${REPO_ROOT}/docs/audio-evidence/nx-loopback-pass-p1-2026-05-11.wav"
P2_WAV="${REPO_ROOT}/docs/audio-evidence/nx-loopback-pass-p2-2026-05-11.wav"
P3_WAV="${REPO_ROOT}/docs/audio-evidence/nx-loopback-pass-p3-2026-05-11.wav"
P1_TEXT="今天天气真好。"
P2_TEXT="人工智能改变了世界。"
P3_TEXT="一二三四五六七八九十。"

for f in "$P1_WAV" "$P2_WAV" "$P3_WAV" "$WORKER" "$PLUGIN"; do
  if [ ! -f "$f" ]; then
    echo "missing required asset: $f" >&2; exit 3
  fi
done

DRIVER="${REPO_ROOT}/scripts/streaming_e2e_verify.py"
if [ ! -f "$DRIVER" ]; then
  echo "driver missing: $DRIVER" >&2; exit 3
fi

CMD=(python3 "$DRIVER"
  --worker "$WORKER"
  --plugin "$PLUGIN"
  --engine-dir "$ENGINE_DIR"
  --multimodal-engine-dir "$MULTIMODAL_ENGINE_DIR"
  --latency-runs "$LATENCY_RUNS"
  --lcs-gate "$LCS_GATE"
  --median-gate-ms "$MEDIAN_GATE_MS"
  --p95-gate-ms "$P95_GATE_MS"
  --max-gen "$MAX_GEN"
  --prompt "p1=${P1_WAV}|${P1_TEXT}"
  --prompt "p2=${P2_WAV}|${P2_TEXT}"
  --prompt "p3=${P3_WAV}|${P3_TEXT}"
)
if [ -n "$MEL_SETTINGS" ]; then
  CMD+=(--mel-settings "$MEL_SETTINGS" --mel-filters "$MEL_FILTERS")
fi
if [ -n "$RESULTS_MD" ]; then
  CMD+=(--results-md "$RESULTS_MD")
fi
if [ -n "$RESULTS_JSON" ]; then
  CMD+=(--results-json "$RESULTS_JSON")
fi

echo "== M5 streaming verification =="
echo "Worker:  $WORKER"
echo "Plugin:  $PLUGIN"
echo "Engine:  $ENGINE_DIR"
echo "MM eng:  $MULTIMODAL_ENGINE_DIR"
if [ -n "$MEL_SETTINGS" ]; then
  echo "PCM:     enabled (mel_settings=$MEL_SETTINGS)"
else
  echo "PCM:     disabled (pass --with-pcm to enable scenario F)"
fi
echo

exec "${CMD[@]}"
