#!/usr/bin/env bash
# HISTORICAL — wrapper used on 2026-05-13 to rebuild the Qwen3-ASR thinker
# engine on Orin NX with max_input_len=256 (highperf-v2 / asr_thinker_full
# _fp8embed). For canonical pipeline + rationale see
# docs/asr-thinker-engine-build-recipe.md.
#
# Usage (env-driven, all optional with defaults):
#   ONNX_DIR=/home/harvest/qwen3-asr-fp8emb-thinker \
#   ENGINE_DIR=/home/harvest/qwen3-models/engines/orin-nx/highperf-v2/asr_thinker_full_fp8embed \
#   MAX_INPUT_LEN=256 MAX_KV=512 MAX_BATCH=1 \
#   bash scripts/build_qwen3_asr_thinker_engine.sh
#
# Prereqs:
#   - TensorRT-Edge-LLM source tree built (build/examples/llm/llm_build exists)
#   - ONNX_DIR contains model.onnx, onnx_model.data, embedding.safetensors,
#     config.json, tokenizer files
#   - embedding.safetensors is FP8 E4M3 with embedding_scale (run
#     scripts/quantize_embedding_safetensors_fp8.py first if it's still FP16)

set -euo pipefail

ONNX_DIR="${ONNX_DIR:?ONNX_DIR is required}"
ENGINE_DIR="${ENGINE_DIR:?ENGINE_DIR is required}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-256}"
MAX_KV="${MAX_KV:-512}"
MAX_BATCH="${MAX_BATCH:-1}"

# Auto-detect llm_build binary
LLM_BUILD="${LLM_BUILD:-}"
if [[ -z "${LLM_BUILD}" ]]; then
  for c in \
    "$HOME/project/repro-qwen3/TensorRT-Edge-LLM/build/examples/llm/llm_build" \
    "$HOME/project/tensorrt-edge-llm/build_sm87/examples/llm/llm_build" \
    "$HOME/project/tensorrt-edge-llm/build/examples/llm/llm_build" ; do
    if [[ -x "$c" ]]; then LLM_BUILD="$c"; break; fi
  done
fi
[[ -x "${LLM_BUILD:-}" ]] || { echo "[fatal] llm_build binary not found; set LLM_BUILD=" >&2; exit 1; }

PLUGIN="${EDGELLM_PLUGIN_PATH:-}"
if [[ -z "${PLUGIN}" ]]; then
  for c in \
    "$HOME/project/repro-qwen3/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so" \
    "$HOME/project/tensorrt-edge-llm/build_sm87/libNvInfer_edgellm_plugin.so" ; do
    if [[ -f "$c" ]]; then PLUGIN="$c"; break; fi
  done
fi
[[ -f "${PLUGIN:-}" ]] || { echo "[fatal] EdgeLLM plugin .so not found" >&2; exit 1; }

# Pre-build sanity: verify ONNX has FP8 embedding
echo "[asr-thinker] ONNX_DIR    = $ONNX_DIR"
echo "[asr-thinker] ENGINE_DIR  = $ENGINE_DIR"
echo "[asr-thinker] llm_build   = $LLM_BUILD"
echo "[asr-thinker] plugin      = $PLUGIN"
echo "[asr-thinker] maxBatch=$MAX_BATCH  maxInputLen=$MAX_INPUT_LEN  maxKV=$MAX_KV"
for f in model.onnx onnx_model.data embedding.safetensors config.json; do
  if [[ ! -f "$ONNX_DIR/$f" ]]; then
    echo "[fatal] missing $ONNX_DIR/$f" >&2; exit 1
  fi
done

# Inspect embedding.safetensors dtype (best-effort, requires python+safetensors)
python3 - "$ONNX_DIR/embedding.safetensors" <<'PY' || echo "[warn] embedding dtype check skipped (python/safetensors unavailable)"
import sys
try:
    from safetensors import safe_open
except Exception:
    sys.exit(0)
with safe_open(sys.argv[1], framework="pt") as f:
    keys = list(f.keys())
    print(f"[asr-thinker] embedding.safetensors keys: {keys}")
    if "embedding" not in keys:
        sys.exit("[fatal] no 'embedding' tensor in safetensors")
    t = f.get_tensor("embedding")
    print(f"[asr-thinker] embedding dtype={t.dtype} shape={tuple(t.shape)}")
    if "scale" not in str(t.dtype).lower() and "float8" not in str(t.dtype) and "embedding_scale" not in keys:
        print("[warn] embedding looks like FP16/FP32 and there is no embedding_scale. "
              "Run scripts/quantize_embedding_safetensors_fp8.py first for FP8 path.")
PY

mkdir -p "$ENGINE_DIR"

LOG_DIR="$ENGINE_DIR/_build_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/llm_build.$(date +%Y%m%dT%H%M%S).log"

export LD_PRELOAD="$PLUGIN${LD_PRELOAD:+:$LD_PRELOAD}"

START=$(date +%s)
echo "[asr-thinker] invoking llm_build, log=$LOG"
"$LLM_BUILD" \
  --onnxDir "$ONNX_DIR" \
  --engineDir "$ENGINE_DIR" \
  --maxBatchSize "$MAX_BATCH" \
  --maxInputLen "$MAX_INPUT_LEN" \
  --maxKVCacheCapacity "$MAX_KV" \
  2>&1 | tee "$LOG"
RC="${PIPESTATUS[0]}"
END=$(date +%s)

echo "[asr-thinker] llm_build exit=$RC wall=$((END-START))s"
[[ "$RC" == "0" ]] || exit "$RC"

echo "[asr-thinker] artifacts:"
ls -la "$ENGINE_DIR"
if [[ -f "$ENGINE_DIR/llm.engine" ]]; then
  md5sum "$ENGINE_DIR/llm.engine"
  du -h "$ENGINE_DIR/llm.engine"
fi
if [[ -f "$ENGINE_DIR/config.json" ]]; then
  echo "[asr-thinker] config.json:"
  cat "$ENGINE_DIR/config.json"
fi
