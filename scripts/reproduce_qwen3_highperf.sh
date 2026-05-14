#!/usr/bin/env bash
# One-shot end-to-end reproduce + verify for the Qwen3 highperf TTS+ASR
# stack on Jetson Orin NX. Idempotent — already-done steps are skipped.
#
# What it does (in order, fail-fast):
#   1. Clone (or fetch) the three repos at the validated branches:
#        suharvest/jetson-local-voice       branch qwen3tts-accurate-20260507
#        suharvest/qwen3-edgellm-jetson     branch main
#        suharvest/TensorRT-Edge-LLM        branch qwen3-tts-highperf-runtime-w8a16
#   2. Initialise EdgeLLM submodules.
#   3. cmake configure + build (-j1; the SM87/CuTe-DSL/EMBEDDED_TARGET
#      defaults are already in CMakeLists at HEAD).
#   4. Symbol-check the built plugin for the OLD W8A16 kernel set.
#   5. Download the HF artifact set + SHA-256 verify.
#   6. Build the slim docker image and start the service.
#   7. Wait for /health, then run verify_reproduction.sh (TTS loopback
#      + voice clone if a reference WAV is provided).
#
# Usage:
#   reproduce_qwen3_highperf.sh \
#       [--workspace ~/project] \
#       [--artifact-set orin-nx-highperf-2026-05-11] \
#       [--artifact-root /opt/models/qwen3-edgellm] \
#       [--service-port 18092] \
#       [--reference path/to/reference.wav] \
#       [--skip-build] [--skip-deploy] [--skip-docker] [--skip-verify]
#
# Environment overrides:
#   HF_ENDPOINT     (e.g. https://hf-mirror.com) — for blocked networks
#   CUDA_DIR        (default /usr/local/cuda-12.6)
#   CUDA_CTK_VER    (default 12.6)
#
# Exit codes:
#   0 — full pass
#   1 — verification check(s) failed (artifact / TTS / clone)
#   2 — build/clone/deploy aborted before verify
set -euo pipefail

WORKSPACE="${HOME}/project"
ARTIFACT_SET="orin-nx-highperf-2026-05-11"
ARTIFACT_ROOT="${QWEN3_ARTIFACT_ROOT:-/opt/models/qwen3-edgellm}"
SERVICE_PORT=18092
REFERENCE_WAV=""
EMBEDDING_FILE=""
SKIP_BUILD=0; SKIP_DEPLOY=0; SKIP_DOCKER=0; SKIP_VERIFY=0
CUDA_DIR="${CUDA_DIR:-/usr/local/cuda-12.6}"
CUDA_CTK_VER="${CUDA_CTK_VER:-12.6}"

while [ $# -gt 0 ]; do
  case "$1" in
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --artifact-set) ARTIFACT_SET="$2"; shift 2 ;;
    --artifact-root) ARTIFACT_ROOT="$2"; shift 2 ;;
    --service-port) SERVICE_PORT="$2"; shift 2 ;;
    --reference) REFERENCE_WAV="$2"; shift 2 ;;
    --embedding) EMBEDDING_FILE="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-deploy) SKIP_DEPLOY=1; shift ;;
    --skip-docker) SKIP_DOCKER=1; shift ;;
    --skip-verify) SKIP_VERIFY=1; shift ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

log()  { printf "\033[36m[%s]\033[0m %s\n" "$(date +%H:%M:%S)" "$*"; }
die()  { printf "\033[31m[%s] FAIL:\033[0m %s\n" "$(date +%H:%M:%S)" "$*" >&2; exit 2; }

EDGELLM="$WORKSPACE/TensorRT-Edge-LLM"
QEJ="$WORKSPACE/qwen3-edgellm-jetson"
JV="$WORKSPACE/jetson-voice"

# Optional commit-hash pinning. When set, each repo is checked out at the exact
# commit AFTER the branch-tip fetch — useful for release reproduction where
# branch tip may drift. Empty (default) keeps current branch-tip behavior.
EDGELLM_COMMIT="${EDGELLM_COMMIT:-}"
QWEN3_EDGELLM_JETSON_COMMIT="${QWEN3_EDGELLM_JETSON_COMMIT:-}"
JETSON_VOICE_COMMIT="${JETSON_VOICE_COMMIT:-}"

# ---------------------------------------------------------------------------
log "step 1/7: ensure three repos at the validated branches"
mkdir -p "$WORKSPACE"
declare -A REPOS=(
  ["$EDGELLM:qwen3-tts-highperf-runtime-w8a16"]="https://github.com/suharvest/TensorRT-Edge-LLM.git"
  ["$QEJ:main"]="https://github.com/suharvest/qwen3-edgellm-jetson.git"
  ["$JV:qwen3tts-accurate-20260507"]="https://github.com/suharvest/jetson-local-voice.git"
)
for spec in "${!REPOS[@]}"; do
  path="${spec%%:*}"
  branch="${spec##*:}"
  url="${REPOS[$spec]}"
  # Per-repo commit pin (empty → branch tip).
  case "$path" in
    "$EDGELLM") pin="$EDGELLM_COMMIT" ;;
    "$QEJ")     pin="$QWEN3_EDGELLM_JETSON_COMMIT" ;;
    "$JV")      pin="$JETSON_VOICE_COMMIT" ;;
    *)          pin="" ;;
  esac
  if [ ! -d "$path/.git" ]; then
    log "  cloning $url -> $path"
    git clone "$url" "$path" || die "clone $url"
  fi
  ( cd "$path"
    git fetch --quiet origin "$branch" 2>/dev/null || true
    if ! git rev-parse --verify "$branch" >/dev/null 2>&1; then
      git checkout -B "$branch" "origin/$branch" >/dev/null 2>&1 || die "checkout $branch in $path"
    else
      git checkout "$branch" >/dev/null 2>&1 || die "checkout $branch in $path"
    fi
    # Best-effort fast-forward; don't blow up if user has local edits.
    git pull --ff-only --quiet 2>/dev/null || log "  $path: pull skipped (local edits or no upstream)"
    if [ -n "$pin" ]; then
      log "  $path: pinning to $pin"
      git checkout --quiet "$pin" || die "pin checkout $pin in $path"
    fi
    echo "  $path @ $(git rev-parse --short HEAD) ($branch${pin:+, pinned})"
  )
done

# ---------------------------------------------------------------------------
if [ $SKIP_BUILD -eq 0 ]; then
  log "step 2/7: EdgeLLM submodules"
  ( cd "$EDGELLM" && git submodule update --init --recursive --quiet ) || die "submodule init"

  log "step 3/7: cmake configure + build"
  export CUDACXX="$CUDA_DIR/bin/nvcc"
  ( cd "$EDGELLM"
    if [ ! -f build/CMakeCache.txt ]; then
      cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRT_PACKAGE_DIR=/usr \
        -DCUDA_DIR="$CUDA_DIR" \
        -DCUDA_CTK_VERSION="$CUDA_CTK_VER" \
        >/dev/null
    fi
    # exampleUtils device-link can get stale on partial rebuilds.
    rm -f build/examples/utils/libexampleUtils.a \
          build/examples/utils/CMakeFiles/exampleUtils.dir/cmake_device_link.o
    cmake --build build --target edgellmCore NvInfer_edgellm_plugin qwen3_tts_worker -j1
  ) || die "cmake build"

  PLUGIN="$EDGELLM/build/libNvInfer_edgellm_plugin.so"
  WORKER="$EDGELLM/build/examples/omni/qwen3_tts_worker"
  [ -f "$PLUGIN" ] || die "plugin missing: $PLUGIN"
  [ -f "$WORKER" ] || die "worker missing: $WORKER"

  log "step 4/7: sanity-check W8A16 kernel symbols in plugin"
  EXPECTED_SYMS=(
    w8a16_hmma_m16n16k16_kernel
    w8a16_m1_output_k_kernel
    w8a16_per_output_output_k_reference_kernel
    w8a16_per_output_reference_kernel
    w8a16_small_m_tiled_kernel
  )
  SYMS=$(nm "$PLUGIN" 2>/dev/null | grep -oE 'w8a16_[a-z0-9_]+_kernel' | sort -u)
  for s in "${EXPECTED_SYMS[@]}"; do
    grep -qx "$s" <<<"$SYMS" || die "plugin missing $s — wrong source revision (re-check git pull)"
  done
  if grep -qE '^w8a16_per_output_tiled(_pair_k)?_kernel$' <<<"$SYMS"; then
    die "plugin contains regressed _tiled / _tiled_pair_k kernels; source needs the OLD W8A16 restore"
  fi
  log "  plugin has the 5 expected W8A16 kernels"
fi

# ---------------------------------------------------------------------------
if [ $SKIP_DEPLOY -eq 0 ]; then
  log "step 5/7: deploy HF artifact set $ARTIFACT_SET → $ARTIFACT_ROOT"
  python3 "$QEJ/scripts/deploy_qwen3_artifacts.py" \
    --set "$ARTIFACT_SET" --root "$ARTIFACT_ROOT" \
    || die "artifact deploy"
  python3 "$QEJ/scripts/deploy_qwen3_artifacts.py" \
    --set "$ARTIFACT_SET" --root "$ARTIFACT_ROOT" --verify-sha256 \
    || die "artifact SHA-256 verify"
fi

# ---------------------------------------------------------------------------
if [ $SKIP_DOCKER -eq 0 ]; then
  log "step 6/7: build slim docker image + start service on port $SERVICE_PORT"
  ( cd "$JV"
    docker build -f Dockerfile.slim.qwen3 -t jetson-voice-qwen3:slim . >/dev/null
  ) || die "docker build"
  docker rm -f jetson_voice_slim >/dev/null 2>&1 || true
  EDGE_LLM_BIN_DIR="$EDGELLM/build"
  JV_WORKER_BIN_DIR="$JV/build/edgellm_voice_worker/workers"
  # qwen3_asr_worker is optional; tolerate missing dir
  EXTRA_MOUNTS=()
  if [ -d "$JV_WORKER_BIN_DIR" ]; then
    EXTRA_MOUNTS+=(-v "$JV_WORKER_BIN_DIR":/opt/jv-workers:ro \
                   -e JETSON_VOICE_WORKER_BUILD=/opt/jv-workers \
                   -e EDGE_LLM_ASR_WORKER_BIN=/opt/jv-workers/qwen3_asr_worker \
                   -e EDGE_LLM_ASR_PLUGIN_PATH=/opt/edgellm-bin/libNvInfer_edgellm_plugin.so)
  fi
  docker run -d --name jetson_voice_slim --runtime nvidia --ipc host \
    -p "${SERVICE_PORT}:8000" \
    -v /usr/local/cuda/lib64:/host-cuda:ro \
    -v /usr/lib/aarch64-linux-gnu:/host-libs:ro \
    -v "$EDGE_LLM_BIN_DIR":/opt/edgellm-bin:ro \
    -v "$ARTIFACT_ROOT":/opt/models/qwen3-edgellm:ro \
    -v "$QEJ":/opt/qwen3-edgellm-jetson:ro \
    "${EXTRA_MOUNTS[@]}" \
    -e LD_LIBRARY_PATH=/host-libs:/host-cuda:/usr/local/lib/python3.10/dist-packages/onnxruntime/capi \
    -e JETSON_VOICE_PROFILE=multilanguage-qwen-highperf-nx \
    -e QWEN3_ARTIFACT_ROOT=/opt/models/qwen3-edgellm \
    -e QWEN3_EDGELLM_JETSON_ROOT=/opt/qwen3-edgellm-jetson \
    -e EDGE_LLM_BASE=/opt/edgellm-bin -e EDGE_LLM_BUILD_DIR=. \
    -e EDGE_LLM_TTS_WORKER_BIN=/opt/edgellm-bin/examples/omni/qwen3_tts_worker \
    -e EDGELLM_PLUGIN_PATH=/opt/edgellm-bin/libNvInfer_edgellm_plugin.so \
    -e EDGE_LLM_TTS_STATEFUL_CODE2WAV=1 \
    -e EDGE_LLM_TTS_STATEFUL_CODE2WAV_ENGINE_DIR=/opt/models/qwen3-edgellm/engines/orin-nx/highperf/code2wav_stateful \
    jetson-voice-qwen3:slim >/dev/null \
    || die "docker run"

  log "  waiting for /health (up to 180s)..."
  for i in $(seq 1 90); do
    if curl -sf -o /dev/null "http://localhost:${SERVICE_PORT}/health"; then
      log "  /health up after ${i}s"
      break
    fi
    sleep 2
  done
  if ! curl -sf -o /dev/null "http://localhost:${SERVICE_PORT}/health"; then
    log "  container logs (last 30):"
    docker logs jetson_voice_slim 2>&1 | tail -30
    die "service did not become healthy"
  fi
fi

# ---------------------------------------------------------------------------
if [ $SKIP_VERIFY -eq 0 ]; then
  log "step 7/7: run verify_reproduction.sh"
  ARGS=(
    --plugin "$EDGELLM/build/libNvInfer_edgellm_plugin.so"
    --artifact-root "$ARTIFACT_ROOT"
    --set "$ARTIFACT_SET"
    --service-url "http://localhost:${SERVICE_PORT}"
  )
  [ -n "$REFERENCE_WAV" ] && ARGS+=(--reference "$REFERENCE_WAV")
  [ -n "$EMBEDDING_FILE" ] && ARGS+=(--embedding "$EMBEDDING_FILE")
  bash "$QEJ/scripts/verify_reproduction.sh" "${ARGS[@]}"
  RC=$?
  if [ $RC -ne 0 ]; then exit $RC; fi
fi

# -----------------------------------------------------------------------
# Smoke test: confirm the freshly-deployed worker can complete a single
# one-shot ASR round-trip against the docker service. Catches "everything
# compiled but engine/plugin ABI mismatch / weights missing" failure modes
# that the artifact verify alone doesn't catch.
# -----------------------------------------------------------------------
if [ $SKIP_VERIFY -eq 0 ]; then
  log "smoke: one-shot ASR via deployed service on :$SERVICE_PORT"
  smoke_audio=$(find "$QEJ/docs/audio-evidence" \
      \( -name 'nano-official-*.wav' -o -name 'nx-highperf-*.wav' \) \
      2>/dev/null | head -1)
  if [ -n "$smoke_audio" ]; then
    set +e
    smoke_resp=$(curl -fsS -m 30 -X POST -F "file=@$smoke_audio" \
        "http://localhost:${SERVICE_PORT}/asr" 2>&1)
    smoke_rc=$?
    set -e
    if [ $smoke_rc -ne 0 ] || ! echo "$smoke_resp" | python3 -c "import sys,json
d=json.load(sys.stdin)
t=(d.get('text') or d.get('transcript') or '').strip()
assert t, 'empty text'" 2>/dev/null; then
      log "smoke: FAIL — service did not return non-empty text"
      log "smoke: response: $smoke_resp"
      exit 1
    fi
    log "smoke: PASS — '${smoke_resp:0:120}...'"
  else
    log "smoke: SKIPPED — no audio sample found under docs/audio-evidence/"
  fi
fi

log "All reproduction checks passed. Service is live at http://localhost:${SERVICE_PORT}"
echo "REPRODUCE_PASS"
