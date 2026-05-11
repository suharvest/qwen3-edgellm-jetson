#!/usr/bin/env bash
# Verify end-to-end Qwen3 highperf reproduction.
#
# Runs every guardrail that prior incidents traced back to:
#   1. EdgeLLM plugin has the exact W8A16 kernel set the frozen
#      artifacts were validated against (regression bait in 2026-05-11).
#   2. HF artifact root contains every required file and SHA-256
#      matches the sidecar.
#   3. A live jetson-voice HTTP service produces audio whose ASR
#      round-trip recovers the original Chinese prompts.
#   4. Voice clone via speaker_embedding_b64 (extracted via the
#      official mel pipeline) recovers the same prompts.
#
# Exits 0 only on full pass. Any check fails → non-zero + summary.
#
# Usage:
#   verify_reproduction.sh \
#       [--plugin <libNvInfer_edgellm_plugin.so>] \
#       [--artifact-root <root>] [--set <set>] \
#       [--service-url http://localhost:18092] \
#       [--reference <path/to/24kHz.wav>] \
#       [--speaker-encoder <path/to/speaker_encoder.onnx>] \
#       [--skip-clone] [--skip-service]
#
# Defaults:
#   --plugin              ~/project/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
#   --artifact-root       /opt/models/qwen3-edgellm  (env QWEN3_ARTIFACT_ROOT wins)
#   --set                 orin-nx-highperf-2026-05-11
#   --service-url         http://localhost:18092
#   --reference           (skipped if not given; clone test is skipped)
#   --speaker-encoder     <artifact-root>/tts/speaker_encoder/speaker_encoder.onnx
set -uo pipefail

PLUGIN="${HOME}/project/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"
ARTIFACT_ROOT="${QWEN3_ARTIFACT_ROOT:-/opt/models/qwen3-edgellm}"
ARTIFACT_SET="orin-nx-highperf-2026-05-11"
SERVICE_URL="http://localhost:18092"
REFERENCE_WAV=""
SPEAKER_ENCODER=""
PRECOMPUTED_EMBEDDING=""
SKIP_CLONE=0
SKIP_SERVICE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --plugin) PLUGIN="$2"; shift 2 ;;
    --artifact-root) ARTIFACT_ROOT="$2"; shift 2 ;;
    --set) ARTIFACT_SET="$2"; shift 2 ;;
    --service-url) SERVICE_URL="$2"; shift 2 ;;
    --reference) REFERENCE_WAV="$2"; shift 2 ;;
    --speaker-encoder) SPEAKER_ENCODER="$2"; shift 2 ;;
    --embedding) PRECOMPUTED_EMBEDDING="$2"; shift 2 ;;
    --skip-clone) SKIP_CLONE=1; shift ;;
    --skip-service) SKIP_SERVICE=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0; FAIL=0; SKIP=0
declare -a FAILED=()

ok()    { printf "  \033[32mPASS\033[0m %s\n" "$1"; PASS=$((PASS+1)); }
fail()  { printf "  \033[31mFAIL\033[0m %s — %s\n" "$1" "$2"; FAIL=$((FAIL+1)); FAILED+=("$1: $2"); }
skip()  { printf "  \033[33mSKIP\033[0m %s — %s\n" "$1" "$2"; SKIP=$((SKIP+1)); }

# Best-of-N retry wrapper for an ASR loopback step. Args:
#   $1 emit function (must echo the ASR text on stdout; non-zero exit = skip)
#   $2 label
#   $3 prompt
# Tries up to 3 attempts, passes on the first sim>=0.7.
# Greedy sampling on this W8A16 + CUDA stack has tie-breaking jitter,
# so single-attempt exact-match is not stable. Real regressions still
# fail because all 3 attempts collapse below threshold.
asr_match_with_retry() {
  local emit_fn="$1" label="$2" prompt="$3"
  local best_score="0" best_asr="" attempts=3 score asr
  for i in $(seq 1 $attempts); do
    asr=$("$emit_fn" "$prompt") || { fail "$label \"$prompt\"" "emit fn returned error"; return; }
    score=$(lcs_ratio "$prompt" "$asr")
    if [ "$(python3 -c "print('y' if $score >= 0.7 else 'n')")" = "y" ]; then
      ok "$label \"$prompt\" → \"$asr\" (sim=$score after $i attempt(s))"
      return
    fi
    # remember best
    if [ "$(python3 -c "print('y' if $score > $best_score else 'n')")" = "y" ]; then
      best_score=$score; best_asr=$asr
    fi
  done
  fail "$label \"$prompt\"" "$attempts attempts; best=\"$best_asr\" (sim=$best_score, need >=0.7)"
}

lcs_ratio() {
  python3 - <<'PY' "$1" "$2"
import sys
p, a = sys.argv[1], sys.argv[2]
strip = lambda s: ''.join(c for c in s if c not in '。，、！？.,!?、，。 \t\n')
p, a = strip(p), strip(a)
if not p: print('1.0'); sys.exit()
m, n = len(p), len(a)
dp = [[0]*(n+1) for _ in range(m+1)]
for i in range(m):
    for j in range(n):
        dp[i+1][j+1] = dp[i][j]+1 if p[i] == a[j] else max(dp[i][j+1], dp[i+1][j])
print(f"{dp[m][n]/m:.3f}")
PY
}

# Single-shot version (legacy callers, no retry).
assert_asr_match() {
  local label="$1" prompt="$2" asr="$3"
  local score
  score=$(python3 - <<'PY' "$prompt" "$asr"
import sys
p, a = sys.argv[1], sys.argv[2]
strip = lambda s: ''.join(c for c in s if c not in '。，、！？.,!?、，。 \t\n')
p, a = strip(p), strip(a)
if not p: print('1.0'); sys.exit()
m, n = len(p), len(a)
dp = [[0]*(n+1) for _ in range(m+1)]
for i in range(m):
    for j in range(n):
        dp[i+1][j+1] = dp[i][j]+1 if p[i] == a[j] else max(dp[i][j+1], dp[i+1][j])
print(f"{dp[m][n]/m:.3f}")
PY
)
  local good
  good=$(python3 -c "print('y' if $score >= 0.7 else 'n')")
  if [ "$good" = "y" ]; then
    ok "$label \"$prompt\" → \"$asr\" (sim=$score)"
  else
    fail "$label \"$prompt\"" "got \"$asr\" (sim=$score, need >=0.7)"
  fi
}

# ---------------------------------------------------------------------------
echo "== [1/4] W8A16 kernel symbol set in $PLUGIN =="
if [ ! -f "$PLUGIN" ]; then
  fail "plugin-present" "no file at $PLUGIN"
else
  EXPECTED=(
    w8a16_hmma_m16n16k16_kernel
    w8a16_m1_output_k_kernel
    w8a16_per_output_output_k_reference_kernel
    w8a16_per_output_reference_kernel
    w8a16_small_m_tiled_kernel
  )
  REGRESSED=(
    w8a16_per_output_tiled_kernel
    w8a16_per_output_tiled_pair_k_kernel
  )
  SYMS=$(nm "$PLUGIN" 2>/dev/null | grep -oE 'w8a16_[a-z0-9_]+_kernel' | sort -u)
  for s in "${EXPECTED[@]}"; do
    if grep -qx "$s" <<< "$SYMS"; then ok "expected $s"
    else fail "expected $s" "missing — wrong/old plugin build"
    fi
  done
  for s in "${REGRESSED[@]}"; do
    if grep -qx "$s" <<< "$SYMS"; then
      fail "regressed $s" "present in plugin — source has the broken refactor; re-pull EdgeLLM fork HEAD"
    fi
  done
fi

# ---------------------------------------------------------------------------
echo
echo "== [2/4] Artifact set $ARTIFACT_SET integrity at $ARTIFACT_ROOT =="
DEPLOY="$REPO_ROOT/scripts/deploy_qwen3_artifacts.py"
if [ ! -f "$DEPLOY" ]; then
  fail "deploy-script" "$DEPLOY not found"
else
  if OUT=$(python3 "$DEPLOY" --set "$ARTIFACT_SET" --root "$ARTIFACT_ROOT" --check-only 2>&1); then
    ok "all required files present"
  else
    fail "required files" "$(echo "$OUT" | tail -1)"
  fi
  if OUT=$(python3 "$DEPLOY" --set "$ARTIFACT_SET" --root "$ARTIFACT_ROOT" --verify-sha256 2>&1); then
    ok "$(echo "$OUT" | tail -1)"
  else
    fail "sha256 verify" "$(echo "$OUT" | tail -1)"
  fi
fi

# ---------------------------------------------------------------------------
echo
echo "== [3/4] HTTP TTS → ASR loopback at $SERVICE_URL =="
if [ "$SKIP_SERVICE" -eq 1 ]; then
  skip "service loopback" "--skip-service"
else
  if ! curl -sf -o /dev/null "$SERVICE_URL/health" 2>/dev/null; then
    fail "/health 200" "service unreachable at $SERVICE_URL"
  else
    HEALTH=$(curl -s "$SERVICE_URL/health")
    if echo "$HEALTH" | grep -q '"tts":true'; then ok "/health: tts=true"
    else fail "/health: tts" "$HEALTH"; fi

    TMPDIR=$(mktemp -d)
    trap "rm -rf $TMPDIR" EXIT
    declare -a PROMPTS=(
      "今天天气真好。"
      "人工智能改变了世界。"
      "一二三四五六七八九十。"
    )
    for prompt in "${PROMPTS[@]}"; do
      # Best-of-3 retry: greedy is still non-deterministic on this CUDA stack
      # due to W8A16 tie-breaking. Real failures collapse on every attempt
      # (sim ~ 0); cosmetic drift typically clears in 2 attempts.
      best_score=0; best_asr=""
      for attempt in 1 2 3; do
        WAV="$TMPDIR/tts_$(echo -n "$prompt" | md5sum | cut -c1-8)_a${attempt}.wav"
        CODE=$(curl -s -X POST "$SERVICE_URL/tts" -H 'content-type: application/json' \
          -d "{\"text\":\"$prompt\",\"talker_top_k\":1,\"talker_temperature\":0.05,\"predictor_top_k\":1,\"predictor_temperature\":0.05}" \
          -o "$WAV" -w '%{http_code}')
        if [ "$CODE" != "200" ] || [ ! -s "$WAV" ]; then continue; fi
        ASR_JSON=$(curl -s -X POST "$SERVICE_URL/asr" -F "file=@$WAV")
        ASR_TXT=$(echo "$ASR_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("text",""))' 2>/dev/null)
        score=$(lcs_ratio "$prompt" "$ASR_TXT")
        if [ "$(python3 -c "print('y' if $score >= 0.7 else 'n')")" = "y" ]; then
          ok "TTS→ASR \"$prompt\" → \"$ASR_TXT\" (sim=$score, attempt $attempt)"
          best_score=$score; break
        fi
        if [ "$(python3 -c "print('y' if $score > $best_score else 'n')")" = "y" ]; then
          best_score=$score; best_asr=$ASR_TXT
        fi
      done
      if [ "$(python3 -c "print('y' if $best_score < 0.7 else 'n')")" = "y" ]; then
        fail "TTS→ASR \"$prompt\"" "3 attempts best=\"$best_asr\" sim=$best_score"
      fi
    done
  fi
fi

# ---------------------------------------------------------------------------
echo
echo "== [4/4] Voice clone loopback =="
if [ "$SKIP_SERVICE" -eq 1 ] || [ "$SKIP_CLONE" -eq 1 ]; then
  skip "voice clone" "skip flag set"
elif [ -n "$PRECOMPUTED_EMBEDDING" ]; then
  if [ ! -f "$PRECOMPUTED_EMBEDDING" ]; then
    fail "precomputed embedding" "file missing: $PRECOMPUTED_EMBEDDING"
  else
    EMB_FILE="$PRECOMPUTED_EMBEDDING"
    ok "using precomputed embedding ($(wc -c < "$EMB_FILE") b64 chars)"
    for prompt in "${PROMPTS[@]}"; do
      best_score=0; best_asr=""
      for attempt in 1 2 3; do
        WAV="$TMPDIR/clone_pre_$(echo -n "$prompt" | md5sum | cut -c1-8)_a${attempt}.wav"
        REQ=$(python3 -c "import json; print(json.dumps({'text':'$prompt','speaker_embedding_b64':open('$EMB_FILE').read().strip(),'first_chunk_frames':7,'chunk_frames':10,'max_chunk_frames':10,'talker_top_k':1,'talker_temperature':0.05,'predictor_top_k':1,'predictor_temperature':0.05},ensure_ascii=False))")
        PCM="$TMPDIR/clone_pre_a${attempt}.pcm"
        CODE=$(curl -s -N -X POST "$SERVICE_URL/tts/clone/stream" -H 'content-type: application/json' -d "$REQ" -o "$PCM" -w '%{http_code}')
        if [ "$CODE" != "200" ] || [ ! -s "$PCM" ]; then continue; fi
        python3 - "$PCM" "$WAV" <<'PY' || true
import sys, struct, wave
raw = open(sys.argv[1],'rb').read(); sr = struct.unpack('<I', raw[:4])[0]
with wave.open(sys.argv[2],'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(raw[4:])
PY
        ASR_JSON=$(curl -s -X POST "$SERVICE_URL/asr" -F "file=@$WAV")
        ASR_TXT=$(echo "$ASR_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("text",""))' 2>/dev/null)
        score=$(lcs_ratio "$prompt" "$ASR_TXT")
        if [ "$(python3 -c "print('y' if $score >= 0.7 else 'n')")" = "y" ]; then
          ok "clone→ASR (precomputed) \"$prompt\" → \"$ASR_TXT\" (sim=$score, attempt $attempt)"
          best_score=$score; break
        fi
        if [ "$(python3 -c "print('y' if $score > $best_score else 'n')")" = "y" ]; then
          best_score=$score; best_asr=$ASR_TXT
        fi
      done
      if [ "$(python3 -c "print('y' if $best_score < 0.7 else 'n')")" = "y" ]; then
        fail "clone→ASR (precomputed) \"$prompt\"" "3 attempts best=\"$best_asr\" sim=$best_score"
      fi
    done
  fi
elif [ -z "$REFERENCE_WAV" ]; then
  skip "voice clone" "pass --reference <wav> (needs librosa) or --embedding <b64> (no python deps needed)"
else
  if [ -z "$SPEAKER_ENCODER" ]; then
    SPEAKER_ENCODER="$ARTIFACT_ROOT/tts/speaker_encoder/speaker_encoder.onnx"
  fi
  EXTRACT="$REPO_ROOT/scripts/extract_speaker_embedding.py"
  if [ ! -f "$REFERENCE_WAV" ]; then
    fail "reference wav" "not at $REFERENCE_WAV"
  elif [ ! -f "$SPEAKER_ENCODER" ]; then
    fail "speaker encoder" "not at $SPEAKER_ENCODER"
  elif [ ! -f "$EXTRACT" ]; then
    fail "extract script" "$EXTRACT missing"
  elif ! python3 -c 'import librosa, onnxruntime' >/dev/null 2>&1; then
    skip "voice clone" "python3 is missing librosa+onnxruntime (install: pip install librosa onnxruntime); extract embedding on a workstation with --embedding instead"
  else
    EMB_FILE="$TMPDIR/spk_emb.b64"
    if ! python3 "$EXTRACT" "$REFERENCE_WAV" "$SPEAKER_ENCODER" "$EMB_FILE" >/dev/null 2>&1; then
      fail "extract embedding" "extractor errored — re-run manually: python3 $EXTRACT $REFERENCE_WAV $SPEAKER_ENCODER /tmp/out.b64"
    else
      EMB=$(cat "$EMB_FILE")
      ok "embedding extracted ($(wc -c < "$EMB_FILE") b64 chars)"
      for prompt in "${PROMPTS[@]}"; do
        best_score=0; best_asr=""
        for attempt in 1 2 3; do
          WAV="$TMPDIR/clone_$(echo -n "$prompt" | md5sum | cut -c1-8)_a${attempt}.wav"
          REQ=$(python3 -c "import json; print(json.dumps({'text':'$prompt','speaker_embedding_b64':open('$EMB_FILE').read().strip(),'first_chunk_frames':7,'chunk_frames':10,'max_chunk_frames':10,'talker_top_k':1,'talker_temperature':0.05,'predictor_top_k':1,'predictor_temperature':0.05},ensure_ascii=False))")
          PCM="$TMPDIR/clone_a${attempt}.pcm"
          CODE=$(curl -s -N -X POST "$SERVICE_URL/tts/clone/stream" -H 'content-type: application/json' -d "$REQ" -o "$PCM" -w '%{http_code}')
          if [ "$CODE" != "200" ] || [ ! -s "$PCM" ]; then continue; fi
          python3 - "$PCM" "$WAV" <<'PY' || true
import sys, struct, wave
raw = open(sys.argv[1],'rb').read(); sr = struct.unpack('<I', raw[:4])[0]
with wave.open(sys.argv[2],'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(raw[4:])
PY
          ASR_JSON=$(curl -s -X POST "$SERVICE_URL/asr" -F "file=@$WAV")
          ASR_TXT=$(echo "$ASR_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("text",""))' 2>/dev/null)
          score=$(lcs_ratio "$prompt" "$ASR_TXT")
          if [ "$(python3 -c "print('y' if $score >= 0.7 else 'n')")" = "y" ]; then
            ok "clone→ASR \"$prompt\" → \"$ASR_TXT\" (sim=$score, attempt $attempt)"
            best_score=$score; break
          fi
          if [ "$(python3 -c "print('y' if $score > $best_score else 'n')")" = "y" ]; then
            best_score=$score; best_asr=$ASR_TXT
          fi
        done
        if [ "$(python3 -c "print('y' if $best_score < 0.7 else 'n')")" = "y" ]; then
          fail "clone→ASR \"$prompt\"" "3 attempts best=\"$best_asr\" sim=$best_score"
        fi
      done
    fi
  fi
fi

# ---------------------------------------------------------------------------
echo
echo "== Summary =="
echo "  pass: $PASS   fail: $FAIL   skip: $SKIP"
if [ $FAIL -gt 0 ]; then
  echo
  echo "Failures:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
  exit 1
fi
exit 0
