# Qwen3 EdgeLLM Jetson

Qwen3 ASR/TTS deployment, build, and validation toolkit for Jetson Orin devices using TensorRT-Edge-LLM.

## Quick start — one shot on Orin NX

```bash
git clone https://github.com/suharvest/qwen3-edgellm-jetson.git
bash qwen3-edgellm-jetson/scripts/reproduce_qwen3_highperf.sh
# add: --reference path/to/24kHz_mono.wav   to also verify voice clone
```

`scripts/reproduce_qwen3_highperf.sh` is idempotent and handles the whole chain: clones the three repos at the validated branches, builds EdgeLLM, downloads + SHA-256-verifies the HF artifact set, builds the slim docker image, brings the service up, then runs `scripts/verify_reproduction.sh` which gates on (1) W8A16 plugin symbol set, (2) artifact integrity, (3) HTTP TTS→ASR loopback on three Chinese prompts (LCS-similarity ≥ 0.7 across up to 3 retries), and (4) voice clone via a real reference WAV. Exit 0 means the entire chain is verified.

If any step fails the script prints which check failed and why — no need to debug from logs by hand.

Prerequisites: Jetson Orin NX with JetPack 6 (TensorRT 10.3.0.30, CUDA 12.6), docker + `--runtime nvidia`, ~10 GB free disk for HF artifacts.

## Architecture

This repository is the Qwen-specific companion to `jetson-voice`:

- `jetson-voice` remains the deployable product service with API, frontend/backend selection, Docker, and device deployment.
- This repo owns Qwen3 ASR/TTS export/build/runtime glue, performance scripts, validation gates, and lessons learned.
- Runtime TensorRT/embedding artifacts live in the Hugging Face model repo `harvestsu/qwen3-edgellm-jetson-artifacts`; ONNX files are reproducible intermediate build products generated locally from official Qwen weights.
- TensorRT-Edge-LLM runtime changes live in the fork `suharvest/TensorRT-Edge-LLM`.

## Runtime Profiles

Two Qwen profiles are maintained:

- `official`: minimal-diff EdgeLLM-compatible path for correctness and upstream review.
- `highperf`: product path for low-latency Qwen3 ASR + Qwen3 TTS dual residency on Jetson Orin.

Jetson Voice consumes these via JSON profiles copied under `configs/profiles/` and the artifact manifest under `deploy/artifacts/qwen3_manifest.json`. `multilanguage-qwen-highperf` targets the Nano artifact set; `multilanguage-qwen-highperf-nx` targets NX-native engines.

## EdgeLLM Fork

Use these TensorRT-Edge-LLM fork branches:

- `official-qwen3-tts-upstream-runtime`: minimal-diff correctness/runtime branch for upstream review.
- `qwen3-tts-highperf-runtime-w8a16`: product high-performance branch used by the current Orin highperf artifacts.

The highperf branch includes the runtime pieces required by the measured path: explicit Qwen3-TTS backend without duplicate generic Talker load, W8A16 plugin/runtime support, CP runtime optimizations, GPU CP kernels, stateful Code2Wav runner, and optional Code2Wav timing profile via `QWEN3_TTS_CODE2WAV_PROFILE=1`.

Do not deploy highperf artifacts against EdgeLLM `main`. A correct checkout must contain:

- `cpp/plugins/w8A16LinearPlugin/`
- `cpp/kernels/qwen3TtsCpKernels/`
- `cpp/multimodal/statefulCode2WavRunner.*`
- highperf Qwen3-TTS worker support for `speaker_embedding_b64`

After changing CUDA sources, clean stale device-link outputs before rebuilding examples:

```bash
rm -f build*/examples/utils/libexampleUtils.a \
      build*/examples/utils/CMakeFiles/exampleUtils.dir/cmake_device_link.o
```

## Contents

Scripts (in priority order):

- `scripts/reproduce_qwen3_highperf.sh` — **one-shot orchestrator**: clone → build → deploy → docker → verify.
- `scripts/verify_reproduction.sh` — gate-only checks (symbol set / artifact SHA-256 / TTS+ASR loopback / clone). Reusable in CI.
- `scripts/deploy_qwen3_artifacts.py` — HF artifact downloader with `--verify-sha256` and `--generate-sidecar`.
- `scripts/extract_speaker_embedding.py` — 24 kHz mono WAV → 1024-d speaker embedding (matches the official Qwen3-TTS mel pipeline; requires `librosa` + `onnxruntime`).
- `scripts/export_qwen3_{asr,tts}_onnx.sh` — official-weight → ONNX export.
- `scripts/build_qwen3_*.{py,sh}` — engine build helpers (historical paths; for re-export only).

Repo content:

- `native/edgellm_voice_worker/` — resident ASR/TTS worker binaries used by Jetson Voice.
- `docs/performance/` — frozen perf records and end-to-end loopback evidence with listenable WAVs.
- `docs/audio-evidence/` — passing and broken WAVs referenced by the perf docs.
- `docs/reproduce-from-zero.md` — step-by-step manual fallback when the one-shot script needs to be debugged.
- `docs/issues/` — open issues with the dev hand-off package.
- `docs/plans/` — implementation notes and negative results that should not be rediscovered.
- `docs/export-from-official-weights.md` — official Qwen3 weight → ONNX export guide.
- `AGENTS.md` — concise operating guide for coding agents working in this repo.
- `deploy/artifacts/qwen3_manifest.json` — HF artifact manifest (file list per artifact set).
- `deploy/artifacts/qwen3_checksums.json` — SHA-256 + byte-size sidecar (per file, per set).
- `configs/profiles/` — Jetson Voice deployment profiles (all use `${QWEN3_ARTIFACT_ROOT}` so they're portable).

## ONNX Export

Users can start from official Qwen3 ASR/TTS Hugging Face snapshots and generate
ONNX locally:

```bash
bash scripts/setup_trt_export_env.sh
scripts/export_qwen3_asr_onnx.sh --model-dir /models/Qwen3-ASR-0.6B --out /tmp/qwen3-asr-onnx
scripts/export_qwen3_tts_onnx.sh --model-dir /models/Qwen3-TTS-0.6B --out /tmp/qwen3-tts-onnx
```

See `docs/export-from-official-weights.md` for the uv environment, Qwen package
dependencies, and highperf post-processing details.

## Artifact Repo

HF repo: <https://huggingface.co/harvestsu/qwen3-edgellm-jetson-artifacts>

Expected artifact layout and required files are recorded in `deploy/artifacts/qwen3_manifest.json`. The shared `tts/tokenizer/` directory must include `tokenizer.json`, `tokenizer_config.json`, and `processed_chat_template.json`; missing sidecars break the C++ tokenizer load on a from-zero device.

Runtime binaries and plugins are built from the EdgeLLM fork or delivered by the runtime image. The model artifact repo intentionally stores engines and model sidecars, not a random local plugin copied from `/tmp`.

Current HF publication status:

| Artifact set | Status | Notes |
|---|---|---|
| `orin-nano-highperf-2026-05-10` | complete | Product highperf Nano artifact set. |
| `orin-nx-highperf-2026-05-11` | complete | Product highperf NX-native artifact set. |
| `orin-nano-official-2026-05-10` | complete | Official/minimal Nano artifact set. |

## Jetson Voice Integration

Preferred: let `scripts/reproduce_qwen3_highperf.sh` (above) build and start everything. For manual deployments of an existing image:

```bash
JETSON_VOICE_PROFILE=multilanguage-qwen-highperf-nx \
QWEN3_HF_REPO_ID=harvestsu/qwen3-edgellm-jetson-artifacts \
docker compose -f deploy/docker-compose.yml up -d
```

(`multilanguage-qwen-highperf` for Nano, `multilanguage-qwen-highperf-nx` for NX.)

Sanity-check the running service in one line:

```bash
bash scripts/verify_reproduction.sh \
    --plugin /opt/edgellm-bin/libNvInfer_edgellm_plugin.so \
    --artifact-root /opt/models/qwen3-edgellm \
    --service-url http://localhost:18092 \
    [--embedding /tmp/precomputed_speaker_emb.b64]
```

## Voice clone

Pre-extract the speaker embedding on a workstation (librosa + onnxruntime) once per voice and pass the base64 to the worker. Don't run the speaker encoder per request.

```bash
python3 scripts/extract_speaker_embedding.py \
    reference.wav speaker_encoder.onnx speaker_emb.b64
curl -X POST http://localhost:18092/tts/clone/stream \
    -H 'content-type: application/json' \
    -d "{\"text\":\"中文文本\",\"speaker_embedding_b64\":\"$(cat speaker_emb.b64)\",\"first_chunk_frames\":7,\"chunk_frames\":10,\"max_chunk_frames\":10}" \
    -o clone.pcm
```

The mel pipeline inside `extract_speaker_embedding.py` matches the official Qwen3-TTS `modeling_qwen3_tts.mel_spectrogram` exactly (magnitude not power, librosa slaney-norm mel, reflect-pad `(n_fft-hop)/2`, `log(clip(x, 1e-5, None))`). Skip any of those four and the Talker collapses to filler tokens (see `docs/performance/qwen3-orin-nx-voice-clone-pass-2026-05-11.md` for the diff).

## Current Baseline (2026-05-11)

End-to-end loopback evidence in `docs/performance/qwen3-orin-nx-loopback-pass-2026-05-11.md` and `docs/performance/qwen3-orin-nx-voice-clone-pass-2026-05-11.md`. Both report exact-match ASR on three Chinese prompts via the slim docker container on Orin NX. Performance numbers in `docs/performance/qwen3-orin-profiles-2026-05-10.md` (V2V `EOS → first audio` 611–637 ms).
