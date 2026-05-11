# Qwen3 EdgeLLM Jetson

Qwen3 ASR/TTS deployment, build, and validation toolkit for Jetson Orin devices using TensorRT-Edge-LLM.

This repository is the Qwen-specific companion to `jetson-voice`:

- `jetson-voice` remains the deployable product service with API, frontend/backend selection, Docker, and device deployment.
- This repo owns Qwen3 ASR/TTS export/build/runtime glue, performance scripts, validation gates, and lessons learned.
- Large ONNX/TensorRT/embedding artifacts live in the Hugging Face model repo `harvestsu/qwen3-edgellm-jetson-artifacts`.

## Runtime Profiles

Two Qwen profiles are maintained:

- `official`: minimal-diff EdgeLLM-compatible path for correctness and upstream review.
- `highperf`: product path for low-latency Qwen3 ASR + Qwen3 TTS dual residency on Jetson Orin.

Jetson Voice consumes these via JSON profiles copied under `configs/profiles/` and the artifact manifest under `deploy/artifacts/qwen3_manifest.json`.

## Contents

- `native/edgellm_voice_worker/`: resident ASR/TTS worker binaries used by Jetson Voice.
- `scripts/`: export, engine build, quantization, stateful Code2Wav, quality gate, and V2V scripts.
- `docs/performance/`: frozen performance records across Orin Nano/NX.
- `docs/plans/`: implementation notes and negative results that should not be rediscovered.
- `deploy/artifacts/qwen3_manifest.json`: HF artifact manifest consumed by Jetson Voice.
- `configs/profiles/`: Jetson Voice deployment profiles.

## Artifact Repo

HF repo: <https://huggingface.co/harvestsu/qwen3-edgellm-jetson-artifacts>

Expected artifact layout and required files are recorded in `deploy/artifacts/qwen3_manifest.json`. Fill checksums and file sizes after uploading the Nano/NX engine sets.

## Jetson Voice Integration

In `jetson-voice`, select Qwen3 with:

```bash
JETSON_VOICE_PROFILE=multilanguage-qwen-highperf \
QWEN3_HF_REPO_ID=harvestsu/qwen3-edgellm-jetson-artifacts \
docker compose -f deploy/docker-compose.yml up -d
```

For local service runs:

```bash
JETSON_VOICE_PROFILE=multilanguage-qwen-highperf \
uvicorn app.main:app --host 0.0.0.0 --port 8621
```

## Current Baseline

See `docs/performance/qwen3-orin-profiles-2026-05-10.md` for the latest frozen numbers. At the time of repo creation, Orin NX highperf V2V smoke remained exact with warm `EOS -> first audio` around `611-637 ms` using the already validated engine set.
