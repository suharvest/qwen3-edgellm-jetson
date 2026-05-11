# Qwen3 EdgeLLM Jetson

Qwen3 ASR/TTS deployment, build, and validation toolkit for Jetson Orin devices using TensorRT-Edge-LLM.

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

- `native/edgellm_voice_worker/`: resident ASR/TTS worker binaries used by Jetson Voice.
- `scripts/`: export, engine build, quantization, stateful Code2Wav, quality gate, and V2V scripts.
- `docs/performance/`: frozen performance records across Orin Nano/NX.
- `docs/reproduce-from-zero.md`: step-by-step reproduction checklist for a new machine.
- `docs/plans/`: implementation notes and negative results that should not be rediscovered.
- `docs/export-from-official-weights.md`: official Qwen3 weight -> ONNX export guide.
- `AGENTS.md`: concise operating guide for coding agents working in this repo.
- `deploy/artifacts/qwen3_manifest.json`: HF artifact manifest consumed by Jetson Voice.
- `configs/profiles/`: Jetson Voice deployment profiles.

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

In `jetson-voice`, select Qwen3 with:

```bash
JETSON_VOICE_PROFILE=multilanguage-qwen-highperf \
QWEN3_HF_REPO_ID=harvestsu/qwen3-edgellm-jetson-artifacts \
docker compose -f deploy/docker-compose.yml up -d
```

On Orin NX, use:

```bash
JETSON_VOICE_PROFILE=multilanguage-qwen-highperf-nx \
QWEN3_HF_REPO_ID=harvestsu/qwen3-edgellm-jetson-artifacts \
docker compose -f deploy/docker-compose.yml up -d
```

See `docs/reproduce-from-zero.md` for the full clone -> artifact download ->
EdgeLLM build -> Jetson Voice run checklist.

For local service runs:

```bash
JETSON_VOICE_PROFILE=multilanguage-qwen-highperf \
uvicorn app.main:app --host 0.0.0.0 --port 8621
```

## Current Baseline

See `docs/performance/qwen3-orin-profiles-2026-05-10.md` for the latest frozen numbers. At the time of repo creation, Orin NX highperf V2V smoke remained exact with warm `EOS -> first audio` around `611-637 ms` using the already validated engine set.

Highperf TTS voice cloning is embedding-based: extract a speaker x-vector once with the Qwen3-TTS `speaker_encoder.onnx`, then pass `speaker_embedding_b64` to the resident TTS worker. The low-latency path should not run the speaker encoder on every synthesis request.
