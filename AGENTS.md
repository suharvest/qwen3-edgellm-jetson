# Agent Guide

This repo is the Qwen3-specific companion to `jetson-voice`.

## Project Boundary

- `jetson-voice`: deployable product service, API, profiles, Docker, backend selection.
- `qwen3-edgellm-jetson`: Qwen3 ASR/TTS export scripts, engine build helpers, artifact manifests, validation scripts, and performance notes.
- `suharvest/TensorRT-Edge-LLM`: runtime/plugin fork. Do not duplicate EdgeLLM runtime patches here.
- `harvestsu/qwen3-edgellm-jetson-artifacts`: HF runtime artifact repo. It stores deployable `.engine` files and required sidecars, not ONNX by default.

## Branches

Use these EdgeLLM fork branches:

- `official-qwen3-tts-upstream-runtime`: minimal-diff correctness path for upstream review.
- `qwen3-tts-highperf-runtime-w8a16`: product high-performance path for current Orin artifacts.

## ONNX Export

Users should be able to start from official Qwen3 ASR/TTS Hugging Face snapshots and generate ONNX locally.

Environment:

```bash
TRT_SRC=$HOME/project/TensorRT-Edge-LLM \
TRT_EXPORT_PROJECT=/tmp/trt-export \
PYTHON=3.12 \
bash scripts/setup_trt_export_env.sh
```

If Qwen Python packages are not installed in the user site, set:

```bash
QWEN_ASR_PKG_DIR=/path/to/qwen_asr
QWEN_TTS_PKG_DIR=/path/to/qwen_tts
QWEN_OMNI_UTILS_PKG_DIR=/path/to/qwen_omni_utils
```

Export wrappers:

```bash
scripts/export_qwen3_asr_onnx.sh --model-dir /models/Qwen3-ASR-0.6B --out /tmp/qwen3-asr-onnx
scripts/export_qwen3_tts_onnx.sh --model-dir /models/Qwen3-TTS-0.6B --out /tmp/qwen3-tts-onnx
```

Detailed instructions live in `docs/export-from-official-weights.md`.

## Runtime Artifacts

HF runtime artifacts are described by `deploy/artifacts/qwen3_manifest.json`.

Current sets:

- `orin-nano-highperf-2026-05-10`: fully published to HF.
- `orin-nx-highperf-2026-05-11`: fully published to HF.
- `orin-nano-official-2026-05-10`: fully published to HF.

Use `scripts/package_qwen3_artifacts.py` to stage artifacts and write checksums before upload.

Do not upload temporary logs, ONNX intermediates, or ad-hoc audio samples to the runtime HF repo unless a new manifest set explicitly calls for them.

Before claiming a profile is reproducible, compare `deploy/artifacts/qwen3_manifest.json`
against the HF repo and make sure every required file exists.

## From-zero Reproduction

Use `docs/reproduce-from-zero.md` as the source of truth. Keep it aligned with:

- `README.md`
- `HF_ARTIFACTS.md`
- `deploy/artifacts/qwen3_manifest.json`
- Jetson Voice `configs/profiles/multilanguage-qwen-*.json`

The highperf path must use the EdgeLLM fork branch
`qwen3-tts-highperf-runtime-w8a16`; EdgeLLM `main` is not enough.

## Validation

Before committing script changes:

```bash
bash -n scripts/*.sh
python3 -m py_compile scripts/*.py
```

For wrapper-only changes, at minimum run:

```bash
scripts/export_qwen3_asr_onnx.sh --help
scripts/export_qwen3_tts_onnx.sh --help
scripts/export_qwen3_asr_onnx.sh --model-dir /tmp/fake-model --out /tmp/qwen3-asr-dry --dry-run
scripts/export_qwen3_tts_onnx.sh --model-dir /tmp/fake-model --out /tmp/qwen3-tts-dry --official-only --dry-run
```

For device work, use the fleet CLI from the parent environment rather than hardcoding SSH credentials.

## Known Decisions

- Full vocab is the product default for ASR and TTS.
- ONNX is generated from official weights, not stored in the default HF runtime repo.
- TensorRT engines are device/tactic specific; build them on the target Jetson class.
- The highperf product path uses W8A16 Talker, CP lm-head pretranspose, stateful Code2Wav, CP decode CUDA graph, and `QWEN3_TTS_ACTIVE_CP_GROUPS=13`.
