# Qwen3 Jetson Reproduction Remaining Work

Date: 2026-05-11

This document is for agents continuing the reproduction/deployment cleanup.
The core model artifacts and code branches are now present; the remaining work
is packaging, clean-room validation, and making the path harder to misuse.

## Current state

Repositories:

- `jetson-voice`: product service and API.
- `qwen3-edgellm-jetson`: Qwen3 export/build/deploy docs, scripts, profiles,
  and artifact manifest.
- `suharvest/TensorRT-Edge-LLM`: EdgeLLM fork with highperf runtime code.
- `harvestsu/qwen3-edgellm-jetson-artifacts`: Hugging Face model artifact repo.

Local path for this repo:

```bash
/Users/harvest/project/qwen3-edgellm-jetson
```

Published HF artifact sets have been checked against
`deploy/artifacts/qwen3_manifest.json`:

```text
orin-nano-highperf-2026-05-10 missing 0
orin-nx-highperf-2026-05-11 missing 0
orin-nano-official-2026-05-10 missing 0
```

Important EdgeLLM branches:

- `qwen3-tts-highperf-runtime-w8a16`: product highperf path.
- `official-qwen3-tts-upstream-runtime`: minimal-diff official/upstream-style path.

## What is not done yet

### 1. Clean-room reproduction

Goal: prove a new checkout can run without relying on old `/tmp` state.

Run on a Jetson device, preferably both:

- Orin NX with `orin-nx-highperf-2026-05-11`
- Orin Nano with `orin-nano-highperf-2026-05-10`

Procedure:

```bash
mkdir -p ~/project/repro-qwen3
cd ~/project/repro-qwen3
git clone https://github.com/suharvest/jetson-local-voice.git jetson-voice
git clone https://github.com/suharvest/qwen3-edgellm-jetson.git
git clone https://github.com/suharvest/TensorRT-Edge-LLM.git
```

Then follow `docs/reproduce-from-zero.md`.

Acceptance:

- `scripts/deploy_qwen3_artifacts.py --check-only` passes.
- EdgeLLM highperf branch builds `qwen3_tts_worker` and plugin from source.
- Jetson Voice starts with the selected Qwen profile.
- TTS streaming emits connected chunks.
- ASR/TTS resident mode starts without OOM.
- Record first-audio latency, total V2V latency, memory, power mode, and commit
  hashes in `docs/performance/`.

### 2. Docker image rebuild and publish

Goal: make `docker compose up` enough for product deployment.

Current limitation: source reproduction is available, but the runtime binary and
plugin are expected to be built from the EdgeLLM fork. They are not stored in
the HF model repo.

Tasks:

- Build a Jetson image that includes:
  - current `jetson-voice`
  - EdgeLLM highperf branch
  - `qwen3_tts_worker`
  - ASR worker if used by the profile
  - matching `libNvInfer_edgellm_plugin.so`
- Ensure profiles still download model artifacts from HF at startup.
- Publish image tag and update Jetson Voice README/compose examples.

Acceptance:

- Fresh device can run Qwen highperf profile with only Docker/compose and HF
  access.
- The container does not depend on `/tmp/qwen3_highperf_bin`.
- Worker/plugin paths are explicit and documented.

### 3. Runtime binary/plugin path cleanup

Goal: remove ambiguity between source-built binaries, `/tmp` binaries, and
container binaries.

Known risk:

- Many historical scripts still default to `/tmp/qwen3_highperf_bin`.
- That path is useful for past experiments but should not be the production
  default in fresh docs.

Tasks:

- Audit `scripts/`, `README.md`, and Jetson Voice profiles for hardcoded
  `/tmp/qwen3_highperf_bin`.
- Keep historical references in performance notes, but mark them as historical.
- Prefer one of:
  - container path, or
  - `~/project/TensorRT-Edge-LLM/build/...` source-build path.

Acceptance:

- New reproduction docs do not require a preexisting `/tmp/qwen3_highperf_bin`.
- Legacy scripts clearly say when they are using historical local binaries.

### 4. Official/minimal profile validation

Goal: prove the official path is not only downloadable but runnable.

HF now contains the official Nano engine set. It still needs a clean run.

Tasks:

- Use `JETSON_VOICE_PROFILE=multilanguage-qwen-official`.
- Download `orin-nano-official-2026-05-10`.
- Build/use the official EdgeLLM branch.
- Run a short TTS and ASR smoke.

Acceptance:

- Official/minimal profile starts.
- TTS semantics are correct on a fixed prompt.
- Any deviation from highperf behavior is documented as expected.

### 5. Voice clone quality gate

Goal: verify highperf voice clone with a real reference embedding, not only
synthetic embedding smoke.

Already done:

- Worker protocol accepts `speaker_embedding_b64`.
- Streaming emits connected chunks in a protocol smoke.

Still needed:

- Use `speaker_encoder.onnx` to extract a real speaker embedding.
- Run `/tts/clone` and `/tts/clone/stream`.
- Save one reference audio and one synthesized output for listening.
- Confirm content correctness and voice similarity are acceptable.

Acceptance:

- Real reference embedding works end to end.
- Low-latency path uses precomputed embedding and does not run speaker encoder
  per request.
- Document the exact extraction command and output embedding format.

### 6. Manifest/checksum hardening

Goal: make HF integrity checks stronger than path existence.

Current state:

- Required paths exist in HF.
- Some staging directories contain checksum inventories, but the manifest does
  not enforce SHA-256.

Tasks:

- Extend `deploy/artifacts/qwen3_manifest.json` or add a sidecar checksum index.
- Make `scripts/deploy_qwen3_artifacts.py --check-only` optionally verify
  SHA-256 for downloaded files.
- Record file sizes for each artifact set.

Acceptance:

- A corrupted partial download fails verification.
- Checksum verification can be skipped for speed but is available for release
  validation.

## Do not redo

Do not re-open these unless a regression is found:

- ASR/TTS vocab pruning as a default path. Product default is full vocab.
- CP group count search for the current default. Current highperf default is
  `QWEN3_TTS_ACTIVE_CP_GROUPS=13`.
- Cross-group sampling fusion. It risks changing stochastic sampling semantics.
- Treating EdgeLLM `main` as compatible with highperf artifacts. It is not.

## Useful verification commands

HF completeness check:

```bash
python3 scripts/deploy_qwen3_artifacts.py \
  --set orin-nx-highperf-2026-05-11 \
  --check-only
```

Static checks in this repo:

```bash
python3 -m json.tool deploy/artifacts/qwen3_manifest.json >/dev/null
bash -n scripts/*.sh
python3 -m py_compile scripts/*.py
git diff --check
```

Recommended first clean-room target:

```bash
JETSON_VOICE_PROFILE=multilanguage-qwen-highperf-nx
QWEN3_ARTIFACT_SET=orin-nx-highperf-2026-05-11
```

