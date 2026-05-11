# Reproduce Qwen3 EdgeLLM Jetson

This is the shortest path for a new machine to reproduce the current released
Qwen3 ASR/TTS flow.

## What is fully published today

All manifest artifact sets are present in Hugging Face:

- `orin-nano-highperf-2026-05-10`: product highperf Nano artifact set.
- `orin-nx-highperf-2026-05-11`: product highperf NX-native artifact set.
- `orin-nano-official-2026-05-10`: official/minimal Nano artifact set for the
  upstream-style path.

Use `scripts/deploy_qwen3_artifacts.py --check-only` before any device run.

## Repositories

Clone these side by side:

```bash
mkdir -p ~/project
cd ~/project
git clone https://github.com/suharvest/jetson-local-voice.git jetson-voice
git clone https://github.com/suharvest/qwen3-edgellm-jetson.git
git clone https://github.com/suharvest/TensorRT-Edge-LLM.git
```

Use the high-performance EdgeLLM branch for the product path:

```bash
cd ~/project/TensorRT-Edge-LLM
git checkout qwen3-tts-highperf-runtime-w8a16
git submodule update --init --recursive
```

Do not use EdgeLLM `main` for highperf artifacts. It does not contain the W8A16
plugin, Qwen3-TTS CP kernels, stateful Code2Wav runner, and speaker embedding
worker support required by this profile.

## Download runtime artifacts

On the Jetson:

```bash
cd ~/project/qwen3-edgellm-jetson
python3 scripts/deploy_qwen3_artifacts.py \
  --set orin-nx-highperf-2026-05-11 \
  --root /opt/models/qwen3-edgellm
```

If `/opt/models` requires root, download to a writable directory and point
Jetson Voice at it with `QWEN3_ARTIFACT_ROOT`.

## Build runtime binaries

Build TensorRT-Edge-LLM on Jetson with TensorRT/CUDA from JetPack. The highperf
worker/plugin must come from the same EdgeLLM branch as the engines:

```bash
cd ~/project/TensorRT-Edge-LLM
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCUDA_DIR=/usr/local/cuda-12.6 \
  -DCUDA_CTK_VERSION=12.6
cmake --build build --target edgellmCore NvInfer_edgellm_plugin qwen3_tts_worker -j2
```

If a CUDA source changed and the example build fails with a fatbin/device-link
symbol error, clean stale example device-link outputs:

```bash
rm -f build/examples/utils/libexampleUtils.a \
      build/examples/utils/CMakeFiles/exampleUtils.dir/cmake_device_link.o
cmake --build build --target exampleUtils qwen3_tts_worker -j1
```

## Run Jetson Voice

The currently complete HF set is NX highperf:

```bash
cd ~/project/jetson-voice
JETSON_VOICE_PROFILE=multilanguage-qwen-highperf-nx \
QWEN3_HF_REPO_ID=harvestsu/qwen3-edgellm-jetson-artifacts \
QWEN3_ARTIFACT_ROOT=/opt/models/qwen3-edgellm \
EDGE_LLM_TTS_WORKER_BIN=~/project/TensorRT-Edge-LLM/build/examples/omni/qwen3_tts_worker \
EDGELLM_PLUGIN_PATH=~/project/TensorRT-Edge-LLM/build/cpp/plugins/libNvInfer_edgellm_plugin.so \
uvicorn app.main:app --host 0.0.0.0 --port 8621
```

For Docker/compose deployment, pass the same profile and artifact env vars into
`deploy/docker-compose.yml`.

## Voice cloning

Highperf voice cloning is embedding-based:

1. Extract a speaker x-vector once with Qwen3-TTS `speaker_encoder.onnx`.
2. Store or cache the float32 embedding bytes.
3. Pass the embedding to Jetson Voice clone APIs; the worker forwards it as
   `speaker_embedding_b64`.

Do not run the speaker encoder inside every low-latency synthesis request.

## Validation checklist

Minimum checks after a fresh deployment:

```bash
python3 scripts/deploy_qwen3_artifacts.py --set orin-nx-highperf-2026-05-11 --check-only
bash -n scripts/*.sh
python3 -m py_compile scripts/*.py
```

Then run a Jetson Voice health check and a short V2V or TTS smoke from the
`scripts/` directory. Compare results with
`docs/performance/qwen3-orin-profiles-2026-05-10.md`.
