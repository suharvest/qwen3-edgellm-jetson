# Reproduce Qwen3 EdgeLLM Jetson

This is the shortest path for a new machine to reproduce the current released
Qwen3 ASR/TTS flow.

> **2026-05-11 update**: this guide has been hardened against the issues that
> blocked the first clean-room reproduction on Orin NX. See `docs/performance/
> qwen3-orin-nx-clean-room-2026-05-11.md` for the original report. The fixes
> below cover JetPack TensorRT 10.3.0.30 + CUDA 12.6.

## What is fully published today

All manifest artifact sets are present in Hugging Face:

- `orin-nano-highperf-2026-05-10`: product highperf Nano artifact set.
- `orin-nx-highperf-2026-05-11`: product highperf NX-native artifact set.
- `orin-nano-official-2026-05-10`: official/minimal Nano artifact set for the
  upstream-style path.

Use `scripts/deploy_qwen3_artifacts.py --check-only` before any device run.
Once the set is downloaded, run again with `--verify-sha256` to catch partial
downloads (sidecar at `deploy/artifacts/qwen3_checksums.json`).

## Repositories

Clone these side by side. All three repos are public:

```bash
mkdir -p ~/project
cd ~/project
git clone https://github.com/suharvest/jetson-local-voice.git jetson-voice
git clone https://github.com/suharvest/qwen3-edgellm-jetson.git
git clone https://github.com/suharvest/TensorRT-Edge-LLM.git
```

**The highperf NX product profile lives on a feature branch of jetson-voice,
not `main`.** Check it out before running:

```bash
cd ~/project/jetson-voice
git checkout qwen3tts-accurate-20260507
```

(Once that branch lands on `main` this step can be removed.)

Pick the EdgeLLM branch that matches the artifact line you intend to run:

```bash
cd ~/project/TensorRT-Edge-LLM
# Highperf product line (W8A16 plugin, CP kernels, stateful Code2Wav, speaker embedding):
git checkout qwen3-tts-highperf-runtime-w8a16
# OR upstream/minimal line:
# git checkout official-qwen3-tts-upstream-runtime
git submodule update --init --recursive
```

Behind a slow or proxied network the `nlohmannJson` submodule fetch is the most
common failure point — re-run `git submodule update --init --recursive
3rdParty/nlohmannJson` if it aborts midway. If `github.com` is unreachable
altogether, rewrite remotes through a mirror:

```bash
git config --global url."https://gh-proxy.com/https://github.com/".insteadOf \
  "https://github.com/"
```

Do not use EdgeLLM `main` for highperf artifacts. It does not contain the W8A16
plugin, Qwen3-TTS CP kernels, stateful Code2Wav runner, and speaker embedding
worker support required by this profile.

The published `orin-{nx,nano}-highperf-2026-05-10/11` engines were validated
against a specific W8A16 kernel set (`w8a16_m1_output_k_kernel`,
`w8a16_hmma_m16n16k16_kernel`, `w8a16_small_m_tiled_kernel`,
`w8a16_per_output_output_k_reference_kernel`,
`w8a16_per_output_reference_kernel`). Commits `8a26eba` and `7ab7f1c` on
this branch carry those kernels and the matching plugin dispatcher. After
building, you can confirm the right kernel set is linked with:

```bash
nm build/libNvInfer_edgellm_plugin.so | grep -oE 'w8a16_[a-z0-9_]+_kernel' | sort -u
```

Five symbols must appear, matching the names above. If you see
`w8a16_per_output_tiled_kernel` / `w8a16_per_output_tiled_pair_k_kernel`
instead, the build is on a regressed source revision and TTS audio will
sound coherent-but-wrong (ASR maps it to a single repeated character).

## Download runtime artifacts

On the Jetson:

```bash
cd ~/project/qwen3-edgellm-jetson
python3 scripts/deploy_qwen3_artifacts.py \
  --set orin-nx-highperf-2026-05-11 \
  --root /opt/models/qwen3-edgellm
```

If `/opt/models` requires sudo and you cannot escalate, download to a writable
directory and export the same path before starting jetson-voice — the profile
JSONs resolve every engine path via `${QWEN3_ARTIFACT_ROOT}`:

```bash
python3 scripts/deploy_qwen3_artifacts.py \
  --set orin-nx-highperf-2026-05-11 \
  --root "$HOME/qwen3-models"
export QWEN3_ARTIFACT_ROOT="$HOME/qwen3-models"
```

Behind a Great-Firewall or otherwise huggingface.co-blocked network, set
`HF_ENDPOINT=https://hf-mirror.com` before running the deploy script.

After the download completes, verify integrity (catches partial files):

```bash
python3 scripts/deploy_qwen3_artifacts.py \
  --set orin-nx-highperf-2026-05-11 \
  --root "${QWEN3_ARTIFACT_ROOT:-/opt/models/qwen3-edgellm}" \
  --verify-sha256
```

## Build runtime binaries

Build TensorRT-Edge-LLM on Jetson with TensorRT 10.3 / CUDA 12.6 from JetPack.
The highperf worker/plugin must come from the same EdgeLLM branch as the
engines.

CMake on JetPack 6 needs an explicit CUDA compiler hint. The highperf
branch already defaults `CMAKE_CUDA_ARCHITECTURES` to `{80;86;87;89}` so
Orin's SM87 cubin gets registered automatically. The
`official-qwen3-tts-upstream-runtime` branch keeps the upstream default
of `52`, which is too old for `__hfma2` in `int4WoQGemvCuda.cu`; on that
branch you MUST pass `-DCMAKE_CUDA_ARCHITECTURES=87` explicitly.

```bash
cd ~/project/TensorRT-Edge-LLM
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc

# Highperf product line (default arch list already includes 87):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCUDA_DIR=/usr/local/cuda-12.6 \
  -DCUDA_CTK_VERSION=12.6

# Official/upstream-reference line (must pin SM87 by hand):
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
#   -DCMAKE_CUDA_ARCHITECTURES=87 \
#   -DTRT_PACKAGE_DIR=/usr -DCUDA_DIR=/usr/local/cuda-12.6 -DCUDA_CTK_VERSION=12.6

# -j1 on 16GB Orin NX; -j2 has been observed OOM-killed during edgellmCore
# kernel compilation. AGX Orin (32GB+) can use higher parallelism.
cmake --build build --target edgellmCore NvInfer_edgellm_plugin qwen3_tts_worker -j1
```

Built artifacts (use these exact paths in env vars):

- Worker: `~/project/TensorRT-Edge-LLM/build/examples/omni/qwen3_tts_worker`
- Plugin: `~/project/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so`
  (NOT `build/cpp/plugins/...` — older docs were wrong)

If a CUDA source changed and the example build fails with a fatbin/device-link
symbol error, clean stale example device-link outputs:

```bash
rm -f build/examples/utils/libexampleUtils.a \
      build/examples/utils/CMakeFiles/exampleUtils.dir/cmake_device_link.o
cmake --build build --target exampleUtils qwen3_tts_worker -j1
```

## Run Jetson Voice

The currently complete HF set is NX highperf. Note: the python backend
expects the EdgeLLM directory name lowercased; create a symlink if you cloned
into `TensorRT-Edge-LLM`:

```bash
ln -sf ~/project/TensorRT-Edge-LLM ~/project/tensorrt-edge-llm
```

Then start the service:

```bash
cd ~/project/jetson-voice
JETSON_VOICE_PROFILE=multilanguage-qwen-highperf-nx \
QWEN3_HF_REPO_ID=harvestsu/qwen3-edgellm-jetson-artifacts \
QWEN3_ARTIFACT_ROOT="${QWEN3_ARTIFACT_ROOT:-/opt/models/qwen3-edgellm}" \
EDGE_LLM_BASE=~/project/tensorrt-edge-llm \
EDGE_LLM_BUILD_DIR=build \
EDGE_LLM_TTS_WORKER_BIN=~/project/tensorrt-edge-llm/build/examples/omni/qwen3_tts_worker \
EDGELLM_PLUGIN_PATH=~/project/tensorrt-edge-llm/build/libNvInfer_edgellm_plugin.so \
uvicorn app.main:app --host 0.0.0.0 --port 8621
```

For Docker/compose deployment, pass the same profile and artifact env vars into
`deploy/docker-compose.yml`. The shipped `docker-compose.yml` is currently the
legacy multilanguage backend; the highperf product compose is tracked in §2 of
`docs/reproduction-remaining-work-2026-05-11.md`.

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
python3 scripts/deploy_qwen3_artifacts.py --set orin-nx-highperf-2026-05-11 --verify-sha256
bash -n scripts/*.sh
python3 -m py_compile scripts/*.py

# Sanity-deserialize an engine against the freshly built plugin (host TRT 10.3):
/usr/src/tensorrt/bin/trtexec \
  --loadEngine="${QWEN3_ARTIFACT_ROOT:-/opt/models/qwen3-edgellm}/engines/orin-nx/highperf/code2wav_stateful/code2wav_stateful.engine" \
  --plugins=~/project/tensorrt-edge-llm/build/libNvInfer_edgellm_plugin.so
```

Then run a Jetson Voice health check and a short V2V or TTS smoke from the
`scripts/` directory. Compare results with
`docs/performance/qwen3-orin-profiles-2026-05-10.md`.

If any engine deserialize fails with an "engine plan ... incompatible" message,
do **not** assume a TRT version mismatch first. Check (in order):
1. `ldd build/examples/omni/qwen3_tts_worker | grep nvinfer` — confirm host
   `/lib/aarch64-linux-gnu/libnvinfer.so.10` is what got linked.
2. `/usr/src/tensorrt/bin/trtexec --loadEngine=<path> --plugins=<plugin.so>`
   — if trtexec accepts the engine the runtime issue is elsewhere
   (typically `LD_LIBRARY_PATH` shadowing the host TRT, or the artifact path
   not actually pointing where the profile expects).
3. `~/.claude/skills/device-gotchas/references/gotchas-jetson.md` line 499+ for
   the documented `LD_LIBRARY_PATH` shadowing pattern.
