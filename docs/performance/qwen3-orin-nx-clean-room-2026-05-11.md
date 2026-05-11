# Qwen3 Orin NX clean-room reproduction — 2026-05-11

Outcome: **BLOCKED at smoke test.** Source-build, artifact download, and SHA-256
verification all succeeded. Engine deserialization fails because the published
HF artifacts were exported with TensorRT 10.4 while the documented JetPack
toolchain ships TensorRT 10.3. Build-from-source path otherwise works end-to-end.

This report covers `~/project/repro-qwen3/` on host `orin-nx`
(Tailscale 100.82.225.102), run as `harvest`, executed 2026-05-11.

## TL;DR findings

1. **Engine plan/runtime mismatch (primary blocker).** Worker dies on
   `IRuntime::deserializeCudaEngine: Error Code 6: API Usage Error (The engine
   plan file is not compatible with this version of TensorRT, expecting library
   version 10.4.0.26 got ..)` for
   `talker_decode_w8a16_outputk.engine` (and presumably the other `.engine`
   files). Host has `libnvinfer-bin 10.3.0.30-1+cuda12.5`. The HF set
   `orin-nx-highperf-2026-05-11` was built against TRT 10.4.0.26.
2. **`docs/reproduce-from-zero.md` cloned the wrong default branch.** Repo
   `suharvest/jetson-local-voice` has the highperf NX profile JSON on branch
   `qwen3tts-accurate-20260507` only. `main` (`0330603`, 2026-04-27) does not
   contain `configs/profiles/multilanguage-qwen-highperf-nx.json`. A plain
   `git clone` per the doc lands you on `main` and downstream wiring breaks.
3. **`qwen3-edgellm-jetson` is a private GitHub repo.** Not cloneable from a
   fresh device without a `gh` token or SSH key. Doc shows public-style HTTPS
   `git clone` which fails. Was rsynced from Mac as a workaround.
4. **`docker-compose.yml` published in `jetson-local-voice` is the legacy
   multilanguage path.** It installs `jetson-qwen3-speech v0.1.0` and does not
   wire `EDGE_LLM_TTS_WORKER_BIN`, `EDGELLM_PLUGIN_PATH`, or `JETSON_VOICE_PROFILE=multilanguage-qwen-highperf-nx`. Following only
   `docker compose up` will not run the highperf product path.
5. **Profile JSON hardcodes `/opt/models/qwen3-edgellm`.** Not parameterized
   by `${QWEN3_ARTIFACT_ROOT}`, so when `/opt/models` is not writable (sudo
   needed on stock Jetson), the doc's `QWEN3_ARTIFACT_ROOT` env var alone is
   not enough — you must either symlink with sudo or sed the profile.
6. **Plugin filename mismatch with doc.** Doc says
   `~/project/TensorRT-Edge-LLM/build/cpp/plugins/libNvInfer_edgellm_plugin.so`;
   actual location after `cmake --build` is
   `~/project/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so` (no
   `cpp/plugins/` prefix).
7. **CMake configure does not auto-discover nvcc on stock JetPack.** Out of
   the box `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr -DCUDA_DIR=/usr/local/cuda-12.6 -DCUDA_CTK_VERSION=12.6` fails with
   `No CMAKE_CUDA_COMPILER could be found`. Must export
   `CUDACXX=/usr/local/cuda-12.6/bin/nvcc` (or prepend `/usr/local/cuda-12.6/bin`
   to PATH). Doc should call this out.
8. **`-j2` cmake build is on the edge of OOM on 16 GB Orin NX.** First attempt
   hung silently; falling back to `-j1` for the `edgellmKernels` stage
   completed cleanly. Doc could note this.
9. **GitHub and huggingface.co are blocked from the device's network (mainland
   China)** — this is environmental, not a doc bug. Worked around with
   `https://gh-proxy.com/` URL rewrite (`git config --global url...insteadOf`)
   and `HF_ENDPOINT=https://hf-mirror.com`.

## Environment

Device:

```
$ uname -a
Linux orinnx 5.15.148-tegra #1 SMP PREEMPT Thu May 22 09:38:12 UTC 2025 aarch64
$ cat /etc/nv_tegra_release | head -1
# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64, DATE: Wed Jan 8 01:49:37 UTC 2025
$ free -h | head -2
               total        used        free      shared  buff/cache   available
Mem:            15Gi       4.8Gi       1.6Gi        65Mi       8.9Gi        10Gi
$ df -h /
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p1  233G  156G   67G  71% /
$ /usr/local/cuda-12.6/bin/nvcc --version | head -3
nvcc: NVIDIA (R) Cuda compiler driver
Built on Wed_Aug_14_10:14:07_PDT_2024
$ dpkg -l | grep -E '^ii  libnvinfer-bin'
ii  libnvinfer-bin  10.3.0.30-1+cuda12.5  arm64  TensorRT binaries
$ python3 --version
Python 3.10.12
```

Power mode: `sudo nvpmodel -q` denied — this device does not allow passwordless
sudo, so power mode could not be queried during the run. tegrastats sample (idle):

```
05-11-2026 01:35:59 RAM 3386/15656MB (lfb 3x4MB) SWAP 1541/7828MB (cached 1MB) CPU [2%@1497,3%@1497,3%@1497,1%@1497,8%@1497,1%@1497,1%@1497,1%@1497] GR3D_FREQ 0% gpu@55.406C VDD_IN 9156mW/9156mW
```

CPU is reporting `@1497 MHz` across all cores, which is consistent with
MAXN-style profile under low load.

## Commit hashes

| Repo | Branch | Commit |
|---|---|---|
| `jetson-voice` (`suharvest/jetson-local-voice`) | `qwen3tts-accurate-20260507` | `bd464053141f318e69345f9d31f5c732a5c3d829` |
| `qwen3-edgellm-jetson` (`suharvest/qwen3-edgellm-jetson`, private — rsynced from Mac) | local | `e6dd2385b04f9f7a7a228a79aaf8f687acb23b44` |
| `TensorRT-Edge-LLM` (`suharvest/TensorRT-Edge-LLM`) | `qwen3-tts-highperf-runtime-w8a16` | `9f248ed6a54a1ff06be7e9ca7621ef5974a45987` |

Sidecar `deploy/artifacts/qwen3_checksums.json` was generated on this clean
checkout and copied back to the Mac.

## Step-by-step

### 1. Clone (deviation A: GitHub blocked, used proxy + correct branch)

```bash
git config --global url."https://gh-proxy.com/https://github.com/".insteadOf "https://github.com/"
mkdir -p ~/project/repro-qwen3 && cd ~/project/repro-qwen3
git clone --branch qwen3tts-accurate-20260507 https://gh-proxy.com/https://github.com/suharvest/jetson-local-voice.git jetson-voice
git clone --branch qwen3-tts-highperf-runtime-w8a16 https://gh-proxy.com/https://github.com/suharvest/TensorRT-Edge-LLM.git TensorRT-Edge-LLM
cd TensorRT-Edge-LLM && git submodule update --init --recursive
# qwen3-edgellm-jetson is private; rsynced from Mac:
#   rsync -az /Users/harvest/project/qwen3-edgellm-jetson/ harvest@orin-nx:project/repro-qwen3/qwen3-edgellm-jetson/
```

### 2. EdgeLLM build

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
cd ~/project/repro-qwen3/TensorRT-Edge-LLM
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr -DCUDA_DIR=/usr/local/cuda-12.6 -DCUDA_CTK_VERSION=12.6
# First attempt with -j2 silently OOM-killed mid-edgellmKernels; resumed with -j1 to completion.
cmake --build build --target edgellmCore NvInfer_edgellm_plugin qwen3_tts_worker -j1
```

Build outcome:

```
[100%] Built target qwen3_tts_worker
BUILD_EXIT=0
Mon May 11 01:33:31 AM EDT 2026
```

Artifacts and md5:

```
bc94948c65caa2aacfa3f24ec5c5a240  /home/harvest/project/repro-qwen3/TensorRT-Edge-LLM/build/examples/omni/qwen3_tts_worker
3d39dcfbcfe9e225343e7c7bb38488b4  /home/harvest/project/repro-qwen3/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
```

`qwen3_asr_worker` was NOT built — it is not part of the doc's recommended
target list and the worker for ASR is launched as a separate binary at runtime.
This is consistent with `app/backends/trt_edge_llm_ipc.py` which falls back to
`examples/llm/llm_inference` when the dedicated ASR worker is missing. The
specific "worker" used by the highperf NX profile for ASR remains
`qwen3_tts_worker`-style only after the engines load successfully — moot in
this run because TTS load fails first.

### 3. Artifact download (deviation B: HF blocked, mirror; deviation C: no
sudo, use `~/qwen3-models`)

```bash
cd ~/project/repro-qwen3/qwen3-edgellm-jetson
HF_ENDPOINT=https://hf-mirror.com python3 scripts/deploy_qwen3_artifacts.py \
  --set orin-nx-highperf-2026-05-11 \
  --root ~/qwen3-models
```

Total `du -sh ~/qwen3-models`: **3.1 GB**, 25 files, including
`talker_decode_w8a16_outputk.engine` (459 MB),
`asr_thinker_full_fp8embed/llm.engine` (1.20 GB),
`audio_encoder.engine` (378 MB),
`code2wav_stateful.engine` (234 MB),
`qwen3_tts_cp.engine` (222 MB).

`--check-only` output:

```
Qwen3 artifact set orin-nx-highperf-2026-05-11 OK at /home/harvest/qwen3-models
```

`--generate-sidecar` then `--verify-sha256` output:

```
Wrote /home/harvest/project/repro-qwen3/qwen3-edgellm-jetson/deploy/artifacts/qwen3_checksums.json
Qwen3 artifact set orin-nx-highperf-2026-05-11 OK at /home/harvest/qwen3-models
SHA-256 verified 25 file(s) for orin-nx-highperf-2026-05-11.
```

Sidecar `qwen3_checksums.json` (5626 bytes) was scp'd back to the Mac at
`/Users/harvest/project/qwen3-edgellm-jetson/deploy/artifacts/qwen3_checksums.json`
(MD5 `20884c91570bf598d515a4e6d030c016`).

### 4. Profile JSON path patch (deviation D)

`configs/profiles/multilanguage-qwen-highperf-nx.json` hardcodes
`/opt/models/qwen3-edgellm/...` for every `EDGE_LLM_*_DIR` env value (10
occurrences). With no `/opt/models` write access, sed-replaced to
`/home/harvest/qwen3-models`.

### 5. Smoke test launch — FAILED

Used the existing local image `jetson-voice-speech:v3.5-clean` (only
all-deps-pre-installed image available on device, since `huggingface.co` block
prevents `transformers` model fetches at runtime, but transformers itself was
pre-installed). Launched on port 8622 with all artifact + worker bindings:

```bash
docker run --rm --runtime nvidia --ipc host -p 8622:8000 \
  -v /home/harvest/qwen3-models:/home/harvest/qwen3-models:ro \
  -v /home/harvest/project/repro-qwen3/TensorRT-Edge-LLM:/home/harvest/project/repro-qwen3/TensorRT-Edge-LLM:ro \
  -v /home/harvest/project/repro-qwen3/jetson-voice/app:/opt/speech/app:ro \
  -v /home/harvest/project/repro-qwen3/jetson-voice/configs:/opt/speech/configs:ro \
  -e LANGUAGE_MODE=multilanguage \
  -e JETSON_VOICE_PROFILE=multilanguage-qwen-highperf-nx \
  -e JETSON_VOICE_PROFILE_DIR=/opt/speech/configs/profiles \
  -e QWEN3_HF_REPO_ID=harvestsu/qwen3-edgellm-jetson-artifacts \
  -e QWEN3_ARTIFACT_ROOT=/home/harvest/qwen3-models \
  -e QWEN3_ARTIFACT_SET=orin-nx-highperf-2026-05-11 \
  -e EDGE_LLM_BASE=/home/harvest/project/repro-qwen3/TensorRT-Edge-LLM \
  -e EDGE_LLM_BUILD_DIR=build \
  -e EDGE_LLM_TTS_WORKER_BIN=/home/harvest/project/repro-qwen3/TensorRT-Edge-LLM/build/examples/omni/qwen3_tts_worker \
  -e EDGELLM_PLUGIN_PATH=/home/harvest/project/repro-qwen3/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so \
  -e CUDA_MODULE_LOADING=LAZY \
  jetson-voice-speech:v3.5-clean
```

Verbatim error from the worker stderr (relayed through Python TTS backend):

```
[JV_MEM] tag=worker_entry_before_plugin   mem_available_mb=12087 mem_free_mb=2126 swap_free_mb=6287
[JV_MEM] tag=worker_after_plugin          mem_available_mb=12087 mem_free_mb=2126 swap_free_mb=6287
[JV_MEM] tag=worker_before_cuda_stream    mem_available_mb=12087 mem_free_mb=2126 swap_free_mb=6287
[JV_MEM] tag=worker_after_cuda_stream     mem_available_mb=12077 mem_free_mb=2116 swap_free_mb=6287
[JV_MEM] tag=worker_before_tts_runtime    mem_available_mb=12077 mem_free_mb=2116 swap_free_mb=6287
[05:36:22.437] [WARNING] [llmRuntimeUtils.cpp:139:collectRopeConfig] rope_scaling is not specified in the model config, using default rope type
[05:36:25.564] [ERROR]  [TensorRT] IRuntime::deserializeCudaEngine: Error Code 6: API Usage Error (The engine plan file is not compatible with this version of TensorRT, expecting library version 10.4.0.26 got ..)
[05:36:25.585] [ERROR]  [qwen3OmniTTSRuntime.cpp:2325:initializeEngineRunners] Failed to load Talker LLM engine: Failed to deserialize Qwen3-TTS Talker engine: /home/harvest/qwen3-models/engines/orin-nx/highperf/talker_w8a16_outputk/talker_decode_w8a16_outputk.engine
[JV_MEM] tag=worker_init_error             mem_available_mb=12013 mem_free_mb=2051 swap_free_mb=6287
Failed to initialize engine runners
ERROR:    Application startup failed. Exiting.
```

The Python wrapper raises:

```
RuntimeError: TTS worker failed to start
File "/opt/speech/app/backends/trt_edge_llm_tts.py", line 473, in _ensure_worker
File "/opt/speech/app/backends/trt_edge_llm_tts.py", line 386, in preload
File "/opt/speech/app/tts_service.py", line 35, in preload
File "/opt/speech/app/main.py", line 150, in startup
```

The container exits with code 3 immediately on startup. No HTTP endpoint ever
binds. **No TTS / ASR / V2V smoke could be executed.**

## What did succeed

- All three repo clones completed at correct commits.
- EdgeLLM submodule init resolved via global URL rewrite (4 submodules: NVTX,
  googletest, nlohmannJson, stb).
- EdgeLLM `-j1` build to `BUILD_EXIT=0`, producing both `qwen3_tts_worker`
  (`bc949...`) and `libNvInfer_edgellm_plugin.so` (`3d39d...`).
- HF `snapshot_download` (via mirror) of all 25 highperf NX files.
- `--check-only` passed.
- `--generate-sidecar` produced the SHA-256 index from this fresh download.
- `--verify-sha256` re-confirmed all 25 files match.
- jetson-voice startup proceeded cleanly through Python preload, plugin load,
  CUDA stream init, and into TRT engine deserialization before failing.

## Recommendations (in `qwen3-edgellm-jetson`)

1. **Re-export HF artifacts using JetPack-shipped TRT 10.3.x (libnvinfer
   10.3.0.30) on a JetPack 6.2 build host**, OR document the exact JetPack /
   TRT version required to run `orin-nx-highperf-2026-05-11`. Right now the
   set is unrunnable on the documented JetPack 6.2 (TRT 10.3) baseline.
2. **Update `docs/reproduce-from-zero.md`**:
   - Specify `--branch qwen3tts-accurate-20260507` for `jetson-local-voice`,
     OR push the highperf NX profile and EdgeLLM TTS streaming changes onto
     `main`.
   - Note `qwen3-edgellm-jetson` is private and how to authenticate
     (`gh auth login`, deploy key, or make it public).
   - Add `export CUDACXX=/usr/local/cuda-12.6/bin/nvcc` to the cmake snippet.
   - Fix plugin path: `build/libNvInfer_edgellm_plugin.so` (not `cpp/plugins/`).
   - Recommend `-j1` (or `-j2` with `MAKEFLAGS="--memory=12G"` style cap) on
     16 GB Orin NX to avoid OOM during `edgellmKernels` stage.
3. **Parameterize the profile JSON.** Either substitute `${QWEN3_ARTIFACT_ROOT}`
   in `EDGE_LLM_*_DIR` values at load time, or templated profile JSON, so
   that a non-sudo install path works without sed.
4. **Publish a docker-compose overlay** that wires highperf product env vars
   and worker/plugin bindmounts, so "fresh clone, `docker compose up`" works
   for the highperf NX path. Current published compose runs the legacy
   multilanguage backend.
5. **Manifest could note expected TRT version per artifact set**, so
   `--check-only` can warn when the host TRT minor version disagrees.

## Files / paths produced (on device)

- `~/project/repro-qwen3/jetson-voice/` (clean clone, branch `qwen3tts-accurate-20260507`)
- `~/project/repro-qwen3/qwen3-edgellm-jetson/` (rsync mirror of Mac at `e6dd238`)
- `~/project/repro-qwen3/TensorRT-Edge-LLM/build/examples/omni/qwen3_tts_worker` (md5 `bc94948c65caa2aacfa3f24ec5c5a240`)
- `~/project/repro-qwen3/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so` (md5 `3d39dcfbcfe9e225343e7c7bb38488b4`)
- `~/qwen3-models/` (3.1 GB, 25 files, all SHA-256-verified)
- `~/project/repro-qwen3/qwen3-edgellm-jetson/deploy/artifacts/qwen3_checksums.json` (also copied to Mac)

## Files modified on Mac

- `/Users/harvest/project/qwen3-edgellm-jetson/deploy/artifacts/qwen3_checksums.json`
  (new, MD5 `20884c91570bf598d515a4e6d030c016`, 5626 bytes)
- `/Users/harvest/project/qwen3-edgellm-jetson/docs/performance/qwen3-orin-nx-clean-room-2026-05-11.md`
  (this report)

---

## CORRECTION (2026-05-11, post-hoc verification)

The "TRT 10.4 vs 10.3 plan mismatch" diagnosis above was **wrong**. Re-tested
on the same orin-nx with the engines downloaded into
`/home/harvest/qwen3-models/` (the fallback path used because `/opt/models`
needed sudo) and the plugin produced by the source build. All five primary
engines deserialize and run on host TRT 10.3:

```
=== code2wav_stateful/code2wav_stateful.engine                          PASSED  (162 qps, 7.08ms)
=== talker_w8a16_outputk/talker_decode_w8a16_outputk.engine             PASSED
=== code_predictor/cp_dir/qwen3_tts_cp.engine                           PASSED
=== asr_thinker_full_fp8embed/llm.engine                                PASSED
=== asr_audio_encoder/audio/audio_encoder.engine                        PASSED
```

Linkage check (`ldd qwen3_tts_worker | grep nvinfer`) confirms the worker
binary resolves to host `/lib/aarch64-linux-gnu/libnvinfer.so.10` (10.3.0.30),
matching the JetPack baseline.

Likely real root cause of the smoke failure: the `multilanguage-qwen-highperf-nx`
profile JSON hardcodes `/opt/models/qwen3-edgellm/...` for every engine path,
but the artifacts landed at `/home/harvest/qwen3-models/...`. The agent applied
a `sed` to the profile JSON in the clean-room copy, but the running service
likely loaded a different profile copy (or the env override was incomplete),
and the resulting "engine plan ... incompatible" error message was misread as a
TRT version mismatch.

Findings #2–#9 (private repo, missing CUDACXX, plugin path doc bug, `-j2`
OOM, fork URL needed, profile root substitution, network blocks, etc.) stand
and should still be addressed.

Action: re-run smoke with `QWEN3_ARTIFACT_ROOT=~/qwen3-models` exported BEFORE
service start, OR symlink `sudo mkdir -p /opt/models && sudo ln -s
~/qwen3-models /opt/models/qwen3-edgellm`, OR add proper `${QWEN3_ARTIFACT_ROOT}`
substitution in the profile JSON.
