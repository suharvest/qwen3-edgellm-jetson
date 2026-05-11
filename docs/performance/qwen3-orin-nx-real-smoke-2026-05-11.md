# Qwen3 Orin NX Highperf Real Smoke + §5 Voice Clone Gate — 2026-05-11

Status: BLOCKED. The published `orin-nx-highperf-2026-05-11` HF artifact set
plus the `qwen3-tts-highperf-runtime-w8a16` branch of TensorRT-Edge-LLM
(commit `9f248ed`) plus the `qwen3tts-accurate-20260507` branch of
jetson-voice does not bring up a working highperf service on Orin NX (Jetpack 6
host with TensorRT 10.3.0.30).

This run was performed under `~/project/repro-qwen3/` on `orin-nx`
(Tailscale 100.82.225.102) with the parallel `TensorRT-Edge-LLM-official`
build untouched.

## Environment

- Host: Orin NX 16GB, JetPack 6 (CUDA 12.6), nvpmodel `40W` (mode 4)
- Host TensorRT: `10.3.0.30-1+cuda12.5` (libnvinfer.so.10.3.0)
- Container TensorRT: `10.4.0` (libnvinfer.so.10.4.0) — incompatible with engines
- EdgeLLM plugin: `~/project/repro-qwen3/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so` md5 `3d39dcfbcfe9e225343e7c7bb38488b4`
- TTS worker (EdgeLLM-shipped, omni variant): `~/project/repro-qwen3/TensorRT-Edge-LLM/build/examples/omni/qwen3_tts_worker` md5 `bc94948c65caa2aacfa3f24ec5c5a240`
- ASR worker (jetson-voice native, freshly built this session): `~/project/repro-qwen3/jetson-voice/build/edgellm_voice_worker/workers/qwen3_asr_worker` md5 `e2b897e473bfa2bacdd3106f3e8bc5fb`
- Speaker encoder (NOT in HF artifact set): `/home/harvest/voice_test/models/qwen3-tts/onnx/speaker_encoder.onnx` (fp32, mel→1024-d embedding)

## Setup approach (Task A)

Profile JSON `multilanguage-qwen-highperf-nx.json` at HEAD already references
`/home/harvest/qwen3-models/...` paths, so no profile-path remediation was
needed. Symlink/edit shenanigans avoided.

Built native ASR worker (was missing from cloned EdgeLLM build):

```
mkdir -p ~/project/repro-qwen3/jetson-voice/build/edgellm_voice_worker
cd ~/project/repro-qwen3/jetson-voice/build/edgellm_voice_worker
CUDACXX=/usr/local/cuda-12.6/bin/nvcc \
EDGE_LLM_BASE=~/project/repro-qwen3/TensorRT-Edge-LLM \
EDGE_LLM_BUILD=~/project/repro-qwen3/TensorRT-Edge-LLM/build \
cmake ~/project/repro-qwen3/jetson-voice/native/edgellm_voice_worker \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release -DEMBEDDED_TARGET=jetson_orin
cmake --build . -j2
```

Built `qwen3_asr_worker` and `qwen3_tts_worker`. Note the native
`qwen3_tts_worker` is the OLD upstream variant (only knows
`--talkerEngineDir/--code2wavEngineDir/--codePredictorEngineDir/--tokenizerDir/--debug`)
— so jetson-voice's resolver was steered to the EdgeLLM omni TTS worker by
mounting only `qwen3_asr_worker` into `JETSON_VOICE_WORKER_BUILD`.

## Service start (Task B)

Used the existing host docker image `jetson-voice-speech:v3.5-clean`
(it has fastapi/sherpa-onnx/onnxruntime preinstalled), but the container
ships TensorRT 10.4. Engines were built with TRT 10.3 (`PASSED` standalone via
host `trtexec`). Running engines under in-container 10.4 produced:

```
[ERROR] [TensorRT] IRuntime::deserializeCudaEngine: Error Code 6: API Usage Error
(The engine plan file is not compatible with this version of TensorRT,
expecting library version 10.4.0.26 got ..)
```

Worked around by mounting host TRT 10.3 libs at `/opt/trt103` and prepending
to `LD_LIBRARY_PATH` so the container loads host `libnvinfer.so.10.3.0`. After
this, ASR engine deserialised in-container.

Final run command:

```
docker run -d --name jetson_voice_repro_smoke \
  --runtime nvidia --ipc host \
  -p 18091:8000 \
  -e JETSON_VOICE_PROFILE=multilanguage-qwen-highperf-nx \
  -e EDGE_LLM_BASE=/opt/edgellm-src -e EDGE_LLM_BUILD_DIR=build \
  -e JETSON_VOICE_WORKER_BUILD=/opt/jv-workers \
  -e EDGELLM_PLUGIN_PATH=/opt/edgellm-src/build/libNvInfer_edgellm_plugin.so \
  -e LD_LIBRARY_PATH=/opt/trt103:/usr/local/lib/python3.10/dist-packages/onnxruntime/capi:/opt/edgellm-src/build:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu \
  -v /tmp/trt103-host:/opt/trt103:ro \
  -v ~/project/repro-qwen3/jetson-voice/app:/opt/speech/app:ro \
  -v ~/project/repro-qwen3/jetson-voice/configs:/opt/speech/configs:ro \
  -v ~/project/repro-qwen3/TensorRT-Edge-LLM:/opt/edgellm-src:ro \
  -v /tmp/jv_workers_filtered:/opt/jv-workers:ro \
  -v ~/qwen3-models:/home/harvest/qwen3-models:ro \
  -v /home/harvest/voice_test/models:/opt/models:ro \
  -w /opt/speech/app jetson-voice-speech:v3.5-clean \
  bash -c 'cd /opt/speech/app && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000'
```

Warmup time: never reached `/health`. Application startup failed (see below).

## TRT engine bring-up — standalone trtexec on host (TRT 10.3)

All five engines load cleanly via `trtexec --skipInference` on host TRT 10.3
with the freshly-built plugin. Excerpt:

| Engine | Path | trtexec |
|---|---|---|
| ASR thinker | `engines/orin-nx/highperf/asr_thinker_full_fp8embed/llm.engine` | PASSED |
| ASR audio encoder | `engines/orin-nx/highperf/asr_audio_encoder/audio/audio_encoder.engine` | PASSED |
| TTS talker | `engines/orin-nx/highperf/talker_w8a16_outputk/talker_decode_w8a16_outputk.engine` | PASSED |
| TTS code predictor | `engines/orin-nx/highperf/code_predictor/cp_dir/qwen3_tts_cp.engine` | PASSED |
| TTS code2wav stateful | `engines/orin-nx/highperf/code2wav_stateful/code2wav_stateful.engine` | PASSED |

So the engines themselves are valid. Failures are at the runtime layer that
wraps them.

## Failures observed at runtime

### 1. ASR worker — `There must be one kernel to implement the MHA`

```
[06:01:58.117] [WARNING] [TensorRT] Using an engine plan file across different
  models of devices is not recommended and is likely to affect performance or
  even cause errors.
[06:02:54.653] [WARNING] [TensorRT] Using an engine plan file across different
  models of devices is not recommended and is likely to affect performance or
  even cause errors.
terminate called after throwing an instance of 'std::runtime_error'
  what():  There must be one kernel to implement the MHA
```

Plan loads fine (`trtexec` PASSED above), but the runtime path inside the ASR
worker (which goes through EdgeLLM's MHA dispatch) cannot find a registered
kernel for the MHA op as recorded in the plan. The "across different models of
devices" warning suggests the plan was originally cooked on a different SM /
device mode than the in-container effective device — but the plugin used for
trtexec is the same .so freshly built on this host. This is likely a kernel
registration mismatch between the published engine and the
`qwen3-tts-highperf-runtime-w8a16` plugin source: the engine references an MHA
kernel variant that the current plugin .so does not register. Re-cooking the
ASR engine on this checkout would presumably fix it but defeats the purpose of
reproducing from the published HF artifact set.

### 2. TTS worker — `scales must be provided for FP8 embedding table`

```
[06:03:02.820] [WARNING] [llmRuntimeUtils.cpp:139:collectRopeConfig] rope_scaling
  is not specified in the model config, using default rope type.
[06:03:13.432] [WARNING] [qwen3OmniTTSRuntime.cpp:779:Qwen3TTSCodePredictorEngine]
  Qwen3-TTS CP decode CUDA graph enabled (experimental; CPU sampling keeps
  token quality)
[JV_MEM] tag=worker_init_error mem_available_mb=9376 mem_free_mb=3329
scales must be provided for FP8 embedding table
```

The worker reaches the embedding table load. `text_embedding.safetensors`
contains both `text_embedding` (F8_E4M3, [151936,2048]) and
`text_embedding_scale` (F32, [151936,16]) — confirmed via metadata dump
(md5 `de947d079fef13d29e115f728ff6a96b`). The runtime's
`Qwen3OmniTTSRuntime::loadTalkerWeights` uses
`textEmbedTensors[0]` as the embedding table without filtering by name. Tensor
file order is `text_embedding_scale` first, `text_embedding` second. So the
runtime is taking the FP32 scales as the table, then later the embedding
kernel sees an FP8 dtype somewhere (or the inverse) and trips the check at
`embeddingKernels.cu:493`. Either the export pipeline writes the wrong tensor
order, or the runtime expects a different naming convention than the published
artifact uses. Either way: a builder/runtime drift between the artifact
producer and the highperf-runtime-w8a16 consumer.

`embedding.safetensors` (the talker codec embedding, separate file) is FP16
and parses fine — that one is not the problem.

## Speaker encoder check (Task D pre-flight)

Speaker encoder ONNX is **not part of the HF artifact set**
`orin-nx-highperf-2026-05-11`. Found locally at
`/home/harvest/voice_test/models/qwen3-tts/onnx/speaker_encoder.onnx`
(staging from a prior export). IO:

- input: `mel` `[1, time, 128]` float32
- output: `speaker_embedding` `[1024]` float32

Cannot proceed with §5 voice clone end-to-end because the TTS worker never
starts (failure #2). The speaker encoder itself is functional but there is
nothing to feed the embedding into.

## Smokes (Task C/D)

NOT EXECUTED — service `/health` never came up.

| Step | Status |
|---|---|
| TTS smoke (`/tts`) | not run |
| TTS streaming (`/tts/stream`) | not run |
| ASR smoke (`/asr`) | not run |
| Voice clone (`/tts/clone`) | not run |
| Voice clone streaming (`/tts/clone/stream`) | not run |

## Service log (filtered)

```
$ grep -iE 'error|crash|fail|oom|warn' /tmp/jv_smoke.log
[WARNING] model_downloader: Qwen3 artifact deploy script missing: /opt/speech/scripts/deploy_qwen3_artifacts.py
[WARNING] backends.trt_edge_llm_asr: TRT-EdgeLLM ASR worker warmup failed: ASR worker exited before response
  [WARNING] [TensorRT] Using an engine plan file across different models of devices ...
  terminate called after throwing an instance of 'std::runtime_error'
  what():  There must be one kernel to implement the MHA
[WARNING] main: ASR warm-up failed: 'TRTEdgeLLMASRBackend' object has no attribute 'transcribe_audio'
[WARNING] backends.trt_edge_llm_tts: Code2Wav not found at .../code2wav_stateful/code2wav.engine — will output RVQ codes only
RuntimeError: TTS worker failed to start: ... scales must be provided for FP8 embedding table
ERROR:    Application startup failed. Exiting.
```

(Container exited; not running at report time.)

## Tegrastats sample (warmup, ~1Hz)

```
05-11-2026 02:06:38 RAM 4772/15656MB SWAP 1499/7828MB CPU [12,24,6,21,1,48,4,10] GR3D 0% gpu@55.5C VDD_IN 9932mW
05-11-2026 02:06:39 RAM 4783/15656MB ... gpu@55.5C VDD_IN 9932mW
05-11-2026 02:06:40 RAM 4780/15656MB ... gpu@55.7C VDD_IN 9932mW
05-11-2026 02:06:41 RAM 4776/15656MB ... gpu@55.7C VDD_IN 9932mW
05-11-2026 02:06:42 RAM 4786/15656MB ... gpu@55.3C VDD_IN 9932mW
```

GPU stayed idle (no inference reached). Memory headroom: 10+GB available
during the failed startup; not a memory issue.

## What the previous "10.4 vs 10.3 plan mismatch" diagnosis got right and wrong

- RIGHT: there IS a 10.3/10.4 split — the `jetson-voice-speech:v3.5-clean`
  docker image carries TRT 10.4 while the host carries 10.3 and the published
  engines were cooked at 10.3. Running them as-is inside the container does
  fail with the "expecting library version 10.4.0.26" error.
- WRONG: the engines themselves are not unusable. They load fine on host TRT
  10.3 (`trtexec` PASSED for all five). The proper bring-up is to either run
  the Python service on the host (which lacks fastapi/sherpa-onnx/etc) or
  shim host TRT libs into the container.
- ALSO WRONG: the `/opt/models` vs `/home/harvest/qwen3-models` path mismatch
  is a non-issue today. The committed
  `multilanguage-qwen-highperf-nx.json` already points at
  `/home/harvest/qwen3-models/...` (a previous fix may have already landed).

## Process maps

Service exited at `Application startup failed`; no live process to `pmap`.
Binary md5s captured above instead.

## Conclusions / what's needed for §5 to proceed

1. **HF artifact gap (§6)**: `text_embedding.safetensors` tensor order /
   naming convention does not match the highperf runtime's expectations.
   Either re-export with `text_embedding` listed first (or rename to
   `embedding`/`embedding_scale`), OR patch `qwen3OmniTTSRuntime.cpp` to
   look up tensors by name not by index.
2. **HF artifact gap (§6)**: `speaker_encoder.onnx` is missing from the
   artifact manifest. §5 voice clone is unreproducible from HF alone.
3. **Engine vs plugin drift**: The ASR thinker engine plan references an MHA
   kernel that the freshly-built plugin doesn't register. Either re-cook the
   engine against the current plugin, or add the missing kernel to the
   plugin. Cannot proceed with ASR smoke until this is resolved.
4. **Documentation gap**: `multilanguage-qwen-highperf-nx` profile assumes the
   container has matching TRT 10.3, but the published `v3.5-clean` image
   ships 10.4. Either publish a new image at TRT 10.3, or document the host
   TRT 10.3 lib mount workaround used here.

## Files of interest (all on `orin-nx`)

- `/tmp/jv_smoke.log` — full service stdout/stderr
- `/tmp/tegra_smoke.log` — tegrastats during run (cleaned up after)
- `/home/harvest/voice_test/models/qwen3-tts/onnx/speaker_encoder.onnx`
- `/home/harvest/qwen3-models/engines/orin-nx/highperf/...`
- `/home/harvest/project/repro-qwen3/jetson-voice/build/edgellm_voice_worker/workers/qwen3_asr_worker`
- `/home/harvest/project/repro-qwen3/TensorRT-Edge-LLM/build/examples/omni/qwen3_tts_worker`
- `/home/harvest/project/repro-qwen3/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so`
