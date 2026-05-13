#!/usr/bin/env python3
"""M3 step 2 measurement spike — per-hop latency capture.

Drives a 5 s WAV through the streaming ASR worker in 500 ms cumulative
chunks WITHOUT prefix prompt logic. Each chunk event points at a mel
safetensors file containing the FULL audio buffered so far (hop k → first
500*(k+1) ms of audio). The worker runs handleRequest from scratch per
hop. We capture per-hop wall-clock + per-stage timings (encoder /
prefill / decode).

Validates / falsifies design doc §15.3 / §15.4 latency estimates BEFORE
step 3 builds prefix-prompt logic on top.

This script is intended to run INSIDE the jetson_voice_slim container on
orin-nx (so it can speak directly to the worker binary).

Usage:
  python3 scripts/test_m3_step2_spike.py \
      --wav /path/to/5s.wav \
      --worker /opt/jv-workers/qwen3_asr_worker \
      --plugin /opt/edgellm-bin/libNvInfer_edgellm_plugin.so \
      --engine-dir /opt/models/qwen3-edgellm/engines/orin-nx/highperf/asr_thinker_full_fp8embed \
      --multimodal-engine-dir /opt/models/qwen3-edgellm/engines/orin-nx/highperf/asr_audio_encoder \
      --results-md /opt/qwen3-edgellm-jetson/docs/plans/m3-spike-step2-results.md
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
import uuid
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
FMIN = 0.0
FMAX = 8000.0
MEL_FLOOR = 1e-10
HOP_SEC = 0.5
N_HOPS = 10  # 5 s audio @ 500 ms hop.


def hz_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank() -> np.ndarray:
    n_freq = N_FFT // 2 + 1
    low_mel = hz_to_mel(np.float64(FMIN))
    high_mel = hz_to_mel(np.float64(FMAX))
    mel_points = np.linspace(low_mel, high_mel, N_MELS + 2, dtype=np.float64)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_freq - 1) * hz_points / FMAX).astype(np.int32)
    bins = np.clip(bins, 0, n_freq - 1)
    fb = np.zeros((N_MELS, n_freq), dtype=np.float64)
    for m in range(1, N_MELS + 1):
        left = int(bins[m - 1])
        center = int(bins[m])
        right = int(bins[m + 1])
        if left != center:
            for i in range(left, center):
                fb[m - 1, i] = (i - left) / (center - left)
        if center != right:
            for i in range(center, right):
                fb[m - 1, i] = (right - i) / (right - center)
    widths = hz_points[2:] - hz_points[:-2]
    fb *= (2.0 / widths)[:, np.newaxis]
    return fb.astype(np.float32)


MEL_FILTERBANK = build_mel_filterbank()


def wav_to_audio(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        sr = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sr


def resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return audio
    new_len = int(round(len(audio) * SAMPLE_RATE / sr))
    src_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def audio_to_mel(audio: np.ndarray) -> np.ndarray:
    pad = N_FFT // 2
    if audio.shape[0] <= 1:
        audio = np.pad(audio, (0, 2 - audio.shape[0]), mode="constant")
    audio = np.pad(audio, (pad, pad), mode="reflect")
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    n_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, N_FFT),
        strides=(audio.strides[0] * HOP_LENGTH, audio.strides[0]),
    )
    stft = np.fft.rfft(frames * window[np.newaxis, :], n=N_FFT, axis=1)
    magnitudes = np.abs(stft[:-1].T).astype(np.float32) ** 2.0
    mel_spec = MEL_FILTERBANK @ magnitudes
    log_spec = np.log10(np.maximum(mel_spec, MEL_FLOOR))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    if log_spec.shape[1] < 100:
        log_spec = np.pad(log_spec, ((0, 0), (0, 100 - log_spec.shape[1])), mode="constant")
    return log_spec[np.newaxis, :, :].astype(np.float32)


def write_safetensors(tensor: np.ndarray, name: str, path: Path) -> None:
    dtype_map = {np.float16: "F16", np.float32: "F32"}
    header = {
        name: {
            "dtype": dtype_map[tensor.dtype.type],
            "shape": list(tensor.shape),
            "data_offsets": [0, tensor.nbytes],
        }
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    with path.open("wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        f.write(tensor.tobytes())


def build_cumulative_mels(audio_16k: np.ndarray, out_dir: Path) -> list[Path]:
    """Build N_HOPS cumulative mel files: hop k covers samples [0 .. (k+1)*hop_samples].

    Returns list of paths in hop order.
    """
    hop_samples = int(SAMPLE_RATE * HOP_SEC)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for k in range(N_HOPS):
        end = (k + 1) * hop_samples
        slice_audio = audio_16k[:end] if end <= len(audio_16k) else audio_16k
        mel = audio_to_mel(slice_audio).astype(np.float16)
        p = out_dir / f"hop_{k:02d}.safetensors"
        write_safetensors(mel, "mel", p)
        paths.append(p)
    return paths


def strip_language_prefix(text: str) -> str:
    if not text.startswith("language "):
        return text
    for lang in (
        "Chinese", "English", "Cantonese", "Japanese", "Korean",
        "French", "German", "Italian", "Portuguese", "Russian", "Spanish",
    ):
        prefix = "language " + lang
        if text.startswith(prefix):
            return text[len(prefix):].lstrip()
    return text


def run_spike(args: argparse.Namespace) -> dict:
    audio_raw, sr = wav_to_audio(Path(args.wav))
    audio = resample_to_16k(audio_raw, sr)
    duration = len(audio) / SAMPLE_RATE
    need_samples = int(SAMPLE_RATE * HOP_SEC * N_HOPS)
    if len(audio) < need_samples:
        # Pad with silence (rare; reproduction WAVs are >=5 s).
        audio = np.pad(audio, (0, need_samples - len(audio)), mode="constant")
        truncated = False
    else:
        audio = audio[:need_samples]
        truncated = duration > HOP_SEC * N_HOPS

    mel_dir = Path(args.mel_dir or "/tmp/m3_spike_mels")
    mel_paths = build_cumulative_mels(audio, mel_dir)

    env = os.environ.copy()
    env["EDGELLM_PLUGIN_PATH"] = args.plugin
    env.setdefault("EDGE_LLM_ASR_CUDA_GRAPH", "0")
    cmd = [args.worker, "--engineDir", args.engine_dir, "--multimodalEngineDir", args.multimodal_engine_dir]
    print(f"[driver] launching worker: {' '.join(cmd)}", file=sys.stderr, flush=True)
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, env=env,
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None

    def readline_or_die(tag: str) -> dict:
        line = proc.stdout.readline()
        if not line:
            err = proc.stderr.read()
            proc.terminate()
            raise RuntimeError(f"worker died ({tag}): {err[-2000:]}")
        return json.loads(line)

    ready = readline_or_die("ready")
    if ready.get("event") != "ready":
        raise RuntimeError(f"unexpected first event: {ready}")
    init_ms = ready.get("init_ms")

    session_id = f"spike_{uuid.uuid4().hex[:8]}"
    begin = {
        "event": "begin",
        "id": session_id,
        "sample_rate": SAMPLE_RATE,
        "chunk_size_sec": HOP_SEC,
    }
    proc.stdin.write(json.dumps(begin) + "\n")
    proc.stdin.flush()
    ack = readline_or_die("begin_ack")
    if ack.get("event") != "begin_ack":
        raise RuntimeError(f"begin_ack expected, got {ack}")

    hops: list[dict] = []
    for k, mp in enumerate(mel_paths):
        is_last = k == len(mel_paths) - 1
        chunk = {"event": "chunk", "id": session_id, "mel_path": str(mp), "last": is_last}
        t0 = time.perf_counter()
        proc.stdin.write(json.dumps(chunk) + "\n")
        proc.stdin.flush()
        resp = readline_or_die(f"hop_{k}")
        driver_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ev = resp.get("event", "")
        if ev not in ("partial", "final"):
            raise RuntimeError(f"unexpected event at hop {k}: {resp}")
        hops.append({
            "hop_id": resp.get("hop_id", k),
            "event": ev,
            "ok": resp.get("ok", False),
            "elapsed_ms": resp.get("elapsed_ms", driver_elapsed_ms),
            "driver_elapsed_ms": driver_elapsed_ms,
            "encoder_ms": resp.get("encoder_ms", 0.0),
            "prefill_ms": resp.get("prefill_ms", 0.0),
            "decode_ms": resp.get("decode_ms", 0.0),
            "text": strip_language_prefix(resp.get("text", "")),
        })

    # Tidy: close worker.
    try:
        proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=5)
    stderr_tail = proc.stderr.read()[-4000:] if proc.stderr else ""

    return {
        "wav": str(args.wav),
        "wav_duration_s": round(duration, 3),
        "truncated": truncated,
        "hop_sec": HOP_SEC,
        "n_hops": N_HOPS,
        "init_ms": init_ms,
        "hops": hops,
        "stderr_tail": stderr_tail,
    }


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return xs_sorted[f]
    return xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f)


def render_table(rows: list[dict]) -> str:
    out = ["| hop | event | elapsed_ms | encoder_ms | prefill_ms | decode_ms | text |",
           "|-----|-------|------------|------------|------------|-----------|------|"]
    for r in rows:
        text = (r["text"] or "").replace("|", "\\|").replace("\n", " ")
        if len(text) > 80:
            text = text[:77] + "..."
        out.append(f"| {r['hop_id']} | {r['event']} | {r['elapsed_ms']:.1f} | "
                   f"{r['encoder_ms']:.1f} | {r['prefill_ms']:.1f} | {r['decode_ms']:.1f} | {text} |")
    return "\n".join(out)


def summarize(result: dict, gate_ms: float) -> str:
    hops = result["hops"]
    # Steady-state: hops 2+ (where prefix would activate). Final hop excluded
    # since it normally fires on `last=true` immediately after a hop.
    steady = [h for h in hops if h["hop_id"] >= 2 and h["event"] == "partial"]
    elapsed = [h["elapsed_ms"] for h in steady]
    med = percentile(elapsed, 50) if elapsed else 0.0
    p95 = percentile(elapsed, 95) if elapsed else 0.0
    verdict = "PASS" if med <= gate_ms else "FAIL"

    lines = []
    lines.append("# M3 Step 2 — Measurement Spike Results")
    lines.append("")
    lines.append(f"WAV: `{result['wav']}` ({result['wav_duration_s']} s, "
                 f"{'truncated to' if result['truncated'] else 'used as'} "
                 f"{N_HOPS * HOP_SEC:.1f} s)")
    lines.append(f"Hop interval: {HOP_SEC*1000:.0f} ms, N hops: {N_HOPS}")
    lines.append(f"Worker init: {result['init_ms']} ms")
    lines.append("")
    lines.append("## Per-hop timings")
    lines.append("")
    lines.append(render_table(hops))
    lines.append("")
    lines.append("## Steady-state stats (hops 2..N-1, partial only)")
    lines.append("")
    lines.append(f"- N samples: {len(elapsed)}")
    lines.append(f"- median elapsed: {med:.1f} ms")
    lines.append(f"- p95 elapsed: {p95:.1f} ms")
    lines.append(f"- gate (median <= {gate_ms:.0f} ms): **{verdict}**")
    lines.append("")
    if verdict == "FAIL":
        lines.append("## Recommendation")
        lines.append("")
        lines.append(f"- Median {med:.0f} ms exceeds hop interval {HOP_SEC*1000:.0f} ms.")
        lines.append("- Options:")
        lines.append("  - Raise `chunk_size_sec` to "
                     f"{max(med / 1000.0 * 1.2, HOP_SEC*2):.2f} s (1.2x median).")
        lines.append("  - Tighten `max_decode_tokens_per_hop` (currently 200, design "
                     "default 64).")
        lines.append("  - Note: step 3 prefix prompt should reduce decode_ms; remeasure "
                     "after step 3 before deciding.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--worker", default="/opt/jv-workers/qwen3_asr_worker")
    parser.add_argument("--plugin", default="/opt/edgellm-bin/libNvInfer_edgellm_plugin.so")
    parser.add_argument("--engine-dir",
        default="/opt/models/qwen3-edgellm/engines/orin-nx/highperf/asr_thinker_full_fp8embed")
    parser.add_argument("--multimodal-engine-dir",
        default="/opt/models/qwen3-edgellm/engines/orin-nx/highperf/asr_audio_encoder")
    parser.add_argument("--mel-dir", default="/tmp/m3_spike_mels")
    parser.add_argument("--results-md", default=None,
        help="Optional path to write summary markdown.")
    parser.add_argument("--results-json", default=None,
        help="Optional path to write raw result JSON.")
    parser.add_argument("--gate-ms", type=float, default=500.0,
        help="Median elapsed gate for hops 2+ (default 500 ms).")
    args = parser.parse_args()

    result = run_spike(args)
    summary = summarize(result, args.gate_ms)
    print(summary)
    if result["stderr_tail"]:
        print("\n--- worker stderr (tail) ---\n" + result["stderr_tail"], file=sys.stderr)

    if args.results_md:
        Path(args.results_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results_md).write_text(summary, encoding="utf-8")
    if args.results_json:
        Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
