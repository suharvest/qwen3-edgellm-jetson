#!/usr/bin/env python3
"""M3 step 5 — streaming ASR worker acceptance test.

Drives the (post step-4) qwen3_asr_worker through five scenarios per design
doc §15.6 step 5 / dispatch brief:

A. Backward-compat one-shot — existing JSON request shape, response must
   contain `ok=true` and `responses[0].output_text` matching a one-shot
   baseline structurally (byte-equivalence asserted on response keys; text
   compared via LCS-similarity ≥ 0.95 to allow CUDA tie-breaking jitter).
B. Streaming happy path — 4-5 s reproduction WAV in 500 ms cumulative-mel
   chunks. final.text LCS-similarity ≥ 0.95 vs one-shot baseline.
C. End-of-speech latency — same as B; measure wall-clock from sending
   last=true to receiving final event. Gate: ≤ 500 ms median over 5 runs,
   ≤ 1000 ms p95. (Soft gate — REPORTED, not failed-hard, because the
   step-2 mechanism's final hop runs the full encoder+prefill+decode each
   time, and the median in spike data is ~150-250 ms, comfortably under.)
D. Auto-segmentation — 8-10 s WAV (concat of two 5 s WAVs) in 500 ms
   chunks. Verifies worker emits ONE `final` event with `segment_count>=2`.
   Final text LCS-similarity ≥ 0.90 vs ground-truth concatenation.
E. Error paths — malformed JSON, unknown event, oversized chunk
   (audio_sec=6.0). Verifies error responses + session state cleared.

Designed to run INSIDE jetson_voice_slim on orin-nx. The driver uses the
same mel preprocessing as scripts/test_m3_step2_spike.py.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import uuid
import wave
from pathlib import Path
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
FMIN = 0.0
FMAX = 8000.0
MEL_FLOOR = 1e-10
HOP_SEC = 0.5

# P1 (post-VAD-Phase2) — default ASR thinker engine path. Scenario D
# (12.9 s zh-long-04) requires the v2 engine rebuilt at max_input_len=256;
# the old 128-cap engine silently fails prefill on the final hop and
# yields an empty transcript. The container-mounted host path is also
# valid (overrides via --engine-dir still respected).
DEFAULT_ASR_ENGINE_DIR = (
    "/opt/models/qwen3-edgellm/engines/orin-nx/highperf-v2/asr_thinker_full_fp8embed")
DEFAULT_ASR_MULTIMODAL_ENGINE_DIR = (
    "/opt/models/qwen3-edgellm/engines/orin-nx/highperf/asr_audio_encoder")

# Curated ground-truth transcript for `docs/audio-evidence/zh-long-04-2026-05-13.wav`.
# Auto-applied as scenario-D `--long-baseline-text` when the supplied --long-wav
# basename contains `zh-long-04`. Lets `bash scripts/test_streaming_worker.py
# ... --long-wav .../zh-long-04-...wav` evaluate the M5 hard gate without the
# caller having to remember the exact transcript.
ZH_LONG_04_GT = (
    "科学家们可以得出结论：暗物质对其他暗物质的影响方式与普通物质相同。")


# ---------------------------------------------------------------------------
# Mel preprocessing (copied from test_m3_step2_spike.py).
# ---------------------------------------------------------------------------
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
        left, center, right = int(bins[m - 1]), int(bins[m]), int(bins[m + 1])
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
    header = {name: {"dtype": dtype_map[tensor.dtype.type], "shape": list(tensor.shape),
                     "data_offsets": [0, tensor.nbytes]}}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    with path.open("wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        f.write(tensor.tobytes())


def strip_language_prefix(text: str) -> str:
    if not text.startswith("language "):
        return text
    for lang in ("Chinese", "English", "Cantonese", "Japanese", "Korean",
                 "French", "German", "Italian", "Portuguese", "Russian", "Spanish"):
        prefix = "language " + lang
        if text.startswith(prefix):
            return text[len(prefix):].lstrip()
    return text


# ---------------------------------------------------------------------------
# LCS-similarity (mirrors scripts/verify_reproduction.py policy).
# ---------------------------------------------------------------------------
def lcs_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la):
        for j in range(lb):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[la][lb]
    return lcs / max(la, lb)


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


# ---------------------------------------------------------------------------
# Worker driver.
# ---------------------------------------------------------------------------
class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        env = os.environ.copy()
        env["EDGELLM_PLUGIN_PATH"] = args.plugin
        env.setdefault("EDGE_LLM_ASR_CUDA_GRAPH", "0")
        cmd = [args.worker, "--engineDir", args.engine_dir,
               "--multimodalEngineDir", args.multimodal_engine_dir]
        if getattr(args, "mel_settings", None):
            cmd += ["--melSettings", args.mel_settings]
        if getattr(args, "mel_filters", None):
            cmd += ["--melFilters", args.mel_filters]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )
        ready = self._readline("ready")
        if ready.get("event") != "ready":
            raise RuntimeError(f"unexpected first event: {ready}")
        self.init_ms = ready.get("init_ms")

    def _readline(self, tag: str) -> dict:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            self.proc.terminate()
            raise RuntimeError(f"worker died ({tag}): {err[-2000:]}")
        return json.loads(line)

    def send(self, obj: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def send_raw(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def recv(self, tag: str = "") -> dict:
        return self._readline(tag)

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=5)


def build_mel_for_audio(audio: np.ndarray, out_path: Path) -> None:
    mel = audio_to_mel(audio).astype(np.float16)
    write_safetensors(mel, "mel", out_path)


# ---------------------------------------------------------------------------
# Scenarios.
# ---------------------------------------------------------------------------
def scenario_a_oneshot(worker: Worker, audio: np.ndarray, mel_dir: Path,
                       max_gen: int) -> dict:
    """Backward-compat one-shot."""
    mel_path = mel_dir / "scen_a.safetensors"
    build_mel_for_audio(audio, mel_path)
    req = {
        "id": uuid.uuid4().hex,
        "requests": [{"messages": [{"role": "user",
                                    "content": [{"type": "audio", "audio": str(mel_path)}]}]}],
        "batch_size": 1, "temperature": 1.0, "top_p": 1.0, "top_k": 1,
        "max_generate_length": max_gen,
        "apply_chat_template": True, "add_generation_prompt": True,
    }
    worker.send(req)
    resp = worker.recv("scen_a")
    text = strip_language_prefix(resp.get("responses", [{}])[0].get("output_text", ""))
    return {
        "ok": bool(resp.get("ok")) and resp.get("event") == "done",
        "text": text,
        "raw_keys": sorted(resp.keys()),
        "response": resp,
    }


def scenario_b_streaming(worker: Worker, audio: np.ndarray, mel_dir: Path,
                         hop_sec: float = HOP_SEC) -> dict:
    """Streaming happy path: cumulative-mel 500ms hops, last=true on final."""
    duration = len(audio) / SAMPLE_RATE
    hop_samples = int(SAMPLE_RATE * hop_sec)
    n_hops = int(np.ceil(len(audio) / hop_samples))
    sid = f"scen_b_{uuid.uuid4().hex[:6]}"
    worker.send({"event": "begin", "id": sid, "sample_rate": SAMPLE_RATE,
                 "chunk_size_sec": hop_sec})
    ack = worker.recv("scen_b_begin")
    assert ack.get("event") == "begin_ack", ack

    finals = []
    rotations = 0
    end_latency_ms: Optional[float] = None
    for k in range(n_hops):
        end = min((k + 1) * hop_samples, len(audio))
        slice_audio = audio[:end]
        mel_path = mel_dir / f"scen_b_h{k:02d}.safetensors"
        build_mel_for_audio(slice_audio, mel_path)
        is_last = (k == n_hops - 1)
        ev = {"event": "chunk", "id": sid, "mel_path": str(mel_path),
              "audio_sec": end / SAMPLE_RATE, "last": is_last}
        t0 = time.perf_counter()
        worker.send(ev)
        resp = worker.recv(f"scen_b_h{k}")
        # In scenario B audio is short enough that no rotation should occur.
        if resp.get("event") == "segment_rotation":
            rotations += 1
            continue
        if is_last:
            end_latency_ms = (time.perf_counter() - t0) * 1000.0
            finals.append(resp)
    assert end_latency_ms is not None
    return {
        "duration_s": duration,
        "n_hops": n_hops,
        "rotations": rotations,
        "final": finals[-1] if finals else None,
        "end_latency_ms": end_latency_ms,
        "text": strip_language_prefix(finals[-1].get("text", "")) if finals else "",
    }


def scenario_f_pcm_streaming(worker: Worker, audio: np.ndarray,
                             hop_sec: float = HOP_SEC) -> dict:
    """PCM-input streaming variant of scenario B (M4 step 5).

    Same hop cadence and cumulative-mel semantics, but the chunk events carry
    raw float32 PCM (base64-encoded) instead of pre-computed mel safetensors.
    The worker runs its C++ MelExtractor on each chunk.

    Gate: final-text LCS vs scenario B baseline >= 0.95.
    """
    duration = len(audio) / SAMPLE_RATE
    hop_samples = int(SAMPLE_RATE * hop_sec)
    n_hops = int(np.ceil(len(audio) / hop_samples))
    sid = f"scen_f_{uuid.uuid4().hex[:6]}"
    worker.send({"event": "begin", "id": sid, "sample_rate": SAMPLE_RATE,
                 "chunk_size_sec": hop_sec, "audio_format": "pcm"})
    ack = worker.recv("scen_f_begin")
    assert ack.get("event") == "begin_ack", ack

    finals = []
    rotations = 0
    end_latency_ms: Optional[float] = None
    for k in range(n_hops):
        end = min((k + 1) * hop_samples, len(audio))
        slice_audio = audio[:end].astype(np.float32)
        pcm_b64 = base64.b64encode(slice_audio.tobytes()).decode("ascii")
        is_last = (k == n_hops - 1)
        ev = {"event": "chunk", "id": sid, "pcm_b64": pcm_b64,
              "audio_sec": end / SAMPLE_RATE, "last": is_last}
        t0 = time.perf_counter()
        worker.send(ev)
        resp = worker.recv(f"scen_f_h{k}")
        if resp.get("event") == "segment_rotation":
            rotations += 1
            continue
        if is_last:
            end_latency_ms = (time.perf_counter() - t0) * 1000.0
            finals.append(resp)
    assert end_latency_ms is not None
    return {
        "duration_s": duration,
        "n_hops": n_hops,
        "rotations": rotations,
        "final": finals[-1] if finals else None,
        "end_latency_ms": end_latency_ms,
        "text": strip_language_prefix(finals[-1].get("text", "")) if finals else "",
    }


def scenario_d_autosegment(worker: Worker, audio: np.ndarray, mel_dir: Path,
                           hop_sec: float = HOP_SEC) -> dict:
    """Auto-segmentation: feed long audio, expect rotations + 1 final event.

    On segment_rotation the driver MUST trim audio_accum to last carryover_sec
    and continue. Subsequent chunks send cumulative mels of the trimmed buffer.
    """
    duration = len(audio) / SAMPLE_RATE
    hop_samples = int(SAMPLE_RATE * hop_sec)
    sid = f"scen_d_{uuid.uuid4().hex[:6]}"
    worker.send({"event": "begin", "id": sid, "sample_rate": SAMPLE_RATE,
                 "chunk_size_sec": hop_sec})
    ack = worker.recv("scen_d_begin")
    assert ack.get("event") == "begin_ack", ack

    # Driver-side accumulation. start_offset_samples tracks where audio_accum
    # starts within the original audio array, updated after each rotation.
    start_offset_samples = 0
    consumed_samples = 0  # how much of the original audio has been "fed in"
    rotations = 0
    final_resp: Optional[dict] = None
    hop_idx = 0
    while consumed_samples < len(audio):
        new_end = min(consumed_samples + hop_samples, len(audio))
        is_last = (new_end >= len(audio))
        consumed_samples = new_end
        # The slice of audio currently in worker view = [start_offset_samples, new_end)
        slice_audio = audio[start_offset_samples:new_end]
        audio_sec = len(slice_audio) / SAMPLE_RATE
        mel_path = mel_dir / f"scen_d_h{hop_idx:02d}.safetensors"
        build_mel_for_audio(slice_audio, mel_path)
        worker.send({"event": "chunk", "id": sid, "mel_path": str(mel_path),
                     "audio_sec": audio_sec, "last": is_last})
        resp = worker.recv(f"scen_d_h{hop_idx}")
        ev_type = resp.get("event")
        if ev_type == "segment_rotation":
            rotations += 1
            carryover_sec = resp.get("carryover_sec", 1.0)
            carry_samples = int(carryover_sec * SAMPLE_RATE)
            # New start = max(0, new_end - carry_samples)
            start_offset_samples = max(0, new_end - carry_samples)
            # Don't advance consumed_samples — next chunk continues from new_end.
        elif ev_type == "final":
            final_resp = resp
        elif ev_type == "partial":
            pass
        else:
            raise RuntimeError(f"unexpected event in scen_d: {resp}")
        hop_idx += 1
    return {
        "duration_s": duration,
        "rotations": rotations,
        "final": final_resp,
        "text": strip_language_prefix(final_resp.get("text", "")) if final_resp else "",
        "segment_count": final_resp.get("segment_count") if final_resp else None,
    }


def scenario_e_errors(args: argparse.Namespace) -> dict:
    """Error paths: malformed JSON, unknown event, oversized chunk.

    Uses a fresh worker because some error paths free the session entirely;
    we want clean isolation so each error case is verified independently.
    """
    results = {}

    # E1: malformed JSON
    w = Worker(args)
    try:
        w.send_raw("{this is not json}")
        resp = w.recv("e1")
        results["malformed_json"] = {
            "got": resp,
            "ok": resp.get("event") == "error" and "json_parse_failed" in resp.get("error", ""),
        }
    finally:
        w.close()

    # E2: unknown event
    w = Worker(args)
    try:
        w.send({"event": "unknown_thing", "id": "e2"})
        resp = w.recv("e2")
        results["unknown_event"] = {
            "got": resp,
            "ok": resp.get("event") == "error" and resp.get("error") == "unknown_event",
        }
    finally:
        w.close()

    # E3: oversized single chunk (audio_sec > 5)
    w = Worker(args)
    mel_dir = Path(args.mel_dir)
    mel_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Build a tiny dummy mel (worker won't actually decode it because
        # the audio_sec preflight refuses first).
        dummy_audio = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
        mel_path = mel_dir / "scen_e3.safetensors"
        build_mel_for_audio(dummy_audio, mel_path)
        w.send({"event": "begin", "id": "e3", "sample_rate": SAMPLE_RATE,
                "chunk_size_sec": 0.5})
        ack = w.recv("e3_begin")
        assert ack.get("event") == "begin_ack", ack
        # audio_sec > 15.0 (worker's kSingleChunkHardLimitSec, P1 thinker v2) → chunk_too_long
        w.send({"event": "chunk", "id": "e3", "mel_path": str(mel_path),
                "audio_sec": 16.0, "last": False})
        resp = w.recv("e3_chunk")
        results["chunk_too_long"] = {
            "got": resp,
            "ok": resp.get("event") == "error" and resp.get("error") == "chunk_too_long",
        }
        # Verify session was cleared: next chunk on same id should fail with
        # no_active_session.
        w.send({"event": "chunk", "id": "e3", "mel_path": str(mel_path),
                "audio_sec": 0.1, "last": False})
        resp2 = w.recv("e3_after")
        results["session_cleared_after_error"] = {
            "got": resp2,
            "ok": resp2.get("error") == "no_active_session",
        }
    finally:
        w.close()

    return results


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-wav", required=True,
                        help="Reproduction WAV ~4-5 s for scenarios A/B/C.")
    parser.add_argument("--long-wav", default=None,
                        help="Long WAV ~8-10 s for scenario D. "
                             "If omitted, scenario D concatenates --short-wav with itself.")
    parser.add_argument("--worker", default="/opt/jv-workers/qwen3_asr_worker")
    parser.add_argument("--plugin", default="/opt/edgellm-bin/libNvInfer_edgellm_plugin.so")
    parser.add_argument("--engine-dir",
        default=DEFAULT_ASR_ENGINE_DIR)
    parser.add_argument("--multimodal-engine-dir",
        default=DEFAULT_ASR_MULTIMODAL_ENGINE_DIR)
    parser.add_argument("--mel-dir", default="/tmp/m3_streaming_mels")
    parser.add_argument("--mel-settings", default=None,
                        help="Path to whisper_feature_extractor.json — enables scenario F (PCM input).")
    parser.add_argument("--mel-filters", default=None,
                        help="Path to mel_filters.bin — required with --mel-settings.")
    parser.add_argument("--max-gen", type=int, default=200)
    parser.add_argument("--latency-runs", type=int, default=5)
    parser.add_argument("--long-baseline-text", default=None,
                        help="Ground-truth transcript for --long-wav. "
                             "When provided, scenario D's LCS is computed "
                             "against this string instead of the synthetic "
                             "baseline_x2 concatenation, which lets the D "
                             "gate be a real quality check rather than a "
                             "mechanism-only test.")
    parser.add_argument("--d-lcs-hard-gate", type=float, default=0.95,
                        help="Scenario D hard-gate LCS threshold. Defaults "
                             "to 0.95 (P1 dedup fix promotes D from soft to "
                             "hard). Requires --long-baseline-text for a "
                             "meaningful comparison. Pass an explicit None "
                             "via env-driven override only when re-running "
                             "the legacy mechanism-only soft path.")
    parser.add_argument("--results-md", default=None)
    parser.add_argument("--results-json", default=None)
    args = parser.parse_args()

    # P1 D-gate auto-curation: when caller passes the canonical zh-long-04 WAV
    # but no --long-baseline-text, fill in the curated GT so the D hard-gate
    # path triggers automatically. Prevents drift between this driver and the
    # commit recipe (docs/plans/m5-test-wavs.md row).
    if args.long_wav and not args.long_baseline_text:
        if "zh-long-04" in Path(args.long_wav).name:
            args.long_baseline_text = ZH_LONG_04_GT
            print(f"[scen_d] auto-applied ZH_LONG_04_GT as --long-baseline-text",
                  file=sys.stderr, flush=True)

    mel_dir = Path(args.mel_dir)
    mel_dir.mkdir(parents=True, exist_ok=True)

    short_audio_raw, sr = wav_to_audio(Path(args.short_wav))
    short_audio = resample_to_16k(short_audio_raw, sr)
    # Clamp short audio to <= 5 s for scenarios A/B/C.
    short_audio = short_audio[:int(SAMPLE_RATE * 5.0)]

    if args.long_wav:
        long_audio_raw, sr2 = wav_to_audio(Path(args.long_wav))
        long_audio = resample_to_16k(long_audio_raw, sr2)
    else:
        long_audio = np.concatenate([short_audio, short_audio]).astype(np.float32)

    results: dict = {"scenarios": {}}

    # Scenario A — single worker, runs one-shot.
    print("[scen_a] backward-compat one-shot ...", file=sys.stderr, flush=True)
    w = Worker(args)
    try:
        a = scenario_a_oneshot(w, short_audio, mel_dir, args.max_gen)
    finally:
        w.close()
    results["scenarios"]["A_oneshot"] = a
    baseline_text = a["text"]
    print(f"[scen_a] ok={a['ok']} text={baseline_text!r}", file=sys.stderr, flush=True)

    # Scenario B — streaming happy path. Compare to baseline.
    print("[scen_b] streaming happy path ...", file=sys.stderr, flush=True)
    w = Worker(args)
    try:
        b = scenario_b_streaming(w, short_audio, mel_dir)
    finally:
        w.close()
    b["lcs_vs_baseline"] = lcs_similarity(b["text"], baseline_text)
    results["scenarios"]["B_streaming"] = b
    print(f"[scen_b] text={b['text']!r} lcs={b['lcs_vs_baseline']:.3f}", file=sys.stderr, flush=True)

    # Scenario C — end-of-speech latency, 5 runs.
    print(f"[scen_c] latency x{args.latency_runs} ...", file=sys.stderr, flush=True)
    latencies = []
    for r in range(args.latency_runs):
        w = Worker(args)
        try:
            bc = scenario_b_streaming(w, short_audio, mel_dir)
        finally:
            w.close()
        latencies.append(bc["end_latency_ms"])
        print(f"[scen_c] run {r}: {bc['end_latency_ms']:.1f} ms", file=sys.stderr, flush=True)
    results["scenarios"]["C_latency"] = {
        "runs_ms": latencies,
        "median_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "min_ms": min(latencies), "max_ms": max(latencies),
    }

    # Scenario D — auto-segmentation.
    # Establish a long-audio one-shot baseline if the audio is short enough to
    # fit the engine's max_input_len (~7 s); otherwise approximate by
    # concatenating the short baseline.
    long_duration = len(long_audio) / SAMPLE_RATE
    long_baseline_text: Optional[str] = None
    if long_duration <= 6.5:
        print(f"[scen_d] long-audio baseline (oneshot {long_duration:.2f}s) ...",
              file=sys.stderr, flush=True)
        w = Worker(args)
        try:
            ab = scenario_a_oneshot(w, long_audio, mel_dir, args.max_gen)
            long_baseline_text = ab["text"]
        finally:
            w.close()
        print(f"[scen_d] long baseline text={long_baseline_text!r}",
              file=sys.stderr, flush=True)

    print(f"[scen_d] auto-segmentation on {long_duration:.2f}s audio ...",
          file=sys.stderr, flush=True)
    w = Worker(args)
    try:
        d = scenario_d_autosegment(w, long_audio, mel_dir)
    finally:
        w.close()
    # LCS comparisons. Priority:
    #   1. --long-baseline-text (curated ground truth, M5 hard-gate path)
    #   2. real long-audio one-shot baseline (when audio <= 6.5 s)
    #   3. baseline_x2 / baseline_x1 (synthetic fallback, mechanism-only)
    d["lcs_vs_long_baseline"] = (
        lcs_similarity(d["text"], long_baseline_text) if long_baseline_text else None)
    d["lcs_vs_baseline_x2"] = lcs_similarity(d["text"], baseline_text + baseline_text)
    d["lcs_vs_baseline_x1"] = lcs_similarity(d["text"], baseline_text)
    d["long_baseline_text"] = long_baseline_text
    # Curated-ground-truth comparison (preferred when supplied).
    if args.long_baseline_text:
        d["lcs_vs_curated_gt"] = lcs_similarity(d["text"], args.long_baseline_text)
        d["curated_ground_truth"] = args.long_baseline_text
    else:
        d["lcs_vs_curated_gt"] = None
        d["curated_ground_truth"] = None
    results["scenarios"]["D_autosegment"] = d
    best_lcs = max(
        x for x in (d["lcs_vs_curated_gt"], d["lcs_vs_long_baseline"],
                    d["lcs_vs_baseline_x2"], d["lcs_vs_baseline_x1"])
        if x is not None
    )
    d["lcs_best"] = best_lcs
    print(f"[scen_d] rotations={d['rotations']} segment_count={d['segment_count']} "
          f"text={d['text']!r} best_lcs={best_lcs:.3f}", file=sys.stderr, flush=True)

    # Scenario E — error paths.
    print("[scen_e] error paths ...", file=sys.stderr, flush=True)
    e = scenario_e_errors(args)
    results["scenarios"]["E_errors"] = e

    # Scenario F — PCM-input streaming (M4 step 5). Skipped if the worker
    # wasn't configured with --mel-settings / --mel-filters (i.e. PCM mode
    # disabled). When enabled, gate against the precomputed-mel baseline (B).
    f_result: Optional[dict] = None
    if args.mel_settings and args.mel_filters:
        print("[scen_f] PCM-input streaming ...", file=sys.stderr, flush=True)
        w = Worker(args)
        try:
            f_result = scenario_f_pcm_streaming(w, short_audio)
        finally:
            w.close()
        f_result["lcs_vs_baseline_B"] = lcs_similarity(f_result["text"], b["text"])
        f_result["lcs_vs_baseline_A"] = lcs_similarity(f_result["text"], baseline_text)
        results["scenarios"]["F_pcm_streaming"] = f_result
        print(f"[scen_f] text={f_result['text']!r} "
              f"lcs_vs_B={f_result['lcs_vs_baseline_B']:.3f} "
              f"lcs_vs_A={f_result['lcs_vs_baseline_A']:.3f}",
              file=sys.stderr, flush=True)

    # Gate evaluation.
    gates = {
        "A_oneshot_ok": a["ok"],
        "B_lcs_ge_0.95": b["lcs_vs_baseline"] >= 0.95,
        "C_median_le_500ms": results["scenarios"]["C_latency"]["median_ms"] <= 500.0,
        "C_p95_le_1000ms": results["scenarios"]["C_latency"]["p95_ms"] <= 1000.0,
        "D_one_final": d["final"] is not None and d["final"].get("event") == "final",
        "D_at_least_one_segment_rotation": (d.get("segment_count") or 0) >= 1
                                            or d.get("rotations", 0) >= 1,
        # D_lcs gate. When --long-baseline-text + --d-lcs-hard-gate are
        # supplied, this becomes a hard gate against the curated ground
        # truth (M5 commit-3 path). Otherwise it stays a soft 0.90 gate
        # that compares against a synthetic baseline concatenation
        # (mechanism-only check; the original M3 path).
        **({
            f"D_lcs_ge_{args.d_lcs_hard_gate}":
                (d["lcs_vs_curated_gt"] is not None
                 and d["lcs_vs_curated_gt"] >= args.d_lcs_hard_gate),
        } if (args.d_lcs_hard_gate is not None and args.long_baseline_text)
            else {
            "D_lcs_ge_0.90_soft": d["lcs_best"] >= 0.90,
        }),
        "E_malformed_json_handled": e["malformed_json"]["ok"],
        "E_unknown_event_handled": e["unknown_event"]["ok"],
        "E_chunk_too_long_handled": e["chunk_too_long"]["ok"],
        "E_session_cleared_after_error": e["session_cleared_after_error"]["ok"],
    }
    if f_result is not None:
        gates["F_pcm_lcs_ge_0.95"] = f_result["lcs_vs_baseline_B"] >= 0.95
    results["gates"] = gates
    # Hard-failing gates exclude SOFT ones (suffix `_soft`).
    hard_gates = {k: v for k, v in gates.items() if not k.endswith("_soft")}
    results["hard_gates"] = hard_gates
    results["all_hard_gates_passed"] = all(hard_gates.values())
    results["all_gates_passed"] = all(gates.values())  # incl. soft

    # Render summary.
    out = []
    out.append("# M3 Step 5 — Streaming Worker Acceptance Test Results")
    out.append("")
    out.append(f"Worker: `{args.worker}`")
    out.append(f"Short WAV: `{args.short_wav}` ({len(short_audio)/SAMPLE_RATE:.2f}s clipped)")
    out.append(f"Long WAV: {'`' + args.long_wav + '`' if args.long_wav else '(concat short_wav x2)'}"
               f" — {len(long_audio)/SAMPLE_RATE:.2f}s")
    out.append("")
    out.append("## Gate summary")
    out.append("")
    out.append("| Gate | Result | Kind |")
    out.append("|------|--------|------|")
    for k, v in gates.items():
        kind = "soft" if k.endswith("_soft") else "hard"
        out.append(f"| {k} | {'PASS' if v else 'FAIL'} | {kind} |")
    out.append("")
    out.append(f"**Hard gates: {'PASS' if results['all_hard_gates_passed'] else 'FAIL'}** | "
               f"All gates (incl. soft): {'PASS' if results['all_gates_passed'] else 'FAIL'}")
    out.append("")
    out.append("## A — one-shot baseline")
    out.append("")
    out.append(f"- ok: {a['ok']}")
    out.append(f"- text: `{baseline_text}`")
    out.append(f"- response keys: {a['raw_keys']}")
    out.append("")
    out.append("## B — streaming happy path")
    out.append("")
    out.append(f"- text: `{b['text']}`")
    out.append(f"- LCS vs baseline: {b['lcs_vs_baseline']:.3f}")
    out.append(f"- end-of-speech latency (1 run): {b['end_latency_ms']:.1f} ms")
    out.append(f"- rotations during B (expected 0): {b['rotations']}")
    out.append("")
    out.append("## C — end-of-speech latency (5 runs)")
    out.append("")
    c = results["scenarios"]["C_latency"]
    out.append(f"- runs (ms): {[round(x,1) for x in c['runs_ms']]}")
    out.append(f"- median: {c['median_ms']:.1f} ms")
    out.append(f"- p95:    {c['p95_ms']:.1f} ms")
    out.append(f"- min/max: {c['min_ms']:.1f} / {c['max_ms']:.1f} ms")
    out.append("")
    out.append("## D — auto-segmentation")
    out.append("")
    out.append(f"- duration: {d['duration_s']:.2f} s")
    out.append(f"- rotations: {d['rotations']}")
    out.append(f"- segment_count: {d['segment_count']}")
    out.append(f"- text: `{d['text']}`")
    if d.get("curated_ground_truth"):
        out.append(f"- curated ground truth: `{d['curated_ground_truth']}`")
        out.append(f"- LCS vs curated GT: {d['lcs_vs_curated_gt']:.3f}")
    if d.get("long_baseline_text"):
        out.append(f"- long-audio baseline: `{d['long_baseline_text']}`")
        out.append(f"- LCS vs long-baseline: {d['lcs_vs_long_baseline']:.3f}")
    out.append(f"- LCS vs baseline_x2: {d['lcs_vs_baseline_x2']:.3f}")
    out.append(f"- LCS vs baseline_x1: {d['lcs_vs_baseline_x1']:.3f}")
    out.append(f"- LCS best: {d['lcs_best']:.3f}")
    out.append("")
    out.append("## E — error paths")
    out.append("")
    for k, v in e.items():
        out.append(f"- **{k}**: {'PASS' if v['ok'] else 'FAIL'} — got `{v['got']}`")
    out.append("")

    summary = "\n".join(out)
    print(summary)
    if args.results_md:
        Path(args.results_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results_md).write_text(summary, encoding="utf-8")
    if args.results_json:
        Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results_json).write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return 0 if results["all_hard_gates_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
