#!/usr/bin/env python3
"""M5 — End-to-end streaming verification on reproduction prompts.

For each (wav, ground_truth_text) tuple, this driver:

  1. Computes a one-shot ASR baseline via the worker's scenario A
     (single `handleRequest` JSON, same path as the HTTP /asr endpoint).
  2. Drives the streaming worker through scenario B (500 ms cumulative-
     mel hops, `mel_path` chunks).
  3. Drives the streaming worker through scenario F (500 ms cumulative
     PCM hops, `pcm_b64` chunks → worker-side C++ mel extractor).

Gates per prompt (all hard):
  - B LCS-similarity vs one-shot baseline >= --lcs-gate (default 0.95)
  - F LCS-similarity vs one-shot baseline >= --lcs-gate
  - Median end-of-speech latency (over --latency-runs streaming runs)
    <= --median-gate-ms (default 500)
  - p95   end-of-speech latency <= --p95-gate-ms (default 1000)

Output:
  - human-readable summary to stdout
  - optional --results-json: full per-prompt detail
  - optional --results-md:   markdown table

Exit code 0 iff every gate on every prompt passes.

Re-uses mel preprocessing and Worker driver from test_streaming_worker.py
(same directory). Designed to run inside `jetson_voice_slim` on
`orin-nx` for the M5 release gate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

# Re-use helpers from the M3 acceptance driver.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_streaming_worker import (  # type: ignore
    SAMPLE_RATE,
    HOP_SEC,
    Worker,
    audio_to_mel,
    lcs_similarity,
    percentile,
    resample_to_16k,
    scenario_a_oneshot,
    scenario_b_streaming,
    scenario_f_pcm_streaming,
    wav_to_audio,
)


def run_one(wav_path: Path, ground_truth: str, args: argparse.Namespace,
            mel_dir: Path) -> dict:
    """Run A + B + F + latency-runs on a single WAV. Returns a result dict."""
    audio_raw, sr = wav_to_audio(wav_path)
    audio = resample_to_16k(audio_raw, sr).astype(np.float32)
    duration_s = len(audio) / SAMPLE_RATE
    # Clamp to <= 5 s (engine max_input_len budget for one-shot).
    audio = audio[: int(SAMPLE_RATE * 5.0)]

    out: dict = {
        "wav": str(wav_path),
        "ground_truth": ground_truth,
        "duration_s": duration_s,
        "clipped_s": len(audio) / SAMPLE_RATE,
    }

    # --- A: one-shot baseline -------------------------------------------
    print(f"  [A] one-shot ...", file=sys.stderr, flush=True)
    w = Worker(args)
    try:
        a = scenario_a_oneshot(w, audio, mel_dir, args.max_gen)
    finally:
        w.close()
    out["A_oneshot"] = {"text": a["text"], "ok": a["ok"]}
    baseline_text = a["text"]
    print(f"      → {baseline_text!r}  ok={a['ok']}", file=sys.stderr, flush=True)

    # --- B: streaming (mel_path) -----------------------------------------
    print(f"  [B] streaming mel_path ...", file=sys.stderr, flush=True)
    w = Worker(args)
    try:
        b = scenario_b_streaming(w, audio, mel_dir)
    finally:
        w.close()
    lcs_b = lcs_similarity(b["text"], baseline_text)
    out["B_streaming_mel"] = {
        "text": b["text"],
        "lcs_vs_baseline": lcs_b,
        "end_latency_ms": b["end_latency_ms"],
        "n_hops": b["n_hops"],
        "rotations": b["rotations"],
    }
    print(f"      → {b['text']!r}  lcs={lcs_b:.3f}  latency={b['end_latency_ms']:.1f}ms",
          file=sys.stderr, flush=True)

    # --- F: streaming (pcm_b64) ------------------------------------------
    if args.mel_settings and args.mel_filters:
        print(f"  [F] streaming pcm_b64 ...", file=sys.stderr, flush=True)
        w = Worker(args)
        try:
            f = scenario_f_pcm_streaming(w, audio)
        finally:
            w.close()
        lcs_f = lcs_similarity(f["text"], baseline_text)
        out["F_streaming_pcm"] = {
            "text": f["text"],
            "lcs_vs_baseline": lcs_f,
            "end_latency_ms": f["end_latency_ms"],
            "n_hops": f["n_hops"],
            "rotations": f["rotations"],
        }
        print(f"      → {f['text']!r}  lcs={lcs_f:.3f}  latency={f['end_latency_ms']:.1f}ms",
              file=sys.stderr, flush=True)
    else:
        out["F_streaming_pcm"] = None
        print(f"  [F] SKIPPED — --mel-settings/--mel-filters not provided",
              file=sys.stderr, flush=True)

    # --- C: latency (run B repeatedly) -----------------------------------
    print(f"  [C] latency x{args.latency_runs} (streaming mel_path) ...",
          file=sys.stderr, flush=True)
    latencies = [b["end_latency_ms"]]
    for r in range(args.latency_runs - 1):
        w = Worker(args)
        try:
            bc = scenario_b_streaming(w, audio, mel_dir)
        finally:
            w.close()
        latencies.append(bc["end_latency_ms"])
    out["C_latency"] = {
        "runs_ms": latencies,
        "median_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }
    print(f"      median={out['C_latency']['median_ms']:.1f}ms  "
          f"p95={out['C_latency']['p95_ms']:.1f}ms",
          file=sys.stderr, flush=True)

    return out


def evaluate_gates(per_prompt: dict, args: argparse.Namespace) -> dict:
    """Compute per-prompt and aggregate gate verdicts."""
    gates_by_prompt: dict = {}
    for pid, res in per_prompt.items():
        g = {
            "A_ok": res["A_oneshot"]["ok"],
            f"B_lcs_ge_{args.lcs_gate}": res["B_streaming_mel"]["lcs_vs_baseline"]
                                          >= args.lcs_gate,
            f"C_median_le_{int(args.median_gate_ms)}ms":
                res["C_latency"]["median_ms"] <= args.median_gate_ms,
            f"C_p95_le_{int(args.p95_gate_ms)}ms":
                res["C_latency"]["p95_ms"] <= args.p95_gate_ms,
        }
        if res.get("F_streaming_pcm") is not None:
            g[f"F_lcs_ge_{args.lcs_gate}"] = (
                res["F_streaming_pcm"]["lcs_vs_baseline"] >= args.lcs_gate
            )
        gates_by_prompt[pid] = g
    all_pass = all(all(g.values()) for g in gates_by_prompt.values())
    return {"by_prompt": gates_by_prompt, "all_pass": all_pass}


def render_md(results: dict, args: argparse.Namespace) -> str:
    out = ["# M5 — End-to-End Streaming Verification Results", ""]
    out.append(f"Worker: `{args.worker}`")
    out.append(f"Gates: LCS ≥ {args.lcs_gate}, median ≤ {args.median_gate_ms} ms, "
               f"p95 ≤ {args.p95_gate_ms} ms")
    out.append("")
    out.append("## Aggregate")
    out.append("")
    out.append(f"**{'PASS' if results['gates']['all_pass'] else 'FAIL'}** — "
               f"{len(results['per_prompt'])} prompts evaluated")
    out.append("")
    out.append("## Per-prompt summary")
    out.append("")
    out.append("| Prompt | Ground truth | A baseline | B (mel) text | B LCS | "
               "F (pcm) text | F LCS | median ms | p95 ms | Verdict |")
    out.append("|--------|--------------|-----------|--------------|------:|"
               "--------------|------:|----------:|-------:|---------|")
    for pid, res in results["per_prompt"].items():
        g = results["gates"]["by_prompt"][pid]
        verdict = "PASS" if all(g.values()) else "FAIL"
        f_text = res["F_streaming_pcm"]["text"] if res.get("F_streaming_pcm") else "—"
        f_lcs = (f"{res['F_streaming_pcm']['lcs_vs_baseline']:.3f}"
                 if res.get("F_streaming_pcm") else "—")
        out.append(
            f"| {pid} | `{res['ground_truth']}` | `{res['A_oneshot']['text']}` | "
            f"`{res['B_streaming_mel']['text']}` | "
            f"{res['B_streaming_mel']['lcs_vs_baseline']:.3f} | "
            f"`{f_text}` | {f_lcs} | "
            f"{res['C_latency']['median_ms']:.1f} | "
            f"{res['C_latency']['p95_ms']:.1f} | {verdict} |"
        )
    out.append("")
    out.append("## Per-prompt gates")
    out.append("")
    for pid, g in results["gates"]["by_prompt"].items():
        out.append(f"### {pid}")
        out.append("")
        for k, v in g.items():
            out.append(f"- {k}: **{'PASS' if v else 'FAIL'}**")
        out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    # Multiple --prompt key=wav,gt entries, OR provide --prompts-json file.
    parser.add_argument("--prompt", action="append", default=[],
                        help="Format: 'id=path/to.wav|ground-truth-text'. Repeatable.")
    parser.add_argument("--prompts-json", default=None,
                        help="JSON file: {id: {wav: ..., text: ...}, ...}")
    parser.add_argument("--worker", default="/opt/jv-workers/qwen3_asr_worker")
    parser.add_argument("--plugin",
                        default="/opt/edgellm-bin/libNvInfer_edgellm_plugin.so")
    parser.add_argument("--engine-dir",
        default="/opt/models/qwen3-edgellm/engines/orin-nx/highperf/asr_thinker_full_fp8embed")
    parser.add_argument("--multimodal-engine-dir",
        default="/opt/models/qwen3-edgellm/engines/orin-nx/highperf/asr_audio_encoder")
    parser.add_argument("--mel-dir", default="/tmp/m5_e2e_mels")
    parser.add_argument("--mel-settings", default=None,
                        help="whisper_feature_extractor.json — enables PCM scenario F.")
    parser.add_argument("--mel-filters", default=None,
                        help="mel_filters.bin — required with --mel-settings.")
    parser.add_argument("--max-gen", type=int, default=200)
    parser.add_argument("--latency-runs", type=int, default=5)
    parser.add_argument("--lcs-gate", type=float, default=0.95)
    parser.add_argument("--median-gate-ms", type=float, default=500.0)
    parser.add_argument("--p95-gate-ms", type=float, default=1000.0)
    parser.add_argument("--results-md", default=None)
    parser.add_argument("--results-json", default=None)
    args = parser.parse_args()

    prompts: dict = {}
    if args.prompts_json:
        prompts.update(json.loads(Path(args.prompts_json).read_text("utf-8")))
    for spec in args.prompt:
        if "=" not in spec or "|" not in spec:
            print(f"bad --prompt spec: {spec!r}", file=sys.stderr)
            return 2
        pid, rest = spec.split("=", 1)
        wav, text = rest.split("|", 1)
        prompts[pid] = {"wav": wav, "text": text}
    if not prompts:
        print("no prompts supplied (use --prompt id=wav|gt or --prompts-json)",
              file=sys.stderr)
        return 2

    mel_dir = Path(args.mel_dir)
    mel_dir.mkdir(parents=True, exist_ok=True)

    per_prompt: dict = {}
    for pid, entry in prompts.items():
        print(f"\n=== prompt {pid}: {entry['wav']} ===", file=sys.stderr, flush=True)
        per_prompt[pid] = run_one(Path(entry["wav"]), entry["text"], args, mel_dir)

    gates = evaluate_gates(per_prompt, args)

    results = {"per_prompt": per_prompt, "gates": gates,
               "settings": {"lcs_gate": args.lcs_gate,
                            "median_gate_ms": args.median_gate_ms,
                            "p95_gate_ms": args.p95_gate_ms,
                            "latency_runs": args.latency_runs}}

    md = render_md(results, args)
    print(md)
    if args.results_md:
        Path(args.results_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results_md).write_text(md, encoding="utf-8")
    if args.results_json:
        Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results_json).write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    return 0 if gates["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
