#!/usr/bin/env python3
"""
Extract WhisperFeatureExtractor settings + mel filter bank from the upstream
Qwen3-ASR model snapshot.

The C++ MelExtractor (native/edgellm_voice_worker/mel_extractor.cpp) loads
these two artifacts at startup. Re-running this script is the canonical way to
regenerate them when upstream model/processor configuration changes.

Run on a host with the qwen-asr Python env, e.g. on wsl2-local:
    fleet exec wsl2-local -- 'source /home/harve/qwen3-asr-vllm-env/bin/activate \\
        && cd <repo> && python scripts/extract_whisper_feature_extractor.py'

Outputs:
    deploy/audio_preprocessing/whisper_feature_extractor.json
    deploy/audio_preprocessing/mel_filters.bin   (float32, [n_freq, n_mels])
    deploy/audio_preprocessing/mel_filters.sha256

See deploy/audio_preprocessing/README.md for layout + regeneration policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import WhisperFeatureExtractor


DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "deploy" / "audio_preprocessing"


def extract(model_name: str, out_dir: Path) -> dict:
    fe = WhisperFeatureExtractor.from_pretrained(model_name)

    mel_filters = np.asarray(fe.mel_filters)
    if mel_filters.ndim != 2:
        raise RuntimeError(f"unexpected mel_filters ndim {mel_filters.ndim}")
    # HF layout: (num_freq_bins, num_mel_filters) — same shape used by
    # transformers.audio_utils.spectrogram via `mel_filters.T @ spec`.
    n_freq, n_mels = mel_filters.shape
    expected_n_freq = fe.n_fft // 2 + 1
    if n_freq != expected_n_freq:
        raise RuntimeError(
            f"mel_filters first dim {n_freq} != n_fft//2+1 ({expected_n_freq})")
    if n_mels != fe.feature_size:
        raise RuntimeError(
            f"mel_filters second dim {n_mels} != feature_size {fe.feature_size}")

    mel_filters_f32 = mel_filters.astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "mel_filters.bin"
    mel_filters_f32.tofile(bin_path)
    sha = hashlib.sha256(bin_path.read_bytes()).hexdigest()
    (out_dir / "mel_filters.sha256").write_text(sha + "  mel_filters.bin\n")

    settings = {
        "source_model": model_name,
        "feature_extractor_type": getattr(fe, "feature_extractor_type", "WhisperFeatureExtractor"),
        "sampling_rate": int(fe.sampling_rate),
        "n_fft": int(fe.n_fft),
        "hop_length": int(fe.hop_length),
        "n_mels": int(fe.feature_size),
        "chunk_length_sec": int(fe.chunk_length),
        "n_samples_max": int(fe.n_samples),
        "padding_value": float(fe.padding_value),
        "dither": float(getattr(fe, "dither", 0.0)),
        "window": "hann_periodic",  # WhisperFeatureExtractor uses periodic hann
        "stft_center": True,
        "stft_pad_mode": "reflect",
        "power": 2.0,
        "mel_floor": 1e-10,
        "log_base": "log10",
        "drop_last_frame": True,  # log_spec[:, :-1]
        "clamp_to_max_minus_8": True,
        "post_normalize": "(x + 4) / 4",
        "mel_filters": {
            "path": "mel_filters.bin",
            "dtype": "float32",
            "shape": [n_freq, n_mels],
            "layout": "row-major; transpose then matmul against power spec",
            "sha256": sha,
        },
    }
    json_path = out_dir / "whisper_feature_extractor.json"
    json_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False))

    return {
        "settings": settings,
        "json_path": str(json_path),
        "bin_path": str(bin_path),
        "sha256": sha,
        "shape": [n_freq, n_mels],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HF model id or local path (default: {DEFAULT_MODEL})")
    p.add_argument("--out", type=Path, default=OUT_DIR,
                   help="output directory (default: deploy/audio_preprocessing/)")
    args = p.parse_args()

    result = extract(args.model, args.out)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
