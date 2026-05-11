# Hugging Face Artifacts

Target repo: `harvestsu/qwen3-edgellm-jetson-artifacts`.

This repository should contain large generated artifacts only:

- ASR thinker engines and config/tokenizer sidecars
- ASR audio encoder engines
- TTS Talker config/tokenizer/text embedding sidecars
- TTS W8A16 explicit Talker engines
- TTS CodePredictor engines and auxiliary tensors
- TTS stateful Code2Wav engines and configs
- per-device manifests/checksums

Do not store these large files in GitHub. Keep the required relative paths in `deploy/artifacts/qwen3_manifest.json` aligned with Jetson Voice profiles.

## Stage and upload

If the source directory already matches the manifest-relative layout:

```bash
python3 scripts/package_qwen3_artifacts.py \
  --set orin-nano-highperf-2026-05-10 \
  --source-root /opt/models/qwen3-edgellm \
  --out /tmp/qwen3-hf-upload

hf upload harvestsu/qwen3-edgellm-jetson-artifacts /tmp/qwen3-hf-upload . \
  --repo-type model \
  --commit-message "Upload orin-nano highperf artifacts"
```

For scattered build outputs, repeat `--map RELATIVE_PATH=/actual/source/file` for each file that is not already under `--source-root`. The packager writes `checksums/<artifact-set>.json` with file sizes and SHA-256 digests.
