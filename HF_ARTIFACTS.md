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
