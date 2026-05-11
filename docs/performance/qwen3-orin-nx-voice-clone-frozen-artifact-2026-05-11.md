# Qwen3 highperf Voice Clone — frozen artifact gate (2026-05-11)

Status: **partial PASS**.

| Path | Result |
|---|---|
| Default speaker (sid=2301) — preset speaker via embTable[speakerId] | ✅ exact-match ASR `今天天气真好。` |
| Alternate speaker (sid=2302) — preset via embTable[speakerId] | ✅ exact-match ASR `今天天气真好。` (different timbre) |
| Out-of-range speaker (sid=2303) — preset not in trained set | ❌ ASR `切。` (degenerate) |
| Voice clone (speaker_embedding_b64 from speaker_encoder.onnx) | ❌ ASR `啊！` / `嗯。` (degenerate) |

## What works

`sid` integer route is correctly wired through the highperf runtime
(commit `6239d5f` "support clone speaker slot"): the prefill kernel
reads `embTable[speakerId]` at row 6 of the assistant preamble, and the
frozen Talker engine handles any speaker ID it was trained against
(observed 2301 and 2302; 2303 out of range). This path is fully usable
for the small set of preset voices that ship with the model.

## What doesn't work (and why)

Zero-shot voice clone via `speaker_embedding_b64` (a raw 1024-d vector
extracted from a reference audio file by
`tts/speaker_encoder/speaker_encoder.onnx`) returns garbage audio. The
ASR transcribes the cloned output as a single character or onomatopoeic
particle regardless of the prompt or the reference voice.

Reason: the frozen Talker engine was trained to consume
`embTable[speakerId]` at the speaker conditioning row — a discrete set
of vectors corresponding to a small list of trained speakers. The
runtime's clone path replaces that row with the raw encoder output,
which is a different vector distribution than anything the Talker saw
at training time. The Talker collapses to a low-entropy filler token.

This is a model-side limitation of the current frozen artifacts, not a
build or runtime bug:

- The W8A16 kernel fix (`8a26eba` + `7ab7f1c`) is in place and verified
  by the `sid`-route exact matches above.
- The runtime correctly routes `speaker_embedding_b64` into row 6 per
  `6239d5f`.
- The Talker engine simply does not generalise to arbitrary speaker
  embeddings.

To unlock real zero-shot voice clone, the Talker needs to be
fine-tuned / re-trained with arbitrary speaker conditioning, then a new
engine baked + uploaded. That is out of scope for the 2026-05-11
clean-room + slim-docker work.

## Audio evidence

| File | Description |
|---|---|
| `docs/audio-evidence/nx-sid-2301-2026-05-11.wav` | Default preset speaker, prompt `今天天气真好。` |
| `docs/audio-evidence/nx-sid-2302-2026-05-11.wav` | Alternate preset speaker, same prompt |
| `docs/audio-evidence/nx-clone-embed-broken-2026-05-11.wav` | `/tts/clone/stream` output, prompt `今天天气真好。`, speaker_embedding from `bench/wavs/S1.wav` — listenable proof of the OOD failure |
| `docs/audio-evidence/voice-clone-reference-S1-2026-05-11.wav` | Reference WAV that the speaker encoder consumed |

## Recommendation

- Document the clone HTTP endpoint as **"reserved for future use; do
  not call against the 2026-05-11 frozen artifacts."**
- Keep `speaker_id` (`sid`) as the public knob for picking a preset
  voice.
- Add a fine-tune branch on the model when zero-shot clone becomes a
  product requirement.
