# M3 Step 2 — Measurement Spike Results

WAV: `/tmp/spike_input.wav` (8.64 s, truncated to 5.0 s)
Hop interval: 500 ms, N hops: 10
Worker init: 11901.062829 ms

## Per-hop timings

| hop | event | elapsed_ms | encoder_ms | prefill_ms | decode_ms | text |
|-----|-------|------------|------------|------------|-----------|------|
| 0 | partial | 958.4 | 269.4 | 134.0 | 491.1 | em. |
| 1 | partial | 118.7 | 16.8 | 18.5 | 76.7 | 语音。 |
| 2 | partial | 140.1 | 11.8 | 16.7 | 106.9 | 语音合成的。 |
| 3 | partial | 151.4 | 8.6 | 17.2 | 121.5 | 语音合成的稳定性。 |
| 4 | partial | 155.8 | 9.2 | 17.4 | 124.3 | 语音合成的稳定性。 |
| 5 | partial | 158.3 | 9.3 | 18.0 | 126.3 | 语音合成的稳定性。 |
| 6 | partial | 161.9 | 9.8 | 18.3 | 126.7 | 语音合成的稳定性。 |
| 7 | partial | 156.5 | 9.8 | 18.4 | 122.5 | 语音合成的稳定性。 |
| 8 | partial | 238.7 | 10.4 | 25.0 | 197.2 | 语音合成的稳定性、讯息清晰度。 |
| 9 | final | 179.5 | 10.5 | 25.2 | 137.9 | 语音合成的稳定性训练。 |

## Steady-state stats (hops 2..N-1, partial only)

- N samples: 7
- median elapsed: 156.5 ms
- p95 elapsed: 215.6 ms
- gate (median <= 500 ms): **PASS**

