# Earlier Single-run Latency Summary

These are historical measurements. The revised manuscript uses the [repeated workstation benchmark](../repeated_workstation_latency/README.md).

## Grouping rule

- All latency summaries below use `predicted_output` grouping only.
- `safe`: the model output is exactly `<SAFE/>`
- `hazard`: the model output is not `<SAFE/>`

## Core comparison

| Model | Output branch | Count | Avg. end-to-end latency (s) | P95 end-to-end (s) | Avg. generate-only (s) | Avg. tokens | Peak GPU mem. allocated (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B-VR-en | safe | 74 | 1.63 | 2.39 | 0.87 | 4.00 | 7.53 |
| Qwen2.5-VL-3B-VR-en | hazard | 126 | 3.36 | 4.41 | 2.50 | 42.72 | 7.53 |
| Qwen2.5-VL-7B-VR-en | safe | 67 | 1.63 | 1.84 | 1.36 | 4.00 | 16.05 |
| Qwen2.5-VL-7B-VR-en | hazard | 133 | 2.65 | 3.09 | 2.36 | 41.56 | 16.05 |

## Direct takeaways

- Both models show a clear latency split between the short `SAFE` branch and the longer hazard-guidance branch.
- The 3B model uses much less GPU memory than the 7B model (`7.53 GB` vs. `16.05 GB` allocated).
- For predicted-safe outputs, the 3B and 7B models have very similar average end-to-end latency (`1.63 s` vs. `1.63 s`).
- For predicted-hazard outputs, the 7B model is faster than the 3B model (`2.65 s` vs. `3.36 s`) under the current tested setting.

## Scope boundary

- These results are the primary formal Inference Efficiency Analysis evidence because they measure the manuscript models `Qwen2.5-VL-3B` and `Qwen2.5-VL-7B` directly.
- **Precision**: `torch_dtype="auto"` resolves to `bfloat16` for both models (`run_server_latency_benchmark.py:198`).
- **Input setting**: Qwen2.5-VL default dynamic resolution processor (`AutoProcessor.from_pretrained`), typical `min_pixels=256×28×28`, `max_pixels=1280×28×28`.
- Any mobile-side prototype results must be treated separately unless they use the same manuscript models under a clearly documented runtime stack.
- If an on-device section is added later, it should be framed as supplementary feasibility evidence rather than merged into this main table unless the model family and deployment path are directly comparable.
