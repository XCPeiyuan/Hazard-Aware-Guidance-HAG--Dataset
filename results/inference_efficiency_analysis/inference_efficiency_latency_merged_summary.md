# [main] Inference Efficiency Analysis Latency Merged Summary

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

## Response-letter draft

To better reflect deployment-time behavior, we regrouped the workstation latency results by the model's actual output type rather than by the source subset from which the images were sampled. Specifically, we separated cases where the model output was exactly `<SAFE/>` from cases where the model produced hazard-aware `<ALERT>`/`<GUIDE>` text. On an NVIDIA RTX A6000 GPU, the fine-tuned Qwen2.5-VL-3B model achieved average end-to-end latencies of `1.63 s` for predicted-safe outputs and `3.36 s` for predicted-hazard outputs, with a peak GPU memory usage of `7.53 GB` allocated (`7.88 GB` reserved). Under the same grouping rule, the fine-tuned Qwen2.5-VL-7B model achieved `1.63 s` for predicted-safe outputs and `2.65 s` for predicted-hazard outputs, with a peak GPU memory usage of `16.05 GB` allocated (`16.55 GB` reserved). These results provide direct workstation-side efficiency evidence while also showing that latency depends strongly on whether the model terminates in the short safe-response branch or generates a longer hazard-guidance response.

If the response letter also mentions our mobile-side prototype exploration, it should be described separately as an exploratory feasibility check rather than as the primary benchmark for the manuscript models. In the current revision package, the formal reviewer-facing deployment evidence is the workstation-side latency/memory measurement for the fine-tuned Qwen2.5-VL-3B and Qwen2.5-VL-7B models, while mobile-side deployment constraints remain part of the limitation discussion.

## Manuscript-ready prose

To provide a deployment-oriented efficiency analysis, we regrouped latency results according to the model's actual output type. Cases where the model output was exactly `<SAFE/>` were treated as the safe-response branch, whereas all other outputs were treated as the hazard-guidance branch. On an NVIDIA RTX A6000 GPU, the fine-tuned Qwen2.5-VL-3B model achieved average end-to-end latencies of `1.63 s` for predicted-safe outputs and `3.36 s` for predicted-hazard outputs, while the fine-tuned Qwen2.5-VL-7B model achieved `1.63 s` and `2.65 s`, respectively. The corresponding peak GPU memory usage was `7.53 GB` allocated (`7.88 GB` reserved) for the 3B model and `16.05 GB` allocated (`16.55 GB` reserved) for the 7B model. In both cases, the hazard-guidance branch was slower because it required much longer generated outputs than the short `<SAFE/>` branch.

If mobile-side prototype evidence is cited elsewhere, the manuscript should explicitly state that such runs are exploratory engineering feasibility checks under a different deployment toolchain, rather than direct on-device benchmarks for the primary `Qwen2.5-VL-3B/7B` models reported in the paper.

## LaTeX table draft

```tex
\begin{table*}[t]
\centering
\caption{Workstation-side inference efficiency of the fine-tuned Qwen2.5-VL models on an NVIDIA RTX A6000 GPU, regrouped by predicted output type.}
\label{tab:workstation_latency_predicted_merged}
\begin{tabular}{lcccccccc}
\toprule
Model & Output branch & Count & Avg. E2E & P95 E2E & Avg. Gen-only & Avg. Tokens & Peak GPU Mem. \\
\midrule
Qwen2.5-VL-3B-VR-en & Predicted-safe   & 74  & 1.63 & 2.39 & 0.87 & 4.00  & 7.53 \\
Qwen2.5-VL-3B-VR-en & Predicted-hazard & 126 & 3.36 & 4.41 & 2.50 & 42.72 & 7.53 \\
Qwen2.5-VL-7B-VR-en & Predicted-safe   & 67  & 1.63 & 1.84 & 1.36 & 4.00  & 16.05 \\
Qwen2.5-VL-7B-VR-en & Predicted-hazard & 133 & 2.65 & 3.09 & 2.36 & 41.56 & 16.05 \\
\bottomrule
\end{tabular}
\end{table*}
```
