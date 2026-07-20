# Inference Efficiency Analysis Workstation Latency Summary

## Grouping rule

- This summary uses `predicted_output` grouping only.
- `safe`: the model output is exactly `<SAFE/>`
- `hazard`: the model output is not `<SAFE/>`

## Core result

- Model: `Qwen2.5-VL-7B-VR-en`
- Hardware: `NVIDIA RTX A6000`
- Precision: `auto`
- Quantization: `none`
- Max new tokens: `128`
- Source run size: `200` images
- Predicted-safe outputs: `67`
- Predicted-hazard outputs: `133`
- Peak GPU memory: `16.05 GB allocated`, `16.55 GB reserved`

## Main numbers

- Predicted-safe outputs: average end-to-end latency `1.63 s`, P50 `1.60 s`, P95 `1.84 s`
- Predicted-hazard outputs: average end-to-end latency `2.65 s`, P50 `2.61 s`, P95 `3.09 s`
- Predicted-safe outputs: average generate-only latency `1.36 s`
- Predicted-hazard outputs: average generate-only latency `2.36 s`
- Predicted-safe outputs: average generated tokens `4.00`
- Predicted-hazard outputs: average generated tokens `41.56`

## Interpretation notes

- The latency gap is primarily driven by the model's actual output branch rather than by the nominal source subset.
- Outputs that terminate as `<SAFE/>` are substantially faster because they are much shorter on average.
- Outputs that contain hazard descriptions and guidance are slower because they require much longer text generation.

## Response-letter draft

To better reflect deployment-time behavior, we regrouped the workstation latency results by the model's actual output type rather than by the source subset from which the images were sampled. Specifically, we separated cases where the model output was exactly `<SAFE/>` from cases where the model produced hazard-aware `<ALERT>`/`<GUIDE>` text. On an NVIDIA RTX A6000 GPU, the fine-tuned Qwen2.5-VL-7B model showed an average end-to-end latency of `1.63 s` for predicted-safe outputs and `2.65 s` for predicted-hazard outputs, with a peak GPU memory usage of `16.05 GB` allocated (`16.55 GB` reserved). This split more directly reflects the practical latency difference between the short safe-response branch and the longer hazard-guidance branch.

## Manuscript-ready prose

To provide a deployment-oriented efficiency analysis, we regrouped latency results according to the model's actual output type. Cases where the model output was exactly `<SAFE/>` were treated as the safe-response branch, whereas all other outputs were treated as the hazard-guidance branch. On an NVIDIA RTX A6000 GPU, the fine-tuned Qwen2.5-VL-7B model achieved an average end-to-end latency of `1.63 s` for predicted-safe outputs and `2.65 s` for predicted-hazard outputs. The corresponding generate-only latencies were `1.36 s` and `2.36 s`, respectively. This difference is consistent with output length: predicted-safe outputs contained `4.00` generated tokens on average, whereas predicted-hazard outputs contained `41.56` tokens on average. The peak GPU memory usage was `16.05 GB` allocated (`16.55 GB` reserved).

## LaTeX table draft

```tex
\begin{table}[t]
\centering
\caption{Workstation-side inference efficiency of the fine-tuned Qwen2.5-VL-7B model, regrouped by the model's actual output type.}
\label{tab:workstation_latency_predicted}
\begin{tabular}{lcccccc}
\toprule
Output branch & Hardware & Precision & Quantization & Avg. E2E Latency (s) & P95 E2E (s) & Peak GPU Mem. (GB) \\
\midrule
Predicted-safe   & NVIDIA RTX A6000 & auto & none & 1.63 & 1.84 & 16.05 \\
Predicted-hazard & NVIDIA RTX A6000 & auto & none & 2.65 & 3.09 & 16.05 \\
\bottomrule
\end{tabular}
\end{table}
```

## Detailed LaTeX table draft

```tex
\begin{table*}[t]
\centering
\caption{Detailed workstation-side inference benchmark for the fine-tuned Qwen2.5-VL-7B model on an NVIDIA RTX A6000 GPU, regrouped by predicted output type.}
\label{tab:workstation_latency_predicted_detailed}
\begin{tabular}{lcccccccc}
\toprule
Output branch & Count & Avg. E2E & SD E2E & P50 E2E & P95 E2E & Avg. Gen-only & Avg. Tokens & Peak GPU Mem. \\
\midrule
Predicted-safe   & 67  & 1.63 & 0.12 & 1.60 & 1.84 & 1.36 & 4.00  & 16.05 \\
Predicted-hazard & 133 & 2.65 & 0.23 & 2.61 & 3.09 & 2.36 & 41.56 & 16.05 \\
\bottomrule
\end{tabular}
\end{table*}
```
