# Inference Efficiency Analysis Workstation Latency Summary

## Grouping rule

- This summary uses `predicted_output` grouping only.
- `safe`: the model output is exactly `<SAFE/>`
- `hazard`: the model output is not `<SAFE/>`

## Core result

- Model: `qwen2.5vl-3b-VR-en`
- Hardware: `NVIDIA RTX A6000`
- Precision: `auto`
- Quantization: `none`
- Max new tokens: `128`
- Source run size: `200` images
- Predicted-safe outputs: `74`
- Predicted-hazard outputs: `126`
- Peak GPU memory: `7.53 GB allocated`, `7.88 GB reserved`

## Main numbers

- Predicted-safe outputs: average end-to-end latency `1.63 s`, P50 `1.51 s`, P95 `2.39 s`
- Predicted-hazard outputs: average end-to-end latency `3.36 s`, P50 `3.31 s`, P95 `4.41 s`
- Predicted-safe outputs: average generate-only latency `0.87 s`
- Predicted-hazard outputs: average generate-only latency `2.50 s`
- Predicted-safe outputs: average generated tokens `4.00`
- Predicted-hazard outputs: average generated tokens `42.72`

## Interpretation notes

- The latency gap is primarily driven by the model's actual output branch rather than by the nominal source subset.
- Outputs that terminate as `<SAFE/>` are substantially faster because they are much shorter on average.
- Outputs that contain hazard descriptions and guidance are slower because they require much longer text generation.

## Response-letter draft

To better reflect deployment-time behavior, we regrouped the workstation latency results by the model's actual output type rather than by the source subset from which the images were sampled. Specifically, we separated cases where the model output was exactly `<SAFE/>` from cases where the model produced hazard-aware `<ALERT>`/`<GUIDE>` text. On an NVIDIA RTX A6000 GPU, the fine-tuned Qwen2.5-VL-3B model showed an average end-to-end latency of `1.63 s` for predicted-safe outputs and `3.36 s` for predicted-hazard outputs, with a peak GPU memory usage of `7.53 GB` allocated (`7.88 GB` reserved). This split more directly reflects the practical latency difference between the short safe-response branch and the longer hazard-guidance branch.

## Manuscript-ready prose

To provide a deployment-oriented efficiency analysis, we regrouped latency results according to the model's actual output type. Cases where the model output was exactly `<SAFE/>` were treated as the safe-response branch, whereas all other outputs were treated as the hazard-guidance branch. On an NVIDIA RTX A6000 GPU, the fine-tuned Qwen2.5-VL-3B model achieved an average end-to-end latency of `1.63 s` for predicted-safe outputs and `3.36 s` for predicted-hazard outputs. The corresponding generate-only latencies were `0.87 s` and `2.50 s`, respectively. This difference is consistent with output length: predicted-safe outputs contained `4.00` generated tokens on average, whereas predicted-hazard outputs contained `42.72` tokens on average. The peak GPU memory usage was `7.53 GB` allocated (`7.88 GB` reserved).

## LaTeX table draft

```tex
\begin{table}[t]
\centering
\caption{Workstation-side inference efficiency of the fine-tuned Qwen2.5-VL-3B model, regrouped by the model's actual output type.}
\label{tab:workstation_latency_predicted_3b}
\begin{tabular}{lcccccc}
\toprule
Output branch & Hardware & Precision & Quantization & Avg. E2E Latency (s) & P95 E2E (s) & Peak GPU Mem. (GB) \\
\midrule
Predicted-safe   & NVIDIA RTX A6000 & auto & none & 1.63 & 2.39 & 7.53 \\
Predicted-hazard & NVIDIA RTX A6000 & auto & none & 3.36 & 4.41 & 7.53 \\
\bottomrule
\end{tabular}
\end{table}
```

## Detailed LaTeX table draft

```tex
\begin{table*}[t]
\centering
\caption{Detailed workstation-side inference benchmark for the fine-tuned Qwen2.5-VL-3B model on an NVIDIA RTX A6000 GPU, regrouped by predicted output type.}
\label{tab:workstation_latency_predicted_3b_detailed}
\begin{tabular}{lcccccccc}
\toprule
Output branch & Count & Avg. E2E & SD E2E & P50 E2E & P95 E2E & Avg. Gen-only & Avg. Tokens & Peak GPU Mem. \\
\midrule
Predicted-safe   & 74  & 1.63 & 0.40 & 1.51 & 2.39 & 0.87 & 4.00  & 7.53 \\
Predicted-hazard & 126 & 3.36 & 0.57 & 3.31 & 4.41 & 2.50 & 42.72 & 7.53 \\
\bottomrule
\end{tabular}
\end{table*}
```
