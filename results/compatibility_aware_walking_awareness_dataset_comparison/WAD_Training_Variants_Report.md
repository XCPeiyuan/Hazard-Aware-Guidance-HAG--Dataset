# WAD 图像输入训练变体评估结果概要

本实验比较两个使用相同 1,000 个 WAD 样本训练的 Qwen2.5-VL 模型：主实验 alter-only supervision，以及辅助 full-JSON multitask supervision。两者均采用 image-only 输入，并分别在本文非-SAFE测试子集和 WAD 测试集上评估。

由于 WalkVLM/WAD 的原始任务是基于图像与 scene summary 的单句 walking reminder 生成，而本文任务是直接由图像生成结构化 ALERT/GUIDE，二者的数据 schema、训练输入和输出目标均不完全一致。因此该实验应作为面向审稿意见的兼容性诊断比较，而不是完全公平的 apples-to-apples benchmark。

当前结果默认排除 Our test 中的 SAFE references，仅评估 254 条非-SAFE 样本；WAD test 使用全部 1,007 条样本。

输出文件：

- `aggregate_metrics.csv`: 汇总指标
- `aggregate_metrics.json`: 汇总指标 JSON
- `bertscore_details.csv`: 逐样本 BERTScore
- `lave_scores.jsonl`: 逐样本 LAVE-style Qwen3-14B judge 输出
- `bertscore_aggregate.json`: BERTScore 汇总
- `lave_aggregate.json`: LAVE 汇总

## Aggregate Metrics

| Dataset | Model | n | BERTScore-F1 | LAVE Hazard | LAVE Guide | LAVE Format | LAVE Overall | Failed LAVE Parses |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| our_test_non_safe | WAD alter-only | 254 | 0.849 | 3.618 | 3.020 | 5.862 | 3.689 | 0 |
| our_test_non_safe | WAD full-JSON | 254 | 0.851 | 4.043 | 2.772 | 6.020 | 4.020 | 0 |
| wad_test | WAD alter-only | 1007 | 0.902 | 4.518 | 4.640 | 7.895 | 5.516 | 0 |
| wad_test | WAD full-JSON | 1007 | 0.900 | 5.288 | 5.485 | 8.049 | 6.148 | 0 |


## English LaTeX Table

```latex
\begin{table}[t]
\centering
\small
\caption{Compatibility-aware comparison with WalkVLM/WAD. Both models use the same Qwen2.5-VL backbone and are evaluated under an image-only inference protocol.}
\label{tab:walkvlm_comparison}
\begin{tabular}{lcccc}
\toprule
Training data & Our test BERTScore-F1 & Our test LLM judge & WAD test BERTScore-F1 & WAD test LLM judge \\
\midrule
WAD alter-only & 0.849 & 3.689 & 0.902 & 5.516 \\
WAD full-JSON & 0.851 & 4.020 & 0.900 & 6.148 \\
\bottomrule
\end{tabular}
\end{table}
```

## Paper-Ready Paragraph

We compared two image-only Qwen2.5-VL variants trained on the same 1,000 WAD samples: direct alter-only supervision and auxiliary full-JSON multitask supervision. Both models were evaluated on the 254 non-SAFE samples from our test set and all 1,007 WAD test samples under an image-only inference protocol. We did not provide WAD scene summaries at test time, because such summaries are not available in our target deployment setting and would constitute privileged information. Since WAD uses summary-conditioned single-sentence walking reminders whereas our method directly predicts structured ALERT/GUIDE outputs, we report semantic BERTScore and a LAVE-style Qwen judge score rather than n-gram overlap metrics. These results are therefore intended as a diagnostic cross-dataset comparison rather than a fully matched benchmark; they show how the two fine-tuning datasets transfer across different annotation schemas and task formulations.

## Caveat Statement

N-gram metrics such as BLEU/ROUGE are not used as primary evidence because schema-shifted safety guidance allows multiple valid phrasings and requires semantic/actionability evaluation. The comparison should be interpreted with the WAD summary-conditioned training objective and the ALERT/GUIDE schema mismatch in mind.
