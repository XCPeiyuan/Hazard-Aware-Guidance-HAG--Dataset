# Earlier WAD Text-comparison Results

For the revised table including structured metrics, see [the additional comparison](../wad_structured_comparison/README.md).

## Experimental Design

To provide a controlled comparison, both the WAD primary baseline and auxiliary WAD variant use the same Qwen2.5-VL backbone, the same 1,000 selected WAD samples, and image-only input. No oracle scene summary is provided at inference time.

- **WAD alter-only:** image plus fixed prompt to the released WAD `alter` target. This is the primary WAD baseline.
- **WAD full-JSON:** image plus fixed prompt to all six released annotation fields; only generated `alter` is evaluated. This is an auxiliary multitask ablation, not the primary baseline and not a reproduction of the original WalkVLM CoT pipeline.
- **R+U:** existing main model results, using structured ALERT/GUIDE output.

The comparison uses 254 non-SAFE references from the Real-world test set. WAD test uses all 1,007 samples. The judge accepts either structured ALERT/GUIDE or a natural-language reminder and evaluates semantic hazard coverage, actionable guidance, format, and overall usefulness.

## Primary Text-metric Results

| Training data / model | Our non-SAFE BERTScore-F1 | Our non-SAFE Judge Overall | WAD test BERTScore-F1 | WAD test Judge Overall |
|---|---|---:|---:|---:|---:|
| R+U (ours) | **0.9334** | **8.2451** | 0.8578 | 6.0119 |
| **WAD full-JSON** (primary) | 0.8506 | 4.0197 | **0.8998** | **6.1480** |
| WAD alter-only (aux.) | 0.8489 | 3.6890 | 0.9021 | 5.5164 |

Interpretation:

- R+U strongly outperforms both WAD variants on the proposed non-SAFE test set (BERTScore-F1 0.9334 vs 0.8506/0.8489; Judge Overall 8.2451 vs 4.0197/3.6890), showing substantially better transfer to the target structured safety-guidance task.
- WAD full-JSON achieves the highest judge score on its own WAD test set (6.1480), consistent with benefiting from the full annotation schema in its native task. R+U achieves a comparable but slightly lower judge score (6.0119) on WAD test, indicating some cross-task guidance transfer.
- WAD alter-only achieves the highest BERTScore on WAD test (0.9021), consistent with single-field fine-tuning maximizing surface-form similarity to the reminder-style references.
- These results support a domain/schema specialization interpretation rather than an unrestricted claim that one dataset is universally superior.

## Auxiliary Full-JSON Ablation

| Test set | WAD alter-only BERTScore-F1 | WAD full-JSON BERTScore-F1 | WAD alter-only Judge | WAD full-JSON Judge |
|---|---:|---:|---:|---:|
| Our non-SAFE, n=254 | 0.8489 | 0.8506 | 3.6890 | 4.0197 |
| WAD test, n=1,007 | **0.9021** | 0.8998 | 5.5164 | **6.1480** |

Paired bootstrap results for full-JSON minus alter-only:

- Our BERTScore-F1: +0.0017, 95% CI [-0.0000, +0.0034].
- Our judge overall: +0.3307, 95% CI [-0.0354, +0.7087].
- WAD BERTScore-F1: -0.0023, 95% CI [-0.0040, -0.0007].
- WAD judge overall: +0.6316, 95% CI [+0.4528, +0.8064].

Full-JSON provides a stable judge-score improvement only on WAD test, where it achieves the best judge score overall (6.15 vs 5.52 alter-only, bootstrap Δ=+0.63 [0.45, 0.81]). On Our test, the improvement is not stable (Δ=+0.33, CI [-0.04, +0.71] crosses zero). Based on this evidence, full-JSON has been elevated to the primary WAD baseline (2026-06-10 update), with alter-only retained as an auxiliary ablation to show the effect of single-field vs full-schema supervision.

## Failure Handling and Validity Checks

- All 2,522 condition-samples produced parseable Qwen3-14B judge JSON.
- All prediction image keys match the corresponding references.
- Full-JSON generation produced 2 truncated outputs on Our test and 4 on WAD test.
- These six extraction failures were retained as `[ALTER_EXTRACTION_FAILED]` and received judge score 0; they were not deleted.
- The WAD alter-only model showed output mode collapse despite 920 unique `alter` targets among the 1,000 training examples. This is a model behavior, not simply duplicated supervision.
- BERTScore uses English `roberta-large`; both references and evaluated outputs are English.

## Evaluation Scope

- WalkVLM/WAD and R+U use different annotation schemas and output objectives.
- WAD originally contains scene summaries and other annotations; the final primary baseline deliberately excludes summaries from model input to prevent privileged-information leakage and match image-only deployment.
- The comparison uses a common Qwen2.5-VL backbone fine-tuned on WAD, not the complete original WalkVLM pipeline.
- The public simplified pipeline does not reproduce all components described in the original paper, including its complete hierarchical/CoT planning system.
- The Qwen3-14B judge is a custom LAVE-style judge, not the official LAVE implementation. It should be described transparently.
- BLEU/ROUGE are not used as primary evidence because multiple safety-valid phrasings and schema differences make n-gram overlap misleading.
