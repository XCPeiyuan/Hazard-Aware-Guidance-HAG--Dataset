# Repeated Workstation Latency Results

| Model | Predicted branch | Observations | Mean E2E (s) | P95 E2E (s) | Mean output tokens | Peak allocated memory (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3B | Safe | 296 | 0.81 | 0.83 | 4.00 | 7.53 |
| 3B | Hazard | 504 | 1.54 | 1.76 | 42.62 | 7.53 |
| 7B | Safe | 264 | 1.03 | 1.06 | 4.00 | 16.05 |
| 7B | Hazard | 536 | 2.02 | 2.32 | 41.37 | 16.05 |

Each model has two single-GPU replicas and two timed rounds per replica over the same 200 benchmark records. Four replicas run concurrently on four RTX A6000 GPUs. Counts are repeated observations, not independent images. Memory is the maximum allocated across each model's replicas and rounds, not a branch-specific peak.

All 1,600 timed inferences succeeded, ended with an end-of-sequence token, and stayed below the 1,024-token generation limit. The reported latency is non-streaming full-output latency under this workstation configuration.

- `pooled_by_model_branch.csv`: full-precision pooled statistics.
- `by_replica_round.csv`: per-replica/round statistics.
- `detailed_timing.csv`: timing-stage summaries.
- `timing_observations.csv`: numeric observations without image identifiers or output text.
- `peak_memory.json`: per-replica peak allocated memory.

See the [protocol and recalculation command](../../scripts/repeated_workstation_latency/README.md). Earlier single-run results remain in `../inference_efficiency_analysis/`; they are not the values reported in the revised latency table.
