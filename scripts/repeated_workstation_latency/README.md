# Repeated Workstation Latency Benchmark

The experiment uses four RTX A6000 GPUs concurrently: two single-GPU replicas of 3B and two of 7B. Each replica loads its model once, performs ten warm-up inferences, and processes the same 200 benchmark records twice. This yields 800 timed observations per model.

Settings are BF16, batch size one, dynamic-resolution preprocessing, checkpoint generation settings, and `max_new_tokens=1024`, without weight quantization. End-to-end latency includes image reading, preprocessing, transfer, generation, and decoding; model loading and warm-up are excluded. CUDA synchronization bounds the generation interval. Results are grouped by normalized predicted output: exactly `<SAFE/>` versus other outputs.

## Included components

- `mr05_worker.py`: original single-GPU worker and inference timing implementation.
- `mr05_common.py`: timing, input, and worker synchronization helpers.
- `mr05_stats.py`: original statistics and full-run validation functions.
- `prompt.txt`: benchmark prompt.
- `recorded_environment.txt`: recorded runtime versions.
- `tests/`: selected original CPU tests for timing/statistical contracts, with import paths adjusted to this directory layout.
- `summarize_public.py`: recomputes pooled statistics from the published numeric timing records.

The worker expects a prepared run directory containing `config/run_config.json`, `sample_manifest.csv`, and `config/prompt.txt`, plus locally accessible images and checkpoints. It also expects GPU assignment and a controller release barrier. The server-specific controller, image manifest, checkpoints, and raw generated text are not bundled; the worker is provided as a reusable experimental component, not a standalone one-command benchmark.

## Recalculate published statistics without a GPU

From this directory:

```bash
python summarize_public.py --input ../../results/repeated_workstation_latency/timing_observations.csv --output pooled_recomputed.json
```

This uses Python's standard library. The output file must not already exist. The public CSV preserves the 1,600 numerical timing observations and branch labels, without image identifiers or generated text. The original full-run validator requires additional raw monitoring and worker files; it is not applicable to this reduced public result directory.

See the [results](../../results/repeated_workstation_latency/README.md).
