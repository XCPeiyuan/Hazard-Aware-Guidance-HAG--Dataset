# Offline Diagnostic Analysis on the Mixed-source R+U Test Set Distance-Aware Object Recall Experiment

## Quick Start

```powershell
$python = 'python'

# Unchanged strict default
& $python '实验数据\Offline Diagnostic Analysis on the Mixed-source R+U Test Set\distance_analysis\run_analysis.py' --dry-run
& $python '实验数据\Offline Diagnostic Analysis on the Mixed-source R+U Test Set\distance_analysis\run_analysis.py'

# Isolated controlled-loose rerun
& $python '实验数据\Offline Diagnostic Analysis on the Mixed-source R+U Test Set\distance_analysis\run_analysis.py' `
  --dry-run --match-mode loose
& $python '实验数据\Offline Diagnostic Analysis on the Mixed-source R+U Test Set\distance_analysis\run_analysis.py' `
  --match-mode loose

# Compare strict and controlled-loose outputs
& $python '实验数据\Offline Diagnostic Analysis on the Mixed-source R+U Test Set\distance_analysis\compare_match_profiles.py'

# Tests
& $python -m pytest '实验数据\Offline Diagnostic Analysis on the Mixed-source R+U Test Set\distance_analysis\tests' `
  -q -p no:cacheprovider `
  --basetemp '实验数据\Offline Diagnostic Analysis on the Mixed-source R+U Test Set\distance_analysis\.pytest_basetemp_manual'
```

## Profiles and Outputs

- `strict` remains the default and writes to `distance_analysis/`.
- `loose` writes to sibling `distance_analysis_loose/` by default.
- The two profiles use independent DeepSeek caches.
- `--output-dir PATH` may override the output directory explicitly.
- Category, distance, SSI, and alert are never used in matching prompts.

## Main Flags

| Flag | Description |
|---|---|
| `--dry-run` | Validate inputs and report expected API workload |
| `--force-refresh-cache` | Delete the selected output directory's cache before running |
| `--max-records N` | Limit to first N GT hazard records |
| `--methods ours fewshot` | Methods to evaluate |
| `--config PATH` | LLM config JSON |
| `--overrides PATH` | Manual match overrides JSONL |
| `--match-mode strict|loose` | DeepSeek matching profile |
| `--output-dir PATH` | Explicit isolated output directory |

## Evidence Scope

- GT source: reviewed test-set JSON.
- Main unit: GT hazard object.
- Matching: rule-first deterministic name matching followed by cached DeepSeek
  name-semantic matching.
- Matching is one-to-one and names-only.
- Distance bins: Near `<=5`, Medium `6-8`, Far `>=9` steps.
- This is offline hazard-object mention recall, not behavioral safety validation.

## Comparison Outputs

`compare_match_profiles.py` writes these files under `distance_analysis_loose/`:

- `strict_loose_comparison.md`
- `strict_loose_comparison.json`
- `loose_matching_changes.jsonl`

The change queue includes newly detected objects, detections lost under loose
matching, and objects paired with a different prediction.
