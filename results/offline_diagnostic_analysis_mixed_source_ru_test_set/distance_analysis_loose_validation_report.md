# Offline Diagnostic Analysis on the Mixed-source R+U Test Set Distance-Aware Object Recall - Validation Report

**Final-ready: YES**

## Experiment Scope

- GT source: reviewed test-set JSON (SSI -> Qwen3-14B alignment -> human review)
- Methods: ours (qwen25vl7b), fewshot (qwen25vl7bfs)
- Matching: deterministic name rules + DeepSeek V4 Flash for unresolved names
- Category and distance are never used in match prompts
- Distance bins: Near <= 5, Medium 6-8, Far >= 9. These are empirical post-hoc bins for this analysis.
- Claim boundary: mention recall across GT categories and distances; not predicted-distance accuracy, collision reduction, or closed-loop safety.

## Input Validation

- Records per source: 863
- GT hazard records: 526
- GT hazard objects: 1092

## Matching Statistics

- Rule matches: 484
- DeepSeek matches: 510
- Unresolved: 1190
- Manual overrides: 0

## Verification

- All tests pass: YES
  - Details: {"command": "C:\\Users\\99681\\miniconda3\\envs\\py311-env\\python.exe -m pytest -q -p no:cacheprovider tests --basetemp C:\\Paper_LaTeX\\New_MTAP_Special_Issue__AI_Driven_Immersive_Multimedia (1)\\实验数据\\Offline Diagnostic Analysis on the Mixed-source R+U Test Set\\distance_analysis\\.pytest_basetemp_validation_3489ab8a944344cca476bb81b100e056", "exit_code": 0, "output_tail": "........................................................................ [ 26%]\n........................................................................ [ 52%]\n........................................................................ [ 79%]\n........................................................                 [100%]\n272 passed in 0.68s"}
- Input hashes unchanged during run: YES
  - Details: {"fewshot_reviewed": {"actual": "b6aeaee0500279baca101461f44b193b6171ae6c2e8cc9420dd1219a5a14f9ee", "expected": "b6aeaee0500279baca101461f44b193b6171ae6c2e8cc9420dd1219a5a14f9ee", "status": "UNCHANGED"}, "gt_reviewed": {"actual": "000e0d5465c412d111b96817e3c087ef5772e2bb945db275bed2686a28a97786", "expected": "000e0d5465c412d111b96817e3c087ef5772e2bb945db275bed2686a28a97786", "status": "UNCHANGED"}, "manual_match_overrides": {"actual": "e7f6017884c092c8c86837b7d5c0b480e2e2737652c05d1c66f4b39a3ca5f55a", "expected": "e7f6017884c092c8c86837b7d5c0b480e2e2737652c05d1c66f4b39a3ca5f55a", "status": "UNCHANGED"}, "ours_reviewed": {"actual": "af4542d6f73c20564eb653ce3952aa451dc1963c5456a67b581d02d44d72cc5e", "expected": "af4542d6f73c20564eb653ce3952aa451dc1963c5456a67b581d02d44d72cc5e", "status": "UNCHANGED"}}
- All matched pairs one-to-one: YES
  - Details: {"duplicate_gt_pairs": [], "duplicate_pred_pairs": [], "matched_pairs": 994}
- No category/distance in match prompts: YES
  - Details: {"forbidden_tokens": []}
- Metrics recomputable from object results: YES
  - Details: {"failures": []}
- Zero-total cells use null: YES
  - Details: {"failures": []}
- No unresolved DeepSeek API failures: YES
  - Details: {"failure_count": 0}
