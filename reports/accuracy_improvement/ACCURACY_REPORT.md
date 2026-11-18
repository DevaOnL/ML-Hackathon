# PathSnap accuracy-improvement addendum

Measured experiment frozen at f4da580f9654d90cc493bea8d4643a89b013ac09f985044a7be2a261af34d85f. The earlier accepted technical report is preserved, not overwritten.

## Paired internal TRAIN16 classifier confirmation

Both methods refitted on identical TRAIN64. Exactly 350 deterministic observations per reserved trip. Earlier releases historically used these trips; not a blind official test.

| Model | Control top-1 | Candidate top-1 | Change | Binary macro F1 control / new | Binary LL control / new |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 91.607% | 91.768% | +0.161 pp | 94.166% / 93.151% | 0.059542 / 0.064316 |
| knn_distance_weighted | 90.589% | 91.750% | +1.161 pp | 93.439% / 93.680% | 0.079269 / 0.069006 |
| bagging | 92.161% | 92.696% | +0.536 pp | 94.708% / 95.778% | 0.048900 / 0.040598 |
| gaussian_nb | 90.554% | 91.089% | +0.536 pp | 92.805% / 92.487% | 0.304175 / 0.130321 |
| hist_gradient_boosting | 92.054% | 93.018% | +0.964 pp | 94.737% / 95.807% | 0.049440 / 0.039301 |

## Dense internal TRAIN16 route confirmation

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| length_f1 | 95.735% | 95.825% | +0.089 pp |
| edge_f1 | 93.719% | 93.839% | +0.120 pp |
| proxy_edge_accuracy | 93.849% | 94.669% | +0.820 pp |
| proxy_edge_macro_f1 | 90.047% | 92.305% | +2.258 pp |
| route_continuity | 100.000% | 100.000% | +0.000 pp |

## Final student-development comparison

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| length_f1 | 97.941% | 78.118% | -19.823 pp |
| edge_f1 | 95.136% | 76.106% | -19.030 pp |
| proxy_edge_accuracy | 93.889% | 75.851% | -18.039 pp |
| proxy_edge_macro_f1 | 90.542% | 74.700% | -15.841 pp |
| route_continuity | 100.000% | 80.000% | -20.000 pp |

## Validation and release

Tests: {"created_at_utc": "2026-11-18T11:02:38.131Z", "hosting_tests": 29, "independent_MAX_reviews_completed": 5, "log": "reports/accuracy_improvement_verified_tests.log", "log_sha256": "85bd4d77ae2dbb400d3dfcb2ed61e07f7893565ec3bbc6b06e6b3b7a705561be", "node_tests": 22, "python_tests": 499, "review_limit": "Automated code/science review, not human/organizer approval.", "scope": "Original regression suites plus new feature/grouped-validation/strict artifact roles/decoder/confirmation/report safety tests. Subsequent source modifications require rerunning relevant tests before release.", "static_smoke": "pass", "status": "pass", "total_tests": 550}

Release decision: {"accepted": false, "all_five_top1_improved": true, "created_at_utc": "2026-11-18T11:13:07.421622+00:00", "deployed": false, "evaluator_used_for_tuning": false, "family_results": [{"family": "logistic_regression", "record_sha256": "768070f769c93ef6d8f593f00943c05d9edae82753553b091b4ccd01a269d7c3", "top1_delta": 0.001607142857142807, "top1_improved": true}, {"family": "knn_distance_weighted", "record_sha256": "263e15480d967a6a991d5f77a542aa6cb9cdaa39538c3c21aca1e4f565abdfac", "top1_delta": 0.011607142857142816, "top1_improved": true}, {"family": "bagging", "record_sha256": "90afa1c0e770fbcf34fd2601e2520f9b7a39421533595e90fe8128bb12f2c2c9", "top1_delta": 0.005357142857142838, "top1_improved": true}, {"family": "gaussian_nb", "record_sha256": "ef727fa6c6f3b16feb02460cd1c6adca8da1164114f397bd313346e6224313f7", "top1_delta": 0.005357142857142838, "top1_improved": true}, {"family": "hist_gradient_boosting", "record_sha256": "c8888b156360db9df3112c42600913a4671965dc0c7b81615be2b0c096385713", "top1_delta": 0.009642857142857175, "top1_improved": true}], "frozen_selection_sha256": "f4da580f9654d90cc493bea8d4643a89b013ac09f985044a7be2a261af34d85f", "inference_entrypoint_sha256": "ec169c097d13c8537eb76add3fe05d36a9dd42ac9feca7c81c1c2a9e2e10f57f", "internal_passed": true, "main_entrypoint_sha256": "2865c848c2b5a2841dcdaa272b4845d8f1fa45c941a45d68fd0e7ad5da2aea4b", "reason": "Preserve original deployment: final frozen candidate did not pass every nonregression gate", "release_manifest_sha256": "ff60a7724a72bb12980a6645852f154ad7bc088bd84d4705c753e5d3ad381c86", "role": "internalhistoricallyexposedTRAINnotblind", "route_summary_sha256": "564d3ec95dee147e6eb1c298d35d23f91082d304ef6515da8d1497f466518089", "source_sha256": "939d7cc817d710911ed4d0470059e194fafc605bb260aca0067f1b2265916cb9", "student_comparison_sha256": "b610e5d25d13ff947571f91b1e2e18a0c75d5e3ec54e639791457c3205d2186e", "student_development_used_for_tuning": false}

## Limits

- Proxy GPS edge labels are geometry-and-motion route projections, not independent timestamp ground truth; geometry-derived target favors geometric predictors.
- All80 TRAIN trips were historically used by earlier releases; prospective16 internal confirmation is not historicallyunseen or an official blind test.
- Student is historically used development data. Results are final comparisons after freeze and did not feed new tuning.
- Evaluator was already scored historically and was not accessed for this new tuning or selection.
- Tuning estimates reuse validation for selection; internal confirmation and paired trip-resampling intervals do not establish independent official-test generalization.
- Individual metric regressions, failures and negative configurations remain visible; no blanket all-accuracies claim.

## Reproduction

1. Use the pinned private authorized TRAIN cache and `experiments/accuracy_improvement/PROTOCOL.json`.
2. Run `python -B -m scripts.accuracy_trees --family FAMILY --stage screen`, then `--stage full`. Existing results are never overwritten; verified identical completed trials may resume.
3. Execute the six authorized OOF routes with `scripts.accuracy_route_validation`; attest original source/model evidence before selecting the decoder.
4. Freeze all families and decoder with `scripts.accuracy_confirmation --freeze`, then run eachfamily once and all16 dense routes via `scripts.accuracy_route_confirmation`.
5. Export the accepted candidate via `scripts.accuracy_export`, perform the final named student comparison and run the complete regression/release integrity gates.

[Protocol](<../../experiments/accuracy_improvement/PROTOCOL.json>) · [Frozen selection](<../../experiments/accuracy_improvement/confirmation/frozen_selection.json>) · [Full measured results](<accuracy_results.json>) · [Comparison CSV](<model_comparison.csv>)
