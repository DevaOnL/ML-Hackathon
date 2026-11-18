# KNN TRAIN-only accuracy tuning scope

- Owner: delegated KNN tuner (parent session-1741df65-9ff5-4d55-a652-64751ae7764d).
- Claim made before implementation: `scripts/accuracy_knn.py`, `tests/test_accuracy_knn.py`, `experiments/accuracy_improvement/knn/**`, `experiments/accuracy_improvement/private/knn/**` only.
- Do not modify released files, model artifacts, live behavior, shared common code, or other families.
- Data authorized for fitting/evaluation: only `experiments/cache/train_edee9a8f151dd25e.npz`; no student/evaluator inputs, references, or results.
- Evaluation: shared three GroupKFold splits, inverse-observation weights, fixed 60,000 stratified weighted bank, seeds 20261118 + 7919 * fold; top-1 primary, macro-F1 and lower log loss secondary.
- Planned implementation: genuine sklearn KNeighborsClassifier, deterministic compact/physics/context feature profiles from lead contract, one exact k-max query reused for inverse-distance votes; exact sklearn zero-distance behavior.
- Resource limit: one or two jobs, max two computation threads; at most three predeclared configurations; phase-0 single-fold screening then full three-fold top two only.
- Initial status above was recorded before implementation.

## Lead handoff superseding execution plan

- Lead coordination now specifies 64 tuning trips and 16 prospective confirmation trips (confirmation is historically seen, not an official blind test). Three GroupKFold splits apply only to the tuning64 subset.
- Approved menu has six scored configurations including a paired v1 baseline: v1/p2/brute/k25; compact/p2/KD/k5,15,35; physics/p2/KD/k15; compact/p1/KD/k15. All banks remain 60,000. The previous nine-score proposal is withdrawn before any cache fit.
- Only the lead-owned central runner launches measured fits and writes measured metrics: fold0 screen, then fresh full3 runs for top2 plus paired baseline. KNN worker performs no duplicate cache fit or scoring.
- Constructor/helper implementation complete. Fourteen synthetic tests pass in 0.105 seconds; actual TRAIN timings, trial metrics, and winner are not yet measured by this worker.
- Exact max-k helper is available for exploratory use. Internal cutoff ties trigger re-queries only on affected rows and those extra query calls are counted; fixed-k probability behavior matches sklearn, including zero distances. Lead chose standard sklearn fixed-k prediction for final promoted comparisons.
- Handoff state: source frozen for lead integration; no TRAIN cache loaded, no model artifacts written by this worker. No held-out inputs/references/results accessed.
