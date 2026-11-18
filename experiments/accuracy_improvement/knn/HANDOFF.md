# KNN source handoff

## Implemented

- [accuracy_knn.py](../../../scripts/accuracy_knn.py) constructs a pure sklearn StandardScaler/KNeighborsClassifier pipeline. `make_knn_pipeline(**parameters, n_jobs=1)` is compatible with the central runner; `make_knn_estimator(parameters, seed, n_jobs)` supplies the direct shared-factory signature. No custom estimator needs serialization.
- Exact Euclidean inverse-distance votes use `p=2`, not squared distances or inverse-squared weights. The predeclared optional `p=1` trial uses inverse Manhattan distance and is a separate metric comparison. Only exact KD/Ball/brute backends are accepted. Jobs above two are rejected.
- `predict_k_grid` queries kmax once per batch for all k values, with exact sklearn zero-distance voting. It counts internal cutoff ties and, by default, re-queries only tied query rows at smaller k, preserving standalone fixed-k backend tie behavior. Extra query calls and transform/query/vote timings are reported. The maximum-k outer boundary is not probed and is explicitly marked unobserved.
- [test_accuracy_knn.py](../../../tests/test_accuracy_knn.py) has 14 passing synthetic tests: independent sklearn parity across exact backends/p1,p2; zero distances and duplicate conflicting labels; nonzero distance-cutoff ties; query-call reuse; bank-only scaling; empty queries; single-class helper behavior; argument/resource validation; shared-factory contract.

## Verification

`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B -m unittest tests.test_accuracy_knn -v`

Result: 14 tests passed, 0.105 seconds. Scientific runtime: Python3.14.6, numpy2.5.5, sklearn1.8.0. No background jobs were started.

Source SHA256: `48504fb802830c97972b3dcbf6fc22481d9457d3e61d9862bd2de52ce8d39782`. Tests SHA256: `c72cf16c5ef13d79802acb42ca5c3adc92d692815fbf8574b2c835423759f747`.

## Actual measurements deliberately deferred

Per [shared lead coordination](../COORDINATION.md), workers must not launch duplicate fits. This worker loaded no TRAIN cache, started zero real TRAIN fits, wrote no model artifacts, and accessed no student/evaluator inputs, references or results. Thus TRAIN trial timings, all-trial metrics, accuracy gain and winner are **not yet measured here**; they must be supplied by the lead central runs.

The [frozen menu](phase0_menu.json) now matches the lead-approved six scored configurations: v1/p2/brute25 control, compact/p2/KD5,15,35, physics/p2/KD15, compact/p1/KD15. The earlier nine-score proposal is withdrawn before any fit. Primary bank60k is fixed; no120k experiment is planned. Shared64tuning/16confirmation partition and shared3GroupKFold apply. Lead runs fold0 screening then fresh full3 top2 plus pairedv1 control; standard fixed-k sklearn prediction is the chosen final path.

## Integration note and honest tradeoffs

The common source-pin logic currently hashes the central factory wrapper module. When that wrapper imports this worker module, the underlying implementation must also be pinned, or the common factory can directly be [make_knn_estimator](../../../scripts/accuracy_knn.py#L83-L91), whose module hash correctly identifies the KNN implementation. The same issue applies to imported linear/Bayes constructors. This was sent to the lead.

Compact or physics features can lose useful information; no improvement is guaranteed. A60k stratified weight-proportional bank approximates weighting because sklearn KNN.fit does not support sample_weight. Exact tree search may be fast in low dimensions but geometry can limit speedups. The shared-vote helper reads sklearn encoded training labels internally; the final estimator remains pure standard sklearn and final full3 prediction need not use this helper. Cutoff-tie correction adds work only for tied rows and is counted.
