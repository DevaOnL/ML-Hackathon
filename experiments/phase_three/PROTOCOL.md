# Phase-three candidate learning protocol (user-selected three models)

This protocol is written and frozen BEFORE any phase-three model is fit, any
family is selected or any student-test array is read. It covers exactly the
families the user selected on 18 November 2026 (docs/PHASE_SCOPE.json,
docs/LEAD_COORDINATION.md, task-16): **Logistic Regression**, a genuine
**distance-weighted KNN**, and **Bagging** (bootstrap aggregation of decision
trees). The organizer five-model minimum is deferred by explicit user
instruction and is NOT satisfied or claimed by this phase. The preserved
five-model CV/HPO evidence and the historical HistGB freeze remain untouched
immutable prior work; nothing here edits experiments/ root files,
artifacts/model.joblib, match_config.json, decoder outputs, or
mapmatch/features.py, mapmatch/learning.py, mapmatch/experiments.py.

## Reproduction

From the repository root, with the numerical stack already installed in
`.venv` and thread pools capped at one:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python experiments/phase_three/run_phase_three.py --stage cv
.venv/bin/python experiments/phase_three/run_phase_three.py --stage select
.venv/bin/python experiments/phase_three/run_phase_three.py --stage hpo
.venv/bin/python experiments/phase_three/run_phase_three.py --stage freeze
.venv/bin/python experiments/phase_three/run_phase_three.py --stage student
```

Each stage writes only under `experiments/phase_three/` and
`artifacts/phase_three/` plus the new module `mapmatch/phase_models.py`. All
stages are idempotent on their own outputs and refuse to run out of order:
`select` requires the three CV records, `hpo` requires the recorded selection,
`freeze` requires HPO, and `student` requires BOTH the recorded selection AND
the model freeze, so no student-test array can exist before the family choice
is recorded on disk.

## Data, folds, features (reused, never rebuilt)

All numbers reuse the already-frozen shared caches; no dataset is regenerated:

- TRAIN: `experiments/cache/train_edee9a8f151dd25e.npz` (+ JSON),
  sha256 `db523c112a6119f16b6f473d6221e3c56b6730ccb2db64dee55433aedbf3ba9e`;
  80 trips, 28,000 sampled observations (deterministic 350/trip),
  269,463 candidate rows, 38 reference-free features.
- Student development set (loaded ONLY after selection+freeze):
  `experiments/cache/student_test_240a8fbfc79205b3.npz` (+ JSON),
  sha256 `ae96abb4784501b2b086c9b4aa2f5668e1c9d7f29abda907b7f402118af81b73`;
  10 trips, 19,160 observations, 175,778 candidate rows.

The three trip-grouped folds come from `mapmatch.experiments.build_group_folds`
on the cached TRAIN dataset (GroupKFold(3) over trip IDs) and are asserted
equal, trip set by trip set, to `experiments/cv_folds.json` before any fit.
Candidate misses (578/28,000 TRAIN) and ambiguous proxies (2,088/28,000) are
retained in every denominator. Labels remain the disclosed route-projection
proxy, never independent per-GPS ground truth. No reference-derived feature
exists; inference features are identical to the deployed schema
(`pathsnap-candidate-v1`, 38 features).

## Families and predeclared defaults

1. `logistic_regression`: StandardScaler + LogisticRegression,
   `{C: 1.0, max_iter: 500, solver: lbfgs}`, seed 20261118 — IDENTICAL to the
   valid saved baseline. Its grouped CV, all80 fit, student evaluation and
   prediction arrays are reused read-only from
   `experiments/cv/logistic_regression.json`,
   `experiments/evaluation/logistic_regression.json`,
   `experiments/predictions/logistic_regression_{oof,student}.npz` and
   `artifacts/baselines/logistic_regression.joblib` after verifying fold
   equality, parameter equality and cache fingerprint; nothing is refit.
2. `knn_distance_weighted`: StandardScaler + KNeighborsClassifier with
   `weights="distance"` (genuine inverse-distance vote weighting), Euclidean
   metric, exact brute-force neighbours, predeclared `n_neighbors=25`.
3. `bagging`: BaggingClassifier over DecisionTreeClassifier base learners —
   genuine bootstrap aggregation (bootstrap=True, row subsampling), NOT a
   relabelled RandomForest/ExtraTrees. Predeclared modest start:
   `{n_estimators: 40, max_samples: 0.7, max_depth: 14, min_samples_leaf: 3}`.

LogisticRegression and KNN use fold-local learned StandardScaler inside each
fit (as the historical scaled families did); Bagging uses no learned
preprocessing, matching the historical tree-family policy.

## KNN reference-bank policy (predeclared, TRAIN-fold-only)

sklearn `KNeighborsClassifier.fit` accepts no `sample_weight`; this limitation
is disclosed rather than hidden. To keep genuine exact KNN tractable and
weighted, each KNN fit (each CV fold and the final all80 fit) first builds a
deterministic reference bank from ITS OWN TRAIN ROWS ONLY:

- Budget fixed at **60,000 rows** (of ~180k fold rows / 269,463 all80 rows),
  never changed by measured performance.
- Class-stratified proportional allocation preserving the positive rate.
- Within each class, weighted sampling without replacement with inclusion
  probability proportional to the saved training weights (inverse candidate
  count per observation, inverse sampled observation count per trip), so the
  bank approximates the observation-equal/trip-equal fitting mass.
- Deterministic seed per fit: `seed + 7919 * fit_index` (fold 0,1,2; final
  all80 fit uses index 3).

Validation rows, other folds' rows and all student rows are never bank
members. Bank composition is recorded with every fit.

## Selection and Bayesian optimization (TRAIN only)

Stage order is enforced on disk: (1) grouped CV of the two new families plus
the reused LR record; (2) family selection recorded in
`experiments/phase_three/selection.json`; (3) HPO on the selected family only;
(4) winner fit + freeze; (5) student diagnostics. The student cache is not
even loaded by stages 1-4.

- Objective (unchanged from the shared protocol): equal-trip mean out-of-fold
  observation top-1 proxy accuracy over all 28,000 TRAIN observations, misses
  included; tie-break by binary macro-F1 then lower log loss.
- HPO budget declared exactly: **8 trials = 1 measured prior (the selected
  family's grouped-CV baseline) + 2 seeded initial-design points + 5 genuine
  Gaussian-process expected-improvement proposals** (Matern-5/2 + white-noise
  surrogate on normalized hyperparameters, analytic EI, incumbent-relative,
  candidate pool 2048 uniform + 1024 incumbent-neighbourhood draws). This is a
  NEW phase-three search; it never reuses or relabels the historical 12-trial
  HistGB history.
- Search spaces (predeclared): LR `C\ log [1e-3, 100]`; KNN
  `n_neighbors\ logint [5, 75]` (weights/metric stay fixed as declared);
  Bagging `n_estimators\ int [30, 80]`, `max_samples\ float [0.5, 0.9]`,
  `max_depth\ int [8, 20]`, `min_samples_leaf\ logint [2, 10]`.

Training observation/trip weighting is applied wherever the estimator supports
`sample_weight` (LogisticRegression, BaggingClassifier); the KNN bank policy
above is the disclosed substitute where it is unsupported.

## Winner freeze

The winning configuration is refit on the EXACT cached all80 TRAIN matrix
(269,463 rows) with saved hyperparameters, timed, and exported atomically
(temp file + rename) to `artifacts/phase_three/model.joblib` as the shared
bundle format `{"estimator", "feature_names", "feature_version", "metadata"}`
containing only standard sklearn classes, loadable by the existing
`mapmatch.learning.load_emission_model` without any new custom-pickle class.
Metadata (`model_metadata.json`), feature schema and an
`artifact_manifest.json` with sha256 mirror the deployed artifact's schema.
A clean-environment smoke load (`.venv-clean`) verifies the bundle, feature
schema, binary classes and finite probabilities on a deterministic fixture
(first 2,000 TRAIN cache rows), then `model_freeze.json` records the frozen
sha256 and a PHASE_MODEL_FROZEN signal is sent to the Lead. After the freeze
the artifact bytes are never rewritten; later supplements are metadata/JSON
only, and the winner's student predictions are produced by loading the frozen
bytes.

If the winning HPO trial is exactly the measured prior, its existing OOF
arrays are reused without refitting; otherwise the winner is confirmatorily
replayed once on the identical three folds (recorded as replay, not as an
extra HPO trial) and must reproduce the winning trial's objective, macro-F1
and log loss within 1e-6, else the stage fails.

## Student diagnostics and final fits (after freeze)

For all three families on all 10 student trips / 19,160 observations:
binary candidate metrics at the fixed 0.5 threshold (accuracy, precision,
recall, macro-F1, log loss, confusion), graph-namespaced directed-edge
observation macro-F1 (`trip_id:canonical_arc_id`), per-observation normalized
categorical NLL with the 1e-7 floor on absent target mass (misses retained in
the unconditional denominator), top-1 including misses, per-trip and
equal-trip summaries, sparse graph-qualified confusion counts,
ambiguity-stratified (ambiguous vs non-ambiguous proxy) top-1/NLL, and
raw-classifier continuity (same canonical edge or head(previous)==tail(current),
no routed connectors; empty assignments discontinuous). Fit, OOF and student
inference timings, parameter descriptions, source versions, train-vs-OOF and
student gaps are recorded per family. LR numbers are the reused saved
evaluation; KNN/Bagging receive fresh all80 fits with their predeclared
defaults (the non-selected families) or the frozen winner artifact (the
selected family). Non-winner fitted models are retained under
`artifacts/phase_three/baselines/`.

Results land in `experiments/phase_three/results.json` and
`experiments/phase_three/model_comparison.csv` using the same schema the
visuals/report pipeline already consumes (`cv`, `evaluations`,
`comparison`, `bayesian`, `coverage_summary` blocks and the 30-column
comparison row). Raw probability caches stay untracked by Git.

## Honest-reporting constraints

No metric, timing, curve or screenshot is invented; every number in the phase
outputs is measured by the code in this directory. The student set is
development data only. Evaluator data was never supplied and is never
accessed. The prior geometry-only baseline and historical five-model/HistGB
results are never relabelled as phase-three evidence.

## Post-measurement source hardening

The measured CV, eight HPO trials, frozen all-80 fit and all-ten student
evaluations completed before the Lead hardened the runner on 18 November 2026.
Those edits add refuse-overwrite/refit checks, saved-array identity assertions,
actual child-runtime versions and a strict probability comparison. No model,
configuration, cached prediction or measurement was changed or rerun. The
original freeze manifest and its clean-load record retain their original bytes.
A separate strict 2,000-row TRAIN fixture check is recorded in
`reports/phase_three/hardened_clean_environment_acceptance.json`; the maximum
selected-Bagging probability difference is zero. The independent implementation
review is `analysis/review/phase_three/model_integration_review.json`.

KNN saved-probability acceptance uses the original training runtime (`.venv`).
A clean-wheel build with identical version strings differs by 6.38e-7 on one
257-row TRAIN diagnostic fixture; the binary decision does not change. This
numerical limitation is recorded in
`reports/phase_three/knn_runtime_diagnostic.json`; it is not hidden by relaxing
the 1e-12 saved-prediction acceptance tolerance.
