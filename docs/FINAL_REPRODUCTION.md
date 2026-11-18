# Reproducing the final five-family experiment

The public checkout contains source, fitted bundles and aggregate measurements. Authorized raw trips, row-level feature caches and prediction archives remain private. A clone needs the supplied data before it can reproduce the numerical experiment. These commands require no evaluator dataset, network service or deployment.

The maintained [final runner](<../experiments/final_five/run_final_five.py>) compares Logistic Regression, distance-weighted KNN, Bagging, GaussianNB and HistGradientBoosting. It reuses the first three families' verified baseline evidence, freshly measures GaussianNB and default/tuned HistGB, and validates the completed historical HistGB Bayesian study. An isolated reproduction records new costs without changing the published measurements.

## Runtime and supplied data

Run commands from the repository root. Prefer the original local virtual environment when available. The tested numerical runtime is Python 3.14.6, NumPy 2.5.5, SciPy 1.18.0, scikit-learn 1.8.0 and joblib 1.5.3, recorded in the [completed training record](<../experiments/final_five/current/train_complete.json>). [requirements.txt](<../requirements.txt>) pins the project packages.

For a clean machine, these are installation instructions, not commands executed by the input checker:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# The final clean stage uses this exact interpreter path:
python3.14 -m venv .venv-clean
.venv-clean/bin/python -m pip install -r requirements.txt
```

Confirm Python 3.14.6 and the package versions above in both environments. Matching versions alone do not establish identical BLAS builds or CPU reduction order. Restoration fails on metric drift beyond absolute tolerance `1e-12`, with relative tolerance zero. It never widens that limit to accommodate another build.

Place the organizer-supplied `Student folder` directory directly under the repository root, retaining its complete `train` (80 trips) and `student_test` (10 trips) subdirectories and each supplied track, graph and route file. Preserve bytes and layout: the builder fingerprints them. Route-derived targets are observation proxies; feature construction remains graph/GPS-only and unchanged. TRAIN selects 350 deterministic observations per trip, giving 28,000 observations and 269,463 candidate rows. Student uses all 19,160 observations and 175,778 rows.

Keep the published prerequisites intact:

- [Whole-trip folds](<../experiments/cv_folds.json>), [shared LR CV](<../experiments/cv/logistic_regression.json>), [shared LR evaluation](<../experiments/evaluation/logistic_regression.json>) and [shared historical summary](<../experiments/results.json>).
- Original phase CV records for [LR](<../experiments/phase_three/cv/logistic_regression.json>), [KNN](<../experiments/phase_three/cv/knn_distance_weighted.json>) and [Bagging](<../experiments/phase_three/cv/bagging.json>), and evaluation records for [LR](<../experiments/phase_three/evaluation/logistic_regression.json>), [KNN](<../experiments/phase_three/evaluation/knn_distance_weighted.json>) and [Bagging](<../experiments/phase_three/evaluation/bagging.json>).
- Published all-80 baseline bundles for [LR](<../artifacts/baselines/logistic_regression.joblib>), [KNN](<../artifacts/phase_three/baselines/knn_distance_weighted.joblib>) and [Bagging](<../artifacts/phase_three/baselines/bagging.joblib>). Student preparation reads these exact estimators, verifies hashes, schema, binary classes, actual parameters and TRAIN trip identities, and fits no replacement model.
- The phase [selection](<../experiments/phase_three/selection.json>), [freeze](<../artifacts/phase_three/model_freeze.json>) and [model](<../artifacts/phase_three/model.joblib>) needed by the raw/LR student helper. The current final [model freeze](<../artifacts/final_five/model_freeze.json>), [model](<../artifacts/final_five/model.joblib>), [metadata](<../artifacts/final_five/model_metadata.json>), [feature schema](<../artifacts/final_five/feature_schema.json>), [decoder freeze](<../reports/final_five/decoder/freeze.json>) and [decoder configuration](<../artifacts/final_five/match_config.json>) are required before final student-input restoration.
- Current final reuse/provenance records summarized in [the final comparison](<../experiments/final_five/current/results.json>) and the historical Bayesian evidence listed below.

The [ignore rules](<../.gitignore>) cover supplied raw data, both caches, canonical prediction archives and the whole private replay-input directory. Do not force-add them. Replay diagnostics can contain proxy targets and identity arrays, including a candidate NPZ retained after a failed publication attempt.

## 1. Inspect the inputs

```bash
.venv/bin/python -B scripts/prepare_final_five_inputs.py
```

The [input preparer](<../scripts/prepare_final_five_inputs.py>) defaults to `--stage check`. It inventories both caches, the folds and all six reused archives: three OOF and three student archives. It performs zero fits, prediction calls, dataset builds or filesystem writes. Student cache/reference arrays are read only after the current final model and decoder freeze checks pass. It has no evaluator access.

Exit codes are 0 for verified inputs, 2 for missing inputs and 1 for verification failure. A clean clone normally reports missing private inputs. Continue with the relevant stages. If the inputs exist and pass, skip rebuilding them. Explicit train/student preparation also validates existing archives without another fit or prediction.

The optional `--report reports/final_five/reproduction_inputs_check.json` creates only that fixed public summary when explicitly requested and refuses an existing report. The published [check summary](<../reports/final_five/reproduction_inputs_check.json>) contains the local verification result, counts and hashes without row-level values. Use the default stdout check for a fresh assessment without replacing published evidence.

## 2. Rebuild the raw caches and LR inputs

Use the existing [raw/LR preparer](<../scripts/prepare_phase_three_inputs.py>), documented in [phase reproduction](<PHASE_REPRODUCTION.md>). In a clean clone where the respective private outputs are absent:

```bash
.venv/bin/python -B scripts/prepare_phase_three_inputs.py --stage train --data-root "Student folder"
# Published model/config freezes must already be present before this command:
.venv/bin/python -B scripts/prepare_phase_three_inputs.py --stage student --data-root "Student folder"
```

TRAIN builds the pinned TRAIN cache and restores LR's grouped OOF probabilities. Student validates the published phase freeze before building the pinned student cache, then predicts with the committed LR baseline. Do not use `--allow-lr-fit` for a reconstruction that claims the published baseline. Both helper stages refuse an existing cache/prediction output. Run them for the missing clean-clone inputs; do not delete working caches to force a rerun.

The helper pins raw fingerprints, cache-container hashes and per-array digests. The original shared run used NumPy 2.5.2; the final runtime uses 2.5.5. A cache-container mismatch still stops execution even if the printed array digests agree. Keep diagnostics and use an original compatible environment; do not edit pins. The new checker verifies every LR aggregate/fold metric at the final runner's stricter `1e-12` tolerance after the older helper's `1e-9` check.

One helper-only transcription correction followed measurement. The student `candidate_distances` digest in the [raw/LR preparer](<../scripts/prepare_phase_three_inputs.py>) is `cd7fdb3a8923e7716894a5599c318aeeb85e0d96e767a8b51b207fe24dba519f`. The earlier literal had `769` in place of `767`. The lead independently verified the array and the unchanged accepted student-container SHA-256 `ae96abb4784501b2b086c9b4aa2f5668e1c9d7f29abda907b7f402118af81b73` before correcting it. No dataset, measured result, historical source hash or tolerance changed.

## 3. Restore missing KNN/Bagging archives

```bash
.venv/bin/python -B scripts/prepare_final_five_inputs.py --stage train --run-id inputs-train-1
.venv/bin/python -B scripts/prepare_final_five_inputs.py --stage student --run-id inputs-student-1
.venv/bin/python -B scripts/prepare_final_five_inputs.py --stage check
```

TRAIN creates only missing KNN/Bagging OOF archives; student creates only missing KNN/Bagging student archives at the original excluded paths expected by the final runner. LR remains the older preparer's responsibility. The new tool accepts no arbitrary destination/root argument, rejects symlinks and traversal run IDs, and uses exclusive creation with non-replacing hard-link publication.

TRAIN uses the original three folds and parameters through [phase_fit and phase_predict](<../experiments/phase_three/run_phase_three.py>). KNN calls the [original reference-bank sampler](<../mapmatch/phase_models.py>): at most 60,000 rows from each fold's TRAIN side, with seed `20261118 + 7919 * fold_index` for fold indices 0, 1 and 2. Its StandardScaler fits inside that selected bank, as the historical implementation did; it does not fit on the whole fold before bank selection. Bagging uses 40 bootstrap trees, depth 14, minimum leaf size 3, sample fraction 0.7 and the original weights/seed.

Before fitting, the tool checks family, parameters, exact train/validation trip sets and the original records' reuse hashes. Every computed fold metric, full OOF metric and identity field must agree before publication. OOF rows must be covered exactly once with finite probabilities in `[0,1]`. Comparison includes confusion matrices and per-trip metrics, not only the selection objective.

Student restoration first checks the current final model/decoder freezes, SHA bindings and explicit TRAIN-only selection declarations. It then loads the actual published KNN/Bagging estimators read-only and predicts all 175,778 student rows. It makes no all-80 fit. Existing archives are checked from stored probabilities without prediction. Full saved metrics and row identities must agree before any new archive is published.

Allow several minutes for missing TRAIN archives. Historical [KNN CV](<../experiments/phase_three/cv/knn_distance_weighted.json>) took about 300 seconds, mostly exact-neighbor prediction; [Bagging CV](<../experiments/phase_three/cv/bagging.json>) took about 204 seconds. Together they can take roughly eight minutes on the measured machine. Full [KNN student prediction](<../experiments/phase_three/evaluation/knn_distance_weighted.json>) took about 187 seconds; [Bagging student prediction](<../experiments/phase_three/evaluation/bagging.json>) took under a second. Another machine/build can differ.

A restoration that does work requires a new run ID. It records fresh fold costs, runtime/source hashes and failure/success diagnostics under the excluded private replay-input directory, with no changes to historical timing records. Metric drift beyond `1e-12` fails the stage and leaves diagnostics. It does not publish newly computed family archives when either family's metric validation fails, loosen checks, or refit to chase a historical score. Regenerated prediction NPZ container bytes can differ; acceptance uses strict metrics and identical identity arrays. Dataset caches retain their stronger pinned-container and per-array requirements.

## 4. Run an isolated final experiment

After the input check passes, choose an unused final-run ID. These commands were not executed by the input checker:

```bash
.venv/bin/python -B experiments/final_five/run_final_five.py --stage train --run-id reproduction-1
.venv/bin/python -B experiments/final_five/run_final_five.py --stage student --run-id reproduction-1
.venv/bin/python -B experiments/final_five/run_final_five.py --stage clean --run-id reproduction-1
```

Reuse the same final-run ID for all three stages. The maintained runner writes isolated experiment, artifact and report trees; guards refuse complete or partial pre-existing outputs. The published bundle, baseline models and history remain unchanged. TRAIN freshly runs GaussianNB plus default/tuned HistGB CV and all-80 fits while validating and reusing LR/KNN/Bagging. This took roughly 30 seconds once the private baseline inputs existed, excluding their reconstruction cost. Student adds fresh final-model probabilities and strict saved-model fixture checks alongside the restored baseline arrays.

The clean stage needs the separate clean environment created above. It requires exact package-version equality and probability replay on the first 2,000 TRAIN candidate rows with `atol=1e-12` and `rtol=0`. The selected [published model freeze](<../artifacts/final_five/model_freeze.json>) binds a fresh all-80 HistGB fit at the saved TRAIN-selected parameters, SHA-256 `c9cc8e9f36838f656a94d07ed69632979df9981ca94a9daf3b52fbce052e4b34`. A new isolated bundle is a new measurement; matching parameters alone do not promise byte-identical joblib output.

### Historical Bayesian evidence

The [current Bayesian provenance](<../experiments/final_five/current/bayesian_optimization.json>) reuses the [historical 12-trial summary](<../experiments/bayesian_optimization.json>) and [trial log](<../experiments/bayesian_trials.jsonl>): one measured baseline prior, three seeded design proposals and eight genuine GP expected-improvement proposals. This is not a newly executed search. The runner reads each trial CV record: [the baseline prior](<../experiments/cv/hist_gradient_boosting.json>) and [01](<../experiments/cv/bayes_hist_gradient_boosting_01.json>), [02](<../experiments/cv/bayes_hist_gradient_boosting_02.json>), [03](<../experiments/cv/bayes_hist_gradient_boosting_03.json>), [04](<../experiments/cv/bayes_hist_gradient_boosting_04.json>), [05](<../experiments/cv/bayes_hist_gradient_boosting_05.json>), [06](<../experiments/cv/bayes_hist_gradient_boosting_06.json>), [07](<../experiments/cv/bayes_hist_gradient_boosting_07.json>), [08](<../experiments/cv/bayes_hist_gradient_boosting_08.json>), [09](<../experiments/cv/bayes_hist_gradient_boosting_09.json>), [10](<../experiments/cv/bayes_hist_gradient_boosting_10.json>) and [11](<../experiments/cv/bayes_hist_gradient_boosting_11.json>). All aggregate numeric logs must remain public. Their private prediction archives are unnecessary for this HPO-provenance validation.

Use the corrected final [current/results.json](<../experiments/final_five/current/results.json>). The first root-level attempt stopped on GaussianNB float32 probability normalization and remains failed-run provenance, explained in the [protocol](<../experiments/final_five/PROTOCOL.md>). The exact measured runner is archived as [run_final_five_measured.py](<../experiments/final_five/executed_source/run_final_five_measured.py>), SHA-256 `6e25f0492c8d48d62ff21ac05e5e5e5f6306ffdc27485458ee5d2f69e20adb65`. Use the maintained runner for reproduction; its later output-safety fixes are documented in [reproduction hardening](<FINAL_REPRODUCTION_HARDENING.md>). Historical source hashes remain unchanged.

## Verification for this preparation change

The [synthetic tests](<../tests/test_prepare_final_five_inputs.py>) use genuine tiny sklearn fits/predictions in temporary directories. They cover missing-stage checks, existing-file no-fit behavior, exact fold/parameter checks, strict metric/identity failures, freeze declarations, traversal/symlink refusal, concurrent destination preservation and private save/reload. They do not read actual private trips or establish a completed real-data regeneration.

```bash
.venv/bin/python -B -m unittest tests.test_prepare_final_five_inputs tests.test_final_five tests.test_phase_input_preparation
.venv/bin/python -B scripts/prepare_final_five_inputs.py --stage check
```

The actual local input check passed with zero fits, prediction calls or cache builds and no evaluator access. It verified both full pinned caches, the original grouped folds, all six reused archives, every saved aggregate/fold metric and the three baseline estimators. The [public check summary](<../reports/final_five/reproduction_inputs_check.json>) contains that result. No heavy real-data CV, HPO, full-student prediction or isolated scientific rerun was started for this change. A clean clone still requires authorized raw data and a compatible numerical build; reconstruction and a fresh scientific run remain explicit commands.
