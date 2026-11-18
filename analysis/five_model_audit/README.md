# Five-model audit: historical GaussianNB + HistGradientBoosting artifacts

Independent verification of the EXISTING historical artifacts for the two remaining model
families, so no one wastes time refitting before the website OpenCode theme. The LR / KNN /
Bagging trio was already completed under phase_three and is out of scope here.

## Verdict

| Family | Variant | Verdict |
|---|---|---|
| gaussian_nb | default | **PASS** |
| hist_gradient_boosting | default | **PASS** |
| hist_gradient_boosting | tuned | **PASS** |

`passed: true` — 66/66 checks, zero failures. Every replay against recorded numbers is
**bit-exact (max delta 0.0)** under the strict 1e-12 tolerance. Nothing needs refitting.

## Scope

- `artifacts/baselines/{gaussian_nb,hist_gradient_boosting,hist_gradient_boosting_tuned}.joblib`
- `artifacts/model.joblib` (frozen tuned HGB serving model) + `model_metadata.json`
- `experiments/predictions/{gaussian_nb,hist_gradient_boosting,hist_gradient_boosting_tuned}_{student,oof}.npz`
- `experiments/evaluation/*.json`, `experiments/cv/*.json`, shared `experiments/cv_folds.json`
- `experiments/cache/train_edee9a8f151dd25e.npz`, `student_test_*.npz/json` (caches read; raw
  data only hashed via manifest)
- HPO records: `bayesian_optimization.json`, `bayesian_trials.jsonl`, `baseline_selection.json`

Guarantees: no refitting, no HPO/decoder reruns, no external HTTP, no access to
`evaluator_test/` or archives, no modification of old artifacts/CV/preds/metrics. Writes are
confined to this directory. Scripts here may read private actual data but emit only
aggregates and hashes.

## How to run

```bash
cd /home/dva/Downloads/ML_Hackathon
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/five_model_audit/verify_gnb_histgb_audit.py
```

Output: `gnb_histgb_audit.json` (this directory). Threads are pinned to match the historical
training environment.

## What the checks establish

- **Provenance / integrity**: raw-input manifest SHA-256 matches; artifact SHA-256 matches
  acceptance records and the frozen `model.joblib`; bundles record the all-80-trip TRAIN
  cache and the shared config (seed 20261118, radius 80 m, max 12 candidates, spacing 40 m);
  caches have the expected shapes and deterministic linspace sampling; TRAIN/student trips are
  disjoint; folds equal the shared `cv_folds.json`.
- **Estimators are what they claim to be**: GNB is a StandardScaler + GaussianNB(var_smoothing=1e-7)
  pipeline; its stored `class_count_`, `theta_`, `var_`, `class_prior_` reproduce the
  weighted closed-form moment equations on the actual scaled TRAIN cache **bit-exactly under
  float32 arithmetic** (the historical fit ran float32 end to end). HGB estimators match their
  recorded parameters and fitted iteration counts. Saved predictions equal the literal Gaussian
  posterior formula from the stored parameters, bit-exactly.
- **Prediction replays (1e-12, actual result 0.0 everywhere)**: saved student/OOF npz arrays
  reproduce all recorded student, train-resubstitution, OOF and per-fold metrics exactly; fold
  bookkeeping closes (validation candidate rows sum to 269463 across 3 folds = 28000
  observations; train+val rows = 269463 per fold). OOF probabilities come from per-fold models
  (verified isolated from the final fit), which were not retained — hence verified via
  saved-array replay, not re-prediction.
- **HPO honesty**: 12 trials with the recorded composition (1 measured prior + 3 seeded + 8 GP-EI),
  winner trial 5 (`bayes_hist_gradient_boosting_05`, objective 0.893392857142857), parameters
  consistent across every record; objective is TRAIN grouped-CV only; baseline selection never
  consulted student_test.
- **Cross-environment**: predictions identical (0.0) between the original `.venv` and a clean
  `.venv-clean` for all three artifacts plus `model.joblib`.

## Findings and disclosures

1. **numpy drift is immaterial.** Artifacts were trained under numpy 2.5.2; current environments
   run 2.5.5. Despite that, the float32 fit accumulation and the historical scoring path
   reproduce every stored parameter, saved prediction and recorded metric bit-exactly (0.0).
2. **GaussianNB ran float32 end to end; live serving uses a promoted wrapper.** The current
   working tree contains an *uncommitted* `learning.predict_positive_proba` fix that promotes
   GaussianNB inputs to float64, because raw float32 likelihoods violate the sum-to-1 guard on
   2 of 269463 TRAIN rows (max deviation 1.52e-5; 0 student rows). The artifacts predate this
   change. Re-scoring the saved GNB artifact through today's wrapper differs from the historical
   predictions by at most ~1.0e-5 in probability on extreme near-clip rows (typical rows
   ≤1.3e-6) and moves student metrics slightly *better* (top1 +6.8e-4, trip_macro_top1 +1.7e-3,
   log_loss −2.6e-3, macro_f1 unchanged) — it removes float32 noise. All historical records
   remain exactly reproducible via the historical float32 path, which this audit uses.
   **The final website model is the tuned HGB, which is unaffected by the promotion.**
3. **Fold records count CANDIDATE rows** (validation_rows sum 269463; 28000 underlying
   observations). OOF predictions come from unretained per-fold models, verified via
   saved-array replay.
4. **student_test access was historical development-only** (after selection, before the final
   report) — not blind. `evaluator_test/` and archives were never accessed by this audit.
5. **`model.joblib` was frozen before the recovery refit** of the remaining baselines; the
   tuned baseline artifact is an independent deterministic refit verified to predict
   identically (max delta 0.0).

## Files

- `verify_gnb_histgb_audit.py` — the audit (read-only with respect to everything outside this
  directory).
- `gnb_histgb_audit.json` — machine-readable report: verdicts, 66 checks with details,
  artifact hashes, replay summaries, HPO inventory, disclosures.
