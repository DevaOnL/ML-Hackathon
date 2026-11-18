# Final five-family protocol

Frozen before new numerical work. This stage extends the completed three-family phase with Gaussian Naive Bayes and HistGradientBoosting. Historical records for these two exist; independent audit checks them, but this stage will also perform new isolated CV and all-80 fits for the two requested additions so the final experiment has current measured training and complete student testing. Existing measured files are never overwritten.

## Data and preprocessing

Use the identical hash-verified TRAIN candidate cache: 80 whole trips, 350 deterministic observations per trip (28,000 observations; 269,463 candidate rows; 38 graph/GPS-only features). Use the existing deterministic three whole-trip GroupKFold partitions. TRAIN route projections provide proxy supervision only. The ten student-development trips (19,160 observations; 175,778 rows) have historical development use and are not globally blind. They are never inputs to fitting, family selection or hyperparameter selection. The new evaluator inputs must remain unopened until model and decoder freezes.

GaussianNB uses a fold-local StandardScaler and genuine GaussianNB(var_smoothing=1e-7) with existing trip/observation candidate weights. HistGradientBoosting uses genuine HistGradientBoostingClassifier(max_iter=120, learning_rate=.09, max_leaf_nodes=23, min_samples_leaf=40, l2_regularization=.5, early_stopping=False, random_state=20261118). No row-random validation or new feature changes. LR, KNN and Bagging reuse their previously verified actual grouped CV and all-80 fits. KNN retains its declared 60,000-row TRAIN-only reference bank.

## Selection and Bayesian optimization

Compare the five untuned families only, using TRAIN out-of-fold equal-trip mean observation top-1 proxy accuracy. Tie-break by OOF binary macro F1, then lower binary log loss. Student/evaluator performance must not select or switch the winner. The previous tuned Bagging remains a disclosed historical variant, not a sixth family or new baseline.

If HistGradientBoosting wins and new default CV reproduces the existing fixed-configuration grouped OOF result, the completed historical 12-trial HistGradientBoosting Bayesian study may be reused only after validating every recorded configuration, fold set, objective and GP-EI metadata. Exactly one measured default prior, three seeded design proposals and eight genuine GP expected-improvement proposals are recorded. New final tuned all-80 fit uses that fixed TRAIN-only winning configuration. The report must distinguish reused search measurements from the fresh fits. No new HPO is fabricated or implied. A differing baseline objective or invalid trial study stops the stage for review.

## GaussianNB numerical precision correction

The first attempt completed three-fold CV but stopped at GNB full-fit scoring: sklearn returned float32 class probabilities whose sum differed from one by up to 1.52e-5. None were nonfinite or outside [0,1]. Strict validation correctly stopped the run before saving a final model. GaussianNB inference now promotes its feature input to float64 before sklearn prediction; all other estimators and the strict validation tolerances are unchanged. The original incomplete attempt remains intact at experiments/final_five/{cv,selection.json,bayesian_optimization.json}; all corrected fresh CV/training/student results are isolated under experiments/final_five/current/. No incomplete measurement is claimed as final. A measured diagnostic found float64 scoring sums within 2.3e-14.

## Fresh training, testing and freeze

Perform and time the new GNB and default/tuned HistGB all-80 fits. Save all three actual estimators with full schema/training provenance. Evaluate all ten student-development trips, raw candidate metrics and observation-edge metrics. Strict fixture replay must reproduce stored predictions at atol=1e-12, rtol=0. Retain identity-qualified arrays privately. Consolidate all five families, tuned HistGB and historical tuned Bagging with distinct provenance.

Re-export the final selected estimator into a new artifacts/final_five bundle, with frozen SHA-256 and strict clean-child probability replay. Preserve phase_three and historical artifacts. Only after all five training/testing acceptance checks pass may website theme work start. Calibrate the final selected model decoder on TRAIN only; disclose any in-sample calibration. Freeze decoder parameters before accessing new evaluator inputs. No student/evaluator-based adjustment or retry parameter tuning.

## Execution safety

Fresh stages refuse any existing complete or partial numerical output before loading data. No overwrite flag. Reproduction must use a new isolated destination rather than editing the published measurements. Model and cache identities, actual current versions, timing, warnings and source hashes are recorded. Tests use synthetic data; no hidden external requests or background fits.
