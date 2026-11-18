# PathSnap requested-trio model card

## Frozen serving model

The selected model is a genuine scikit-learn `BaggingClassifier` with 48
bootstrapped decision trees, `max_samples=0.7160628448832399`, tree depth 13
and minimum leaf size 2. It scores binary candidate-arc compatibility from
38 graph/GPS features. A directed Viterbi decoder turns those scores into
a chronological connected route. It never reads reference routes during
prediction.

- Artifact: [model.joblib](<model.joblib>)
- SHA-256: `c7f2572d5d037d07a5c07710fbb1ffefd4ff13fbe866c8940022e4902669c733`
- Decoder: [match_config.json](<match_config.json>)
- Feature order: [feature_schema.json](<feature_schema.json>)
- Recorded fit and selection: [model_metadata.json](<model_metadata.json>)
- Frozen file identities: [artifact_manifest.json](<artifact_manifest.json>)
- Live service: <https://ml.d3va.dev>

## Training and selection

The phase includes exactly Logistic Regression, distance-weighted KNN and
Bagging. The tuned Bagging variant is the same family, not a fourth model
family. The two remaining family deliverables are deferred by user instruction.
Historical experiments remain separate provenance.

Training uses 80 trips, 28,000 deterministically sampled GPS observations and
269,463 candidate rows. Three whole-trip TRAIN folds select the family by
trip-macro observation top-1 accuracy. The Bayesian search has eight total
trials: one measured baseline prior, two seeded proposals and five Gaussian
process expected-improvement proposals. The tuned winner scores 0.8925
versus 0.891464 for default Bagging on that TRAIN objective.

LR is a fold-local StandardScaler followed by binary LogisticRegression.
KNN is a fold-local StandardScaler followed by genuine Euclidean,
inverse-distance-weighted KNeighborsClassifier (`k=25`, brute force). Its
predeclared bank contains 60,000 class-stratified weighted TRAIN rows, with
fold-local selection and no validation or student rows. Default Bagging has
40 bootstrap decision trees.

## Measured development results

All three families and the tuned winner were tested on all ten
student-development trips: 19,160 GPS observations and 175,778 candidate
rows. Observation targets are projections onto the supplied reference route;
they are proxy labels, not independently provided point-aligned truth.

| Candidate classifier | Student observation top-1 | Binary candidate macro-F1 |
| --- | ---: | ---: |
| Logistic Regression | 0.895198 | 0.932936 |
| Distance-weighted KNN | 0.890501 | 0.927488 |
| Default Bagging | 0.904802 | 0.943727 |
| Frozen tuned Bagging | 0.903392 | 0.943575 |

Default Bagging is slightly better on student top-1. The original TRAIN-only
selection is retained; no winner or hyperparameter was changed after seeing
these scores. Candidate misses remain in the denominator (578 TRAIN and
19 student observations). No candidate sets are empty in these caches.

The frozen classifier plus HMM has mean per-trip directed-arc multiset F1
0.949238, length-weighted F1 0.978880, edit similarity 0.913668 and continuity
1.0. All ten routes are connected; zero are exact sequence matches. The
geometry-HMM baseline has length-weighted F1 0.973448. The prior historical
HistGradientBoosting result was slightly higher than current Bagging and is
not relabelled as an improvement by the requested trio.

## Limitations and independence

Student trips were used in historical development, so this is not a globally
blind final test. Decoder calibration uses eight TRAIN trips after the
all-80 classifier fit and is therefore in-sample calibration, not held-out
validation. Categorical emission NLL is a normalized candidate score, not
a posterior probability for the HMM route. One student reference contains a
broken join; references are scored as supplied. These results do not
establish road-safety or production-navigation suitability.

After the freeze, testing and deployment, the user reported adding ten
evaluator trips and asked for a separate website section later. Those files
were not inspected or processed for this report and did not influence model
or decoder selection. No evaluator performance is claimed.

## Runtime and reproducibility

Python 3.14.6, NumPy 2.5.5, SciPy 1.18.0, scikit-learn 1.8.0 and Joblib 1.5.3
are the measured runtime versions. Selected Bagging probabilities match
exactly on a 2,000-row TRAIN fixture in the original and clean runtimes.
A different KNN numerical build with identical version strings differs by
6.38e-7 on one TRAIN diagnostic row; strict saved-prediction acceptance
therefore uses the original training runtime and retains a 1e-12 tolerance.
The KNN limitation is disclosed rather than hidden by a relaxed threshold.

The [independent actual-model check](<../../reports/phase_three/model_acceptance.json>),
[clean-runtime check](<../../reports/phase_three/hardened_clean_environment_acceptance.json>)
and [full delivery reconciliation](<../../reports/phase_three/delivery_verification.json>)
verify actual saved artifacts. Ten real public HTTPS responses match final
CLI route geometry, observation states and scores. Actual API calls with all
reference files removed succeed for LR, KNN and Bagging, while reference
metrics correctly remain unavailable. No prediction cache or Gaussian
fallback substitutes for the learned estimator.

Only load trusted Joblib files. A fresh clone needs authorized local raw
inputs for serving or rebuilding excluded row-level caches. See the
[reproduction guide](<../../docs/PHASE_REPRODUCTION.md>). A deliberate new
numerical run requires an explicit isolated `--run-id`; completed or partial
stages refuse accidental overwrites. The current frozen artifact is ready
for inference and does not require refitting.
