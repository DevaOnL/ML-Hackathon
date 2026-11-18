# Accuracy-improvement experiment

The user requested higher accuracies after the reviewed five-family release. This iteration runs on an isolated branch; the released model, decoder, report, prediction archives and existing website remain unchanged until measured improvements pass validation. New code is opt-in rather than a weakened version of the old model loader.

## Comparison design

The [preregistered protocol](<../experiments/accuracy_improvement/PROTOCOL.json>) pins the unchanged TRAIN cache, proxy-target definition, candidate pool and every family configuration. TRAIN has 80 trips; a hash-based split reserves 16 for a single confirmation comparison in this iteration and uses 64 for tuning with three whole-trip folds. Earlier releases trained on all 80, so these reserved trips are **not historically unseen data** and are not presented as a new official test set. Tuning workers never score reserved rows.

Each of Logistic Regression, weighted KNN, Bagging, GaussianNB and HistGradientBoosting has six fixed screen configurations. Fold 0 screens them; the top two and the original-feature control undergo complete three-fold validation. After the user requested an expedited finish, Bagging was reduced to the leading screened candidate plus the paired original-feature control, before any confirmation scoring; the optional second full-CV finalist was stopped and its records preserved in the budget amendment. These are tuning estimates, not unbiased final test scores. New baseline controls are fitted on the identical partitions, numerical precision, targets and observations as the proposed feature variants. The earlier 80-trip CV results remain historical and are not used as the denominator for claims about 64-trip improvements.

The primary classification objective remains unconditional equal-trip observation top-1 accuracy. Missing proxy targets still count as wrong. Binary candidate accuracy, binary macro F1, raw binary log loss, graph-qualified observation edge F1, per-trip deltas, coverage and timing remain separate results. We will retain declines rather than label every changed metric an improvement.

## What changes

The [opt-in feature transformer](<../mapmatch/accuracy_features.py>) adds deterministic graph/GPS candidate geometry and interactions without route access, labels, graph-local identifiers or learned cross-trip preprocessing. Its context schema retains the 38 old columns as an unchanged prefix; compact and physics profiles reduce correlated distances and direction terms for KNN and GaussianNB. Learned scaling, skew transforms and feature reduction fit only within each TRAIN fitting fold. GaussianNB predicts in float64 and all model outputs must pass strict finite, bounds and sum-to-one checks before baseline-compatible clipping.

The [common experiment runner](<../scripts/accuracy_common.py>) enforces exact preregistered whole-trip folds and parameters, source hashes, fixed KNN sampling and private row-output locations. The [central menu runner](<../scripts/accuracy_trees.py>) records all measured attempts and permits resumption only when saved trial provenance matches; it refuses overwriting existing model/probability outputs.

Decoder experiments must use the corresponding fold-trained estimators to score every original GPS observation, not the sampled 350-row classifier validation archive alone. Initial decoder comparisons keep the candidate policy and dense observation denominator fixed. Changing candidate count changes candidate-relative predictors and requires a separately declared distribution/feature-policy experiment. Routes remain connected directed walks, and real repeated arcs/cycles cannot be removed merely to inflate a score.

## Confirmation and deployment gate

Features, family parameters, ranking rules and the route decoder freeze before the reserved comparison. Student-development results may be reported only after that freeze and retain their historical development-data label. The evaluator set has already been scored; it cannot become a new blind tuning or confirmation set. No tuning worker reads it.

No production model is replaced solely because a tuning score rises. Replacement requires a reproducible, strictly validated candidate with paired gains, disclosure of any metric decline, clean loading/inference and connected-route checks, full regression tests and independent automated review. Human/organizer approval is not implied by coding-agent review.
