# Completed TRAIN-only tuning

All five prescribed model families improved the preregistered primary observation top-1 metric against freshly fitted original-feature controls on the same64 TRAIN trips and three whole-trip folds. These are model-selection estimates, not independent confirmation or blind-test results.

| Model | Original-feature control | Frozen candidate | Change |
|---|---:|---:|---:|
| Logistic Regression | 87.634% | 88.246% | +0.612 percentage points |
| Distance-weighted KNN | 86.696% | 88.228% | +1.531 points |
| Bagging | 88.420% | 89.379% | +0.960 points |
| Gaussian Naive Bayes | 86.478% | 87.446% | +0.969 points |
| HistGradientBoosting | 88.344% | 89.402% | +1.058 points |

Feature/parameter/decoder decisions were frozen before reserved16 scoring. The reserved TRAIN trips were historically used by earlier releases and cannot be called newly blind. Binary F1 and log-loss are separate; some secondary metrics decline. Full paired confirmation results and route metrics are reported separately when complete.

The expedited Bagging budget retained all six measured fold0 screens, then completed the leading finalist and control through three folds. The optional second finalist was stopped at the user request before any confirmation scoring; its records remain preserved.

[Protocol](<../../experiments/accuracy_improvement/PROTOCOL.json>) · [Budget amendment](<../../experiments/accuracy_improvement/BUDGET_AMENDMENT.json>) · [Frozen selection](<../../experiments/accuracy_improvement/confirmation/frozen_selection.json>)
