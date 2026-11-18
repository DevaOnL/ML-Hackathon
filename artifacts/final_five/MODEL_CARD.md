# PathSnap final five-family model card

## Selected artifact

The final selected estimator is scikit-learn HistGradientBoostingClassifier. It is a genuine fitted model, not a lookup table or a geometry-only substitute. Final bundle: [model.joblib](<model.joblib>), SHA-256 `c9cc8e9f36838f656a94d07ed69632979df9981ca94a9daf3b52fbce052e4b34` (147,906 bytes). Its parameters are 104 boosting iterations, learning rate 0.15902391896503631, max_leaf_nodes 22, min_samples_leaf 38, l2_regularization 0.8604220152998319, random_state 20261118, early_stopping False.

## Training and five-family selection

The exact comparison comprises Logistic Regression, distance-weighted KNN, bootstrap Bagging, Gaussian Naive Bayes and HistGradientBoosting. LR/KNN/Bagging actual fits and whole-trip CV from the completed three-family phase are reused read-only and verified against saved prediction identities and metrics. GNB, default HistGB and tuned HistGB were freshly cross-validated and fitted for this final stage. Tuned HistGB is a configuration of the fifth family, not a sixth model family.

Training uses all 80 TRAIN trips: 350 deterministic observations per trip (28,000 observations), 269,463 candidate rows and 38 graph/GPS-derived features. All learned preprocessing is fitted inside each TRAIN fold; only LR/KNN/GNB need StandardScaler. KNN stores a declared 60,000-row TRAIN-only stratified reference bank; that is part of its fitting, not student sampling. Candidate fitting weights balance observations and trips.

Family selection uses only out-of-fold equal-trip mean observation top-1 proxy accuracy across the same three whole-trip GroupKFold partitions. HistGB default scored 0.8926785714285714, Bagging 0.8914642857142857, LR 0.884, KNN 0.8767857142857143 and GaussianNB 0.8734285714285714. Student and evaluator metrics never choose a winner.

The previously completed TRAIN-only HistGB Bayesian study is reused after validating its complete trial evidence: one measured default prior, three seeded proposals and eight Gaussian-process expected-improvement proposals, 12 total. Fresh default and tuned CV reproduce its recorded objective; the tuned winner scores 0.893392857142857. The final tuned estimator was freshly refitted on all 80 TRAIN trips in 4.396 seconds, before its new student test. Historical Bagging optimization remains separately preserved and does not change this choice.

## Student-development testing

Every family and the tuned HistGB variant are evaluated on all ten student-development trips (19,160 observations; 175,778 candidate rows). This partition has historical development use and is not a globally blind final test. All candidate labels are reference-route projection proxies, not independent per-GPS ground truth. There are 578 TRAIN and 19 student observations whose proxy target is missing from the candidate set; there are no empty candidate sets.

Tuned HistGB pooled observation top-1 is 0.9067849686847599, binary macro F1 0.9447159610168164 and binary log loss 0.05549988431377391. The final results record is [results.json](<../../experiments/final_five/current/results.json>); its measured values take precedence over rounded prose. Observation-edge macro F1, raw classifier continuity, candidate categorical NLL and directed-arc multiset route F1 have distinct denominators and must not be relabelled as the same accuracy.

## Precision correction and actual verification

The first attempted final run stopped at full-fit GNB scoring because sklearn float32 class-probability sums deviated by up to 1.52e-5. It saved no final model. GNB prediction now promotes only its input to float64; probability validity checks were not relaxed. Corrected fresh GNB CV/training/testing records are under experiments/final_five/current/. The incomplete earlier attempt is preserved but is not final evidence. Other estimators are unaffected.

All six saved configurations pass a strict 257-row student fixture against their measured probabilities (absolute tolerance 1e-12, zero relative tolerance). A separate clean Python environment loads the selected bundle and reproduces a hash-pinned 2,000-row TRAIN fixture exactly, with actual child versions recorded. Python 3.14.6, NumPy 2.5.5, SciPy 1.18.0, scikit-learn 1.8.0 and joblib 1.5.3. The older cross-build KNN floating-point limitation remains disclosed in the three-family report; it does not affect the selected HistGB strict replay.

## Inference limits

Features use the offline track, including surrounding/future observations. This is not a causal turn-by-turn navigation system. Inference reads GPS and graph only; reference routes are opened by a separate scorer only after prediction. Decoding uses directed graph transitions and refuses disconnected teleportation. Any TRAIN decoder calibration after the all-80 fit is in-sample and must be labelled as such. The final decoder has a separate freeze record.

The ten newly supplied evaluator trips were unopened at model freeze and stayed unopened until the decoder freeze and a preregistered evaluator protocol. Subsequent frozen inference covered all 19,674 observations, producing ten connected predictions before any private reference was parsed. Their trip-macro directed-arc F1 is 0.9226442590786567 and length F1 0.96580634225047; exact routes are 0/10. These scores never changed selection, feature schema, hyperparameters, candidate widths or decoder parameters. Metrics remain N/A where references or required probabilities are absent. See [evaluator acceptance](<../../reports/final_five/evaluator/acceptance.json>).

## Provenance

AI-assisted development and administrative GitHub account use are disclosed in the repository. Nothing in this card asserts independent human review, a physical two-laptop venue demonstration, a completed viva, a plagiarism clearance or unseen evaluation results.
