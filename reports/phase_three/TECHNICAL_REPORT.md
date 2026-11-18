# PathSnap Technical Report: Phase Three

Map matching converts a sequence of noisy GPS positions into the most plausible ordered list of directed road segments. PathSnap does this in two stages: a supervised classifier scores every candidate road-arc observation for a GPS point, and a Viterbi decoder chains the per-point choices into a coherent route under a road-network transition model. This report documents the phase-three deliverable: three classifier families (logistic regression, distance-weighted KNN, bagged decision trees), a Bayesian-tuned variant of the selected family, and the frozen decoder evaluated on the 10-trip student development set.

Measured metrics and experimental settings are bound to recorded evidence by `scripts/build_phase_three_report.py`. Its final-render guards check model identity, the decoder freeze chain, figure hashes and live delivery records; `reports/phase_three/report_manifest.json` records the input SHA-256 hashes. Definitions and methodological explanations are reviewed prose, not automatically proved by the numeric checks. Repository: https://github.com/DevaOnL/ML-Hackathon.

This project was developed with AI coding assistance under the user’s direction. Section 9 describes that assistance and the limits of the verification.

> Scope caveat, stated up front. The competition brief asks for five model families. The user's instruction for this phase, recorded in `docs/PHASE_SCOPE.json`, limits the deliverable to three families plus a tuned variant of the selected one and defers the remaining two. The earlier five-family cross-validation is retained in Appendix A as provenance only. The identical Logistic Regression baseline is explicitly reused; other historical-family results are not presented as current-phase models, and this report does not claim five-family compliance.

<!--pagebreak-->

<!--toc-->

## 1 Problem and phase scope

The task provides 90 development trips. Each trip is a `.track` file of timestamped positions, a reference `.route` of directed zero-based arc ids, and its own directed road graph (`.nodes` and `.arcs` files). A submission must reproduce the reference arc sequence for held-out trips. After the current model was frozen, tested and deployed, the user reported adding ten evaluator trips and requested a separate website section later. Those inputs were not listed, inspected or processed for this report. Every score below concerns development data; no evaluator result is claimed.

The user's instruction fixes the scope of this phase:

- learn exactly three classifier families: logistic regression, distance-weighted KNN, and bagging (bagged decision trees);
- select among them by grouped cross-validation on TRAIN trips, without using student scores in this phase’s selection;
- tune only the selected family, with a Bayesian search of fixed budget;
- freeze the winner (model artifact plus match configuration) and only then run the student-test evaluation, route decoding, and live deployment checks.

The selection record (`experiments/phase_three/selection.json`) names bagging as the selected family under the objective out_of_fold_trip_macro_top1_accuracy. The consolidated results file records the serving model as bagging. Selection and tuning were recorded before this phase’s new student evaluation. Logistic Regression reuses a valid historical fit and evaluation, and earlier student-development results were already known. These are development results, not a globally blind final test. The builder re-checks the three-family scope.

## 2 Data, partitions, and exploratory analysis

### 2.1 Inventory and partitions

The completed byte and row audit covers 90 trips in 360 files: 227,577 track rows, 34,813 route rows, 4,061,945 arc rows, and 2,024,174 node rows. The corpus splits into 80 training trips and 10 student-test trips. This completed audit covers only those original development trips and makes no inventory claim about the newly reported evaluator inputs. Training used 28,000 labeled observations (a frozen ceiling of 350 per trip), which expand into 269,463 candidate feature rows under the candidate policy below. For 578 TRAIN observations the proxy target arc is absent from the generated candidate set; these target misses remain in the accuracy and NLL denominators. The student partition holds 19,160 GPS points across 10 trips, expanding to 175,778 candidate rows and 19,160 scored observations, with 19 proxy-target misses. Neither cache contains an empty candidate set: a missing target is different from having no candidate. Student rows are excluded from estimator fitting, family selection, HPO and this phase’s decoder search.

A sampled KD tree retrieves candidate road segments using 40 m index spacing. Exact point-to-segment projections, not the index samples, determine candidate distances and positions. The primary radius is 80 m and the initial limit is 12 candidates; radius expansion and deterministic width retries are described in section 5. Every candidate has 38 graph/GPS features, including distance, heading alignment at several windows, motion, projection fraction, road length, degrees and local ambiguity. No reference-route membership or road ID is an input feature. Degenerate directions and unavailable motion use the pipeline’s finite zero-valued encodings; malformed raw coordinates are rejected rather than imputed. Windows use surrounding GPS observations, including future samples, so this is an offline matcher rather than a causal real-time navigation system.

The supplied reference is an ordered route, not one label per GPS point. For supervision and offline scoring, each point is projected onto the entire reference route, independently of the generated candidates. The proxy target minimizes squared lateral distance plus a weak GPS-heading disagreement penalty. Each candidate receives a binary label for equality to that canonical directed proxy arc. The heading penalty is 32 square metres at full motion reliability; a best-versus-second score gap below 4 square metres marks an ambiguous proxy. Parallel roads, stops, intersections, repeated traversals and reference defects can make these derived targets uncertain. Geometry-based labels also favor geometry-based predictors. Reference routes are never read by prediction.

![Dataset inventory: file, row, and partition totals for all 80 training plus 10 student-development trips, sourced from the completed byte/row audit of the 360 original files.](visualizations/plots/dataset_inventory.png)

![Missing, invalid, and anomalous value audit across the development corpus, with repeated coordinates and unusually large GPS jumps distinguished from malformed rows.](visualizations/plots/missing_invalid_and_anomalies.png)

### 2.2 Timestamps and motion

The audit records 201,053 one-second TRAIN inter-observation intervals; the largest gap in any training trip is 153,064.0 seconds. The decoder's transition weights use shortest-path geometry, so gaps of this size are handled by the routing slack described in section 5 rather than by resampling.

![Distribution of timestamp gaps between consecutive GPS samples across training and student-development trips.](visualizations/plots/timestamp_gaps.png)

![Distributions of speed, heading change, and point-to-point displacement for training and student-development tracks.](visualizations/plots/trajectory_distributions.png)

### 2.3 Road networks

Each trip ships its own directed network. Across the 80 training trips the per-trip graphs contain 20,038 strong components and 3,121 weak components in total; the student graphs contain 227 weak components. Reference routes occasionally span multiple strong components of their graph, which is recorded as a reference property in section 2.5 rather than smoothed away.

![Per-trip road network sizes and connectivity, contrasting training and student-development graphs.](visualizations/plots/road_network_sizes_connectivity.png)

### 2.4 Geographic coverage

The training corpus contributes 208,417 GPS points and the student corpus 19,160. The geographic overlap plot is descriptive; no geographically blocked validation was performed. These scores do not establish generalization to unseen regions or road networks.

![Geographic coverage of GPS points across all development trips, training and student partitions overlaid.](visualizations/plots/gps_geographic_coverage.png)

### 2.5 Reference route defects

The reference routes contain defects that the pipeline treats as ground truth exactly as supplied. The training references contain 13 broken joins (consecutive route arcs that do not share an endpoint). The student references contain 1 broken join, in trip 00000076; the trip-specific audit independently records 1. The student references also contain 255 repeated-arc extra occurrences; these concentrate in trip 00000056, which is the only student trip (1 of 10) whose reference route spans more than one strong component, and 1 student trips contain repeated arcs in total.

Prediction cannot consult or repair a reference-route defect. Reconstruction joins selected states only through real directed shortest-path arcs; it fails explicitly if no connected path exists. Consecutive observations on one arc are collapsed, but nonconsecutive repeated traversals and real cycles are retained. Route-level scores are therefore computed against the references as supplied, defects included.

## 3 Model families and cross-validated design

### 3.0 Shared setup

All families share the candidate policy and are compared using 3 whole-trip TRAIN folds over 80 trips, with seed 20261118. The selection objective is equal-trip mean observation top-1 proxy accuracy (out_of_fold_trip_macro_top1_accuracy). Scalers, class weights, reference banks and estimators are fitted only on the training side of each fold. All candidates belonging to a GPS observation stay in the same trip and fold. The same folds support the 8-trial HPO budget.

| Model | Family | Exact configuration |
|---|---|---|
| Logistic regression | scaled linear discriminative logistic regression | {"C": 1.0, "max_iter": 500, "solver": "lbfgs"} |
| Distance-weighted KNN | distance-weighted k-nearest-neighbour candidate classifier over a class-stratified TRAIN reference bank | {"n_neighbors": 25} |
| Bagged decision trees | genuine bootstrap aggregation of decision-tree base learners | {"max_depth": 14, "max_samples": 0.7, "min_samples_leaf": 3, "n_estimators": 40} |
| Tuned bagging | genuine bootstrap aggregation of decision-tree base learners | {"max_depth": 13, "max_samples": 0.7160628448832399, "min_samples_leaf": 2, "n_estimators": 48} |
Table: Actual estimator families and initial or tuned configurations. The tuned variant is still Bagging, not a fourth family.

| Model | CV binary F1 | CV top-1 | Student binary F1 | Student binary LL | Fit (s) | Raw continuity |
|---|---|---|---|---|---|---|
| Logistic regression | 0.9328 | 0.8840 | 0.9329 | 0.0690 | 1.0 | 0.9287 |
| Distance-weighted KNN | 0.9272 | 0.8768 | 0.9275 | 0.0960 | 0.0 | 0.9230 |
| Bagged decision trees | 0.9424 | 0.8915 | 0.9437 | 0.0562 | 103.8 | 0.9322 |
| Tuned bagging | 0.9425 | 0.8925 | 0.9436 | 0.0560 | 133.1 | 0.9313 |
Table: TRAIN cross-validation versus student-development classification. Binary F1 is candidate-row macro F1. CV F1 is a mean over folds; student F1 and log loss (LL) pool candidate rows. CV top-1 is also trip-macro because TRAIN sampling is balanced. Fit time excludes feature generation and prediction; KNN’s small fit time is not its inference latency. Raw continuity is observation-assignment adjacency, not a decoded route score. Logistic Regression records are explicitly reused.

### 3.1 Logistic regression

Logistic Regression supplies a fast linear baseline on the engineered distance and heading features. Its sigmoid score is usable as an emission weight, but neither linear separability nor probability calibration was established by these experiments.

The model is binary LogisticRegression after StandardScaler, trained with weighted binary cross-entropy and L2 regularization using L-BFGS. Its initial parameters are {"C": 1.0, "max_iter": 500, "solver": "lbfgs"}. The existing all-80 fit, valid fold-local CV and prediction arrays are reused read-only, with schema, identity, metric and saved-model replay checks; they were not re-fit to manufacture a new result. Its prose family description in the results file is "scaled linear discriminative logistic regression".

Student-development performance appears in the comparison and detail tables; the CV-to-student difference is quantified in section 3.4.

### 3.2 Distance-weighted KNN

Distance-weighted KNN is included as a non-parametric contrast: it makes no linearity assumption and its decision boundary follows the local geometry of the feature space, which is a reasonable fit when candidate correctness is dominated by distance-like features. It is a lazy learner, so training is trivial while inference cost grows with the training set; the fit and inference timings in the tables below record that trade-off explicitly.

Class scores are distance-weighted vote shares over the {"n_neighbors": 25} neighbors, converted to a distribution for the log-loss column. All families’ binary log losses use the same labels and clipping policy and can be compared on that basis. None has a separate calibration study; KNN’s distance-weighted vote shares can produce extreme scores. Features are standardized within each fit, and neighbors use genuine brute-force Euclidean distance with inverse-distance weights. At inference time each fit scores against a frozen reference bank of 60,000 rows: class-stratified, training-weight-proportional sampling without replacement over each fit's own TRAIN rows, drawn under a deterministic seed and fixed before any student measurement, so the bank never reads student data.

### 3.3 Bagged decision trees

Bagged decision trees (bootstrap-aggregated {"max_depth": 14, "max_samples": 0.7, "min_samples_leaf": 3, "n_estimators": 40} ensemble) reduce the variance of individual trees while capturing the non-linear interactions between lateral distance, heading change, and motion speed that a linear model averages away. Bootstrap aggregation averages decision-tree class scores; no out-of-bag accuracy estimate is used for selection.

Trees split by Gini impurity and need no feature standardization. Zero encodings are ordinary numeric feature values, not a special missing-data treatment. This family was selected under the phase objective (section 1), and section 4 tunes it.

### 3.4 Per-family detail and overfitting evidence

| Model | CV binary F1 (std) | CV binary LL | CV inference (s) | Student top-1 |
|---|---|---|---|---|
| Logistic regression | 0.9328 +/- 0.0086 | 0.0744 | 0.045 | 0.8952 |
| Distance-weighted KNN | 0.9272 +/- 0.0087 | 0.0941 | 299.056 | 0.8905 |
| Bagged decision trees | 0.9424 +/- 0.0068 | 0.0572 | 1.174 | 0.9048 |
| Tuned bagging | 0.9425 +/- 0.0062 | 0.0575 | 1.377 | 0.9034 |
Table: Candidate-binary CV variation and prediction timing. Standard deviation is across the three folds; CV inference seconds sum all validation-fold predictions. Student top-1 pools GPS observations, including proxy-target candidate misses.

| Model | Raw continuity | Obs. edge macro F1 | Categorical NLL | Target miss rate |
|---|---|---|---|---|
| Logistic regression | 0.9287 | 0.8071 | 0.3476 | 0.0010 |
| Distance-weighted KNN | 0.9230 | 0.7955 | 0.5217 | 0.0010 |
| Bagged decision trees | 0.9322 | 0.8271 | 0.2824 | 0.0010 |
| Tuned bagging | 0.9313 | 0.8248 | 0.2817 | 0.0010 |
Table: Student observation metrics on the fixed all-family cache. Edge classes are graph-qualified. Raw continuity is the fraction of adjacent observation assignments on the same canonical arc or a directly connected directed arc, with no routed connectors. It does not measure reference agreement. Section 5.4 defines the NLL and distinguishes these denominators.

![Actual three-family training and inference timings, with a separate panel for each quantity; only the microseconds-per-observation panel uses a logarithmic axis.](visualizations/phase_three/plots/phase_three_training_inference_times.png)

![Candidate-binary confusion matrices for the three families and the tuned Bagging variant on all ten student-development trips.](visualizations/phase_three/plots/phase_three_confusion_matrices_all.png)

![Observation edge F1 and categorical emission NLL under the actual route-projection proxy targets, with distinct candidate-binary and observation denominators.](visualizations/phase_three/plots/phase_three_observation_level_metrics.png)

![Accuracy and NLL by recorded proxy ambiguity stratum; the 19 target-miss observations remain in overall scores rather than being silently discarded.](visualizations/phase_three/plots/phase_three_ambiguity_strata.png)

The CV-to-student difference is a diagnostic of these two partitions. CV predictions are out of fold; student metrics come from the all-80 fits, including the explicitly reused Logistic Regression record. Computed from the same frozen results file, the student macro F1 differs from the CV macro F1 by +0.0001 for logistic regression, +0.0003 for KNN, +0.0013 for bagging, and +0.0011 for the tuned variant; positive values mean the student partition scored higher. The corresponding top-1 differences are +0.0112, +0.0137, +0.0133, and +0.0109. The reported F1 gaps compare candidate-binary scores; CV F1 is the mean over folds, whereas student F1 is pooled over candidate rows. CV top-1 equals trip-macro top-1 because every TRAIN trip contributes the same 350 observations; the student top-1 column pools observations of unequal-length trips. Small or positive gaps do not establish absence of overfitting or independent test performance.

![TRAIN-CV versus student candidate-binary F1; development gaps are descriptive and do not establish a blind generalization bound.](visualizations/phase_three/plots/phase_three_generalization_macro_f1.png)

## 4 Tuning the selected family

The selected family, bagging, was tuned using GaussianProcessRegressor Matern-5/2 surrogate + analytic expected improvement, maximizing maximize out-of-fold equal-trip mean observation top1 proxy accuracy; fixed 3-fold GroupKFold on all 80 TRAIN trips on the same grouped folds. The budget was 8 trials in total: one measured prior, 2 seeded initial-design points, and 5 Gaussian-process model-based trials; the search took 1489.6 s. Tuning scored TRAIN cross-validation folds only; the student partition was never used to select or re-tune the configuration. The search space:

```json
{"max_depth": ["int", 8, 20], "max_samples": ["float", 0.5, 0.9], "min_samples_leaf": ["logint", 2, 10], "n_estimators": ["int", 30, 80]}
```

The best configuration found:

```json
{"max_depth": 13, "max_samples": 0.7160628448832399, "min_samples_leaf": 2, "n_estimators": 48}
```

The best objective reached was 0.8925 against 0.8915 for the untuned baseline. On the CV macro F1 scale, tuning moved the selected family by +0.0000; the student columns carry the corresponding development comparison. Default Bagging has slightly higher student top-1 than the tuned model. The TRAIN-only winner is retained; the model was not switched after seeing student results. The trial trace below shows every evaluated point, including the prior.

![Bayesian optimization trial trace for the selected family: objective value per trial, prior point included, with the best trial marked.](visualizations/phase_three/plots/phase_three_bayesian_trial_trace.png)

![Tuned versus untuned comparison of the selected family across CV and student-test metrics.](visualizations/phase_three/plots/phase_three_tuned_vs_untuned.png)

### 4.1 Frozen model feature importance

The plot below reads the frozen 48-tree Bagging artifact without refitting. It averages each tree’s impurity-based feature importances across the 38 features. Heading alignment and relative candidate distance rank highly. These are Gini split importances, which can favor high-cardinality or correlated predictors; they do not establish causal relevance or independent predictive value.

![Mean tree-impurity feature importance from the actual frozen tuned Bagging model; read-only artifact loading with model SHA-256 verification, no new fit.](visualizations/phase_three/plots/phase_three_feature_importance.png)

## 5 Route decoding: emission model, transitions, reconstruction

### 5.1 Emission model and scale fit

The decoder consumes the classifier's per-candidate probabilities as HMM emission weights. The geometry and transition scales reuse historical TRAIN-residual statistics (recorded in `reports/decoder/scale_fit.json`; its SHA-256 is pinned in the report manifest). The fitted geometry scale is sigma_m = 7.161. The raw transition-residual scale was 1.424672666285088 m and is clamped to its declared floor, giving beta_m = 2.000. No Baum-Welch re-estimation runs at inference time: the emission scale is frozen with the model, which is what makes the deployment reproducible. The frozen configuration is A_fitted, recorded at 2026-11-18T07:11:40.099678+00:00; the full freeze record (config hash, model hash, frozen timestamp) lives in `reports/phase_three/decoder/freeze.json` and the calibration plan in `reports/phase_three/decoder/calibration_plan.json`.

That configuration was chosen by a predeclared search over three candidates (`A_fitted`, `B_twice_beta`, `C_classifier_up`) scored on the 8 TRAIN calibration trips, in-sample after the all-trips fit; the per-trip results are in `reports/phase_three/decoder/search_results.jsonl`. The three configurations scored mean length F1 of 0.9531, 0.9516, and 0.9523 respectively, and A_fitted was selected. This is a development choice, not a held-out measurement, and it is disclosed as such. The freeze record also states plainly that earlier-phase student results existed as prior context when this phase began; the predeclared search itself read only TRAIN predictions and references, never student scores, and the student ablation of section 5.3 started only after the freeze.

### 5.2 Transitions and reconstruction

For candidate distance d and binary positive score p, the learned emission cost is classifier_weight * [-log(clip(p))] + geometry_weight * 0.5 * (d/sigma_m)^2. The frozen classifier and geometry weights are 1.000 and 1.000. Transition cost is |directed_network_distance - GPS_chord_distance| / beta_m, with a routing search horizon initialized to the GPS chord plus 300 m. It is not a speed-matching likelihood. Radius search may expand for an empty local set. A disconnected decode deterministically retries candidate widths from the initial 12 through 24 up to the limit 48; 48 is a maximum width, not a retry count. Decoding is Viterbi over the directed graph; loops and repeated arcs are preserved, and no connector arc is invented for the reference defects of section 2.5. Route multiset and length F1 plus ordered edit similarity are reported in the ablation table.

### 5.3 Ablation on all 10 student trips

The decoder ablation ran on 10 student trips in 81.9 s with no failed trips, comparing the raw classifier (argmax per observation, no sequence model) against the full classifier-plus-HMM pipeline. The geometry-HMM row reuses the historical default configuration where the phase did not rerun it.

| Variant | Top-1 proxy acc. | Route edge F1 | Length F1 | Edit sim. | Continuity | Obs. NLL | Miss rate | Seconds |
|---|---|---|---|---|---|---|---|---|
| Classifier only (raw) | 0.9162 | 0.7636 | 0.8174 | 0.6265 | 0.7422 | 0.2520 | 0.0005 | 0.9 |
| Classifier + HMM (this phase) | 0.9376 | 0.9492 | 0.9789 | 0.9137 | 1.0000 | 0.2520 | 0.0005 | 6.1 |
| Geometry + HMM (historical default, reused) | 0.9173 | 0.9262 | 0.9734 | 0.8747 | 1.0000 | 1.1082 | 0.0005 | 5.0 |
Table: Decoder ablation over all 10 student-development trips, trip-macro averages. Edit sim. is ordered edit similarity over the arc sequence; Obs. NLL is the per-observation normalized categorical NLL defined in section 5.4.

Pooling route overlap across trips, the classifier-plus-HMM variant reaches length F1 0.9814 and route edge F1 0.9546, against 0.8020 and 0.7534 for the raw classifier; the reused geometry default reaches 0.9784 and 0.9303 pooled. Trip-macro route metrics weight trips equally. Pooled route edge F1 sums matched and total arc occurrences across trips; pooled length F1 instead sums matched and total arc metres. Neither is observation-weighted. Observation accuracy and emission NLL are pooled by GPS count separately.

The routes actually delivered with this submission come from the frozen final CLI, and its evaluation over the ten student trips matches the classifier-plus-HMM row of the table: macro length F1 0.9789, macro edge F1 0.9492, and ordered edit similarity 0.9137, with 10 of 10 predictions fully connected and 0 exact route matches. The references themselves contain 1 discontinuity, which is scored as supplied rather than bridged. The per-trip route files are sha256-bound, trip by trip, in the offline delivery verification record, and the live HTTP API reproduces the same per-trip geometry hashes; the builder guards both bindings on every final render.

![Decoder ablation across all 10 student trips: per-trip route-level scores for the raw classifier, the classifier-plus-HMM pipeline, and the reused geometry default.](visualizations/phase_three/plots/phase_three_route_ablation.png)

### 5.4 Metric taxonomy

Candidate-binary macro F1 averages positive and negative class F1 over candidate rows; its binary log loss uses those same rows. Observation edge macro F1 instead treats trip_id:canonical_directed_arc_id as a multiclass label and averages across the union of reference and predicted edge classes. Observation top-1 accuracy requires equality to the single route-projection proxy target, not membership in any part of the reference route. Candidate-target misses remain incorrect.

For categorical emission NLL, positive scores are clipped, normalized within each observation, and the probability mass assigned to the proxy target is scored with a 1e-7 floor; an absent target has zero mass before the floor. This is an emission diagnostic, not a calibration guarantee or an HMM posterior. The paired raw/HMM ablation shares candidates and emission scores, so its NLL is identical for those two rows. The all-family cache and decoder retry policy can produce different observation metrics and must not be treated as identical probability fixtures.

Route edge F1 counts directed arc occurrences as a multiset, preserving repeated traversals; length F1 weights those occurrences by arc length. Ordered edit similarity is 1 minus Levenshtein distance divided by the larger route length in arc rows. These are explicit local metrics, not a claim about an unavailable organizer scorer. Route continuity is the valid-join fraction of the emitted route and is distinct from raw observation-assignment continuity.

## 6 Route map

The figure below is the decoded route for development trip 00000009 under the frozen phase configuration, drawn against the trip's road graph and GPS trace. The interactive version, `visualizations/phase_three/maps/phase_three_development_trip_00000009_final.html`, renders offline from local files. This trip’s reference has no broken joins. Defects in other references, including trip 00000076, are still scored as supplied.

![Decoded route for student-development trip 00000009 under the frozen phase-three configuration: GPS trace, road graph, chosen directed arcs, and reference route overlaid.](visualizations/phase_three/maps/phase_three_development_trip_00000009_final.png)

## 7 Deployment: API, web UI, and live verification

The service exposes the matcher over HTTP. A request carries only a trip identifier; coordinates are read server-side from the local development corpus and never appear in a request body or a log line. The local origin is http://127.0.0.1:8000 and the public origin is https://ml.d3va.dev, serving the frozen bagging model; the public verification record was checked 2026-11-18T07:29:13.065747+00:00 and reports the deployment ready with the frozen model and match-configuration hashes.

Live evidence was collected against the public origin: 10 of the 10 student trips matched successfully through the HTTP API (evidence recorded 2026-11-18T07:22:49.748936+00:00; the public verification record targets https://ml.d3va.dev and was checked 2026-11-18T07:29:13.065747+00:00). The evidence directory `reports/phase_three/live/` contains the match records, the browser capture, and the API terminal capture. The four screenshots below and the terminal capture are bound by sha256 into those JSON evidence files, and the builder's screenshots_hash_bound guard refuses a final render unless every hash matches; the terminal capture must specifically be bound in `reports/phase_three/live/api_terminal_evidence.json`.

![Web UI, desktop layout: trip selection and matched route summary on the public deployment.](reports/phase_three/live/screenshots/ui_desktop.png)

![Web UI, route map panel: decoded route drawn over the road graph for a selected student-development trip.](reports/phase_three/live/screenshots/ui_route_map.png)

![Web UI, mobile layout: the same trip matched on a narrow viewport.](reports/phase_three/live/screenshots/ui_mobile.png)

![Receiving API terminal: real service journal entries reconciled to a public HTTPS match request, including its request ID and model/configuration identities.](reports/phase_three/live/screenshots/api_terminal.png)

## 8 Reproducibility

The phase-three run used Python 3.14.6, NumPy 2.5.5, scikit-learn 1.8.0, and SciPy 1.18.0. The earlier clean-install record (prefix `.venv-clean`) pins NumPy 2.5.5 and scikit-learn 1.8.0 and recorded 108 passing unit tests at that checkpoint; this is not the current release-suite count. The historical five-family run of Appendix A trained under NumPy 2.5.2; the version divergence between that run and this phase is disclosed here rather than hidden behind a single version string.

Runtime determinism is recorded in three acceptance files rather than asserted. Replaying the KNN artifact's saved probabilities on the original runtime reproduces them exactly, with a maximum absolute probability delta of 0.0 on a 257-row TRAIN fixture. Its fitted reference bank has 60,000 rows; bank size is not replay-fixture size. Rebuilding the same KNN probabilities in an isolated clean environment (same NumPy, SciPy, and scikit-learn versions, different low-level distributions) is not bit-identical: the maximum absolute delta is 6.38e-07 on a fixture of 257 deterministic TRAIN rows, no student read, with 1 rows over a 1e-12 tolerance, while every binary threshold decision comes out equal; the original runtime is therefore retained instead of claiming cross-build bit identity. The selected bagging model replayed in the hardened clean environment reproduces all 2,000 fixture predictions with a maximum absolute probability delta of 0.0.

The frozen model artifact is `artifacts/phase_three/model.joblib` with SHA-256 c7f2572d5d037d07a5c07710fbb1ffefd4ff13fbe866c8940022e4902669c733. The freeze chain is checked on final render. Existing training, HPO, decoder search, route exports and public all-ten captures are complete and must not be repeated merely to regenerate this document. The following read-only checks use the repository root and the project virtual environment; use a new output path for delivery reconciliation. Full preprocessing and deliberate isolated new-run instructions are in `docs/PHASE_REPRODUCTION.md`.

```bash
# read-only figure and decoder checks; no training or route reruns
.venv/bin/python visualizations/phase_three/generate_phase_three.py status
.venv/bin/python visualizations/phase_three/generate_phase_three.py verify
.venv/bin/python scripts/phase_three_decoder.py self-check

# authorized local data required; writes only a NEW reconciliation report
.venv/bin/python scripts/verify_phase_three_delivery.py \
  --live reports/phase_three/live \
  --output reports/phase_three/new_delivery_reconciliation.json

# report readiness; --final regenerates only the document and its manifest
.venv/bin/python scripts/build_phase_three_report.py --check
.venv/bin/python scripts/build_phase_three_report.py --final
```

## 9 Limitations, honest disclosure, and AI provenance

The user reported adding ten evaluator trips after this model’s freeze, test and deployment and asked to add their website section later. They remain uninspected and unevaluated here. Student trips are development data with historical experimental use, so these scores are not an independent estimate of unseen evaluator performance. The five-family requirement of the brief is deferred by user instruction (section 1 caveat), and the historical five-family numbers in Appendix A are provenance, not current results. Top-1 accuracy is a proxy metric. Route-level metrics are local scoring definitions; no official organizer score is asserted. The geometry-only baseline of Appendix A is not learned-final evidence for this phase. Reference defects (section 2.5) are scored as supplied.

The user chose the requested three families and deferred the other two. AI coding assistance produced implementation, experiments, figures and report drafts; the Lead reviewed the combined changes and ran numerical and regression checks. Account attribution in repository operations does not imply independent human authorship, as disclosed in the development provenance. Figures and browser captures were checked by hashes, OCR, layout geometry, browser errors and pixel tests. No human visual inspection is claimed. Numeric bindings help detect stale evidence but do not prove prose conclusions.

<!--pagebreak-->

## Appendix A Historical five-family cross-validation (provenance only)

Before the phase instruction narrowed the scope, a five-family cross-validation was run on the same folds. Its numbers are retained for provenance. KNN and Bagging were measured in the requested-trio phase; the identical valid Logistic Regression fit and CV are reused with explicit provenance.

| Model | Historical CV macro F1 |
|---|---|
| Logistic regression | 0.9328 |
| Gaussian naive Bayes | 0.9208 |
| Random forest | 0.9425 |
| Extra trees | 0.9427 |
| Hist gradient boosting | 0.9427 |
Table: Historical five-family cross-validated macro F1, from the pre-phase run (NumPy 2.5.2 era). Historical families are not substitutes for the current requested trio. Logistic Regression is the explicitly reused common baseline.

The geometry-only HMM baseline (no learned classifier, historical default configuration) reached macro length F1 0.9734, macro edge F1 0.9262, and macro edit similarity 0.8747 on the student partition in that era. It is recorded here because the decoder ablation of section 5.3 reuses its default configuration; it is not evidence about the learned final model of this phase.

One earlier delivery of this repository, frozen before the phase instruction narrowed the scope to the current trio, delivered macro length F1 0.9794 and macro edge F1 0.9514 on the same ten-trip student partition through its own final CLI. That length F1 is +0.0005 against this phase's delivered 0.9789: slightly higher, from a model outside the requested trio and a different freeze. The earlier result is included so the report does not imply a numerical improvement over every historical model. It belongs to a different freeze outside the requested trio and was not redeployed.
