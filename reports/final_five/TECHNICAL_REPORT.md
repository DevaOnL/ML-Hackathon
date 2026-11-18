# 1. Summary and scope

PathSnap matches an ordered GPS track to a directed road graph. A supervised binary classifier scores each nearby candidate arc; the scores become observation-level emission potentials for a graph-constrained HMM/Viterbi decoder. The delivery includes the learned model, command-line predictions, a two-section web interface and evidence that the API returns the same routes and observation assignments as the offline run.

The final comparison covers exactly five families: ["logistic_regression", "knn_distance_weighted", "bagging", "gaussian_nb", "hist_gradient_boosting"]. The selected family is hist_gradient_boosting; the delivered configuration is hist_gradient_boosting_tuned, which is a tuned configuration of that family, not a sixth family. The corrected model-comparison stage completed at 2026-11-18T08:26:57.159973+00:00. The default-to-tuned TRAIN selection objective moved from 0.8927 to 0.8934, a change of +0.0007. Selection used whole-trip TRAIN cross-validation, not student or evaluator scores.

After freezing model and decoder, the student-development CLI achieved trip-macro directed-arc F1 0.9514 and length-weighted F1 0.9794. The later evaluator pass achieved 0.9226 and 0.9658, respectively, over 10 trips. All reported route scores use the complete supplied reference arc sequence. Observation accuracy and observation macro F1 use route-derived proxy labels, not independently supplied GPS-to-road truth.

Two limits affect interpretation. The ten student trips were historically used as development data, so this report does not call them globally blind. The evaluator audit found no matching IDs or exact track/network files against the checked TRAIN and student partitions; that finding does not prove geographical independence or official unseen status. Evaluator exact-route matches were 0; weak trip 15 is retained in every aggregate.

# 2. Dataset and exploratory audit

## 2.1 File meaning and partition roles

The original student archive contains 90 trips and 360 files, with four files per trip. A track row is longitude, latitude and timestamp in seconds. Node rows are longitude and latitude. Arc rows contain zero-based tail/head node IDs, and an arc's row index is its original directed ID. A route is an ordered sequence of those IDs; its length need not equal the GPS row count. Parallel arcs and repeated traversals remain valid data.

The supplied split is 80 TRAIN trips and ten student-development trips. The full structural audit read every file and checked numerical formats, byte hashes, ranges, endpoints, timestamp order and cross-file consistency. The TRAIN networks contain 3,875,129 unique directed node pairs summed over their trip-local graphs; this is not a count for one shared global graph. The audit's pooled observation distributions and its equally weighted per-trip distributions answer different questions and are labelled separately in the figures. See [audit summary](<analysis/data_audit/summary.json>) and [per-trip audit](<analysis/data_audit/per_trip.json>).

The student role is recorded as “historically used student-development set; not globally blind”. Those trips were visible in earlier development and earlier model/decoder work. For the final phase, family selection, HPO and the fixed decoder search read TRAIN objectives only; final student results were not fed back to revise the frozen settings. This narrower boundary is supportable. A claim that student data never influenced any prior design decision would not be.

## 2.2 Structural findings and what was preserved

The audit found 13 broken consecutive-arc joins in TRAIN reference routes and 1 in student reference routes. These are discontinuities in supplied reference walks, not discontinuous GPS observations. Student trip 00000076, not TRAIN, has 1 reference break; the decoder evidence locates it at join position 12. No claim is made that it was removed from training: it was never a TRAIN trip. References were not repaired to improve a score.

TRAIN also contains 18 steps above 100 m/s if the timestamp unit is seconds, 7,289 intervals differing from one second, 89 extra nodes sharing coordinates and 572 duplicate-coordinate track rows. The audit found 0 nonpositive time differences and 67 reference arc occurrences beyond first visits. Fast steps, gaps and repeated observations are disclosed rather than silently discarded. A disconnected full network does not itself invalidate a trip: the route must be connected within the relevant directed component.

Distances are computed in metres using a per-graph local equirectangular projection centred on the graph's mean longitude/latitude. Longitude scale includes the cosine of the origin latitude. This is an approximation for local networks, not a universal planar CRS or a claim of geodesic accuracy over arbitrary extents. The implementation validates finite geographic coordinates and original arc endpoints before projecting; it does not use Euclidean degrees as metres. See [geometry/parsers](<mapmatch/data.py#L13-L74>).

## 2.3 Exploratory figures

These twelve figures are existing measured EDA assets, reused read-only and hash-bound by [EDA figure manifest](<visualizations/figure_manifest.json>). They cover inventory, missing/invalid values, trip sizes, timestamp gaps, connectivity, geographic coverage and all supplied train/student trip galleries. A reference overlay is exploratory context, not proof of per-observation labels.

![Dataset inventory: actual trip, file and observation counts](visualizations/plots/dataset_inventory.png)

![Missing values, invalid values and recorded anomalies](visualizations/plots/missing_invalid_and_anomalies.png)

![Trajectory size and duration distributions](visualizations/plots/trajectory_distributions.png)

![Actual timestamp gaps rather than assumed uniform 1 Hz sampling](visualizations/plots/timestamp_gaps.png)

![Road-network sizes and directed connectivity](visualizations/plots/road_network_sizes_connectivity.png)

![Geographic coverage of the supplied student material](visualizations/plots/gps_geographic_coverage.png)

![TRAIN trip gallery, part 1](visualizations/plots/gps_trip_gallery_train_01.png)

![TRAIN trip gallery, part 2](visualizations/plots/gps_trip_gallery_train_02.png)

![TRAIN trip gallery, part 3](visualizations/plots/gps_trip_gallery_train_03.png)

![TRAIN trip gallery, part 4](visualizations/plots/gps_trip_gallery_train_04.png)

![Student-development trip gallery](visualizations/plots/gps_trip_gallery_student_test_01.png)

![Representative directed road network and supplied reference route](visualizations/plots/representative_road_route.png)

# 3. Candidate features and target construction

## 3.1 Graph/GPS-only inference features

The frozen schema has 38 ordered numerical features, version pathsnap-candidate-v1. It combines lateral distance and transformed/ranked distance, candidate-set context, projection fraction and endpoint proximity, arc length, signed heading alignment at several windows, motion reliability, neighbouring lateral distances, local speed/turn/gap descriptors, endpoint degree and reverse-arc availability. Exact names and definitions are persisted in [feature schema](<artifacts/final_five/feature_schema.json>) and implemented in [feature extractor](<mapmatch/features.py#L17-L180>).

Candidates start within an 80 m radius with width 12. Radius expansion is triggered by an empty candidate set and capped at 1000 m. A target miss is different: candidates can exist while none represents the reference-derived proxy edge. Candidate construction and feature extraction accept graph and track only; references cannot enter them. Nearest-neighbour scaling for LR/KNN and GaussianNB's standardization are fitted inside each training fold, not on held-out trips. Tree models use the deterministic numerical features without learned scaling.

Several features deliberately use centred windows, up to eight rows on either side, and a forward-neighbour lateral distance. They are valid for the delivered whole-track offline/request-response task. They are not causal streaming features. A real-time point-by-point deployment would require a separately declared latency window or a new causal feature policy and validation; this report claims neither.

## 3.2 Proxy labels: the exact implemented rule

No per-GPS road labels were supplied. For supervised targets and postprediction observation scoring, the code searches every distinct canonical directed segment occurring anywhere in the supplied reference route. For observation t and candidate reference segment e, it minimizes the following score:


offset_score(t,e) = lateral_distance(t,e)^2 + 32 m² * reliability(t) * [1 - signed_heading_alignment(t,e)]

Here reliability is min(motion-window displacement / 12 m, 1). This label window uses at most three neighbouring rows each way and stops at local gaps above 20 s. A winning-to-runner-up score margin below 4 m² is marked ambiguous. The 32 m² coefficient and 4 m² ambiguity rule are different quantities; neither is a measured claim of true GPS assignment accuracy. The exact policy is [reference_observation_labels](<mapmatch/features.py#L191-L234>).

The label search is independent of the generated candidate set. If the selected canonical reference edge is absent from that set, the observation remains in the top-1 denominator and counts wrong. Repeated route traversals are collapsed only for the observation label's geometric search; route metrics retain their multiplicity. Intersections, opposite directions, stops, nearby parallel roads, repeated visits and reference defects remain sources of proxy uncertainty. Geometry-derived targets can favour geometric predictors.

# 4. Experimental protocol and metric definitions

## 4.1 Shared populations and honest reuse

The saved experiment protocol uses all 80 TRAIN trips, with 350 deterministic observations per trip: 28,000 observations and 269,463 candidate rows. The 3-fold GroupKFold split holds out whole trips; no trip contributes candidate rows to both training and validation in a fold. Weights give equal total mass to an observation and inverse-count correction across trips. KNN cannot take sample weights, so its declared 60,000-row class-stratified reference bank samples within each class without replacement in proportion to those weights. Its scaler is fitted only on that fit's bank.

The primary objective is equal-trip mean out-of-fold observation top-1 proxy accuracy. The legacy objective text says “maximize out-of-fold equal-trip mean observation top1 proxy accuracy; fixed 3/5-fold GroupKFold on all 80 TRAIN trips”; the actual delivered records use three folds. All student observations are retained: 19,160 observations and 175,778 candidate rows. There are 578 TRAIN target misses (2.0643%; 27,422 present targets) and 19 student target misses (0.0992%; 19,141 present targets). These counts are not empty-set counts.

The LR, KNN and default Bagging fits/CV were reused from verified prior experiments with the same features, sampling, folds and parameters. GaussianNB, default HistGB and tuned HistGB were freshly fitted in the corrected final run; the acceptance records ["gaussian_nb", "hist_gradient_boosting", "hist_gradient_boosting_tuned"] fresh fits. The completed HistGB HPO study was reused rather than rerun. Reuse avoids misrepresenting old numerical work as new computation. The final selected model was refitted on the all-80 training sample, not on student or evaluator data. See [frozen protocol](<experiments/final_five/PROTOCOL.md>) and [corrected measured results](<experiments/final_five/current/results.json>).

## 4.2 Three different metric populations

1. Binary candidate metrics treat each candidate row as positive/negative at threshold 0.5. The two-class confusion matrices, binary macro F1, positive-class precision/recall and binary log loss all use these rows. Many candidate negatives make binary accuracy unsuitable as the sole headline.
2. Observation metrics choose one road per GPS observation. Top-1 is correct assignments divided by all observations, including absent-target misses. Canonical directed road classes are namespaced by trip so unrelated graph-local IDs are not merged. Pooled edge macro F1 and an equal-trip mean of per-trip edge F1 are separate quantities. Equal-trip top-1 and observation-pooled top-1 also differ on unequal trip lengths.
3. Route metrics compare the complete reconstructed route to the complete supplied reference walk. Directed-arc multiset overlap counts repeated occurrences, length overlap weights those occurrences by projected segment length, and edit similarity measures sequence order. These are not candidate-conditioned route scores and are not the binary candidate F1.

For an observation-level categorical emission log score, positive binary probabilities are clipped, then normalized within that observation's candidates. The unconditional NLL includes missing targets using a 1e-7 floor; the conditional NLL drops only absent-target observations and must be labelled “given candidate”. Neither is an HMM posterior log loss or a general calibration guarantee. The decoder's relative path cost includes geometric and transition terms and is a different object again. See [candidate metrics](<mapmatch/learning.py#L126-L208>), [sequence recount](<experiments/final_five/current/sequence_classification.json>) and [route scorer](<mapmatch/metrics.py>).

## 4.3 Mandatory nine-column comparison

This table preserves the requested columns. CV Macro F1 and Student Macro F1 are binary candidate-row measures. Top-1 Accuracy is equal-trip student observation accuracy; the selection objective itself is TRAIN CV top-1, reported immediately below. Log Loss is student binary candidate log loss. Inference Time is the measured estimator probability call over the fixed student candidate matrix, in seconds, excluding candidate construction, features, routing, scoring, HTTP and browser rendering. Reused rows retain their original measured timings; these are not repeated hardware-normalized benchmarks. Route continuity is N/A here because the independent classifier does not reconstruct a route; the raw adjacency diagnostic and HMM route continuity are reported separately.

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route continuity | Student-Test Macro F1-score | Log Loss | Inference Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | scaled linear logistic | {"C": 1.0, "max_iter": 500, "solver": "lbfgs"} | 0.9330 | 0.9046 | N/A | 0.9329 | 0.0690 | 0.026 |
| Distance-weighted KNN | nonparametric neighbours | {"n_neighbors": 25} | 0.9275 | 0.9027 | N/A | 0.9275 | 0.0960 | 187.078 |
| Bagging | bootstrap tree ensemble | {"max_depth": 14, "max_samples": 0.7, "min_samples_leaf": 3, "n_estimators": 40} | 0.9426 | 0.9179 | N/A | 0.9437 | 0.0562 | 0.872 |
| GaussianNB | Gaussian generative Bayes | {"var_smoothing": 1e-07} | 0.9211 | 0.8982 | N/A | 0.9222 | 0.3591 | 0.105 |
| HistGB default | histogram boosted trees | {"l2_regularization": 0.5, "learning_rate": 0.09, "max_iter": 120, "max_leaf_nodes": 23, "min_samples_leaf": 40} | 0.9429 | 0.9186 | N/A | 0.9442 | 0.0552 | 0.577 |
| HistGB tuned | same boosted-tree family | {"l2_regularization": 0.8604220152998319, "learning_rate": 0.15902391896503631, "max_iter": 104, "max_leaf_nodes": 22, "min_samples_leaf": 38} | 0.9416 | 0.9196 | N/A | 0.9447 | 0.0555 | 0.499 |

The family-selection CV top-1 values are LR 0.8840, KNN 0.8768, Bagging 0.8915, GaussianNB 0.8734 and default HistGB 0.8927. Tuned HistGB reaches 0.8934. A sixth row represents a second configuration of HistGB only.

## 4.4 Observation and continuity diagnostics

The table below keeps pooled and trip-macro values explicit. “Raw continuity” means adjacent assignments are the same canonical edge or directly head-to-tail connected, without inserted routing connectors; it is not validation of a full route. The edge-change version excludes same-edge pairs, which otherwise dominate this statistic. It includes observed time gaps. The conditional NLL can look better by excluding misses, so the unconditional score is reported alongside it.

| Model | Pooled observation top-1 | Pooled edge macro F1 | Trip-macro edge F1 | Unconditional emission NLL | NLL given candidate | Raw adjacency continuity | Edge-change continuity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.8952 | 0.8071 | 0.8107 | 0.3476 | 0.3319 | 0.9287 | 0.7204 |
| knn_distance_weighted | 0.8905 | 0.7955 | 0.8045 | 0.5217 | 0.5062 | 0.9230 | 0.7081 |
| bagging | 0.9048 | 0.8271 | 0.8365 | 0.2824 | 0.2667 | 0.9322 | 0.7203 |
| gaussian_nb | 0.8884 | 0.7933 | 0.7935 | 0.9260 | 0.9109 | 0.9229 | 0.7093 |
| hist_gradient_boosting | 0.9066 | 0.8306 | 0.8387 | 0.2791 | 0.2633 | 0.9335 | 0.7247 |
| hist_gradient_boosting_tuned | 0.9068 | 0.8346 | 0.8433 | 0.2767 | 0.2609 | 0.9343 | 0.7275 |

The candidate-row accuracy, positive-class precision and positive-class recall below are not road-class macro precision/recall. They document the binary decision shown by the confusion matrices.

| Model | Binary row accuracy | Positive-class precision | Positive-class recall | Binary macro F1 |
| --- | --- | --- | --- | --- |
| logistic_regression | 0.9737 | 0.8713 | 0.8902 | 0.9329 |
| knn_distance_weighted | 0.9724 | 0.8895 | 0.8522 | 0.9275 |
| bagging | 0.9783 | 0.9084 | 0.8910 | 0.9437 |
| gaussian_nb | 0.9693 | 0.8465 | 0.8774 | 0.9222 |
| hist_gradient_boosting | 0.9783 | 0.8979 | 0.9034 | 0.9442 |
| hist_gradient_boosting_tuned | 0.9785 | 0.8985 | 0.9046 | 0.9447 |

## 4.5 Fit diagnostics and their limits

| Model | TRAIN resubstitution top-1 | Grouped-CV top-1 | TRAIN binary macro F1 | TRAIN binary log loss | OOF binary log loss | All-80 fit time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.8843 | 0.8840 | 0.9348 | 0.0675 | 0.0744 | 1.021 |
| knn_distance_weighted | 0.9077 | 0.8768 | 0.9412 | 0.0723 | 0.0941 | 0.024 |
| bagging | 0.9406 | 0.8915 | 0.9716 | 0.0306 | 0.0572 | 103.756 |
| gaussian_nb | 0.8740 | 0.8734 | 0.9212 | 0.3370 | 0.3392 | 0.174 |
| hist_gradient_boosting | 0.9085 | 0.8927 | 0.9540 | 0.0438 | 0.0570 | 5.350 |
| hist_gradient_boosting_tuned | 0.9140 | 0.8934 | 0.9567 | 0.0409 | 0.0573 | 4.396 |

Resubstitution and held-out metrics are useful diagnostics, but their difference is not an isolated causal estimate of overfitting: fold training sets are smaller, observations are sampled for TRAIN, and student trips have another length/composition distribution. A student score exceeding a TRAIN score is not evidence that no overfitting exists. None of these estimator pipelines saved epoch-wise training/validation histories. Such curves are N/A for each family; measured fit/CV/HPO summaries are supplied instead.

# 5. The five individual model families

## 5.1 Logistic Regression

**Why and suitability.** A scaled linear classifier is a small, fast baseline for tabular distance, direction and topology evidence. It tests how much the engineered features explain without a nonlinear ensemble. The initial configuration is {"C": 1.0, "max_iter": 500, "solver": "lbfgs"}; L-BFGS minimizes regularized binary logistic loss, while selection evaluates the common grouped-CV observation objective. Fold-local StandardScaler prevents validation information entering the learned scaling.

**Strengths and limits.** The model is cheap to fit and score, but a linear log-odds surface cannot directly learn arbitrary distance-by-motion-by-topology interactions. Its all-80 fit took 1.021 s and its saved student probability call 0.026 s. Student binary precision/recall were 0.8713 / 0.8902, binary macro F1 0.9329, pooled observation accuracy 0.8952 and pooled road-class macro F1 0.8071.

**Observed fit behaviour and epoch applicability.** Resubstitution top-1 0.8843 is close to grouped-CV 0.8840; this is consistent with limited in-sample gain for this configuration. It does not establish universal underfitting or prove absence of bias. Epoch curves are N/A: the saved solver fit has a maximum iteration limit, not a logged epoch-by-epoch training/validation history. This final row is a verified reused fit.

## 5.2 Distance-weighted KNN

**Why and suitability.** KNN is a genuinely different distance-based family: standardized candidate features are compared by exact Euclidean neighbour search, and inverse-distance class votes form the probabilities. Initial parameters are {"n_neighbors": 25}; fixed mechanics are distance weights, Minkowski p=2 and brute-force neighbours. There is no optimized parametric training loss. Bank construction and k are the relevant model-capacity choices; k was not changed from these final baseline results.

**Strengths and limits.** Local examples can express nonlinear decisions without fitting trees. Memory and query cost grow with the reference bank, and a 38-dimensional distance can be a poor summary when features are correlated or irrelevant. The declared bank preserves class proportions and approximates observation/trip weights through sampling, rather than claiming exact sample-weight support. Fit time 0.024 s mainly reflects storing the bank; the student probability call took 187.078 s, and summed CV query time was 299.1 s. These must not be described as end-to-end service latency.

**Test metrics and fit behaviour.** Student binary precision/recall 0.8895 / 0.8522, binary macro F1 0.9275, pooled observation accuracy 0.8905 and pooled edge macro F1 0.7955 trail the stronger tree configurations. Resubstitution top-1 0.9077 exceeds grouped-CV 0.8768; locality and self-neighbour effects can contribute, so this is reported as an in-sample advantage, not diagnosed from student data alone. Epoch curves are N/A because KNN stores a bank and has no epoch training process. The fit is verified historical reuse.

## 5.3 Bagging

**Why and suitability.** This is genuine bootstrap aggregation of DecisionTreeClassifier base learners, not an alias for RandomForest or HistGB. The initial configuration is {"max_depth": 14, "max_samples": 0.7, "min_samples_leaf": 3, "n_estimators": 40}; bootstrap samples use 70% of rows, trees are depth-limited and leaves require a minimum sample count. Each tree greedily reduces Gini impurity; averaged binary probabilities are evaluated by the same CV objective and log-loss diagnostics. There is no differentiable ensemble epoch loss.

**Strengths and limits.** Bagging captures nonlinear interactions and reduces the variance of an individual tree. Fitting multiple trees costs more and does not by itself constrain road-sequence connectivity. All-80 fit time was 103.756 s, summed CV training 202.3 s, and student probability inference 0.872 s. Student binary precision/recall 0.9084 / 0.8910, macro F1 0.9437, pooled observation accuracy 0.9048 and pooled edge macro F1 0.8271 make it a competitive baseline.

**Observed fit behaviour and epoch applicability.** Resubstitution top-1 0.9406 versus grouped-CV 0.8915 shows a larger in-sample advantage than LR. Depth/leaf limits constrain capacity but do not remove that gap. It is not justified to conclude either guaranteed generalization or complete overfit from these aggregates. Epoch curves are N/A; no per-tree training/validation curve was saved. The final default fit is reused; the separately tuned historical Bagging model appears only in Appendix A.

## 5.4 Gaussian Naive Bayes

**Why and suitability.** GaussianNB supplies a fast generative comparator. It estimates class priors and per-feature Gaussian means/variances, using conditional independence to form a class likelihood; it does not optimize logistic loss. Initial configuration {"var_smoothing": 1e-07} specifies numerical variance smoothing. Its scaling is learned inside each fit. The speed and simple distributional assumptions make it a useful counterpoint to nearest neighbours and nonlinear ensembles.

**Strengths and limits.** Many distance, motion and rank features are deliberately related. Conditional independence is therefore a restrictive approximation and its probabilities can be overconfident. The observed student binary log loss 0.3591 is substantially worse than the tree alternatives even though pooled observation accuracy is 0.8884. Student precision/recall are 0.8465 / 0.8774, binary macro F1 0.9222 and pooled edge macro F1 0.7933. Fit and probability inference took 0.174 s and 0.105 s.

**Numerical correction and fit behaviour.** The preserved pre-correction float32 diagnostic measured maximum probability-row-sum error 1.52e-05. Its 0 out-of-bounds entries and 0 affected rows refer to bounds violations, not to row-sum violations. That first attempt stopped before a final artifact was saved. Promoting only GaussianNB scoring input to float64 fixed the observed sum problem without relaxing the probability guard; the corrected current run is the reported evidence. Resubstitution top-1 0.8740 and CV 0.8734 are close, but weak held-out performance can reflect model assumptions as well as data noise. Epoch curves are N/A because this is a closed-form statistical fit, not epoch training. See [diagnostic](<reports/final_five/gnb_probability_diagnostic.json>) and [scoring implementation](<mapmatch/learning.py#L28-L45>).

## 5.5 Histogram Gradient Boosting

**Why and suitability.** HistGradientBoosting models nonlinear tabular interactions with histogram-binned decision trees, adding trees to reduce binary log loss. It can capture combinations such as distance evidence that becomes less reliable at a turn or after a gap. The default configuration is {"l2_regularization": 0.5, "learning_rate": 0.09, "max_iter": 120, "max_leaf_nodes": 23, "min_samples_leaf": 40}; the delivered tuned configuration is {"l2_regularization": 0.8604220152998319, "learning_rate": 0.15902391896503631, "max_iter": 104, "max_leaf_nodes": 22, "min_samples_leaf": 38}. Early stopping is disabled so there is no hidden row-random validation split inside a grouped-CV fit.

**Strengths and limits.** The family gave the best untuned TRAIN trip-macro top-1 objective among the five. It is much cheaper to query than the measured KNN baseline, but learning emission scores still cannot guarantee connected roads; the HMM supplies the graph constraint. Default/tuned all-80 fits took 5.350 / 4.396 s, and student probability inference 0.577 / 0.499 s. Tuned student binary precision/recall are 0.8985 / 0.9046, binary macro F1 0.9447, pooled observation accuracy 0.9068 and pooled edge macro F1 0.8346.

**Observed fit behaviour and epoch applicability.** Default resubstitution top-1 0.9085 exceeds CV 0.8927; tuned resubstitution 0.9140 exceeds CV 0.8934. The tuned gap is larger, so no claim of reduced overfitting follows merely from tuning. Trees are boosting iterations, not recorded neural-network epochs; epoch-wise train/validation curves are N/A. The current default and tuned fits are fresh. Their parameter study is historical reuse, explained next.

# 6. Model selection, tuning and measured plots

## 6.1 Selection rule and retained Bayesian study

The primary TRAIN criterion selects the default HistGB family. The retained study used GaussianProcessRegressor Matern-5/2 surrogate + analytic expected improvement, with 12 measured trials including the prior baseline, 3 seeded initial-design trials and 8 model-based proposals. The current final phase executed 0 fresh HPO trials. The historical study duration was 162.6 s; it is not a new final-run runtime. Its reuse note reads: “Actual completed historical TRAIN-only GP-EI study; not newly executed trials. Fresh default CV reproduces the prior objective.”. Fresh default/tuned CV and all-80 refits validate the selected configuration, rather than claiming the study itself was rerun.

The search space was {"l2_regularization": ["log", 0.01, 20.0], "learning_rate": ["log", 0.025, 0.2], "max_iter": ["int", 70, 200], "max_leaf_nodes": ["int", 7, 47], "min_samples_leaf": ["logint", 15, 160]}. The selected parameters are {"l2_regularization": 0.8604220152998319, "learning_rate": 0.15902391896503631, "max_iter": 104, "max_leaf_nodes": 22, "min_samples_leaf": 38}. The historical baseline objective 0.8927 became 0.8934, gain +0.0007. Equal-trip fold means/stds are default 0.8922 / 0.0281 and tuned 0.8929 / 0.0290; summed tuned CV training was 8.9 s.

The improvement is small and objective-specific. Tuned CV binary macro F1 0.9416 is lower than default 0.9429; tuned CV binary log loss 0.0573 is worse than default 0.0570. Student binary log loss also moves from 0.0552 to 0.0555, despite a student equal-trip top-1 change +0.0010. The tuned configuration did not win every metric. This tradeoff is accepted because the selection objective was declared in advance. Twelve trials are too few, and their acquisition is too adaptive, to treat scatterplot associations as causal hyperparameter importance.

## 6.2 Comparison figures and confusion matrices

All plots below are actual saved measurements, bound by [final figure manifest](<visualizations/final_five/figure_manifest.json>). No smooth learning curve has been invented. Confusion matrices have classes 0/1 for candidate membership at threshold 0.5; they are not confusion matrices of road IDs or one row per GPS observation.

![Primary TRAIN whole-trip CV selection objective](visualizations/final_five/plots/final_five_primary_objective.png)

![Binary candidate CV macro F1 by configuration](visualizations/final_five/plots/final_five_cv_macro_f1.png)

![Binary candidate log loss, OOF and student](visualizations/final_five/plots/final_five_binary_log_loss.png)

![Resubstitution, grouped-CV and student loss partitions](visualizations/final_five/plots/final_five_log_loss_partitions.png)

![Student observation top-1 and binary candidate macro F1: separate metrics](visualizations/final_five/plots/final_five_student_top1_macro_f1.png)

![Selection comparison under the declared objective](visualizations/final_five/plots/final_five_selection_comparison.png)

![Logistic Regression binary candidate confusion matrix](visualizations/final_five/plots/final_five_confusion_logistic_regression.png)

![KNN binary candidate confusion matrix](visualizations/final_five/plots/final_five_confusion_knn_distance_weighted.png)

![Bagging binary candidate confusion matrix](visualizations/final_five/plots/final_five_confusion_bagging.png)

![GaussianNB binary candidate confusion matrix](visualizations/final_five/plots/final_five_confusion_gaussian_nb.png)

![Default HistGB binary candidate confusion matrix](visualizations/final_five/plots/final_five_confusion_hist_gradient_boosting.png)

![Tuned HistGB binary candidate confusion matrix](visualizations/final_five/plots/final_five_confusion_hist_gradient_boosting_tuned.png)

![All six binary candidate confusion matrices](visualizations/final_five/plots/final_five_confusion_matrices_all.png)

![Measured fold stability; descriptive variation, not a confidence guarantee](visualizations/final_five/plots/final_five_fold_stability.png)

![Resubstitution and held-out differences with distinct population labels](visualizations/final_five/plots/final_five_generalization_gap.png)

![Actual training and candidate-matrix inference timings](visualizations/final_five/plots/final_five_timing.png)

![Retained twelve-trial HPO objective history, not new trials](visualizations/final_five/plots/final_five_hpo_progress.png)

![Observed HPO parameter associations, not causal importance](visualizations/final_five/plots/final_five_hpo_parameters.png)

![Tuned versus default HistGB tradeoffs](visualizations/final_five/plots/final_five_tuned_vs_untuned.png)

![Observation edge metrics, separate from candidate binary F1](visualizations/final_five/plots/final_five_observation_edge_metrics.png)

![Raw assignment adjacency diagnostics, not reconstructed-route continuity](visualizations/final_five/plots/final_five_continuity.png)

![Candidate, observation and trip denominators](visualizations/final_five/plots/final_five_denominators.png)

# 7. Frozen HMM decoder and all-ten ablation

## 7.1 Emission and transition costs

For candidate c at observation t, the delivered decoder minimizes a sum of emission and transition costs. Binary positive probabilities are clipped to [1e-7, 1-1e-7] and normalized within the candidate set to q_t(c). They are emission potentials, not normalized HMM posterior probabilities.

emission_cost(t,c) = geometry_weight * 0.5 * [lateral_distance(t,c) / sigma]^2 - classifier_weight * log(max(q_t(c), 1e-12))

transition_cost(a,b) = abs(directed_network_distance(a,b) - GPS_chord_distance) / beta

The directed network distance includes the remaining fraction of the first arc, a directed graph path and the initial fraction of the next arc. On the same arc, a projection moving no further back than the configured tolerance can be treated as forward progress/jitter; that tolerance does not authorize a disconnected jump. Viterbi chooses the minimum-cost sequence with backpointers. Shortest-path reconstruction inserts actual graph arcs between selected states and preserves real cycles and repeated traversals. If reachability still fails after bounded search expansion and the fixed candidate-width retries, inference raises an error rather than fabricating teleport edges. See [matcher](<mapmatch/matcher.py#L98-L251>).

Sigma controls the spatial emission penalty and beta controls discrepancy between network travel and GPS chord, both in metres. They are not learning rates, class thresholds or speed limits. Historical TRAIN-only residual/transition fitting was reused because it is geometric rather than classifier-specific; it used TRAIN references, so “reference-free calibration” would be false. The final model already fitted all 80 TRAIN trips before the 8-trip decoder search. This is in-sample calibration, not held-out route generalization.

## 7.2 Preregistered configurations and freeze

The fixed search compared A fitted scales, B twice beta, and C classifier weight 2 with geometric weight 0.5 on the preregistered TRAIN8 set. Its objective was mean exact-original-ID length-weighted repeated-arc route F1, with a failed trip contributing zero. Scores were A length/edge 0.9533 / 0.9382, B 0.9521 / 0.9383, and C 0.9523 / 0.9408. It selected A_fitted and froze at 2026-11-18T08:37:21.093991+00:00, before the final student ablation and evaluator access.

The delivered settings are sigma 7.161 m, beta 2.000 m, classifier weight 1.00 and geometric weight 1.00. Initial radius is 80 m, routing slack 300 m, backward tolerance 20 m and graph-index sample spacing 40 m. Width begins at 12 and can widen 12 -> 24 -> 48 only after genuine reachability failure. Observation distance is zero, so every original observation is decoded. Graph-index spacing does not mean GPS observations were downsampled. See [calibration plan](<reports/final_five/decoder/calibration_plan.json>) and [decoder freeze](<reports/final_five/decoder/freeze.json>).

## 7.3 Paired student-development ablation

The ablation contains 10 trips and all 19,160 observations, with 0 failed trips. Raw classifier argmax and learned HMM use the same final candidates/probabilities. The raw route comparator collapses adjacent identical IDs but does not insert shortest-path connectors; it can therefore be disconnected. Its F1 is still scored against the full reference route, including arcs not present in candidate sets. This is a deliberately weak pointwise route baseline, not an alternative connected decoder.

| Method | Trip-macro observation proxy accuracy | Complete-reference route edge F1 | Complete-reference length F1 |
| --- | --- | --- | --- |
| classifier_only_raw | 0.9200 | 0.7704 | 0.8227 |
| classifier_plus_HMM | 0.9389 | 0.9514 | 0.9794 |
| geometry_HMM_default_reused | N/A in this table | 0.9262 | 0.9734 |

The HMM raises route edge F1 by +0.1810 over this raw comparator. Mean measured HMM inference time is 4.39 s per trip. The geometry-only row is an actual historical all-ten run reused as context, not a freshly rerun third final model. Differences do not isolate every effect: the paired raw/learned-HMM comparison shares candidates and scores, while the historical geometry baseline can differ in configuration and candidate behaviour.

The separately executed CLI matches the ablation route hashes and reaches route edge F1 0.9514, length F1 0.9794, edit similarity 0.9172 and continuity 1.0000. The student reference break remains disclosed, while predictions are validated as connected directed walks. See [paired ablation](<reports/final_five/decoder/student_ablation.json>) and [actual CLI verification](<reports/final_five/offline_delivery_verification.json>).

![Actual three-method all-ten route ablation](visualizations/final_five/plots/final_five_route_ablation.png)

# 8. Postfreeze evaluator pass

## 8.1 Access chronology and overlap audit

The model-stage record said “not accessed; final decoder must freeze before evaluator work”; that was its status at model completion, not a statement that the completed project never evaluated the new inputs. The model froze at 2026-11-18T08:26:50.372940+00:00 and the decoder at 2026-11-18T08:37:21.093991+00:00. The preregistered evaluator's first graph/track file open began 2026-11-18T08:52:25.539066+00:00. One actual all-ten CLI completed at 2026-11-18T08:53:12.792273+00:00; reference scoring began only at 2026-11-18T08:53:38.239789+00:00, after full prediction verification. No fit, feature revision, HPO, decoder adjustment or prediction retry followed those scores. See [evaluator protocol](<docs/FINAL_EVALUATOR_PROTOCOL.md>) and [chronology/acceptance](<reports/final_five/evaluator/acceptance.json>).

The audit checked exact track, nodes and arcs hashes against all 80 frozen TRAIN trips and 10 student trips. Same-ID matches were []; any identical input file was False and any full-trip duplicate False. Matching IDs alone would not prove duplicate data, and unmatched checksums do not rule out geographic overlap, transformed copies or other prior exposure. The dataset is described here as the supplied postfreeze evaluator set, not an independently certified unseen official benchmark.

## 8.2 Results, proxy denominators and weak cases

All 19,674 original observations were predicted and proxy-scored across 10 trips. Macro route edge F1 is 0.9226, length F1 0.9658, edit similarity 0.8793 and route continuity 1.0000. References have 0 broken joins. Exact routes: 0. Trip 15's edge F1 0.5788 and length F1 0.7751 are materially weaker than the aggregate; neither the trip nor its errors were removed.

Trip-macro classifier top-1 is 0.9378; trip-macro decoded road-class F1 is 0.8743. Over all observations, pooled decoded accuracy/F1 are 0.9305 / 0.8884, versus raw classifier 0.9328 / 0.8449. The pooled classifier accuracy is slightly higher than decoded accuracy while decoded macro F1 is higher. The report retains that tradeoff and does not claim the HMM improves every metric. All observation measures are the same geometry-and-motion proxies described earlier, not independently labelled GPS truth.

The offline CLI saved classifier IDs and matched states but not complete candidate probability arrays or memberships. Offline evaluator candidate NLL and miss rate are therefore N/A with an explicit reason, and no new inference was run to fill them. The later live/API audit scores probabilities it actually records during fresh API requests; those results must be labelled as API measurements rather than retroactively inserted into the offline pass. HMM posterior NLL remains unavailable. Missing metrics are not replaced with zero.

The following per-trip categorical NLL and candidate miss rates come from the actual twenty-trip ASGI API integration pass, after offline evaluation. Its request-local audit records every candidate probability used by the decoder, and the scorer runs after inference. These are current measured API values, not missing offline CLI arrays and not binary candidate log losses. The ASGI test exercises the real estimator without sockets; the separately recorded HTTPS/browser tests supply public-network proof. Source: [actual API integration](<reports/final_five/api_integration_acceptance.json>) and [postprediction scoring](<backend/scoring.py#L99-L118>).

| Evaluator trip | Actual API emission categorical NLL | Actual API candidate-miss fraction |
| --- | --- | --- |
| 00000015 | 0.358403 | 0.000000 |
| 00000017 | 0.343905 | 0.007429 |
| 00000023 | 0.244751 | 0.004367 |
| 00000030 | 0.571056 | 0.004587 |
| 00000049 | 0.036796 | 0.000000 |
| 00000059 | 0.180535 | 0.001940 |
| 00000067 | 0.071287 | 0.000000 |
| 00000073 | 0.069809 | 0.000000 |
| 00000087 | 0.121270 | 0.000000 |
| 00000094 | 0.153870 | 0.000000 |

The evaluator passed its 190 invariant checks and 9 synthetic guard tests. The ZIP contains ten flat predicted route files only, SHA-256 20a62accc83d0d5221339093f4f07a34efee04018a14cb464d66ed2fce1b655b. Private references, raw coordinates, candidate arrays and observation CSVs are excluded from public evaluator reports. See [all per-trip evaluator metrics](<reports/final_five/evaluator/route_metrics.json>), [input audit](<reports/final_five/evaluator/input_audit.json>) and [privacy/test verification](<reports/final_five/evaluator/test_and_privacy_verification.json>).

# 9. API, website and genuine browser evidence

## 9.1 Delivered system

The user-authorized delivery is the public HTTPS site https://ml.d3va.dev, reusing the dedicated API service and existing tunnel. It replaces the requested physical two-laptop arrangement for this user; it does not prove a venue demonstration occurred. Development and evaluator sections have separate allowlisted trip registries, with the same frozen model/configuration and inference path. Requests identify dataset and trip, not client-supplied filesystem paths, model files or decoder overrides. References are accessed only by the scoring phase after prediction and are not sent as reference geometry in match responses.

The measured HTTPS evidence has 20 successful trip matches, with all twenty requested flag True. The live health response reports frozen model SHA c9cc8e9f36838f656a94d07ed69632979df9981ca94a9daf3b52fbce052e4b34. API integration verifies route, predicted geometry and matched-state hashes against offline predictions for ten development and ten evaluator trips. A model hash alone would not prove route parity, so the array and geometry comparisons matter. [API integration evidence](<reports/final_five/api_integration_acceptance.json>), [consolidated parity](<reports/final_five/delivery_verification.json>) and [live API evidence](<reports/final_five/live/live_api.json>) supply the checks.

The interface uses an original PathSnap dark/light design informed by the explicitly visited OpenCode reference after model acceptance. Local Leaflet assets avoid third-party basemap requests in the recorded test. The UI shows request/model identity, route and raw GPS layers, metric definitions, and separate inference/scoring timing. It preserves theme preference and handles dataset changes without presenting a stale result as another trip. The browser evidence is actual Chromium execution without mocked responses or request interception. Its browser-error count is 0 and console-error count 0; these are properties of the recorded sessions, not a promise that every browser/environment is error-free.

## 9.2 Desktop, themes, evaluator and mobile

The captures below show the deployed page, not fixture screens. The evidence tests dark/light themes, both datasets and widths 320, 390, 768 and 1440 with no horizontal overflow. The development browser request ID is e5c74616aa5f40f7ac9acbf2459d9678 and evaluator request ID cdfbc4474f3840ec8f271e9e79e90635. The evaluator screenshot example uses trip 87 with proxy decoded accuracy 0.9907; it is one trip and is not the all-ten mean. Keyboard checks include dataset-tab navigation, modal Escape/focus return and theme persistence. Changing theme or reloading does not silently launch a new match. See [browser evidence](<reports/final_five/live/browser_evidence.json>).

![Actual deployed dark desktop view](reports/final_five/live/screenshots/ui_desktop.png)

![Actual raw GPS and predicted route map panel](reports/final_five/live/screenshots/ui_route_map.png)

![Actual deployed light theme](reports/final_five/live/screenshots/ui_light.png)

![Actual separate evaluator section](reports/final_five/live/screenshots/ui_evaluator.png)

![Actual narrow 320-pixel mobile view](reports/final_five/live/screenshots/ui_mobile_320.png)

![Actual 390-pixel mobile view](reports/final_five/live/screenshots/ui_mobile.png)

The full-page captures above preserve page context; the very tall mobile overviews are not intended for reading individual controls in print. The following enlarged details are lossless rectangular crops of those same screenshots. Their [crop manifest](<reports/final_five/report_images/manifest.json>) binds each original screenshot hash, exact coordinates, output hash and pixel-equality check. No content was redrawn, relabelled or recaptured.

![Student-development dataset control, original-pixel detail](reports/final_five/report_images/development_dataset_controls.png)

![Selected development trip and genuine run control, original-pixel detail](reports/final_five/report_images/development_selected_run.png)

![Development result scores, original-pixel detail](reports/final_five/report_images/development_scores.png)

![Development model and request identity, original-pixel detail](reports/final_five/report_images/development_execution.png)

![Separate evaluator dataset control, original-pixel detail](reports/final_five/report_images/evaluator_dataset_controls.png)

![Evaluator result scores, original-pixel detail](reports/final_five/report_images/evaluator_scores.png)

![Light-theme result scores, original-pixel detail](reports/final_five/report_images/light_scores.png)

![390-pixel mobile selected trip and run control, original-pixel detail](reports/final_five/report_images/mobile_390_selected_run.png)

![390-pixel mobile result cards, original-pixel detail](reports/final_five/report_images/mobile_390_scores.png)

![320-pixel mobile result cards, original-pixel detail](reports/final_five/report_images/mobile_320_scores.png)

## 9.3 Receiving-terminal correlation

The genuine Alacritty screenshot displays the dedicated service's received-request journal. The browser initiated request e5c74616aa5f40f7ac9acbf2459d9678 for the development section; the terminal did not originate the HTTP request. The log records match start 2026-11-18T09:06:25.487067+00:00, completion 2026-11-18T09:06:28.707597+00:00 and an HTTP event with the same ID e5c74616aa5f40f7ac9acbf2459d9678. In this allowlisted transport event, the model hash is absent; model/config identity comes from the same-ID match-start/completion events and browser response. It would be false to require every transport event to carry the model hash or to replace the real log with typed sample output.

The terminal record marks validity True and OCR privacy guard True. These are automated provenance/privacy checks, not claims that a human independently inspected the capture. See [terminal evidence](<reports/final_five/live/api_terminal_evidence.json>).

![Genuine receiving terminal with browser-correlated request events](reports/final_five/live/screenshots/api_terminal.png)

![Receiving-terminal request and full model/configuration hashes, original-pixel detail](reports/final_five/report_images/terminal_identity.png)

![Receiving-terminal completion and timings, original-pixel detail](reports/final_five/report_images/terminal_completion.png)

# 10. Actual final-model Folium map

The final explanatory map is a distinct artifact from the interactive application's privacy boundary. It uses the delivered tuned HistGB prediction for development trip 00000009, with 1049 GPS points, 444 predicted arc occurrences and 429 reference arc occurrences. Route edge/length F1 are 0.9691 / 0.9739. Folium version 1.0.0 generated the actual HTML; a browser rendered its PNG/PDF. The map is not an old Bagging route with a new label.

The map draws every predicted arc in CLI order, including repeated occurrences, using supplied endpoints. It does not invent missing road shape or splice route gaps for appearance. Network, GPS, prediction and supplied reference layers are labelled. The development reference geometry is intentionally present for this analysis-only artifact, so its manifest says not to serve it through the application. No evaluator reference geometry belongs in a public map. Tooltip time ranges pooled by arc ID are not reconstructed per-traversal timestamps.

Open the [interactive Folium map](<visualizations/final_five/maps/final_development_trip_00000009.html>); the [map metadata](<visualizations/final_five/maps/final_development_trip_00000009.metadata.json>) and [map manifest](<visualizations/final_five/map_manifest.json>) bind source/prediction/model/config hashes and browser rendering checks.

![Actual final tuned-HistGB development trip 9 Folium rendering](visualizations/final_five/maps/final_development_trip_00000009.png)

# 11. Reproducibility, requirements and remaining responsibilities

## 11.1 Frozen artifacts and clean replay

The selected model froze at 2026-11-18T08:26:50.372940+00:00, size 147,906 bytes, deployment-export SHA-256 c9cc8e9f36838f656a94d07ed69632979df9981ca94a9daf3b52fbce052e4b34. The separately recorded comparison artifact has a different byte hash, listed below; it must not be confused with the frozen deployment export. The metadata and feature-schema hashes are c354885e22158e9da14e0f8bc8e01d3e7c6fbe810e196192c6b59c7cbdbbf4c5 and 55f203a1e30ab6369ff937840c2bfa4cb180dc36172a2ca47047714c05fb8d9a. Its fit declaration is “TRAIN only; student and evaluator not read by this stage”, with seed 20261118, feature version pathsnap-candidate-v1, all-80 fit time 4.40 s and sampled TRAIN candidate probability inference 0.770 s. That latter timing is TRAIN inference, not student or API latency; the selected student probability call was 0.499 s.

Software at the final fit: Python 3.14.6, scikit-learn 1.8.0, NumPy 2.5.5, SciPy 1.18.0 and joblib 1.5.3. A fresh-process clean-load check on the fixed 2,000-row TRAIN fixture reproduced probabilities with maximum absolute difference 0 at tolerance 1e-12, using matching recorded package versions. This demonstrates that particular reload, not a guarantee across all platforms, dependencies, compiler builds or arbitrary future environments. Trusted local joblib loading is required because pickle-based artifacts can execute code.

| Compared configuration | Accepted artifact SHA-256 |
| --- | --- |
| logistic_regression | 29157349d52a993d33cae78ed244805a9f087e79b6dc914aaa3b4461c02a58c9 |
| knn_distance_weighted | 564e6f85f9190d0ff19e75967613e4d56dbe573aaf80ef8780ed8374f8537246 |
| bagging | 619bc3d267fec686d3ee0d212b3d13a4c8d9bd46e35893f3f570253d232822b6 |
| gaussian_nb | f79c03821ee50a9cdb9912031dd56df0f0e54379bb48ad63116315e5400b8579 |
| hist_gradient_boosting | 7974b1eacc2787ade3becd48b0d1079bd673be83f15183bdad0839a4fc71ce85 |
| hist_gradient_boosting_tuned comparison artifact (not the deployment export) | 47ea723d350567b6eeba827744e2c1a26c9baf915d322e29a590f45922ed8eeb |


An independent AI reviewer recorded 9 verified findings and no critical defects, including an actual 111/111 verification check run. This is evidence review, not an independent human viva or organizer approval. The final test record reports 421 total tests, 421 passed and 0 failures. Test provenance and actual counts are taken from [final test record](<reports/final_five/test_results.json>) rather than inferred from a successful command alone.

The [report builder](<scripts/build_final_report.py>) resolves this manuscript's registered fact tokens from cited measured evidence and should refuse a final release while required facts/assets/guards fail. Its manifest records the builder/template/input hashes. These bindings cover the current and historical metric tables; they do not mean every prose statement or every number in an image is automatically validated. Formatting or page count cannot turn missing evidence into a completed requirement.

## 11.2 Human/organizer exceptions and honest completion status

The current requirements matrix has 47 rows: 41 evidence-complete, 0 partial, 0 delegated/in progress, 0 not done, 0 user-deferred, 1 explicit user exceptions and 5 human/external blockers. The builder calls 47 rows administratively resolved under its terminal-status rule. That rule includes accepted exceptions and human/external blockers; “resolved” is not the same as “all organizer obligations performed.” The full source-located matrix follows in Appendix B.

AI agents implemented and reviewed substantial parts of this project under human direction. This disclosure does not establish organizer permission for AI assistance, cross-team independence, plagiarism clearance or human authorship. Administrative-account repository actions are not independent human review. Each participating member must be able to explain the data, model choices, proxy labels, selection, decoder, limitations and deployment in a viva; no evidence here proves that happened or requires an invented five-person team size. A physical two-laptop venue demonstration remains a human task if organizers require it despite the user-authorized website substitute. The before-lunch notice has no calendar date, so no evaluation date is inferred. See [requirements audit](<docs/FINAL_REQUIREMENTS.md>), [development provenance](<docs/DEVELOPMENT_PROVENANCE.md>) and [delivery guide](<docs/FINAL_DELIVERY_GUIDE.md>).

# 12. Conclusions

The selected HistGB configuration met the declared TRAIN whole-trip top-1 objective and was frozen before final decoder calibration/evaluator access. The delivered HMM turns independent candidate scores into connected directed routes and improves the student full-reference route comparator, while the evaluator retains a difficult trip, zero exact routes and an observation-accuracy tradeoff. These are measured outcomes, not a claim of solved map matching or independently verified GPS labels. CLI/API/browser identity checks bind the software demonstration to the frozen artifacts. Human viva, event policy and plagiarism decisions remain external.

# Appendix A. Historical experiments, kept separate from the final five

RandomForest and ExtraTrees were genuinely experimented models in the earlier broader comparison; tuned Bagging belongs to the earlier three-family phase. They are documented to preserve the experiment record, not counted as extra final families or substituted for the required five. Their original CV/student metrics and timings are reported without refits or claims that they were freshly tested now. Historical GaussianNB precision-era numbers are not substituted for corrected current GaussianNB.

RandomForest uses bootstrapped trees with feature subsampling; ExtraTrees adds randomized split thresholds and is not identical to Bagging. Tuned Bagging remains an ensemble of ordinary bootstrap decision trees. Each tree uses impurity-driven splits, with no epoch histories. This historical appendix is not a post-evaluator model-selection opportunity. The final scope was fixed to LR/KNN/Bagging/GaussianNB/HistGB before these final delivery results.

| Historical model | Exact saved configuration | CV binary macro F1 | CV trip-macro top-1 | Student pooled top-1 | Student trip-macro top-1 | Student binary macro F1 | Student binary log loss | Probability inference (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForest, historical | {"max_depth": 16, "max_features": 0.7, "min_samples_leaf": 4, "n_estimators": 100} | 0.9427 | 0.8914 | 0.9050 | 0.9181 | 0.9446 | 0.0548 | 0.400 |
| ExtraTrees, historical | {"max_depth": 22, "max_features": 0.8, "min_samples_leaf": 3, "n_estimators": 100} | 0.9429 | 0.8918 | 0.9040 | 0.9163 | 0.9422 | 0.0556 | 0.491 |
| Tuned Bagging, historical | {"max_depth": 13, "max_samples": 0.7160628448832399, "min_samples_leaf": 2, "n_estimators": 48} | 0.9426 | 0.8925 | 0.9034 | 0.9159 | 0.9436 | 0.0560 | 0.938 |

Historical all-80 fit times were RandomForest 292.539 s, ExtraTrees 62.150 s and tuned Bagging 133.064 s. They are workload- and machine-dependent saved timings. Their source evidence is [historical broader results](<experiments/results.json>) and [historical three-family results](<experiments/phase_three/results.json>), not the current all-five result file.

# Appendix B. Original requirement traceability

The matrix below retains requirement classifications, source document/paragraph locators, evidence and current status. Exceptions are visible rather than converted to fabricated technical completion.

| id | area | requirement | status |
| --- | --- | --- | --- |
| R-DS-01 | dataset | Verify trip organisation, file naming, row counts, data types and column meanings; do not assume every trip is identical or that a GPS row corresponds to a network row. | done_evidence_exists |
| R-DS-02 | dataset | 3-column whitespace .track (lon, lat, ts seconds, ~1 Hz), 2-column .nodes, integer from/to .arcs (row index = arc_id), 1-column .route of chronological ground-truth arc_ids. | done_evidence_exists |
| R-DS-03 | dataset | Identify trips per partition, observations per trip, file availability/sizes, types/columns, missing or invalid GPS, timestamp ordering and gaps, geographic extent, network nodes/segments, connectivity, isolated nodes/disconnected components, anomalous trips, reference-route availability and structure. | done_evidence_exists |
| R-DS-04 | dataset | Justify projected CRS or geodesic distance (degrees-Euclidean is not physical distance), check coordinate order and plausibility, account for noise/gaps/irregular timestamps. | done_evidence_exists |
| R-DS-05 | dataset | Do not assume supplied routes are independently verified physical truth; identify limitations, state assumptions, avoid overstating label certainty. | done_evidence_exists |
| R-DS-06 | dataset | 80 training trips for fitting/development/CV/optimisation; 10 student-testing trips for development-stage comparison; 10 evaluator-testing trips for final evaluation only. | done_evidence_exists |
| R-DS-07 | dataset | No trip split across partitions; evaluator data not used for training, feature selection, algorithm or hyperparameter choice; no manual alteration from known routes; no hardcoded answers; preprocessing fitted on permitted training data only; evaluator ground truth kept out of the pipeline. | done_evidence_exists |
| R-DS-08 | dataset | Dataset inventory, summary statistics, coordinate/pattern plots, interactive Folium maps, network size/connectivity tables, data-quality record. | done_evidence_exists |
| R-DS-09 | dataset | At the end of exploration each team should be able to explain how the raw dataset is organised, what each file represents, how components relate, quality concerns and assumptions. | human_external_blocker |
| R-MT-01 | models | Verbatim: 'Teams must train and evaluate at leastFIVE different models before selecting their final model.' Selection must be supported by experimental results, performance comparisons and technical reasoning; direct selection without meaningful experiments incurs deductions. | done_evidence_exists |
| R-MT-02 | models | Students are strongly encouraged to explore models from multiple algorithm families; the family table is examples and suggestions, not a fixed list. | done_evidence_exists |
| R-MT-03 | models | Train >=5 distinct models; record each configuration; fit and cross-validate on the 80 training trips; compare on the ten student-testing trips; apply preprocessing consistently; avoid training-accuracy-only selection; record training and inference time where practical; retain logs, seeds, configurations and results. | done_evidence_exists |
| R-MT-04 | models | Report, where applicable: accuracy, precision, recall, F1, confusion matrix, log loss, observation-level Top-1 accuracy, candidate miss rate, route continuity, sequence-level performance, training and inference time; explain the primary selection metric and why it is appropriate. | done_evidence_exists |
| R-MT-05 | models | Comparison must identify model name, algorithm family, important configuration, training/CV performance, student-testing performance, precision, recall, F1, log loss where applicable, computational efficiency, route continuity. | done_evidence_exists |
| R-MT-06 | models | Selection must consider generalisation performance, stability across validation folds, probability quality, inference speed, model complexity, decoder compatibility; 'Do not select a model solely because it achieves high training accuracy.' | done_evidence_exists |
| R-MT-07 | models | After the initial comparison, run a Bayesian search on a promising model: justified search space, stated objective, group-aware CV on the 80 training trips, recorded trials, best configuration, refit, evaluation on the ten student-testing trips, tuned-vs-untuned comparison, reported computational cost. | done_evidence_exists |
| R-MT-08 | models | The evaluator-testing partition must not be used to choose the search space, objective, hyperparameters, model or thresholds. | done_evidence_exists |
| R-HM-01 | hmm | Route reconstruction should use the selected model's outputs consistently; sequential information plus network connectivity produce a coherent route. | done_evidence_exists |
| R-HM-02 | hmm | Describe hidden states; preserve candidate-to-segment correspondence; explain emission and transition modelling; explain the Viterbi recurrence; handle missing candidates and invalid transitions; handle GPS gaps; return a predicted segment per relevant observation; compare classifier-only with classifier-plus-HMM predictions (the negative control); report pointwise and sequence-level performance; analyse route continuity. | done_evidence_exists |
| R-HM-03 | hmm | Optimise decoder parameters using training and student-testing data only. | done_evidence_exists |
| R-SY-01 | system | Data-processing implementation, model-experimentation implementation, Bayesian-optimisation implementation, HMM/Viterbi implementation, API (FastAPI), UI, model and configuration artifacts, technical report, GitHub repository. | done_evidence_exists |
| R-SY-02 | system | Laptop 1 hosts the trained classifier, HMM/Viterbi, road-network information and server-side evaluation; Laptop 2 displays the ten evaluation-trip options, sends the selection to the API and shows the returned route and results. Final-evaluation steps 1-9 run API on Laptop 1 and UI on Laptop 2. | user_exception |
| R-SY-03 | system | The UI displays the ten reserved trips; evaluator selects any one; UI sends the trip ID; the API runs the complete prediction pipeline and computes accuracy from its internal reference data; UI displays route and results. Must work for all ten, not one demo example. | done_evidence_exists |
| R-SY-04 | system | Ground-truth labels must remain server-side and never be exposed to the prediction model or UI; hardcoded outputs and bypassing the API are not permitted; repositories will be examined. | done_evidence_exists |
| R-SY-05 | system | Students are required to show the evaluator the API logs on the laptop receiving the inference requests, on the endpoint. | done_evidence_exists |
| R-DL-01 | deliverables | Repository contains data processing, feature engineering, model training (every model experimented with), model evaluation, hyperparameter optimisation, HMM and Viterbi, API, UI, deployment configuration, visualisations (Folium map generation), and supporting results (CSVs, logs, configs, summaries). | done_evidence_exists |
| R-DL-02 | deliverables | All source committed (not only the final model); saved weights and configs subject to size constraints; README with installation, execution and deployment instructions; clearly named folders; accessible to evaluators; no secrets committed. | done_evidence_exists |
| R-DL-03 | deliverables | One consolidated PDF per team, following the required structure, with actual experimental results, tables, graphs and geographical visualisations; every reported result verifiable from the repository; no separate per-model reports. | done_evidence_exists |
| R-DL-04 | deliverables | Answer how the data was preprocessed and what patterns were observed; which features were extracted and why; include missing-value analysis, GPS trajectory plots, road-network visualisations, at least one Folium map; clearly mention which partition each analysis used. | done_evidence_exists |
| R-DL-05 | deliverables | Table columns exactly: Model, Algorithm Family, Key Configuration, CV Macro F1-score, Top-1 Accuracy, Route continuity, Student-Test Macro F1-score, Log Loss, Inference Time; actual results; N/A cells explained. | done_evidence_exists |
| R-DL-06 | deliverables | For every model: why the algorithm suits PathSnap; strengths and limitations; important hyperparameters and initial values; loss function or optimisation objective; student-testing performance; observation-level Top-1, Macro F1 and Log Loss where applicable; overfitting/underfitting with supporting evidence. No textbook-only answers. | done_evidence_exists |
| R-DL-07 | deliverables | CV Macro F1 comparison across all models; student-testing Top-1 and Macro F1 comparisons; Log Loss comparison where applicable; confusion matrices optional; training and validation loss plots only for models producing epoch-wise training histories; every graph from actual results; model-comparison plots justifying selection; no artificial training curves. | done_evidence_exists |
| R-DL-08 | deliverables | State hyperparameters searched, ranges or distributions, objective function, trial count, best values, tuned-vs-untuned comparison on CV Macro F1, student Top-1, student Macro F1, Log Loss where applicable, and inference time, using the same metrics and partitions for both. | done_evidence_exists |
| R-DL-09 | deliverables | Explain hidden states, emission and transition calculation, missing candidates and GPS gaps, disconnected-segment handling; whether HMM decoding improved the route, supported by experimental results; compare classifier-only vs classifier-plus-HMM; discuss cases where decoding did not help. | done_evidence_exists |
| R-DL-10 | deliverables | Mandatory Folium maps from the team's actual implementation showing GPS observations, road-network geometry, reference route where available, and predicted route; distinguishable colours/markers, labels or legend, appropriate extent and zoom, comparability, layer controls; at least one map screenshot/export in the PDF; interactive HTML or generator code in the repo; no unrelated or fabricated maps. | done_evidence_exists |
| R-DL-11 | deliverables | Screenshots of the working UI; screenshots of successful API responses on the CMD of the receiving laptop; submitted source must match the demonstrated implementation. | done_evidence_exists |
| R-DL-12 | deliverables | Reports and GitHub submissions face integrity checks: plagiarism checks on content, cross-team comparison of code/configs/weights/results, verification of reported metrics against actual outputs, technical questioning; the report must represent the team's actual work; every member must be able to explain it. | human_external_blocker |
| R-DL-13 | deliverables | Notice on the title table: 'There will be first phase evaluation before lunch break.' | human_external_blocker |
| R-AP-01 | assessment | Train and compare at least five models before selecting a final one; support selection with recorded results; explain strengths and limitations of the chosen approach; demonstrate Bayesian HPO; retain the search record; evaluate models and the decoder using permitted development data. Selecting without meaningful experimentation may cost marks. | done_evidence_exists |
| R-AP-02 | assessment | Each team independently implements, trains, evaluates and selects its own models; sharing model files, weights, configurations, code, derived datasets or results between teams is prohibited; submissions are compared across teams and sections; teams must explain selection, implementation, training and results during evaluation. | human_external_blocker |
| R-AP-03 | assessment | Prohibited: fabricating metrics, graphs or execution results; fabricating citations or implementation details; copying another team's report; submitting explanations the team cannot defend. Teams remain responsible for accuracy and originality of all content. | done_evidence_exists |
| R-AP-04 | assessment | Organisers may ask team members to explain dataset structure, preprocessing, feature engineering, model comparisons and selection, Bayesian HPO, HMM implementation, Viterbi decoding, API, UI and experimental results; teams should demonstrate they understand and can reproduce their submitted work. | human_external_blocker |
| R-AP-05 | assessment | Organisers compare source code, repository history, model architectures, weights, hyperparameter configurations, experimental results, reports, graphs and tables; similarity may trigger review; no single similarity measure is automatic proof of plagiarism. | done_evidence_exists |
| R-US-01 | user_scope | User order: complete Logistic Regression, KNN and Bagging (done), then Gaussian Naive Bayes and HistGradientBoosting, for the entire set of deliverables across all five original DOCX. This latest instruction overrides the earlier three-family deferral recorded in docs/PHASE_SCOPE.json. | done_evidence_exists |
| R-US-02 | user_scope | User order: AFTER all five models are trained and tested, visit the OpenCode website and use the kind theme for the project website. Not present in the organizer documents. | done_evidence_exists |
| R-US-03 | user_scope | User-approved replacement of the brief's separate UI laptop with a real public website: same-origin UI/API, loopback-bound API, server-private reference data, isolated restartable services, no credentials in Git. | done_evidence_exists |
| R-US-04 | user_scope | The ten human-reported evaluator trips and their separate website section are handled only AFTER the final model/decoder freeze; the new inputs are not read before that point. | done_evidence_exists |

# Appendix C. Organizer-document provenance

The five original documents are private event material. The requirement audit re-extracted their paragraphs/tables and checked the synthesis's 480 nonempty paragraphs against the originals. This report cites the originals by filename/hash and the source locators in the matrix; it does not redistribute the documents. Source: [provenance summary](<analysis/final_requirements/provenance_summary.md>).

| Original document | SHA-256 |
| --- | --- |
| Main Guidelines.docx | 84279a8748ec6de9707ddbdcf13eadd230339d88d567cee6837c225df48274d2 |
| Dataset Exploration.docx | 87d93a36979cec6e57647506768e2275ffe437ffdf3b175f2344cf68e55109e2 |
| Models and Training Guidelines.docx | e85fae3503eba502485469e41ea69fad5feaf0a6c8b3c55248a9c772580cc3f1 |
| Deliverables.docx | 5cc83562dedf54f1a82c30fc024654c07278a314856ef4f3b35322feaa865e26 |
| Assessment and Plagiarism Policy.docx | 023de0bdfadc5d7bb2faa598b79cfe38c0c660ee4ce9d5bf9b9fa258042b2b53 |

# Appendix D. Implementation and library references

The local code/evidence citations establish the project's actual settings and results. The links below identify the libraries and algorithm APIs used; this review made no fresh HTTP requests and does not treat a documentation page as evidence that this project achieved a score.

- [scikit-learn LogisticRegression API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
- [scikit-learn KNeighborsClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).
- [scikit-learn BaggingClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html).
- [scikit-learn GaussianNB API](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html).
- [scikit-learn HistGradientBoostingClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
- [scikit-learn GroupKFold API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) and [GaussianProcessRegressor API](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html).
- [Folium documentation](https://python-visualization.github.io/folium/latest/) and [Leaflet reference](https://leafletjs.com/reference.html).
- [FastAPI documentation](https://fastapi.tiangolo.com/) for the request/response framework; the project's actual boundary is implemented in [backend service](<backend/service.py>) and [scorer](<backend/scoring.py>).

The decoder equations in Section 7 document this repository's weighted emission/transition implementation, rather than claiming byte-for-byte reproduction of a named published HMM. Local shortest-path and dynamic-programming code, frozen configuration hashes and route checks are the reproduction specification.
