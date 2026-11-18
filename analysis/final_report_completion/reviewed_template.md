# 1. Summary and scope

PathSnap matches an ordered GPS track to a directed road graph. A supervised binary classifier scores each nearby candidate arc; the scores become observation-level emission potentials for a graph-constrained HMM/Viterbi decoder. The delivery includes the learned model, command-line predictions, a two-section web interface and evidence that the API returns the same routes and observation assignments as the offline run.

The final comparison covers exactly five families: {{fact:families_json}}. The selected family is {{fact:sel_model_run}}; the delivered configuration is {{fact:final_run_name}}, which is a tuned configuration of that family, not a sixth family. The corrected model-comparison stage completed at {{fact:phase_completed}}. The default-to-tuned TRAIN selection objective moved from {{fact:cv_obj_hgb}} to {{fact:cv_obj_tuned}}, a change of {{fact:cv_gain}}. Selection used whole-trip TRAIN cross-validation, not student or evaluator scores.

After freezing model and decoder, the student-development CLI achieved trip-macro directed-arc F1 {{fact:off_edge_f1}} and length-weighted F1 {{fact:off_len_f1}}. The later evaluator pass achieved {{fact:eval_edge_f1}} and {{fact:eval_len_f1}}, respectively, over {{fact:eval_trips}} trips. All reported route scores use the complete supplied reference arc sequence. Observation accuracy and observation macro F1 use route-derived proxy labels, not independently supplied GPS-to-road truth.

Two limits affect interpretation. The ten student trips were historically used as development data, so this report does not call them globally blind. The evaluator audit found no matching IDs or exact track/network files against the checked TRAIN and student partitions; that finding does not prove geographical independence or official unseen status. Evaluator exact-route matches were {{fact:evaluator_exact}}; weak trip 15 is retained in every aggregate.

# 2. Dataset and exploratory audit

## 2.1 File meaning and partition roles

The original student archive contains {{fact:aud_trips}} trips and {{fact:aud_files}} files, with four files per trip. A track row is longitude, latitude and timestamp in seconds. Node rows are longitude and latitude. Arc rows contain zero-based tail/head node IDs, and an arc's row index is its original directed ID. A route is an ordered sequence of those IDs; its length need not equal the GPS row count. Parallel arcs and repeated traversals remain valid data.

The supplied split is 80 TRAIN trips and ten student-development trips. The full structural audit read every file and checked numerical formats, byte hashes, ranges, endpoints, timestamp order and cross-file consistency. The TRAIN networks contain {{fact:aud_arcs_pairs}} unique directed node pairs summed over their trip-local graphs; this is not a count for one shared global graph. The audit's pooled observation distributions and its equally weighted per-trip distributions answer different questions and are labelled separately in the figures. See [audit summary](<analysis/data_audit/summary.json>) and [per-trip audit](<analysis/data_audit/per_trip.json>).

The student role is recorded as “{{fact:student_role}}”. Those trips were visible in earlier development and earlier model/decoder work. For the final phase, family selection, HPO and the fixed decoder search read TRAIN objectives only; final student results were not fed back to revise the frozen settings. This narrower boundary is supportable. A claim that student data never influenced any prior design decision would not be.

## 2.2 Structural findings and what was preserved

The audit found {{fact:aud_broken_train}} broken consecutive-arc joins in TRAIN reference routes and {{fact:aud_broken_student}} in student reference routes. These are discontinuities in supplied reference walks, not discontinuous GPS observations. Student trip 00000076, not TRAIN, has {{fact:trip76_broken}} reference break; the decoder evidence locates it at join position 12. No claim is made that it was removed from training: it was never a TRAIN trip. References were not repaired to improve a score.

TRAIN also contains {{fact:aud_steps_over}} steps above 100 m/s if the timestamp unit is seconds, {{fact:aud_time_delta}} intervals differing from one second, {{fact:aud_dup_nodes}} extra nodes sharing coordinates and {{fact:aud_dup_rows}} duplicate-coordinate track rows. The audit found {{fact:aud_nonpos}} nonpositive time differences and {{fact:aud_repeated}} reference arc occurrences beyond first visits. Fast steps, gaps and repeated observations are disclosed rather than silently discarded. A disconnected full network does not itself invalidate a trip: the route must be connected within the relevant directed component.

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

The frozen schema has {{fact:feat_count}} ordered numerical features, version {{fact:feat_version}}. It combines lateral distance and transformed/ranked distance, candidate-set context, projection fraction and endpoint proximity, arc length, signed heading alignment at several windows, motion reliability, neighbouring lateral distances, local speed/turn/gap descriptors, endpoint degree and reverse-arc availability. Exact names and definitions are persisted in [feature schema](<artifacts/final_five/feature_schema.json>) and implemented in [feature extractor](<mapmatch/features.py#L17-L180>).

Candidates start within an 80 m radius with width 12. Radius expansion is triggered by an empty candidate set and capped at 1000 m. A target miss is different: candidates can exist while none represents the reference-derived proxy edge. Candidate construction and feature extraction accept graph and track only; references cannot enter them. Nearest-neighbour scaling for LR/KNN and GaussianNB's standardization are fitted inside each training fold, not on held-out trips. Tree models use the deterministic numerical features without learned scaling.

Several features deliberately use centred windows, up to eight rows on either side, and a forward-neighbour lateral distance. They are valid for the delivered whole-track offline/request-response task. They are not causal streaming features. A real-time point-by-point deployment would require a separately declared latency window or a new causal feature policy and validation; this report claims neither.

## 3.2 Proxy labels: the exact implemented rule

No per-GPS road labels were supplied. For supervised targets and postprediction observation scoring, the code searches every distinct canonical directed segment occurring anywhere in the supplied reference route. For observation t and candidate reference segment e, it minimizes the following score:


offset_score(t,e) = lateral_distance(t,e)^2 + 32 m² * reliability(t) * [1 - signed_heading_alignment(t,e)]

Here reliability is min(motion-window displacement / 12 m, 1). This label window uses at most three neighbouring rows each way and stops at local gaps above 20 s. A winning-to-runner-up score margin below 4 m² is marked ambiguous. The 32 m² coefficient and 4 m² ambiguity rule are different quantities; neither is a measured claim of true GPS assignment accuracy. The exact policy is [reference_observation_labels](<mapmatch/features.py#L191-L234>).

The label search is independent of the generated candidate set. If the selected canonical reference edge is absent from that set, the observation remains in the top-1 denominator and counts wrong. Repeated route traversals are collapsed only for the observation label's geometric search; route metrics retain their multiplicity. Intersections, opposite directions, stops, nearby parallel roads, repeated visits and reference defects remain sources of proxy uncertainty. Geometry-derived targets can favour geometric predictors.

# 4. Experimental protocol and metric definitions

## 4.1 Shared populations and honest reuse

The saved experiment protocol uses all {{fact:mm_train_trips}} TRAIN trips, with 350 deterministic observations per trip: {{fact:train_obs}} observations and {{fact:train_rows}} candidate rows. The {{fact:train_cv_folds}}-fold GroupKFold split holds out whole trips; no trip contributes candidate rows to both training and validation in a fold. Weights give equal total mass to an observation and inverse-count correction across trips. KNN cannot take sample weights, so its declared 60,000-row class-stratified reference bank samples within each class without replacement in proportion to those weights. Its scaler is fitted only on that fit's bank.

The primary objective is equal-trip mean out-of-fold observation top-1 proxy accuracy. The legacy objective text says “{{fact:cv_protocol}}”; the actual delivered records use three folds. All student observations are retained: {{fact:student_obs}} observations and {{fact:student_rows}} candidate rows. There are {{fact:train_miss}} TRAIN target misses ({{fact:train_miss_rate}}; {{fact:train_present}} present targets) and {{fact:student_miss}} student target misses ({{fact:student_miss_rate}}; {{fact:student_present}} present targets). These counts are not empty-set counts.

The LR, KNN and default Bagging fits/CV were reused from verified prior experiments with the same features, sampling, folds and parameters. GaussianNB, default HistGB and tuned HistGB were freshly fitted in the corrected final run; the acceptance records {{fact:accept_fresh}} fresh fits. The completed HistGB HPO study was reused rather than rerun. Reuse avoids misrepresenting old numerical work as new computation. The final selected model was refitted on the all-80 training sample, not on student or evaluator data. See [frozen protocol](<experiments/final_five/PROTOCOL.md>) and [corrected measured results](<experiments/final_five/current/results.json>).

## 4.2 Three different metric populations

1. Binary candidate metrics treat each candidate row as positive/negative at threshold 0.5. The two-class confusion matrices, binary macro F1, positive-class precision/recall and binary log loss all use these rows. Many candidate negatives make binary accuracy unsuitable as the sole headline.
2. Observation metrics choose one road per GPS observation. Top-1 is correct assignments divided by all observations, including absent-target misses. Canonical directed road classes are namespaced by trip so unrelated graph-local IDs are not merged. Pooled edge macro F1 and an equal-trip mean of per-trip edge F1 are separate quantities. Equal-trip top-1 and observation-pooled top-1 also differ on unequal trip lengths.
3. Route metrics compare the complete reconstructed route to the complete supplied reference walk. Directed-arc multiset overlap counts repeated occurrences, length overlap weights those occurrences by projected segment length, and edit similarity measures sequence order. These are not candidate-conditioned route scores and are not the binary candidate F1.

For an observation-level categorical emission log score, positive binary probabilities are clipped, then normalized within that observation's candidates. The unconditional NLL includes missing targets using a 1e-7 floor; the conditional NLL drops only absent-target observations and must be labelled “given candidate”. Neither is an HMM posterior log loss or a general calibration guarantee. The decoder's relative path cost includes geometric and transition terms and is a different object again. See [candidate metrics](<mapmatch/learning.py#L126-L208>), [sequence recount](<experiments/final_five/current/sequence_classification.json>) and [route scorer](<mapmatch/metrics.py>).

## 4.3 Mandatory nine-column comparison

This table preserves the requested columns. CV Macro F1 and Student Macro F1 are binary candidate-row measures. Top-1 Accuracy is equal-trip student observation accuracy; the selection objective itself is TRAIN CV top-1, reported immediately below. Log Loss is student binary candidate log loss. Inference Time is the measured estimator probability call over the fixed student candidate matrix, in seconds, excluding candidate construction, features, routing, scoring, HTTP and browser rendering. Reused rows retain their original measured timings; these are not repeated hardware-normalized benchmarks. Route continuity is N/A here because the independent classifier does not reconstruct a route; the raw adjacency diagnostic and HMM route continuity are reported separately.

| Model | Algorithm Family | Key Config | CV Macro F1 | Top-1 Accuracy | Route continuity | Student Macro F1 | Log Loss | Inference Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | scaled linear logistic | {{fact:params_lr}} | {{fact:cv_f1_lr}} | {{fact:st_top1_lr}} | N/A: raw classifier has no reconstructed route | {{fact:st_f1_lr}} | {{fact:st_ll_lr}} | {{fact:lr_infer_s}} |
| knn_distance_weighted | nonparametric neighbours | {{fact:params_knn}} | {{fact:cv_f1_knn}} | {{fact:st_top1_knn}} | N/A: raw classifier has no reconstructed route | {{fact:st_f1_knn}} | {{fact:st_ll_knn}} | {{fact:knn_infer_s}} |
| bagging | bootstrap tree ensemble | {{fact:params_bag}} | {{fact:cv_f1_bag}} | {{fact:st_top1_bag}} | N/A: raw classifier has no reconstructed route | {{fact:st_f1_bag}} | {{fact:st_ll_bag}} | {{fact:bag_infer_s}} |
| gaussian_nb | Gaussian generative Bayes | {{fact:params_gnb}} | {{fact:cv_f1_gnb}} | {{fact:st_top1_gnb}} | N/A: raw classifier has no reconstructed route | {{fact:st_f1_gnb}} | {{fact:st_ll_gnb}} | {{fact:gnb_infer_s}} |
| hist_gradient_boosting | histogram boosted trees | {{fact:params_hgb}} | {{fact:cv_f1_hgb}} | {{fact:st_top1_hgb}} | N/A: raw classifier has no reconstructed route | {{fact:st_f1_hgb}} | {{fact:st_ll_hgb}} | {{fact:hgb_infer_s}} |
| hist_gradient_boosting_tuned | same boosted-tree family | {{fact:params_tuned}} | {{fact:cv_f1_tuned}} | {{fact:st_top1_tuned}} | N/A: raw classifier has no reconstructed route | {{fact:st_f1_tuned}} | {{fact:st_ll_tuned}} | {{fact:tuned_infer_s}} |

The family-selection CV top-1 values are LR {{fact:cv_obj_lr}}, KNN {{fact:cv_obj_knn}}, Bagging {{fact:cv_obj_bag}}, GaussianNB {{fact:cv_obj_gnb}} and default HistGB {{fact:cv_obj_hgb}}. Tuned HistGB reaches {{fact:cv_obj_tuned}}. A sixth row represents a second configuration of HistGB only.

## 4.4 Observation and continuity diagnostics

The table below keeps pooled and trip-macro values explicit. “Raw continuity” means adjacent assignments are the same canonical edge or directly head-to-tail connected, without inserted routing connectors; it is not validation of a full route. The edge-change version excludes same-edge pairs, which otherwise dominate this statistic. It includes observed time gaps. The conditional NLL can look better by excluding misses, so the unconditional score is reported alongside it.

| Model | Pooled observation top-1 | Pooled edge macro F1 | Trip-macro edge F1 | Unconditional emission NLL | NLL given candidate | Raw adjacency continuity | Edge-change continuity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | {{fact:lr_pooled_top1}} | {{fact:lr_pooled_edge_f1}} | {{fact:seq_edge_lr}} | {{fact:lr_all_nll}} | {{fact:seq_nll_lr}} | {{fact:lr_raw_continuity}} | {{fact:lr_change_continuity}} |
| knn_distance_weighted | {{fact:knn_pooled_top1}} | {{fact:knn_pooled_edge_f1}} | {{fact:seq_edge_knn}} | {{fact:knn_all_nll}} | {{fact:seq_nll_knn}} | {{fact:knn_raw_continuity}} | {{fact:knn_change_continuity}} |
| bagging | {{fact:bag_pooled_top1}} | {{fact:bag_pooled_edge_f1}} | {{fact:seq_edge_bag}} | {{fact:bag_all_nll}} | {{fact:seq_nll_bag}} | {{fact:bag_raw_continuity}} | {{fact:bag_change_continuity}} |
| gaussian_nb | {{fact:gnb_pooled_top1}} | {{fact:gnb_pooled_edge_f1}} | {{fact:seq_edge_gnb}} | {{fact:gnb_all_nll}} | {{fact:seq_nll_gnb}} | {{fact:gnb_raw_continuity}} | {{fact:gnb_change_continuity}} |
| hist_gradient_boosting | {{fact:hgb_pooled_top1}} | {{fact:hgb_pooled_edge_f1}} | {{fact:seq_edge_hgb}} | {{fact:hgb_all_nll}} | {{fact:seq_nll_hgb}} | {{fact:hgb_raw_continuity}} | {{fact:hgb_change_continuity}} |
| hist_gradient_boosting_tuned | {{fact:tuned_pooled_top1}} | {{fact:tuned_pooled_edge_f1}} | {{fact:seq_edge_tuned}} | {{fact:tuned_all_nll}} | {{fact:seq_nll_tuned}} | {{fact:tuned_raw_continuity}} | {{fact:tuned_change_continuity}} |

The candidate-row accuracy, positive-class precision and positive-class recall below are not road-class macro precision/recall. They document the binary decision shown by the confusion matrices.

| Model | Binary row accuracy | Positive-class precision | Positive-class recall | Binary macro F1 |
| --- | --- | --- | --- | --- |
| logistic_regression | {{fact:lr_binary_accuracy}} | {{fact:lr_binary_precision}} | {{fact:lr_binary_recall}} | {{fact:st_f1_lr}} |
| knn_distance_weighted | {{fact:knn_binary_accuracy}} | {{fact:knn_binary_precision}} | {{fact:knn_binary_recall}} | {{fact:st_f1_knn}} |
| bagging | {{fact:bag_binary_accuracy}} | {{fact:bag_binary_precision}} | {{fact:bag_binary_recall}} | {{fact:st_f1_bag}} |
| gaussian_nb | {{fact:gnb_binary_accuracy}} | {{fact:gnb_binary_precision}} | {{fact:gnb_binary_recall}} | {{fact:st_f1_gnb}} |
| hist_gradient_boosting | {{fact:hgb_binary_accuracy}} | {{fact:hgb_binary_precision}} | {{fact:hgb_binary_recall}} | {{fact:st_f1_hgb}} |
| hist_gradient_boosting_tuned | {{fact:tuned_binary_accuracy}} | {{fact:tuned_binary_precision}} | {{fact:tuned_binary_recall}} | {{fact:st_f1_tuned}} |

## 4.5 Fit diagnostics and their limits

| Model | TRAIN resubstitution top-1 | Grouped-CV top-1 | TRAIN binary macro F1 | TRAIN binary log loss | OOF binary log loss | All-80 fit time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | {{fact:lr_train_top1}} | {{fact:cv_obj_lr}} | {{fact:lr_train_binary_f1}} | {{fact:lr_train_ll}} | {{fact:cv_ll_lr}} | {{fact:lr_fit_s}} |
| knn_distance_weighted | {{fact:knn_train_top1}} | {{fact:cv_obj_knn}} | {{fact:knn_train_binary_f1}} | {{fact:knn_train_ll}} | {{fact:cv_ll_knn}} | {{fact:knn_fit_s}} |
| bagging | {{fact:bag_train_top1}} | {{fact:cv_obj_bag}} | {{fact:bag_train_binary_f1}} | {{fact:bag_train_ll}} | {{fact:cv_ll_bag}} | {{fact:bag_fit_s}} |
| gaussian_nb | {{fact:gnb_train_top1}} | {{fact:cv_obj_gnb}} | {{fact:gnb_train_binary_f1}} | {{fact:gnb_train_ll}} | {{fact:cv_ll_gnb}} | {{fact:gnb_fit_s}} |
| hist_gradient_boosting | {{fact:hgb_train_top1}} | {{fact:cv_obj_hgb}} | {{fact:hgb_train_binary_f1}} | {{fact:hgb_train_ll}} | {{fact:cv_ll_hgb}} | {{fact:hgb_fit_s}} |
| hist_gradient_boosting_tuned | {{fact:tuned_train_top1}} | {{fact:cv_obj_tuned}} | {{fact:tuned_train_binary_f1}} | {{fact:tuned_train_ll}} | {{fact:cv_ll_tuned}} | {{fact:tuned_fit_s}} |

Resubstitution and held-out metrics are useful diagnostics, but their difference is not an isolated causal estimate of overfitting: fold training sets are smaller, observations are sampled for TRAIN, and student trips have another length/composition distribution. A student score exceeding a TRAIN score is not evidence that no overfitting exists. None of these estimator pipelines saved epoch-wise training/validation histories. Such curves are N/A for each family; measured fit/CV/HPO summaries are supplied instead.

# 5. The five individual model families

## 5.1 Logistic Regression

**Why and suitability.** A scaled linear classifier is a small, fast baseline for tabular distance, direction and topology evidence. It tests how much the engineered features explain without a nonlinear ensemble. The initial configuration is {{fact:params_lr}}; L-BFGS minimizes regularized binary logistic loss, while selection evaluates the common grouped-CV observation objective. Fold-local StandardScaler prevents validation information entering the learned scaling.

**Strengths and limits.** The model is cheap to fit and score, but a linear log-odds surface cannot directly learn arbitrary distance-by-motion-by-topology interactions. Its all-80 fit took {{fact:lr_fit_s}} s and its saved student probability call {{fact:lr_infer_s}} s. Student binary precision/recall were {{fact:lr_binary_precision}} / {{fact:lr_binary_recall}}, binary macro F1 {{fact:st_f1_lr}}, pooled observation accuracy {{fact:lr_pooled_top1}} and pooled road-class macro F1 {{fact:lr_pooled_edge_f1}}.

**Observed fit behaviour and epoch applicability.** Resubstitution top-1 {{fact:lr_train_top1}} is close to grouped-CV {{fact:cv_obj_lr}}; this is consistent with limited in-sample gain for this configuration. It does not establish universal underfitting or prove absence of bias. Epoch curves are N/A: the saved solver fit has a maximum iteration limit, not a logged epoch-by-epoch training/validation history. This final row is a verified reused fit.

## 5.2 Distance-weighted KNN

**Why and suitability.** KNN is a genuinely different distance-based family: standardized candidate features are compared by exact Euclidean neighbour search, and inverse-distance class votes form the probabilities. Initial parameters are {{fact:params_knn}}; fixed mechanics are distance weights, Minkowski p=2 and brute-force neighbours. There is no optimized parametric training loss. Bank construction and k are the relevant model-capacity choices; k was not changed from these final baseline results.

**Strengths and limits.** Local examples can express nonlinear decisions without fitting trees. Memory and query cost grow with the reference bank, and a 38-dimensional distance can be a poor summary when features are correlated or irrelevant. The declared bank preserves class proportions and approximates observation/trip weights through sampling, rather than claiming exact sample-weight support. Fit time {{fact:knn_fit_s}} s mainly reflects storing the bank; the student probability call took {{fact:knn_infer_s}} s, and summed CV query time was {{fact:cv_sec_knn_inf}} s. These must not be described as end-to-end service latency.

**Test metrics and fit behaviour.** Student binary precision/recall {{fact:knn_binary_precision}} / {{fact:knn_binary_recall}}, binary macro F1 {{fact:st_f1_knn}}, pooled observation accuracy {{fact:knn_pooled_top1}} and pooled edge macro F1 {{fact:knn_pooled_edge_f1}} trail the stronger tree configurations. Resubstitution top-1 {{fact:knn_train_top1}} exceeds grouped-CV {{fact:cv_obj_knn}}; locality and self-neighbour effects can contribute, so this is reported as an in-sample advantage, not diagnosed from student data alone. Epoch curves are N/A because KNN stores a bank and has no epoch training process. The fit is verified historical reuse.

## 5.3 Bagging

**Why and suitability.** This is genuine bootstrap aggregation of DecisionTreeClassifier base learners, not an alias for RandomForest or HistGB. The initial configuration is {{fact:params_bag}}; bootstrap samples use 70% of rows, trees are depth-limited and leaves require a minimum sample count. Each tree greedily reduces Gini impurity; averaged binary probabilities are evaluated by the same CV objective and log-loss diagnostics. There is no differentiable ensemble epoch loss.

**Strengths and limits.** Bagging captures nonlinear interactions and reduces the variance of an individual tree. Fitting multiple trees costs more and does not by itself constrain road-sequence connectivity. All-80 fit time was {{fact:bag_fit_s}} s, summed CV training {{fact:cv_sec_bag}} s, and student probability inference {{fact:bag_infer_s}} s. Student binary precision/recall {{fact:bag_binary_precision}} / {{fact:bag_binary_recall}}, macro F1 {{fact:st_f1_bag}}, pooled observation accuracy {{fact:bag_pooled_top1}} and pooled edge macro F1 {{fact:bag_pooled_edge_f1}} make it a competitive baseline.

**Observed fit behaviour and epoch applicability.** Resubstitution top-1 {{fact:bag_train_top1}} versus grouped-CV {{fact:cv_obj_bag}} shows a larger in-sample advantage than LR. Depth/leaf limits constrain capacity but do not remove that gap. It is not justified to conclude either guaranteed generalization or complete overfit from these aggregates. Epoch curves are N/A; no per-tree training/validation curve was saved. The final default fit is reused; the separately tuned historical Bagging model appears only in Appendix A.

## 5.4 Gaussian Naive Bayes

**Why and suitability.** GaussianNB supplies a fast generative comparator. It estimates class priors and per-feature Gaussian means/variances, using conditional independence to form a class likelihood; it does not optimize logistic loss. Initial configuration {{fact:params_gnb}} specifies numerical variance smoothing. Its scaling is learned inside each fit. The speed and simple distributional assumptions make it a useful counterpoint to nearest neighbours and nonlinear ensembles.

**Strengths and limits.** Many distance, motion and rank features are deliberately related. Conditional independence is therefore a restrictive approximation and its probabilities can be overconfident. The observed student binary log loss {{fact:st_ll_gnb}} is substantially worse than the tree alternatives even though pooled observation accuracy is {{fact:gnb_pooled_top1}}. Student precision/recall are {{fact:gnb_binary_precision}} / {{fact:gnb_binary_recall}}, binary macro F1 {{fact:st_f1_gnb}} and pooled edge macro F1 {{fact:gnb_pooled_edge_f1}}. Fit and probability inference took {{fact:gnb_fit_s}} s and {{fact:gnb_infer_s}} s.

**Numerical correction and fit behaviour.** The preserved pre-correction float32 diagnostic measured maximum probability-row-sum error {{fact:gnb_rowsum}}. Its {{fact:gnb_outside}} out-of-bounds entries and {{fact:gnb_affected}} affected rows refer to bounds violations, not to row-sum violations. That first attempt stopped before a final artifact was saved. Promoting only GaussianNB scoring input to float64 fixed the observed sum problem without relaxing the probability guard; the corrected current run is the reported evidence. Resubstitution top-1 {{fact:gnb_train_top1}} and CV {{fact:cv_obj_gnb}} are close, but weak held-out performance can reflect model assumptions as well as data noise. Epoch curves are N/A because this is a closed-form statistical fit, not epoch training. See [diagnostic](<reports/final_five/gnb_probability_diagnostic.json>) and [scoring implementation](<mapmatch/learning.py#L28-L45>).

## 5.5 Histogram Gradient Boosting

**Why and suitability.** HistGradientBoosting models nonlinear tabular interactions with histogram-binned decision trees, adding trees to reduce binary log loss. It can capture combinations such as distance evidence that becomes less reliable at a turn or after a gap. The default configuration is {{fact:params_hgb}}; the delivered tuned configuration is {{fact:params_tuned}}. Early stopping is disabled so there is no hidden row-random validation split inside a grouped-CV fit.

**Strengths and limits.** The family gave the best untuned TRAIN trip-macro top-1 objective among the five. It is much cheaper to query than the measured KNN baseline, but learning emission scores still cannot guarantee connected roads; the HMM supplies the graph constraint. Default/tuned all-80 fits took {{fact:hgb_fit_s}} / {{fact:tuned_fit_s}} s, and student probability inference {{fact:hgb_infer_s}} / {{fact:tuned_infer_s}} s. Tuned student binary precision/recall are {{fact:st_prec_tuned}} / {{fact:st_rec_tuned}}, binary macro F1 {{fact:st_f1_tuned}}, pooled observation accuracy {{fact:tuned_pooled_top1}} and pooled edge macro F1 {{fact:tuned_pooled_edge_f1}}.

**Observed fit behaviour and epoch applicability.** Default resubstitution top-1 {{fact:hgb_train_top1}} exceeds CV {{fact:cv_obj_hgb}}; tuned resubstitution {{fact:tuned_train_top1}} exceeds CV {{fact:cv_obj_tuned}}. The tuned gap is larger, so no claim of reduced overfitting follows merely from tuning. Trees are boosting iterations, not recorded neural-network epochs; epoch-wise train/validation curves are N/A. The current default and tuned fits are fresh. Their parameter study is historical reuse, explained next.

# 6. Model selection, tuning and measured plots

## 6.1 Selection rule and retained Bayesian study

The primary TRAIN criterion selects the default HistGB family. The retained study used {{fact:hpo_method}}, with {{fact:hpo_trials}} measured trials including the prior baseline, {{fact:hpo_seeded}} seeded initial-design trials and {{fact:hpo_modelbased}} model-based proposals. The current final phase executed {{fact:hpo_fresh_trials}} fresh HPO trials. The historical study duration was {{fact:hpo_seconds}} s; it is not a new final-run runtime. Its reuse note reads: “{{fact:hpo_reuse_note}}”. Fresh default/tuned CV and all-80 refits validate the selected configuration, rather than claiming the study itself was rerun.

The search space was {{fact:hpo_space}}. The selected parameters are {{fact:hpo_best_params}}. The historical baseline objective {{fact:hpo_baseline}} became {{fact:hpo_best}}, gain {{fact:hpo_gain}}. Equal-trip fold means/stds are default {{fact:cv_t1m_hgb}} / {{fact:cv_t1s_hgb}} and tuned {{fact:cv_t1m_tuned}} / {{fact:cv_t1s_tuned}}; summed tuned CV training was {{fact:cv_sec_tuned}} s.

The improvement is small and objective-specific. Tuned CV binary macro F1 {{fact:cv_f1_tuned}} is lower than default {{fact:cv_f1_hgb}}; tuned CV binary log loss {{fact:cv_ll_tuned}} is worse than default {{fact:cv_ll_hgb}}. Student binary log loss also moves from {{fact:st_ll_hgb}} to {{fact:st_ll_tuned}}, despite a student equal-trip top-1 change {{fact:st_gain}}. The tuned configuration did not win every metric. This tradeoff is accepted because the selection objective was declared in advance. Twelve trials are too few, and their acquisition is too adaptive, to treat scatterplot associations as causal hyperparameter importance.

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

Sigma controls the spatial emission penalty and beta controls discrepancy between network travel and GPS chord, both in metres. They are not learning rates, class thresholds or speed limits. Historical TRAIN-only residual/transition fitting was reused because it is geometric rather than classifier-specific; it used TRAIN references, so “reference-free calibration” would be false. The final model already fitted all {{fact:dplan_trips}} TRAIN trips before the {{fact:dec_search_trips}}-trip decoder search. This is in-sample calibration, not held-out route generalization.

## 7.2 Preregistered configurations and freeze

The fixed search compared A fitted scales, B twice beta, and C classifier weight 2 with geometric weight 0.5 on the preregistered TRAIN8 set. Its objective was mean exact-original-ID length-weighted repeated-arc route F1, with a failed trip contributing zero. Scores were A length/edge {{fact:dec_a_lenf1}} / {{fact:dec_a_edgef1}}, B {{fact:dec_b_lenf1}} / {{fact:dec_b_edgef1}}, and C {{fact:dec_c_lenf1}} / {{fact:dec_c_edgef1}}. It selected {{fact:dec_selected}} and froze at {{fact:dec_frozen_utc}}, before the final student ablation and evaluator access.

The delivered settings are sigma {{fact:dec_sigma}} m, beta {{fact:dec_beta}} m, classifier weight {{fact:dec_cls_w}} and geometric weight {{fact:dec_geo_w}}. Initial radius is {{fact:dec_radius}} m, routing slack {{fact:dec_slack}} m, backward tolerance {{fact:dec_btol}} m and graph-index sample spacing {{fact:dec_spacing}} m. Width begins at {{fact:dec_maxcand}} and can widen 12 -> 24 -> {{fact:dec_retry}} only after genuine reachability failure. Observation distance is zero, so every original observation is decoded. Graph-index spacing does not mean GPS observations were downsampled. See [calibration plan](<reports/final_five/decoder/calibration_plan.json>) and [decoder freeze](<reports/final_five/decoder/freeze.json>).

## 7.3 Paired student-development ablation

The ablation contains {{fact:abl_trips}} trips and all {{fact:abl_obs}} observations, with {{fact:abl_failed}} failed trips. Raw classifier argmax and learned HMM use the same final candidates/probabilities. The raw route comparator collapses adjacent identical IDs but does not insert shortest-path connectors; it can therefore be disconnected. Its F1 is still scored against the full reference route, including arcs not present in candidate sets. This is a deliberately weak pointwise route baseline, not an alternative connected decoder.

| Method | Trip-macro observation proxy accuracy | Complete-reference route edge F1 | Complete-reference length F1 |
| --- | --- | --- | --- |
| classifier_only_raw | {{fact:abl_raw_top1}} | {{fact:abl_raw_edge}} | {{fact:abl_raw_len}} |
| classifier_plus_HMM | {{fact:abl_hmm_top1}} | {{fact:abl_hmm_edge}} | {{fact:abl_hmm_len}} |
| geometry_HMM_default_reused | N/A in this table | {{fact:abl_geo_edge}} | {{fact:abl_geo_len}} |

The HMM raises route edge F1 by {{fact:abl_hmm_edge_gain}} over this raw comparator. Mean measured HMM inference time is {{fact:abl_hmm_seconds}} s per trip. The geometry-only row is an actual historical all-ten run reused as context, not a freshly rerun third final model. Differences do not isolate every effect: the paired raw/learned-HMM comparison shares candidates and scores, while the historical geometry baseline can differ in configuration and candidate behaviour.

The separately executed CLI matches the ablation route hashes and reaches route edge F1 {{fact:off_edge_f1}}, length F1 {{fact:off_len_f1}}, edit similarity {{fact:off_edit}} and continuity {{fact:off_cont}}. The student reference break remains disclosed, while predictions are validated as connected directed walks. See [paired ablation](<reports/final_five/decoder/student_ablation.json>) and [actual CLI verification](<reports/final_five/offline_delivery_verification.json>).

![Actual three-method all-ten route ablation](visualizations/final_five/plots/final_five_route_ablation.png)

# 8. Postfreeze evaluator pass

## 8.1 Access chronology and overlap audit

The model-stage record said “{{fact:evaluator_status}}”; that was its status at model completion, not a statement that the completed project never evaluated the new inputs. The model froze at {{fact:frozen_utc}} and the decoder at {{fact:dec_frozen_utc}}. The preregistered evaluator's first graph/track file open began {{fact:evaluator_first_open}}. One actual all-ten CLI completed at {{fact:evaluator_predictions_completed}}; reference scoring began only at {{fact:evaluator_reference_start}}, after full prediction verification. No fit, feature revision, HPO, decoder adjustment or prediction retry followed those scores. See [evaluator protocol](<docs/FINAL_EVALUATOR_PROTOCOL.md>) and [chronology/acceptance](<reports/final_five/evaluator/acceptance.json>).

The audit checked exact track, nodes and arcs hashes against all {{fact:overlap_train_count}} frozen TRAIN trips and {{fact:overlap_student_count}} student trips. Same-ID matches were {{fact:overlap_id_matches}}; any identical input file was {{fact:overlap_any_file}} and any full-trip duplicate {{fact:overlap_exact_duplicate}}. Matching IDs alone would not prove duplicate data, and unmatched checksums do not rule out geographic overlap, transformed copies or other prior exposure. The dataset is described here as the supplied postfreeze evaluator set, not an independently certified unseen official benchmark.

## 8.2 Results, proxy denominators and weak cases

All {{fact:evaluator_obs}} original observations were predicted and proxy-scored across {{fact:eval_trips}} trips. Macro route edge F1 is {{fact:eval_edge_f1}}, length F1 {{fact:eval_len_f1}}, edit similarity {{fact:eval_edit}} and route continuity {{fact:eval_cont}}. References have {{fact:evaluator_refbreaks}} broken joins. Exact routes: {{fact:evaluator_exact}}. Trip 15's edge F1 {{fact:evaluator_weak15_edge}} and length F1 {{fact:evaluator_weak15_length}} are materially weaker than the aggregate; neither the trip nor its errors were removed.

Trip-macro classifier top-1 is {{fact:eval_top1}}; trip-macro decoded road-class F1 is {{fact:eval_mf1}}. Over all observations, pooled decoded accuracy/F1 are {{fact:evaluator_decoded_accuracy}} / {{fact:evaluator_decoded_f1}}, versus raw classifier {{fact:evaluator_classifier_accuracy}} / {{fact:evaluator_classifier_f1}}. The pooled classifier accuracy is slightly higher than decoded accuracy while decoded macro F1 is higher. The report retains that tradeoff and does not claim the HMM improves every metric. All observation measures are the same geometry-and-motion proxies described earlier, not independently labelled GPS truth.

The offline CLI saved classifier IDs and matched states but not complete candidate probability arrays or memberships. Offline evaluator candidate NLL and miss rate are therefore N/A with an explicit reason, and no new inference was run to fill them. The later live/API audit scores probabilities it actually records during fresh API requests; those results must be labelled as API measurements rather than retroactively inserted into the offline pass. HMM posterior NLL remains unavailable. Missing metrics are not replaced with zero.

The following per-trip categorical NLL and candidate miss rates come from the actual twenty-trip ASGI API integration pass, after offline evaluation. Its request-local audit records every candidate probability used by the decoder, and the scorer runs after inference. These are current measured API values, not missing offline CLI arrays and not binary candidate log losses. The ASGI test exercises the real estimator without sockets; the separately recorded HTTPS/browser tests supply public-network proof. Source: [actual API integration](<reports/final_five/api_integration_acceptance.json>) and [postprediction scoring](<backend/scoring.py#L99-L118>).

| Evaluator trip | Actual API emission categorical NLL | Actual API candidate-miss fraction |
| --- | --- | --- |
| 00000015 | {{fact:api_eval_15_nll}} | {{fact:api_eval_15_miss}} |
| 00000017 | {{fact:api_eval_17_nll}} | {{fact:api_eval_17_miss}} |
| 00000023 | {{fact:api_eval_23_nll}} | {{fact:api_eval_23_miss}} |
| 00000030 | {{fact:api_eval_30_nll}} | {{fact:api_eval_30_miss}} |
| 00000049 | {{fact:api_eval_49_nll}} | {{fact:api_eval_49_miss}} |
| 00000059 | {{fact:api_eval_59_nll}} | {{fact:api_eval_59_miss}} |
| 00000067 | {{fact:api_eval_67_nll}} | {{fact:api_eval_67_miss}} |
| 00000073 | {{fact:api_eval_73_nll}} | {{fact:api_eval_73_miss}} |
| 00000087 | {{fact:api_eval_87_nll}} | {{fact:api_eval_87_miss}} |
| 00000094 | {{fact:api_eval_94_nll}} | {{fact:api_eval_94_miss}} |

The evaluator passed its 190 invariant checks and {{fact:evaluator_guard_tests}} synthetic guard tests. The ZIP contains ten flat predicted route files only, SHA-256 {{fact:evaluator_zip_sha}}. Private references, raw coordinates, candidate arrays and observation CSVs are excluded from public evaluator reports. See [all per-trip evaluator metrics](<reports/final_five/evaluator/route_metrics.json>), [input audit](<reports/final_five/evaluator/input_audit.json>) and [privacy/test verification](<reports/final_five/evaluator/test_and_privacy_verification.json>).

# 9. API, website and genuine browser evidence

## 9.1 Delivered system

The user-authorized delivery is the public HTTPS site {{fact:live_base}}, reusing the dedicated API service and existing tunnel. It replaces the requested physical two-laptop arrangement for this user; it does not prove a venue demonstration occurred. Development and evaluator sections have separate allowlisted trip registries, with the same frozen model/configuration and inference path. Requests identify dataset and trip, not client-supplied filesystem paths, model files or decoder overrides. References are accessed only by the scoring phase after prediction and are not sent as reference geometry in match responses.

The measured HTTPS evidence has {{fact:live_matches}} successful trip matches, with all twenty requested flag {{fact:live_requested}}. The live health response reports frozen model SHA {{fact:live_health_sha}}. API integration verifies route, predicted geometry and matched-state hashes against offline predictions for ten development and ten evaluator trips. A model hash alone would not prove route parity, so the array and geometry comparisons matter. [API integration evidence](<reports/final_five/api_integration_acceptance.json>), [consolidated parity](<reports/final_five/delivery_verification.json>) and [live API evidence](<reports/final_five/live/live_api.json>) supply the checks.

The interface uses an original PathSnap dark/light design informed by the explicitly visited OpenCode reference after model acceptance. Local Leaflet assets avoid third-party basemap requests in the recorded test. The UI shows request/model identity, route and raw GPS layers, metric definitions, and separate inference/scoring timing. It preserves theme preference and handles dataset changes without presenting a stale result as another trip. The browser evidence is actual Chromium execution without mocked responses or request interception. Its browser-error count is {{fact:browser_errors}} and console-error count {{fact:browser_console}}; these are properties of the recorded sessions, not a promise that every browser/environment is error-free.

## 9.2 Desktop, themes, evaluator and mobile

The captures below show the deployed page, not fixture screens. The evidence tests dark/light themes, both datasets and widths 320, 390, 768 and 1440 with no horizontal overflow. The development browser request ID is {{fact:browser_live_request_id}} and evaluator request ID {{fact:browser_evaluator_request_id}}. The evaluator screenshot example uses trip 87 with proxy decoded accuracy {{fact:browser_evaluator_accuracy}}; it is one trip and is not the all-ten mean. Keyboard checks include dataset-tab navigation, modal Escape/focus return and theme persistence. Changing theme or reloading does not silently launch a new match. See [browser evidence](<reports/final_five/live/browser_evidence.json>).

![Actual deployed dark desktop view](reports/final_five/live/screenshots/ui_desktop.png)

![Actual raw GPS and predicted route map panel](reports/final_five/live/screenshots/ui_route_map.png)

![Actual deployed light theme](reports/final_five/live/screenshots/ui_light.png)

![Actual separate evaluator section](reports/final_five/live/screenshots/ui_evaluator.png)

![Actual narrow 320-pixel mobile view](reports/final_five/live/screenshots/ui_mobile_320.png)

![Actual 390-pixel mobile view](reports/final_five/live/screenshots/ui_mobile.png)

## 9.3 Receiving-terminal correlation

The genuine Alacritty screenshot displays the dedicated service's received-request journal. The browser initiated request {{fact:terminal_request_id}} for the {{fact:terminal_dataset}} section; the terminal did not originate the HTTP request. The log records match start {{fact:terminal_start_utc}}, completion {{fact:terminal_finish_utc}} and an HTTP event with the same ID {{fact:terminal_http_request_id}}. In this allowlisted transport event, the model hash is absent; model/config identity comes from the same-ID match-start/completion events and browser response. It would be false to require every transport event to carry the model hash or to replace the real log with typed sample output.

The terminal record marks validity {{fact:term_valid}} and OCR privacy guard {{fact:term_ocr}}. These are automated provenance/privacy checks, not claims that a human independently inspected the capture. See [terminal evidence](<reports/final_five/live/api_terminal_evidence.json>).

![Genuine receiving terminal with browser-correlated request events](reports/final_five/live/screenshots/api_terminal.png)

# 10. Actual final-model Folium map

The final explanatory map is a distinct artifact from the interactive application's privacy boundary. It uses the delivered tuned HistGB prediction for development trip 00000009, with {{fact:map_points}} GPS points, {{fact:map_pred_arcs}} predicted arc occurrences and {{fact:map_ref_arcs}} reference arc occurrences. Route edge/length F1 are {{fact:map_edge_f1}} / {{fact:map_length_f1}}. Folium version {{fact:map_folium_version}} generated the actual HTML; a browser rendered its PNG/PDF. The map is not an old Bagging route with a new label.

The map draws every predicted arc in CLI order, including repeated occurrences, using supplied endpoints. It does not invent missing road shape or splice route gaps for appearance. Network, GPS, prediction and supplied reference layers are labelled. The development reference geometry is intentionally present for this analysis-only artifact, so its manifest says not to serve it through the application. No evaluator reference geometry belongs in a public map. Tooltip time ranges pooled by arc ID are not reconstructed per-traversal timestamps.

Open the [interactive Folium map](<visualizations/final_five/maps/final_development_trip_00000009.html>); the [map metadata](<visualizations/final_five/maps/final_development_trip_00000009.metadata.json>) and [map manifest](<visualizations/final_five/map_manifest.json>) bind source/prediction/model/config hashes and browser rendering checks.

![Actual final tuned-HistGB development trip 9 Folium rendering](visualizations/final_five/maps/final_development_trip_00000009.png)

# 11. Reproducibility, requirements and remaining responsibilities

## 11.1 Frozen artifacts and clean replay

The selected model froze at {{fact:frozen_utc}}, size {{fact:frozen_bytes}} bytes, SHA-256 {{fact:sha_tuned}}. The metadata and feature-schema hashes are {{fact:mmeta_sha}} and {{fact:fschema_sha}}. Its fit declaration is “{{fact:mm_fit_data}}”, with seed {{fact:mm_seed}}, feature version {{fact:mm_feature_version}}, all-80 fit time {{fact:mm_fit_seconds}} s and sampled TRAIN candidate probability inference {{fact:mm_inf_seconds}} s. That latter timing is TRAIN inference, not student or API latency; the selected student probability call was {{fact:st_inf_tuned}} s.

Software at the final fit: Python {{fact:ver_python}}, scikit-learn {{fact:ver_sklearn}}, NumPy {{fact:ver_numpy}}, SciPy {{fact:ver_scipy}} and joblib {{fact:ver_joblib}}. A fresh-process clean-load check on the fixed 2,000-row TRAIN fixture reproduced probabilities with maximum absolute difference {{fact:clean_delta}} at tolerance {{fact:clean_atol}}, using matching recorded package versions. This demonstrates that particular reload, not a guarantee across all platforms, dependencies, compiler builds or arbitrary future environments. Trusted local joblib loading is required because pickle-based artifacts can execute code.

| Compared configuration | Accepted artifact SHA-256 |
| --- | --- |
| logistic_regression | {{fact:sha_lr}} |
| knn_distance_weighted | {{fact:sha_knn}} |
| bagging | {{fact:sha_bag}} |
| gaussian_nb | {{fact:sha_gnb}} |
| hist_gradient_boosting | {{fact:sha_hgb}} |
| hist_gradient_boosting_tuned | {{fact:sha_tuned}} |


An independent AI reviewer recorded {{fact:ind_findings}} verified findings and no critical defects, including an actual 111/111 verification check run. This is evidence review, not an independent human viva or organizer approval. The final test record reports {{fact:tests_total}} total tests, {{fact:tests_passed}} passed and {{fact:tests_failed}} failures. Test provenance and actual counts are taken from [final test record](<reports/final_five/test_results.json>) rather than inferred from a successful command alone.

The [report builder](<scripts/build_final_report.py>) resolves this manuscript's registered fact tokens from cited measured evidence and should refuse a final release while required facts/assets/guards fail. Its manifest records the builder/template/input hashes. These bindings cover the current and historical metric tables; they do not mean every prose statement or every number in an image is automatically validated. Formatting or page count cannot turn missing evidence into a completed requirement.

## 11.2 Human/organizer exceptions and honest completion status

The current requirements matrix has {{fact:req_total}} rows: {{fact:req_done}} evidence-complete, {{fact:req_partial}} partial, {{fact:req_inprogress}} delegated/in progress, {{fact:req_notdone}} not done, {{fact:req_deferred}} user-deferred, {{fact:req_exception}} explicit user exceptions and {{fact:req_human}} human/external blockers. The builder calls {{fact:req_resolved}} rows administratively resolved under its terminal-status rule. That rule includes accepted exceptions and human/external blockers; “resolved” is not the same as “all organizer obligations performed.” The full source-located matrix follows in Appendix B.

AI agents implemented and reviewed substantial parts of this project under human direction. This disclosure does not establish organizer permission for AI assistance, cross-team independence, plagiarism clearance or human authorship. Administrative-account repository actions are not independent human review. Each participating member must be able to explain the data, model choices, proxy labels, selection, decoder, limitations and deployment in a viva; no evidence here proves that happened or requires an invented five-person team size. A physical two-laptop venue demonstration remains a human task if organizers require it despite the user-authorized website substitute. The before-lunch notice has no calendar date, so no evaluation date is inferred. See [requirements audit](<docs/FINAL_REQUIREMENTS.md>), [development provenance](<docs/DEVELOPMENT_PROVENANCE.md>) and [delivery guide](<docs/FINAL_DELIVERY_GUIDE.md>).

# 12. Conclusions

The selected HistGB configuration met the declared TRAIN whole-trip top-1 objective and was frozen before final decoder calibration/evaluator access. The delivered HMM turns independent candidate scores into connected directed routes and improves the student full-reference route comparator, while the evaluator retains a difficult trip, zero exact routes and an observation-accuracy tradeoff. These are measured outcomes, not a claim of solved map matching or independently verified GPS labels. CLI/API/browser identity checks bind the software demonstration to the frozen artifacts. Human viva, event policy and plagiarism decisions remain external.

# Appendix A. Historical experiments, kept separate from the final five

RandomForest and ExtraTrees were genuinely experimented models in the earlier broader comparison; tuned Bagging belongs to the earlier three-family phase. They are documented to preserve the experiment record, not counted as extra final families or substituted for the required five. Their original CV/student metrics and timings are reported without refits or claims that they were freshly tested now. Historical GaussianNB precision-era numbers are not substituted for corrected current GaussianNB.

RandomForest uses bootstrapped trees with feature subsampling; ExtraTrees adds randomized split thresholds and is not identical to Bagging. Tuned Bagging remains an ensemble of ordinary bootstrap decision trees. Each tree uses impurity-driven splits, with no epoch histories. This historical appendix is not a post-evaluator model-selection opportunity. The final scope was fixed to LR/KNN/Bagging/GaussianNB/HistGB before these final delivery results.

| Historical model | Exact saved configuration | CV binary macro F1 | CV trip-macro top-1 | Student pooled top-1 | Student trip-macro top-1 | Student binary macro F1 | Student binary log loss | Probability inference (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForest, historical | {{fact:historical_rf_params}} | {{fact:historical_rf_cv_f1}} | {{fact:historical_rf_cv_obj}} | {{fact:historical_rf_student_top1}} | {{fact:historical_rf_student_trip_top1}} | {{fact:historical_rf_student_f1}} | {{fact:historical_rf_student_ll}} | {{fact:historical_rf_infer_s}} |
| ExtraTrees, historical | {{fact:historical_et_params}} | {{fact:historical_et_cv_f1}} | {{fact:historical_et_cv_obj}} | {{fact:historical_et_student_top1}} | {{fact:historical_et_student_trip_top1}} | {{fact:historical_et_student_f1}} | {{fact:historical_et_student_ll}} | {{fact:historical_et_infer_s}} |
| Tuned Bagging, historical | {{fact:historical_bt_params}} | {{fact:historical_bt_cv_f1}} | {{fact:historical_bt_cv_obj}} | {{fact:historical_bt_student_top1}} | {{fact:historical_bt_student_trip_top1}} | {{fact:historical_bt_student_f1}} | {{fact:historical_bt_student_ll}} | {{fact:historical_bt_infer_s}} |

Historical all-80 fit times were RandomForest {{fact:historical_rf_fit_s}} s, ExtraTrees {{fact:historical_et_fit_s}} s and tuned Bagging {{fact:historical_bt_fit_s}} s. They are workload- and machine-dependent saved timings. Their source evidence is [historical broader results](<experiments/results.json>) and [historical three-family results](<experiments/phase_three/results.json>), not the current all-five result file.

# Appendix B. Original requirement traceability

The matrix below retains requirement classifications, source document/paragraph locators, evidence and current status. Exceptions are visible rather than converted to fabricated technical completion.

{{table:requirements}}

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
