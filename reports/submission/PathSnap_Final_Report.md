# PathSnap — Technical Report

## 2. Dataset Understanding and Exploratory Data Analysis

| Partition | Trips | GPS observations | Nodes / directed arcs |
| Training | 80 | 208,417 | 1,933,485 / 3,876,904 |
| Student testing | 10 | 19,160 | 90,689 / 185,041 |
| Evaluator testing | 10 | 19,674 | Final scoring only; not used for training |
**Preprocessing.** Each trip has a .track (longitude, latitude, timestamp), .nodes (longitude, latitude), .arcs (tail, head) and .route (ordered directed arc IDs). Arc IDs are zero-based row indices local to a trip. Every file was parsed and checked for finite values, valid coordinates/endpoints and ordered timestamps. No missing, NaN or nonfinite fields were found, so no imputation was needed. Local equirectangular projection converts coordinates to metres; repeated GPS rows, parallel arcs, cycles and reference order are preserved.
| Quality check | TRAIN80 | Student10 |
| Missing files / blank rows / invalid coordinates | 0 / 0 / 0 | 0 / 0 / 0 |
| Nonpositive time intervals / isolated nodes | 0 / 0 | 0 / 0 |
| Median / longest GPS gap | 1 s / 153,064 s | 1 s / 210 s |
| Repeated-coordinate GPS rows / broken reference joins | 572 / 13 | 6 / 1 |
**Patterns.** Most intervals are one second, but long gaps and uneven trip sizes rule out assuming uniform motion or randomly splitting GPS rows. Median trip sizes are 1,529 TRAIN and 1,420 student observations. Trip networks may have several directed components (median strong-component counts 127 and 53.5 respectively); disconnected references and nearby parallel roads limit label certainty.
**Features and targets.** The base 38 features describe distance/rank, projection, arc length, heading/reliability, neighbouring offsets, motion/gap context and endpoint degree. The final 70-feature variant adds stateless geometry-energy, rank and best-rival-margin transforms. They distinguish close parallel/opposite roads using motion and connectivity. Baseline candidate pools start at 12; the final route-tuned pipeline uses 24, retry 48 on reachability failure, within 80 m. References supply route sequences, not per-GPS labels: training labels use the closest reference-route segment plus a weak heading term. All observation scores therefore measure this proxy, not independent GPS truth. LR/KNN/GaussianNB scaling is learned inside each TRAIN fold.
Measured TRAIN/student trip-size and timestamp patterns. A raw GPS trajectory and its actual road network appear in the layered Folium map in Section 6.

![Measured TRAIN/student trip-size and timestamp patterns. A raw GPS trajectory and its actual road network appear in the layered Folium map in Section6.](<../../reports/submission/figures/eda.png>)

<font color="#486879"><link href="https://github.com/DevaOnL/ML-Hackathon/blob/feat/train-only-accuracy-improvement/mapmatch/features.py">Source: Data and feature implementation</link></font>

## 3. Model Experimentation and Comparison

Baselines use three-fold whole-trip CV on TRAIN 80 (350 observations/trip, 28,000 total). The final 70-feature model uses the separately fixed TRAIN 64 CV; its complete route decoder is chosen by equal-trip route-edge F1 on all 64 dense held-out trajectories. All student results use 19,160 observations. Historical student development is not blind; no evaluator data choose settings. All tested families and tuned variants are retained.

### 3.1 Overall model comparison

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route continuity | Student-Test Macro F1-score | Log Loss | Inference Time |
| Logistic Regression | linear | C=1;500 iterations | 0.9330 | 89.52% | N/A: no route | 0.9329 | 0.0690 | 0.026s |
| Distance-weighted KNN | instance | k=25;distance weights;60k bank | 0.9275 | 89.05% | N/A: no route | 0.9275 | 0.0960 | 187.078s |
| Bagging | ensemble | 40 trees;depth14;leaf3;sample0.7 | 0.9426 | 90.48% | N/A: no route | 0.9437 | 0.0562 | 0.872s |
| Bagging tuned | ensemble | 48 trees;depth13;leaf2;sample0.7161 | 0.9426 | 90.34% | N/A: no route | 0.9436 | 0.0560 | 0.938s |
| GaussianNB | probabilistic | smoothing10⁻⁷ | 0.9211 | 88.84% | N/A: no route | 0.9222 | 0.3591 | 0.105s |
| Random Forest | ensemble | 100 trees;depth16;leaf4;features0.7 | 0.9427 | 90.50% | N/A: no route | 0.9446 | 0.0548 | 0.400s |
| Extra Trees | ensemble | 100 trees;depth22;leaf3;features0.8 | 0.9429 | 90.40% | N/A: no route | 0.9422 | 0.0556 | 0.491s |
| HistGB | boosting | 120 rounds;rate0.09;23 leaves | 0.9429 | 90.66% | N/A: no route | 0.9442 | 0.0552 | 0.577s |
| HistGB tuned | boosting | 104 rounds;rate0.1590;22 leaves | 0.9416 | 90.68% | N/A: no route | 0.9447 | 0.0555 | 0.499s |
| HistGB final (70F) | boosting | 70F;180 rounds;31 leaves | 0.9475 | 91.62% | N/A: no route | 0.9534 | 0.0299 | 1.637s |
CV/Student Macro F1 is a **binary candidate-row** metric at threshold 0.5; binary log loss uses unthresholded clipped probabilities. Top-1 is pooled **observation** accuracy; all candidate misses count wrong. Standalone classifiers do not reconstruct routes, so route continuity is N/A. Timing measures the student probability call only, excluding features/routing; retained timings are not repeated hardware-normalised benchmarks.
| Model | Binary accuracy | Positive precision / recall | Observation edge Macro F1 | Observation NLL | Raw adjacency | Fit time |
| Logistic Regression | 0.9737 | 0.8713 / 0.8902 | 0.8071 | 0.3476 | 92.87% | 1.02s |
| Distance-weighted KNN | 0.9724 | 0.8895 / 0.8522 | 0.7955 | 0.5217 | 92.30% | 0.02s |
| Bagging | 0.9783 | 0.9084 / 0.8910 | 0.8271 | 0.2824 | 93.22% | 103.76s |
| Bagging tuned | 0.9783 | 0.9071 / 0.8917 | 0.8248 | 0.2817 | 93.13% | 133.06s |
| GaussianNB | 0.9693 | 0.8465 / 0.8774 | 0.7933 | 0.9260 | 92.29% | 0.17s |
| Random Forest | 0.9787 | 0.9091 / 0.8935 | 0.8284 | 0.2807 | 93.31% | 292.54s |
| Extra Trees | 0.9778 | 0.9051 / 0.8889 | 0.8271 | 0.2882 | 93.08% | 62.15s |
| HistGB | 0.9783 | 0.8979 / 0.9034 | 0.8306 | 0.2791 | 93.35% | 5.35s |
| HistGB tuned | 0.9785 | 0.8985 / 0.9046 | 0.8346 | 0.2767 | 93.43% | 4.40s |
| HistGB final (70F) | 0.9884 | 0.9186 / 0.9075 | 0.8565 | 0.2495 | 93.50% | 8.11s |
Observation Macro F1 uses trip-qualified directed-road classes. Observation NLL normalises positive candidate scores per GPS row and floors absent targets at 10⁻⁷; it is not binary log loss. Raw adjacency checks same-edge/head-to-tail neighbours without inserting connectors. Candidate misses: baseline TRAIN 578/28,000; student 19/19,160. Final K24 student 12/19,160=0.0626%. Full-route scores appear in Section 5.
Final 70F CV: TRAIN 64/22,400 observations, K12; final student metrics: all 19,160 observations, K24 (retry 48). This wider candidate policy is explicit; older rows retain their original 80-trip CV and K12 classification metrics. Route continuity for the actual final pipeline is 1.0000 (Section 5), not classifier-only adjacency.
<font color="#486879"><link href="https://github.com/DevaOnL/ML-Hackathon/blob/feat/train-only-accuracy-improvement/experiments/target96/student_comparison/summary.json">Source: Final comparison data</link></font>

## 3.2 Results and interpretation for each model

Student results below use observation-level road classes; binary scores are reported separately in Section 3.1.

### Logistic Regression

A fast scaled linear baseline tests whether distance, heading and local context already separate correct candidates. Its limited nonlinear interactions are the main restriction. Initial settings: C=1, L2 regularisation, L-BFGS, 500 iterations; weighted binary log loss is minimised. Student Top-1 89.520%; observation edge Macro F1 0.8071; observation NLL 0.3476. Training Top-1 88.432% versus grouped-CV 88.400%. The almost equal training/CV scores show little memorisation; its lower CV score than trees suggests limited model capacity.

### Distance-weighted KNN

Neighbour voting can model local nonlinear candidate similarity without assuming a functional boundary. It requires scaling and is costly for a large candidate matrix. Initial settings: k=25, Euclidean distance and inverse-distance votes; a weighted class-stratified 60,000-row TRAIN bank bounds memory. There is no iterative training loss; k and the bank determine the estimate. Student Top-1 89.050%; observation edge Macro F1 0.7955; observation NLL 0.5217. Training Top-1 90.768% versus grouped-CV 87.679%. The 3.09-point train/CV gap indicates overfitting; 187.08 s student inference is unsuitable compared with historical 38-feature HistGB's0.50 s.

### Bagging

Bootstrapped decision trees capture nonlinear interactions and reduce single-tree variance, at greater fit cost. Initial settings:40 trees, max_depth 14, min_samples_leaf 3, max_samples 0.7; trees minimise weighted Gini impurity and their probabilities are averaged. Student Top-1 90.480%; observation edge Macro F1 0.8271; observation NLL 0.2824. Training Top-1 94.064% versus grouped-CV 89.146%. The 4.92-point gap suggests overfitting. A Bayesian variant used 48 trees, depth 13, leaf 2 and sample 0.7161: CV Top-1 improved 89.146%→89.250%, but student Top-1 fell 90.480%→90.339% (edge Macro F1 0.8248; NLL 0.2817), so it was not the final model.

### Gaussian Naive Bayes

A cheap probabilistic baseline tests whether per-feature class distributions suffice. Correlated geometric features violate conditional independence and can make probabilities overconfident. Initial var_smoothing=10⁻⁷; weighted Gaussian class means/variances are fitted by likelihood, without gradient training. Student Top-1 88.836%; observation edge Macro F1 0.7933; observation NLL 0.9260. Training Top-1 87.400% versus grouped-CV 87.343%. The small train/CV gap is consistent with limited capacity; binary student log loss 0.3591 is much worse than boosted trees 0.0552.
<font color="#486879"><link href="https://github.com/DevaOnL/ML-Hackathon/blob/feat/train-only-accuracy-improvement/mapmatch/phase_models.py">Source: KNN/Bagging training and bank policy</link></font>

## 3.2 Results and interpretation for each model (continued)


### Random Forest

Bagged trees with feature subsampling capture nonlinear distance/heading interactions while reducing correlated-tree variance. Initial settings:100 trees, depth 16, min_samples_leaf 4, max_features 0.7; weighted Gini splits with averaged probabilities. Student Top-1 90.496%; observation edge Macro F1 0.8284; observation NLL 0.2807. Training Top-1 95.289% versus grouped-CV 89.139%. The 6.15-point training/CV gap is larger than HistGB's; higher training accuracy did not justify selecting it. Student binary log loss 0.0548 is good, but its CV Top-1 is below HistGB.

### Extra Trees

Randomised split thresholds provide a fast nonlinear ensemble and an alternative variance/bias tradeoff. Initial settings:100 trees, depth 22, min_samples_leaf 3, max_features 0.8, bootstrap=False; random candidate splits are compared by weighted Gini impurity. Student Top-1 90.397%; observation edge Macro F1 0.8271; observation NLL 0.2882. Training Top-1 96.961% versus grouped-CV 89.182%. The 7.78-point training/CV gap is the largest among these baselines; stronger memorisation did not improve student Top-1 over Bagging/HistGB.

### HistGradientBoosting

Sequential trees optimise probability estimates for interacting distance, heading and motion features; histogram splits keep fitting/inference practical. Initial settings:120 iterations, learning_rate 0.09,23 leaves, min_samples_leaf 40, L2=0.5, early_stopping=False. The loss is weighted binary log loss. Student Top-1 90.658%; observation edge Macro F1 0.8306; observation NLL 0.2791. Training Top-1 90.850% versus grouped-CV 89.268%. Bayesian tuning gives 104 iterations, rate 0.159024,22 leaves, leaf minimum 38, L2=0.860422. Tuned student results are Top-1 90.678%, edge Macro F1 0.8346 and NLL 0.2767. Its training/CV gap rises from 1.58 to 2.06 points: tuning does not prove reduced overfitting.

### Final route-tuned HistGB

The final 70-feature model uses 180 rounds, rate 0.1, 31 leaves, leaf minimum 30 and L2=2; objective remains weighted binary log loss. All 80 TRAIN trips provide the final fit. Its Top-1 is 92.957% in training versus 89.402% in TRAIN 64 CV, so overfitting remains visible. Student pooled Top-1 is 91.618%, observation edge Macro F1 0.8565 and NLL 0.2495. This classifier with the route-tuned HMM reaches 98.212% route-edge F1; classifier accuracy is a separate metric.
None of these estimators recorded epoch-wise train/validation histories; no artificial training curves are supplied. Actual cross-model plots follow. All model configurations/trials remain available in the repository.
<font color="#486879"><link href="https://github.com/DevaOnL/ML-Hackathon/blob/feat/train-only-accuracy-improvement/mapmatch/experiments.py">Source: Initial estimators and loss implementation</link></font>

## 3.3 Measured model visualisations

All trained algorithm families and tuned configurations. Baseline CV uses TRAIN 80; final 70F CV uses TRAIN 64 (K12). Final student results use K24, older rows use their recorded candidate pools; all use the same 10 trips/19,160 GPS rows. Macro F1/log loss are binary candidate-row metrics.

![All trained algorithm families and tuned configurations. Baseline CV uses TRAIN80; final70F CV uses TRAIN64 (K12). Final student results use K24, older rows use their recorded candidate pools; all use the same10 trips/19,160 GPS rows. Macro F1/log loss are binary candidate-row metrics.](<../../reports/submission/figures/model_comparison.png>)


### Other tested feature configurations

A later TRAIN-only search tested compact 12-, physics 3- and context 70-feature variants of the five main families. The following candidates were fitted on the same 64 trips as their controls and scored once on 16 internally reserved TRAIN trips. Earlier releases had used all 80 trips; these are internal comparisons, not blind/student-test scores.
| Family | Selected profile / configuration | Control Top-1 | Candidate Top-1 |
| Logistic Regression | compact;C=1 | 91.607% | 91.768% |
| KNN | compact;k=35,p=2 | 90.589% | 91.750% |
| Bagging | context;64 trees,depth18,leaf6 | 92.161% | 92.696% |
| GaussianNB | compact;smoothing0.1 | 90.554% | 91.089% |
| HistGB | context;180 rounds,rate0.1,31 leaves | 92.054% | 93.018% |
The objectives are unchanged. LR/GaussianNB binary Macro F1 decreased despite higher Top-1; separate full-student scoring for those four candidates is N/A (not run). The earlier context HistGB with fixedK 12 failed 2/10 student routes (edge F1 0.7611), so that pipeline was rejected. The final model is the separately frozen route-tuned pipeline in Sections 4–5; prior failed results remain unchanged.
| Full-route student metric | Previous final | New final |
| Equal-trip route-edge F1 | 95.136% | 98.212% |
| Length-weighted F1 | 0.9794 | 0.9884 |
| Continuity / failed trips | 1.000 /0 | 1.000 /0 |
<font color="#486879"><link href="https://github.com/DevaOnL/ML-Hackathon/blob/feat/train-only-accuracy-improvement/reports/accuracy_improvement/accuracy_results.json">Source: Additional tested configurations and confirmation results</link></font>

## 4. Bayesian Hyperparameter Optimisation

The baseline HistGB Bayesian search used a Gaussian-process Matérn-5/2 surrogate and expected improvement. Its objectives used three-fold whole-trip CV on TRAIN 80, maximising equal-trip observation Top-1. This favours the correct road for every trip rather than the many negative candidate rows. Student/evaluator scores were not search objectives.
| Hyperparameter | Range / distribution | Reason |
| learning_rate | 0.025–0.2;log-uniform | Explore small and large boosting steps |
| max_iter | 70–200;integer | Bound capacity and fitting cost |
| max_leaf_nodes | 7–47;integer | Control tree complexity |
| min_samples_leaf | 15–160;log-integer | Regularise rare/noisy candidates |
| l2_regularization | 0.01–20;log-uniform | Regularise leaf values |
**Budget and best values.**12 recorded trials comprise one measured baseline, three seeded initial designs and eight model-based proposals. Search time was 162.61 s. The best Bayesian baseline configuration is 104 iterations, rate 0.1590239,22 leaves, minimum leaf 38 and L2=0.8604220; it was refitted on all 80 TRAIN trips.
| Same-data metric | Untuned | Tuned |
| TRAIN CV Top-1 | 0.8927 | 0.8934 |
| TRAIN CV binary Macro F1 | 0.9429 | 0.9416 |
| Student observation Top-1 | 0.9066 | 0.9068 |
| Student observation edge Macro F1 | 0.8306 | 0.8346 |
| Student binary Macro F1 | 0.9442 | 0.9447 |
| Student binary log loss | 0.0552 | 0.0555 |
| Student observation NLL | 0.2791 | 0.2767 |
| Student probability-call time | 0.577s | 0.499s |
| All-TRAIN fit time | 5.350s | 4.396s |
**Did tuning help?** CV Top-1 rises by 0.071 percentage points; student Top-1 rises by 0.021 points and inference is faster. Student edge/binary Macro F1 and observation NLL improve slightly. CV binary Macro F1 and student binary log loss worsen. The gain is modest and does not justify an across-the-board improvement claim.
**Final route tuning.** A separate TRAIN-only search optimised route-edge F1:12 joint settings on 18 OOF trips; then 9 precision settings on 18, advancing two plus a paired control to all 64 dense OOF trips. Candidate width 12/24, beta 2/5/10, geometry 0.1/0.25/1, reverse cost 0/1/3/8 and appended-arc cost 0/0.25/0.5/1/2/5 were tested in fixed menus. Selected:K24, beta 5, geometry 0.25, reverse 3 at GPS chord≤20 m and appended-arc cost 5. All 64 edge F1 is 0.8494→0.9103, with the same 3 failed TRAIN routes scored zero in both means. The final 10-trip comparison was run after freezing this choice; it is development evidence, not a new blind test.
<font color="#486879"><link href="https://github.com/DevaOnL/ML-Hackathon/blob/feat/train-only-accuracy-improvement/experiments/final_five/current/bayesian_optimization.json">Source: Search ranges, trial objectives and best values</link></font>

## 5. HMM and Viterbi Route Reconstruction

**Hidden states.** Each state is a candidate directed arc and the GPS point's projected fraction along it. Initial candidates are the closest 24 within 80 m; an empty set expands the radius, up to 1000 m. Every original GPS row is decoded.
**Emissions.** The selected classifier's clipped positive scores p are normalised within each observation, q=p/Σp. A candidate's final emission cost is E=0.25×½(d/σ)²−log(q), combining lateral distance d with learned compatibility; σ=7.16065 m. These are HMM emission potentials, not an HMM posterior probability.
**Transitions and Viterbi.** Let D be directed shortest-metre travel distance, g the GPS chord, n the exact number of arcs reconstruction appends, and R an endpoint-opposite edge switch with g≤20 m. Final cost T=|D−g|/5+5n+3R; unreachable transitions remain infinite. Vₜ(j)=Eₜ(j)+minᵢ[Vₜ₋₁(i)+Tᵢⱼ] stores backpointers, then reconstructs those same shortest-metre graph arcs. Finite arc/reversal costs discourage short excursions before decoding; no output arcs are removed.
**Gaps and disconnected segments.** At a gap, transitions still use observed displacement and directed reachability; heading windows stop across gaps above 20 s. Routing starts at the GPS chord plus 300 m slack; if all transitions fail, the search expands to at least 5000 m or ten times that horizon. Continued reachability failure widens candidate width 24→48 and recomputes candidate-relative features/emissions. If no connected sequence exists, inference reports failure rather than teleporting or fabricating edges. Real cycles and repeated traversals remain in the output.
**Measured effect.** Both methods below use the same final candidate sets and classifier probabilities on all 10 student trips/19,160 observations. Classifier-only picks each local argmax, collapses adjacent equal arcs and adds no connecting paths. Route F1 compares the full supplied reference sequence; repeated arcs count. Values are equal-trip means.
| Student metric | Classifier only | Classifier + HMM |
| Observation proxy accuracy | 0.9283 | 0.9495 |
| Observation edge Macro F1 | 0.8578 | 0.9445 |
| Directed-arc route F1 | 0.7645 | 0.9821 |
| Length-weighted route F1 | 0.8171 | 0.9884 |
| Sequence edit similarity | 0.6286 | 0.9721 |
| Route continuity | 0.7571 | 1.0000 |
| Mean prediction time/trip | 0.164s | 6.097s |
The final pipeline improves route-edge F1 from 95.136% to **98.212%** on the same 10 trips/19,160 GPS rows, with zero failures and 100% continuity. Length F1 rises 0.9794→0.9884; edit similarity 0.9172→0.9721. This is route F1, not classification accuracy. Every trip improves pointwise versus its own raw classifier, but improvement is not universal across models: original trip 00000007 proxy accuracy fell 97.267%→96.307% with HMM while its route F1 improved. The reference break on trip 00000076 is retained.
**Frozen evaluator results.** On 10 trips/19,674 observations, edge F1 0.9849, length F1 0.9844, edit similarity 0.9743 and continuity 1.0000, with zero failures. Every prediction was sealed before references opened; no evaluator score changed the model. These are local results, not organiser-certified scores.
Classifier-only timing is the probability call; HMM timing includes full feature generation, candidates and routing. Their difference is not isolated DP overhead.
<font color="#486879"><link href="https://github.com/DevaOnL/ML-Hackathon/blob/feat/train-only-accuracy-improvement/experiments/target96/student_comparison/summary.json">Source: Paired classifier/HMM experiment</link></font>

## 6. Geographical Visualisation Using Folium

Student trip 00000009 shows original GPS observations, its supplied directed road graph, the reference route and the submitted model's predicted route. The screenshot is exported from the actual interactive Folium map; no road geometry or prediction was drawn manually.
Folium map of student trip 00000009. Road/GPS/reference/prediction layers use distinct colours, labelled controls and an automatically fitted extent.

![Folium map of student trip00000009. Road/GPS/reference/prediction layers use distinct colours, labelled controls and an automatically fitted extent.](<../../visualizations/target96/accepted_development_trip_00000009.png>)

The map contains 1,049 GPS points, 6,745 nodes and 13,106 directed arcs. The accepted prediction has 433 original arc traversals versus 429 in the reference; repeated occurrences remain. It is connected. Directed-arc F1 is 0.9768, length-weighted F1 0.9766 and sequence edit similarity 0.9654. Its exact route/model/policy hashes match the final comparison.
The layer control lets the reader isolate GPS, network, reference and prediction; the legend identifies each. Clickable features show their original IDs. This is a development/reference map; evaluator reference geometry is not delivered to the browser.
**Interactive map:** <link href="https://github.com/DevaOnL/ML-Hackathon/blob/feat/train-only-accuracy-improvement/visualizations/target96/accepted_development_trip_00000009.html">Folium HTML in the repository</link>. **Generator:** <link href="https://github.com/DevaOnL/ML-Hackathon/blob/feat/train-only-accuracy-improvement/scripts/generate_target96_map.py">map generator</link>.
<font color="#486879"><link href="https://github.com/DevaOnL/ML-Hackathon/blob/feat/train-only-accuracy-improvement/reports/target96/map_evidence.json">Source: Map source and measured route statistics</link></font>

## 7. API, UI and Deployment

**Repository:** <link href="https://github.com/DevaOnL/ML-Hackathon">github.com/DevaOnL/ML-Hackathon</link>. **Application:** <link href="https://ml.d3va.dev">ml.d3va.dev</link>. Laptop 1 runs FastAPI, the saved HistGB classifier, road graphs, Viterbi and private reference scoring. Laptop 2 opens the browser UI, selects one of ten trips and displays the returned GPS/road/prediction layers and scores. No model training or truth file is sent by the UI.
**Request flow.** GET /health checks readiness; GET /datasets lists available partitions; GET /datasets/{id}/trips lists exact trip IDs. POST /datasets/{id}/match receives only {trip_id}. The server loads graph/track, generates candidates, predicts and decodes, then separately reads the reference to score. The response contains route geometry, metrics and timing. The browser clears stale results when changing trips/datasets; references remain server-side.
Working UI: dataset/trip selection, reconstructed route and measured scores. These are screenshots of the submitted application.
![Working UI controls](<../../reports/target96/live/report_images/development_dataset_controls.png>)
![Working route map](<../../reports/target96/live/ui_route_map.png>)
![Measured UI scores](<../../reports/target96/live/report_images/development_scores.png>)
![Actual HTTP200 response on the receiving host](<../../reports/target96/terminal/api_response.png>)
Receiving-host terminal: an actual HTTPS POST for student trip 00000009 returns HTTP 200, route scores and timings. Selected JSON fields are shown without changing their values; geometry is omitted for readability. This verifies the real response on the API host, not a physical two-laptop demonstration.
**Deployment.** The existing HTTPS tunnel forwards to the API on 127.0.0.1:8000; the same service serves the UI. Dependencies and run commands are in README. The selected model/config are loaded once; inference is limited to one request at a time and predicted routes are not response-cached. All ten student and ten evaluator trips are supported. The existing service now loads the accepted 70-feature HistGB with the frozen route-tuned policy; health and public-response geometry match its exact model and route hashes. The earlier fixedK 12 failed pipeline remains rejected.
