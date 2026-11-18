# Original requirements and final evidence

The five original organizer DOCX files were independently re-extracted, including nested tables, and matched their prior XML extracts exactly. The sixth synthesis reproduces the five originals’ 480 nonempty paragraphs in order and adds no requirement. [Source provenance](<../analysis/final_requirements/provenance_summary.md>) records document hashes and locator conventions. The original gap audit is preserved as [initial review](<../analysis/final_requirements/initial_review.md>) and [initial matrix](<../analysis/final_requirements/initial_requirements_matrix.json>); it is historical, not a description of the finished five-family stage.

The live [47-row requirements matrix](<../analysis/final_requirements/requirements_matrix.json>) retains every source locator and classification. The Lead advances a row only when its named evidence exists; pending final report/repository rows are resolved after their actual checks. This is AI-assisted technical traceability, not an organizer’s certification.

## Five models, selection and tuning

Models P12/P54/P73, Deliverables P64 and Assessment P19 require at least five trained/tested models. The final scope includes Logistic Regression, distance-weighted KNN, Bagging, GaussianNB and HistGradientBoosting. The [acceptance](<../reports/final_five/all_five_model_acceptance.json>) and [independent current-model review](<../analysis/final_model_review/final_review_report.json>) verify genuine estimators, all 80 TRAIN trips, the same three whole-trip folds and all ten student-development trips. Verified first-three-family fits are reused honestly; GNB/default HistGB/tuned HistGB are fresh final-stage fits.

The [five-family result record](<../experiments/final_five/current/results.json>) contains configurations, candidate metrics, fold stability and measured costs. The primary family-selection objective is TRAIN equal-trip mean observation top-1 proxy accuracy. The [12-trial HistGB Bayesian study](<../experiments/final_five/current/bayesian_optimization.json>) is previously measured work reused after trial/fold/configuration validation, followed by fresh default/tuned CV and all-80 refitting. It is not a new set of twelve trials. Neither student nor evaluator results select the model.

Deliverables P65 also requires every actually experimented model to be documented. Earlier RandomForest/ExtraTrees and tuned Bagging results belong in a clearly separated historical appendix, not hidden or counted as extra final-scope families. The [historical results](<../experiments/results.json>) and immutable [three-family report](<../reports/phase_three/PathSnap_Technical_Report.pdf>) preserve them.

## Plots and training histories

Deliverables P139–P147 requires CV macro F1, student top-1/macro F1, applicable log loss and selection-justifying comparison plots. The [final figure manifest](<../visualizations/final_five/figure_manifest.json>) and [asset checks](<../visualizations/final_five/asset_verification.json>) bind actual model-comparison and decoder-ablation plots to measured inputs. Twelve original EDA figures are reused read-only.

Epoch-wise training/validation loss plots apply only when a model produces such a history. None of these five estimator pipelines records an epoch-wise history, so it is stated N/A for each. Resubstitution/CV/student loss and measured HPO trajectories are provided; they must not be represented as artificial epoch curves or causal hyperparameter-importance estimates.

## HMM, routes and reference isolation

The [decoder freeze](<../reports/final_five/decoder/freeze.json>) follows a preregistered three-configuration search on eight TRAIN trips. Because the model already fitted all TRAIN80, this is in-sample calibration. [Classifier-only versus HMM ablation](<../reports/final_five/decoder/student_ablation.json>) covers all 19,160 student observations. [CLI acceptance](<../reports/final_five/offline_delivery_verification.json>) verifies connected directed routes, chronological observation states and exact route/hash parity for all ten.

The new evaluator inputs were parsed only after model and decoder freezes and [preregistration](<FINAL_EVALUATOR_PROTOCOL.md>). All 19,674 observations were predicted before references were parsed. [Evaluator acceptance](<../reports/final_five/evaluator/acceptance.json>) retains 190 checks and [all per-trip metrics](<../reports/final_five/evaluator/route_metrics.json>), including weak trip 15 and zero exact routes. Label-derived observation metrics remain proxies. Directed-arc route metrics preserve traversal multiplicity. No match against TRAIN/student IDs or per-extension file contents proves only that specific audit, not geographic independence or official unseen status.

## Website and deployment

The user requested the public [ml.d3va.dev](https://ml.d3va.dev) website instead of the brief’s physical two-laptop arrangement. The existing dedicated service/tunnel is reused, not replaced. [Activation evidence](<../reports/final_five/deployment_activation.json>) pins the selected HistGB hash, both ready ten-trip sections and the six allowlisted runtime-setting changes. [Actual twenty-trip API acceptance](<../reports/final_five/api_integration_acceptance.json>) checks model probabilities, reference isolation and CLI parity. [Dataset contract](<EVALUATOR_SECTION_PLAN.md>) describes separate registries and shared model/configuration/inference slot.

[OpenCode](https://opencode.ai/) was visited only after all-five training/testing passed. The [reference record](<OPENCODE_DESIGN_REFERENCE.md>) and [website design](<WEBSITE_DESIGN.md>) document the original PathSnap theme, local Leaflet, dark/light preference, keyboard controls, strict dataset identity and stale-response handling. The final report must include the actual UI and receiving-terminal screenshots with real request IDs, not fixture pages or typed sample logs.

## Human and organizer responsibilities

These cannot be completed by a remote coding agent:

- A physical two-laptop venue demonstration, if organizers insist despite the user-authorized public-site substitute. The [two-laptop instructions](<../deploy/TWO_LAPTOPS.md>) remain available.
- Each participating member’s viva and ability to explain the data, features, models, tuning, decoder, metrics and deployment. No source requires a particular five-person team size.
- Organizer plagiarism clearance, permission for AI assistance, cross-team independence checks and authorship declarations. [Development provenance](<DEVELOPMENT_PROVENANCE.md>) discloses assistance and administrative account use; repository activity is not independent human review.
- Confirmation of evaluation scheduling. The before-lunch notice has no calendar date in the source.

[Delivery guide](<FINAL_DELIVERY_GUIDE.md>) gives the demo procedure and private-file boundaries. No final software/report acceptance is a claim that those human or organizer tasks have occurred.
