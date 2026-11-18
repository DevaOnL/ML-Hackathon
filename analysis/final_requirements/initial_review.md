# PathSnap original requirements: status and gaps

This report is the independent audit of what the five original organizer documents actually require, measured against the workspace as of 18 November 2026. Sources are the five DOCX files in PathSnap_student_materials/; every claim below carries a locator into analysis/final_requirements/requirements_matrix.json, which holds the full 47-row matrix. The source text was re-extracted from the DOCX binaries and matched the existing extracts exactly, and the sixth "Complete Guidelines" file was confirmed to be the five documents concatenated, so nothing in it changes this audit.

## Where things stand

| Area | Requirements | State in one line |
|---|---|---|
| Dataset and partitions | R-DS-01 to R-DS-09 | Complete with evidence; carried forward into the final report |
| Five-model experimentation | R-MT-01 to R-MT-08 | Three families done, two pending the final-five audit |
| HMM and Viterbi | R-HM-01 to R-HM-03 | Done for the frozen Bagging model; re-freeze needed for the final winner |
| System and deployment | R-SY-01 to R-SY-05 | Public website replaces the second laptop by user order; evaluator section deferred |
| Report and repository | R-DL-01 to R-DL-13 | Phase-three PDF and repo merged; final-five PDF, charts and screenshots outstanding |
| Assessment and integrity | R-AP-01 to R-AP-05 | Mechanical parts upheld; viva and defence are human work |
| User-scope items | R-US-01 to R-US-04 | Five-family order active; OpenCode site sequenced after it |

## The five-model rule

The wording in the Models and Training Guidelines (P12) is "Teams must train and evaluate at leastFIVE different models before selecting their final model" (the missing space is in the original). The same floor appears at Models P54 and P73, Deliverables P64, and Assessment P19, so it is mandatory in four separate places. The consolidated report table (Deliverables P66 to P121) has columns for exactly five model rows and nine metrics per row.

What exists today: Logistic Regression, distance-weighted KNN and Bagging are trained, cross-validated on the 80 training trips, and evaluated on the ten student-testing trips, with a Bayesian search completed for Bagging. Gaussian Naive Bayes and HistGradientBoosting have historical artifacts fitted on all 80 trips with predictions for all ten evaluator trips. Those artifacts are with the model-audit child right now; until that audit confirms their identity and validity, no number from them should enter the final table. The matrix marks this as partially done, and the honest sentence for the Lead is: the project currently has a three-model claim with full evidence, and a five-model claim only after GNB and HistGB are either adopted from verified artifacts or re-run under the final protocol.

Multiple algorithm families are "strongly encouraged" (Models P10), and the family tables at Models P15 to P51 are labelled suggestions rather than a fixed list. The five chosen families span that table: linear (LR), probabilistic (GNB), instance-based (KNN), bagging ensemble, and boosting. The five-family scope also matches the user's latest order, which supersedes the earlier three-family deferral recorded in PHASE_SCOPE.json.

Selection must weigh generalisation, fold stability, probability quality, inference speed and decoder compatibility, and "Do not select a model solely because it achieves high training accuracy" (Models P92). The phase-three selection followed that rule, using TRAIN-only grouped CV, and the same analysis has to be repeated across all five before the final freeze.

## Training curves, stated honestly

The user asked about training loss and learning curves. The report requirements (Deliverables P143 and P147) ask for training and validation loss plots "only for models that produce epoch-wise training histories", and forbid artificial training curves for algorithms that do not produce training histories. None of the five families in scope produces epoch-wise histories under the current sklearn pipeline, so the correct treatment is a stated N/A with the reason, per model, in the report. Fabricating learning curves to fill the section would violate Deliverables P145 and the authenticity rules in Assessment P36. What the documents do require for every model are the comparison charts: CV Macro F1 across models, student-testing Top-1 and Macro F1, Log Loss where applicable, and selection-justifying comparison plots (Deliverables P139 to P146). Those charts exist for three families and must be regenerated for all five.

## Bayesian hyperparameter optimisation

The full protocol (Models P97 to P105, Deliverables P150 to P159, Assessment P20 and P21) asks for the search space and its justification, the objective, group-aware CV on the training partition, the trial count, best values, a tuned-versus-untuned comparison on identical partitions, and the retained search record. The Bagging search meets this: 8 trials, TRAIN-only objective, configs, trial log and objective CSV preserved in reports/phase_three/decoder/. A historical 12-trial TRAIN-only search exists for HistGradientBoosting; the model-audit child must verify its trial study before any number is reused. If the five-family comparison selects a model whose search was never done, a fresh Bayesian search for that model becomes a blocking item before the final report.

The evaluator partition must stay out of every optimisation choice (Models P106). This has held so far and needs a final check at the freeze.

## The negative control and the decoder

The phrase "negative control" does not appear in any of the five documents. The closest requirement, and the one the user's question maps to, is the classifier-only versus classifier-plus-HMM comparison (Models P122, Deliverables P169 to P172): pointwise accuracy of raw classifier predictions against decoded routes, with a discussion of where decoding helped and where it did not. The phase-three ablation covers this for Bagging. It has to be re-run or re-verified for whichever model wins the five-family comparison, because the decoder consumes the selected model's probabilities (Models P107 to P111) and its parameters may only be tuned on training and student data (Models P124).

## Deployment and the laptop question

Yes, the brief mandates two laptops. Main Guidelines Table 3 (P20 to P29) and the final-evaluation steps (Models P146 to P158) put the model, HMM, road network and evaluation on Laptop 1 and the UI on Laptop 2, with the UI sending trip selections to the API and displaying returned routes and scores.

The user replaced the second laptop with a public website, and that replacement is deployed and verified at https://ml.d3va.dev: same-origin UI and API, loopback-bound API, reference data kept server-side, real HTTPS inference requests reconciled against CLI geometry in delivery_verification.json. The matrix records this as a user exception, and the report should state the substitution plainly rather than imply the original architecture was used. If the organisers insist on the literal setup at the venue, setting up and demonstrating two physical laptops is work only the team can do, and it is listed as a human external item. The API request logs the brief requires (Models P160) exist for the phase-three verification and will need regenerating from the final site.

The ten evaluation trips must be selectable end to end, with accuracy computed server-side and ground truth never reaching the model or the UI (Models P147 to P160, Assessment P57 and P58). That section stays deferred until after the final model and decoder freeze per the user's order; the new inputs have not been read, and that holds.

## Report, repository and the website

The consolidated PDF (Deliverables P43 to P49) must be one document per team whose every number is verifiable from the repository. The 23-page phase-three PDF meets this with 209 checked evidence facts. The final-five version does not exist yet and is the largest outstanding deliverable: the five-row comparison table, the GNB and HistGB question subsections, the all-five charts, the HPO section for the final model, the refreshed HMM ablation, and new UI and API screenshots. Those screenshots must come from the OpenCode website with the kind theme, which the user ordered only after all five models are trained and tested, so report section 7 evidence is sequenced behind it.

The repository side is in good shape: PR #5 merged to main carries all eleven required components from Deliverables Table 2, and the repo rules (all source committed, README, folders, no secrets) hold. A secret scan before the final merge is the one remaining check.

On the ZIP question: the string "zip" does not appear in any of the five documents. The archive arrives as a dataset and the EDA duties sit in Dataset Exploration sections 3, 4 and 8 plus Deliverables 2.2, all of which the data audit and the report already cover. There is no separate organiser requirement about exploring the ZIP container itself.

## What only the humans can do

These are disclosed rather than faked, per the audit's terms.

The viva. Assessment P41 to P54 lets organisers question any member on the dataset, preprocessing, features, model comparisons, selection, Bayesian HPO, the HMM, Viterbi, the API, the UI and the results. No document states a five-member team; the rule binds every member of whatever size the team is. An AI agent cannot make a person understand the work. The practical step is a team walkthrough of the final report and a live reproduction run from the repository before evaluation.

The plagiarism defence. The policy prohibits inter-team sharing and fabricated results (Assessment P26 to P40), and organisers will compare code, history, weights and reports across teams (Assessment P59 to P69). The workspace shows no inter-team sharing and the reported numbers all trace to measured artifacts. The work is AI-assisted at the owner's request, disclosed in docs/DEVELOPMENT_PROVENANCE.md, and that disclosure should stay in the submission. Claiming unaided human authorship would be the dishonest path.

The schedule. The Deliverables title table carries the note "There will be first phase evaluation before lunch break." No date or time appears anywhere in the materials. The user needs to confirm the event schedule so a submission is ready before that checkpoint; this audit will not guess it.

## Evidence the Lead must hold before building the final deliverables

1. Final-five model proof: for LR, KNN, Bagging, GaussianNB and HistGradientBoosting, the fitted artifact, configuration and seed, TRAIN grouped-CV metrics, student-test metrics and timing, all under one protocol. The GNB and HistGB rows depend on the model-audit child's verdict; nothing from the historical all-80 artifacts enters the table before that.
2. The training and comparison plots listed in Deliverables 3.3, regenerated from the five-model result files, with the epoch-wise loss plot question answered as a stated N/A per model.
3. HPO quantitative evidence for whichever model is finally selected: search space, objective, trial count, best values, the tuned-versus-untuned metric table on identical partitions, and the retained trial log. Bagging's search is complete; a different winner needs its own search.
4. The HMM ablation for the final model: classifier-only versus classifier-plus-HMM on the student partition, with continuity metrics.
5. Only after the freeze: the evaluator section, the OpenCode website with the kind theme, the regenerated request logs, and the final PDF and screenshots.

Until items 1 to 4 exist, any "five models" statement in the report or the website would be ahead of the evidence.
