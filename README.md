# PathSnap

A learned GPS-to-road map matcher with directed HMM/Viterbi reconstruction, a FastAPI service and an offline-capable browser map. Model inference uses graph/GPS features; private reference routes are opened only by a separate scorer after prediction.

- Website: [ml.d3va.dev](https://ml.d3va.dev)
- Repository: [DevaOnL/ML-Hackathon](https://github.com/DevaOnL/ML-Hackathon)
- Final measured route result: **98.212% directed-route edge F1**, up from **95.136%**, on the same ten student-development trips and19,160 GPS observations; zero failures and 100% continuity. This is route F1, not classification accuracy or blind-test evidence. [Measured comparison](<experiments/target96/student_comparison/summary.json>)
- Selected pipeline: [70-feature HistGB](<artifacts/target96/model.joblib>), [training metadata](<artifacts/target96/training_metadata.json>), [decoder](<artifacts/target96/match_config.json>), [arc policy](<artifacts/target96/arc_policy.json>), [model card](<artifacts/target96/MODEL_CARD.md>) and [acceptance](<artifacts/target96/acceptance.json>). The existing public service now serves this accepted release.
- Concise assignment report: [final PDF](<reports/submission/PathSnap_Final_Report.pdf>); exact new route submissions: [student development ZIP](<outputs/target96/candidate/submission.zip>) and [evaluator ZIP](<outputs/target96/evaluator/submission.zip>). Frozen evaluator route-edge F1 is **98.487%**, with zero failures. Historical five-family measurements and previous accepted artifacts remain below unchanged.

## Previous classification-accuracy study

A new TRAIN-only study improved all five families’ observation top-1 accuracy in paired internal confirmation: Logistic Regression **91.607% → 91.768%**, KNN **90.589% → 91.750%**, Bagging **92.161% → 92.696%**, GaussianNB **90.554% → 91.089%**, and HistGradientBoosting **92.054% → 93.018%**. Both controls and candidates fit on the same64 TRAIN trips; these16 reserved trips were historically exposed to earlier releases, so this is not blind-test evidence. Secondary metrics do not all improve.

The new HGB/context candidate also improved both route F1 metrics internally. Its final frozen student-development comparison had **2/10 route failures**, so it was **not accepted or deployed**. That failed fixed-K12 release remains rejected. The new route-primary study above subsequently met its target with a separately frozen decoder; historical artifacts below are preserved. [Measured accuracy addendum](<reports/accuracy_improvement/ACCURACY_REPORT.md>), [comparison CSV](<reports/accuracy_improvement/model_comparison.csv>), [frozen protocol](<experiments/accuracy_improvement/PROTOCOL.json>) and [release decision](<reports/accuracy_improvement/release_decision.json>) retain every tradeoff and the expedited Bagging budget deviation.

## Five trained and tested families

All five families have genuine fitted estimators, three-fold whole-trip TRAIN validation and testing on all ten student-development trips. The final stage preserves the completed Logistic Regression/KNN/Bagging fits and measurements, and adds fresh GaussianNB/default HistGB/tuned HistGB CV, all-80-trip fits and all-ten-trip tests. Tuned HistGB is a configuration of the fifth family, not a sixth family.

| Candidate classifier | TRAIN trip-macro CV top-1 | Student pooled top-1 | Student binary macro F1 | Student binary log loss |
|---|---:|---:|---:|---:|
| Logistic Regression | 88.400% | 89.520% | 0.93294 | 0.06900 |
| Distance-weighted KNN | 87.679% | 89.050% | 0.92749 | 0.09604 |
| Bagging, default | 89.146% | 90.480% | 0.94373 | 0.05622 |
| Gaussian Naive Bayes | 87.343% | 88.836% | 0.92221 | 0.35910 |
| HistGradientBoosting, default | 89.268% | 90.658% | 0.94423 | 0.05522 |
| HistGradientBoosting, tuned | **89.339%** | **90.678%** | **0.94472** | 0.05550 |

Family selection uses TRAIN equal-trip mean observation top-1 against route-derived proxy targets. Student scores do not choose the winner. The previously measured HistGB study contains one default prior, three seeded proposals and eight Gaussian-process expected-improvement proposals (12 total); its folds, parameters and objectives were revalidated, and fresh default/tuned CV reproduces them. No new HPO trials are implied. [Selection and reuse protocol](<experiments/final_five/PROTOCOL.md>), [model acceptance](<reports/final_five/all_five_model_acceptance.json>) and [independent current-model review](<analysis/final_model_review/final_review_report.json>) provide the evidence.

TRAIN uses 80 trips, 350 observations per trip, 269,463 candidate rows and 38 features. Student-development testing uses all 19,160 observations and 175,778 candidate rows. KNN uses a declared 60,000-row TRAIN-only weighted stratified reference bank with fold-local scaling. There are 578 TRAIN and 19 student proxy-target candidate misses, but no empty candidate sets. The student partition has historical development use and is not a globally blind test.

The first GNB full-fit scoring attempt correctly stopped when float32 class probabilities failed the sum-to-one guard. Only GNB scoring inputs were promoted to float64; strict probability checks were not relaxed. The final table uses the corrected fresh run. The incomplete earlier attempt remains separate. The selected HistGB is unaffected.

## Route reconstruction and isolated evaluator test

The final HistGB artifact was frozen before final decoder calibration. Three predeclared configurations were compared on eight TRAIN trips; because the classifier had been fitted on TRAIN80, this is explicitly in-sample decoder calibration. Selected scales are sigma 7.160645916267228 m and beta 2 m, with unit classifier/geometry weights. Directed no-teleport Viterbi uses widths 12 → 24 → 48 only when retry is needed. [Decoder freeze](<reports/final_five/decoder/freeze.json>) records the choice before student ablation or evaluator access.

| Fixed final decoder | Trips | Observations | Trip-macro directed-arc F1 | Trip-macro length F1 | Connected |
|---|---:|---:|---:|---:|---:|
| Student development | 10 | 19,160 | 0.95136 | 0.97941 | 10/10 |
| Newly supplied evaluator | 10 | 19,674 | 0.92264 | 0.96581 | 10/10 |

Student classifier-only length F1 is 0.82274; the historical geometry-HMM baseline is 0.97345. HMM sequencing provides the main improvement. Route F1 retains repeated directed-arc occurrences; it is not an unordered unique-edge score. Binary macro F1, observation-edge macro F1, categorical candidate NLL and route F1 are different metrics. Exact routes are 0/10 in each partition.

New evaluator graph/track parsing began only after model and decoder freezes and a written preregistration. All ten predictions finished and passed verification before any evaluator reference was parsed. No evaluator ID or per-extension input-content match was found against TRAIN80/student10; this does not prove geographic independence or official unseen status. The weakest evaluator trip, 15, has arc F1 0.57883 and length F1 0.77508. Its errors are retained. Full results and limitations: [evaluator metrics](<reports/final_five/evaluator/route_metrics.json>), [input audit](<reports/final_five/evaluator/input_audit.json>), [190-check acceptance](<reports/final_five/evaluator/acceptance.json>) and [preregistered protocol](<docs/FINAL_EVALUATOR_PROTOCOL.md>).

Predicted route archives contain exactly ten flat `.route` members each: [student routes](<outputs/final_student_test/submission.zip>) and [evaluator routes](<outputs/final_evaluator_test/submission.zip>). Raw inputs, references, row-level probability arrays, diagnostic arrays and observation CSVs are not redistributed.

## Install and run

Use Python 3.14 and Node 24 for the tested workflow. Exact measured versions are recorded with the experiments; selected-model strict replay was also checked in a separate clean environment.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -B -m unittest discover -s infrastructure -p test_hosting.py -v
node frontend/smoke.mjs
node --test tests/frontend_adapter.test.mjs
```

With authorized local inputs, the following legacy CLI reproduces the preserved38-feature baseline, not the new final70-feature pipeline. The accepted final pipeline is served with explicit target96 artifact paths (next block).

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m mapmatch predict \
  --input "Student folder/student_test" --output outputs/my_student_run \
  --model artifacts/final_five/model.joblib \
  --config artifacts/final_five/match_config.json
.venv/bin/python -m mapmatch evaluate \
  --input "Student folder/student_test" --predictions outputs/my_student_run \
  --output outputs/my_student_run/evaluation.json
```

Prediction never opens reference routes; evaluation is explicit and separate. Existing prediction directories are refused. [Complete final-five reproduction](<docs/FINAL_REPRODUCTION.md>) explains how to rebuild missing private inputs without changing measured evidence. [Post-measurement output-safety hardening](<docs/FINAL_REPRODUCTION_HARDENING.md>) records the maintained runner changes. New reproductions use a fresh `--run-id`, never the published result directory.

For local API execution of the accepted final pipeline, provide trusted dataset roots and explicitly select its artifacts:

```bash
PATHSNAP_MODEL=artifacts/target96/model.joblib \
PATHSNAP_MODEL_METADATA=artifacts/target96/training_metadata.json \
PATHSNAP_MATCH_CONFIG=artifacts/target96/match_config.json \
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

The backend and CLI default to the final-five artifact paths and fail rather than substituting a geometry-only model. The historical generic model loader is retained for explicit old-artifact replay. [API documentation](<deploy/API.md>), [deployment instructions](<deploy/DEPLOYMENT.md>) and [two-laptop alternative](<deploy/TWO_LAPTOPS.md>) describe configuration. Do not run a second server to update the existing public deployment; use its scoped lifecycle tooling.

## Website and API

The existing public URL now runs the accepted70-feature HistGB plus route-tuned decoder. [Deployment receipt](<reports/target96/deployment.json>) verifies HTTP200 health with the exact new model hash; [genuine browser evidence](<reports/target96/live/browser_evidence.json>) verifies actual public response geometry against the frozen final route. [Actual receiving-host response](<reports/target96/terminal/api_response.json>) and its screenshot are preserved. Earlier [release verification](<reports/final_five/delivery_verification.json>) remains historical evidence, not validation of this new model. No physical two-laptop demonstration or manual visual inspection is claimed.

The site has separate Student development and Evaluator sections, an original OpenCode-inspired monochrome theme, working dark/light preference and keyboard dataset navigation. [OpenCode was inspected](<docs/OPENCODE_DESIGN_REFERENCE.md>) only after the five-model training/testing gate. No OpenCode branding, fonts or runtime scripts are copied. Leaflet is bundled locally; the default map makes no external tile requests. [Website implementation](<docs/WEBSITE_DESIGN.md>) documents behavior.

The API exposes `/health`, `/datasets`, `/datasets/development/trips`, `/datasets/evaluator/trips` and dataset-specific `/match` endpoints. POST bodies contain only `trip_id`. Legacy `/trips` and `/match` still address student development. Both sections use the same frozen model/configuration and one shared inference slot; graph registries and private reference stores remain separate. No predicted routes or API responses are cached. Unavailable scores are null/N/A. See the [dataset contract](<docs/EVALUATOR_SECTION_PLAN.md>).

## Evidence, requirements and provenance

**Preserved earlier-release deliverables:** [Previous technical report (PDF)](<reports/final_five/PathSnap_Final_Technical_Report.pdf>) · [readable report source](<reports/final_five/TECHNICAL_REPORT.md>) · [student-development prediction ZIP](<outputs/final_student_test/submission.zip>) · [evaluator prediction ZIP](<outputs/final_evaluator_test/submission.zip>) · [delivery guide](<docs/FINAL_DELIVERY_GUIDE.md>).

The complete synthetic release suite passed **421 tests** (370 Python, 29 hosting, 22 Node), plus the static frontend smoke check, from an exact staged checkout without private datasets or probability arrays. The [release test record](<reports/final_five/test_results.json>) binds all four passing logs. Report generation checks 360 resolved facts, 50 public inputs and 55 image bindings, including twelve source-pixel-verified readable screenshot details.

The [47-row original-document audit](<analysis/final_requirements/requirements_matrix.json>) was checked against all five source DOCX documents, including nested tables. The sixth synthesis is their exact ordered concatenation. [Requirement review](<docs/FINAL_REQUIREMENTS.md>) distinguishes mandatory requirements, suggestions and human/external work. Epoch-wise loss curves are N/A for this pipeline; measured CV, testing, log-loss, timing, confusion and HPO charts are provided instead.

The original [three-family report](<reports/phase_three/PathSnap_Technical_Report.pdf>) and its published evidence remain immutable history, not the current five-family final result. Earlier RandomForest/ExtraTrees experiments are retained in the [historical experiment record](<experiments/results.json>) and documented separately rather than counted as final-scope families.

Development is AI-assisted. Authorized GitHub account operations do not imply four independent human authors or reviewers. [Development provenance](<docs/DEVELOPMENT_PROVENANCE.md>) and [publication boundaries](<docs/FINAL_PUBLICATION_POLICY.md>) make that explicit. The user requested a public same-origin website instead of the brief’s physical two-laptop setup. Venue demonstration, team viva, organizer deadlines and plagiarism clearance remain human/organizer responsibilities; no software artifact certifies them.
