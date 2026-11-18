# PathSnap

A learned GPS-to-road map matcher with directed HMM/Viterbi reconstruction, a FastAPI service and a browser evaluator. The project targets the supplied PathSnap hackathon brief.

Public repository: https://github.com/DevaOnL/ML-Hackathon

Public deployment: **https://ml.d3va.dev**. Standard HTTPS, a real browser reconstruction and dedicated-service receiving-terminal events have been verified. This computer serves the same-origin UI/API through a dedicated Cloudflare tunnel, following the user's deployment choice instead of the brief's separate UI laptop. The live service now loads the TRAIN-selected Bagging model; `/health` reports the frozen artifact hash. All ten real HTTPS match requests were reconciled with the final CLI route geometry, states and scores.

## Current phase: three requested families

The user selected **Logistic Regression, distance-weighted KNN, and Bagging** for this phase. The organizer's five-family minimum and the other two model deliverables are deferred until further instruction. Earlier HistGradientBoosting and five-model experiments remain preserved historical work, separate from the new phase's model, metrics and report.

Current phase outputs use the [experiment directory](<experiments/phase_three/>), [artifacts](<artifacts/phase_three/>), [report evidence](<reports/phase_three/>), [visualizations](<visualizations/phase_three/>), and a separate final-route directory. The CLI and backend default to the requested-trio artifact paths; they never silently fall back to the older model. The phase protocol records the three-family comparison, fixed KNN reference-bank budget, and TRAIN-only Bayesian search.

## Completed model results

All three families were trained on 80 TRAIN trips and evaluated on all ten
student-development trips (19,160 observations). The table reports raw
observation top-1 against route-derived proxy labels, before HMM decoding.

| Candidate classifier | TRAIN trip-grouped CV top-1 | Student top-1 |
| --- | ---: | ---: |
| Logistic Regression | 88.40% | 89.52% |
| Distance-weighted KNN | 87.68% | 89.05% |
| Bagging, default parameters | 89.15% | 90.48% |
| Bagging, TRAIN-selected tuned parameters | 89.25% | 90.34% |

The tuned Bagging model remains frozen because selection used TRAIN-only CV,
even though default Bagging scores slightly higher on student top-1. Its HMM
produces connected routes for all ten trips: mean route length-weighted F1 is
97.89%, directed-arc multiset F1 is 94.92%, and continuity is 100%. No reserved-evaluator
result is claimed. The other two model families remain deferred.

See the [model comparison](<experiments/phase_three/model_comparison.csv>),
[independent actual-model acceptance](<reports/phase_three/model_acceptance.json>)
and [CLI/HTTPS delivery verification](<reports/phase_three/delivery_verification.json>).
The [technical report](<reports/phase_three/PathSnap_Technical_Report.pdf>)
is the final 23-page consolidated document, with 209 checked evidence facts,
20 embedded figures and 18 passing final-render guards. Its
[layout and OCR verification](<analysis/review/phase_three/pdf_release_layout_final/layout_report.json>)
passed without defects. The release suite passed 179 Python tests, 28 hosting
tests, eight frontend adapter tests and the static frontend smoke check.

## Setup

Use Python **3.14** for the exact pinned environment below. The reusable source supports Python 3.10 or newer with compatible dependencies, but the submitted model is trained with scikit-learn 1.8.0; use the pinned runtime versions for loading saved artifacts.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The local development environment uses system scientific packages through a virtual environment; the clean-install instructions above do not depend on that arrangement. Never load an untrusted Joblib/pickle artifact.

## Supply the data

The original dataset and organiser materials are not redistributed. Extract the supplied dataset archive into the project root so the trip folders are under `Student folder/train` (80 trips) and `Student folder/student_test` (ten trips). A trip contains matching `.nodes`, `.arcs`, `.track` and `.route` filenames. No downloads or hardcoded route answers are used during inference.

- Nodes: longitude/latitude, zero-based physical row IDs.
- Arcs: directed source/destination node IDs; zero-based physical row is the arc ID.
- Track: longitude/latitude/timestamp seconds, strictly increasing original times.
- Reference route: chronological arc-ID sequence, **not** a per-observation label file.

Raw inputs stay read-only. `.route` files are read only by explicit fitting, auditing or postprediction evaluation, never by the matcher. The user reported ten new evaluator trips after the current model was frozen, tested and deployed. They remain uninspected and unassessed in this report; their separate website section is queued for later.

## Reproduce the requested-trio phase

The measured phase runner deliberately requires verified shared caches and the saved Logistic Regression baseline. Raw data, caches and per-observation reference labels are not redistributed. A fresh clone can serve the bundled frozen model after the authorized dataset is supplied. Rebuilding the full experimental state requires the preprocessing and baseline preparation described in the phase reproduction guide; do not mistake a missing cache for permission to silently change the sample, folds or metrics.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Only for a deliberate NEW numerical experiment after input preparation:
python experiments/phase_three/run_phase_three.py --stage cv --run-id reproduction-1
python experiments/phase_three/run_phase_three.py --stage select --run-id reproduction-1
python experiments/phase_three/run_phase_three.py --stage hpo --run-id reproduction-1
python experiments/phase_three/run_phase_three.py --stage freeze --run-id reproduction-1
python experiments/phase_three/run_phase_three.py --stage student --run-id reproduction-1
```

These commands write only to isolated `experiments/phase_three_runs/reproduction-1/`
and `artifacts/phase_three_runs/reproduction-1/` trees. Existing completed or
partial stages refuse overwrites; omit `--run-id` only to address the original
phase, whose completed stages also refuse reruns. The bundled frozen model is
ready for inference and does not require training again. See the
[preparation guide](<docs/PHASE_REPRODUCTION.md>) before starting a new run.

The three trip-grouped TRAIN folds use 80 trips with 350 deterministically sampled observations per trip. KNN fits a predeclared 60,000-row training-fold-only reference bank. Bagging is a genuine bootstrap `BaggingClassifier` of decision trees. Eight Bayesian trials include one measured baseline prior, two seeded proposals and five Gaussian-process expected-improvement proposals. Family selection and tuning precede the new phase's student diagnostics; the earlier historical student results were already available and are disclosed as prior development context.

### Historical five-model reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m mapmatch.experiments \
  --data-root "Student folder" --output experiments --artifacts artifacts \
  --train-observations-per-trip 350 --student-observations-per-trip 0 \
  --folds 3 --bayesian-trials 12 --n-jobs 2
```

The historical runner above trains five probabilistic models, keeps whole trips together in every fold, fits preprocessing inside each training fold, then runs Gaussian-process expected-improvement Bayesian search on the selected family. It is not the requested three-family phase runner and is retained for provenance; do not rerun it as part of this phase. It saves real fold results, trial records, coverage, parameters and trained artifacts. The final fit uses the 80 training trips; all 19,160 observations in the ten student-development trips are evaluated after training/search.

Observation-level supervision is derived from reference-route geometry with a weak motion-direction tie-break. Accuracy, F1 and candidate-miss results based on this alignment are **proxy-label metrics**, not independently supplied GPS ground truth. Binary candidate F1 and directed-edge observation F1 are reported separately. Route-sequence metrics preserve repeated traversals; edit similarity measures order.

## Predict and score

```bash
python -m mapmatch predict --input "Student folder/student_test" --output outputs/new_run
python -m mapmatch evaluate --input "Student folder/student_test" \
  --predictions outputs/new_run --output reports/new_route_metrics.json
python -m mapmatch validate --input "Student folder/train" --output reports/new_input_check.json
```

Prediction opens only graph and GPS inputs. The output directory must be new/empty and outside the input tree. Scoring/validation output files must be new and outside input/prediction trees. A complete run writes original arc-ID routes, diagnostics, observation assignments and a deterministic flat route ZIP. Any failed trip prevents creation of a misleading complete archive. `--baseline` explicitly selects a geometry-only diagnostic; the public application requires the learned model.

## Run the website locally

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m uvicorn backend.main:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

Open http://127.0.0.1:8000. The same process serves the allowlisted UI and `/health`, `/trips`, `/match`. Every reconstruction request runs fresh learned inference. The server scores references only after matching and returns predicted geometry, raw observations and aggregate results. Reference routes, labels, model files, source files and development Folium maps are never served by the UI.

The service discovers the ten configured trip IDs; it does not hardcode them. Missing model/data, invalid IDs, invalid input, busy inference and scoring failures have explicit states. Leaflet is bundled locally; route/GPS geometry works without internet map tiles. OpenStreetMap tiles are optional.

Public hosting uses dedicated restartable user services and a named Cloudflare tunnel to the loopback API. The tunnel exposes the application only, not the surrounding filesystem or other local websites. See the [hosting guide](<infrastructure/HOSTING.md>) for setup/status and the [deployment directory](<deploy/>) for service configuration. Do not commit Cloudflare certificates, tokens, private references or `.env` files.

## Verify

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
node frontend/smoke.mjs
node --test tests/frontend_adapter.test.mjs
```

Independent tests cover physical row-ID preservation, metric projection, segment interiors, directed multigraph routing, zero-length arcs, loops, stationarity, disconnected roads, exact Viterbi enumeration, bounded candidate recovery, learned-emission use, repeat-aware scores, protected CLI output and API reference isolation. Browser/API integration evidence and exact dependency versions are recorded in `reports/`.

## Layout

- `mapmatch/`: data, spatial indexing, routing, features, training, optimisation, matcher and CLI.
- `backend/`: FastAPI application and private server-side scorer.
- `frontend/`: standalone browser interface and vendored Leaflet.
- `artifacts/`: trained classifiers, feature schema, model metadata and decoder configuration.
- `experiments/`: actual experiment protocol, folds, comparisons and trials.
- `scripts/`: decoder calibration, integration evidence and consolidated report generation.
- `visualizations/`: EDA/model charts and development-only Folium maps.
- `tests/`: independent Python tests; frontend adapter checks live with the UI source.
- `reports/`: consolidated PDF, measured results and verification evidence.
- `infrastructure/`, `deploy/`: public-host and local service setup.

## Integrity and limitations

Original dataset hashes and parsing evidence are recorded under `analysis/data_audit/`. Some supplied references contain disconnected joins or GPS/reference mismatches; they are retained and disclosed. Candidate misses remain in headline metrics. No reserved-evaluator result is claimed. Model-training and serving versions are recorded because scikit-learn artifacts are version-sensitive.

This is AI-assisted work. Account-based commit/push operations follow the owner’s request and do not claim four independent human authors. See the [development provenance](<docs/DEVELOPMENT_PROVENANCE.md>) for the development record.
