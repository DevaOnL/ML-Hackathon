# PathSnap

A learned GPS-to-road map matcher with directed HMM/Viterbi reconstruction, a FastAPI service and a browser evaluator. The project targets the supplied PathSnap hackathon brief.

Public repository: https://github.com/DevaOnL/ML-Hackathon

Requested public deployment: **https://ml.d3va.dev**. The user chose a same-origin website hosted from this computer through Cloudflare instead of the brief’s separate UI laptop. Public deployment is not yet verified; check the deployment status before treating this URL as live.

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

Raw inputs stay read-only. `.route` files are read only by explicit fitting, auditing or postprediction evaluation, never by the matcher. Reserved evaluator trips are separate from the ten permitted student-development trips and have not been provided.

## Reproduce the experiments

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m mapmatch.experiments \
  --data-root "Student folder" --output experiments --artifacts artifacts \
  --train-observations-per-trip 350 --student-observations-per-trip 0 \
  --folds 3 --bayesian-trials 12 --n-jobs 2
```

The runner trains five probabilistic models, keeps whole trips together in every fold, fits preprocessing inside each training fold, then runs Gaussian-process expected-improvement Bayesian search on the selected family. It saves real fold results, trial records, coverage, parameters and trained artifacts. The final fit uses the 80 training trips; all 19,160 observations in the ten student-development trips are evaluated after training/search.

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

Public hosting uses dedicated restartable user services and a named Cloudflare tunnel to the loopback API. The tunnel exposes the application only, not the surrounding filesystem or other local websites. See `infrastructure/` for setup/status and `deploy/` for service configuration. Do not commit Cloudflare certificates, tokens, private references or `.env` files.

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

This is AI-assisted work. Account-based commit/push operations follow the owner’s request and do not claim four independent human authors. See `docs/DEVELOPMENT_PROVENANCE.md` for the development record.
