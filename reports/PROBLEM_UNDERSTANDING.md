# Problem understanding

The 51.47-second video asks us to recover a route from noisy GPS observations on a road network. The six PathSnap documents added later define the full submission: train and compare at least five models, use trip-grouped cross-validation and Bayesian optimisation, integrate learned classifier outputs into HMM/Viterbi route reconstruction, build a FastAPI service and evaluator UI, and submit a public repository with one consolidated PDF. The complete document repeats the five supporting documents; recursive XML inspection recovered nested table rows missed by the initial extraction.

## Current user-scoped phase

On18November2026 the user explicitly limited the completed phase to **Logistic Regression, KNN (or a distance-based classifier), and Bagging**, with the other two model deliverables deferred until later instruction. The current phase selects only among those three using TRAIN-only grouped validation and a new Bayesian search; prior HistGradientBoosting/five-model artifacts remain historical work, not this phase's final model. The organizer's five-model minimum is therefore deferred, not claimed satisfied by this phase. Current phase inputs/results/paths are recorded in `docs/PHASE_SCOPE.json`.

Both verified local Whisper base.en and tiny.en transcribed the complete51.456-second narration independently; they agree after normalizing case, punctuation and project-name spacing. It introduces urban GPS drift and road-sequence reconstruction, with no additional model/metric/deployment constraints. Machine-ASR confidence, timestamps and exact weight/audio hashes are recorded locally in `analysis/video_audio/narration_audit.json`; this is not a claim of manual listening.

## Input data

The original dataset archive contains 360 files: 80 training trips and ten student-testing trips, each with four headerless whitespace-separated files. Every file was read completely, numerically parsed and checked against its archived bytes. The original inputs remain unchanged.

| File | Schema and identifier convention |
|---|---|
| `.nodes` | Longitude, latitude in decimal degrees; physical row index is the zero-based node ID. |
| `.arcs` | Source-node ID, destination-node ID; physical row index is the zero-based directed arc ID. |
| `.track` | Longitude, latitude, timestamp in seconds. Keep the original order and variable time gaps. |
| `.route` | One chronological sequence of traversed original arc IDs. Its rows are not aligned one-to-one with GPS observations. |

There are 2,024,174 node rows, 4,061,945 arc rows, 227,577 GPS observations and 34,813 reference-route rows across the 90 supplied trips. Original node/arc order is retained. Duplicate directed arc pairs, zero-length geometry, disconnected graph components and some disconnected reference-route joins require explicit treatment.

## Supervision and evaluation

Candidate classifiers need observation-level targets, whereas the supplied route files give route sequences. Training and scoring therefore derive proxy observation labels from the nearest reference-route segment with a weak motion-direction tie-break. This alignment uses reference geometry only in fitting or postprediction scoring. It is never an inference feature. Pointwise accuracy/F1 and candidate misses are labelled as proxy metrics throughout the results and interface. Ordered-route overlap/edit metrics compare the predicted and supplied sequences directly.

Cross-validation keeps entire training trips together. Models and Bayesian search fit only training folds; fitted preprocessing stays inside each fold. Final classifiers refit using training data. The ten student-testing trips are permitted development comparisons under the written guidelines. The separate ten reserved evaluator trips have not been supplied and are not claimed as tested. The earlier independent prototype used an ID-modulo-five training holdout; it is retained as exploratory evidence and does not replace the final grouped-CV experiment.

## Route reconstruction

A state is a projected point on an original directed arc. Exact point-to-segment candidates include the interiors of long roads. Distances use a local longitude/latitude projection in metres. Learned classifier probabilities and lateral distance form emission costs. Directed shortest paths supply transition costs and recover intervening arcs. Log-space Viterbi chooses a sequence; reconstruction preserves loops and repeated traversals.

If nearest-candidate pruning removes a reachable road, the matcher retries the whole sequence with a bounded larger candidate set. It never resets the route across a disconnected component or invents an arc. Unrecoverable cases return explicit errors. All observations are used by the final configuration; optional thinning exists only as an explicit diagnostic setting.

## Deployment change authorised by the user

The written brief describes an API laptop and a separate UI laptop. The user explicitly replaced that arrangement with a public website at **https://ml.d3va.dev**, served from this computer through Cloudflare. The same-origin site retains the intended boundary: the browser sends only a selected trip ID; the server runs fresh inference, then scores private references and returns predicted geometry plus aggregate metrics. Reference labels/geometry are excluded from UI assets and responses. Cloudflare authentication and real public checks must succeed before the site is described as live.

## Submission and acceptance

The two final deliverables are the public repository and one consolidated PDF. Reproducible prediction `.route` files, diagnostics, comparison CSVs, trial records and maps are supporting repository artifacts. No official numeric leaderboard formula is supplied, so local metric definitions and denominators are stated explicitly. Required evidence includes actual EDA, cross-model and optimisation plots, a development-only Folium reference/prediction map, UI screenshots and received API request/response evidence. No reserved-evaluator or unperformed deployment results may be inferred from local tests.

Detailed source inspection is recorded under `analysis/data_audit/`, `analysis/video/` and `analysis/instructions_review/`. Implementation ownership and acceptance steps are in `docs/IMPLEMENTATION_PLAN.md`.
