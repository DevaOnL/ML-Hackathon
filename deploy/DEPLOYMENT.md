# PathSnap deployment

## Current requested topology

The requested website is **https://ml.d3va.dev**. This computer hosts the fitted
classifier, graph/GPS inputs, HMM/Viterbi decoder, private reference scoring, API,
and allowlisted UI assets. A dedicated Cloudflare Tunnel forwards HTTPS to the
loopback API. Browser clients use the same origin for selection and inference;
there is no need to copy any dataset, model, or reference to the viewing device.

The public-site request replaces the physical second-laptop topology for this
installation. The [two-laptop alternative](<TWO_LAPTOPS.md>) is retained to explain
and reproduce the original evaluation architecture. Infrastructure-specific
DNS/tunnel/service setup is owned separately in the infrastructure directory;
this guide does not embed credentials or claim that an unverified DNS change is
live. Readiness must be checked after deployment.

## Local preparation

Run from the repository root. Python 3.14 is recommended; the current environment
was checked with Python 3.14.6. Dependencies in [the serving requirements](<requirements.txt>)
match the training environment's scikit-learn/numpy/scipy/joblib versions.
Only use trusted local joblib files: pickle-based models can execute Python code.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r deploy/requirements.txt
```

Do not start a second listener while a managed deployment is running. For a
manual, local-only instance **after training and integration are complete**:

```bash
bash deploy/start_backend.sh
```

[The launcher](<start_backend.sh>) runs one Uvicorn worker at
`http://127.0.0.1:8000`, disables unsanitized access logs, caps HTTP concurrency,
and sets numerical thread defaults to one. It does not train a model or fake
missing results. Default inference concurrency is one; a second simultaneous
match gets retryable HTTP 429 instead of an unbounded queue.

The import entry point is `backend.main:app` in [the backend entry module](<../backend/main.py>).
Opening `http://127.0.0.1:8000/` uses the same static UI/API as public hosting.
The current machine already has an unrelated service on port 8080; do not reuse
or stop it. Manual host/port changes require explicit configuration.

## Required trusted assets

- The model at `artifacts/model.joblib`, produced by the submitted training code,
  with matching `artifacts/model_metadata.json`. The UI receives only safe model
  name/hash/feature-count/seed metadata, never training IDs or estimator internals.
- Decoder JSON at `artifacts/match_config.json`. The default path is loaded if
  present; an explicitly configured missing file is an error. Effective settings
  including defaults are hashed in health, requests and logs. Final model/config
  assets must be frozen before a persistent deployment starts.
- Ten complete input trip prefixes: `<id>.track`, `<id>.nodes`, `<id>.arcs`.
  Both flat files and one subfolder per trip are accepted. IDs are discovered
  from actual files and need not match the development dataset's IDs.
- Server-private `<id>.route` files for scoring. They may be next to inputs, or
  in a completely separate private root. They are never static files and are
  opened only after live prediction has finished. Missing references make scores
  unavailable; they never trigger precomputed predictions.

The provided data contains 80 training and ten student-testing trips. The ten
hidden evaluator trips are **not supplied** in this checkout. Development-trip
results are not final evaluator performance. Installing new evaluator files is a
configuration change at final evaluation, not permission to tune on them.

## Environment

[The example environment](<env.example>) contains only nonsecret placeholders.
When using a trusted shell env file, export its variables before launch; Uvicorn
does not implicitly read this example.

| Variable | Default / behavior |
| --- | --- |
| `PATHSNAP_DATA_DIR` | `Student folder/student_test` relative to repository root |
| `PATHSNAP_DATASET_ROLE` | `development` for the default root, otherwise `evaluator`; may explicitly select one |
| `PATHSNAP_REFERENCE_DIR` | Omitted: private .route beside each input; set to a separate private root if available |
| `PATHSNAP_MODEL` | `artifacts/model.joblib` |
| `PATHSNAP_MODEL_METADATA` | `model_metadata.json` next to the chosen model |
| `PATHSNAP_MATCH_CONFIG` | `artifacts/match_config.json` if present; an explicit nonempty value must exist and validate |
| `PATHSNAP_CORS_ORIGINS` | Empty: same-origin hosting only; otherwise comma-separated exact http(s) origins, no wildcards/paths |
| `PATHSNAP_GRAPH_CACHE_SIZE` | 2 input graphs; no model outputs or labels are cached |
| `PATHSNAP_INFERENCE_CONCURRENCY` | 1 active match, additional matches HTTP 429 |
| `PATHSNAP_HOST` | Launcher only: 127.0.0.1; keep loopback for Cloudflare Tunnel |
| `PATHSNAP_PORT` | Launcher only: 8000 |
| `PATHSNAP_PYTHON` | Launcher only: repository .venv/bin/python |

Paths are trusted server configuration, never request properties. Relative paths
resolve from the repository root. Restart the managed API after changing an
already-loaded model/configuration/dataset. A missing model is retried lazily so
health remains visible during first-time training, but a loaded model is not
silently swapped underneath an active run.

### Hidden evaluator handoff

At final evaluation, obtain the ten reserved input trips and private references
through the authorized process. Keep references on this computer only, then set:

```bash
export PATHSNAP_DATA_DIR=/private/pathsnap/evaluator_inputs
export PATHSNAP_REFERENCE_DIR=/private/pathsnap/evaluator_references
export PATHSNAP_DATASET_ROLE=evaluator
export PATHSNAP_MODEL=artifacts/model.joblib
export PATHSNAP_MATCH_CONFIG=artifacts/match_config.json
```

Restart using the documented managed process. Verify health says evaluator role,
ten trips and the same frozen model/config hashes. `GET /trips` must display the
newly discovered ten IDs without source edits. The prediction model never sees
the route files, and no evaluator labels may inform training, threshold changes,
feature selection, model choice or tuning.

## Public-site deployment and verification

The intended chain is browser → Cloudflare HTTPS/Tunnel → 127.0.0.1:8000 → fresh
classifier/HMM → postprediction private scorer → predicted geometry and aggregate
scores. The backend serves only the exact UI file set and Leaflet assets; source,
reports, datasets, routes, weights, hidden files, docs and listings return 404.
Direct same-origin hosting needs no CORS entry. For this Cloudflare deployment,
set `PATHSNAP_CORS_ORIGINS=https://ml.d3va.dev` defensively and trust forwarded
headers only from the loopback tunnel process. An Origin-present request must
match the ASGI scheme/Host origin or this exact allowlist; disallowed and null
origins get 403 before inference. No-Origin CLI requests remain allowed for the
public demo. Do not expose the project using a generic repository-root HTTP server.

Keep the API process and tunnel under the separately provisioned service manager.
Do not start a replacement server on another port and assume it updates this
website. Never place Cloudflare tokens, tunnel credentials or certificates in
source control, Docker context or UI configuration.

Run these checks against the actual requested origin, not just a local listener:

```bash
curl --fail --silent --show-error https://ml.d3va.dev/health
curl --fail --silent --show-error https://ml.d3va.dev/trips
```

Require `status:"ready"`, `model_loaded:true`, role stated accurately, and ten
trip options. Select a returned ID in the real UI and click inference. Repeat a
trip: each click must produce a new request ID and new model/decoder work, not a
saved response. Test all ten development IDs before claiming integration complete.

A small HTTP smoke utility is provided in [the smoke client](<smoke_api.py>):

```bash
.venv/bin/python -B deploy/smoke_api.py --base https://ml.d3va.dev
.venv/bin/python -B deploy/smoke_api.py --base https://ml.d3va.dev --all
```

The second command runs live inference on all ten registered trips. The client
refuses evaluator-role inference unless `--allow-evaluator` is explicitly added
for the authorized final demonstration. Never use hidden evaluator scores for
tuning. It reports score status/timings, not reference labels.

Capture the browser route/result display and receiving-machine JSON
`match_completed` event for the same request ID. Completion logs include trip ID,
model/config hashes, inference/scoring time and aggregate scores. No screenshots
or metric values should be invented. See [the API contract](<API.md>) for exact
metric definitions and the explicit proxy-label limitations.

## Optional containers

[The Dockerfile](<Dockerfile>) packages only Python serving code and UI assets;
[the Docker context whitelist](<Dockerfile.dockerignore>) excludes private datasets,
weights, logs and credentials from the context. Inputs, references and artifacts
are read-only mounts in [the compose template](<compose.yaml>). The container runs
as a nonroot user with dropped capabilities and a read-only root filesystem.

```bash
export PATHSNAP_HOST_DATA_DIR="/absolute/path/to/Student folder/student_test"
export PATHSNAP_HOST_REFERENCE_DIR="/absolute/path/to/private/references"
export PATHSNAP_HOST_ARTIFACTS="/absolute/path/to/artifacts"
export PATHSNAP_DATASET_ROLE=development
# Choose readable mount permissions for container UID 10001; never make secrets public.
docker compose -f deploy/compose.yaml up --build -d
```

For development only, reference and data mount sources may point to the same
original student-testing root. Prefer separate roots for final evaluation.
The compose port binds loopback; a tunnel on this host can forward it. Do not run
this alongside the existing persistent API on the same port. This template is an
alternative deployment; container build/run is not claimed without a successful
local Docker check.

## Failure visibility

- Missing/invalid model: health not ready, match 503; never a Gaussian fallback
  disguised as learned inference.
- Wrong number/incomplete trips: health error and no partial/hardcoded menu.
- Matching failure: sanitized 500 with request ID; no fabricated route.
- No reference: live route, scores null and explicit unavailable status.
- Scoring failure: live route preserved with error status; private exception
  contents are not returned.
- Busy: retryable 429. A lost browser connection does not create a hidden cached
  result; let active work finish before retrying.

An unauthenticated public demo is not a general confidential-data service.
Raw GPS and predictions are intentionally visible for its ten registered trips.
Keep sensitive personal trips off the public dataset; retain private references
and credentials only on the server. Optional map tiles contact OpenStreetMap when
the user enables them; the self-hosted route overlay works without basemap tiles.
