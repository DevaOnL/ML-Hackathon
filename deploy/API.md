# PathSnap HTTP contract

The API is implemented in [the FastAPI application](<../backend/app.py>) and
[the inference runtime](<../backend/service.py>). The deployed public site uses
the same origin for its UI and API. There is no upload, arbitrary-path,
model-download, reference-download, directory-listing, or public documentation
endpoint.

## GET /health

Returns HTTP 200 with readiness information. A reachable process is not
necessarily ready: require both `status == "ready"` and `model_loaded == true`.

- `status`: `ready`, `not_ready`, or `error`.
- `model_loaded`, `model_name`, `algorithm`, `model_hash` (SHA-256).
- `model_metadata_loaded`, `match_config_hash` (SHA-256 of effective sorted
  decoder configuration, including defaults).
- `dataset_role`: `development` for the supplied student-testing root;
  `evaluator` for another configured root unless explicitly overridden.
- `trip_count`, `expected_trip_count` (10 in environment-based deployment),
  `inference_concurrency`, `prediction_cache: false`.
- `errors`: sanitized `{code,message}` entries, with no filesystem paths,
  model parameters, credentials, labels, or reference geometry.

## GET /trips

Returns `{trips:[{trip_id,observation_count,duration_s},...],count:10,
expected_count:10,dataset_role}`. All ten IDs are discovered dynamically, never
hardcoded. Metadata comes only from raw GPS observations.

Discovery accepts flat trip prefixes or one subfolder per trip, requires matching
GPS/node/arc files, rejects duplicate/unsafe IDs and symlinks escaping the root,
and fails explicitly if the count is not ten. Reference files are not opened.

## POST /match

Send `Content-Type: application/json`, no compression, no extra fields, and at
most 1024 bytes. The only request property is a string ID from the current
`/trips` response:

```json
{"trip_id":"new_trip_001"}
```

The ID above illustrates the format, not a registered default. Clients cannot
choose paths, supply labels, replace graphs/models, or request precomputed routes.

### Successful response

| Field | Content |
| --- | --- |
| `request_id` | Server-generated identifier, also in `X-Request-ID` |
| `trip_id`, `dataset_role` | Registered selection and explicit partition role |
| `predicted_route` | GeoJSON FeatureCollection of connected predicted LineString parts |
| `raw_gps` | GeoJSON FeatureCollection with the raw GPS LineString, or Point for one observation |
| `observation_assignments` | Predicted-only `{observation_index,edge_id,fraction,distance_m,coordinate}` |
| `metrics` | Allowlisted aggregate scores below; unavailable values are null |
| `metrics_status` | `ok`, `partial`, `unavailable`, or `error` |
| `metric_notes` | Fixed explanations, never exception text or private contents |
| `proxy_reference` | true; observation references are route-derived proxies |
| `timings` | `inference_ms`, `scoring_ms`, `total_ms` |
| `model` | Safe `name`, `sha256`, `algorithm`, optional feature count/seed |
| `match_config_hash` | Effective decoder configuration hash |
| `diagnostics` | Predicted-only counts, length, breaks, lateral distances, learned-emission and candidate-retry counts |

All coordinates use **[longitude, latitude]** in decimal degrees. Predicted
features carry `kind: "predicted_route"`, `part_index`, and `arc_count`. Raw GPS
properties contain `kind: "raw_gps"`, `observation_count`, and raw `timestamps`.
No artificial line joins disconnected predicted parts. The current matcher
requires connected decoding or returns an error, never a fabricated connector.

### Metrics

Observation scores use [route-projection/motion proxy alignment](<../mapmatch/features.py>).
Supplied references are road sequences, not timestamp-aligned per-GPS labels.
These values must not be presented as official organizer scores or independently
verified physical ground truth.

- `accuracy`, `precision`, `recall`, `macro_f1`: HMM observation-level canonical
  directed-edge classification. Precision/recall/F1 are macro averages over the
  union of true and predicted edge classes. Duplicate identical directed arc rows
  are canonicalized for these observation metrics.
- `top1_accuracy`: classifier-only candidate argmax versus the proxy edge;
  candidate misses count as incorrect. It is **not** HMM accuracy.
- `classifier_accuracy`, `classifier_precision`, `classifier_recall`,
  `classifier_macro_f1`: corresponding classifier-only observation metrics.
- `candidate_miss_rate`: fraction of scored observations whose proxy edge is
  absent from the final candidate set, including bounded decoder retries.
- `log_loss`: mean categorical negative log probability of the proxy edge under
  candidate-normalized emissions. Target probability mass is floored at 1e-7
  (including missing targets), matching offline categorical evaluation. This is
  a nonnegative natural-log scalar, **not a percentage**. It differs from
  training's binary candidate-row log loss.
- `route_continuity`: fraction of successive predicted arcs sharing a directed
  endpoint (1 for a one-arc route); requires no reference.
- `route_edge_f1`, `route_length_f1`: directed arc multiset-overlap F1 retaining
  repeats, count-based and length-weighted respectively.
- `sequence_similarity`: one minus Levenshtein distance divided by the longer
  route length. Order is evaluated separately from multiset overlap.
- `scored_observation_fraction`: valid proxy labels divided by decoded
  observations. Any trusted-config subsampling is reflected in decoded counts;
  the shipped configuration retains all observations.

Route definitions are implemented in [the route metrics module](<../mapmatch/metrics.py>).
No reference route, reference length, reference edge IDs, reference per-observation
labels, confusion-matrix class labels, or reference geometry is returned.

### Computation order and timing

1. Validate registered-ID membership and acquire a bounded inference slot.
2. Load/cache graph input, create a **fresh matcher** and request-local learned
   emission audit, and compute classifier potentials plus HMM/Viterbi predictions.
3. Require actual learned calls for every decoded observation. An absent trained
   artifact never permits a geometry-only substitute.
4. **Only after prediction returns**, resolve/open the private reference and
   compute aggregate scores. The matcher never receives references.
5. Build allowlisted JSON. Predictions, candidate audits, labels and response
   objects are not cached. GPS metadata/model and a bounded graph LRU may persist.

`inference_ms` includes graph loading, candidates, features/probabilities and
Viterbi. `scoring_ms` includes private reference lookup/alignment and scoring.
`total_ms` also includes request-local setup/serialization construction, not
network transfer or browser rendering. First-request artifact loading, if
training finishes after startup, appears in total rather than decoder time.

### Errors

```json
{"detail":{"code":"...","message":"...","request_id":"..."}}
```

No request body, private path, traceback, label, or reference geometry is echoed.

| HTTP | Condition |
| --- | --- |
| 422 | Invalid body/extra fields/type, unsafe ID, or unregistered ID |
| 403 | Origin-present browser request is neither same-origin nor explicitly allowed |
| 413 | Body larger than 1024 bytes |
| 415 | Non-JSON or compressed body |
| 408 | Body did not arrive within ten seconds |
| 429 | Inference slot occupied; `Retry-After: 3` |
| 503 | Missing/invalid model or unavailable dataset/configuration |
| 500 | Live decoding, learned-audit, or unexpected internal failure |
| 404 | Nonpublic static/data/docs/source resource |

Missing references do not replace valid predictions: accuracy-like fields are
null and `metrics_status` is `unavailable`. A scorer failure also preserves the
live route, sets `metrics_status: "error"`, and logs only the exception type.

Every response has `Cache-Control: no-store` and a request ID. CORS uses only
exact configured origins, never `*`, without credentials. Origin-present match
requests are also explicitly rejected before model work if neither same-origin
nor allowed; `Origin: null` is rejected. Requests with no Origin header remain
allowed for CLI/evaluator clients. This is not authentication: the public
deployment intentionally exposes its fixed ten-trip demo. One inference is admitted at a time by default; excess work
is rejected rather than queued without bound. The launcher also bounds Uvicorn
connection concurrency.

Structured JSON events include `match_started`, `match_completed`,
`match_rejected`, `scoring_failed` and `http_request`. Completion events contain
request ID, registered trip ID, model/config hashes, measured timings, score
status and aggregate scores. Receiving-laptop screenshots can show these events
without exposing reference labels or geometry.
