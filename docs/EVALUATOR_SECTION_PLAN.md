# Evaluator dataset API: contract and build plan

This is the contract between the backend dataset API and the website, plus the build plan behind it. The backend work stays inside `backend/` and one new test file; the three existing endpoints keep their exact request and response shapes. The evaluator dataset stays switched off until the Lead sets explicit server configuration after the model freeze, so the UI must be able to render the evaluator entry as `not_configured` from the first deploy.

## Endpoints

Three legacy endpoints are unchanged: `GET /health`, `GET /trips`, and `POST /match` with body `{"trip_id": "..."}`.

Three endpoints are added.

### GET /datasets

Returns the catalogue of datasets this server serves. Public dataset IDs are a closed list, `development` and `evaluator`; the server accepts nothing else. No filesystem path appears anywhere in the response.

```json
{
  "datasets": [
    {"id": "development", "label": "Student development", "role": "development",
     "available": true, "trip_count": 10, "expected_trip_count": 10, "status": "ready"},
    {"id": "evaluator", "label": "Evaluator", "role": "evaluator",
     "available": false, "trip_count": 0, "expected_trip_count": 10, "status": "not_configured"}
  ]
}
```

Status values are `ready`, `not_ready`, `error`, and `not_configured` (evaluator only). `available` is true when that dataset's trips are registered. The evaluator entry reads `not_configured` until the server is given an explicit evaluator directory; a directory that merely sits next to the development data is never discovered implicitly.

### GET /datasets/{dataset_id}/trips

Same shape as `GET /trips`, with `dataset_id` added and `dataset_role` matching the selected dataset.

```json
{"trips": [{"trip_id": "00000001", "observation_count": 4, "duration_s": 3.0}],
 "count": 1, "expected_count": 10, "dataset_id": "development", "dataset_role": "development"}
```

### POST /datasets/{dataset_id}/match

The body is exactly `{"trip_id": "..."}`: one string field, no extras, the same validation as `POST /match`. The response is the existing match response with `dataset_id` added; `dataset_role` is already part of it.

## Errors the UI should expect

| Situation | Status | detail.code |
|---|---|---|
| dataset_id outside the closed list, or a traversal attempt | 404 | unknown_dataset |
| evaluator requested while not configured | 503 | dataset_not_configured |
| configured dataset unavailable (missing or invalid directory) | 503 | dataset_missing and neighbours |
| trip_id missing, mistyped, or extra fields present | 422 | invalid_request or invalid_trip_id |
| another inference in flight on any dataset | 429 | inference_busy (Retry-After: 3) |
| body over 1024 bytes | 413 | request_too_large |
| content type other than application/json | 415 | unsupported_media_type |
| cross-origin POST without an allowlisted Origin | 403 | origin_not_allowed |

Error bodies keep the legacy shape `{"detail": {"code", "message", "request_id"}}` and never contain paths, labels, or geometry.

## Global health

`GET /health` keeps its legacy fields and status. When the evaluator dataset is configured, one extra key `datasets` carries the same summaries as `GET /datasets`, so a broken evaluator directory is visible in health while the development dataset keeps serving. When the evaluator is not configured, the `/health` output is unchanged.

## Server-side configuration

Trusted environment only; no client can select paths, configs, or model files:

- `PATHSNAP_EVALUATOR_DATA_DIR`: trip dataset directory for the evaluator partition
- `PATHSNAP_EVALUATOR_REFERENCE_DIR`: private reference directory, optional; references are looked up only after a prediction completes
- `PATHSNAP_EVALUATOR_EXPECTED_TRIP_COUNT`: default 10

Until `PATHSNAP_EVALUATOR_DATA_DIR` is set, every evaluator route answers 503 and the catalogue reports `not_configured`. The Lead enables it after the final model freeze; nothing in the UI can.

## Rules the implementation keeps

- One frozen classifier, one decoder configuration, and one concurrency budget are shared across both datasets. An inference in flight on development returns 429 for evaluator requests too.
- Each dataset gets its own trip registry and graph cache, so the same trip ID appearing in both partitions resolves inside its own dataset and its own graph.
- Reference routes stay server-private: the reference store is consulted only after the decoder returns, and only aggregate metrics reach the response.
- No prediction or result is ever cached.
- The public boundary (origin check, 1024-byte limit, content type check, security headers, request IDs) applies to the new POST route exactly as it applies to `/match`.
- Static file serving stays limited to the allowlisted frontend assets; `/datasets/...` is API only.

## Build order

1. This contract.
2. `backend/config.py`: evaluator settings fields, appended at the end of the dataclass so positional construction in existing tests keeps working.
3. `backend/datasets.py`: a `DatasetService` wrapping the primary `InferenceRuntime`, building the evaluator runtime from the primary settings via `dataclasses.replace` when and only when an explicit evaluator directory is configured.
4. `backend/app.py`: the three new routes plus the `datasets` health summary; `backend/boundary.py`: POST guarding and log route names for the new match path.
5. `tests/test_datasets.py`: synthetic fixtures only. Two dataset folders share one trip ID with different geometry, a stub classifier runs under the real geometry scoring, reference-present and reference-absent cases are covered, plus unknown and traversal IDs, strict bodies, origin and size limits, the shared busy semaphore, and the catalogue states.

No evaluator input is read by any test; every fixture is synthetic. No new server or service is started here; the Lead activates the runtime configuration later and validates the real twenty calls.

## Frontend integration notes

The intended flow: fetch `GET /datasets`, build a dataset selector from the catalogue (disable or hide the evaluator entry while its status is `not_configured`), then list trips with `GET /datasets/{id}/trips` and match with `POST /datasets/{id}/match`. The legacy `/trips` and `/match` keep working for the current single-dataset view, so the selector can land before anything else changes. Treat 503 `dataset_not_configured` as a stable state until the next health poll, and 429 with `Retry-After` as retryable. Same-origin requests need no CORS setup; the allowlist behaviour matches the existing endpoints.
