# Final source review

Verdict: GO with one low-severity provenance observation. No critical, high or medium blocker was found in the reviewed source. This is an automated review, not a human sign-off or service-activation approval.

The [JSON report](<analysis/final_code_review/review.json>) records source locations, findings, commands, outcomes and exclusions. [Source hashes](<analysis/final_code_review/reviewed_source_hashes.json>) pin 33 reviewed files by their exact working-tree bytes.

## Finding

CR-01, low: the secondary runtime returns the primary's already-loaded classifier, then inherited startup logic hashes the current artifact file again. If a trusted operator replaces that file between the two startups, both datasets still share one model object but can report different model hashes. This requires local asset mutation; there is no HTTP model-selection path. The [synthetic result](<analysis/final_code_review/synthetic_security_probes.json#L336-L344>) shows one loader call, shared model/config identity and the mismatched hashes. The relevant code is in [datasets.py:38–63](<backend/datasets.py#L38-L63>) and [service.py:139–177](<backend/service.py#L139-L177>).

Keep the bundle read-only through startup and serving. A future revision can copy the primary model's hash and public metadata with the shared object under the primary lock. No frozen source was edited for this review.

## Checks run

| Check | Outcome | Evidence |
|---|---|---|
| Complete Python unit discovery | 291 passed | [Python log](<analysis/final_code_review/python_unit_tests.log#L590-L594>) |
| Hosting discovery | 29 passed | [Hosting log](<analysis/final_code_review/hosting_discovery_tests.log#L31-L34>) |
| Node adapter and in-memory DOM tests | 22 passed | [Node log](<analysis/final_code_review/frontend_unit_tests.log>) |
| Frontend static smoke, without an API URL | Passed | [Smoke log](<analysis/final_code_review/frontend_smoke.log>) |
| Direct-ASGI edge cases and cancellation | Passed; CR-01 reproduced separately | [Probe output](<analysis/final_code_review/synthetic_security_probes.json>) |

The first hosting command used package imports and failed to find its sibling module. Discovery from the infrastructure directory passed; the [initial error](<analysis/final_code_review/hosting_unit_tests.log>) remains recorded. The Python suite includes tiny generated-data estimator fits. No production training, HPO, actual model fixture or all-20 inference was run.

The additional [probe script](<analysis/final_code_review/synthetic_security_probes.py>) called ASGI in memory. It checked 25 unknown-dataset/method combinations before body receive, 27 origin/type/encoding/size cases across all inference paths, and six unsupported-method/path combinations. It also cancelled a pending request: the other dataset remained busy until the worker finished, then the semaphore became available. No socket or listening service was started.

## Source conclusions

- Dataset registries and graph caches are separate, including when trip IDs overlap. The model, decoder and inference budget are shared. Results and emission audits are request-local.
- Learned prediction and output validation finish before the first private-reference lookup. Scoring failures retain that prediction and mark metrics unavailable; response, note and log allowlists do not serialize reference arrays or arbitrary diagnostics.
- Unknown dataset IDs fail before body receive. The same size, media-type, compression and origin guards cover all inference paths. Strict request bodies reject extra fields.
- UI success, failure and finally paths check API base, dataset, trip and generation. The in-memory DOM test confirms a late same-ID result cannot clear a newer dataset run's loading state. Unavailable catalogues remain unavailable; invalid metric values become N/A.
- GeoJSON parsing drops supplied properties and extra coordinate components. Invalid or out-of-range longitude/latitude and unsupported geometry fail. API-derived text uses text nodes; the Leaflet attribution HTML strings are fixed literals.
- Active CLI/backend defaults select final-five paths without silent model fallback. The generic no-argument loading helper retains its documented historical default; callers outside those entry points should pass the desired artifact explicitly.
- GNB promotion is limited to float64 scoring inputs. Probability validity checks remain strict. Its earlier failed attempt is disclosed; the selected HistGB path is unaffected.
- The [exact runner diff](<analysis/final_code_review/runner_output_guard.diff>) contains only output-safety changes. The preserved measured source is SHA-256 `6e25f0492c8d48d62ff21ac05e5e5e5f6306ffdc27485458ee5d2f69e20adb65`; the maintained runner is `42b6bddd738518fb6eb533593b7b10f984cf1dcd1493de528c5350f1dfd2d46e`.
- Publication exemptions are twelve exact aggregate-report paths and one document. Nearby unknown, nested, raw and cache paths remain refused, and exempt paths still undergo secret scanning. The five synthetic publication tests pass.

## Limits

Real model/probability acceptance and live all-20/public deployment checks were assigned to other agents and were not duplicated. This review did not inspect raw/evaluator inputs, perform Git operations, certify the final staged publication set or review unfinished peer report/figure/reproduction deliverables. Real-browser visual and accessibility checks are outside this synthetic test result.

The one-slot semaphore bounds concurrent inference; it is not per-client rate limiting or authentication. Keep the Lead's exact-index publication gate and frozen-asset deployment checks.
