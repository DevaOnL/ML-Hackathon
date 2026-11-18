# PathSnap website design handoff

## Scope and reference

This update keeps PathSnap's name and original route mark. The charcoal surfaces, system monospace type, fine borders and bracketed controls follow the observations in [the design reference](<docs/OPENCODE_DESIGN_REFERENCE.md>). No OpenCode artwork, fonts, copy or scripts are included.

The reference was collected after the five-family training and testing check passed at 2026-11-18T08:26:57Z. This frontend work does not change the selected classifier or frozen decoder.

## Theme and layout

Dark is the default. The header toggle switches to light and saves only the theme string under `pathsnap.theme`. Invalid or inaccessible browser storage falls back to dark; a toggle still works for the current tab if saving fails. Its accessible action label describes the next theme, and `aria-pressed="true"` means light is active.

[The stylesheet](<frontend/styles.css>) contains one theme implementation, with variables for both palettes. It uses installed monospace fonts, blue route geometry and amber GPS observations. Local Leaflet assets provide map controls. The geometry canvas works without street tiles; the optional basemap remains an explicit online opt-in. Theme changes leave the current geometry and scores intact.

The source defines a two-column desktop workspace, a stacked layout below 1024 pixels, smaller controls below 768 pixels, and a single-column score grid below 400 pixels. Visible focus styles and reduced-motion rules are included. These are source-level layout choices, not a claim of browser or visual verification.

## Dataset sections

The two tabs remain visible as Student development and Evaluator. Counts come from the server. An unconfigured evaluator displays its returned zero count with "Not configured" and stays disabled; the page never invents ten available trips. A configured but unavailable section is also disabled. Refresh checks availability again; no background inference or automatic retry runs.

The active panel contains only the selected dataset's trip catalogue. Changing sections immediately clears the selected trip, map and execution record, then loads the new section. Results are not retained for switching back. Each section must return ten distinct trips and agree with its catalogue summary before reconstruction is enabled. Incomplete catalogues remain visible with a readiness notice.

Tabs use `tablist`, `tab` and `tabpanel` roles with a roving tab stop. Left/Right wrap across enabled tabs; Home/End select the first/last enabled tab. Focus stays on the activated tab, and the shared panel's accessible label follows it. Trip choices remain native radios with their normal keyboard behavior.

## API boundary

[The fixed dataset contract](<docs/EVALUATOR_SECTION_PLAN.md>) is authoritative. [The adapter](<frontend/api.mjs>) accepts exactly the public dataset IDs `development` and `evaluator`. It drops unknown response fields and validates required identities.

| Request | Frontend behavior |
| --- | --- |
| `GET /health` | Reads readiness and the actual model name/hash for the header badge. No model name or frozen hash is hardcoded. |
| `GET /datasets` | Requires both dataset summaries, using `id`, `label`, `role`, `available`, `trip_count`, `expected_trip_count` and `status`. |
| `GET /datasets/{id}/trips` | Requires echoed `dataset_id` and `dataset_role` to match the requested section. Checks counts and string trip IDs. |
| `POST /datasets/{id}/match` | Sends exactly `{"trip_id":"selected API-returned string"}`. Requires the trip and both echoed dataset identities to match. |

Legacy trip/match normalizers remain available for the preserved legacy API. The new UI requires the dataset endpoints; it does not silently treat an old single-dataset response as evaluator data. There is no model comparison endpoint, request or placeholder. The five-model comparison belongs in the technical report. The badge shows what the running server reports.

## Request and result safety

[The application](<frontend/app.mjs>) snapshots the API base, dataset, trip and request generation for every run. A section change or reconnect invalidates the generation and stops waiting for the old request. All result, error and cleanup paths check that snapshot before changing the current UI. This also covers leaving a section and returning to the same trip ID before an earlier response arrives.

Stopping a browser request does not claim to stop server computation. The UI says that the server may still be processing. Busy responses remain explicit and do not trigger a retry. Health and catalogue requests time out after 15 seconds; match requests wait up to 180 seconds.

Every reconstruction makes a fresh POST with `cache: "no-store"`. Only returned, allowlisted predicted-route and GPS GeoJSON reach Leaflet. Geometry properties are discarded; disconnected route parts stay separate. No trip coordinates, predictions, labels, references or model code are bundled in the app. Results stay in current page memory and are never persisted. The only local preferences are theme and an optional API base URL.

Scores come from allowlisted server fields. Missing or invalid values display N/A, including reference-unavailable evaluator scores. Proxy-aligned observation metrics retain their visible caveat and server notes. The execution record identifies the dataset, trip through the summary, request and model hash.

## Checks and deployment handoff

Run from the repository root without starting a service:

~~~bash
node --check frontend/app.mjs
node --check frontend/api.mjs
node --test tests/frontend_adapter.test.mjs
node frontend/smoke.mjs
~~~

[The synthetic tests](<tests/frontend_adapter.test.mjs>) cover unavailable evaluator state, two sections, closed dataset IDs, missing/conflicting response identities, exact POST bodies, stale same-trip results and local theme behavior. An in-memory DOM harness exercises application events; it does not replace browser testing. [The smoke check](<frontend/smoke.mjs>) retains the local Leaflet/asset/ARIA guards and adds section, identity, freshness, theme and unsupported-endpoint checks.

No service was started, restarted or deployed for this source handoff. No raw evaluator data or private deployment configuration was accessed. Existing browser evidence and [the earlier browser checker](<frontend/browser_check.py>) remain unchanged. The Lead owns deployment and genuine public screenshots, responsive checks and two real POST captures.

Stable capture selectors:

| Control or record | Selector |
| --- | --- |
| Dataset tabs | `#tab-development`, `#tab-evaluator` |
| Dataset counts and active panel | `#count-development`, `#count-evaluator`, `#workspace-panel` |
| Theme action | `#theme-toggle`, `#theme-toggle-label` |
| Trip choices | `#trip-list input[name="trip-choice"]` |
| Fresh reconstruction | `#match-button`, `#cancel-match` |
| Returned identity | `#result-dataset`, `#result-request`, `#result-hash` |
| Scores and map | `#metric-accuracy`, `#metric-f1`, `#metric-continuity`, `#route-map` |

After activation, verify both real section counts, two fresh requests with different request IDs/results, theme persistence after reload, disabled/unavailable handling where applicable, keyboard navigation and modal focus. Inspect actual image captures before making visual-quality claims. Keep new browser evidence separate from the earlier phase's captures.
