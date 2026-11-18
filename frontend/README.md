# PathSnap evaluator workspace

A build-free, vanilla HTML/CSS/JavaScript dashboard for live map matching. It calls the model API for every run; there are no bundled trips, sample routes, cached predictions, or client-side evaluation labels.

## Primary deployment: one public website

The user-approved deployment is **[https://ml.d3va.dev](https://ml.d3va.dev)**. This replaces the original second-laptop presentation requirement: the application runs on the host computer and Cloudflare exposes the UI and model API at the same HTTPS origin.

Open the website, wait for **API ready**, select any of the ten API-returned trips, and choose **Reconstruct route**. No API configuration is needed for the public site. The browser uses its current origin for these requests:

- GET /health — trained-model and configuration readiness.
- GET /trips — all available trip IDs and non-label metadata.
- POST /match — JSON containing only the selected trip ID.

The backend serves an explicit allowlist of frontend assets. Do not expose the repository root, source datasets, internal development maps, artifacts, or report folders through a general-purpose public file server. The deployment/tunnel is managed separately; this frontend does not start a service or change DNS.

### Run the combined UI and API locally

From the repository root, after the backend dependencies, trained model, and data paths are configured:

~~~bash
bash deploy/start_backend.sh
~~~

The launcher defaults to loopback port 8000. Open http://127.0.0.1:8000; it serves this dashboard and the API together. The public Cloudflare tunnel should point to the configured local backend address. Keep the loopback binding for the public tunnel unless a separate trusted-LAN setup is explicitly required.

## Optional separate static UI / original two-laptop setup

No Node package installation, bundler, CDN, or build step is required. Python can serve the frontend on a separate laptop. The recommended allowlisted server is:

~~~bash
python3 -B deploy/serve_ui.py --host 0.0.0.0 --port 8090
~~~

For a local-only static preview without the backend package, Python’s standard server also works. The directory restriction is essential:

~~~bash
python3 -m http.server 8090 --bind 127.0.0.1 --directory frontend
~~~

Open http://127.0.0.1:8090. A static server does not implement API endpoints, so open **API settings**, enter the actual backend base URL, and choose **Save & reconnect**. For the original trusted-LAN arrangement, use Laptop 1’s LAN address, for example the scheme/address/port of its model server; do not enter localhost unless the model API runs on the same device as the browser. Allow the UI’s exact origin in the backend’s PATHSNAP_CORS_ORIGINS configuration, bind the API to the trusted-LAN interface, and permit its port through the server firewall.

Port 8090 is only an alternative preview port; do not use another application’s existing listener. No static server is needed for the combined/public deployment.

## Using the workspace

1. **Check readiness.** The header reports connectivity and the sidebar identifies the API host. The UI fetches health and the catalogue in parallel. Inference stays disabled unless the model reports ready and exactly ten distinct trips are available. A non-evaluation/development catalogue is labelled accordingly.
2. **Select a trip.** All trip choices come from GET /trips; none are embedded in HTML. Each option shows the actual ID and, when provided, observation count and duration. The list scrolls on smaller screens. Native radio controls support Tab, Shift+Tab, arrow keys, and Space.
3. **Run live inference.** Reconstruct route sends only JSON {"trip_id": "the selected API-returned ID"}. It waits for a fresh API response, with no-store caching. Selection and connection changes are disabled during a run to avoid associating a result with the wrong trip.
4. **Inspect the map.** Teal is the predicted route; orange dashes and points are the original noisy GPS observations. Independent route parts stay independent: the UI never draws a line across disconnected predictions. S/E markers identify the geometry’s endpoints. Use layer checkboxes, Fit view, or the keyboard-enabled Leaflet map to compare them.
5. **Review measured results.** Summary cards display the server’s HMM observation accuracy (metrics.accuracy), route continuity, inference timing, and total timing. The comparison strip separately displays observation-level HMM macro F1 and classifier Top-1 accuracy; the latter is never presented as decoder accuracy. Expand Scores & execution details for additional allowlisted scores, scoring time, model name/algorithm/hash, and request ID. Client round-trip timing is measured separately and clearly labelled. Null, missing, negative, non-finite, or invalid ratios display N/A; unavailable values never become fabricated zeroes.
6. **Read the evaluation caveat.** When the API flags proxy_reference, a visible notice states that observation scores use proxy alignment rather than official time-aligned labels. Server status and safe textual notes are shown separately. No reference route, reference road IDs, or ground-truth labels are requested or displayed.

### API settings

The default for an HTTP(S) website is window.location.origin. Advanced settings can point to another deployment, and an override is persisted only as the browser-local pathsnap.apiBaseUrl value. **Use website default** removes the need for a cross-origin address when saved. No result, coordinate, selected trip, or prediction is persisted.

An HTTPS page rejects HTTP API overrides to prevent mixed-content failures. URLs with credentials, query strings, or fragments are rejected. If browser storage is unavailable, the configured address still works for the current tab.

## Offline map behavior

Leaflet 1.9.4, its CSS, images, and BSD 2-Clause license are bundled under [vendor/leaflet](<vendor/leaflet/leaflet.js>). Typography uses installed system fonts. The default geometry view makes **no external map-tile request** and remains useful without internet access: both returned traces, points, endpoints, scale, zoom, pan, and Fit view work on a neutral geographic canvas.

“Offline ready” describes map rendering, not offline inference. A reachable model API is always required for a new run. The public website itself also requires a network connection; no service worker silently caches old predictions.

The optional Street basemap checkbox requests [OpenStreetMap](https://www.openstreetmap.org/copyright) tiles only after opt-in. Attribution stays visible. Tile errors or a 12-second tile timeout switch back to geometry view without losing the returned route. The UI does not download an offline tile pack.

## Loading, errors, and accessibility

- Health/catalogue requests time out after 15 seconds; inference waits up to 180 seconds. Gateway timeout responses are reported distinctly. No request is retried automatically.
- **Stop waiting** aborts the browser fetch only; it does not claim to cancel server computation. Check server logs before rerunning a long request. The same warning is given on a timeout.
- HTTP 422, 429, 500, 502, 503, and 504 receive actionable messages without echoing raw server internals. A 503 disables further runs until reconnecting. No placeholder route is shown after a failed run.
- A new selection or run clears the previous route and scores, preventing stale results from appearing under a different trip.
- Semantic sections, labelled native controls, an accessible modal with Escape handling, a skip link, visible focus rings, polite live announcements, alert notices, and reduced-motion styles are included. Colour is reinforced by line style and text labels. The layout adapts down to 320-pixel screens.

## Agreed API adapter contract

The adapter in [api.mjs](<api.mjs>) implements the contract agreed with the backend teammate before integration:

| Endpoint | Required public data |
| --- | --- |
| GET /health | status ready/not_ready/error; model_loaded; optional model_name, algorithm, model_hash; trip_count, expected_trip_count, dataset_role |
| GET /trips | trips array of trip_id, observation_count, duration_s; expected_count=10; dataset_role |
| POST /match | request_id, matching trip_id, predicted_route, raw_gps, metrics, timings, model, metrics_status, metric_notes, proxy_reference |

GeoJSON coordinates are [longitude, latitude]. The route may be a Feature or FeatureCollection of separate LineString parts; MultiLineString and Point are also supported. Original observations may be a FeatureCollection, LineString, or Point. The adapter clones geometry only and discards every GeoJSON property, along with all unused response fields (including predicted observation assignments). Invalid coordinates cause an explicit contract error instead of a misleading map.

Metrics use numeric fractions from 0 to 1 or null, except conventional nonnegative log loss. Times are milliseconds. The UI reads only documented aggregate metric keys. Model metadata uses name, algorithm, and sha256. Error detail payloads are never rendered verbatim.

## Checks without starting a server

Node is optional for checks, not a runtime/build dependency:

~~~bash
node --check frontend/app.mjs
node --check frontend/api.mjs
node frontend/smoke.mjs
~~~

The static smoke check validates DOM targets, accessible references, complete local Leaflet files, assets, and source integrity. It does not invent API responses or launch a service. If an actual API is already running, optional live checks are:

~~~bash
node frontend/smoke.mjs --api https://ml.d3va.dev
# Sends one real inference request for the first API-returned trip:
node frontend/smoke.mjs --api https://ml.d3va.dev --match-first
~~~

Independent adapter unit checks live in the repository’s test scope, maintained by the verification teammate. The optional [browser_check.py](<browser_check.py>) uses separately installed Playwright/Chromium against an already-running deployment; it never starts a server or injects responses. Example: `python3 -B frontend/browser_check.py --url http://127.0.0.1:8000`.

Genuine local browser checks were performed against the actual backend at http://127.0.0.1:8000 while it correctly reported a missing model/configuration. They verified all ten real API trip options, a disabled inference action with explicit readiness notice, local Leaflet initialization, keyboard radio selection, modal Tab/Escape/focus behavior, API URL validation/reset, and no horizontal overflow at 320/390/768 pixels. No browser console errors or external default requests were observed. The [machine-readable record](<validation/browser-check.json>), [desktop capture](<validation/desktop-not-ready.png>), and [390-pixel mobile capture](<validation/mobile-390-not-ready.png>) describe this not-ready state; they do **not** claim a successful model run. The image-input tool was unavailable for pixel inspection; screenshots were captured by the real browser and layout was verified through DOM dimensions. Final model-ready/public-HTTPS route runs and captures remain separate integration evidence.

## Third-party assets

[Leaflet 1.9.4](https://leafletjs.com) distribution files were downloaded from the [versioned unpkg distribution](https://unpkg.com/leaflet@1.9.4/dist/) and are bundled unchanged. Its [BSD 2-Clause license](<vendor/leaflet/LICENSE>) is included. Original UI artwork is inline SVG; no external font or icon dependency is used.
