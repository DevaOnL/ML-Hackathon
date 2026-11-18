import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import * as api from "../frontend/api.mjs";
import {
  ApiError, defaultApiBase, normalizeApiBase, normalizeHealth, normalizeTrips,
  normalizeGeoJSON, geometryPositions, geometryParts, normalizeMatch, requestJson,
  normalizeDatasets, normalizeDatasetId, normalizeDatasetTrips, datasetTripsPath, datasetMatchPath,
  DATASET_IDS, isCurrentMatch, normalizeTheme, themeToggleState, THEME_STORAGE_KEY,
} from "../frontend/api.mjs";

const https = { protocol: "https:", origin: "https://ml.d3va.dev" };
const privateSentinel = "SYNTHETIC_PRIVATE_REFERENCE_7839124";
const routeGeometry = { type: "FeatureCollection", features: [
  { type: "Feature", properties: { reference: privateSentinel, popup: "<script>unsafe</script>" },
    geometry: { type: "LineString", coordinates: [[8, 50, 999], [8.001, 50.001, 999]] } },
] };

test("public HTTPS deployment is same origin with strict API configuration", () => {
  assert.equal(defaultApiBase(https), "https://ml.d3va.dev");
  assert.equal(normalizeApiBase("", https), "https://ml.d3va.dev");
  assert.equal(normalizeApiBase("https://api.example///", https), "https://api.example");
  for (const value of ["http://api.example", "file:///etc/passwd", "javascript:alert(1)",
                       "https://user:pass@api.example", "https://api.example?token=secret",
                       "https://api.example#reference", "not a URL"]) {
    assert.throws(() => normalizeApiBase(value, https), ApiError);
  }
});

test("health readiness requires explicit loaded model and strips untrusted server fields", () => {
  const health = normalizeHealth({ status: "ready", model_loaded: true, model_name: "Fitted classifier",
    trip_count: 10, expected_trip_count: 10, dataset_role: "development", reference: privateSentinel });
  assert.equal(health.ready, true);
  assert.equal(health.tripCount, 10);
  assert.equal(health.datasetRole, "development");
  assert.equal(JSON.stringify(health).includes(privateSentinel), false);
  assert.equal(normalizeHealth({ status: "ready", model_loaded: false }).ready, false);
  assert.equal(normalizeHealth({ status: "unknown", model_loaded: true }).ready, false);
  assert.throws(() => normalizeHealth({ status: "ready", model_loaded: "true" }), ApiError);
});

test("trip catalogue is dynamic, retains exact string IDs and rejects duplicate IDs", () => {
  const catalogue = normalizeTrips({ trips: [
    { trip_id: "00000009", observation_count: 123, duration_s: 145, route: privateSentinel },
    { trip_id: "00000002", observation_count: 99, duration_s: 101 },
  ], expected_count: 10, dataset_role: "development" });
  assert.deepEqual(catalogue.trips.map(trip => trip.id), ["00000009", "00000002"]);
  assert.equal(catalogue.trips[0].requestId, "00000009");
  assert.equal(JSON.stringify(catalogue).includes(privateSentinel), false);
  assert.throws(() => normalizeTrips({ trips: [{ trip_id: "same" }, { trip_id: "same" }] }), ApiError);
  assert.throws(() => normalizeTrips({ trips: [{ trip_id: " " }] }), ApiError);
  assert.throws(() => normalizeTrips({ wrong: [] }), ApiError);
});

test("GeoJSON geometry keeps lon/lat only and discards all supplied properties", () => {
  const normalized = normalizeGeoJSON(routeGeometry, "prediction");
  assert.deepEqual(normalized.features[0].properties, {});
  assert.deepEqual(geometryPositions(normalized), [[8, 50], [8.001, 50.001]]);
  assert.equal(JSON.stringify(normalized).includes(privateSentinel), false);
  const parts = normalizeGeoJSON({ type: "MultiLineString", coordinates: [
    [[8, 50], [8.1, 50]], [[8.2, 50], [8.3, 50]],
  ] }, "prediction");
  assert.equal(geometryParts(parts), 2);
  assert.equal(geometryPositions(parts).length, 4);
  for (const geometry of [
    { type: "LineString", coordinates: [[8, 50]] },
    { type: "Point", coordinates: [181, 50] },
    { type: "Point", coordinates: [8, 91] },
    { type: "Point", coordinates: [NaN, 50] },
    { type: "Point", coordinates: [8, Infinity] },
    { type: "Point", coordinates: ["8", 50] },
    { type: "Polygon", coordinates: [] },
  ]) assert.throws(() => normalizeGeoJSON(geometry, "prediction"), ApiError);
});

test("match response accepts only selected trip and allowlisted geometry/scalar metrics", () => {
  const payload = {
    trip_id: "00000009", request_id: "request-1", predicted_route: routeGeometry,
    raw_gps: { type: "Point", coordinates: [8, 50] }, reference_route: privateSentinel,
    metrics: { top1_accuracy: 0.9, accuracy: 0.8, macro_f1: 0.75, log_loss: 0.2,
      route_continuity: 1, candidate_miss_rate: 0.05, reference_array: privateSentinel },
    metrics_status: "ok", proxy_reference: true,
    metric_notes: ["Proxy-aligned development metrics."],
    timings: { inference_ms: 10, scoring_ms: 2, total_ms: 12, server_path: privateSentinel },
    model: { name: "Trained model", algorithm: "learned_classifier_hmm_viterbi",
      sha256: "a".repeat(64), reference: privateSentinel },
  };
  const result = normalizeMatch(payload, "00000009");
  assert.equal(result.metrics.top1_accuracy, 0.9);
  assert.equal(result.metrics.accuracy, 0.8);
  assert.equal(result.proxyReference, true);
  assert.deepEqual(result.timings, { inferenceMs: 10, scoringMs: 2, totalMs: 12 });
  assert.equal(JSON.stringify(result).includes(privateSentinel), false);
  assert.throws(() => normalizeMatch(payload, "00000002"), ApiError);
  const missing = normalizeMatch({ ...payload, metrics: { accuracy: 2, log_loss: -1, macro_f1: NaN },
    timings: { total_ms: Infinity } }, "00000009");
  assert.equal(missing.metrics.accuracy, null);
  assert.equal(missing.metrics.log_loss, null);
  assert.equal(missing.metrics.macro_f1, null);
  assert.equal(missing.timings.totalMs, null);
});

test("match transport sends only selected trip, no-store and no labels or geometry", async () => {
  const requests = [];
  const result = await requestJson("https://ml.d3va.dev", "/match", {
    method: "POST", body: { trip_id: "00000009" },
    fetchImpl: async (url, options) => {
      requests.push({ url, options });
      return { ok: true, status: 200, json: async () => ({ trip_id: "00000009" }) };
    },
  });
  assert.equal(result.trip_id, "00000009");
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "https://ml.d3va.dev/match");
  assert.equal(requests[0].options.cache, "no-store");
  assert.equal(requests[0].options.credentials, "same-origin");
  assert.equal(requests[0].options.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(requests[0].options.body), { trip_id: "00000009" });
});

test("HTTP and network errors never interpolate sensitive server error details", async () => {
  await assert.rejects(requestJson("https://ml.d3va.dev", "/match", {
    fetchImpl: async () => ({ ok: false, status: 500, json: async () => ({ detail: privateSentinel }) }),
  }), error => error instanceof ApiError && error.status === 500 && !error.message.includes(privateSentinel));
  await assert.rejects(requestJson("https://ml.d3va.dev", "/health", {
    fetchImpl: async () => { throw new Error(privateSentinel); },
  }), error => error instanceof ApiError && error.kind === "network" && !error.message.includes(privateSentinel));
  for (const payload of [[], null, "wrong response"]) {
    await assert.rejects(requestJson("https://ml.d3va.dev", "/health", {
      fetchImpl: async () => ({ ok: true, status: 200, json: async () => payload }),
    }), ApiError);
  }
});

test("timeout and caller abort remain distinct and never claim server computation cancelled", async () => {
  const hangingFetch = async (_url, { signal }) => await new Promise((_resolve, reject) => {
    if (signal.aborted) reject(new Error("aborted"));
    else signal.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
  });
  await assert.rejects(requestJson("https://ml.d3va.dev", "/match", {
    timeoutMs: 5, fetchImpl: hangingFetch,
  }), error => error.kind === "timeout" && error.message.includes("still be processing"));
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(requestJson("https://ml.d3va.dev", "/match", {
    signal: controller.signal, fetchImpl: hangingFetch,
  }), error => error.kind === "cancelled" && error.message.includes("may still finish"));
});

// Every catalogue and coordinate in this file is synthetic.
function datasetSummary(id, overrides = {}) {
  return { id, label: id === "development" ? "Student development" : "Evaluator", role: id,
    available: true, trip_count: 10, expected_trip_count: 10, status: "ready", ...overrides };
}
function datasetPayload(evaluator = {}) {
  return { datasets: [datasetSummary("development"), datasetSummary("evaluator", evaluator)] };
}
function tripsPayload(id, count = 10) {
  return { dataset_id: id, dataset_role: id, count, expected_count: 10,
    trips: Array.from({ length: count }, (_, index) => ({ trip_id: "synthetic-" + index, observation_count: 2, duration_s: 1 })) };
}
function matchPayload(id, overrides = {}) {
  return { trip_id: "synthetic-0", dataset_id: id, dataset_role: id,
    predicted_route: routeGeometry, raw_gps: { type: "Point", coordinates: [8, 50] },
    request_id: "synthetic-" + id, metrics_status: "unavailable", metrics: {},
    timings: { inference_ms: 4, scoring_ms: 1, total_ms: 5 },
    model: { name: "Synthetic frozen model", sha256: "b".repeat(64) }, ...overrides };
}

test("unset evaluator retains real zero count and not_configured state", () => {
  const result = normalizeDatasets(datasetPayload({ available: false, trip_count: 0, status: "not_configured", private: privateSentinel }));
  assert.deepEqual(result.datasets.map(({ id, available, tripCount, expectedCount, status }) => ({ id, available, tripCount, expectedCount, status })), [
    { id: "development", available: true, tripCount: 10, expectedCount: 10, status: "ready" },
    { id: "evaluator", available: false, tripCount: 0, expectedCount: 10, status: "not_configured" },
  ]);
  assert.equal(JSON.stringify(result).includes(privateSentinel), false);
});

test("two dataset sections keep exact expected_trip_count and normalize order", () => {
  const payload = datasetPayload({ expected_trip_count: 7, trip_count: 3, status: "not_ready" });
  payload.datasets.reverse();
  const result = normalizeDatasets(payload);
  assert.deepEqual(result.datasets.map(dataset => dataset.id), ["development", "evaluator"]);
  assert.equal(result.datasets[1].expectedCount, 7);
  assert.equal(result.datasets[1].tripCount, 3);
  assert.equal(result.datasets[1].available, true);
  assert(Object.isFrozen(DATASET_IDS));
});

test("dataset paths accept only the exact closed development/evaluator enum", () => {
  for (const id of DATASET_IDS) {
    assert.equal(normalizeDatasetId(id), id);
    assert.equal(datasetTripsPath(id), "/datasets/" + id + "/trips");
    assert.equal(datasetMatchPath(id), "/datasets/" + id + "/match");
  }
  for (const id of [undefined, null, 1, "", "new_dataset", "Development", " evaluator", "evaluator ", "../evaluator", "%65valuator", "evaluator/match", "__proto__"]) {
    for (const normalize of [normalizeDatasetId, datasetTripsPath, datasetMatchPath]) assert.throws(() => normalize(id), ApiError);
  }
});

test("dataset catalogue rejects missing, duplicate, unknown, and conflicting identities", () => {
  const invalid = [ {}, { datasets: [] }, { datasets: [datasetSummary("development")] },
    { datasets: [datasetSummary("development"), datasetSummary("development")] },
    { datasets: [datasetSummary("development"), datasetSummary("unknown")] },
    datasetPayload({ role: "development" }), datasetPayload({ available: undefined }),
    datasetPayload({ expected_trip_count: undefined }), datasetPayload({ trip_count: -1 }),
    datasetPayload({ status: "mystery" }), datasetPayload({ status: "not_configured" }),
    { datasets: [datasetSummary("development", { status: "not_configured" }), datasetSummary("evaluator")] },
  ];
  for (const payload of invalid) assert.throws(() => normalizeDatasets(payload), ApiError);
});

test("dataset trips require echoed dataset id and role, valid count, and string IDs", () => {
  for (const id of DATASET_IDS) {
    const payload = tripsPayload(id);
    const result = normalizeDatasetTrips(payload, id);
    assert.equal(result.datasetId, id);
    assert.equal(result.trips[0].id, "synthetic-0");
    for (const change of [{ dataset_id: undefined }, { dataset_role: undefined }, { dataset_role: "other" },
      { dataset_id: id === "development" ? "evaluator" : "development" }, { count: 9 }, { count: undefined }, { expected_count: undefined }]) {
      assert.throws(() => normalizeDatasetTrips({ ...payload, ...change }, id), ApiError);
    }
  }
  assert.throws(() => normalizeDatasetTrips({ ...tripsPayload("development", 1), trips: [{ trip_id: 4 }] }, "development"), ApiError);
});

test("dataset match identity is mandatory even when both sections share a trip ID", () => {
  const payload = matchPayload("development");
  assert.equal(normalizeMatch(payload, "synthetic-0", "development").datasetId, "development");
  assert.equal(normalizeMatch(matchPayload("evaluator"), "synthetic-0", "evaluator").datasetId, "evaluator");
  for (const change of [{ dataset_id: undefined }, { dataset_role: undefined }, { dataset_role: "evaluator" }, { dataset_id: "evaluator" }, { dataset_id: "unknown" }, { trip_id: "other" }]) {
    assert.throws(() => normalizeMatch({ ...payload, ...change }, "synthetic-0", "development"), ApiError);
  }
  const { dataset_id, dataset_role, ...legacy } = payload;
  assert.equal(normalizeMatch(legacy, "synthetic-0").datasetId, null);
  assert.equal(normalizeMatch(payload, "synthetic-0", "development").metrics.accuracy, null);
});

test("freshness checks reject same-trip switches, switch backs, reconnects, and old runs", () => {
  const request = { version: 1, apiBase: "https://api.example", datasetId: "development", tripId: "shared" };
  const current = { matchVersion: 1, apiBase: request.apiBase, datasetId: "development", selectedId: "shared" };
  assert.equal(isCurrentMatch(request, current), true);
  for (const change of [{ datasetId: "evaluator" }, { matchVersion: 3 }, { selectedId: "other" }, { apiBase: "https://other.example" }]) {
    assert.equal(isCurrentMatch(request, { ...current, ...change }), false);
  }
});

test("theme primitives default to dark and give accurate accessible toggle actions", () => {
  for (const value of [undefined, null, "", "dark", "LIGHT", "unexpected", {}]) assert.equal(normalizeTheme(value), "dark");
  assert.equal(normalizeTheme("light"), "light");
  assert.deepEqual(themeToggleState("dark"), { theme: "dark", pressed: false, action: "Switch to light theme" });
  assert.deepEqual(themeToggleState("light"), { theme: "light", pressed: true, action: "Switch to dark theme" });
  assert.equal(THEME_STORAGE_KEY, "pathsnap.theme");
});

test("dataset transport sends one exact trip_id field to each selected match endpoint", async () => {
  for (const id of DATASET_IDS) {
    const calls = [];
    await requestJson("https://api.example", datasetMatchPath(id), { method: "POST", body: { trip_id: "synthetic-0" },
      fetchImpl: async (url, options) => { calls.push({ url, options }); return { ok: true, status: 200, json: async () => matchPayload(id) }; },
    });
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, "https://api.example/datasets/" + id + "/match");
    assert.deepEqual(JSON.parse(calls[0].options.body), { trip_id: "synthetic-0" });
    assert.equal(calls[0].options.cache, "no-store");
  }
});

test("transport rejects an aborted late body even if a fetch implementation ignores abort", async () => {
  const controller = new AbortController();
  await assert.rejects(requestJson("https://api.example", "/datasets/development/match", { signal: controller.signal,
    fetchImpl: async () => ({ ok: true, status: 200, json: async () => { controller.abort(); return matchPayload("development"); } }),
  }), error => error.kind === "cancelled");
});

// Minimal in-memory DOM exercises app wiring without a server, browser, or data files.
const appSource = await readFile(new URL("../frontend/app.mjs", import.meta.url), "utf8");
const htmlSource = await readFile(new URL("../frontend/index.html", import.meta.url), "utf8");
function appHarness({ evaluator = {}, storage = new Map(), fetchOverride } = {}) {
  let document;
  class Node {
    constructor(tag = "div") { this.tagName = tag; this.children = []; this.dataset = {}; this.attributes = new Map(); this.listeners = {}; this.hidden = false; this.disabled = false; this.checked = false; this.textContent = "";
      const classes = new Set(); this.classList = { add: name => classes.add(name), remove: name => classes.delete(name), toggle: (name, enabled) => enabled ? classes.add(name) : classes.delete(name) };
    }
    setAttribute(name, value) { this.attributes.set(name, String(value)); }
    getAttribute(name) { return this.attributes.get(name) ?? null; }
    removeAttribute(name) { this.attributes.delete(name); }
    append(...nodes) { for (const node of nodes) { if (node.tagName === "fragment") this.children.push(...node.children); else this.children.push(node); } }
    replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
    addEventListener(type, callback) { (this.listeners[type] ??= []).push(callback); }
    focus() { document.activeElement = this; }
    select() {}
    showModal() { this.open = true; }
    close() { this.open = false; }
    getClientRects() { return [1]; }
    fire(type, event = {}) { return this.listeners[type]?.map(callback => callback({ currentTarget: this, preventDefault() {}, ...event })); }
  }
  const nodes = new Map([...htmlSource.matchAll(/id="([^"]+)"/g)].map(([, id]) => [id, new Node()]));
  nodes.get("tab-development").dataset.dataset = "development";
  nodes.get("tab-evaluator").dataset.dataset = "evaluator";
  const descendants = node => [node, ...node.children.flatMap(descendants)];
  document = { readyState: "loading", documentElement: new Node("html"), activeElement: null,
    getElementById: id => { assert(nodes.has(id), "Missing DOM node " + id); return nodes.get(id); },
    createElement: tag => new Node(tag), createTextNode: value => Object.assign(new Node("text"), { textContent: value }), createDocumentFragment: () => new Node("fragment"),
    querySelectorAll: () => descendants(nodes.get("trip-list")).filter(node => node.name === "trip-choice"), addEventListener() {},
  };
  const requests = [];
  const response = payload => ({ ok: true, status: 200, json: async () => payload });
  const fetchImpl = async (url, options) => {
    const path = new URL(url).pathname;
    requests.push({ path, options });
    if (fetchOverride) { const override = fetchOverride(path, options); if (override !== undefined) return override; }
    if (path === "/health") return response({ status: "ready", model_loaded: true, model_name: "Synthetic frozen model", model_hash: "b".repeat(64), dataset_role: "development", trip_count: 10 });
    if (path === "/datasets") return response(datasetPayload(evaluator));
    const match = path.match(/^\/datasets\/(development|evaluator)\/(trips|match)$/);
    assert(match, "Unexpected request " + path);
    return response(match[2] === "trips" ? tripsPayload(match[1]) : matchPayload(match[1]));
  };
  const localStorage = { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) };
  const imported = appSource.replace(/^import \{[\s\S]*?\} from "\.\/api\.mjs";/, 'const { ' + Object.keys(api).join(', ') + ' } = api;');
  const context = { api: { ...api, requestJson: (base, path, options) => api.requestJson(base, path, { ...options, fetchImpl }) }, document, localStorage, AbortController,
    Intl, URL, performance, setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1, clearInterval() {}, requestAnimationFrame: callback => callback(), window: { addEventListener() {} } };
  vm.createContext(context);
  vm.runInContext(imported + '\n globalThis.driver = { init, connect, switchDataset, selectTrip, matchTrip, state };', context);
  return { ...context.driver, nodes, document, requests, storage, response, radios: () => document.querySelectorAll() };
}
async function settledApp() { for (let pass = 0; pass < 8; pass++) await new Promise(resolve => setImmediate(resolve)); }

test("app leaves evaluator disabled and never requests its trips while unconfigured", async () => {
  const app = appHarness({ evaluator: { available: false, trip_count: 0, status: "not_configured" } });
  await app.connect();
  assert.equal(app.state.ready, true);
  assert.equal(app.radios().length, 10);
  assert.equal(app.nodes.get("tab-evaluator").disabled, true);
  assert.match(app.nodes.get("count-evaluator").textContent, /^0 \/ 10 · Not configured$/);
  app.switchDataset("evaluator");
  assert.equal(app.requests.some(request => request.path.includes("/evaluator/")), false);
  assert.equal(app.nodes.get("model-badge-name").textContent, "Synthetic frozen model");
});

test("app switches separate radio catalogues with keyboard tab focus and clears prior output", async () => {
  const app = appHarness();
  app.init(); await settledApp();
  app.selectTrip("synthetic-0"); await app.matchTrip();
  assert.equal(app.nodes.get("result-dataset").textContent, "Student development (development)");
  assert.equal(app.nodes.get("metric-accuracy").textContent, "N/A");
  app.nodes.get("tab-development").fire("keydown", { key: "ArrowRight" });
  assert.equal(app.nodes.get("selected-trip").textContent, "No trip selected");
  assert.equal(app.nodes.get("result-request").textContent, "—");
  await settledApp();
  assert.equal(app.state.datasetId, "evaluator");
  assert.equal(app.document.activeElement, app.nodes.get("tab-evaluator"));
  assert.equal(app.nodes.get("tab-evaluator").getAttribute("aria-selected"), "true");
  assert.equal(app.nodes.get("tab-evaluator").tabIndex, 0);
  assert.equal(app.nodes.get("tab-development").tabIndex, -1);
  assert.equal(app.nodes.get("workspace-panel").getAttribute("aria-labelledby"), "tab-evaluator");
  assert.equal(app.radios().length, 10);
  app.selectTrip("synthetic-0"); await app.matchTrip();
  assert.equal(app.nodes.get("result-dataset").textContent, "Evaluator (evaluator)");
  assert.deepEqual(app.requests.filter(request => request.options.method === "POST").map(request => [request.path, JSON.parse(request.options.body)]), [
    ["/datasets/development/match", { trip_id: "synthetic-0" }], ["/datasets/evaluator/match", { trip_id: "synthetic-0" }],
  ]);
  app.nodes.get("tab-evaluator").fire("keydown", { key: "Home" }); await settledApp();
  assert.equal(app.state.datasetId, "development");
  app.nodes.get("tab-development").fire("keydown", { key: "End" }); await settledApp();
  assert.equal(app.state.datasetId, "evaluator");
});

test("late same-trip result cannot overwrite a new dataset run or its loading state", async () => {
  const pending = [];
  const app = appHarness({ fetchOverride: (path) => path.endsWith("/match") ? new Promise(resolve => pending.push({ path, resolve })) : undefined });
  await app.connect(); app.selectTrip("synthetic-0");
  const developmentRun = app.matchTrip();
  app.switchDataset("evaluator"); await settledApp(); app.selectTrip("synthetic-0");
  const evaluatorRun = app.matchTrip();
  pending[0].resolve(app.response(matchPayload("development"))); await developmentRun;
  assert.equal(app.state.datasetId, "evaluator");
  assert.equal(app.state.loading, true);
  assert.equal(app.nodes.get("map-loading").hidden, false);
  assert.equal(app.nodes.get("result-request").textContent, "—");
  pending[1].resolve(app.response(matchPayload("evaluator"))); await evaluatorRun;
  assert.equal(app.state.loading, false);
  assert.equal(app.nodes.get("result-request").textContent, "synthetic-evaluator");
  assert.equal(app.nodes.get("result-dataset").textContent, "Evaluator (evaluator)");
});

test("app restores local light preference and toggles dark without storing predictions", async () => {
  const storage = new Map([[THEME_STORAGE_KEY, "light"]]);
  const app = appHarness({ storage }); app.init(); await settledApp();
  assert.equal(app.document.documentElement.dataset.theme, "light");
  assert.equal(app.nodes.get("theme-toggle").getAttribute("aria-pressed"), "true");
  app.nodes.get("theme-toggle").fire("click");
  assert.equal(app.document.documentElement.dataset.theme, "dark");
  assert.equal(app.nodes.get("theme-toggle").getAttribute("aria-label"), "Switch to light theme");
  assert.deepEqual([...storage.entries()], [[THEME_STORAGE_KEY, "dark"]]);
  const noPreference = appHarness(); noPreference.init(); await settledApp();
  assert.equal(noPreference.document.documentElement.dataset.theme, "dark");
});
