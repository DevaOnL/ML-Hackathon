import assert from "node:assert/strict";
import test from "node:test";
import {
  ApiError, defaultApiBase, normalizeApiBase, normalizeHealth, normalizeTrips,
  normalizeGeoJSON, geometryPositions, geometryParts, normalizeMatch, requestJson,
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
