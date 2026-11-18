/**
 * Public PathSnap API boundary. No dataset files, cached predictions, labels,
 * reference geometry, or local matching logic belong in this module.
 *
 * Agreed service contract: GET /health; GET /trips; POST /match {trip_id}.
 * Only documented scalar metadata and geometry are retained by the adapter.
 */
export const API_STORAGE_KEY = "pathsnap.apiBaseUrl";
export const MATCH_TIMEOUT_MS = 180_000;
export const CATALOG_TIMEOUT_MS = 15_000;
const MAX_COORDINATES = 250_000;
const MAX_FEATURES = 20_000;

export class ApiError extends Error {
  constructor(message, { kind = "response", status = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

export function defaultApiBase(locationLike = globalThis.location) {
  if (locationLike && /^https?:$/.test(locationLike.protocol)) return locationLike.origin;
  return "http://127.0.0.1:8000";
}

export function normalizeApiBase(value, locationLike = globalThis.location) {
  const raw = String(value ?? "").trim() || defaultApiBase(locationLike);
  let url;
  try { url = new URL(raw); } catch {
    throw new ApiError("Enter a complete API URL beginning with http:// or https://.", { kind: "configuration" });
  }
  if (!/^https?:$/.test(url.protocol) || !url.hostname) {
    throw new ApiError("The API address must use HTTP or HTTPS.", { kind: "configuration" });
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new ApiError("Use an API base URL without credentials, a query, or a fragment.", { kind: "configuration" });
  }
  if (locationLike?.protocol === "https:" && url.protocol !== "https:") {
    throw new ApiError("An HTTPS website needs an HTTPS API. Use this website’s default API or another HTTPS address.", { kind: "configuration" });
  }
  return url.href.replace(/\/+$/, "");
}

export function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function shortText(value, maximum = 180) {
  return typeof value === "string" ? value.replace(/[\u0000-\u001F\u007F]/g, " ").trim().slice(0, maximum) : "";
}

export function nonNegative(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
}

export function ratio(value) {
  const number = nonNegative(value);
  return number !== null && number <= 1 ? number : null;
}

function integer(value) {
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function httpError(status) {
  const messages = {
    400: "The API could not accept this request. Refresh the trip list and try again.",
    404: "The API endpoint was not found. Check the base URL in API settings.",
    413: "The API rejected the request size. The client sends only a trip ID; check the server configuration.",
    422: "The API rejected this trip ID. Refresh the trip list, choose a returned trip, and try again.",
    429: "The model server is busy. Wait for the current request to finish, then try again.",
    500: "The model server could not complete this request. Check its logs, then retry.",
    502: "The website could not reach the model server. Check the API service and tunnel, then retry.",
    503: "The model API is not ready. Check model loading and the trip catalogue on the server, then reconnect.",
    504: "The website’s gateway timed out. The server may still be processing; check its logs before retrying.",
  };
  // Do not interpolate server error details: validation lists can include input
  // values and operational information not intended for this UI.
  return new ApiError(messages[status] || "The API returned HTTP " + status + ". Check the model server and retry.", { kind: "http", status });
}

export async function requestJson(base, path, {
  method = "GET", body, signal, timeoutMs = CATALOG_TIMEOUT_MS,
  fetchImpl = globalThis.fetch,
} = {}) {
  const controller = new AbortController();
  let timedOut = false;
  const abort = () => controller.abort();
  if (signal?.aborted) controller.abort();
  else signal?.addEventListener("abort", abort, { once: true });
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  try {
    const response = await fetchImpl(base + path, {
      method,
      headers: body === undefined ? { Accept: "application/json" } : { Accept: "application/json", "Content-Type": "application/json" },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      cache: "no-store",
      credentials: "same-origin",
      signal: controller.signal,
    });
    let payload;
    try { payload = await response.json(); } catch {
      if (!response.ok) throw httpError(response.status);
      throw new ApiError("The API returned a non-JSON response. Check that the base URL points to the model API, not a static file server.");
    }
    if (!response.ok) throw httpError(response.status);
    if (!isRecord(payload)) throw new ApiError("The API response must be a JSON object. Check the documented service contract.");
    return payload;
  } catch (error) {
    if (timedOut) throw new ApiError("No response within " + Math.round(timeoutMs / 1000) + " seconds. The server may still be processing. Check its logs before trying again.", { kind: "timeout" });
    if (signal?.aborted || controller.signal.aborted) throw new ApiError("Stopped waiting for the API. The server may still finish this request.", { kind: "cancelled" });
    if (error instanceof ApiError) throw error;
    throw new ApiError("The model API could not be reached. Check the service, address, network or tunnel, and CORS when using a separate API host.", { kind: "network" });
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abort);
  }
}

export function normalizeHealth(payload) {
  if (!isRecord(payload) || typeof payload.model_loaded !== "boolean") {
    throw new ApiError("The health response is missing model_loaded. Check that this address is the PathSnap API.");
  }
  const status = ["ready", "not_ready", "error"].includes(payload.status) ? payload.status : "not_ready";
  return {
    ready: status === "ready" && payload.model_loaded,
    status,
    modelLoaded: payload.model_loaded,
    modelName: shortText(payload.model_name),
    algorithm: shortText(payload.algorithm),
    modelHash: shortText(payload.model_hash, 64),
    datasetRole: ["development", "evaluator"].includes(payload.dataset_role) ? payload.dataset_role : "",
    tripCount: integer(payload.trip_count),
    expectedCount: integer(payload.expected_trip_count) ?? 10,
  };
}

export function normalizeTrips(payload) {
  if (!isRecord(payload) || !Array.isArray(payload.trips)) {
    throw new ApiError("The trip response is missing its trips array. Check the API contract.");
  }
  const seen = new Set();
  const trips = payload.trips.map((entry) => {
    if (!isRecord(entry) || !["string", "number"].includes(typeof entry.trip_id)) {
      throw new ApiError("The API returned a trip without a valid trip_id.");
    }
    const id = String(entry.trip_id);
    if (!id.trim() || id.length > 240 || (typeof entry.trip_id === "number" && !Number.isSafeInteger(entry.trip_id))) {
      throw new ApiError("The API returned an invalid trip ID. Refresh after checking the server catalogue.");
    }
    if (seen.has(id)) throw new ApiError("The API returned duplicate trip IDs. Check the server catalogue.");
    seen.add(id);
    return { id, requestId: entry.trip_id, observationCount: integer(entry.observation_count), durationSeconds: nonNegative(entry.duration_s) };
  });
  return {
    trips,
    expectedCount: integer(payload.expected_count) ?? 10,
    datasetRole: ["development", "evaluator"].includes(payload.dataset_role) ? payload.dataset_role : "",
  };
}

/** Clone only permitted geometry; GeoJSON properties are never rendered. */
export function normalizeGeoJSON(input, label) {
  let coordinateCount = 0;
  const position = (value) => {
    if (!Array.isArray(value) || value.length < 2 || !value.slice(0, 2).every((n) => typeof n === "number" && Number.isFinite(n))) {
      throw new ApiError("The API returned invalid coordinates in " + label + ".");
    }
    if (Math.abs(value[0]) > 180 || Math.abs(value[1]) > 90) {
      throw new ApiError("The API returned out-of-range coordinates in " + label + ". GeoJSON must use [longitude, latitude].");
    }
    if (++coordinateCount > MAX_COORDINATES) throw new ApiError("The " + label + " geometry is too large to display safely.");
    return value.slice(0, 2);
  };
  const line = (value) => {
    if (!Array.isArray(value) || value.length < 2) throw new ApiError("The API returned an invalid LineString in " + label + ".");
    return value.map(position);
  };
  const geometry = (value) => {
    if (!isRecord(value)) throw new ApiError("The API returned no valid geometry in " + label + ".");
    switch (value.type) {
      case "Point": return { type: "Point", coordinates: position(value.coordinates) };
      case "MultiPoint":
        if (!Array.isArray(value.coordinates)) break;
        return { type: "MultiPoint", coordinates: value.coordinates.map(position) };
      case "LineString": return { type: "LineString", coordinates: line(value.coordinates) };
      case "MultiLineString":
        if (!Array.isArray(value.coordinates)) break;
        return { type: "MultiLineString", coordinates: value.coordinates.map(line) };
      default: break;
    }
    throw new ApiError("Unsupported geometry in " + label + ". Expected Point, MultiPoint, LineString or MultiLineString.");
  };
  const feature = (value) => {
    if (!isRecord(value)) throw new ApiError("The API returned an invalid feature in " + label + ".");
    return { type: "Feature", properties: {}, geometry: geometry(value.type === "Feature" ? value.geometry : value) };
  };
  if (!isRecord(input)) throw new ApiError("The API response is missing " + label + " GeoJSON.");
  if (input.type === "FeatureCollection") {
    if (!Array.isArray(input.features) || input.features.length > MAX_FEATURES) throw new ApiError("The API returned an invalid feature collection in " + label + ".");
    return { type: "FeatureCollection", features: input.features.map(feature) };
  }
  return { type: "FeatureCollection", features: [feature(input)] };
}

export function geometryPositions(collection) {
  const points = [];
  for (const { geometry } of collection.features) {
    if (geometry.type === "Point") points.push(geometry.coordinates);
    else if (["LineString", "MultiPoint"].includes(geometry.type)) for (const point of geometry.coordinates) points.push(point);
    else if (geometry.type === "MultiLineString") for (const line of geometry.coordinates) for (const point of line) points.push(point);
  }
  return points;
}

export function geometryParts(collection) {
  return collection.features.reduce((count, { geometry }) => count + (geometry.type === "MultiLineString" ? geometry.coordinates.length : 1), 0);
}

export const METRIC_DEFINITIONS = [
  ["accuracy", "HMM observation accuracy", "ratio"],
  ["precision", "HMM precision", "ratio"],
  ["recall", "HMM recall", "ratio"],
  ["macro_f1", "HMM macro F1", "ratio"],
  ["top1_accuracy", "Classifier Top-1 accuracy", "ratio"],
  ["candidate_miss_rate", "Candidate miss rate", "ratio"],
  ["log_loss", "Log loss", "number"],
  ["route_continuity", "Route continuity", "ratio"],
  ["route_edge_f1", "Route edge F1", "ratio"],
  ["route_length_f1", "Route length F1", "ratio"],
  ["sequence_similarity", "Sequence similarity", "ratio"],
  ["classifier_accuracy", "Classifier observation accuracy", "ratio"],
  ["classifier_precision", "Classifier precision", "ratio"],
  ["classifier_recall", "Classifier recall", "ratio"],
  ["classifier_macro_f1", "Classifier macro F1", "ratio"],
];

export function normalizeMatch(payload, selectedId) {
  if (!isRecord(payload)) throw new ApiError("The match response is not a JSON object.");
  if (String(payload.trip_id) !== String(selectedId)) throw new ApiError("The API returned a different trip. The result was not displayed; refresh the catalogue and retry.");
  const metricsIn = isRecord(payload.metrics) ? payload.metrics : {};
  const timingsIn = isRecord(payload.timings) ? payload.timings : {};
  const modelIn = isRecord(payload.model) ? payload.model : {};
  const metrics = {};
  for (const [key, , type] of METRIC_DEFINITIONS) {
    if (key.startsWith("classifier_") && !(key in metricsIn)) continue;
    metrics[key] = (type === "ratio" ? ratio : nonNegative)(metricsIn[key]);
  }
  const notes = Array.isArray(payload.metric_notes) ? payload.metric_notes.filter((note) => typeof note === "string").map((note) => shortText(note, 1000)).filter(Boolean).slice(0, 20) : [];
  return {
    tripId: String(payload.trip_id),
    requestId: shortText(payload.request_id, 140),
    predictedRoute: normalizeGeoJSON(payload.predicted_route, "predicted_route"),
    rawGps: normalizeGeoJSON(payload.raw_gps, "raw_gps"),
    metrics,
    metricsStatus: ["ok", "partial", "unavailable", "error"].includes(payload.metrics_status) ? payload.metrics_status : "unspecified",
    metricNotes: notes,
    proxyReference: payload.proxy_reference === true,
    timings: { inferenceMs: nonNegative(timingsIn.inference_ms), scoringMs: nonNegative(timingsIn.scoring_ms), totalMs: nonNegative(timingsIn.total_ms) },
    model: { name: shortText(modelIn.name), algorithm: shortText(modelIn.algorithm), sha256: shortText(modelIn.sha256, 64) },
  };
}
