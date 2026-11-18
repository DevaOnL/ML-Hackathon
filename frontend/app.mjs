import {
  API_STORAGE_KEY, THEME_STORAGE_KEY, MATCH_TIMEOUT_MS, ApiError, METRIC_DEFINITIONS,
  defaultApiBase, normalizeApiBase, requestJson, normalizeHealth, normalizeDatasets,
  DATASET_IDS, normalizeDatasetTrips, datasetTripsPath, datasetMatchPath,
  normalizeMatch, geometryPositions, geometryParts, isCurrentMatch, normalizeTheme, themeToggleState,
} from "./api.mjs";

const $ = (id) => document.getElementById(id);
const numberFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });
const decimalFormat = new Intl.NumberFormat(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const state = {
  apiBase: defaultApiBase(), health: null, trips: [], selectedId: null,
  datasets: [], datasetId: "development", matchVersion: 0,
  ready: false, connecting: false, loading: false, result: null,
  connectVersion: 0, connectController: null, matchController: null,
  map: null, routeLayer: null, gpsLayer: null, mapBounds: null,
  basemap: null, basemapTimer: null, elapsedTimer: null, toastTimer: null,
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function text(id, value) { $(id).textContent = value; }
function announce(message) { text("live-status", message); }
function selectedTrip() { return state.trips.find((trip) => trip.id === state.selectedId) || null; }
function selectedDataset() { return state.datasets.find((dataset) => dataset.id === state.datasetId) || null; }
function datasetLabel() { return selectedDataset()?.label || (state.datasetId === "evaluator" ? "Evaluator" : "Student development"); }
function formatCount(value) { return value === null ? "N/A" : numberFormat.format(value); }
function formatPercent(value) { return value === null ? "N/A" : decimalFormat.format(value * 100) + "%"; }
function durationLabel(value) {
  if (value === null) return null;
  const rounded = Math.round(value);
  if (rounded < 60) return rounded + " s";
  const minutes = Math.floor(rounded / 60);
  const seconds = rounded % 60;
  return seconds ? minutes + "m " + seconds + "s" : minutes + " min";
}
function timingParts(value) {
  if (value === null) return { value: "N/A", unit: "" };
  if (value < 1000) return { value: value < 10 ? decimalFormat.format(value) : numberFormat.format(value), unit: "ms" };
  return { value: decimalFormat.format(value / 1000), unit: "s" };
}
function timingLabel(value) {
  const parts = timingParts(value);
  return parts.value + (parts.unit ? " " + parts.unit : "");
}
function metricValue(id, value, type) {
  const node = $(id);
  node.replaceChildren();
  node.classList.toggle("is-na", value === null);
  if (type === "time") {
    const parts = timingParts(value);
    node.append(document.createTextNode(parts.value));
    if (parts.unit) node.append(element("small", "", parts.unit));
  } else node.textContent = formatPercent(value);
}
function toast(message) {
  clearTimeout(state.toastTimer);
  text("toast", message);
  $("toast").hidden = false;
  state.toastTimer = setTimeout(() => { $("toast").hidden = true; }, 6500);
}
function setConnection(label, status) {
  text("connection-label", label);
  $("connection-badge").dataset.state = status;
}
function connectionNotice(title, copy) {
  text("connection-notice-title", title);
  text("connection-notice-copy", copy);
  $("connection-notice").hidden = false;
}
function runNotice(title, copy, tone = "warning") {
  text("run-notice-title", title);
  text("run-notice-copy", copy);
  $("run-notice").dataset.tone = tone;
  $("run-notice").hidden = false;
}
function workflow(current) {
  const ids = ["step-select", "step-run", "step-review"];
  ids.forEach((id, index) => {
    const node = $(id);
    node.classList.toggle("is-current", index === current);
    node.classList.toggle("is-done", index < current);
    if (index === current) node.setAttribute("aria-current", "step");
    else node.removeAttribute("aria-current");
  });
}
function updateApiLabel() {
  const url = new URL(state.apiBase);
  text("api-host", url.host + (url.pathname === "/" ? "" : url.pathname));
  $("api-host").title = state.apiBase;
}
function updateControls() {
  const selected = selectedTrip();
  $("match-button").disabled = !state.ready || !selected || state.loading || state.connecting;
  $("match-button-label").textContent = state.loading ? "Reconstructing…" : state.result ? "Run again" : "Reconstruct route";
  $("match-button-arrow").hidden = state.loading;
  $("match-button-spinner").hidden = !state.loading;
  $("cancel-match").hidden = !state.loading;
  $("refresh-trips").disabled = state.loading || state.connecting;
  $("retry-connection").disabled = state.loading || state.connecting;
  $("open-settings").disabled = state.loading;
  $("fit-map").disabled = !state.map || !state.mapBounds || state.loading;
  $("show-route").disabled = !state.routeLayer || state.loading;
  $("show-gps").disabled = !state.gpsLayer || state.loading;
  $("show-basemap").disabled = !state.map;
  for (const radio of document.querySelectorAll('input[name="trip-choice"]')) radio.disabled = state.loading || state.connecting;
  renderDatasetTabs();
  if (state.loading) text("run-help", "Live request · up to " + MATCH_TIMEOUT_MS / 1000 + " seconds");
  else if (state.connecting) text("run-help", "Checking the API and available trips…");
  else if (!state.ready) text("run-help", "Reconnect when the model API is ready.");
  else text("run-help", "Inference and scoring run on the server.");
}

function emptyTripList(title, copy, isLoading = false) {
  const wrapper = element("div", "list-state");
  if (isLoading) {
    const spinner = element("span", "spinner");
    spinner.setAttribute("aria-hidden", "true");
    wrapper.append(spinner);
  }
  wrapper.append(element("strong", "", title));
  if (copy) wrapper.append(element("p", "", copy));
  $("trip-list").replaceChildren(wrapper);
}
function renderTrips() {
  if (!state.trips.length) {
    emptyTripList("No trips returned", "Check the API catalogue, then refresh.");
    return;
  }
  const fragment = document.createDocumentFragment();
  state.trips.forEach((trip, index) => {
    const label = element("label", "trip-option");
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "trip-choice";
    radio.value = trip.id;
    radio.checked = trip.id === state.selectedId;
    radio.addEventListener("change", () => selectTrip(trip.id));
    const ordinal = element("span", "trip-ordinal", String(index + 1).padStart(2, "0"));
    ordinal.setAttribute("aria-hidden", "true");
    const copy = element("span", "trip-copy");
    copy.append(element("span", "trip-id", trip.id));
    const metadata = [];
    if (trip.observationCount !== null) metadata.push(formatCount(trip.observationCount) + " GPS points");
    if (trip.durationSeconds !== null) metadata.push(durationLabel(trip.durationSeconds));
    copy.append(element("span", "trip-metadata", metadata.join(" · ") || "Ready for reconstruction"));
    const check = element("span", "trip-radio");
    check.setAttribute("aria-hidden", "true");
    label.append(radio, ordinal, copy, check);
    fragment.append(label);
  });
  $("trip-list").replaceChildren(fragment);
}
function selectTrip(id) {
  if (state.loading || state.selectedId === id) return;
  state.selectedId = id;
  ++state.matchVersion;
  resetResult();
  updateSelection();
  announce("Selected " + id + ". " + (state.ready ? "Ready to reconstruct." : "The API must be ready before running."));
}
function updateSelection() {
  const trip = selectedTrip();
  text("selected-trip", trip?.id || "No trip selected");
  const count = trip?.observationCount;
  text("selection-detail", trip ? count === null ? "Ready to send to the model API." : formatCount(count) + " original GPS observations" : "Choose a trip above to begin.");
  workflow(trip ? 1 : 0);
  updateControls();
}

function updateModelBadge() {
  const health = state.health;
  text("model-badge-name", health?.modelName || "Not reported");
  text("model-badge-hash", health?.modelHash ? health.modelHash.slice(0, 12) : "");
  $("model-badge").title = health ? [health.modelName, health.algorithm, health.modelHash].filter(Boolean).join(" · ") || "Model metadata not reported by /health" : "Waiting for model metadata from /health";
}
function renderDatasetTabs() {
  for (const id of DATASET_IDS) {
    const dataset = state.datasets.find((entry) => entry.id === id);
    const tab = $("tab-" + id);
    const selected = id === state.datasetId;
    tab.disabled = !dataset?.available;
    tab.tabIndex = selected && !tab.disabled ? 0 : -1;
    tab.setAttribute("aria-selected", String(selected));
    const availability = !dataset ? "Not checked" : dataset.status === "not_configured" ? "Not configured" : !dataset.available ? "Unavailable" : "";
    text("count-" + id, dataset ? dataset.tripCount + " / " + dataset.expectedCount + (availability ? " · " + availability : "") : "Not checked");
    tab.title = availability || dataset.label;
    tab.setAttribute("aria-label", (id === "development" ? "Student development" : "Evaluator") + ": " + (dataset ? dataset.tripCount + " trips available; " + dataset.expectedCount + " expected" + (availability ? ". " + availability : "") : "availability not checked"));
  }
  $("workspace-panel").setAttribute("aria-labelledby", "tab-" + state.datasetId);
  $("workspace-panel").setAttribute("aria-busy", String(state.connecting));
  text("dataset-label", datasetLabel() + " trips");
  const evaluator = state.datasets.find((entry) => entry.id === "evaluator");
  text("dataset-mode-note", !evaluator ? "Dataset availability has not been confirmed." : evaluator.status === "not_configured" ? "Evaluator not configured on the server. Development trips remain separate." : !evaluator.available ? "Evaluator unavailable. Check the server, then refresh." : "Separate trip catalogues. Both use the same frozen model.");
}
function stopPendingMatch() {
  ++state.matchVersion;
  state.matchController?.abort();
  state.matchController = null;
  clearInterval(state.elapsedTimer);
  state.elapsedTimer = null;
  state.loading = false;
  $("map-loading").hidden = true;
  $("results-section").setAttribute("aria-busy", "false");
}
function switchDataset(id) {
  const dataset = state.datasets.find((entry) => entry.id === id);
  if (!dataset?.available || state.datasetId === id) return;
  if (state.loading) toast("Stopped waiting for the previous dataset. The server may still be processing that request.");
  state.datasetId = id;
  state.selectedId = null;
  void connect();
}
function datasetTabKeydown(event) {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  const tabs = DATASET_IDS.map((id) => $("tab-" + id)).filter((tab) => !tab.disabled);
  if (!tabs.length) return;
  event.preventDefault();
  const current = tabs.indexOf(event.currentTarget);
  const index = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (current + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
  tabs[index].focus();
  switchDataset(tabs[index].dataset.dataset);
}
function applyTheme(value) {
  const preference = themeToggleState(value);
  document.documentElement.dataset.theme = preference.theme;
  $("theme-toggle").setAttribute("aria-pressed", String(preference.pressed));
  $("theme-toggle").setAttribute("aria-label", preference.action);
  $("theme-toggle").title = preference.action;
  text("theme-toggle-label", preference.theme === "light" ? "dark" : "light");
}

async function connect() {
  stopPendingMatch();
  state.connectController?.abort();
  const controller = new AbortController();
  const version = ++state.connectVersion;
  const base = state.apiBase;
  state.connectController = controller;
  state.connecting = true;
  state.ready = false;
  state.health = null;
  state.trips = [];
  resetResult();
  updateModelBadge();
  setConnection("Connecting to API", "connecting");
  $("connection-notice").hidden = true;
  $("trip-list").setAttribute("aria-busy", "true");
  text("trip-count", "Not loaded");
  $("trip-count").classList.remove("is-warning");
  emptyTripList("Loading available trips…", "Checking model and dataset readiness.", true);
  updateSelection();
  announce("Connecting to the model API and loading dataset availability.");
  const [healthResult, datasetsResult] = await Promise.allSettled([
    requestJson(base, "/health", { signal: controller.signal }).then(normalizeHealth),
    requestJson(base, "/datasets", { signal: controller.signal }).then(normalizeDatasets),
  ]);
  if (version !== state.connectVersion) return;
  const issues = [];
  const missingHealth = healthResult.status !== "fulfilled";
  if (!missingHealth) {
    state.health = healthResult.value;
    if (!state.health.ready) issues.push(state.health.modelLoaded ? "The API is reachable, but its model or configuration is not ready." : "The API is reachable, but its trained model is not loaded. Load the model on the server, then reconnect.");
  } else issues.push(healthResult.reason instanceof ApiError ? healthResult.reason.message : "The health check failed.");
  updateModelBadge();
  if (datasetsResult.status === "fulfilled") {
    state.datasets = datasetsResult.value.datasets;
    if (!selectedDataset()?.available) {
      const fallback = state.datasets.find((entry) => entry.available);
      if (fallback && fallback.id !== state.datasetId) { state.datasetId = fallback.id; state.selectedId = null; }
    }
    const dataset = selectedDataset();
    renderDatasetTabs();
    if (dataset?.available) {
      const requestedDatasetId = dataset.id;
      try {
        const payload = await requestJson(base, datasetTripsPath(requestedDatasetId), { signal: controller.signal });
        if (version !== state.connectVersion) return;
        const catalogue = normalizeDatasetTrips(payload, requestedDatasetId);
        state.trips = catalogue.trips;
        text("trip-count", state.trips.length + " / " + catalogue.expectedCount);
        $("trip-count").setAttribute("aria-label", state.trips.length + " trips returned; " + catalogue.expectedCount + " expected");
        const complete = state.trips.length === 10 && catalogue.expectedCount === 10;
        $("trip-count").classList.toggle("is-warning", !complete);
        if (!complete) issues.push("The API returned " + state.trips.length + " trips with " + catalogue.expectedCount + " expected; this section requires 10. All returned options are shown. Check the catalogue before running.");
        if (state.trips.length !== dataset.tripCount || catalogue.expectedCount !== dataset.expectedCount) issues.push("The dataset summary and trip catalogue disagree on the trip count. Refresh after checking the server.");
        if (dataset.status !== "ready") issues.push(dataset.label + " is registered but reports " + dataset.status + ". Check the server, then refresh.");
        if (state.health?.datasetRole === requestedDatasetId && state.health.tripCount !== null && state.health.tripCount !== state.trips.length) issues.push("Health and the selected trip catalogue disagree on the trip count.");
        renderTrips();
      } catch (error) {
        if (version !== state.connectVersion) return;
        issues.push(error instanceof ApiError ? error.message : "The trip catalogue could not be loaded.");
        emptyTripList("Could not load trips", "Retry the connection or check API settings.");
      }
    } else {
      issues.push(dataset?.status === "not_configured" ? "The evaluator dataset is not configured on the server." : "The selected dataset is unavailable on the server.");
      text("trip-count", dataset ? dataset.tripCount + " / " + dataset.expectedCount : "Not loaded");
      emptyTripList("Dataset unavailable", "No trips were loaded. Check server configuration, then refresh.");
    }
  } else {
    state.datasets = [];
    issues.push(datasetsResult.reason instanceof ApiError ? datasetsResult.reason.message : "Dataset availability could not be loaded.");
    emptyTripList("Could not load datasets", "This interface requires the dataset API. Check the server, then reconnect.");
  }
  if (!state.trips.some((trip) => trip.id === state.selectedId)) state.selectedId = null;
  state.connectController = null;
  state.connecting = false;
  state.ready = issues.length === 0;
  $("trip-list").setAttribute("aria-busy", "false");
  if (state.ready) {
    setConnection("API ready", "ready");
    announce(datasetLabel() + " ready. All " + state.trips.length + " trips are available. Choose a trip to reconstruct.");
  } else {
    setConnection(missingHealth ? "API unavailable" : "API not ready", missingHealth ? "offline" : "warning");
    connectionNotice(missingHealth ? "Cannot reach the model API" : "The API needs attention", [...new Set(issues)].join(" "));
    announce("The API is not ready. " + issues[0]);
  }
  updateSelection();
}

function initMap() {
  if (!globalThis.L) {
    $("map-empty").hidden = true;
    $("map-failure").hidden = false;
    return;
  }
  const L = globalThis.L;
  try {
    state.map = L.map("route-map", { zoomControl: false, preferCanvas: true, attributionControl: true, worldCopyJump: true, minZoom: 1, maxZoom: 20 }).setView([0, 0], 2);
    state.map.attributionControl.setPrefix('<a href="https://leafletjs.com" target="_blank" rel="noopener noreferrer">Leaflet</a>');
    L.control.zoom({ position: "topright" }).addTo(state.map);
    L.control.scale({ imperial: false, position: "bottomright", maxWidth: 110 }).addTo(state.map);
    if (globalThis.ResizeObserver) new ResizeObserver(() => state.map?.invalidateSize({ pan: false })).observe($("map-stage"));
  } catch {
    state.map = null;
    $("map-empty").hidden = true;
    $("map-failure").hidden = false;
  }
}
function clearMap() {
  if (state.map) {
    if (state.routeLayer) state.map.removeLayer(state.routeLayer);
    if (state.gpsLayer) state.map.removeLayer(state.gpsLayer);
  }
  state.routeLayer = null;
  state.gpsLayer = null;
  state.mapBounds = null;
  $("map-empty").hidden = !state.map;
  $("map-run-label").hidden = true;
  $("map-endpoints").hidden = true;
  text("map-heading", "Route geometry");
  text("map-state-badge", "Awaiting a trip");
  $("map-state-badge").dataset.state = "";
  $("route-map").setAttribute("aria-label", "Interactive route map. No route loaded. Use arrow keys to pan, plus and minus to zoom.");
  $("show-route").checked = true;
  $("show-gps").checked = true;
}
function fitMap() {
  if (!state.map || !state.mapBounds) return;
  state.map.invalidateSize({ pan: false });
  state.map.fitBounds(state.mapBounds, { padding: [42, 42], maxZoom: 17, animate: false });
}
function addEndpoint(group, coordinate, label, isEnd) {
  const L = globalThis.L;
  const color = isEnd ? "#1d4ed8" : "#1b1b1b";
  const marker = L.circleMarker([coordinate[1], coordinate[0]], { radius: 9, color: "#ffffff", weight: 2, fillColor: color, fillOpacity: 1 });
  const tooltip = element("span", "", label);
  marker.bindTooltip(tooltip, { permanent: true, direction: "center", opacity: 1, className: "endpoint-tooltip" + (isEnd ? " endpoint-tooltip-end" : "") });
  marker.addTo(group);
}
function drawMap(result) {
  const routePoints = geometryPositions(result.predictedRoute);
  const gpsPoints = geometryPositions(result.rawGps);
  const parts = geometryParts(result.predictedRoute);
  if (!state.map) return { routePoints, gpsPoints, parts, mapUnavailable: true };
  const L = globalThis.L;
  const routeGroup = L.featureGroup();
  const gpsGroup = L.featureGroup();
  if (routePoints.length) {
    L.geoJSON(result.predictedRoute, {
      style: { color: "#ffffff", weight: 8, opacity: .93, lineCap: "round", lineJoin: "round" },
      pointToLayer: (_feature, latlng) => L.circleMarker(latlng, { radius: 7, color: "white", weight: 2, fillColor: "#3e7bf5", fillOpacity: 1 }),
      interactive: false,
    }).addTo(routeGroup);
    L.geoJSON(result.predictedRoute, {
      style: { color: "#3e7bf5", weight: 4, opacity: .96, lineCap: "round", lineJoin: "round" },
      pointToLayer: (_feature, latlng) => L.circleMarker(latlng, { radius: 4, color: "#3e7bf5", weight: 1, fillColor: "#3e7bf5", fillOpacity: 1 }),
      interactive: false,
    }).addTo(routeGroup);
  }
  if (gpsPoints.length) {
    L.geoJSON(result.rawGps, {
      style: { color: "#cf8c2f", weight: 1.7, opacity: .78, dashArray: "4 7" },
      pointToLayer: (_feature, latlng) => L.circleMarker(latlng, { radius: 3, color: "#cf8c2f", weight: 1.2, fillColor: "#fff8ee", fillOpacity: .95 }),
      interactive: false,
    }).addTo(gpsGroup);
    // The complete original line is always displayed. For unusually dense
    // traces, thin only the decorative point markers to keep interaction fluid.
    const stride = Math.max(1, Math.ceil(gpsPoints.length / 5000));
    gpsPoints.forEach((coordinate, index) => {
      if (index % stride && index !== gpsPoints.length - 1) return;
      const marker = L.circleMarker([coordinate[1], coordinate[0]], { radius: 3, color: "#cf8c2f", weight: 1.1, fillColor: "#fff8ee", fillOpacity: .95 });
      const label = element("span", "", "GPS observation " + (index + 1) + " · " + coordinate[1].toFixed(5) + ", " + coordinate[0].toFixed(5));
      marker.bindTooltip(label, { direction: "top" });
      marker.addTo(gpsGroup);
    });
  }
  const endpoints = routePoints.length ? routePoints : gpsPoints;
  if (endpoints.length) {
    const endpointGroup = routePoints.length ? routeGroup : gpsGroup;
    addEndpoint(endpointGroup, endpoints[0], endpoints.length === 1 ? "S/E" : "S", false);
    if (endpoints.length > 1) addEndpoint(endpointGroup, endpoints[endpoints.length - 1], "E", true);
  }
  if (gpsPoints.length) state.gpsLayer = gpsGroup.addTo(state.map);
  if (routePoints.length) state.routeLayer = routeGroup.addTo(state.map);
  // Add route above the noisy trace but keep GPS observation dots readable.
  if (state.routeLayer) state.routeLayer.bringToFront();
  const allPoints = [...routePoints, ...gpsPoints];
  if (allPoints.length) {
    const bounds = L.latLngBounds(allPoints.map((p) => [p[1], p[0]]));
    state.mapBounds = bounds;
    $("map-empty").hidden = true;
    text("map-heading", "Route · " + result.tripId);
    text("map-run-label-text", formatCount(gpsPoints.length) + " GPS observations · " + parts + (parts === 1 ? " route part" : " route parts"));
    $("map-run-label").hidden = false;
    $("map-endpoints").hidden = !endpoints.length;
    $("route-map").setAttribute("aria-label", "Route for " + result.tripId + ": " + parts + " predicted route parts and " + gpsPoints.length + " original GPS observations. Use layer checkboxes to compare. Arrow keys pan; plus and minus zoom.");
    requestAnimationFrame(fitMap);
  } else {
    text("map-empty-title", "The API returned no drawable geometry");
    text("map-empty-copy", "The execution record below is still available. Check the server’s geometry output before retrying.");
  }
  return { routePoints, gpsPoints, parts, mapUnavailable: false };
}
function toggleLayer(kind) {
  const group = kind === "route" ? state.routeLayer : state.gpsLayer;
  const checked = $(kind === "route" ? "show-route" : "show-gps").checked;
  if (!state.map || !group) return;
  if (checked) group.addTo(state.map);
  else state.map.removeLayer(group);
}
function disableBasemap(message = "Geometry view · offline ready") {
  clearTimeout(state.basemapTimer);
  if (state.basemap && state.map) state.map.removeLayer(state.basemap);
  state.basemap = null;
  $("show-basemap").checked = false;
  text("basemap-status", message);
}
function toggleBasemap() {
  if (!state.map) return;
  if (!$("show-basemap").checked) { disableBasemap(); return; }
  if (state.basemap) return;
  text("basemap-status", "Loading optional street tiles…");
  let loaded = false;
  const fail = () => {
    if (!state.basemap) return;
    disableBasemap("Tiles unavailable · geometry remains visible");
    toast("Street tiles could not load. The route and GPS still work in geometry view.");
  };
  state.basemap = globalThis.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors',
    crossOrigin: true,
  });
  state.basemap.on("tileload", () => {
    if (!state.basemap) return;
    loaded = true;
    clearTimeout(state.basemapTimer);
    text("basemap-status", "Street basemap · OpenStreetMap");
  });
  state.basemap.once("tileerror", fail);
  state.basemap.addTo(state.map);
  state.basemapTimer = setTimeout(() => { if (!loaded) fail(); }, 12_000);
}

function resetResult() {
  state.result = null;
  clearMap();
  $("run-notice").hidden = true;
  $("proxy-notice").hidden = true;
  $("metric-notes").hidden = true;
  $("metric-notes").replaceChildren();
  text("map-empty-title", "Your next route starts here");
  text("map-empty-copy", "Choose a trip and run reconstruction. The route and GPS trace will appear together.");
  text("results-state", "No run yet");
  for (const id of ["metric-accuracy", "metric-continuity", "metric-inference", "metric-total", "metric-f1", "metric-classifier"]) {
    text(id, "—"); $(id).classList.remove("is-na");
  }
  text("metric-accuracy-note", "Available after a run");
  text("metric-continuity-note", "Available after a run");
  text("metric-inference-note", "Server timing");
  text("metric-total-note", "Inference + scoring");
  for (const id of ["result-model", "result-algorithm", "result-hash", "result-request", "result-dataset", "result-observations", "result-parts", "result-scoring", "result-roundtrip"]) text(id, "—");
  const row = element("div"); row.append(element("dt", "", "Evaluation metrics"), element("dd", "", "Run a trip to view"));
  $("additional-metrics").replaceChildren(row);
  text("details-hint", "No result yet");
}
function renderResult(result, roundTripMs) {
  const geometry = drawMap(result);
  metricValue("metric-accuracy", result.metrics.accuracy, "ratio");
  text("metric-f1", formatPercent(result.metrics.macro_f1));
  text("metric-classifier", formatPercent(result.metrics.top1_accuracy));
  metricValue("metric-continuity", result.metrics.route_continuity, "ratio");
  metricValue("metric-inference", result.timings.inferenceMs, "time");
  metricValue("metric-total", result.timings.totalMs, "time");
  text("metric-accuracy-note", result.metrics.accuracy === null ? "Not available from API" : result.proxyReference ? "Observations · proxy-aligned" : "Observation accuracy");
  text("metric-continuity-note", result.metrics.route_continuity === null ? "Not available from API" : "Server-reported continuity");
  text("metric-inference-note", result.timings.inferenceMs === null ? "Not available from API" : "Server-side prediction");
  text("metric-total-note", result.timings.totalMs === null ? "Not available from API" : "Server inference + scoring");
  const metricsFragment = document.createDocumentFragment();
  for (const [key, label, type] of METRIC_DEFINITIONS) {
    if (!(key in result.metrics)) continue;
    const value = result.metrics[key];
    const row = element("div");
    row.append(element("dt", "", label), element("dd", "", type === "ratio" ? formatPercent(value) : value === null ? "N/A" : value.toLocaleString(undefined, { maximumFractionDigits: 4 })));
    metricsFragment.append(row);
  }
  $("additional-metrics").replaceChildren(metricsFragment);
  text("result-model", result.model.name || "Not supplied");
  text("result-algorithm", result.model.algorithm || "Not supplied");
  text("result-hash", result.model.sha256 || "Not supplied");
  text("result-request", result.requestId || "Not supplied");
  text("result-dataset", datasetLabel() + " (" + result.datasetId + ")");
  text("result-observations", formatCount(geometry.gpsPoints.length));
  text("result-parts", formatCount(geometry.parts));
  text("result-scoring", timingLabel(result.timings.scoringMs));
  text("result-roundtrip", timingLabel(roundTripMs));
  text("details-hint", "Scoring: " + (result.metricsStatus === "unspecified" ? "status not supplied" : result.metricsStatus));
  $("proxy-notice").hidden = !result.proxyReference;
  if (result.metricNotes.length) {
    $("metric-notes").replaceChildren(...result.metricNotes.map((note) => element("p", "", note)));
    $("metric-notes").hidden = false;
  }
  const warnings = [];
  if (geometry.mapUnavailable) warnings.push("The local map library is unavailable. Serve the complete frontend folder, including vendor assets, to view the returned geometry.");
  if (!geometry.routePoints.length) warnings.push("The API returned no predicted route geometry. GPS and server results are shown where available.");
  if (!geometry.gpsPoints.length) warnings.push("The API returned no original GPS geometry. The predicted route and server results are shown where available.");
  if (["partial", "unavailable", "error"].includes(result.metricsStatus)) warnings.push("Server evaluation is " + result.metricsStatus + ". N/A means the API did not supply a valid score; see the evaluation notes in the details below.");
  if (warnings.length) runNotice("Result returned with a note", warnings.join(" "));
  text("results-state", "Live result · " + datasetLabel() + " · " + result.tripId);
  text("map-state-badge", geometry.routePoints.length ? "Result received" : "Check geometry");
  $("map-state-badge").dataset.state = geometry.routePoints.length ? "complete" : "error";
  workflow(2);
  announce("Result received for " + result.tripId + ". " + geometry.gpsPoints.length + " GPS observations. HMM observation accuracy " + formatPercent(result.metrics.accuracy) + ", HMM macro F1 " + formatPercent(result.metrics.macro_f1) + ", route continuity " + formatPercent(result.metrics.route_continuity) + "." + (result.proxyReference ? " Observation evaluation is proxy-aligned, not official time-aligned accuracy." : ""));
}

async function matchTrip() {
  const trip = selectedTrip();
  if (!trip || !state.ready || state.loading || state.connecting) return;
  resetResult();
  state.loading = true;
  const controller = new AbortController();
  const request = { version: ++state.matchVersion, apiBase: state.apiBase, datasetId: state.datasetId, tripId: trip.id };
  state.matchController = controller;
  const start = performance.now();
  $("map-loading").hidden = false;
  $("map-empty").hidden = true;
  $("results-section").setAttribute("aria-busy", "true");
  text("map-state-badge", "Processing");
  $("map-state-badge").dataset.state = "loading";
  text("results-state", "Waiting for the API…");
  text("run-elapsed", "0.0 s");
  state.elapsedTimer = setInterval(() => text("run-elapsed", ((performance.now() - start) / 1000).toFixed(1) + " s"), 200);
  updateControls();
  workflow(1);
  announce("Reconstructing " + trip.id + " on the model server. Waiting for inference and scoring.");
  try {
    const payload = await requestJson(request.apiBase, datasetMatchPath(request.datasetId), { method: "POST", body: { trip_id: trip.requestId }, signal: controller.signal, timeoutMs: MATCH_TIMEOUT_MS });
    if (!isCurrentMatch(request, state)) return;
    const roundTripMs = performance.now() - start;
    const result = normalizeMatch(payload, trip.id, request.datasetId);
    state.result = result;
    renderResult(result, roundTripMs);
  } catch (error) {
    if (!isCurrentMatch(request, state)) return;
    state.result = null;
    const known = error instanceof ApiError ? error : new ApiError("The returned result could not be displayed. Check the API contract and browser console, then retry.");
    const cancelled = known.kind === "cancelled";
    runNotice(cancelled ? "Stopped waiting" : known.kind === "timeout" ? "The request timed out" : "Couldn’t reconstruct this trip", known.message, cancelled ? "warning" : "error");
    text("map-state-badge", cancelled ? "Stopped waiting" : "Run unsuccessful");
    $("map-state-badge").dataset.state = "error";
    text("results-state", "No completed result");
    $("map-empty").hidden = !state.map;
    text("map-empty-title", cancelled ? "Ready when you are" : "No route to display yet");
    text("map-empty-copy", cancelled ? "The API may still be working. Check the server before running again." : "Resolve the message below, then run the selected trip again.");
    if (known.status === 503) {
      state.ready = false;
      setConnection("API not ready", "warning");
      connectionNotice("The model service is not ready", "The API returned HTTP 503. Check model loading and server configuration, then retry the connection.");
    }
    announce((cancelled ? "Stopped waiting. " : "Reconstruction failed. ") + known.message);
  } finally {
    if (!isCurrentMatch(request, state)) return;
    clearInterval(state.elapsedTimer);
    state.elapsedTimer = null;
    state.matchController = null;
    state.loading = false;
    $("map-loading").hidden = true;
    $("results-section").setAttribute("aria-busy", "false");
    updateControls();
  }
}

function showSettings() {
  if (state.loading) return;
  $("api-base-input").value = state.apiBase;
  $("api-base-input").removeAttribute("aria-invalid");
  $("settings-error").hidden = true;
  $("settings-dialog").showModal();
  $("api-base-input").focus();
  $("api-base-input").select();
}
function closeSettings() { $("settings-dialog").close(); }
function saveSettings(event) {
  event.preventDefault();
  let base;
  try { base = normalizeApiBase($("api-base-input").value); } catch (error) {
    $("api-base-input").setAttribute("aria-invalid", "true");
    text("settings-error", error.message);
    $("settings-error").hidden = false;
    $("api-base-input").focus();
    return;
  }
  let saved = true;
  try {
    if (base === defaultApiBase()) localStorage.removeItem(API_STORAGE_KEY);
    else localStorage.setItem(API_STORAGE_KEY, base);
  } catch { saved = false; }
  const changed = base !== state.apiBase;
  state.apiBase = base;
  if (changed) { state.selectedId = null; state.datasets = []; state.datasetId = "development"; }
  updateApiLabel();
  closeSettings();
  toast(saved ? "API address updated. Reconnecting…" : "API address updated for this tab. Browser storage is unavailable.");
  void connect();
}
function init() {
  let savedTheme = null;
  try { savedTheme = localStorage.getItem(THEME_STORAGE_KEY); } catch {}
  applyTheme(savedTheme);
  $("theme-toggle").addEventListener("click", () => {
    const next = normalizeTheme(document.documentElement.dataset.theme) === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem(THEME_STORAGE_KEY, next); } catch { toast("Theme changed for this tab. Browser storage is unavailable."); }
  });
  for (const id of DATASET_IDS) {
    $("tab-" + id).addEventListener("click", () => switchDataset(id));
    $("tab-" + id).addEventListener("keydown", datasetTabKeydown);
  }
  let invalidSavedAddress = false;
  try {
    const saved = localStorage.getItem(API_STORAGE_KEY);
    if (saved) state.apiBase = normalizeApiBase(saved);
  } catch { invalidSavedAddress = true; }
  updateApiLabel();
  initMap();
  $("open-settings").addEventListener("click", showSettings);
  $("close-settings").addEventListener("click", closeSettings);
  $("cancel-settings").addEventListener("click", closeSettings);
  $("settings-form").addEventListener("submit", saveSettings);
  $("settings-dialog").addEventListener("keydown", (event) => {
    if (event.key !== "Tab") return;
    const controls = [...$("settings-dialog").querySelectorAll('button:not(:disabled), input:not(:disabled), [tabindex="0"]')].filter((node) => !node.hidden && node.getClientRects().length);
    const first = controls[0];
    const last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
  });
  $("use-default-api").addEventListener("click", () => { $("api-base-input").value = defaultApiBase(); $("api-base-input").focus(); });
  $("refresh-trips").addEventListener("click", () => void connect());
  $("retry-connection").addEventListener("click", () => void connect());
  $("match-button").addEventListener("click", () => void matchTrip());
  $("cancel-match").addEventListener("click", () => state.matchController?.abort());
  $("fit-map").addEventListener("click", fitMap);
  $("show-route").addEventListener("change", () => toggleLayer("route"));
  $("show-gps").addEventListener("change", () => toggleLayer("gps"));
  $("show-basemap").addEventListener("change", toggleBasemap);
  window.addEventListener("offline", () => { if (!state.loading) toast("The browser reports no network. An already rendered route stays available; new runs need the model API."); });
  window.addEventListener("pagehide", () => {
    state.connectController?.abort(); state.matchController?.abort();
    clearInterval(state.elapsedTimer); clearTimeout(state.toastTimer); clearTimeout(state.basemapTimer);
  });
  if (invalidSavedAddress) toast("The saved API address could not be used. Using this website’s default connection.");
  void connect();
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, { once: true });
else init();
