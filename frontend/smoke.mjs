#!/usr/bin/env node
/** Frontend-only validation. Never starts a server or injects sample results.
 * Static checks: node frontend/smoke.mjs
 * Live API checks: node frontend/smoke.mjs --api https://ml.d3va.dev
 * One real inference: append --match-first (can be computationally expensive).
 */
import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve, dirname, extname, sep } from "node:path";
import {
  normalizeApiBase, requestJson, normalizeHealth, normalizeTrips,
  normalizeMatch, geometryPositions, MATCH_TIMEOUT_MS,
} from "./api.mjs";

const root = dirname(fileURLToPath(import.meta.url));
const options = process.argv.slice(2);
let apiBase = null;
let matchFirst = false;
for (let index = 0; index < options.length; index++) {
  if (options[index] === "--api") {
    assert(options[index + 1], "--api requires a complete URL");
    apiBase = normalizeApiBase(options[++index], { protocol: "http:", origin: "http://localhost" });
  } else if (options[index] === "--match-first") matchFirst = true;
  else throw new Error("Unknown option: " + options[index]);
}
assert(!matchFirst || apiBase, "--match-first requires --api");
const html = await readFile(resolve(root, "index.html"), "utf8");
const app = await readFile(resolve(root, "app.mjs"), "utf8");
const css = await readFile(resolve(root, "styles.css"), "utf8");
const api = await readFile(resolve(root, "api.mjs"), "utf8");
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
assert.equal(new Set(ids).size, ids.length, "HTML IDs must be unique");
const requiredIds = [...app.matchAll(/\$\("([^"]+)"\)/g)].map((match) => match[1]);
for (const id of requiredIds) assert(ids.includes(id), "Missing application target: " + id);
for (const match of html.matchAll(/\b(?:aria-labelledby|aria-describedby)="([^"]+)"/g)) {
  for (const id of match[1].split(/\s+/)) assert(ids.includes(id), "Broken accessible reference: " + id);
}
for (const match of html.matchAll(/\b(?:src|href)="(\.\/[^"#]+)"/g)) {
  const path = resolve(root, match[1]);
  assert(path.startsWith(root + sep), "Asset path must stay in frontend");
  assert((await stat(path)).isFile(), "Missing local asset: " + match[1]);
}
for (const match of css.matchAll(/url\((?:["'])?(\.\/[^)'"\s]+)(?:["'])?\)/g)) assert((await stat(resolve(root, match[1]))).isFile());
const leaflet = await readFile(resolve(root, "vendor/leaflet/leaflet.js"), "utf8");
assert(leaflet.includes("1.9.4"), "Unexpected Leaflet version");
assert(leaflet.length > 140_000 && leaflet.length < 200_000, "Leaflet JS download may be incomplete");
assert((await readFile(resolve(root, "vendor/leaflet/LICENSE"), "utf8")).includes("BSD 2-Clause"), "Missing Leaflet license");
for (const name of ["layers.png", "layers-2x.png", "marker-icon.png", "marker-icon-2x.png", "marker-shadow.png"]) {
  const image = await readFile(resolve(root, "vendor/leaflet/images", name));
  assert(image.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])), "Invalid vendored PNG: " + name);
}
assert(!app.includes("innerHTML"), "Use text nodes for API-derived content");
assert(!html.includes("fonts.googleapis.com"), "System fonts should keep the UI offline-ready");
assert(!/<script[^>]+src="https?:/i.test(html), "Runtime scripts must be local");
assert(!/<input[^>]+name="trip-choice"/.test(html), "Trips must be populated by the API, not static HTML");
assert(!/localStorage\.setItem\([^,]*(?:result|route|trip)/i.test(app), "Trip outputs must not be persisted");
assert(css.includes("prefers-reduced-motion"), "Missing reduced-motion support");
assert(html.includes('role="radiogroup"') && html.includes('aria-live="polite"'), "Missing accessible trip/status controls");
assert(api.includes('"/health"') === false, "Endpoints should remain caller-controlled within the adapter");
console.log("PASS: static assets, complete offline Leaflet, DOM targets, ARIA references, and source integrity");

if (apiBase) {
  const [healthPayload, tripPayload] = await Promise.all([
    requestJson(apiBase, "/health"), requestJson(apiBase, "/trips"),
  ]);
  const health = normalizeHealth(healthPayload);
  const catalogue = normalizeTrips(tripPayload);
  assert(health.ready, "API health must report ready with its model loaded");
  assert.equal(catalogue.trips.length, 10, "The API must return all 10 trips");
  assert.equal(catalogue.expectedCount, 10, "The API must expect 10 trips");
  console.log("PASS: live API health and complete 10-trip catalogue");
  if (matchFirst) {
    const trip = catalogue.trips[0];
    const payload = await requestJson(apiBase, "/match", { method: "POST", body: { trip_id: trip.requestId }, timeoutMs: MATCH_TIMEOUT_MS });
    const result = normalizeMatch(payload, trip.id);
    assert(geometryPositions(result.predictedRoute).length, "Live predicted route must be drawable");
    assert(geometryPositions(result.rawGps).length, "Live GPS observations must be drawable");
    assert(result.requestId, "Live response must identify its request");
    assert(result.model.sha256, "Live response must identify its model");
    const inspectKeys = (value) => {
      if (!value || typeof value !== "object") return;
      for (const [key, child] of Object.entries(value)) {
        assert(!/(ground_truth|reference_route|reference_edges|true_labels|y_true|route_labels)/i.test(key), "Forbidden reference data key in live response: " + key);
        inspectKeys(child);
      }
    };
    inspectKeys(payload);
    console.log("PASS: one real POST /match response, route and observations, model identity, and no reference-data keys");
    console.log("Request: " + result.requestId + "; scoring status: " + result.metricsStatus + "; proxy-aligned: " + result.proxyReference);
  }
}
