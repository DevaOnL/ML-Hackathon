"""Collect actual network/browser evidence from the existing PathSnap service.

No server, request interception, route mocks, substitute predictions or references
are created here. API and browser stages can run separately without repeating
an expensive all-trip inference loop.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_KEYS = {"reference_route", "reference_edges", "reference_geometry", "ground_truth", "true_labels", "reference_arc_ids"}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_public(value):
    if isinstance(value, dict):
        if PRIVATE_KEYS.intersection(value):
            raise RuntimeError("private reference field exposed in public JSON")
        for inner in value.values():
            inspect_public(inner)
    elif isinstance(value, list):
        for inner in value:
            inspect_public(inner)


def integer_array_hash(values):
    import struct
    return hashlib.sha256(b"".join(struct.pack("<q", int(v)) for v in values)).hexdigest()


def expected_freeze(freeze_path=None):
    freeze_path = Path(freeze_path or ROOT / "reports/phase_three/decoder/freeze.json")
    freeze = json.loads(freeze_path.read_text())
    model = ROOT / freeze["model_path"]
    config = model.parent / "match_config.json"
    if sha256(model) != freeze["model_sha256"] or sha256(config) != freeze["config_file_sha256"]:
        raise RuntimeError("frozen model/config bytes changed")
    # The backend hashes compact JSON; the decoder evidence hashes spaced JSON.
    compact = json.dumps(freeze["config"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    return freeze, hashlib.sha256(compact.encode()).hexdigest()


def check_health(health, freeze_path=None):
    freeze, config_hash = expected_freeze(freeze_path)
    if (health.get("status") != "ready" or not health.get("model_loaded")
            or health.get("model_hash") != freeze["model_sha256"]
            or health.get("match_config_hash") != config_hash
            or health.get("trip_count") != 10 or health.get("prediction_cache") is not False):
        raise RuntimeError("running service does not match the ready frozen learned configuration")
    return freeze, config_hash


def summarize_response(payload, selected, status, elapsed):
    inspect_public(payload)
    row = {"trip_id": selected, "http_status": status, "client_elapsed_ms": elapsed,
           "request_id": payload.get("request_id"), "timings": payload.get("timings"),
           "metrics": payload.get("metrics"), "metrics_status": payload.get("metrics_status"),
           "diagnostics": payload.get("diagnostics"), "model": payload.get("model"),
           "match_config_hash": payload.get("match_config_hash")}
    if status == 200:
        features = payload.get("predicted_route", {}).get("features", [])
        assignments = payload.get("observation_assignments", [])
        # Public route geometry is grouped by connected part, not one feature per arc.
        # Original observation edge IDs are public predictions and preserve state order.
        states = [a["edge_id"] for a in assignments]
        row.update(prediction_feature_count=len(features),
                   raw_feature_count=len(payload.get("raw_gps", {}).get("features", [])),
                   observation_assignment_count=len(assignments),
                   predicted_geometry_sha256=hashlib.sha256(json.dumps(payload["predicted_route"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                   matched_edge_array_sha256=integer_array_hash(states))
        if not row["request_id"] or payload.get("trip_id") != selected:
            raise RuntimeError("response/request identity mismatch")
    return row


def collect_api(base, output, matches=False, trip_id=None, freeze_path=None):
    evidence = {"collected_at_utc": utcnow(), "base_url": base,
                "source": "actual HTTP requests to the running service", "mocked": False,
                "checks": [], "matches": [], "all_ten_requested": bool(matches and not trip_id)}
    with httpx.Client(base_url=base, timeout=httpx.Timeout(240, connect=20)) as client:
        health = client.get("/health")
        catalog = client.get("/trips")
        health.raise_for_status()
        catalog.raise_for_status()
        evidence.update(health=health.json(), health_http_status=health.status_code, catalog=catalog.json())
        ids = [t["trip_id"] for t in catalog.json()["trips"]]
        if len(ids) != 10 or len(set(ids)) != 10:
            raise RuntimeError("expected ten distinct registered trips")
        public = {"/", "/app.mjs", "/styles.css", "/vendor/leaflet/leaflet.js"}
        for path in (*sorted(public), "/.env", "/.git/config", "/artifacts/model.joblib",
                     "/visualizations/", "/docs", "/openapi.json",
                     "/Student%20folder/student_test/00000000/00000000.route"):
            response = client.get(path)
            expected = 200 if path in public else 404
            evidence["checks"].append({"path": path, "status": response.status_code, "expected": expected,
                                       "cache_control": response.headers.get("cache-control"),
                                       "content_type": response.headers.get("content-type")})
            if response.status_code != expected:
                raise RuntimeError("unexpected public/private static response: " + path)
        for body in ({"trip_id": "../../etc/passwd"}, {"trip_id": 0}, {"trip_id": ids[0], "reference_route": [0]}):
            response = client.post("/match", json=body)
            evidence["checks"].append({"invalid_request": body, "status": response.status_code})
            if response.status_code != 422:
                raise RuntimeError("invalid request accepted")
        foreign = client.post("/match", json={"trip_id": ids[0]}, headers={"Origin": "https://foreign-origin.invalid"})
        evidence["checks"].append({"foreign_origin_status": foreign.status_code})
        if foreign.status_code != 403:
            raise RuntimeError("foreign origin accepted")
        if matches:
            frozen, compact_hash = check_health(evidence["health"], freeze_path)
            evidence["freeze"] = {"model_sha256": frozen["model_sha256"],
                                  "config_file_sha256": frozen["config_file_sha256"],
                                  "decoder_spaced_config_sha256": frozen["effective_config_sha256"],
                                  "api_compact_config_sha256": compact_hash}
            if trip_id and trip_id not in ids:
                raise RuntimeError("unknown selected trip")
            for selected in ([trip_id] if trip_id else ids):
                start = time.perf_counter()
                response = client.post("/match", json={"trip_id": selected})
                row = summarize_response(response.json(), selected, response.status_code,
                                         (time.perf_counter() - start) * 1000)
                evidence["matches"].append(row)
                dump(output / "live_api.json", evidence)
                print(json.dumps(row, allow_nan=False), flush=True)
                if response.status_code != 200:
                    raise RuntimeError("live inference failed for " + selected)
                if row["model"]["sha256"] != frozen["model_sha256"] or row["match_config_hash"] != compact_hash:
                    raise RuntimeError("response artifact hashes differ from freeze")
                if row["metrics_status"] != "ok":
                    raise RuntimeError("reference scoring unavailable; no all-ten score claim permitted")
    evidence.update(completed_at_utc=utcnow(), attempted_count=len(evidence["matches"]),
                    successful_count=sum(r["http_status"] == 200 for r in evidence["matches"]))
    dump(output / "live_api.json", evidence)
    return evidence


def collect_browser(base, output, ready=False, trip_id="00000009", freeze_path=None):
    from playwright.sync_api import sync_playwright
    screenshots = output / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    evidence = {"collected_at_utc": utcnow(), "base_url": base, "mocked": False,
                "source": "actual Chromium and network; no interception or route mocks",
                "browser_errors": [], "console_errors": [], "external_requests": [], "checks": []}
    origin = urlsplit(base).netloc
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1512, "height": 1060}, device_scale_factor=1)
        evidence["browser_version"] = browser.version
        page.on("pageerror", lambda e: evidence["browser_errors"].append(str(e)))
        page.on("console", lambda m: evidence["console_errors"].append(m.text) if m.type == "error" else None)
        page.on("request", lambda r: evidence["external_requests"].append(r.url)
                if urlsplit(r.url).netloc and urlsplit(r.url).netloc != origin else None)
        response = page.goto(base, wait_until="networkidle")
        if response is None or response.status != 200:
            raise RuntimeError("website did not return 200")
        page.wait_for_function('() => document.querySelectorAll("#trip-list input[type=radio]").length===10')
        evidence["checks"].append({"trip_options": page.locator("#trip-list input[type=radio]").count(),
                                   "api_host": page.locator("#api-host").inner_text(),
                                   "connection": page.locator("#connection-label").inner_text()})
        if ready:
            health = page.request.get(base.rstrip("/") + "/health").json()
            check_health(health, freeze_path)
            page.wait_for_function('() => document.querySelectorAll("#trip-list input[type=radio]:not(:disabled)").length===10', timeout=30000)
            selected = page.locator('#trip-list input[type=radio][value="' + trip_id + '"]')
            if selected.count() != 1:
                raise RuntimeError("browser trip missing from genuine catalog")
            selected.check()
            page.wait_for_function('() => !document.getElementById("match-button").disabled', timeout=30000)
            started = time.perf_counter()
            with page.expect_response(lambda r: r.url.endswith("/match") and r.request.method == "POST", timeout=240000) as info:
                page.locator("#match-button").click()
            payload = info.value.json()
            if info.value.status != 200:
                raise RuntimeError("browser real inference failed")
            inspect_public(payload)
            page.wait_for_function('() => !document.getElementById("match-button").disabled', timeout=240000)
            page.wait_for_function('() => document.getElementById("metric-accuracy").textContent.trim()!=="—"', timeout=15000)
            page.locator("#result-details").evaluate("el=>el.open=true")
            page.locator("#route-map").scroll_into_view_if_needed()
            evidence["response"] = summarize_response(payload, trip_id, info.value.status,
                                                      (time.perf_counter() - started) * 1000)
            # Leaflet intentionally uses a Canvas renderer; SVG path count is not
            # evidence of a blank map. Read actual canvas alpha/color pixels.
            page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
            canvas = page.locator("#route-map canvas").evaluate_all("""elements => elements.map(c => {
              const pixels=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
              let drawn=0, route=0, gps=0;
              for(let i=0;i<pixels.length;i+=4){
                const r=pixels[i],g=pixels[i+1],b=pixels[i+2],a=pixels[i+3];
                if(a>100) drawn++;
                if(a>100 && g>r*1.4 && g>b*.8) route++;
                if(a>100 && r>g*1.3 && g>b*1.2) gps++;
              }
              return {width:c.width,height:c.height,drawn_pixels:drawn,route_color_pixels:route,gps_color_pixels:gps};
            })""")
            if (not canvas or sum(c["route_color_pixels"] for c in canvas) < 10
                    or sum(c["gps_color_pixels"] for c in canvas) < 10):
                raise RuntimeError("real map canvas missing predicted-route or original-GPS pixels")
            displayed_request = page.locator("#result-request").inner_text()
            if displayed_request != payload["request_id"]:
                raise RuntimeError("UI request ID differs from actual POST response")
            evidence["checks"].append({"real_post_status": info.value.status,
                                       "hmm_accuracy_display": page.locator("#metric-accuracy").inner_text(),
                                       "continuity_display": page.locator("#metric-continuity").inner_text(),
                                       "request_id": displayed_request, "leaflet_canvas": canvas,
                                       "map_accessible_label": page.locator("#route-map").get_attribute("aria-label")})
            page.screenshot(path=str(screenshots / "ui_desktop.png"), full_page=True)
            page.locator("#route-map").screenshot(path=str(screenshots / "ui_route_map.png"))
        else:
            page.screenshot(path=str(screenshots / "ui_not_ready.png"), full_page=True)
        for width in (320, 390, 768, 1440):
            page.set_viewport_size({"width": width, "height": 844})
            # Poll a measured layout condition, not a fixed animation delay.
            page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
            overflow = page.evaluate("() => document.documentElement.scrollWidth>window.innerWidth")
            evidence["checks"].append({"viewport_width": width, "horizontal_overflow": overflow})
            if overflow:
                raise RuntimeError("horizontal overflow at width " + str(width))
            if width == 390 and ready:
                page.screenshot(path=str(screenshots / "ui_mobile.png"), full_page=True)
        page.locator("#open-settings").click()
        page.wait_for_function('() => document.getElementById("settings-dialog").open')
        page.keyboard.press("Escape")
        closed = not page.locator("#settings-dialog").evaluate("el=>el.open")
        evidence["checks"].append({"settings_escape_closes": closed,
                                   "focus_restored_to_settings": page.locator("#open-settings").evaluate("el=>el===document.activeElement")})
        if not closed:
            raise RuntimeError("Escape did not close settings")
        browser.close()
    evidence["screenshots"] = [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size}
                               for p in sorted(screenshots.glob("ui_*.png"))]
    evidence["completed_at_utc"] = utcnow()
    dump(output / "browser_evidence.json", evidence)
    if evidence["browser_errors"] or evidence["console_errors"] or evidence["external_requests"]:
        raise RuntimeError("browser console, script, or external-network error")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase_three/live")
    parser.add_argument("--freeze", type=Path, default=ROOT / "reports/phase_three/decoder/freeze.json")
    parser.add_argument("--matches", action="store_true")
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--browser-only", action="store_true", help="do not repeat the API collection")
    parser.add_argument("--trip", help="single API diagnostic; omit for all ten")
    parser.add_argument("--browser-trip", default="00000009")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    result = None if args.browser_only else collect_api(args.base, args.output, args.matches, args.trip, args.freeze)
    if args.browser or args.browser_only:
        collect_browser(args.base, args.output, args.matches, args.browser_trip, args.freeze)
    print(json.dumps({"base_url": args.base, "matches": len(result["matches"]) if result else "API stage not repeated",
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
