#!/usr/bin/env python3
"""Record actual public20, Chromium two-section and real journal terminal evidence.

This never starts a server, replaces predictions, installs infrastructure, changes
DNS/units/tunnels, intercepts browser requests or reads private runtime settings.
Each stage is separate, so the twenty-trip loop is never repeated for screenshots.
All original phase-three outputs remain untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.verify_final_api import (
    FINAL, MODEL_SHA256, CONFIG_SHA256, COMPACT_CONFIG_SHA256, DATASET_IDS,
    REQUIRED_ROW_CHECKS, load_json, sha256, require, require_checks,
    load_offline_expected, validate_api_acceptance, verify_manifest, source_manifest,
    response_checks, summarize, health_checks, utcnow,
)
from scripts.live_evidence import dump

BASE = "https://ml.d3va.dev"
LIVE = ROOT / FINAL / "live"
ASSETS = {"/": "frontend/index.html", "/styles.css": "frontend/styles.css",
          "/app.mjs": "frontend/app.mjs", "/api.mjs": "frontend/api.mjs"}
WIDTHS = (320, 390, 768, 1440)


def no_existing(path):
    require(not path.exists() and not path.is_symlink(), "refuse to overwrite genuine evidence: " + path.name)


def png_dimensions(path):
    with Path(path).open("rb") as stream:
        header = stream.read(24)
    require(header[:8] == b"\x89PNG\r\n\x1a\n" and len(header) == 24, "screenshot is not a complete PNG header")
    width, height = struct.unpack(">II", header[16:24])
    require(width > 0 and height > 0, "screenshot has empty dimensions")
    return width, height


def picture_record(path, **details):
    width, height = png_dimensions(path)
    return {"path": str(Path(path).relative_to(ROOT)), "sha256": sha256(path),
            "width": width, "height": height, "bytes": Path(path).stat().st_size, **details}


def metric_equality(actual, expected):
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False
    return all((value is None and expected[key] is None) or
               (isinstance(value, (int, float)) and isinstance(expected[key], (int, float))
                and abs(value - expected[key]) <= 1e-12) for key, value in actual.items())


def gates():
    expected, offline = load_offline_expected()
    integration = load_json(ROOT / FINAL / "api_integration_acceptance.json")
    validate_api_acceptance(integration, offline, expected)
    verify_manifest(integration["source_sha256_after"])
    activation = load_json(ROOT / FINAL / "deployment_activation.json")
    require(activation.get("validation_passed") is True and activation.get("applied") is True and activation.get("lead_go_confirmed") is True, "no authorized successful deployment activation exists")
    require(activation.get("model_sha256") == MODEL_SHA256 and activation.get("match_config_hash") == COMPACT_CONFIG_SHA256 and activation.get("config_file_sha256") == CONFIG_SHA256, "activation frozen identity mismatch")
    require(activation.get("installed_before") == activation.get("installed_after"), "installed units/launcher/runtime bytes were not preserved")
    # Serving source must remain exact; evidence-collector revisions are separately
    # SHA recorded at each stage and cannot alter the server's predictions.
    serving = {k: v for k, v in activation.get("source_sha256", {}).items() if k.startswith(("backend/", "frontend/", "mapmatch/", "infrastructure/"))}
    verify_manifest(serving)
    require(activation.get("unrelated_environment_preserved") is True, "activation did not preserve unrelated environment")
    return expected, offline, integration, activation


def verify_row(row, offline, integration):
    dataset_id, tid = row["dataset_id"], row["trip_id"]
    require(dataset_id in DATASET_IDS and tid in offline[dataset_id], "unregistered evidence dataset/trip")
    checks = response_checks(row, offline[dataset_id][tid], dataset_id)
    checks["all_metrics_equal_actual_asgi"] = metric_equality(row.get("metrics"), integration["datasets"][dataset_id]["trips"][tid]["metrics"])
    require_checks(checks, dataset_id + ":" + tid + ": recorded response", REQUIRED_ROW_CHECKS)
    return checks


def collect_api():
    import httpx
    expected, offline, integration, activation = gates()
    output = LIVE / "live_api.json"
    no_existing(output)
    before = source_manifest(include_frontend=True)
    evidence = {"collected_at_utc": utcnow(), "base_url": BASE, "mocked": False,
                "source": "real TLS-verified HTTP to the already-running authorized public origin",
                "all_twenty_requested": True, "attempted_count": 0, "successful_count": 0,
                "model_sha256": MODEL_SHA256, "config_file_sha256": CONFIG_SHA256,
                "match_config_hash": COMPACT_CONFIG_SHA256,
                "matches": [], "datasets": {}, "passed": False,
                "source_sha256_before": before, "collector_sha256": sha256(Path(__file__))}
    with httpx.Client(base_url=BASE, timeout=240, follow_redirects=False, trust_env=False) as client:
        health_response, catalogue_response = client.get("/health"), client.get("/datasets")
        require(health_response.status_code == catalogue_response.status_code == 200, "public health/catalogue HTTP failure")
        health, catalogue = health_response.json(), catalogue_response.json()
        evidence["health"], evidence["catalogue"] = health, catalogue
        evidence["health_checks"] = health_checks(health, catalogue)
        require_checks(evidence["health_checks"], "public ready health/catalogue")
        evidence["static_assets"] = []
        for url, relative in ASSETS.items():
            response = client.get(url, headers={"Cache-Control": "no-cache"})
            public_hash = hashlib.sha256(response.content).hexdigest()
            record = {"url": BASE + url, "source": relative, "http_status": response.status_code,
                      "sha256": public_hash, "source_sha256": sha256(ROOT / relative)}
            record["matches_current_source"] = response.status_code == 200 and record["sha256"] == record["source_sha256"]
            require(record["matches_current_source"], "existing public static bytes are stale")
            evidence["static_assets"].append(record)
        for dataset_id in DATASET_IDS:
            response = client.get("/datasets/" + dataset_id + "/trips")
            require(response.status_code == 200, "public dataset trip catalogue failed")
            catalogue = response.json()
            checks = {"exact_ten_registered_ids": len(catalogue["trips"]) == 10 and {r["trip_id"] for r in catalogue["trips"]} == set(offline[dataset_id]),
                      "dataset_identity": catalogue.get("dataset_id") == catalogue.get("dataset_role") == dataset_id}
            require_checks(checks, "public " + dataset_id + " catalogue")
            evidence["datasets"][dataset_id] = {"trip_count": 10, "catalogue": catalogue, "checks": checks}
            for tid in sorted(offline[dataset_id]):
                evidence["attempted_count"] += 1
                started = time.perf_counter()
                response = client.post("/datasets/" + dataset_id + "/match", json={"trip_id": tid})
                require(response.status_code == 200, "public genuine inference HTTP failure")
                row = summarize(response.json(), dataset_id, tid, response.status_code, (time.perf_counter()-started)*1000, offline[dataset_id][tid])
                row["checks"] = verify_row(row, offline, integration)
                row["checks"]["request_id_header"] = response.headers.get("x-request-id") == row["request_id"]
                row["checks"]["no_store_header"] = "no-store" in response.headers.get("cache-control", "")
                row["request_body_keys"] = ["trip_id"]
                require_checks(row["checks"], "public genuine response")
                evidence["matches"].append(row)
                evidence["successful_count"] += 1
                dump(output, evidence)
                print(json.dumps({"event": "actual_public_final_trip_passed", "dataset_id": dataset_id, "trip_id": tid, "request_id": row["request_id"], "route_length_f1": row["metrics"]["route_length_f1"]}), flush=True)
    rows = evidence["matches"]
    require(evidence["attempted_count"] == evidence["successful_count"] == len(rows) == len({(r["dataset_id"], r["trip_id"]) for r in rows}) == len({r["request_id"] for r in rows}) == 20, "public evidence is not exact twenty fresh successful requests")
    evidence["source_sha256_after"] = source_manifest(include_frontend=True)
    require(before == evidence["source_sha256_after"], "source changed during public20 collection")
    evidence.update(passed=True, completed_at_utc=utcnow())
    dump(output, evidence)
    return evidence


def wait_catalogue(page, dataset_id):
    page.wait_for_function("id => document.getElementById('tab-'+id).getAttribute('aria-selected') === 'true' && document.getElementById('workspace-panel').getAttribute('aria-labelledby') === 'tab-'+id && document.getElementById('trip-list').getAttribute('aria-busy') === 'false' && document.querySelectorAll('#trip-list input[name=trip-choice]').length === 10", arg=dataset_id, timeout=30000)
    require(page.locator("#connection-label").inner_text() == "API ready", "browser dataset not ready")


def animation_frame(page):
    page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")


def map_pixels(page):
    animation_frame(page)
    canvases = page.locator("#route-map canvas").evaluate_all("""elements => elements.map(c => {
      const pixels=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
      let drawn=0,route=0,gps=0;
      for(let i=0;i<pixels.length;i+=4){const r=pixels[i],g=pixels[i+1],b=pixels[i+2],a=pixels[i+3];
        if(a>100)drawn++; if(a>100&&b>r*1.5&&b>g*1.3)route++;
        if(a>100&&r>g*1.3&&g>b*1.2)gps++;}
      return {width:c.width,height:c.height,drawn_pixels:drawn,route_color_pixels:route,gps_color_pixels:gps};
    })""")
    counts = {"canvases": len(canvases), "leaflet_panes": page.locator("#route-map .leaflet-pane").count(),
              "leaflet_controls": page.locator("#route-map .leaflet-control").count(),
              "canvas_pixels": canvases}
    require(counts["canvases"] > 0 and counts["leaflet_panes"] > 0 and counts["leaflet_controls"] > 0 and sum(c["route_color_pixels"] for c in canvases) > 10 and sum(c["gps_color_pixels"] for c in canvases) > 10, "actual Leaflet map lacks predicted route/GPS pixels or controls")
    return counts


def layout_record(page, dataset_id, theme, width):
    page.set_viewport_size({"width": width, "height": 1000 if width >= 768 else 844})
    animation_frame(page)
    measured = page.evaluate("""() => ({viewport_width:innerWidth, document_width:document.documentElement.scrollWidth,
       horizontal_overflow:document.documentElement.scrollWidth>innerWidth,
       body_width:document.body.scrollWidth})""")
    measured.update(dataset_id=dataset_id, theme=theme)
    require(not measured["horizontal_overflow"], "horizontal overflow at " + str(width) + " in " + dataset_id + "/" + theme)
    return measured


def screenshot(page, output, filename, dataset_id, theme, element=None):
    target = output / "screenshots" / filename
    no_existing(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if element:
        page.locator(element).screenshot(path=str(target))
    else:
        page.screenshot(path=str(target), full_page=True)
    return picture_record(target, dataset_id=dataset_id, theme=theme)


def collect_browser():
    from playwright.sync_api import sync_playwright
    expected, offline, integration, activation = gates()
    output = LIVE / "browser_evidence.json"
    no_existing(output)
    require(load_json(LIVE / "live_api.json").get("passed") is True, "complete public20 evidence must precede browser collection")
    before = source_manifest(include_frontend=True)
    evidence = {"collected_at_utc": utcnow(), "base_url": BASE, "mocked": False, "passed": False,
                "source": "actual Chromium page and POST responses; no interception/mocks or replacement server",
                "browser_errors": [], "console_errors": [], "external_requests": [], "responses": [],
                "screenshots": [], "layout_checks": [], "datasets": {}, "checks": {},
                "source_sha256_before": before, "collector_sha256": sha256(Path(__file__)),
                "visual_review": "DOM, canvas-pixel and screenshot provenance checks only; no manual visual inspection claimed by collector"}
    posted = []
    observed_assets = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1080}, device_scale_factor=1, service_workers="block")
        page = context.new_page()
        evidence["browser_version"] = browser.version
        page.on("pageerror", lambda error: evidence["browser_errors"].append(str(error)))
        page.on("console", lambda message: evidence["console_errors"].append(message.text) if message.type == "error" else None)
        def request_seen(request):
            parsed = urlsplit(request.url)
            if parsed.netloc and parsed.netloc != urlsplit(BASE).netloc:
                evidence["external_requests"].append(request.url)
            if request.method == "POST":
                posted.append({"url": request.url, "body": request.post_data_json})
        page.on("request", request_seen)
        def response_seen(response):
            parsed = urlsplit(response.url)
            if parsed.netloc == urlsplit(BASE).netloc and parsed.path in ASSETS:
                observed_assets[parsed.path] = {"http_status": response.status, "sha256": hashlib.sha256(response.body()).hexdigest()}
        page.on("response", response_seen)
        initial = page.goto(BASE, wait_until="networkidle")
        require(initial is not None and initial.status == 200 and urlsplit(page.url).netloc == urlsplit(BASE).netloc, "existing public page failed to load")
        wait_catalogue(page, "development")
        page.wait_for_function("() => !document.getElementById('tab-evaluator').disabled")
        require(page.locator("#theme-toggle").get_attribute("aria-pressed") == "false", "fresh browser did not default to dark")
        evidence["checks"]["dark_default"] = True
        evidence["checks"]["both_actual_ten_counts"] = all(page.locator("#count-" + dataset).inner_text() == "10 / 10" for dataset in DATASET_IDS)
        require(evidence["checks"]["both_actual_ten_counts"], "UI did not show two actual ten-trip sections")
        # Genuine keyboard interaction; no JS invocation of application handlers.
        keyboard = []
        page.locator("#tab-development").focus()
        for key, dataset in (("ArrowRight", "evaluator"), ("ArrowRight", "development"), ("End", "evaluator"), ("Home", "development"), ("ArrowLeft", "evaluator"), ("ArrowLeft", "development")):
            page.keyboard.press(key)
            wait_catalogue(page, dataset)
            focused = page.locator("#tab-" + dataset).evaluate("el => el === document.activeElement")
            require(focused, "keyboard dataset tab focus was not retained")
            keyboard.append({"key": key, "selected_dataset": dataset, "focus_retained": focused})
        evidence["keyboard_checks"] = keyboard
        for dataset_id in DATASET_IDS:
            if dataset_id == "evaluator":
                page.locator("#tab-evaluator").click()
                wait_catalogue(page, dataset_id)
            available = [element.get_attribute("value") for element in page.locator('#trip-list input[name="trip-choice"]').all()]
            require(len(available) == 10 and set(available) == set(offline[dataset_id]), "browser catalogue differs from actual offline ten")
            tid = "00000009" if dataset_id == "development" and "00000009" in available else min(available, key=lambda key: abs(offline[dataset_id][key]["observations"]-1500))
            radio = page.locator('#trip-list input[name="trip-choice"][value="' + tid + '"]')
            # The actual native radio is visually hidden; clicking its label is
            # the real user interaction and avoids forced/synthetic checking.
            radio.locator("..").click()
            require(radio.is_checked(), "real trip label click did not select native radio")
            page.wait_for_function("() => !document.getElementById('match-button').disabled")
            started = time.perf_counter()
            with page.expect_response(lambda response: response.url == BASE + "/datasets/" + dataset_id + "/match" and response.request.method == "POST", timeout=240000) as info:
                page.locator("#match-button").click()
            response = info.value
            require(response.status == 200, "genuine browser POST inference failed")
            payload = response.json()
            row = summarize(payload, dataset_id, tid, response.status, (time.perf_counter()-started)*1000, offline[dataset_id][tid])
            row["checks"] = verify_row(row, offline, integration)
            row["checks"]["exact_request_body"] = response.request.post_data_json == {"trip_id": tid}
            require_checks(row["checks"], "browser genuine response")
            page.wait_for_function("id => document.getElementById('result-request').textContent === id && !document.getElementById('match-button').disabled", arg=row["request_id"], timeout=30000)
            page.locator("#result-details").evaluate("el => el.open = true")
            require(page.locator("#result-hash").inner_text() == MODEL_SHA256, "UI displayed model differs from actual response")
            require(dataset_id in page.locator("#result-dataset").inner_text().lower() or (dataset_id == "development" and "student" in page.locator("#result-dataset").inner_text().lower()), "UI dataset identity missing")
            page.locator("#route-map").scroll_into_view_if_needed()
            pixels = map_pixels(page)
            evidence["responses"].append(row)
            if dataset_id == "development":
                evidence["response"] = row
            evidence["datasets"][dataset_id] = {"trip_count": 10, "trip_ids": available, "selected_trip_id": tid,
                                                  "request_id": row["request_id"], "displayed_request_id": page.locator("#result-request").inner_text(),
                                                  "displayed_model_hash": page.locator("#result-hash").inner_text(),
                                                  "accuracy_display": page.locator("#metric-accuracy").inner_text(),
                                                  "map": pixels}
            for theme in ("dark", "light"):
                desired = "false" if theme == "dark" else "true"
                if page.locator("#theme-toggle").get_attribute("aria-pressed") != desired:
                    count_before = len(posted)
                    page.locator("#theme-toggle").click()
                    animation_frame(page)
                    require(len(posted) == count_before and page.locator("#result-request").inner_text() == row["request_id"], "theme switch performed inference or lost current result")
                require(page.locator("#theme-toggle").get_attribute("aria-pressed") == desired, "theme toggle state inconsistent")
                for width in WIDTHS:
                    evidence["layout_checks"].append(layout_record(page, dataset_id, theme, width))
                    if dataset_id == "evaluator" and theme == "dark" and width == 390:
                        evidence["screenshots"].append(screenshot(page, LIVE, "ui_mobile.png", dataset_id, theme))
                    if dataset_id == "evaluator" and theme == "dark" and width == 320:
                        evidence["screenshots"].append(screenshot(page, LIVE, "ui_mobile_320.png", dataset_id, theme))
                page.set_viewport_size({"width": 1440, "height": 1080})
                animation_frame(page)
                if dataset_id == "development" and theme == "dark":
                    evidence["screenshots"].append(screenshot(page, LIVE, "ui_desktop.png", dataset_id, theme))
                    evidence["screenshots"].append(screenshot(page, LIVE, "ui_route_map.png", dataset_id, theme, "#route-map"))
                if dataset_id == "development" and theme == "light":
                    evidence["screenshots"].append(screenshot(page, LIVE, "ui_light.png", dataset_id, theme))
                if dataset_id == "evaluator" and theme == "dark":
                    evidence["screenshots"].append(screenshot(page, LIVE, "ui_evaluator.png", dataset_id, theme))
            require(page.locator("#result-request").inner_text() == row["request_id"], "responsive/theme changes altered result identity")
        evidence["checks"]["theme_changes_no_post"] = len(posted) == 2
        evidence["checks"]["two_exact_dataset_posts"] = len(posted) == 2 and all(posted[index] == {"url": BASE + "/datasets/" + row["dataset_id"] + "/match", "body": {"trip_id": row["trip_id"]}} for index, row in enumerate(evidence["responses"]))
        evidence["checks"]["fresh_two_request_ids"] = len({r["request_id"] for r in evidence["responses"]}) == 2
        page.locator("#open-settings").click()
        page.wait_for_function("() => document.getElementById('settings-dialog').open")
        page.keyboard.press("Escape")
        require(not page.locator("#settings-dialog").evaluate("el => el.open"), "settings modal failed Escape close")
        evidence["checks"]["modal_escape_and_focus"] = page.locator("#open-settings").evaluate("el => el === document.activeElement")
        # A real page refresh verifies saved light theme and current static bytes.
        page.reload(wait_until="networkidle")
        wait_catalogue(page, "development")
        evidence["checks"]["theme_persists_after_reload"] = page.locator("#theme-toggle").get_attribute("aria-pressed") == "true"
        evidence["checks"]["reload_no_post"] = len(posted) == 2
        evidence["static_assets"] = [{"url": BASE + url, "http_status": observed_assets.get(url, {}).get("http_status"),
                                      "sha256": observed_assets.get(url, {}).get("sha256"), "source_sha256": sha256(ROOT / relative),
                                      "matches_current_source": observed_assets.get(url, {}).get("http_status") == 200 and observed_assets.get(url, {}).get("sha256") == sha256(ROOT / relative)} for url, relative in ASSETS.items()]
        evidence["checks"]["refreshed_current_static_bytes"] = all(row["matches_current_source"] for row in evidence["static_assets"])
        evidence["checks"]["clean_browser"] = not evidence["browser_errors"] and not evidence["console_errors"] and not evidence["external_requests"]
        evidence["post_requests"] = posted
        require_checks(evidence["checks"], "genuine browser checks")
        browser.close()
    evidence["source_sha256_after"] = source_manifest(include_frontend=True)
    require(before == evidence["source_sha256_after"], "source changed during browser collection")
    evidence.update(passed=True, completed_at_utc=utcnow())
    dump(output, evidence)
    return evidence


def terminal_checks(terminal, browser):
    responses = {row["request_id"]: row for row in browser.get("responses", [])}
    request_id = terminal.get("request_id")
    known = responses.get(request_id, {})
    events = terminal.get("events", [])
    started = [e for e in events if e.get("event") == "match_started"]
    completed = [e for e in events if e.get("event") == "match_completed"]
    image = ROOT / terminal.get("image", "missing")
    return {"genuine_unmocked": terminal.get("mocked") is False and terminal.get("valid") is True,
            "ocr_privacy_guard": terminal.get("ocr_privacy_guard_passed") is True,
            "browser_request_identity": bool(known), "one_started_and_completed": len(started) == len(completed) == 1,
            "original_frozen_events": bool(events) and all(e.get("request_id") == request_id and e.get("model_hash") == MODEL_SHA256 and e.get("match_config_hash") == COMPACT_CONFIG_SHA256 and e.get("trip_id") == known.get("trip_id") and e.get("dataset_role") == known.get("dataset_id") for e in started + completed),
            "image_sha256": image.is_file() and sha256(image) == terminal.get("image_sha256"),
            "dedicated_api_journal": terminal.get("journal_unit") == "path-snap-api.service"}


def collect_terminal():
    gates()
    from scripts.capture_api_terminal import capture, load_events
    browser = load_json(LIVE / "browser_evidence.json")
    require(browser.get("passed") is True and len(browser.get("responses", [])) == 2, "terminal requires completed genuine browser evidence")
    no_existing(LIVE / "api_terminal_evidence.json")
    request_id = browser["response"]["request_id"]
    since = datetime.fromisoformat(browser["collected_at_utc"]).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    events = load_events(request_id, since)
    require(all(e.get("model_hash") == MODEL_SHA256 and e.get("match_config_hash") == COMPACT_CONFIG_SHA256 for e in events if e.get("event") in {"match_started", "match_completed"}), "journal does not identify frozen inference")
    capture(SimpleNamespace(request_id=request_id, since=since, output=LIVE))
    terminal = load_json(LIVE / "api_terminal_evidence.json")
    require_checks(terminal_checks(terminal, browser), "genuine journal terminal")
    return terminal


def validate_browser(browser, offline, integration):
    require(browser.get("passed") is True and browser.get("mocked") is False and browser.get("base_url") == BASE, "genuine public browser evidence missing")
    require(not browser.get("browser_errors") and not browser.get("console_errors") and not browser.get("external_requests"), "browser errors or external runtime asset requests present")
    required = {"dark_default", "both_actual_ten_counts", "theme_changes_no_post", "two_exact_dataset_posts", "fresh_two_request_ids", "modal_escape_and_focus", "theme_persists_after_reload", "reload_no_post", "refreshed_current_static_bytes", "clean_browser"}
    require_checks(browser.get("checks"), "browser acceptance", required)
    rows = browser.get("responses", [])
    require(len(rows) == 2 and {r.get("dataset_id") for r in rows} == set(DATASET_IDS), "browser did not execute one actual POST in each dataset")
    require(len({r.get("request_id") for r in rows}) == 2 and browser.get("response") in rows, "browser response request identity mismatch")
    for row in rows:
        verify_row(row, offline, integration)
        require_checks(row.get("checks"), "browser saved per-trip checks", REQUIRED_ROW_CHECKS | {"exact_request_body"})
    layout = browser.get("layout_checks", [])
    require(len(layout) == 16 and {(r.get("dataset_id"), r.get("theme"), r.get("viewport_width")) for r in layout} == {(d, theme, width) for d in DATASET_IDS for theme in ("dark", "light") for width in WIDTHS}, "all dataset/theme/width layout combinations not checked")
    require(all(r.get("horizontal_overflow") is False and r.get("document_width", 0) <= r["viewport_width"] for r in layout), "responsive overflow check failed")
    required_images = {"ui_desktop.png", "ui_route_map.png", "ui_mobile.png", "ui_mobile_320.png", "ui_light.png"}
    pictures = browser.get("screenshots", [])
    require(required_images.issubset({Path(p["path"]).name for p in pictures}), "required genuine UI screenshots missing")
    for picture in pictures:
        path = (ROOT / picture["path"]).resolve()
        require(path.is_relative_to(LIVE.resolve()) and sha256(path) == picture["sha256"] and png_dimensions(path) == (picture["width"], picture["height"]) and picture.get("bytes", 0) > 2000, "browser screenshot identity or dimensions invalid")
    require(set(browser.get("datasets", {})) == set(DATASET_IDS), "browser dataset evidence incomplete")
    for dataset, details in browser["datasets"].items():
        require(details.get("trip_count") == 10 and set(details.get("trip_ids", [])) == set(offline[dataset]), "browser trip list count differs from actual ten")
        pixels = details.get("map", {})
        require(pixels.get("canvases", 0) > 0 and pixels.get("leaflet_panes", 0) > 0 and pixels.get("leaflet_controls", 0) > 0 and sum(r["route_color_pixels"] for r in pixels.get("canvas_pixels", [])) > 10 and sum(r["gps_color_pixels"] for r in pixels.get("canvas_pixels", [])) > 10, "browser actual Leaflet pixels missing")
    return True


def reconcile():
    expected, offline, integration, activation = gates()
    target = ROOT / FINAL / "delivery_verification.json"
    no_existing(target)
    live = load_json(LIVE / "live_api.json")
    browser = load_json(LIVE / "browser_evidence.json")
    terminal = load_json(LIVE / "api_terminal_evidence.json")
    require(live.get("passed") is True and live.get("mocked") is False and live.get("base_url") == BASE and live.get("all_twenty_requested") is True, "genuine public20 evidence missing")
    rows = live.get("matches", [])
    pairs = {(dataset, tid) for dataset in DATASET_IDS for tid in offline[dataset]}
    require(live.get("attempted_count") == live.get("successful_count") == len(rows) == 20 and {(r.get("dataset_id"), r.get("trip_id")) for r in rows} == pairs, "public exact twenty inference coverage missing")
    require(len({r.get("request_id") for r in rows}) == 20, "public requests reused IDs")
    require_checks(live.get("health_checks"), "public health checks")
    require(all(s.get("matches_current_source") is True for s in live.get("static_assets", [])) and len(live.get("static_assets", [])) == 4, "public static source evidence incomplete")
    verify_manifest(live["source_sha256_after"])
    require(live["source_sha256_before"] == live["source_sha256_after"], "public collection source changed")
    validate_browser(browser, offline, integration)
    verify_manifest(browser["source_sha256_after"])
    require(browser["source_sha256_before"] == browser["source_sha256_after"], "browser collection source changed")
    terminal_map = terminal_checks(terminal, browser)
    require_checks(terminal_map, "terminal reconciliation")
    from scripts.capture_api_terminal import terminal_frame_flags
    ocr = (LIVE / "api_terminal_ocr.txt").read_text()
    require_checks(terminal_frame_flags(ocr, terminal["request_id"]), "persisted actual terminal OCR")
    require(not any(marker in ocr for marker in ("/home/", "runtime.env", "Bearer ", "tunnel-credentials")), "terminal OCR privacy marker guard failed")
    report = {"checked_at_utc": utcnow(), "passed": False, "base_url": BASE,
              "source": "per-trip reconciliation of actual offline20, in-process20, public20, browser2 and real journal terminal; no inference repeated",
              **expected, "datasets": {}, "checks": {}, "defects": [], "terminal_checks": terminal_map,
              "counts": {"offline": 20, "in_process_asgi": 20, "public_post": 20, "additional_asgi_repeat": 1, "browser_post": 2, "receiving_terminals": 1},
              "evaluator_overlap_audit": integration["evaluator_overlap_audit"]}
    public_by_pair = {(r["dataset_id"], r["trip_id"]): r for r in rows}
    browser_by_pair = {(r["dataset_id"], r["trip_id"]): r for r in browser["responses"]}
    for dataset in DATASET_IDS:
        group = {"trip_count": 10, "trips": {}}
        report["datasets"][dataset] = group
        for tid, reference in offline[dataset].items():
            api_row = integration["datasets"][dataset]["trips"][tid]
            public = public_by_pair[(dataset, tid)]
            require_checks(public.get("checks"), "public saved checks", REQUIRED_ROW_CHECKS | {"all_metrics_equal_actual_asgi", "request_id_header", "no_store_header"})
            trip = {"offline_checks": reference["checks"], "api_checks": api_row["checks"],
                    "actual_cli_checks": api_row["cli_fixture_checks"], "candidate_audit_checks": api_row["candidate_audit"]["checks"],
                    "public_checks": verify_row(public, offline, integration),
                    "observations": reference["observations"], "route_sha256": reference["route_sha256"],
                    "matched_edge_array_sha256": reference["matched_edge_array_sha256"], "predicted_geometry_sha256": reference["predicted_geometry_sha256"],
                    "metrics_status": public["metrics_status"], "offline_route_metrics": reference.get("metrics"),
                    "api_metrics": api_row["metrics"], "public_metrics": public["metrics"],
                    "api_request_id": api_row["request_id"], "public_request_id": public["request_id"]}
            if (dataset, tid) in browser_by_pair:
                trip["browser_checks"] = verify_row(browser_by_pair[(dataset, tid)], offline, integration)
                trip["browser_request_id"] = browser_by_pair[(dataset, tid)]["request_id"]
            group["trips"][tid] = trip
            for key, check_map in trip.items():
                if key.endswith("checks"):
                    require_checks(check_map, dataset + ":" + tid + ":" + key)
        group["observation_count"] = sum(r["observations"] for r in offline[dataset].values())
        group["trip_macro"] = {key: sum(r["metrics"][key] for r in offline[dataset].values()) / 10 for key in ("edge_f1", "length_f1", "edit_similarity", "route_continuity") if all(r.get("metrics_status") == "ok" for r in offline[dataset].values())}
    report["checks"] = {"all_twenty_offline_per_trip_checks": True, "all_twenty_actual_asgi_per_trip_checks": True,
                        "all_twenty_public_per_trip_checks": True, "two_actual_browser_dataset_responses": True,
                        "terminal_matches_recorded_browser_request": True, "all_screenshot_hashes_and_dimensions": True,
                        "all_sixteen_responsive_theme_dataset_layouts": True, "public_static_bytes_equal_reviewed_source": True,
                        "frozen_model_and_full_config_unchanged": True, "no_mocked_inference_or_browser": True,
                        "private_references_not_in_public_responses": True}
    report["evidence_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in (
        ROOT / FINAL / "offline_delivery_verification.json", ROOT / FINAL / "evaluator/acceptance.json",
        ROOT / FINAL / "api_integration_acceptance.json", ROOT / FINAL / "deployment_activation.json",
        LIVE / "live_api.json", LIVE / "browser_evidence.json", LIVE / "api_terminal_evidence.json")}
    report["passed"] = True
    dump(target, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("api", "browser", "terminal", "reconcile", "check"))
    args = parser.parse_args(argv)
    if args.stage == "check":
        expected, offline, integration, activation = gates()
        print(json.dumps({"passed": True, "network_requests": 0, "model_sha256": MODEL_SHA256, "trip_count": 20}))
        return
    result = {"api": collect_api, "browser": collect_browser, "terminal": collect_terminal, "reconcile": reconcile}[args.stage]()
    print(json.dumps({"stage": args.stage, "base_url": BASE, "passed": result.get("passed", result.get("valid")),
                      "attempted_count": result.get("attempted_count"), "successful_count": result.get("successful_count")}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
