"""Genuine browser check against an already-running PathSnap deployment.

No server is started and no responses, metrics, routes, or screenshots are faked.
Requires separately installed Playwright and its Chromium browser.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--browser", default=None)
    parser.add_argument("--output", default=str(Path(__file__).parent / "validation"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = args.url.rstrip("/")
    origin = "{0.scheme}://{0.netloc}".format(urlsplit(base))
    record = {"url": base, "captured_utc": datetime.now(timezone.utc).isoformat(), "checks": [], "screenshots": []}
    with sync_playwright() as playwright:
        executable = args.browser
        if executable is None and not Path(playwright.chromium.executable_path).is_file():
            available = sorted((Path.home() / ".cache/ms-playwright").glob("chromium-*/chrome-linux64/chrome"))
            if available:
                executable = str(available[-1])
        browser = playwright.chromium.launch(headless=True, executable_path=executable)
        context = browser.new_context(viewport={"width": 1440, "height": 1080}, device_scale_factor=1, reduced_motion="reduce")
        page = context.new_page()
        console_errors = []
        page_errors = []
        requests = []
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on("request", lambda request: requests.append(request.url))
        response = page.goto(base, wait_until="networkidle")
        assert response and response.status == 200, "UI page must be served successfully"
        expect(page.locator("#trip-list")).to_have_attribute("aria-busy", "false")
        health = context.request.get(base + "/health").json()
        trips = context.request.get(base + "/trips").json()["trips"]
        record["health_status"] = health.get("status")
        record["model_loaded"] = health.get("model_loaded")
        record["trip_count"] = len(trips)
        radios = page.locator('input[name="trip-choice"]')
        assert radios.count() == len(trips) == 10, "All ten live API trips must be represented"
        assert radios.evaluate_all("nodes => nodes.map(node => node.value)") == [str(trip["trip_id"]) for trip in trips]
        record["checks"].append("All 10 trip choices match actual GET /trips IDs in API order")
        assert page.locator("#map-failure").is_hidden(), "Vendored Leaflet must initialize"
        assert page.locator(".leaflet-container").count() == 1
        assert not any("tile.openstreetmap.org" in url for url in requests), "Default map must not request remote tiles"
        record["checks"].append("Local Leaflet initializes with no external default map-tile requests")
        radios.first.focus()
        page.keyboard.press("Space")
        first_id = radios.first.input_value()
        assert page.locator("#selected-trip").inner_text() == first_id
        page.keyboard.press("ArrowRight")
        second_id = radios.nth(1).input_value()
        assert radios.nth(1).is_checked()
        assert page.locator("#selected-trip").inner_text() == second_id
        assert page.evaluate("document.activeElement.name") == "trip-choice"
        record["checks"].append("Keyboard Space/ArrowRight selects native radio options and updates selected-trip summary")
        if health.get("status") != "ready" or not health.get("model_loaded"):
            assert page.locator("#match-button").is_disabled()
            assert page.locator("#connection-notice").is_visible()
            assert page.locator("#connection-label").inner_text() in {"API not ready", "API unavailable"}
            assert page.locator("#metric-accuracy").inner_text() == "—"
            assert page.locator("#map-empty").is_visible()
            record["checks"].append("Actual missing/unready model yields explicit notice, disabled inference, empty map, and no invented scores")
        settings = page.locator("#open-settings")
        settings.focus()
        page.keyboard.press("Enter")
        assert page.locator("#settings-dialog").is_visible()
        assert page.evaluate("document.activeElement.id") == "api-base-input"
        for _ in range(9):
            page.keyboard.press("Tab")
            assert page.evaluate("document.activeElement.closest('#settings-dialog') !== null"), "Dialog must contain keyboard focus"
        page.keyboard.press("Escape")
        assert not page.locator("#settings-dialog").is_visible()
        assert page.evaluate("document.activeElement.id") == "open-settings"
        record["checks"].append("Settings opens from keyboard, traps Tab focus, Escape closes, and focus returns")
        settings.click()
        page.locator("#api-base-input").fill("file:///etc/passwd")
        page.locator('#settings-form button[type="submit"]').click()
        assert page.locator("#settings-error").is_visible()
        assert page.locator("#api-base-input").get_attribute("aria-invalid") == "true"
        page.locator("#use-default-api").click()
        assert page.locator("#api-base-input").input_value() == origin
        page.locator('#settings-form button[type="submit"]').click()
        expect(page.locator("#trip-list")).to_have_attribute("aria-busy", "false")
        assert page.locator("#settings-dialog").is_hidden()
        assert page.evaluate("localStorage.getItem('pathsnap.apiBaseUrl')") is None
        record["checks"].append("Invalid API scheme rejected; website-default reconnect remains same-origin and clears override")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Desktop must not horizontally overflow"
        page.locator("#selected-trip").scroll_into_view_if_needed()
        page.evaluate("window.scrollTo(0, 0)")
        suffix = "ready" if health.get("status") == "ready" and health.get("model_loaded") else "not-ready"
        desktop = output / ("desktop-" + suffix + ".png")
        page.screenshot(path=str(desktop), full_page=True)
        record["screenshots"].append(desktop.name)
        for width in [390, 320, 768]:
            page.set_viewport_size({"width": width, "height": 844})
            page.evaluate("window.scrollTo(0, 0)")
            page.locator("#page-title").bounding_box()
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Layout must fit width " + str(width)
            assert page.locator("#open-settings").is_visible()
            assert page.locator("#trip-list").is_visible()
            if width == 390:
                mobile = output / ("mobile-390-" + suffix + ".png")
                page.screenshot(path=str(mobile), full_page=True)
                record["screenshots"].append(mobile.name)
            record["checks"].append("No horizontal overflow at viewport width " + str(width))
        record["console_errors"] = console_errors
        record["page_errors"] = page_errors
        record["external_requests"] = sorted(set(url for url in requests if not url.startswith(origin + "/")))
        assert not page_errors, "No uncaught application errors are expected"
        assert not console_errors, "No browser console errors are expected"
        assert not record["external_requests"], "Default UI must have only same-origin requests"
        record["checks"].append("No browser console errors, uncaught application errors, or external default requests")
        browser.close()
    record["model_dependent_checks"] = "Not run: this check does not trigger inference. Lead captures live route runs after the trained artifact is ready."
    (output / "browser-check.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
