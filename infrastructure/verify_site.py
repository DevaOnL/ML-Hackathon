#!/usr/bin/env python3
"""Verify the actual same-origin PathSnap site without saving raw GPS/model data.

Default target is the authorized public HTTPS hostname; --local checks only8000.
--static-only permits the explicitly authorized pre-training local stage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from launch_service import HOSTNAME, ORIGIN


class VerificationError(Exception):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


class Client:
    def __init__(self, base: str):
        self.base = base
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def request(self, path: str, body: dict | None = None, *, origin: str | None = None) -> tuple[int, dict, bytes, float]:
        headers = {"User-Agent": "PathSnap-deployment-verification/1"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if origin is not None:
            headers["Origin"] = origin
        request = Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                          headers=headers, method="POST" if body is not None else "GET")
        started = time.perf_counter()
        try:
            with self.opener.open(request, timeout=180 if body is not None else 15) as response:
                return response.status, {key.lower(): value for key, value in response.headers.items()}, response.read(32 * 1024 * 1024), (time.perf_counter() - started) * 1000
        except HTTPError as exc:
            return exc.code, {key.lower(): value for key, value in exc.headers.items()}, exc.read(65536), (time.perf_counter() - started) * 1000
        except (URLError, OSError):
            raise VerificationError("Connection/TLS verification failed for the exact target; no insecure fallback was attempted.") from None


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def parse_json(payload: bytes) -> dict:
    try:
        result = json.loads(payload)
    except ValueError:
        raise VerificationError("The endpoint did not return valid JSON; unexpected content was not saved.") from None
    ensure(isinstance(result, dict), "The endpoint returned an unexpected JSON shape.")
    return result


def forbidden_keys(value) -> set[str]:
    forbidden = {"reference_route", "reference_geometry", "ground_truth", "ground_truth_route", "reference_arc_ids",
                 "route_file", "model_path", "data_dir", "reference_dir", "apiToken", "TunnelSecret"}
    found = set()
    if isinstance(value, dict):
        found.update(forbidden.intersection(value))
        for nested in value.values():
            found.update(forbidden_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(forbidden_keys(nested))
    return found


def verify(args: argparse.Namespace) -> dict:
    base = ORIGIN if args.local else "https://" + HOSTNAME
    ensure(not args.static_only or args.local, "Static/notready mode is restricted to the local target.")
    client = Client(base)
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "target": base,
              "stage": "local-static-notready" if args.static_only else "ready", "checks": [], "matches": []}
    for path, expected in [("/", 200), ("/styles.css", 200), ("/app.mjs", 200),
                           ("/api.mjs", 200), ("/vendor/leaflet/leaflet.js", 200),
                           ("/.env", 404), ("/artifacts/model.joblib", 404),
                           ("/Student%20folder/student_test/", 404), ("/docs", 404)]:
        status, headers, payload, latency = client.request(path)
        ensure(status == expected, f"Unexpected HTTP status for {path}: {status}, expected {expected}.")
        if path == "/":
            ensure(b"PathSnap" in payload, "The HTTPS root is not the PathSnap interface.")
            ensure("text/html" in headers.get("content-type", ""), "The root has an unexpected content type.")
        report["checks"].append({"path": path, "status": status, "latency_ms": round(latency, 3)})
    status, _, payload, _ = client.request("/health")
    ensure(status == 200, f"Health endpoint returned HTTP {status}.")
    health = parse_json(payload)
    ensure(health.get("trip_count") == 10, "Health did not discover exactly ten configured trips.")
    if not args.static_only:
        ensure(health.get("status") == "ready" and health.get("model_loaded") is True, "The trained-model service is not ready.")
        ensure(health.get("model_metadata_loaded") is True and health.get("model_hash") and health.get("match_config_hash"), "Model/config provenance is incomplete.")
    report["health"] = {key: health.get(key) for key in ("status", "model_loaded", "model_metadata_loaded", "model_name",
                              "model_hash", "match_config_hash", "trip_count", "dataset_role", "inference_concurrency")}
    report["health"]["error_codes"] = [item.get("code") for item in health.get("errors", [])]
    status, _, payload, _ = client.request("/trips")
    ensure(status == 200, "Trip registry was not available.")
    trips = parse_json(payload)
    ids = [trip.get("trip_id") for trip in trips.get("trips", [])]
    ensure(len(ids) == 10 and len(set(ids)) == 10 and all(isinstance(item, str) for item in ids), "Trip registry is not ten unique IDs.")
    report["trip_ids"] = ids
    # Origin-present hostile POSTs must be denied before inference/model work.
    status, _, _, _ = client.request("/match", {"trip_id": ids[0]}, origin="https://untrusted.invalid")
    ensure(status == 403, f"Cross-origin inference was not rejected: HTTP {status}.")
    report["checks"].append({"path": "/match", "origin": "untrusted", "status": status})
    if args.static_only:
        ensure(health.get("model_loaded") is False and health.get("status") != "ready", "Static-stage check unexpectedly found a ready model; run the ready-stage verification instead.")
        return report
    selected_ids = ids if args.all_trips else [args.trip or ids[0]]
    ensure(all(item in ids for item in selected_ids), "Requested verification trip is not in the live registry.")
    for trip_id in selected_ids:
        status, headers, payload, latency = client.request("/match", {"trip_id": trip_id}, origin=base)
        ensure(status == 200, f"Live match failed for {trip_id}: HTTP {status}.")
        result = parse_json(payload)
        diagnostics = result.get("diagnostics", {})
        ensure(result.get("trip_id") == trip_id, "Match response identifies the wrong trip.")
        ensure(diagnostics.get("uses_reference_route") is False, "Match provenance does not exclude reference input.")
        ensure(diagnostics.get("learned_emission_observations", 0) > 0, "Match did not report trained classifier use.")
        ensure(diagnostics.get("route_arc_count", 0) > 0 and diagnostics.get("break_count") == 0, "Match route is empty or discontinuous.")
        ensure(result.get("predicted_route", {}).get("type") == "FeatureCollection", "Predicted route GeoJSON is unavailable.")
        ensure(result.get("raw_gps", {}).get("type") == "FeatureCollection", "Raw GPS GeoJSON is unavailable.")
        ensure(result.get("match_config_hash") == health.get("match_config_hash"), "Decoder hash changed during verification.")
        ensure(not forbidden_keys(result), "A server-private response field was exposed.")
        report["matches"].append({"trip_id": trip_id, "status": status, "request_id": result.get("request_id"),
                                  "latency_ms": round(latency, 3), "response_sha256": hashlib.sha256(payload).hexdigest(),
                                  "timings": result.get("timings"), "metrics_status": result.get("metrics_status"),
                                  "metrics": result.get("metrics"), "diagnostics": diagnostics})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--all-trips", action="store_true")
    parser.add_argument("--trip")
    parser.add_argument("--output", type=Path, help="Optional sanitized verification evidence JSON; contains no raw route geometry.")
    args = parser.parse_args(argv)
    try:
        report = verify(args)
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            path = args.output.expanduser().absolute()
            root = Path(__file__).resolve().parent
            ensure(path.resolve().is_relative_to(root), "Evidence output must stay within infrastructure/.")
            ensure(not path.exists(), "Refusing to overwrite existing verification evidence; choose a new filename.")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as output:
                output.write(text)
        print(text, end="")
        return 0
    except (VerificationError, OSError, ValueError, TypeError, KeyError) as exc:
        message = str(exc) if isinstance(exc, VerificationError) else "Verification failed while parsing or saving sanitized evidence."
        print("PathSnap site verification failed: " + message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
