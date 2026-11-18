"""Exercise an existing PathSnap API. This client never starts a server.

Default performs health, metadata and one live match; --all checks all ten.
Hidden evaluator execution requires --allow-evaluator to prevent accidental use
of reserved scores during development. Never use evaluator results for tuning.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

FORBIDDEN_KEYS = {"reference_route", "reference_geometry", "reference_edges", "reference_labels", "true_edges", "true_labels", "ground_truth"}


def _check_public(value):
    if isinstance(value, dict):
        if FORBIDDEN_KEYS.intersection(value):
            raise ValueError("Response unexpectedly contains a private reference field.")
        for item in value.values():
            _check_public(item)
    elif isinstance(value, list):
        for item in value:
            _check_public(item)


def request(base, path, *, payload=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if body is None else {"Content-Type": "application/json"}
    req = Request(base + path, data=body, headers=headers, method="GET" if body is None else "POST")
    try:
        with urlopen(req, timeout=240) as response:
            value = json.load(response)
            request_id = response.headers.get("X-Request-ID")
            if not request_id or response.headers.get("Cache-Control") != "no-store":
                raise ValueError("Missing request ID/no-store public response headers.")
    except HTTPError as exc:
        try:
            code = json.load(exc).get("detail", {}).get("code", "http_error")
        except (ValueError, AttributeError):
            code = "http_error"
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {code}") from None
    _check_public(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--all", action="store_true", help="run all ten dynamically discovered trips")
    parser.add_argument("--trip", help="run one ID returned by /trips")
    parser.add_argument("--allow-evaluator", action="store_true", help="acknowledge authorized final-only evaluator execution")
    args = parser.parse_args()
    base = args.base.rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path:
        parser.error("--base must be a plain http(s) origin without credentials or a path")
    if args.all and args.trip:
        parser.error("choose --all or --trip")
    health = request(base, "/health")
    print(json.dumps({"event": "health", **health}, sort_keys=True))
    if not health.get("model_loaded") or health.get("status") != "ready":
        raise RuntimeError("The API is reachable but the trained inference service is not ready.")
    listing = request(base, "/trips")
    trips = listing.get("trips", [])
    if len(trips) != 10 or listing.get("count") != 10:
        raise ValueError("The deployed API must register exactly ten trips.")
    if any(set(trip) != {"trip_id", "observation_count", "duration_s"} for trip in trips):
        raise ValueError("Trip listing is not metadata-only.")
    ids = [trip["trip_id"] for trip in trips]
    if len(set(ids)) != 10 or any(not isinstance(trip_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", trip_id) is None for trip_id in ids):
        raise ValueError("The trip registry returned unsafe or duplicate IDs.")
    role = listing.get("dataset_role")
    print(json.dumps({"event": "trips", "dataset_role": role, "trip_ids": ids}, sort_keys=True))
    if role == "evaluator" and not args.allow_evaluator:
        raise RuntimeError("Evaluator data is reserved for final evaluation; add --allow-evaluator only for authorized final demonstration, never development tuning.")
    chosen = ids if args.all else [args.trip or ids[0]]
    if any(trip_id not in ids for trip_id in chosen):
        parser.error("--trip must be an ID returned by the current /trips endpoint")
    for trip_id in chosen:
        result = request(base, "/match", payload={"trip_id": trip_id})
        if result.get("trip_id") != trip_id or not result.get("request_id") or result.get("proxy_reference") is not True:
            raise ValueError("Match response lost selection, request ID, or proxy-label disclosure.")
        for key in ("predicted_route", "raw_gps"):
            geometry = result.get(key, {})
            if geometry.get("type") != "FeatureCollection" or not geometry.get("features"):
                raise ValueError("Match response contains no live geometry.")
        diagnostics = result.get("diagnostics", {})
        if diagnostics.get("learned_emission_observations", 0) < 1 or diagnostics.get("uses_reference_route") is not False:
            raise ValueError("Match response does not attest learned, reference-free inference.")
        print(json.dumps({"event": "match", "trip_id": trip_id, "request_id": result["request_id"],
                          "dataset_role": result.get("dataset_role"), "model": result.get("model"),
                          "match_config_hash": result.get("match_config_hash"),
                          "metrics_status": result.get("metrics_status"), "metrics": result.get("metrics"),
                          "timings": result.get("timings"), "diagnostics": diagnostics}, sort_keys=True))
        if result.get("metrics_status") == "error":
            raise RuntimeError("Live prediction succeeded but server scoring failed; inspect the matching request ID in server logs.")
    print(json.dumps({"event": "passed", "live_matches": len(chosen), "dataset_role": role}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, URLError, TimeoutError) as exc:
        print(f"PathSnap smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
