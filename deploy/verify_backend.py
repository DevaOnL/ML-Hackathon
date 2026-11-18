"""Run actual ten-trip FastAPI checks in-process; never bind a socket.

Requires frozen trained model/configuration and httpx (requirements-check.txt).
Saves aggregate verification evidence only, never served route outputs or labels.
Development partition only; use the authorized public UI for final evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from backend.app import create_app
from backend.config import Settings

FORBIDDEN_KEYS = {"reference_route", "reference_geometry", "reference_edges", "reference_labels", "true_edges", "true_labels", "ground_truth"}


def check_public(value):
    if isinstance(value, dict):
        if FORBIDDEN_KEYS.intersection(value):
            raise AssertionError("A private reference field entered public JSON.")
        for item in value.values():
            check_public(item)
    elif isinstance(value, list):
        for item in value:
            check_public(item)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(settings):
    return {
        "model_sha256": sha256(settings.model_path),
        "metadata_sha256": sha256(settings.metadata_path),
        "match_config_file_sha256": sha256(settings.match_config_path),
    }


def geometry_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="deploy/verification/local_all10.json")
    parser.add_argument("--max-request-seconds", type=float, default=95.0)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT / "deploy"):
        parser.error("--output must remain within deploy/")
    if args.max_request_seconds <= 0:
        parser.error("--max-request-seconds must be positive")
    settings = Settings.from_env()
    if settings.dataset_role != "development":
        raise RuntimeError("This development verification script refuses evaluator-role datasets.")
    before = fingerprint(settings)  # Fail before inference if final artifacts are absent.
    evidence = {"check": "actual_trained_fastapi_all10", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "transport": "in_process_TestClient_no_network_listener", "dataset_role": settings.dataset_role,
                "artifacts_before": before, "all_observations_required": True,
                "max_request_seconds_budget": args.max_request_seconds, "trips": [], "errors": []}
    started = time.perf_counter()
    try:
        with TestClient(create_app(settings)) as client:
            health = client.get("/health")
            assert health.status_code == 200
            evidence["health"] = health.json()
            assert evidence["health"]["status"] == "ready" and evidence["health"]["model_loaded"]
            listing = client.get("/trips")
            assert listing.status_code == 200
            trips = listing.json()["trips"]
            assert len(trips) == 10 and listing.json()["count"] == 10
            assert all(set(trip) == {"trip_id", "observation_count", "duration_s"} for trip in trips)
            ids = [trip["trip_id"] for trip in trips]
            assert len(set(ids)) == 10
            seen_request_ids = set()
            first_route_hash = None
            for trip in trips + [trips[0]]:
                repeat = len(evidence["trips"]) == 10
                wall_started = time.perf_counter()
                response = client.post("/match", json={"trip_id": trip["trip_id"]})
                elapsed = time.perf_counter() - wall_started
                record = {"trip_id": trip["trip_id"], "repeat_first": repeat,
                          "http_status": response.status_code, "http_wall_seconds": elapsed,
                          "response_bytes": len(response.content)}
                if response.status_code != 200:
                    record["error_code"] = response.json().get("detail", {}).get("code")
                    evidence["trips"].append(record)
                    raise AssertionError("An actual registered trip failed live API inference.")
                body = response.json()
                check_public(body)
                assert response.headers["cache-control"] == "no-store"
                assert response.headers["x-request-id"] == body["request_id"]
                assert body["request_id"] not in seen_request_ids
                seen_request_ids.add(body["request_id"])
                assert body["trip_id"] == trip["trip_id"] and body["dataset_role"] == "development"
                assert body["model"]["sha256"] == before["model_sha256"]
                assert body["match_config_hash"] == evidence["health"]["match_config_hash"]
                assert body["proxy_reference"] is True and body["metrics_status"] == "ok"
                diagnostics = body["diagnostics"]
                assert diagnostics["uses_reference_route"] is False
                assert diagnostics["observation_count"] == trip["observation_count"]
                assert diagnostics["matched_observation_count"] == trip["observation_count"]
                assert diagnostics["learned_emission_observations"] == trip["observation_count"]
                assert len(body["observation_assignments"]) == trip["observation_count"]
                assert diagnostics["route_arc_count"] > 0 and diagnostics["break_count"] == 0
                assert body["metrics"]["route_continuity"] == 1.0
                for name, value in body["metrics"].items():
                    assert isinstance(value, (int, float)) and math.isfinite(value), (name, value)
                    assert value >= 0 and (name == "log_loss" or value <= 1), (name, value)
                assert body["metrics"]["scored_observation_fraction"] == 1.0
                for key in ("predicted_route", "raw_gps"):
                    assert body[key]["type"] == "FeatureCollection" and body[key]["features"]
                route_hash = geometry_hash(body["predicted_route"])
                if first_route_hash is None:
                    first_route_hash = route_hash
                if repeat:
                    assert route_hash == first_route_hash, "Repeated actual inference was not deterministic."
                record.update({"request_id": body["request_id"], "model": body["model"],
                               "match_config_hash": body["match_config_hash"], "timings": body["timings"],
                               "metrics_status": body["metrics_status"], "metrics": body["metrics"],
                               "diagnostics": diagnostics, "predicted_geometry_sha256": route_hash,
                               "within_gateway_budget": elapsed < args.max_request_seconds})
                evidence["trips"].append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
                assert elapsed < args.max_request_seconds, "A trip exceeded the deployment gateway request budget."
            # These invalid and nonpublic requests must not trigger fresh inference.
            assert client.post("/match", json={"trip_id": "../../etc/passwd"}).status_code == 422
            assert client.post("/match", json={"trip_id": ids[0]}, headers={"Origin": "https://attacker.invalid"}).status_code == 403
            for path in ("/docs", "/openapi.json", "/artifacts/model.joblib", "/README.md", "/analysis/", "/Student%20folder/"):
                assert client.get(path).status_code == 404
            evidence["private_resource_checks"] = "passed"
            evidence["repeat_request_id_and_live_emission_checks"] = "passed"
            evidence["health_after"] = client.get("/health").json()
        evidence["artifacts_after"] = fingerprint(settings)
        assert evidence["artifacts_after"] == before, "Artifacts changed during the verification run."
        evidence["passed"] = True
    except Exception as exc:
        evidence["passed"] = False
        evidence["errors"].append({"type": type(exc).__name__, "message": str(exc)[:300]})
        raise
    finally:
        evidence["elapsed_seconds"] = time.perf_counter() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"event": "verification_saved", "path": str(output.relative_to(ROOT)),
                          "passed": evidence.get("passed"), "elapsed_seconds": evidence["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
