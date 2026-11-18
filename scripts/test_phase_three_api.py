#!/usr/bin/env python3
"""Actual frozen Bagging -> loader -> API -> all-ten geometry/state/score parity.

In-process ASGI TestClient only; this is NOT public-network evidence. No model,
scorer, matcher, or reference provider is replaced. No HTTP server is started.
Only structured test logs are suppressed; privacy checks inspect real responses.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from backend.app import create_app
from backend.config import Settings
from scripts.live_evidence import check_health, summarize_response
from scripts.verify_phase_three_delivery import public_row_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase_three/api_integration_acceptance.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refuse to overwrite actual API acceptance")
    offline = json.loads((ROOT/"reports/phase_three/offline_delivery_verification.json").read_text())
    if not offline.get("passed"):
        raise RuntimeError("all-ten CLI/decoder offline parity has not passed")
    settings = Settings.from_env()
    expected_model = (ROOT/"artifacts/phase_three/model.joblib").resolve()
    if settings.model_path != expected_model or not settings.match_config_required:
        raise RuntimeError("test defaults do not select frozen requested-trio model/config")
    rows = []
    metric_names = {"route_edge_f1":"edge_f1", "route_length_f1":"length_f1",
                    "sequence_similarity":"edit_similarity", "route_continuity":"route_continuity"}
    with patch("backend.service.log_event"), patch("backend.boundary.log_event"):
        with TestClient(create_app(settings)) as client:
            health = client.get("/health").json()
            check_health(health)
            catalogue = client.get("/trips").json()
            ids = {row["trip_id"] for row in catalogue["trips"]}
            if ids != set(offline["trips"]):
                raise AssertionError("actual API trip catalogue differs from tested CLI")
            for tid in sorted(ids):
                started = time.perf_counter()
                response = client.post("/match", json={"trip_id":tid})
                payload = response.json()
                row = summarize_response(payload,tid,response.status_code,1000*(time.perf_counter()-started))
                checks = public_row_checks(row,offline["trips"][tid],offline["model_sha256"],offline["api_compact_config_sha256"])
                checks["private_reference_not_in_response"] = "reference_route" not in payload
                for api_name, offline_name in metric_names.items():
                    checks["metric_parity:"+api_name] = abs(payload["metrics"][api_name]-offline["trips"][tid]["metrics"][offline_name]) <= 1e-12
                if not all(checks.values()):
                    raise AssertionError(tid+": "+str([k for k,v in checks.items() if not v]))
                rows.append({"trip_id":tid,"request_id":row["request_id"],"checks":checks,"metrics":payload["metrics"],
                             "predicted_geometry_sha256":row["predicted_geometry_sha256"],
                             "matched_edge_array_sha256":row["matched_edge_array_sha256"],"timings":payload["timings"]})
                print(json.dumps({"event":"actual_api_integration_passed","trip_id":tid,"route_length_f1":payload["metrics"]["route_length_f1"]}),flush=True)
    if len({r["request_id"] for r in rows}) != 10:
        raise AssertionError("inference request IDs were not fresh")
    report = {"checked_at_utc":datetime.now(timezone.utc).isoformat(),"passed":True,"trip_count":10,
              "actual_estimator":"BaggingClassifier", "transport":"in-process ASGI TestClient, no sockets, not public live proof",
              "mocked_model_or_inference":False,"model_sha256":offline["model_sha256"],
              "match_config_hash":offline["api_compact_config_sha256"],"prediction_cache":False,"trips":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("ACTUAL_ALL_TEN_API_INTEGRATION_PASSED",flush=True)


if __name__ == "__main__":
    main()
