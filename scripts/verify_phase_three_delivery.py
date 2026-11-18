#!/usr/bin/env python3
"""Reconcile frozen decoder, CLI routes/ZIP, and recorded genuine HTTP evidence.

This command never fits a model, runs inference, opens a server, or makes HTTP
requests. References are opened ONLY to recompute offline route scores after
prediction. The public geometry hash is reconstructed from predicted arcs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.geometry import predicted_geojson
from mapmatch.data import discover_trips, load_graph, load_route, load_track, validate_route
from mapmatch.metrics import score_routes
from scripts.live_evidence import integer_array_hash


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_hash(graph, route):
    value = predicted_geojson(graph, route)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def public_row_checks(row, expected, model_hash, config_hash):
    """Compare one recorded live result to already checked local prediction."""
    return {
        "http_200": row.get("http_status") == 200,
        "scoring_ok": row.get("metrics_status") == "ok",
        "model_hash": row.get("model", {}).get("sha256") == model_hash,
        "config_hash": row.get("match_config_hash") == config_hash,
        "request_id_present": bool(row.get("request_id")),
        "matched_states": row.get("matched_edge_array_sha256") == expected["matched_edge_array_sha256"],
        "predicted_geometry": row.get("predicted_geometry_sha256") == expected["predicted_geometry_sha256"],
        "observation_count": row.get("observation_assignment_count") == expected["observations"],
        "learned_every_observation": row.get("diagnostics", {}).get("learned_emission_observations") == expected["observations"],
        "no_reference_in_prediction": row.get("diagnostics", {}).get("uses_reference_route") is False,
        "length_f1": abs(row.get("metrics", {}).get("route_length_f1", -1) - expected["metrics"]["length_f1"]) < 1e-10,
        "edge_f1": abs(row.get("metrics", {}).get("route_edge_f1", -1) - expected["metrics"]["edge_f1"]) < 1e-10,
    }


def verify(args):
    decoder = args.decoder.resolve()
    outputs = args.routes.resolve()
    frozen = read_json(decoder / "freeze.json")
    model = ROOT / frozen["model_path"]
    config_file = model.parent / "match_config.json"
    states = read_json(decoder / "student_route_checksums.json")
    config = read_json(config_file)
    run = read_json(outputs / "run_config.json")
    compact_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "offline reconciliation of existing genuine measurements; no model inference or network calls",
        "model_sha256": sha256(model), "config_file_sha256": sha256(config_file),
        "api_compact_config_sha256": compact_hash,
        "checks": {}, "trips": {}, "defects": [],
    }
    checks = report["checks"]
    checks["model_hash_matches_all_records"] = report["model_sha256"] == frozen["model_sha256"] == states["model_sha256"] == run["model_sha256"]
    checks["config_hash_matches_all_records"] = report["config_file_sha256"] == frozen["config_file_sha256"] == states["config_file_sha256"]
    checks["config_matches_cli"] = config == run["config"] == frozen["config"]
    checks["learned_not_baseline"] = run["baseline"] is False and run["uses_reference_routes"] is False
    checks["no_cli_failures"] = read_json(outputs / "failures.json") == []
    prefixes = {p.name: p for p in discover_trips(args.data)}
    with (outputs / "index.csv").open() as stream:
        index = list(csv.DictReader(stream))
    checks["exact_ten_distinct_trips"] = len(index) == len(prefixes) == len(states["trips"]) == 10 and {r["trip_id"] for r in index} == set(prefixes) == set(states["trips"])
    expected_names = {tid + ".route" for tid in prefixes}
    with ZipFile(outputs / "submission.zip") as archive:
        checks["zip_flat_ten_routes"] = len(archive.namelist()) == 10 and set(archive.namelist()) == expected_names
        checks["zip_no_corrupt_member"] = archive.testzip() is None
        for row in index:
            tid = row["trip_id"]
            prefix = prefixes[tid]
            route_file = outputs / "routes" / (tid + ".route")
            graph = load_graph(prefix)
            track = load_track(prefix.with_suffix(".track"))
            route = load_route(route_file)
            with (outputs / "observations" / (tid + ".csv")).open() as stream:
                observations = list(csv.DictReader(stream))
            edges = [int(o["predicted_arc_id"]) for o in observations]
            reference = states["trips"][tid]["classifier_plus_HMM"]
            metric = score_routes(graph, route, load_route(prefix.with_suffix(".route")))
            trip_checks = {
                "cli_status": row["status"] == reference["status"] == "ok",
                "route_bytes": sha256(route_file) == row["sha256"] == reference["route_file_sha256"],
                "archive_bytes": archive.read(tid + ".route") == route_file.read_bytes(),
                "arc_count": len(route) == int(row["arc_count"]) == reference["predicted_arc_count"],
                "valid_connected_directed_arcs": validate_route(graph, route)["valid"] and bool(reference["connected"]),
                "decoder_matched_states": integer_array_hash(edges) == reference["matched_edge_array_sha256"],
                "original_gps_order": [int(o["observation_index"]) for o in observations] == list(range(len(track))),
                "original_timestamps": np.array_equal([float(o["timestamp_s"]) for o in observations], track[:, 2]),
            }
            report["trips"][tid] = {
                "checks": trip_checks, "observations": len(track), "predicted_arc_count": len(route),
                "route_sha256": sha256(route_file), "matched_edge_array_sha256": integer_array_hash(edges),
                "predicted_geometry_sha256": geometry_hash(graph, route), "metrics": metric,
            }
    report["trip_macro"] = {k: float(np.mean([t["metrics"][k] for t in report["trips"].values()]))
                            for k in ("edge_f1", "length_f1", "edit_similarity", "route_continuity")}
    report["exact_route_count"] = sum(t["metrics"]["exact_match"] for t in report["trips"].values())
    if args.live:
        live = read_json(args.live / "live_api.json")
        browser = read_json(args.live / "browser_evidence.json")
        checks["genuine_public_origin"] = live["mocked"] is False and live["base_url"] == browser["base_url"] == "https://ml.d3va.dev" and browser["mocked"] is False
        checks["all_ten_real_posts"] = live["all_ten_requested"] is True and live["successful_count"] == live["attempted_count"] == len(live["matches"]) == 10 and {r["trip_id"] for r in live["matches"]} == set(prefixes)
        checks["unique_http_request_ids"] = len({r["request_id"] for r in live["matches"]}) == 10
        checks["browser_clean"] = not browser["browser_errors"] and not browser["console_errors"] and not browser["external_requests"]
        all_rows = [("api", r) for r in live["matches"]] + [("browser", browser["response"])]
        for source, row in all_rows:
            tid = row["trip_id"]
            report["trips"][tid][source + "_checks"] = public_row_checks(row, report["trips"][tid], report["model_sha256"], compact_hash)
        screenshots = browser["screenshots"]
        checks["browser_screenshot_hashes"] = bool(screenshots) and all(sha256(ROOT / s["path"]) == s["sha256"] for s in screenshots)
        terminal = read_json(args.live / "api_terminal_evidence.json")
        known_ids = {r["request_id"] for _, r in all_rows}
        checks["terminal_real_recorded_request"] = terminal["mocked"] is False and terminal["valid"] is True and terminal["ocr_privacy_guard_passed"] is True and terminal["request_id"] in known_ids
        checks["terminal_screenshot_hash"] = sha256(ROOT / terminal["image"]) == terminal["image_sha256"]
        completed = [e for e in terminal["events"] if e["event"] == "match_completed"]
        checks["terminal_frozen_model_event"] = len(completed) == 1 and completed[0]["model_hash"] == report["model_sha256"] and completed[0]["match_config_hash"] == compact_hash
    report["defects"] += [name for name, passed in checks.items() if not passed]
    for tid, trip in report["trips"].items():
        for group in ("checks", "api_checks", "browser_checks"):
            report["defects"] += [tid + ":" + group + ":" + name for name, passed in trip.get(group, {}).items() if not passed]
    report["passed"] = not report["defects"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", type=Path, default=ROOT / "reports/phase_three/decoder")
    parser.add_argument("--routes", type=Path, default=ROOT / "outputs/phase_three_student_test")
    parser.add_argument("--data", type=Path, default=ROOT / "Student folder/student_test")
    parser.add_argument("--live", type=Path, help="Also reconcile recorded public API, browser and terminal evidence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    for source in (args.decoder, args.routes, args.data, *([args.live] if args.live else [])):
        if target.is_relative_to(source.resolve()):
            parser.error("verification output must be outside input evidence/data trees")
    if target.exists():
        parser.error("refuse to overwrite existing verification; choose a new output")
    report = verify(args)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "checks", "trip_macro", "exact_route_count", "defects")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
