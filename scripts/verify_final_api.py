#!/usr/bin/env python3
"""Genuine frozen-model ASGI integration for development10 and evaluator10.

No socket server, model refit, prediction lookup, mock, environment-file read or
network request is used. The optional scorer observer records scalar audit facts
and unconditionally delegates to the actual private post-prediction scorer.
Evaluator inputs are opened only after its separate offline acceptance passes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_SHA256 = "c9cc8e9f36838f656a94d07ed69632979df9981ca94a9daf3b52fbce052e4b34"
CONFIG_SHA256 = "10405aa6b6f370e41d4ddcb01207fd1e8f58900fd0395e541ee6bc78146168c5"
COMPACT_CONFIG_SHA256 = "e326b1afda125897cd784ab335fbcd7c869fcd70714c5203ef5c2c71977b3288"
DATASET_IDS = ("development", "evaluator")
FAMILIES = {"logistic_regression", "knn_distance_weighted", "bagging", "gaussian_nb", "hist_gradient_boosting"}
FINAL = Path("reports/final_five")
MODEL = Path("artifacts/final_five/model.joblib")
METADATA = Path("artifacts/final_five/model_metadata.json")
CONFIG = Path("artifacts/final_five/match_config.json")
OUTPUTS = {"development": Path("outputs/final_student_test"), "evaluator": Path("outputs/final_evaluator_test")}
REQUIRED_ROW_CHECKS = frozenset({
    "http_200", "scoring_ok", "model_hash", "config_hash", "request_id_present",
    "matched_states", "predicted_geometry", "observation_count", "learned_every_observation",
    "no_reference_in_prediction", "length_f1", "edge_f1", "dataset_identity",
    "model_name", "all_observation_counts", "finite_json", "metric_ranges",
    "private_reference_not_in_response", "full_original_observation_order", "raw_gps_count",
    "metric_parity:sequence_similarity", "metric_parity:route_continuity",
})
REQUIRED_AUDIT_CHECKS = frozenset({
    "actual_emission_model", "actual_histgb_estimator", "all_observations_audited",
    "all_candidate_probabilities_finite_positive_normalized", "post_prediction_complete",
    "frozen_full_configuration", "classifier_was_called", "actual_private_scorer_called",
})
REQUIRED_INTEGRATION_CHECKS = frozenset({
    "exact_twenty_unique_dataset_trip_pairs", "unique_request_ids", "no_prediction_cache",
    "shared_actual_model", "shared_frozen_config", "shared_concurrency_gate",
    "separate_registries_and_graph_caches", "frozen_files_unchanged", "frozen_full_config_unchanged",
    "inference_sources_unchanged", "both_catalogues_ready", "repeated_trip_fresh_inference",
})


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_checks(checks, label, required=()):
    require(isinstance(checks, dict) and bool(checks), label + ": check map is empty or invalid")
    require(set(required).issubset(checks), label + ": required checks are missing")
    require(all(isinstance(k, str) and k and v is True for k, v in checks.items()), label + ": a check did not explicitly pass")


def validate_frozen_bundle(root=ROOT):
    """Read only already-public model/review/decoder records, never raw inputs."""
    root = Path(root)
    acceptance = load_json(root / FINAL / "all_five_model_acceptance.json")
    model_freeze = load_json(root / "artifacts/final_five/model_freeze.json")
    clean = load_json(root / "artifacts/final_five/clean_load_report.json")
    review = load_json(root / "analysis/final_model_review/final_review_report.json")
    raw_review = load_json(root / "analysis/final_model_review/verification_raw_output.json")
    toy_review = load_json(root / "analysis/final_model_review/toy_mapmatcher_raw_output.json")
    decoder = load_json(root / FINAL / "decoder/freeze.json")
    config = load_json(root / CONFIG)
    metadata = load_json(root / METADATA)
    require(acceptance.get("passed") is True and acceptance.get("exactly_five_families") is True, "all-five model acceptance has not passed")
    require(set(acceptance.get("families", [])) == FAMILIES, "all-five family coverage mismatch")
    require(acceptance.get("all80_train_trips") is True and acceptance.get("all10_student_trips") is True, "all-five train/test coverage is incomplete")
    require(acceptance.get("student_used_for_selection") is False, "student selection leakage guard failed")
    models = acceptance.get("all_models", {})
    require(set(models) == FAMILIES | {"hist_gradient_boosting_tuned"}, "actual fitted estimator coverage mismatch")
    for name, record in models.items():
        require(record.get("passed") is True and record.get("training_trip_count") == 80 and record.get("student_trip_count") == 10, name + ": real train/test acceptance missing")
        require(record.get("strict_fixture_rows", 0) >= 257 and math.isfinite(record.get("max_probability_delta", math.inf)) and record["max_probability_delta"] <= 1e-12, name + ": strict probability replay failed")
        artifact = (root / record["artifact"]).resolve()
        require(artifact.is_relative_to(root.resolve()) and sha256(artifact) == record["sha256"], name + ": fitted artifact bytes changed")
    require(review.get("passed") is True and review.get("critical_defects") == [], "independent model review has not passed cleanly")
    for report, count, label in ((raw_review, 111, "independent actual checks"), (toy_review, 10, "independent real matcher checks")):
        rows = report.get("checks", [])
        require(len(rows) == count and all(r.get("passed") is True for r in rows), label + " incomplete or failed")
    require(clean.get("passed") is True and clean.get("atol") == 1e-12 and clean.get("rtol") == 0 and clean.get("max_probability_delta", math.inf) <= 1e-12, "strict clean-load replay has not passed")
    require(clean.get("versions") == clean.get("parent_versions"), "clean-load dependency versions disagree")
    hashes = [sha256(root / MODEL), acceptance.get("selected_model_sha256"), model_freeze.get("model_sha256"), clean.get("model_sha256"), decoder.get("model_sha256"), review.get("selected_model", {}).get("frozen_model_sha256")]
    require(all(value == MODEL_SHA256 for value in hashes), "pinned final model SHA mismatch")
    require(sha256(root / CONFIG) == decoder.get("config_file_sha256") == CONFIG_SHA256, "pinned final configuration SHA mismatch")
    require(config == decoder.get("config") and json_hash(config) == COMPACT_CONFIG_SHA256, "full frozen configuration mismatch")
    require(sha256(root / METADATA) == model_freeze.get("metadata_sha256"), "frozen model metadata bytes changed")
    require(metadata.get("model_name") == "hist_gradient_boosting" and metadata.get("run_name") == "hist_gradient_boosting_tuned", "wrong frozen estimator sidecar")
    require(model_freeze.get("student_used_for_selection") is False and model_freeze.get("evaluator_used_for_selection") is False, "frozen model selection guards failed")
    return {"model_sha256": MODEL_SHA256, "config_file_sha256": CONFIG_SHA256,
            "match_config_hash": COMPACT_CONFIG_SHA256, "model_name": "hist_gradient_boosting",
            "frozen_config": config, "metadata_sha256": sha256(root / METADATA)}


def load_offline_expected(root=ROOT):
    """Evaluator acceptance is checked BEFORE constructing Settings or a registry."""
    root = Path(root)
    expected = validate_frozen_bundle(root)
    development = load_json(root / FINAL / "offline_delivery_verification.json")
    require(development.get("passed") is True, "development offline acceptance has not passed")
    require_checks(development.get("checks"), "development offline")
    require(development.get("model_sha256") == MODEL_SHA256 and development.get("config_file_sha256") == CONFIG_SHA256 and development.get("api_compact_config_sha256") == COMPACT_CONFIG_SHA256, "development offline frozen identity mismatch")
    evaluator = load_json(root / FINAL / "evaluator/acceptance.json")
    require(evaluator.get("passed") is True and evaluator.get("trip_count") == 10, "evaluator offline all-ten acceptance has not passed")
    require_checks(evaluator.get("checks"), "evaluator offline")
    provenance = evaluator.get("freeze_provenance", {})
    require(provenance.get("model_sha256") == MODEL_SHA256 and provenance.get("config_file_sha256") == CONFIG_SHA256 and provenance.get("api_compact_config_sha256") == COMPACT_CONFIG_SHA256, "evaluator offline frozen identity mismatch")
    datasets = {"development": development.get("trips", {}), "evaluator": evaluator.get("expected_trips", {})}
    for dataset_id, trips in datasets.items():
        require(isinstance(trips, dict) and len(trips) == 10, dataset_id + ": need exactly ten offline trip records")
        for tid, row in trips.items():
            require_checks(row.get("checks"), dataset_id + ":" + tid + ":offline")
            require(isinstance(row.get("observations"), int) and row["observations"] > 0, "invalid offline observation count")
            for name in ("route_sha256", "matched_edge_array_sha256", "predicted_geometry_sha256"):
                require(isinstance(row.get(name), str) and len(row[name]) == 64, dataset_id + ":" + tid + ": missing " + name)
            status = row.get("metrics_status", "ok" if dataset_id == "development" else None)
            require(status in {"ok", "unavailable"}, dataset_id + ":" + tid + ": offline scoring failed or incomplete")
            if status == "ok":
                require(all(isinstance(row.get("metrics", {}).get(k), (int, float)) and math.isfinite(row["metrics"][k]) for k in ("edge_f1", "length_f1", "edit_similarity", "route_continuity")), "offline route scores missing")
            row["metrics_status"] = status
    return expected, datasets


def source_manifest(root=ROOT, include_frontend=False):
    root = Path(root)
    paths = sorted(set(root.glob("backend/*.py")) | set(root.glob("mapmatch/*.py")))
    paths += [root / "scripts/verify_final_api.py", root / "scripts/live_evidence.py", root / "scripts/verify_phase_three_delivery.py"]
    if include_frontend:
        paths += sorted(p for p in (root / "frontend").rglob("*") if p.is_file())
        paths += [root / "infrastructure/launch_service.py", root / "scripts/activate_final_five.py", root / "scripts/final_live_evidence.py"]
    return {str(path.relative_to(root)): sha256(path) for path in paths}


def verify_manifest(manifest, root=ROOT):
    require(isinstance(manifest, dict) and bool(manifest), "source hash manifest missing")
    for relative, digest in manifest.items():
        path = (Path(root) / relative).resolve()
        require(path.is_relative_to(Path(root).resolve()) and sha256(path) == digest, "reviewed source bytes changed")


def finite_json(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_json(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_json(v) for v in value)
    return False


def assert_public_payload(payload):
    from scripts.live_evidence import inspect_public
    inspect_public(payload)
    forbidden = {"reference_path", "reference_dir", "reference_content", "reference_nodes", "reference_labels", "proxy_labels", "proxy_arrays", "truth", "y_true", "true_edges", "labels", "candidate_probabilities", "reference_route_path"}
    def visit(value):
        if isinstance(value, dict):
            require(not forbidden.intersection(value), "private reference/proxy array field exposed")
            for inner in value.values():
                visit(inner)
        elif isinstance(value, list):
            for inner in value:
                visit(inner)
        elif isinstance(value, str):
            require(not any(marker in value for marker in (str(ROOT), "/home/", "runtime.env", "tunnel-credentials", "Bearer ")), "private path or credential marker exposed")
    visit(payload)
    allowed = {"request_id", "trip_id", "dataset_id", "dataset_role", "predicted_route", "raw_gps", "observation_assignments", "metrics", "metrics_status", "metric_notes", "proxy_reference", "diagnostics", "model", "match_config_hash", "timings"}
    require(set(payload).issubset(allowed), "unexpected top-level public response field")
    require(finite_json(payload), "nonfinite or invalid JSON value")


def catalogue_checks(catalogue):
    rows = catalogue.get("datasets", [])
    allowed = {"id", "label", "role", "available", "trip_count", "expected_trip_count", "status"}
    return {"exact_two_datasets": len(rows) == 2 and {r.get("id") for r in rows} == set(DATASET_IDS),
            "exact_public_catalogue_fields": all(set(row) == allowed for row in rows),
            "both_ready_ten": len(rows) == 2 and all(row.get("available") is True and row.get("trip_count") == row.get("expected_trip_count") == 10 and row.get("status") == "ready" and row.get("role") == row.get("id") for row in rows)}


def health_checks(health, catalogue):
    result = {"ready": health.get("status") == "ready" and health.get("model_loaded") is True,
              "selected_histgb": health.get("model_hash") == MODEL_SHA256 and health.get("model_name") == "hist_gradient_boosting",
              "frozen_config": health.get("match_config_hash") == COMPACT_CONFIG_SHA256,
              "prediction_cache_disabled": health.get("prediction_cache") is False,
              "development_ten": health.get("trip_count") == health.get("expected_trip_count") == 10 and health.get("dataset_role") == "development",
              "health_catalogue_equal": health.get("datasets") == catalogue.get("datasets")}
    result.update(catalogue_checks(catalogue))
    return result


def metric_ranges(metrics):
    if not isinstance(metrics, dict):
        return False
    return all(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 and (k == "log_loss" or v <= 1 + 1e-12)) for k, v in metrics.items())


def response_checks(row, offline, dataset_id, payload=None):
    from scripts.verify_phase_three_delivery import public_row_checks
    checks = public_row_checks(row, offline, MODEL_SHA256, COMPACT_CONFIG_SHA256) if offline["metrics_status"] == "ok" else {
        "http_200": row.get("http_status") == 200, "scoring_ok": row.get("metrics_status") == "unavailable",
        "model_hash": row.get("model", {}).get("sha256") == MODEL_SHA256,
        "config_hash": row.get("match_config_hash") == COMPACT_CONFIG_SHA256,
        "request_id_present": bool(row.get("request_id")),
        "matched_states": row.get("matched_edge_array_sha256") == offline["matched_edge_array_sha256"],
        "predicted_geometry": row.get("predicted_geometry_sha256") == offline["predicted_geometry_sha256"],
        "observation_count": row.get("observation_assignment_count") == offline["observations"],
        "learned_every_observation": row.get("diagnostics", {}).get("learned_emission_observations") == offline["observations"],
        "no_reference_in_prediction": row.get("diagnostics", {}).get("uses_reference_route") is False,
        "length_f1": row.get("metrics", {}).get("route_length_f1") is None,
        "edge_f1": row.get("metrics", {}).get("route_edge_f1") is None,
    }
    n = offline["observations"]
    checks.update(dataset_identity=row.get("dataset_id") == dataset_id,
                  model_name=row.get("model", {}).get("name") == "hist_gradient_boosting",
                  all_observation_counts=all(row.get("diagnostics", {}).get(k) == n for k in ("observation_count", "matched_observation_count", "learned_emission_observations")),
                  finite_json=finite_json(row), metric_ranges=metric_ranges(row.get("metrics")))
    for public, private in (("sequence_similarity", "edit_similarity"), ("route_continuity", "route_continuity")):
        actual = row.get("metrics", {}).get(public)
        target = offline.get("metrics", {}).get(private)
        checks["metric_parity:" + public] = (actual is None and target is None) or (isinstance(actual, (int, float)) and isinstance(target, (int, float)) and abs(actual-target) <= 1e-12)
    if payload is not None:
        try:
            assert_public_payload(payload)
            private_ok = True
        except ValueError:
            private_ok = False
        assignments = payload.get("observation_assignments", [])
        gps = payload.get("raw_gps", {}).get("features", [])
        checks.update(private_reference_not_in_response=private_ok,
                      full_original_observation_order=[r.get("observation_index") for r in assignments] == list(range(n)),
                      raw_gps_count=len(gps) == 1 and gps[0].get("properties", {}).get("observation_count") == n and len(gps[0].get("properties", {}).get("timestamps", [])) == n)
    else:
        # These facts are captured from the actual complete response before it is discarded.
        checks.update({key: row.get("response_checks", {}).get(key) is True for key in ("private_reference_not_in_response", "full_original_observation_order", "raw_gps_count")})
    return checks


def summarize(payload, dataset_id, tid, status, elapsed, offline):
    from scripts.live_evidence import summarize_response
    row = summarize_response(payload, tid, status, elapsed)
    row["dataset_id"] = payload.get("dataset_id")
    checks = response_checks(row, offline, dataset_id, payload)
    row["response_checks"] = checks
    return row


class PostPredictionAudit:
    """Transparent measurement, not a mock scorer or a replacement prediction."""
    def __init__(self, frozen_config):
        self.frozen_config = frozen_config
        self.rows = []

    def __call__(self, graph, track, result, model, config, reference_path):
        import numpy as np
        from backend.scoring import score_prediction
        from mapmatch.learning import EmissionModel
        from sklearn.ensemble import HistGradientBoostingClassifier
        records = model.records
        maximum_error = 0.0
        all_probabilities_valid = True
        candidate_count = 0
        digest = hashlib.sha256()
        for index in sorted(records):
            record = records[index]
            p = record.probabilities
            error = abs(float(np.sum(p)) - 1.0)
            maximum_error = max(maximum_error, error)
            all_probabilities_valid &= bool(len(p) and np.isfinite(p).all() and np.all(p > 0) and np.all(p <= 1) and error <= 1e-12 and p.shape == record.edges.shape)
            candidate_count += len(p)
            digest.update(np.asarray([index], dtype="<i8").tobytes())
            digest.update(np.asarray(record.edges, dtype="<i8").tobytes())
            digest.update(np.asarray(p, dtype="<f8").tobytes())
        checks = {
            "actual_emission_model": isinstance(model.model, EmissionModel),
            "actual_histgb_estimator": isinstance(model.model.estimator, HistGradientBoostingClassifier),
            "all_observations_audited": set(records) == set(range(len(track))),
            "all_candidate_probabilities_finite_positive_normalized": bool(all_probabilities_valid),
            "post_prediction_complete": np.array_equal(result.observation_indices, np.arange(len(track))) and len(result.matched_edges) == len(track) and len(result.route) > 0,
            "frozen_full_configuration": asdict(config) == self.frozen_config,
            "classifier_was_called": model.call_count >= len(track),
        }
        # The real scorer performs all reference loading, proxy alignment and NLL.
        summary = score_prediction(graph, track, result, model, config, reference_path)
        checks["actual_private_scorer_called"] = True
        self.rows.append({"checks": checks, "learned_emission_observations": len(records),
                          "learned_candidate_probability_count": candidate_count,
                          "classifier_call_count": model.call_count,
                          "probability_sum_error_max": maximum_error,
                          "candidate_probability_sha256": digest.hexdigest(),
                          "reference_available_after_prediction": reference_path is not None,
                          "metric_status": summary.metrics_status})
        return summary


def cli_fixture_checks(runtime, dataset_id, tid, expected, root=ROOT):
    """Open only existing CLI predictions, graph/GPS; never open a reference."""
    import numpy as np
    from mapmatch.data import load_route
    from scripts.live_evidence import integer_array_hash
    from scripts.verify_phase_three_delivery import geometry_hash
    trip = runtime.registry.get(tid)
    graph = runtime.graphs.get(trip)
    route_file = Path(root) / OUTPUTS[dataset_id] / "routes" / (tid + ".route")
    route = load_route(route_file)
    with (Path(root) / OUTPUTS[dataset_id] / "observations" / (tid + ".csv")).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    checks = {"actual_cli_route_bytes": sha256(route_file) == expected["route_sha256"],
              "actual_cli_geometry": geometry_hash(graph, route) == expected["predicted_geometry_sha256"],
              "actual_cli_states": integer_array_hash([int(r["predicted_arc_id"]) for r in rows]) == expected["matched_edge_array_sha256"],
              "actual_cli_count": len(rows) == len(trip.track) == expected["observations"],
              "actual_cli_original_order": [int(r["observation_index"]) for r in rows] == list(range(len(trip.track))),
              "actual_cli_original_timestamps": np.array_equal([float(r["timestamp_s"]) for r in rows], trip.track[:, 2])}
    return checks


def validate_api_acceptance(report, datasets, expected):
    require(report.get("passed") is True and report.get("trip_count") == 20 and report.get("mocked_model_or_inference") is False, "actual unmocked API20 acceptance missing")
    require(report.get("model_sha256") == expected["model_sha256"] and report.get("config_file_sha256") == expected["config_file_sha256"] and report.get("match_config_hash") == expected["match_config_hash"], "API20 frozen identity mismatch")
    require(report.get("frozen_config_before") == report.get("frozen_config_after") == expected["frozen_config"], "API20 full configuration before/after mismatch")
    require_checks(report.get("checks"), "API20 global", REQUIRED_INTEGRATION_CHECKS)
    require(set(report.get("datasets", {})) == set(DATASET_IDS), "API20 dataset coverage missing")
    ids = []
    for dataset_id in DATASET_IDS:
        group = report["datasets"][dataset_id]
        require(group.get("trip_count") == 10 and set(group.get("trips", {})) == set(datasets[dataset_id]), "API20 per-dataset ten-trip coverage missing")
        require_checks(group.get("checks"), "API20 dataset " + dataset_id)
        for tid, row in group["trips"].items():
            require(row.get("trip_id") == tid and row.get("dataset_id") == dataset_id, "API20 row identity mismatch")
            require_checks(row.get("checks"), "API20 " + dataset_id + ":" + tid, REQUIRED_ROW_CHECKS)
            require_checks(response_checks(row, datasets[dataset_id][tid], dataset_id), "API20 recomputed " + dataset_id + ":" + tid, REQUIRED_ROW_CHECKS)
            require_checks(row.get("cli_fixture_checks"), "API20 actual CLI fixture")
            require_checks(row.get("candidate_audit", {}).get("checks"), "API20 real candidate audit", REQUIRED_AUDIT_CHECKS)
            audit = row["candidate_audit"]
            require(audit.get("learned_emission_observations") == datasets[dataset_id][tid]["observations"] and audit.get("learned_candidate_probability_count", 0) >= audit["learned_emission_observations"], "API20 learned candidate count mismatch")
            require(math.isfinite(audit.get("probability_sum_error_max", math.inf)) and audit["probability_sum_error_max"] <= 1e-12, "API20 candidate probabilities inconsistent")
            ids.append(row["request_id"])
    require(len(ids) == len(set(ids)) == 20, "API20 request identities not unique")
    repeated = report.get("repeated_request", {})
    require(repeated.get("excluded_from_twenty_headline") is True and repeated.get("request_id") not in ids, "extra repeat not separately and freshly identified")
    require_checks(repeated.get("checks"), "separate repeat")
    return True


def run_integration(root=ROOT):
    from fastapi.testclient import TestClient
    from backend.app import create_app
    from backend.config import Settings
    from backend.service import InferenceRuntime
    from mapmatch.learning import EmissionModel
    expected, offline = load_offline_expected(root)  # Must precede all actual evaluator input reads.
    before_sources = source_manifest(root)
    before_files = {str(p): sha256(Path(root) / p) for p in (MODEL, METADATA, CONFIG)}
    settings = Settings(data_dir=Path(root) / "Student folder/student_test", reference_dir=None,
                        model_path=Path(root) / MODEL, metadata_path=Path(root) / METADATA,
                        match_config_path=Path(root) / CONFIG, match_config_required=True,
                        expected_trip_count=10, dataset_role="development", graph_cache_size=2,
                        inference_concurrency=1, frontend_dir=Path(root) / "frontend",
                        evaluator_data_dir=Path(root) / "evaluator_test",
                        evaluator_reference_dir=Path(root) / "evaluator_test", evaluator_expected_trip_count=10)
    observer = PostPredictionAudit(expected["frozen_config"])
    runtime = InferenceRuntime(settings, scorer=observer)
    app = create_app(settings, runtime=runtime)
    report = {"checked_at_utc": utcnow(), "passed": False, "trip_count": 20, "dataset_count": 2,
              "transport": "in-process ASGI TestClient; no sockets and not public-network proof",
              "actual_estimator": "HistGradientBoostingClassifier", "mocked_model_or_inference": False,
              "post_prediction_observer": "scalar-only transparent observer delegates backend.scoring.score_prediction unchanged",
              **expected, "prediction_cache": False, "frozen_config_before": expected["frozen_config"],
              "source_sha256_before": before_sources, "datasets": {}, "checks": {}}
    with TestClient(app) as client:
        health_response, catalogue_response = client.get("/health"), client.get("/datasets")
        require(health_response.status_code == catalogue_response.status_code == 200, "health/catalogue unavailable")
        health, catalogue = health_response.json(), catalogue_response.json()
        require_checks(health_checks(health, catalogue), "actual ready ASGI datasets")
        report["health"], report["catalogue"] = health, catalogue
        for dataset_id in DATASET_IDS:
            catalogue_response = client.get("/datasets/" + dataset_id + "/trips")
            require(catalogue_response.status_code == 200, "dataset trips unavailable")
            trips = catalogue_response.json()
            group_checks = {"exact_ten_registered_ids": len(trips["trips"]) == 10 and {r["trip_id"] for r in trips["trips"]} == set(offline[dataset_id]),
                            "dataset_role": trips.get("dataset_id") == trips.get("dataset_role") == dataset_id}
            require_checks(group_checks, dataset_id + " catalogue")
            group = {"trip_count": 10, "checks": group_checks, "trips": {}}
            report["datasets"][dataset_id] = group
            for tid in sorted(offline[dataset_id]):
                fixture_checks = cli_fixture_checks(app.state.datasets.runtime_for(dataset_id), dataset_id, tid, offline[dataset_id][tid], root)
                require_checks(fixture_checks, dataset_id + ":" + tid + ": actual CLI fixture")
                before_audits = len(observer.rows)
                started = time.perf_counter()
                response = client.post("/datasets/" + dataset_id + "/match", json={"trip_id": tid})
                payload = response.json()
                require(response.status_code == 200, dataset_id + ":" + tid + ": real inference HTTP failure")
                require(len(observer.rows) == before_audits + 1, "actual scorer/audit was not called exactly once")
                row = summarize(payload, dataset_id, tid, response.status_code, (time.perf_counter()-started)*1000, offline[dataset_id][tid])
                row.update(checks=row["response_checks"], cli_fixture_checks=fixture_checks, candidate_audit=observer.rows[-1])
                row["checks"]["request_id_header"] = response.headers.get("x-request-id") == row["request_id"]
                row["checks"]["no_store_header"] = "no-store" in response.headers.get("cache-control", "")
                require_checks(row["checks"], dataset_id + ":" + tid, REQUIRED_ROW_CHECKS)
                require_checks(row["candidate_audit"]["checks"], "actual learned audit", REQUIRED_AUDIT_CHECKS)
                group["trips"][tid] = row
                print(json.dumps({"event": "actual_final_api_trip_passed", "dataset_id": dataset_id, "trip_id": tid,
                                  "observations": offline[dataset_id][tid]["observations"], "route_length_f1": row["metrics"]["route_length_f1"], "request_id": row["request_id"]}), flush=True)
        # A separately counted repeat proves the real scorer/audit executes again.
        tid = sorted(offline["development"])[0]
        old = report["datasets"]["development"]["trips"][tid]
        count = len(observer.rows)
        started = time.perf_counter()
        response = client.post("/datasets/development/match", json={"trip_id": tid})
        repeated = summarize(response.json(), "development", tid, response.status_code, (time.perf_counter()-started)*1000, offline["development"][tid])
        repeat_checks = response_checks(repeated, offline["development"][tid], "development")
        repeat_checks.update(new_request_id=repeated["request_id"] != old["request_id"], actual_new_audit=len(observer.rows) == count + 1,
                             same_real_prediction=repeated["predicted_geometry_sha256"] == old["predicted_geometry_sha256"] and repeated["matched_edge_array_sha256"] == old["matched_edge_array_sha256"])
        require_checks(repeat_checks, "separate repeated inference")
        report["repeated_request"] = {**repeated, "checks": repeat_checks, "excluded_from_twenty_headline": True}
        primary, evaluator = app.state.datasets.primary, app.state.datasets.evaluator
        report["checks"].update(shared_actual_model=isinstance(primary.model, EmissionModel) and primary.model is evaluator.model,
                                  shared_frozen_config=primary.match_config is evaluator.match_config,
                                  shared_concurrency_gate=primary._slots is evaluator._slots,
                                  separate_registries_and_graph_caches=primary.registry is not evaluator.registry and primary.graphs is not evaluator.graphs,
                                  frozen_full_config_unchanged=asdict(primary.match_config) == asdict(evaluator.match_config) == expected["frozen_config"])
        report["frozen_config_after"] = asdict(primary.match_config)
    rows = [row for group in report["datasets"].values() for row in group["trips"].values()]
    report["checks"].update(exact_twenty_unique_dataset_trip_pairs=len(rows) == len({(r["dataset_id"], r["trip_id"]) for r in rows}) == 20,
                              unique_request_ids=len({r["request_id"] for r in rows}) == 20,
                              no_prediction_cache=health["prediction_cache"] is False,
                              both_catalogues_ready=all(catalogue_checks(catalogue).values()),
                              frozen_files_unchanged=before_files == {str(p): sha256(Path(root) / p) for p in (MODEL, METADATA, CONFIG)},
                              inference_sources_unchanged=before_sources == source_manifest(root),
                              repeated_trip_fresh_inference=all(repeat_checks.values()))
    report["source_sha256_after"] = source_manifest(root)
    audit = load_json(Path(root) / FINAL / "evaluator/input_audit.json")
    report["evaluator_overlap_audit"] = {"source": str(FINAL / "evaluator/input_audit.json"), "sha256": sha256(Path(root) / FINAL / "evaluator/input_audit.json"),
                                         "overlap_report": audit.get("overlap_report"),
                                         "caveat": "Hash/ID overlap evidence only; no claim of geographical or official blind independence."}
    report["passed"] = True
    validate_api_acceptance(report, offline, expected)
    report["completed_at_utc"] = utcnow()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / FINAL / "api_integration_acceptance.json")
    parser.add_argument("--check", action="store_true", help="Validate frozen/offline gates only; no raw input or inference")
    args = parser.parse_args(argv)
    if args.check:
        expected, offline = load_offline_expected()
        print(json.dumps({"passed": True, "raw_inputs_opened": False, "dataset_trip_counts": {k: len(v) for k, v in offline.items()}, "model_sha256": expected["model_sha256"], "match_config_hash": expected["match_config_hash"]}, indent=2))
        return
    target = args.output.resolve()
    require(target == (ROOT / FINAL / "api_integration_acceptance.json").resolve(), "only the task-owned canonical acceptance path is permitted")
    require(not target.exists() and not target.is_symlink(), "refuse to overwrite actual API acceptance")
    report = run_integration()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "actual_twenty_trip_count": 20, "additional_repeat_count": 1, "model_sha256": MODEL_SHA256, "match_config_hash": COMPACT_CONFIG_SHA256}), flush=True)


if __name__ == "__main__":
    main()
