#!/usr/bin/env python3
"""One fixed postfreeze evaluator pass; references open only after all predictions."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mapmatch.data import discover_trips, load_graph, load_route, load_track, validate_route
from mapmatch.features import reference_observation_labels, PROXY_LABEL_POLICY
from mapmatch.metrics import score_routes
from backend.scoring import classification_metrics
from scripts.verify_phase_three_delivery import geometry_hash, integer_array_hash

MODEL_SHA = "c9cc8e9f36838f656a94d07ed69632979df9981ca94a9daf3b52fbce052e4b34"
CONFIG_SHA = "10405aa6b6f370e41d4ddcb01207fd1e8f58900fd0395e541ee6bc78146168c5"
IDS = tuple(f"{i:08d}" for i in (15, 17, 23, 30, 49, 59, 67, 73, 87, 94))
INPUT = ROOT / "evaluator_test"
OUTPUT = ROOT / "outputs/final_evaluator_test"
REPORT = ROOT / "reports/final_five/evaluator"
PROTOCOL = ROOT / "docs/FINAL_EVALUATOR_PROTOCOL.md"
MODEL = ROOT / "artifacts/final_five/model.joblib"
CONFIG = ROOT / "artifacts/final_five/match_config.json"
SUFFIXES = (".track", ".nodes", ".arcs")
COMMAND = [".venv/bin/python", "-u", "-B", "-m", "mapmatch", "predict", "--input", "evaluator_test", "--output", "outputs/final_evaluator_test", "--model", "artifacts/final_five/model.joblib", "--config", "artifacts/final_five/match_config.json"]
NLL_REASON = "The actual CLI saves classifier argmax IDs, not full candidate probabilities or membership. No inference was repeated to reconstruct NLL or candidate misses."
ROUTE_KEYS = ("edge_precision", "edge_recall", "edge_f1", "length_precision", "length_recall", "length_f1", "duplicate_canonical_length_f1", "edit_similarity", "route_continuity")
PROXY_KEYS = ("accuracy", "precision", "recall", "macro_f1", "classifier_accuracy", "classifier_precision", "classifier_recall", "classifier_macro_f1", "top1_accuracy", "scored_observation_fraction", "ambiguous_observation_fraction")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(value, reason):
    if not value:
        raise ValueError(reason)


def safe_path(path, root, *, must_exist=True):
    path, root = Path(path), Path(root)
    require(path.is_absolute() and root.is_absolute(), "paths must be explicit absolute paths internally")
    require(path.is_relative_to(root), "path is outside trusted root")
    for part in (path, *path.parents):
        require(not part.is_symlink(), "symlinks are forbidden")
        if part == root:
            break
    require(path.resolve().is_relative_to(root.resolve()), "resolved path is outside trusted root")
    if must_exist:
        require(path.exists(), "required path is absent")
    return path


def refuse_existing(*paths):
    for path in paths:
        require(not Path(path).exists() and not Path(path).is_symlink(), "refuse overwrite of evaluator evidence")


def write_json(path, value):
    path = Path(path)
    require(any(path.is_relative_to(base) for base in (REPORT, OUTPUT)), "write outside exclusive evaluator scope")
    safe_path(path, ROOT, must_exist=False)
    refuse_existing(path)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    require(str(ROOT) not in text and '"coordinates"' not in text, "private data in public report")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def parsed_time(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "freeze timestamp lacks timezone")
    return result


def validate_freeze():
    """All evidence and artifact checks precede every evaluator input open."""
    now = datetime.now(timezone.utc)
    protocol = PROTOCOL.read_text(encoding="utf-8")
    match = re.search(r"Preregistered at (\S+), before", protocol)
    require(match is not None, "missing preregistration timestamp")
    registered = parsed_time(match.group(1))
    paths = {
        "model_freeze": "artifacts/final_five/model_freeze.json",
        "decoder_freeze": "reports/final_five/decoder/freeze.json",
        "all_five_acceptance": "reports/final_five/all_five_model_acceptance.json",
        "clean_load": "artifacts/final_five/clean_load_report.json",
        "independent_review": "analysis/final_model_review/final_review_report.json",
        "student_delivery": "reports/final_five/offline_delivery_verification.json",
        "metadata": "artifacts/final_five/model_metadata.json",
        "feature_schema": "artifacts/final_five/feature_schema.json",
    }
    evidence = {name: read_json(safe_path(ROOT / relative, ROOT)) for name, relative in paths.items()}
    mf, df, acceptance, clean, review, delivery, metadata = (evidence[k] for k in ("model_freeze", "decoder_freeze", "all_five_acceptance", "clean_load", "independent_review", "student_delivery", "metadata"))
    require(sha256(safe_path(MODEL, ROOT)) == MODEL_SHA == mf["model_sha256"] == df["model_sha256"] == acceptance["selected_model_sha256"] == clean["model_sha256"] == review["selected_model"]["frozen_model_sha256"] == delivery["model_sha256"], "selected model hash mismatch")
    require(sha256(safe_path(CONFIG, ROOT)) == CONFIG_SHA == df["config_file_sha256"] == delivery["config_file_sha256"], "decoder hash mismatch")
    require(sha256(ROOT / paths["metadata"]) == mf["metadata_sha256"] and sha256(ROOT / paths["feature_schema"]) == mf["feature_schema_sha256"], "frozen metadata or schema changed")
    require(MODEL.stat().st_size == mf["bytes"], "model byte count mismatch")
    require(parsed_time(mf["frozen_at_utc"]) < registered and parsed_time(df["frozen_utc"]) < registered <= now, "freeze or preregistration is not in the past")
    require(not mf["evaluator_used_for_selection"] and not mf["student_used_for_selection"], "selection boundary failed")
    require(acceptance["passed"] is True and acceptance["exactly_five_families"] is True and len(acceptance["families"]) == 5 and not acceptance["evaluator_accessed"], "all-five acceptance failed")
    require(clean["passed"] is True and clean["max_probability_delta"] <= 1e-12, "clean load failed")
    require(review["passed"] is True and review["check_counts"]["main_verification_checks"] == "111/111" and not review["critical_defects"], "independent review failed")
    require(delivery["passed"] is True and len(delivery["trips"]) == 10 and all(delivery["checks"].values()) and all(all(t["checks"].values()) for t in delivery["trips"].values()), "student route verification failed")
    require(parsed_time(delivery["checked_at_utc"]) < registered, "student verification must predate evaluator preregistration")
    artifact_hashes = {}
    for name, row in acceptance["all_models"].items():
        require(row["passed"] is True and row["training_trip_count"] == 80 and row["student_trip_count"] == 10, "family acceptance incomplete")
        path = safe_path(ROOT / row["artifact"], ROOT)
        require(sha256(path) == row["sha256"], "accepted model artifact changed")
        artifact_hashes[name] = row["sha256"]
    for name, expected in df["source_sha256"].items():
        require(sha256(safe_path(ROOT / name, ROOT)) == expected, "frozen inference or scoring source changed")
    config = read_json(CONFIG)
    require(config == df["config"] and config["max_candidates"] == 12 and config["candidate_retry_limit"] == 48 and config["observation_distance_m"] == 0, "fixed decoder policy mismatch")
    from mapmatch.learning import load_model
    bundle = load_model(MODEL)
    require(type(bundle["estimator"]).__name__ == "HistGradientBoostingClassifier", "wrong selected estimator")
    require(metadata["training_trip_count"] == len(set(metadata["training_trip_ids"])) == len(metadata["training_trip_ids"]) == 80, "TRAIN metadata is not all eighty unique trips")
    for key in ("model_name", "run_name", "parameters", "training_trip_ids", "feature_version"):
        require(bundle["metadata"][key] == metadata[key], "model and frozen metadata disagree")
    require(all(bundle["estimator"].get_params()[k] == v for k, v in metadata["parameters"].items()), "selected estimator parameters changed")
    return {
        "checked_at_utc": utc(), "passed": True,
        "model_sha256": MODEL_SHA, "config_file_sha256": CONFIG_SHA,
        "api_compact_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "model_frozen_at_utc": mf["frozen_at_utc"], "decoder_frozen_at_utc": df["frozen_utc"],
        "preregistered_at_utc": registered.isoformat(), "protocol_sha256": sha256(PROTOCOL),
        "evidence_sha256": {name: sha256(ROOT / relative) for name, relative in paths.items()},
        "accepted_artifact_sha256": artifact_hashes, "source_sha256": df["source_sha256"],
        "training_trip_ids": metadata["training_trip_ids"], "student_trip_ids": sorted(delivery["trips"]),
        "candidate_width_policy": [12, 24, 48], "selected_model_loaded_without_inference": True,
    }


def exact_trips(root, ids):
    safe_path(root, ROOT)
    for path in root.rglob("*"):
        safe_path(path, root)
    prefixes = discover_trips(root)
    require(len(prefixes) == len(ids) and {p.name for p in prefixes} == set(ids), "trip set differs from preregistered set")
    for prefix in prefixes:
        require(prefix == root / prefix.name / prefix.name, "trip files must use exact id/id paths")
        for extension in SUFFIXES:
            require(safe_path(prefix.with_suffix(extension), root).is_file(), "missing regular trip file")
        reference = prefix.with_suffix(".route")
        safe_path(reference, root, must_exist=False)
    return {p.name: p for p in prefixes}


def input_hashes(prefix):
    return {extension: sha256(prefix.with_suffix(extension)) for extension in SUFFIXES}


def structural_audit(prefix):
    graph, track = load_graph(prefix), load_track(prefix.with_suffix(".track"))
    projected = graph.project(track[:, :2])
    require(np.isfinite(graph.xy).all() and np.isfinite(projected).all() and np.isfinite(graph.edge_lengths).all(), "nonfinite local projection")
    delta = np.diff(track[:, 2])
    return {
        "trip_id": prefix.name, "valid": True, "nodes": len(graph.nodes), "arcs": len(graph.arcs), "observations": len(track),
        "duration_s": float(track[-1, 2] - track[0, 2]),
        "timestamp_range_s": [float(track[0, 2]), float(track[-1, 2])],
        "timestamp_step_range_s": [float(delta.min()), float(delta.max())] if len(delta) else None,
        "graph_span_m": np.ptp(graph.xy, axis=0).tolist(), "track_span_m": np.ptp(projected, axis=0).tolist(),
        "edge_length_range_m": [float(graph.edge_lengths.min()), float(graph.edge_lengths.max())],
        "zero_length_arc_count": int(np.sum(graph.edge_lengths == 0)),
        "duplicate_directed_arc_rows": int(len(graph.arcs) - len(np.unique(graph.arcs, axis=0))),
        "sha256_by_extension": {**input_hashes(prefix), ".route": None},
        "reference_hash_status": "deferred until all predictions pass verification", "reference_parsed": False,
        "finite_local_coordinates": True, "strict_original_timestamp_order": True,
    }


def overlap_audit(rows, training_ids, student_ids):
    groups = (("TRAIN80", ROOT / "Student folder/train", training_ids), ("student_development10", ROOT / "Student folder/student_test", student_ids))
    duplicates, extension_matches, id_matches = [], [], []
    for fold, folder, ids in groups:
        comparison = exact_trips(folder, ids)
        hashes = {tid: input_hashes(prefix) for tid, prefix in comparison.items()}
        for row in rows:
            tid, values = row["trip_id"], row["sha256_by_extension"]
            if tid in comparison:
                id_matches.append({"evaluator_trip_id": tid, "fold": fold, "comparison_trip_id": tid})
            for other, digest in hashes.items():
                matched = [ext for ext in SUFFIXES if values[ext] == digest[ext]]
                if matched:
                    extension_matches.append({"evaluator_trip_id": tid, "fold": fold, "comparison_trip_id": other, "matching_extensions": matched})
                if len(matched) == len(SUFFIXES):
                    duplicates.append({"evaluator_trip_id": tid, "fold": fold, "comparison_trip_id": other})
    return {"training_trip_count_checked": len(training_ids), "student_trip_count_checked": len(student_ids),
            "id_matches": id_matches, "full_trip_duplicates": duplicates, "per_extension_matches": extension_matches,
            "any_identical_input_file": bool(extension_matches), "any_full_trip_duplicate": bool(duplicates),
            "independent_holdout_claimed": False,
            "interpretation": "Exact input hashes checked against all frozen TRAIN80 and student-development10. ID matches alone do not establish overlap. Any duplicate content limits independence; absence of exact duplicates does not prove geographical or official unseen status."}


def current_inputs(audit, freeze):
    require(audit["freeze_validation_sha256"] == sha256(REPORT / "freeze_validation.json"), "pre-input freeze evidence changed")
    require(audit["protocol_sha256"] == freeze["protocol_sha256"], "protocol changed after first input")
    require(parsed_time(audit["first_input_open_started_utc"]) > parsed_time(freeze["preregistered_at_utc"]), "input preceded preregistration")
    prefixes = exact_trips(INPUT, IDS)
    for row in audit["trips"]:
        require(input_hashes(prefixes[row["trip_id"]]) == {k: row["sha256_by_extension"][k] for k in SUFFIXES}, "audited evaluator inputs changed")
    return prefixes


def stage_audit():
    refuse_existing(REPORT / "freeze_validation.json", REPORT / "stage_plan.json", REPORT / "input_audit_pre_prediction.json", REPORT / "prediction_attempt.json")
    safe_path(OUTPUT, ROOT, must_exist=False)
    require(not OUTPUT.exists() or not any(OUTPUT.iterdir()), "CLI output must be empty before audit")
    frozen = validate_freeze()
    write_json(REPORT / "freeze_validation.json", frozen)
    write_json(REPORT / "stage_plan.json", {"created_at_utc": utc(), "objective": "fixed postfreeze all-ten final evaluation", "state": "planned; no prediction complete", "trip_ids": IDS, "command": COMMAND,
        "stages": ["audit all ten graph/track files and TRAIN80/student10 overlap", "one actual original-venv learned CLI", "verify every prediction and original observation", "only then open references and score"],
        "no_adjustment": True, "references_read": False, "candidate_width_policy": [12, 24, 48]})
    first_open = utc()
    prefixes = exact_trips(INPUT, IDS)
    rows = [structural_audit(prefixes[tid]) for tid in IDS]
    overlap = overlap_audit(rows, frozen["training_trip_ids"], frozen["student_trip_ids"])
    validate_freeze()
    audit = {"trip_count": len(rows), "first_input_open_started_utc": first_open, "completed_at_utc": utc(),
             "protocol_sha256": frozen["protocol_sha256"], "freeze_validation_sha256": sha256(REPORT / "freeze_validation.json"),
             "model_sha256": MODEL_SHA, "config_file_sha256": CONFIG_SHA, "references_parsed": False, "trips": rows, "overlap_report": overlap}
    write_json(REPORT / "input_audit_pre_prediction.json", audit)
    return audit


def verify_predictions(audit, frozen):
    prefixes = current_inputs(audit, frozen)
    config, run = read_json(CONFIG), read_json(OUTPUT / "run_config.json")
    with (OUTPUT / "index.csv").open(newline="") as stream:
        index = list(csv.DictReader(stream))
    checks = {"exact_ten_successful_index_rows": len(index) == len(IDS) and {r["trip_id"] for r in index} == set(IDS) and all(r["status"] == "ok" for r in index),
              "learned_model_hash": run["model_sha256"] == MODEL_SHA, "fixed_config": run["config"] == config,
              "learned_not_baseline": run["baseline"] is False, "no_reference_in_prediction": run["uses_reference_routes"] is False,
              "all_requested_trips": run["trip_ids"] == list(IDS), "no_failures": read_json(OUTPUT / "failures.json") == []}
    expected = {}
    with ZipFile(OUTPUT / "submission.zip") as archive:
        checks["zip_flat_exact_ten_routes"] = len(archive.namelist()) == len(IDS) and set(archive.namelist()) == {tid + ".route" for tid in IDS}
        checks["zip_integrity"] = archive.testzip() is None
        checks["no_stale_route_files"] = {p.name for p in (OUTPUT / "routes").iterdir()} == {tid + ".route" for tid in IDS}
        for item in index:
            tid, prefix = item["trip_id"], prefixes[item["trip_id"]]
            graph, track = load_graph(prefix), load_track(prefix.with_suffix(".track"))
            path = safe_path(OUTPUT / "routes" / (tid + ".route"), OUTPUT)
            route = load_route(path)
            diagnostics = read_json(OUTPUT / "diagnostics" / (tid + ".json"))
            with (OUTPUT / "observations" / (tid + ".csv")).open(newline="") as stream:
                observations = list(csv.DictReader(stream))
            edges = np.array([int(row["predicted_arc_id"]) for row in observations], dtype=np.int64)
            classifier = np.array([int(row["classifier_only_arc_id"]) for row in observations], dtype=np.int64)
            numeric = np.asarray([[float(row[k]) for k in ("timestamp_s", "fraction", "lateral_distance_m")] for row in observations])
            width = diagnostics["effective_max_candidates"]
            tries = diagnostics["candidate_retry_attempts"]
            expected_tries = [12, 24, 48][: [12, 24, 48].index(width)] if width in (12, 24, 48) else None
            effective_config = {**config, "max_candidates": width}
            tc = {"nonempty_connected_directed_route": bool(len(route)) and validate_route(graph, route)["valid"],
                  "route_sha_matches_index": sha256(path) == item["sha256"], "route_bytes_match_zip": archive.read(tid + ".route") == path.read_bytes(),
                  "route_count_matches_index": len(route) == int(item["arc_count"]) == diagnostics["route_arc_count"],
                  "all_original_observations": len(observations) == len(track) == diagnostics["observation_count"] == diagnostics["matched_observation_count"],
                  "original_observation_order": [int(row["observation_index"]) for row in observations] == list(range(len(track))),
                  "original_timestamps": np.array_equal(numeric[:, 0], track[:, 2]),
                  "finite_saved_observation_fields": bool(np.isfinite(numeric).all()),
                  "valid_matched_and_classifier_ids": bool(np.all((edges >= 0) & (edges < len(graph.arcs)))) and bool(np.all((classifier >= 0) & (classifier < len(graph.arcs)))),
                  "valid_projection_fields": bool(np.all((numeric[:, 1] >= 0) & (numeric[:, 1] <= 1)) and np.all(numeric[:, 2] >= 0)),
                  "classifier_columns_match_actual_diagnostics": classifier.tolist() == diagnostics["classifier_only_edges"],
                  "learned_emission_model_loaded": diagnostics["emission_model"] == "learned_classifier_plus_geometry" and run["baseline"] is False and run["model_sha256"] == MODEL_SHA,
                  "no_reference_used": diagnostics["uses_reference_route"] is False,
                  "fixed_requested_and_effective_config": diagnostics["requested_config"] == config and diagnostics["config"] == effective_config,
                  "bounded_fixed_retry_widths": expected_tries is not None and [t["max_candidates"] for t in tries] == expected_tries and all("no connected directed transition" in t["failure"] for t in tries),
                  "finite_learned_path_cost": math.isfinite(diagnostics["negative_log_likelihood_relative"]),
                  "no_routing_breaks": diagnostics["breaks"] == []}
            expected[tid] = {"checks": tc, "observations": len(track), "learned_emission_observations": len(classifier),
                "learned_emission_count_basis": "Saved actual classifier IDs for every original observation; frozen matcher validates probability shape, finite nonnegative values and positive sum before recording each ID.",
                "predicted_arc_count": len(route), "route_sha256": sha256(path), "matched_edge_array_sha256": integer_array_hash(edges),
                "classifier_only_edge_array_sha256": integer_array_hash(classifier), "predicted_geometry_sha256": geometry_hash(graph, route),
                "effective_max_candidates": width, "retry_widths": [t["max_candidates"] for t in tries],
                "elapsed_s": float(item["elapsed_s"]),
                "probability_audit": {"finite_probability_validation_passed_by_frozen_matcher": tc["learned_emission_model_loaded"] and tc["finite_learned_path_cost"] and tc["all_original_observations"],
                    "raw_probability_arrays_serialized": False, "independent_saved_probability_sum_check": None,
                    "unavailable_reason": NLL_REASON}, "metrics": None}
    return {"checked_at_utc": utc(), "model_sha256": MODEL_SHA, "config_file_sha256": CONFIG_SHA,
            "api_compact_config_sha256": frozen["api_compact_config_sha256"], "checks": checks, "trips": expected,
            "submission_zip_sha256": sha256(OUTPUT / "submission.zip"), "submission_zip_member_count": len(IDS),
            "passed": all(checks.values()) and len(expected) == len(IDS) and all(all(t["checks"].values()) for t in expected.values()),
            "reference_files_parsed": False}


def stage_predict():
    refuse_existing(REPORT / "prediction_attempt.json", REPORT / "cli_result.json", REPORT / "prediction_verification.json")
    frozen = validate_freeze()
    audit = read_json(REPORT / "input_audit_pre_prediction.json")
    current_inputs(audit, frozen)
    safe_path(OUTPUT, ROOT, must_exist=False)
    require(not OUTPUT.exists() or not any(OUTPUT.iterdir()), "refuse nonempty CLI output")
    write_json(REPORT / "prediction_attempt.json", {"started_at_utc": utc(), "attempt_number": 1, "command": COMMAND,
        "model_sha256": MODEL_SHA, "config_file_sha256": CONFIG_SHA, "references_read": False, "state": "started; result not yet known"})
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    start = time.perf_counter()
    result = subprocess.run(COMMAND, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    elapsed, finished = time.perf_counter() - start, utc()
    log = OUTPUT / "observations/actual_cli.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        stream.write(result.stdout)
    write_json(REPORT / "cli_result.json", {"completed_at_utc": finished, "exit_code": result.returncode, "elapsed_s": elapsed, "attempt_number": 1,
        "private_log_sha256": sha256(log), "references_read": False, "success": result.returncode == 0})
    require(result.returncode == 0, "actual CLI failed; partial results preserved and retries prohibited")
    frozen = validate_freeze()
    report = verify_predictions(audit, frozen)
    write_json(REPORT / "prediction_verification.json", report)
    require(report["passed"], "prediction verification failed; references remain unopened")
    return report


def mean_available(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def stage_score():
    refuse_existing(REPORT / "reference_scoring_started.json", REPORT / "route_metrics.json", REPORT / "input_audit.json", REPORT / "acceptance.json", REPORT / "expected_trips.json")
    frozen = validate_freeze()
    audit = read_json(REPORT / "input_audit_pre_prediction.json")
    result = read_json(REPORT / "cli_result.json")
    verified = read_json(REPORT / "prediction_verification.json")
    require(result["exit_code"] == 0 and verified["passed"] is True, "all-ten prediction success is required before reference scoring")
    current = verify_predictions(audit, frozen)
    require(current["passed"] and current["submission_zip_sha256"] == verified["submission_zip_sha256"] and current["trips"] == verified["trips"], "predictions changed after verification")
    prefixes = exact_trips(INPUT, IDS)
    started = utc()
    require(parsed_time(result["completed_at_utc"]) <= parsed_time(verified["checked_at_utc"]) < parsed_time(started), "invalid prediction-before-scoring chronology")
    write_json(REPORT / "reference_scoring_started.json", {"started_at_utc": started, "all_ten_predictions_verified": True,
        "prediction_verification_sha256": sha256(REPORT / "prediction_verification.json"), "references_previously_parsed_by_this_evaluator": False})
    rows, truths, predictions, classifiers = [], [], [], []
    namespace = 0
    final_inputs = {row["trip_id"]: row for row in audit["trips"]}
    for tid in IDS:
        prefix = prefixes[tid]
        graph, track = load_graph(prefix), load_track(prefix.with_suffix(".track"))
        refpath = prefix.with_suffix(".route")
        row = {"trip_id": tid, **dict.fromkeys((*ROUTE_KEYS, *PROXY_KEYS, "exact_match", "reference_discontinuity_count", "candidate_miss_rate", "log_loss")), "hmm_posterior_nll": None,
               "candidate_nll_unavailable_reason": NLL_REASON, "observations": len(track), "metrics_status": "unavailable", "unavailable_reason": None}
        if not refpath.is_file():
            row["unavailable_reason"] = "No private reference file supplied."
            final_inputs[tid]["reference_hash_status"] = "missing reference"
        else:
            safe_path(refpath, INPUT)
            final_inputs[tid]["sha256_by_extension"][".route"] = sha256(refpath)
            final_inputs[tid]["reference_hash_status"] = "hashed only after all predictions verified"
            try:
                reference = load_route(refpath)
                final_inputs[tid]["reference_parsed"] = True
                require(len(reference) and np.all((reference >= 0) & (reference < len(graph.arcs))), "empty or invalid exact original reference arc IDs")
                route = load_route(OUTPUT / "routes" / (tid + ".route"))
                row.update(score_routes(graph, route, reference))
                row["reference_repeated_arc_occurrences"] = int(len(reference) - len(np.unique(reference)))
                row["reference_zero_length_arc_occurrences"] = int(np.sum(graph.edge_lengths[reference] == 0))
                with (OUTPUT / "observations" / (tid + ".csv")).open(newline="") as stream:
                    observations = list(csv.DictReader(stream))
                matched = np.array([int(o["predicted_arc_id"]) for o in observations], dtype=np.int64)
                classifier = np.array([int(o["classifier_only_arc_id"]) for o in observations], dtype=np.int64)
                labels = reference_observation_labels(graph, track, reference)
                truth = graph.canonical_edges[labels.edges]
                matched, classifier = graph.canonical_edges[matched], graph.canonical_edges[classifier]
                row.update(classification_metrics(truth, matched))
                row.update({"classifier_" + k: v for k, v in classification_metrics(truth, classifier).items()})
                row["top1_accuracy"] = row["classifier_accuracy"]
                row["scored_observation_fraction"] = 1.0
                row["ambiguous_observation_fraction"] = float(np.mean(labels.ambiguous))
                row["metrics_status"] = "ok; candidate NLL unavailable from saved CLI"
                truths.append(truth + namespace)
                predictions.append(matched + namespace)
                classifiers.append(classifier + namespace)
            except (ValueError, OSError) as exc:
                row["unavailable_reason"] = "Reference scoring unavailable: " + type(exc).__name__ + ". No reference repair or prediction changes were made."
        verified["trips"][tid]["metrics"] = {k: row.get(k) for k in (*ROUTE_KEYS, "exact_match", "reference_discontinuity_count")}
        verified["trips"][tid]["proxy_metrics"] = {k: row.get(k) for k in PROXY_KEYS}
        verified["trips"][tid]["metrics_status"] = "ok" if row["edge_f1"] is not None and row["accuracy"] is not None else ("unavailable" if not refpath.is_file() else "error")
        rows.append(row)
        namespace += len(graph.arcs) + 1
    validate_freeze()
    current_inputs(audit, frozen)
    available = sum(row["edge_f1"] is not None for row in rows)
    proxy_summary = {"observations_scored": sum(len(v) for v in truths), "candidate_categorical_nll": None, "candidate_miss_rate": None, "unavailable_reason": NLL_REASON, "hmm_posterior_nll": None}
    if truths:
        all_truth = np.concatenate(truths)
        proxy_summary["decoded"] = classification_metrics(all_truth, np.concatenate(predictions))
        proxy_summary["classifier_only"] = classification_metrics(all_truth, np.concatenate(classifiers))
    provenance = {"model_sha256": MODEL_SHA, "config_file_sha256": CONFIG_SHA, "api_compact_config_sha256": frozen["api_compact_config_sha256"],
        "model_frozen_at_utc": frozen["model_frozen_at_utc"], "decoder_frozen_at_utc": frozen["decoder_frozen_at_utc"],
        "protocol_sha256": frozen["protocol_sha256"], "freeze_validation_sha256": sha256(REPORT / "freeze_validation.json")}
    metrics = {"summary": {"trip_count": len(rows), "reference_scored_trip_count": available,
        "macro": {k: mean_available(rows, k) for k in (*ROUTE_KEYS, *PROXY_KEYS, "candidate_miss_rate", "log_loss")},
        "macro_contributing_trip_counts": {k: sum(row.get(k) is not None for row in rows) for k in (*ROUTE_KEYS, *PROXY_KEYS, "candidate_miss_rate", "log_loss")},
        "exact_route_count": sum(row["exact_match"] is True for row in rows), "connected_prediction_count": sum(t["checks"]["nonempty_connected_directed_route"] for t in verified["trips"].values()),
        "reference_discontinuities": sum(row["reference_discontinuity_count"] or 0 for row in rows), "full_observation_proxy": proxy_summary,
        "evaluation_only": True, "official_score": False, "proxy_not_true_gps_labels": True}, "trips": rows,
        "models": {"selected_family": "hist_gradient_boosting", "selected_run": "hist_gradient_boosting_tuned"}, "freeze_provenance": provenance,
        "reference_scoring_started_at_utc": started, "completed_at_utc": utc(), "proxy_label_policy": PROXY_LABEL_POLICY,
        "overlap_report": audit["overlap_report"], "metric_definition": "Exact original directed arc multiset overlap and length-weighted occurrences; sequence edit separately. Canonical edge proxy classes are graph-namespaced across trips for pooled observation macro F1."}
    write_json(REPORT / "route_metrics.json", metrics)
    final_audit = {**audit, "trips": list(final_inputs.values()), "references_parsed": any(row["reference_parsed"] for row in final_inputs.values()), "finalized_at_utc": utc()}
    write_json(REPORT / "input_audit.json", final_audit)
    expected = {"trip_count": len(IDS), "trips": verified["trips"], "freeze_provenance": provenance, "submission_zip_sha256": verified["submission_zip_sha256"]}
    write_json(REPORT / "expected_trips.json", expected)
    checks = {**verified["checks"], "frozen_before_first_input": parsed_time(frozen["decoder_frozen_at_utc"]) < parsed_time(audit["first_input_open_started_utc"]),
        "preregistered_before_first_input": parsed_time(frozen["preregistered_at_utc"]) < parsed_time(audit["first_input_open_started_utc"]),
        "prediction_before_reference_scoring": parsed_time(verified["checked_at_utc"]) < parsed_time(started), "single_actual_cli_attempt": result["attempt_number"] == 1,
        "all_ten_route_metrics_available": available == len(IDS), "all_original_observations_scored": proxy_summary["observations_scored"] == sum(t["observations"] for t in verified["trips"].values()),
        "final_evaluation_only": True, "frozen_hashes_unchanged_at_all_phases": True, "overlap_disclosed": True, "no_reinference_for_scoring": True}
    acceptance = {"checked_at_utc": utc(), "passed": all(checks.values()) and all(all(t["checks"].values()) for t in verified["trips"].values()), "checks": checks,
        "trip_count": len(IDS), "expected_trips": verified["trips"], "freeze_provenance": provenance,
        "submission_zip_sha256": verified["submission_zip_sha256"], "submission_zip_member_count": len(IDS), "overlap_report": audit["overlap_report"],
        "chronology": {"preregistered_at_utc": frozen["preregistered_at_utc"], "first_input_open_started_utc": audit["first_input_open_started_utc"], "prediction_completed_at_utc": result["completed_at_utc"], "all_ten_verified_at_utc": verified["checked_at_utc"], "reference_scoring_started_at_utc": started},
        "unavailable_metrics": {"candidate_categorical_nll": NLL_REASON, "candidate_miss_rate": NLL_REASON, "hmm_posterior_nll": "No HMM posterior probabilities were computed or saved by the CLI."},
        "privacy": {"raw_inputs_private": True, "references_private": True, "observations_private_ignored": True, "public_reports_contain_no_coordinates_or_absolute_raw_paths": True, "zip_contains_predicted_routes_only": True},
        "report_sha256": {name: sha256(REPORT / name) for name in ("route_metrics.json", "input_audit.json", "prediction_verification.json", "expected_trips.json")}}
    write_json(REPORT / "acceptance.json", acceptance)
    require(acceptance["passed"], "final acceptance contains failed checks")
    return {"passed": True, **metrics["summary"], "submission_zip_sha256": verified["submission_zip_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "predict", "score", "all"))
    args = parser.parse_args(argv)
    stage = args.stage
    try:
        outcome = None
        for name, function in (("audit", stage_audit), ("predict", stage_predict), ("score", stage_score)):
            if args.stage in (name, "all"):
                stage = name
                outcome = function()
                print(json.dumps({"stage": name, "status": "complete", "trip_count": outcome.get("trip_count", len(outcome.get("trips", {}))), "passed": outcome.get("passed")}), flush=True)
        if args.stage in ("score", "all"):
            print(json.dumps(outcome, indent=2, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        failure = {"recorded_at_utc": utc(), "stage": stage, "state": "failed; no automatic retry or tuning", "error_type": type(exc).__name__, "model_sha256_expected": MODEL_SHA, "config_sha256_expected": CONFIG_SHA,
                   "detail": "Inspect the local exception with the evaluator owner; private error text is excluded from public JSON. Partial evidence is preserved."}
        if not (REPORT / (stage + "_failure.json")).exists():
            write_json(REPORT / (stage + "_failure.json"), failure)
        print(json.dumps(failure), file=sys.stderr)
        print(type(exc).__name__ + ": " + str(exc).replace(str(ROOT), "<workspace>"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
