#!/usr/bin/env python3
"""One frozen target96 evaluator pass; all ten predictions precede any truth IO.

This is evaluation only, not selection, retuning, deployment, or an official
organizer score. Original evaluator inputs never enter the submission archive.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import target96_runtime as runtime
from backend.emissions import AuditedEmissionModel
from backend.geometry import prediction_arrays
from mapmatch.data import discover_trips, load_graph, load_route, load_track, validate_route
from mapmatch.matcher import MatchConfig
from mapmatch.metrics import score_routes

INPUT = ROOT / "evaluator_test"
OUTPUT = ROOT / "outputs/target96/evaluator"
EXPERIMENT = ROOT / "experiments/target96/evaluator"
MANIFEST = ROOT / "artifacts/target96/release_manifest.json"
COMPARISON = ROOT / "experiments/target96/student_comparison/summary.json"
BASELINE = ROOT / "reports/final_five/evaluator/route_metrics.json"
IDS = tuple(f"{i:08d}" for i in (15, 17, 23, 30, 49, 59, 67, 73, 87, 94))
EXPECTED_OBSERVATIONS = 19674
KEYS = ("edge_precision", "edge_recall", "edge_f1", "length_precision",
        "length_recall", "length_f1", "duplicate_canonical_length_f1",
        "edit_similarity", "route_continuity")
SOURCES = ("scripts/target96_evaluate.py", "backend/target96_runtime.py",
           "backend/emissions.py", "backend/geometry.py", "mapmatch/data.py",
           "mapmatch/candidates.py", "mapmatch/routing.py", "mapmatch/metrics.py")
sha256 = runtime.sha256


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def safe_path(path, base=ROOT):
    path = Path(path)
    require(path.is_absolute() and path.is_relative_to(base), "path outside assigned scope")
    require(path.resolve().is_relative_to(base.resolve()), "resolved path outside assigned scope")
    for parent in (path, *path.parents):
        require(not parent.is_symlink(), "symlinks forbidden")
        if parent == base:
            break
    return path


def new_json(path, value):
    require(any(path.is_relative_to(base) for base in (EXPERIMENT, OUTPUT)), "write outside evaluator output scope")
    safe_path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    require(str(ROOT) not in text, "absolute private path in evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


class ReferenceBarrier:
    """Audit input reference opens, including libraries bypassing our loader."""

    def __init__(self, input_root, ids):
        self.input_root = input_root.resolve()
        self.ids = set(ids)
        self.completed = set()
        self.sealed = False
        self.reference_opens = 0
        self.first_reference_open_utc = None

    def audit(self, event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0]))
        if path.suffix == ".route" and path.resolve().is_relative_to(self.input_root):
            require(self.sealed and self.completed == self.ids, "reference opened before all ten predictions")
            self.reference_opens += 1
            self.first_reference_open_utc = self.first_reference_open_utc or utc()

    def prediction_completed(self, trip):
        require(not self.sealed and trip in self.ids and trip not in self.completed, "invalid prediction barrier transition")
        self.completed.add(trip)

    def seal(self):
        require(self.completed == self.ids and not self.sealed and self.reference_opens == 0, "all prediction barrier not satisfied")
        self.sealed = True


def release_snapshot():
    manifest, loader, factory = runtime.release_components(MANIFEST)
    accepted = read_json(MANIFEST.parent / "acceptance.json")
    require(sha256(COMPARISON) == accepted["comparison_sha256"], "accepted development comparison pin differs")
    snapshot = {
        "manifest_sha256": sha256(MANIFEST), "acceptance_sha256": sha256(MANIFEST.parent / "acceptance.json"),
        "model_sha256": manifest["model_sha256"], "config_sha256": manifest["files"]["match_config.json"],
        "arc_policy_sha256": manifest["files"]["arc_policy.json"],
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
        "release_source_sha256": manifest["source_sha256"],
    }
    return snapshot, manifest, loader, factory


def unchanged(snapshot):
    require(release_snapshot()[0] == snapshot, "accepted release or evaluation program drift")


def exact_inputs():
    safe_path(INPUT)
    prefixes = discover_trips(INPUT)
    require(len(prefixes) == 10 and {p.name for p in prefixes} == set(IDS), "evaluator trip set differs")
    for prefix in prefixes:
        require(prefix == INPUT / prefix.name / prefix.name, "evaluator paths must be id/id")
        for suffix in (".nodes", ".arcs", ".track"):
            require(safe_path(prefix.with_suffix(suffix), INPUT).is_file(), "required evaluator input missing")
    return {p.name: p for p in prefixes}


def input_hashes(prefix):
    return {suffix: sha256(prefix.with_suffix(suffix)) for suffix in (".nodes", ".arcs", ".track")}


def new_route(path, route):
    safe_path(path, OUTPUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        np.savetxt(stream, route, fmt="%d")


def submission(routes):
    destination = OUTPUT / "submission.zip"
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
        for trip in IDS:
            member = ZipInfo(trip + ".route", (1980, 1, 1, 0, 0, 0))
            member.compress_type = ZIP_DEFLATED
            member.external_attr = 0o644 << 16
            archive.writestr(member, routes[trip].read_bytes())
    with ZipFile(destination) as archive:
        require(archive.namelist() == [tid + ".route" for tid in IDS] and archive.testzip() is None, "flat ten-route ZIP invalid")
        require(all(archive.read(tid + ".route") == routes[tid].read_bytes() for tid in IDS), "ZIP route byte mismatch")
    return {"path": relative(destination), "sha256": sha256(destination), "flat_route_members": 10,
            "only_predictions": True, "byte_parity_verified": True}


def baseline_description(manifest, candidate):
    baseline = read_json(BASELINE)
    pins = baseline["freeze_provenance"]
    require(pins["model_sha256"] == manifest["baseline_model_sha256"]
            and pins["config_file_sha256"] == manifest["baseline_config_sha256"], "baseline evidence identity differs")
    require({r["trip_id"] for r in baseline["trips"]} == set(IDS)
            and baseline["summary"]["trip_count"] == 10, "baseline comparison denominator differs")
    macro = {key: baseline["summary"]["macro"][key] for key in KEYS}
    return {"path": relative(BASELINE), "sha256": sha256(BASELINE), "macro": macro,
            "delta_candidate_minus_baseline": {key: candidate[key] - macro[key] for key in KEYS},
            "interpretation": "Descriptive only; previous frozen predictions, no baseline rerun and no selection feedback.",
            "used_to_choose_settings": False}


def run():
    snapshot, manifest, loader, factory = release_snapshot()
    for path in (EXPERIMENT, OUTPUT):
        safe_path(path)
        require(not path.exists(), "refuse repeat/overwrite of evaluator run")
    config = MatchConfig(**read_json(MANIFEST.parent / "match_config.json"))
    require(asdict(config) == manifest["configuration"]["match_config"], "frozen decoder aliases differ")
    model = loader(MANIFEST.parent / "model.joblib")
    barrier = ReferenceBarrier(INPUT, IDS)
    sys.addaudithook(barrier.audit)
    plan = {"created_at_utc": utc(), "trip_ids": IDS, "expected_observations": EXPECTED_OBSERVATIONS,
            "release": snapshot, "configuration": manifest["configuration"],
            "all_predictions_before_any_reference": True, "evaluation_only": True,
            "selection_feedback_allowed": False, "deployment_changed": False,
            "official_score": False, "independent_blind_test_claimed": False,
            "timing_note": "Sequential uncached factory+match, two inference threads; graph/track parsing excluded."}
    new_json(EXPERIMENT / "run_plan.json", plan)
    prefixes = exact_inputs()
    predictions, routes, rows, graphs = {}, {}, [], {}
    for trip in IDS:
        prefix = prefixes[trip]
        hashes = input_hashes(prefix)
        graph, track = load_graph(prefix), load_track(prefix.with_suffix(".track"))
        audit = AuditedEmissionModel(model)
        start = time.perf_counter()
        try:
            with threadpool_limits(limits=2):
                result = factory(graph, config, emission_model=audit).match(track)
            elapsed = time.perf_counter() - start
            route, indices, edges, fractions, distances = prediction_arrays(graph, track, result)
            require(np.array_equal(indices, np.arange(len(track))), "prediction is not dense all-GPS")
            require(validate_route(graph, route)["valid"] and not result.breaks, "predicted route not connected")
            require(audit.call_count > 0 and set(audit.records) == set(range(len(track))), "learned score audit incomplete")
            require(result.diagnostics.get("uses_reference_route") is False, "prediction reports reference access")
            require(input_hashes(prefix) == hashes, "input changed during prediction")
            path = OUTPUT / "routes" / (trip + ".route")
            new_route(path, route)
            diagnostic_path = OUTPUT / "diagnostics" / (trip + ".npz")
            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            with diagnostic_path.open("xb") as stream:
                np.savez_compressed(stream, observation_indices=indices, matched_edges=edges,
                                    matched_fractions=fractions, matched_distances=distances)
            diagnostics = {k: v for k, v in result.diagnostics.items() if k != "classifier_only_edges"}
            record = {"trip_id": trip, "success": True, "observations": len(track),
                      "elapsed_seconds": elapsed, "route_arc_count": len(route),
                      "route_path": relative(path), "route_sha256": sha256(path),
                      "diagnostics_path": relative(diagnostic_path), "diagnostics_sha256": sha256(diagnostic_path),
                      "input_sha256": hashes, "reference_opened": False, "learned_emission_observations": len(audit.records),
                      "dense_original_arc_ids_connected": True, "diagnostics": diagnostics}
            new_json(EXPERIMENT / "predictions" / (trip + ".json"), record)
            predictions[trip] = record
            routes[trip] = path
            graphs[trip] = graph
            barrier.prediction_completed(trip)
            print(json.dumps({"event": "target96_evaluator_prediction", "trip_id": trip,
                              "observations": len(track), "elapsed_seconds": elapsed}), flush=True)
        except Exception as exc:
            new_json(EXPERIMENT / "failure.json", {"trip_id": trip, "exception_type": type(exc).__name__,
                     "predicted_trip_count": len(predictions), "references_opened": barrier.reference_opens,
                     "evaluation_complete": False, "no_retuning_permitted": True})
            raise
        finally:
            del audit
    total = sum(row["observations"] for row in predictions.values())
    require(total == EXPECTED_OBSERVATIONS, "evaluator all-GPS denominator differs")
    unchanged(snapshot)
    zip_evidence = submission(routes)
    completed = {"completed_at_utc": utc(), "trip_count": 10, "observations": total,
                 "references_opened": barrier.reference_opens, "all_dense_connected_learned": True,
                 "route_sha256": {tid: predictions[tid]["route_sha256"] for tid in IDS}, "submission": zip_evidence}
    new_json(EXPERIMENT / "all_predictions_complete.json", completed)
    barrier.seal()
    scoring_start = utc()
    for trip in IDS:
        require(sha256(routes[trip]) == predictions[trip]["route_sha256"], "prediction changed before reference scoring")
        reference_path = safe_path(prefixes[trip].with_suffix(".route"), INPUT)
        reference = load_route(reference_path)
        prediction = load_route(routes[trip])
        metrics = score_routes(graphs[trip], prediction, reference)
        require(all(np.isfinite(metrics[key]) for key in KEYS), "nonfinite route score")
        row = {k: v for k, v in predictions[trip].items() if k != "diagnostics"}
        row.update({"reference_loaded_after_all_predictions": True, "metrics": metrics,
                    "reference_sha256": sha256(reference_path)})
        new_json(EXPERIMENT / "trips" / (trip + ".json"), row)
        rows.append(row)
    macro = {key: float(np.mean([row["metrics"][key] for row in rows])) for key in KEYS}
    unchanged(snapshot)
    for trip in IDS:
        require(input_hashes(prefixes[trip]) == predictions[trip]["input_sha256"]
                and sha256(routes[trip]) == predictions[trip]["route_sha256"], "input/prediction drift after scoring")
    summary = {"schema": "pathsnap-target96-evaluator-v1", "completed_at_utc": utc(),
               "evaluation_only": True, "official_score": False, "selection_feedback_used": False,
               "independent_blind_test_claimed": False, "model_configuration_unchanged": True,
               "trip_count": 10, "observations": total, "failed_trips": 0, "macro": macro,
               "connected_predictions": 10, "exact_route_count": sum(r["metrics"]["exact_match"] for r in rows),
               "reference_discontinuities": sum(r["metrics"]["reference_discontinuity_count"] for r in rows),
               "mean_seconds": float(np.mean([r["elapsed_seconds"] for r in rows])),
               "total_seconds": float(sum(r["elapsed_seconds"] for r in rows)),
               "release": snapshot, "configuration_name": manifest["configuration"]["name"],
               "all_predictions_completed_at_utc": completed["completed_at_utc"],
               "reference_scoring_started_at_utc": scoring_start,
               "first_reference_open_utc": barrier.first_reference_open_utc,
               "no_reference_before_all_predictions": True, "reference_open_audit_events": barrier.reference_opens,
               "submission": zip_evidence, "baseline": baseline_description(manifest, macro),
               "trips": [{"trip_id": row["trip_id"], "observations": row["observations"],
                          "elapsed_seconds": row["elapsed_seconds"], "metrics": row["metrics"],
                          "route_sha256": row["route_sha256"],
                          "record_path": relative(EXPERIMENT / "trips" / (row["trip_id"] + ".json")),
                          "record_sha256": sha256(EXPERIMENT / "trips" / (row["trip_id"] + ".json"))} for row in rows]}
    new_json(EXPERIMENT / "summary.json", summary)
    print(json.dumps({"event": "target96_evaluator_complete", "trips": 10, "observations": total,
                      "macro": macro, "mean_seconds": summary["mean_seconds"],
                      "submission": zip_evidence, "summary": relative(EXPERIMENT / "summary.json")}), flush=True)
    return summary


class FixtureUnitTests(unittest.TestCase):
    """Synthetic unit checks only; never load datasets/models or perform inference."""

    def test_reference_barrier_blocks_early_truth_and_only_seals_complete_set(self):
        with tempfile.TemporaryDirectory(prefix="target96-eval-unit-") as directory:
            root = Path(directory)
            gate = ReferenceBarrier(root, ("a", "b"))
            event = (str(root / "a.route"), "r", 0)
            with self.assertRaises(ValueError):
                gate.audit("open", event)
            gate.prediction_completed("a")
            with self.assertRaises(ValueError):
                gate.seal()
            gate.prediction_completed("b")
            gate.seal()
            gate.audit("open", event)
            self.assertEqual(gate.reference_opens, 1)
            with self.assertRaises(ValueError):
                gate.prediction_completed("a")

    def test_barrier_does_not_read_or_block_graph_track_or_predicted_route(self):
        gate = ReferenceBarrier(Path("/unit-input"), ("a",))
        for path in ("/unit-input/a.track", "/unit-input/a.nodes", "/unit-output/a.route"):
            gate.audit("open", (path, "r", 0))
        self.assertEqual(gate.reference_opens, 0)

    def test_route_metrics_keep_original_ids_repeats_and_continuity(self):
        from mapmatch.data import Graph
        graph = Graph(np.array([[0., 0.], [.001, 0.], [.002, 0.]]), np.array([[0, 1], [1, 2], [1, 0]]))
        self.assertTrue(validate_route(graph, [0, 2, 0, 1])["valid"])
        score = score_routes(graph, [0, 2, 0, 1], [0, 1])
        self.assertAlmostEqual(score["edge_precision"], .5)
        self.assertEqual(score["route_continuity"], 1.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run synthetic UNIT tests only")
    parser.add_argument("--run", action="store_true", help="perform the single authorized frozen evaluator pass")
    args = parser.parse_args()
    require(args.self_test != args.run, "choose exactly one of --self-test or --run")
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(FixtureUnitTests)
        outcome = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if outcome.wasSuccessful() else 1)
    run()


if __name__ == "__main__":
    main()
