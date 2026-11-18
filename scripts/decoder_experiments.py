#!/usr/bin/env python3
"""TRAIN-only decoder calibration and full-observation student-development ablation.

This script does not train a classifier or change matcher implementation. Reference
routes are used for calibration targets or AFTER predictions, never for inference.
Only reports/decoder/ and artifacts/match_config.json are written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

# Set before importing numerical packages; every experiment is single-threaded.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import numpy as np

from mapmatch.data import (discover_trips, load_graph, load_route, load_track,
                           validate_route)
from mapmatch.features import reference_observation_labels
from mapmatch.matcher import MapMatcher, MatchConfig
from mapmatch.metrics import score_routes

REPORT = PROJECT / "reports" / "decoder"
CONFIG_PATH = PROJECT / "artifacts" / "match_config.json"
CALIBRATION_IDS = tuple(f"{i:08d}" for i in range(1, 72, 10))
SIGMA_BOUNDS = (3.0, 50.0)
BETA_BOUNDS = (2.0, 100.0)
SOURCES = ("scripts/decoder_experiments.py", "mapmatch/data.py",
           "mapmatch/candidates.py", "mapmatch/matcher.py",
           "mapmatch/routing.py", "mapmatch/features.py",
           "mapmatch/learning.py", "mapmatch/metrics.py",
           "backend/emissions.py", "backend/scoring.py")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_hashes():
    return {p: sha256(PROJECT / p) for p in SOURCES}


def write_json(path, value):
    path = Path(path)
    if path != CONFIG_PATH and not path.resolve().is_relative_to(REPORT.resolve()):
        raise ValueError("Experiment writes must stay in reports/decoder/")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    path = Path(path)
    if not path.resolve().is_relative_to(REPORT.resolve()):
        raise ValueError("CSV must stay in reports/decoder/")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def progress(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def summary(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "mean": float(values.mean()),
            "median": float(np.median(values)),
            **{f"p{q:g}": float(np.percentile(values, q)) for q in (5, 25, 75, 90, 95, 99)},
            "max": float(values.max())}


def sample_indices(n, count, start=0):
    if n <= start:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.linspace(start, n - 1, min(count, n - start), dtype=np.int64))


def plan():
    return {
        "declared_utc": utcnow(), "version": 1,
        "training_root": "Student folder/train", "train_trip_count": 80,
        "scale_sampling": "256 evenly spaced original observations/trip; 128 evenly spaced adjacent original-observation pairs/trip; deterministic linspace integer indices",
        "sigma_estimator": "median absolute perpendicular residual to the proxy-labelled reference arc supporting line / 0.6744897501960817; zero-length arc uses exact point residual; all sampled observations enter, no outlier removal",
        "sigma_stability_bounds_m": list(SIGMA_BOUNDS),
        "beta_estimator": "median absolute (forward contiguous reference-route projected distance - observed chord) / ln(2); minimum nonnegative forward occurrence span; same-edge backward <=20m treated as jitter; unidentified reference spans reported, no speed filter",
        "beta_stability_bounds_m": list(BETA_BOUNDS),
        "calibration_trip_ids": list(CALIBRATION_IDS),
        "calibration_rule": "first eight sorted TRAIN IDs whose numeric ID modulo 10 is 1",
        "objective": "mean exact-original-ID length-weighted repeated-arc route F1 across the eight TRAIN calibration trips; a failed trip contributes zero",
        "selection_tie_break": "predeclared order A, B, C for exactly equal objectives",
        "configurations": [
            {"name": "A_fitted", "beta_multiplier": 1.0, "classifier_weight": 1.0, "geometry_weight": 1.0},
            {"name": "B_twice_beta", "beta_multiplier": 2.0, "classifier_weight": 1.0, "geometry_weight": 1.0},
            {"name": "C_classifier_up", "beta_multiplier": 1.0, "classifier_weight": 2.0, "geometry_weight": 0.5},
        ],
        "freeze": "Write final plain MatchConfig artifact before student-development ablation; never revise from those results",
        "limitations": ["Calibration uses route-derived proxy alignment, not timestamp-aligned truth.",
                         "Final classifier was fitted on all TRAIN80, so decoder calibration TRAIN8 is in-sample and may be optimistic.",
                         "This is robust geometric scale fitting plus a small supervised decoder selection, not Baum-Welch or full HMM parameter learning.",
                         "Student-development10 is not the absent reserved evaluator10."],
    }


def ensure_plan():
    path = REPORT / "calibration_plan.json"
    if path.exists():
        existing = read_json(path)
        if existing["calibration_trip_ids"] != list(CALIBRATION_IDS):
            raise ValueError("Existing calibration plan differs")
        return existing
    value = plan()
    write_json(path, value)
    return value


def project_selected(graph, xy, edges):
    starts = graph.xy[graph.tails[edges]]
    vectors = graph.xy[graph.heads[edges]] - starts
    length2 = np.einsum("ij,ij->i", vectors, vectors)
    offset = xy - starts
    fraction = np.clip(np.divide(np.einsum("ij,ij->i", offset, vectors), length2,
                                 out=np.zeros(len(edges)), where=length2 > 1e-12), 0, 1)
    exact = np.linalg.norm(starts + fraction[:, None] * vectors - xy, axis=1)
    cross = vectors[:, 0] * offset[:, 1] - vectors[:, 1] * offset[:, 0]
    perpendicular = np.divide(np.abs(cross), np.sqrt(length2), out=exact.copy(), where=length2 > 1e-12)
    return fraction, exact, perpendicular, length2 <= 1e-12


def reference_span_distance(graph, route, occurrence, prefix, break_prefix, e1, f1, e2, f2):
    """No prediction or graph routing: only labelled reference occurrence spans."""
    best = math.inf
    forward_seen = connected_seen = False
    for left in occurrence.get(int(e1), ()):
        for right in occurrence.get(int(e2), ()):
            if right < left:
                continue
            forward_seen = True
            if break_prefix[right] != break_prefix[left]:
                continue
            connected_seen = True
            distance = float(prefix[right] + f2 * graph.edge_lengths[e2]
                             - prefix[left] - f1 * graph.edge_lengths[e1])
            if right == left and distance >= -20.0:
                best = min(best, max(distance, 0.0))
            elif distance >= -1e-8:
                best = min(best, max(distance, 0.0))
    if math.isfinite(best):
        return best, "identified"
    if not forward_seen:
        return None, "no_forward_reference_occurrence"
    if not connected_seen:
        return None, "reference_span_crosses_discontinuity"
    return None, "backward_proxy_without_reference_loop"


def calibrate(args):
    declaration = ensure_plan()
    if (REPORT / "scale_fit.json").exists():
        raise ValueError("scale_fit.json exists; retain the original experiment rather than overwrite it")
    prefixes = discover_trips(args.train_root)
    if len(prefixes) != 80:
        raise ValueError("Scale fitting requires all 80 TRAIN trips")
    ids = [p.name for p in prefixes if int(p.name) % 10 == 1][:8]
    if ids != list(CALIBRATION_IDS):
        raise ValueError("TRAIN calibration IDs differ from declaration")
    residual_rows, pair_rows, trips = [], [], []
    perpendicular_all, exact_all, mismatch_all = [], [], []
    start = time.perf_counter()
    for prefix in prefixes:
        graph = load_graph(prefix)
        track = load_track(prefix.with_suffix(".track"))
        route = load_route(prefix.with_suffix(".route"))
        route_validation = validate_route(graph, route)
        if route_validation["invalid_positions"] or not len(route):
            raise ValueError("Cannot calibrate with invalid reference arc IDs")
        route = graph.canonical_edges[route]
        sigma_indices = sample_indices(len(track), 256)
        second = sample_indices(len(track), 128, 1)
        all_indices = np.unique(np.concatenate((sigma_indices, second, second - 1)))
        labels = reference_observation_labels(graph, track, route, all_indices)
        xy = graph.project(track[:, :2])
        fractions, distances, perpendicular, degenerate = project_selected(graph, xy[all_indices], labels.edges)
        lookup = {int(i): k for k, i in enumerate(all_indices)}
        local_perp, local_exact = [], []
        for i in sigma_indices:
            k = lookup[int(i)]
            row = {"trip_id": prefix.name, "observation_index": int(i),
                   "perpendicular_residual_m": float(perpendicular[k]),
                   "exact_segment_residual_m": float(distances[k]),
                   "proxy_ambiguous": bool(labels.ambiguous[k]),
                   "degenerate_segment_fallback": bool(degenerate[k])}
            residual_rows.append(row)
            local_perp.append(row["perpendicular_residual_m"])
            local_exact.append(row["exact_segment_residual_m"])
        occurrence = defaultdict(list)
        for i, edge in enumerate(route):
            occurrence[int(edge)].append(i)
        prefix_length = np.concatenate(([0.0], np.cumsum(graph.edge_lengths[route])))
        broken = graph.heads[route[:-1]] != graph.tails[route[1:]]
        break_prefix = np.concatenate(([0], np.cumsum(broken)))
        local_mismatch, reasons = [], Counter()
        for i in second:
            before, after = lookup[int(i - 1)], lookup[int(i)]
            distance, reason = reference_span_distance(graph, route, occurrence, prefix_length, break_prefix,
                                                       int(labels.edges[before]), float(fractions[before]),
                                                       int(labels.edges[after]), float(fractions[after]))
            chord = float(np.linalg.norm(xy[i] - xy[i - 1]))
            mismatch = abs(distance - chord) if distance is not None else None
            reasons[reason] += 1
            if mismatch is not None:
                local_mismatch.append(mismatch)
            pair_rows.append({"trip_id": prefix.name, "previous_observation_index": int(i - 1),
                              "observation_index": int(i), "gap_s": float(track[i, 2] - track[i - 1, 2]),
                              "gps_chord_m": chord, "reference_along_m": distance,
                              "absolute_mismatch_m": mismatch, "status": reason})
        perpendicular_all.extend(local_perp)
        exact_all.extend(local_exact)
        mismatch_all.extend(local_mismatch)
        row = {"trip_id": prefix.name, "observation_count": len(track),
               "sigma_sample_count": len(sigma_indices), "pair_sample_count": len(second),
               "pair_status_counts": dict(reasons),
               "perpendicular_residual_m": summary(local_perp),
               "exact_segment_residual_m": summary(local_exact),
               "route_chord_absolute_mismatch_m": summary(local_mismatch),
               "reference_discontinuities": len(route_validation["discontinuities"]),
               "source_sha256": {s: sha256(prefix.with_suffix(s)) for s in (".nodes", ".arcs", ".track", ".route")}}
        trips.append(row)
        progress({"stage": "scale_fit", "trip_id": prefix.name, "sigma_median_m": row["perpendicular_residual_m"]["median"],
                  "identified_beta_pairs": len(local_mismatch), "unidentified_beta_pairs": len(second) - len(local_mismatch)})
    sigma_raw = float(np.median(perpendicular_all) / 0.6744897501960817)
    beta_raw = float(np.median(mismatch_all) / math.log(2.0))
    fitted = {"sigma_m": float(np.clip(sigma_raw, *SIGMA_BOUNDS)),
              "beta_m": float(np.clip(beta_raw, *BETA_BOUNDS))}
    value = {"completed_utc": utcnow(), "plan_sha256": canonical_hash(declaration),
             "elapsed_seconds": time.perf_counter() - start, "training_trip_count": len(trips),
             "sigma_unclamped_m": sigma_raw, "beta_unclamped_m": beta_raw,
             "sigma_clamped": not SIGMA_BOUNDS[0] <= sigma_raw <= SIGMA_BOUNDS[1],
             "beta_clamped": not BETA_BOUNDS[0] <= beta_raw <= BETA_BOUNDS[1],
             "fitted": fitted, "perpendicular_residual_m": summary(perpendicular_all),
             "exact_segment_residual_m": summary(exact_all),
             "route_chord_absolute_mismatch_m": summary(mismatch_all),
             "sigma_observations_excluded": 0,
             "beta_pair_status_counts": dict(Counter(r["status"] for r in pair_rows)),
             "reference_discontinuity_count": sum(t["reference_discontinuities"] for t in trips),
             "source_sha256": source_hashes(), "trips": trips}
    write_csv(REPORT / "training_residual_samples.csv", residual_rows)
    write_csv(REPORT / "training_transition_samples.csv", pair_rows)
    write_json(REPORT / "scale_fit.json", value)
    progress({"stage": "scale_fit_complete", **fitted, "sigma_unclamped_m": sigma_raw, "beta_unclamped_m": beta_raw})


def overlap_only(graph, prediction, reference):
    p, r = Counter(map(int, prediction)), Counter(map(int, reference))
    common = p & r
    common_count = sum(common.values())
    common_length = sum(float(graph.edge_lengths[e]) * n for e, n in common.items())
    plen, rlen = float(graph.edge_lengths[prediction].sum()), float(graph.edge_lengths[reference].sum())
    return {"edge_f1": 2 * common_count / max(len(prediction) + len(reference), 1),
            "length_f1": 2 * common_length / (plen + rlen) if plen + rlen else 0.0,
            "predicted_arc_count": len(prediction), "reference_arc_count": len(reference),
            "predicted_length_m": plen, "reference_length_m": rlen}


def search(args):
    declaration = ensure_plan()
    scale = read_json(REPORT / "scale_fit.json")
    if CONFIG_PATH.exists() or (REPORT / "freeze.json").exists() or (REPORT / "search_results.jsonl").exists():
        raise ValueError("Config/search already exists; preserve frozen experiment")
    from mapmatch.learning import load_emission_model
    model_path = Path(args.model)
    model_hash = sha256(model_path)
    model = load_emission_model(model_path)
    configs = []
    for item in declaration["configurations"]:
        config = MatchConfig(sigma_m=scale["fitted"]["sigma_m"],
                             beta_m=float(np.clip(scale["fitted"]["beta_m"] * item["beta_multiplier"], *BETA_BOUNDS)),
                             classifier_weight=item["classifier_weight"], geometry_weight=item["geometry_weight"])
        configs.append((item["name"], config))
    write_json(REPORT / "search_configs.json", {"created_utc": utcnow(), "model_sha256": model_hash,
                                                "plan_sha256": canonical_hash(declaration),
                                                "configs": {n: asdict(c) for n, c in configs},
                                                "source_sha256": source_hashes()})
    by_id = {p.name: p for p in discover_trips(args.train_root)}
    records = []
    start = time.perf_counter()
    with (REPORT / "search_results.jsonl").open("w", encoding="utf-8") as log:
        for trip_id in CALIBRATION_IDS:
            prefix = by_id[trip_id]
            graph = load_graph(prefix)
            track = load_track(prefix.with_suffix(".track"))
            # Calibration labels are not supplied to MapMatcher.
            for name, config in configs:
                trial_start = time.perf_counter()
                row = {"configuration": name, "trip_id": trip_id, "observations": len(track)}
                try:
                    result = MapMatcher(graph, config, emission_model=model).match(track)
                    if not np.array_equal(result.observation_indices, np.arange(len(track))):
                        raise AssertionError("Calibration silently dropped observations")
                    reference = load_route(prefix.with_suffix(".route"))
                    row.update(overlap_only(graph, result.route, reference))
                    row.update({"status": "ok", "effective_max_candidates": result.diagnostics["effective_max_candidates"],
                                "candidate_retry_attempts": result.diagnostics["candidate_retry_attempts"],
                                "connected": validate_route(graph, result.route)["valid"],
                                "route_array_sha256": hashlib.sha256(np.asarray(result.route, dtype="<i8").tobytes()).hexdigest()})
                except Exception as exc:
                    row.update({"status": "failed", "length_f1": 0.0, "edge_f1": 0.0,
                                "error_type": type(exc).__name__, "error": str(exc),
                                "traceback": traceback.format_exc()})
                row["elapsed_seconds"] = time.perf_counter() - trial_start
                records.append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                progress({"stage": "search", **row})
    if sha256(model_path) != model_hash:
        raise RuntimeError("Learned model changed during decoder selection; no config frozen")
    objectives = []
    for name, config in configs:
        rows = [r for r in records if r["configuration"] == name]
        objectives.append({"configuration": name, "mean_length_f1": float(np.mean([r["length_f1"] for r in rows])),
                           "mean_edge_f1": float(np.mean([r["edge_f1"] for r in rows])),
                           "successful_trips": sum(r["status"] == "ok" for r in rows),
                           "total_trips": len(rows), "total_seconds": sum(r["elapsed_seconds"] for r in rows)})
    winner = max(range(len(objectives)), key=lambda k: objectives[k]["mean_length_f1"])
    name, config = configs[winner]
    write_json(CONFIG_PATH, asdict(config))
    frozen = {"frozen_utc": utcnow(), "selected_configuration": name,
              "config": asdict(config), "config_file_sha256": sha256(CONFIG_PATH),
              "effective_config_sha256": canonical_hash(asdict(config)),
              "model_sha256": model_hash, "model_path": str(model_path.resolve().relative_to(PROJECT)),
              "model_metadata": {k: model.metadata.get(k) for k in ("model_name", "run_name", "family", "parameters", "feature_version", "training_trip_count", "training_observations", "training_candidate_rows", "fit_data")},
              "plan_sha256": canonical_hash(declaration),
              "calibration_trip_ids": list(CALIBRATION_IDS), "student_results_seen_for_selection": False,
              "student_results_used_for_selection": False,
              "prior_geometry_student_baseline_available": True,
              "selection_boundary_note": "Earlier geometry-only student baseline was available as context; this predeclared search reads only TRAIN predictions/references and never reads student scores. Learned student ablation starts after this freeze.",
              "objective": declaration["objective"], "objectives": objectives,
              "elapsed_seconds": time.perf_counter() - start, "source_sha256": source_hashes()}
    write_csv(REPORT / "search_objectives.csv", objectives)
    write_json(REPORT / "freeze.json", frozen)
    progress({"stage": "CONFIG_FROZEN", "selected_configuration": name,
              "config_file_sha256": frozen["config_file_sha256"],
              "effective_config_sha256": frozen["effective_config_sha256"], "config": asdict(config),
              "objectives": objectives})


def standalone_classifier(graph, track, model, effective_config, audit):
    """Time fresh generation/scoring on the same final effective candidate width."""
    from mapmatch.candidates import CandidateIndex
    start = time.perf_counter()
    index = CandidateIndex(graph, effective_config.sample_spacing_m)
    xy = graph.project(track[:, :2])
    candidates = []
    for point in xy:
        radius = effective_config.candidate_radius_m
        while True:
            c = index.query(point, radius, effective_config.max_candidates)
            if len(c):
                candidates.append(c)
                break
            if radius >= effective_config.max_radius_m:
                raise AssertionError("Candidate generation disagreed with HMM")
            radius = min(radius * 2, effective_config.max_radius_m)
    probability = model.score_candidates_batch(graph, track, np.arange(len(track)), candidates)
    prediction = []
    for i, (c, p) in enumerate(zip(candidates, probability)):
        record = audit.records[i]
        if not np.array_equal(c.edges, record.edges) or not np.allclose(p, record.probabilities, rtol=1e-12, atol=1e-15):
            raise AssertionError("Standalone classifier did not use final HMM candidate probabilities")
        prediction.append(int(c.edges[np.argmax(p)]))
    prediction = np.asarray(prediction, dtype=np.int64)
    adjacent_collapse(prediction)
    return prediction, time.perf_counter() - start


def adjacent_collapse(edges):
    edges = np.asarray(edges, dtype=np.int64)
    return edges[np.concatenate(([True], edges[1:] != edges[:-1]))] if len(edges) else edges


def metric_row(graph, trip_id, method, track, prediction, route, reference, truth, seconds, extra=None):
    from backend.scoring import classification_metrics
    metrics = score_routes(graph, route, reference)
    classification = classification_metrics(graph.canonical_edges[truth], graph.canonical_edges[prediction])
    validation = validate_route(graph, route)
    row = {"trip_id": trip_id, "method": method, "status": "ok", "observations": len(track),
           **{f"proxy_{k}": v for k, v in classification.items()},
           "route_edge_precision": metrics["edge_precision"], "route_edge_recall": metrics["edge_recall"],
           "route_edge_f1": metrics["edge_f1"], "route_length_precision": metrics["length_precision"],
           "route_length_recall": metrics["length_recall"], "route_length_f1": metrics["length_f1"],
           "route_edit_distance": metrics["edit_distance"], "route_edit_similarity": metrics["edit_similarity"],
           "route_continuity": metrics["route_continuity"], "route_discontinuities": len(validation["discontinuities"]),
           "prediction_connected": validation["valid"],
           "reference_discontinuities": metrics["reference_discontinuity_count"],
           "predicted_arc_count": metrics["predicted_arc_count"], "reference_arc_count": metrics["reference_arc_count"],
           "common_arc_count": metrics["common_arc_count"],
           "predicted_length_m": metrics["predicted_length_m"], "reference_length_m": metrics["reference_length_m"],
           "common_length_m": metrics["common_length_m"],
           "inference_seconds": seconds, "posterior_log_loss": None,
           "route_file_sha256": hashlib.sha256("".join(f"{int(e)}\n" for e in route).encode()).hexdigest(),
           "matched_edge_array_sha256": hashlib.sha256(np.asarray(prediction, dtype="<i8").tobytes()).hexdigest()}
    if extra:
        row.update(extra)
    return row


def aggregate(rows, pooled, attempted_trip_count):
    from backend.scoring import classification_metrics
    result = {}
    for method in dict.fromkeys(r["method"] for r in rows):
        rr = [r for r in rows if r["method"] == method and r["status"] == "ok"]
        keys = [k for k, v in rr[0].items() if isinstance(v, (float, int)) and not isinstance(v, bool)] if rr else []
        means = {k: float(np.mean([r[k] for r in rr])) for k in keys}
        obs = sum(r["observations"] for r in rr)
        joins = sum(max(r["predicted_arc_count"] - 1, 0) for r in rr)
        micro = {"observations": obs, "successful_trips": len(rr),
                 "connected_trips": sum(r["prediction_connected"] for r in rr),
                 "route_edge_f1": 2 * sum(r["common_arc_count"] for r in rr) / max(sum(r["predicted_arc_count"] + r["reference_arc_count"] for r in rr), 1),
                 "route_length_f1": 2 * sum(r["common_length_m"] for r in rr) / max(sum(r["predicted_length_m"] + r["reference_length_m"] for r in rr), 1e-15),
                 "route_continuity": 1 - sum(r["route_discontinuities"] for r in rr) / joins if joins else 1.0,
                 "route_discontinuities": sum(r["route_discontinuities"] for r in rr),
                 "route_edit_similarity": 1 - sum(r["route_edit_distance"] for r in rr) / max(sum(max(r["predicted_arc_count"], r["reference_arc_count"], 1) for r in rr), 1),
                 "total_inference_seconds": sum(r["inference_seconds"] for r in rr),
                 "scored_observation_fraction": 1.0 if obs else None,
                 "candidate_miss_count": sum(r.get("candidate_miss_count", 0) for r in rr),
                 "proxy_ambiguous_count": sum(r.get("proxy_ambiguous_count", 0) for r in rr)}
        if method in pooled and pooled[method]:
            truth = np.concatenate([t for t, p in pooled[method]])
            pred = np.concatenate([p for t, p in pooled[method]])
            micro.update({f"proxy_{k}": v for k, v in classification_metrics(truth, pred).items()})
        for key in ("candidate_miss_rate", "categorical_emission_log_loss", "classifier_top1_accuracy"):
            available = [r for r in rr if isinstance(r.get(key), (float, int))]
            micro[key] = sum(r[key] * r["observations"] for r in available) / sum(r["observations"] for r in available) if available else None
        failure_count = attempted_trip_count - len(rr)
        zero_inclusive = {k: sum(float(r[k]) for r in rr) / attempted_trip_count
                          for k in ("proxy_accuracy", "proxy_precision", "proxy_recall", "proxy_macro_f1",
                                    "route_edge_f1", "route_length_f1", "route_edit_similarity", "route_continuity")}
        result[method] = {"attempted_trip_count": attempted_trip_count, "successful_trip_count": len(rr),
                          "failed_or_unavailable_trip_count": failure_count,
                          "all_ten_headline_eligible": attempted_trip_count == 10 and failure_count == 0,
                          "trip_macro_denominator": len(rr),
                          "trip_macro": means, "trip_macro_failures_as_zero": zero_inclusive, "pooled": micro}
    return result


def baseline_row(prefix, graph, track, reference, truth, output):
    trip_id = prefix.name
    run = read_json(output / "run_config.json")
    if not run.get("baseline") or run.get("model_sha256") is not None:
        raise ValueError("Not a geometry-only baseline artifact")
    diagnostics = read_json(output / "diagnostics" / (trip_id + ".json"))
    effective = MatchConfig(**diagnostics["config"])
    route_path = output / "routes" / (trip_id + ".route")
    route = load_route(route_path)
    with (output / "index.csv").open(newline="", encoding="utf-8") as stream:
        idx = {r["trip_id"]: r for r in csv.DictReader(stream)}[trip_id]
    if sha256(route_path) != idx["sha256"]:
        raise ValueError("Baseline route checksum differs from original index")
    with (output / "observations" / (trip_id + ".csv")).open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    indices = np.array([int(r["observation_index"]) for r in records])
    if not np.array_equal(indices, np.arange(len(track))):
        raise AssertionError("Baseline was not all-observation")
    predicted = np.array([int(r["predicted_arc_id"]) for r in records])
    # Reconstruct only candidate sets/probabilities, not predictions or routes.
    matcher = MapMatcher(graph, effective)
    xy = graph.project(track[:, :2])
    probability, missed = [], []
    for i, label in enumerate(graph.canonical_edges[truth]):
        candidates, _ = matcher._candidates(xy[i])
        costs = 0.5 * effective.geometry_weight * (candidates.distances / effective.sigma_m) ** 2
        mass = np.exp(-(costs - costs.min()))
        mass /= mass.sum()
        found = graph.canonical_edges[candidates.edges] == label
        probability.append(max(float(mass[found].sum()), 1e-7))
        missed.append(not bool(found.any()))
    row = metric_row(graph, trip_id, "geometry_HMM_default_reused", track, predicted, route, reference, truth, float(idx["elapsed_s"]),
                     {"candidate_miss_rate": float(np.mean(missed)), "candidate_miss_count": int(np.sum(missed)),
                      "scored_observation_fraction": 1.0,
                      "categorical_emission_log_loss": float(-np.mean(np.log(probability))),
                      "classifier_top1_accuracy": None, "effective_max_candidates": effective.max_candidates,
                      "baseline_route_sha256": idx["sha256"],
                      "baseline_diagnostics_sha256": sha256(output / "diagnostics" / (trip_id + ".json")),
                      "baseline_observations_sha256": sha256(output / "observations" / (trip_id + ".csv")),
                      "inference_timing_note": "Reused original CLI elapsed_s includes graph load; not controlled against fresh learned timings"})
    return row, predicted


def ablate(args):
    from backend.emissions import AuditedEmissionModel
    from backend.scoring import score_prediction
    from mapmatch.learning import load_emission_model
    frozen = read_json(REPORT / "freeze.json")
    config = MatchConfig(**read_json(CONFIG_PATH))
    if sha256(CONFIG_PATH) != frozen["config_file_sha256"] or sha256(args.model) != frozen["model_sha256"]:
        raise ValueError("Frozen model/config changed")
    if (REPORT / "student_ablation.json").exists() or (REPORT / "student_ablation.jsonl").exists():
        raise ValueError("Student ablation already exists; do not overwrite reported evidence")
    model = load_emission_model(args.model)
    prefixes = discover_trips(args.student_root)
    if len(prefixes) != 10:
        raise ValueError("Ablation requires all ten student-development trips")
    expected_methods = ["classifier_only_raw", "classifier_plus_HMM"]
    if args.baseline_output:
        expected_methods.append("geometry_HMM_default_reused")
    write_json(REPORT / "ablation_run.json", {"started_utc": utcnow(),
               "student_trip_ids": [p.name for p in prefixes],
               "freeze_file_sha256": sha256(REPORT / "freeze.json"),
               "model_sha256": frozen["model_sha256"], "config_sha256": frozen["config_file_sha256"],
               "config": asdict(config), "source_sha256": source_hashes(),
               "baseline_output": str(Path(args.baseline_output).resolve().relative_to(PROJECT)) if args.baseline_output else None,
               "baseline_provenance_sha256": {n: sha256(Path(args.baseline_output) / n) for n in ("run_config.json", "index.csv")} if args.baseline_output else None,
               "pointwise_definition": "Canonical directed arc labels from reference geometry-and-motion proxy; all original observations, none excluded",
               "classifier_only_sequence": "Per-observation argmax of the exact final effective HMM candidates/probabilities; collapse adjacent identical IDs only; no connectors inserted",
               "log_loss_definition": "Categorical NLL of normalized pre-HMM classifier emissions including candidate misses at1e-7; same emission NLL for classifier and HMM, NOT HMM posterior NLL or binary candidate log loss; geometry baseline uses normalized geometric emissions at same1e-7floor",
               "pooled_macro_f1": "Macro over union of (trip_id, canonical directed arc ID) truth/prediction labels; graph-local edge numbers are never conflated",
               "inference_uses_reference": False,
               "reserved_evaluator_data_available": False,
               "student_results_used_to_revise_config": False})
    rows, pooled = [], defaultdict(list)
    offset = 0
    start = time.perf_counter()
    with (REPORT / "student_ablation.jsonl").open("w", encoding="utf-8") as log:
        for prefix in prefixes:
            trip_start = time.perf_counter()
            try:
                graph = load_graph(prefix)
                track = load_track(prefix.with_suffix(".track"))
                audited = AuditedEmissionModel(model)
                decode_start = time.perf_counter()
                result = MapMatcher(graph, config, emission_model=audited).match(track)
                decode_seconds = time.perf_counter() - decode_start
                if not np.array_equal(result.observation_indices, np.arange(len(track))):
                    raise AssertionError("Final HMM dropped original observations")
                if set(audited.records) != set(range(len(track))):
                    raise AssertionError("Final effective candidate audit incomplete")
                if result.breaks or not validate_route(graph, result.route)["valid"]:
                    raise AssertionError("HMM returned a teleport or disconnected output")
                effective_config = MatchConfig(**result.diagnostics["config"])
                standalone, classifier_seconds = standalone_classifier(graph, track, model, effective_config, audited)
                # Deliberately first student reference read, after both predictions.
                score = score_prediction(graph, track, result, audited, config, prefix.with_suffix(".route"))
                if score.metrics_status != "ok" or score.metrics["scored_observation_fraction"] != 1.0:
                    raise AssertionError("Backend scorer did not score every original observation")
                reference = load_route(prefix.with_suffix(".route"))
                labels = reference_observation_labels(graph, track, reference)
                truth = labels.edges
                classifier = np.array([int(audited.records[i].edges[np.argmax(audited.records[i].probabilities)]) for i in range(len(track))])
                if (not np.array_equal(classifier, np.asarray(result.diagnostics["classifier_only_edges"]))
                        or not np.array_equal(classifier, standalone)):
                    raise AssertionError("Classifier ablation candidates differ from decoder candidates")
                raw_sequence = adjacent_collapse(classifier)
                shared = {"candidate_miss_rate": score.metrics["candidate_miss_rate"],
                          "categorical_emission_log_loss": score.metrics["log_loss"],
                          "classifier_top1_accuracy": score.metrics["top1_accuracy"],
                          "proxy_ambiguous_fraction": float(np.mean(labels.ambiguous)),
                          "scored_observation_fraction": 1.0,
                          "effective_max_candidates": result.diagnostics["effective_max_candidates"],
                          "candidate_retry_count": len(result.diagnostics["candidate_retry_attempts"]),
                          "candidate_miss_count": int(round(score.metrics["candidate_miss_rate"] * len(track))),
                          "proxy_ambiguous_count": int(np.sum(labels.ambiguous)),
                          "source_input_sha256": {s: sha256(prefix.with_suffix(s)) for s in (".nodes", ".arcs", ".track", ".route")},
                          "candidate_retry_attempts": result.diagnostics["candidate_retry_attempts"],
                          "inference_timing_note": "Fresh MapMatcher including candidate index and all retries; excludes model/graph load and private scoring"}
                # Scoring time is not included in measured decoding latency.
                r_hmm = metric_row(graph, prefix.name, "classifier_plus_HMM", track, result.matched_edges, result.route, reference, truth, decode_seconds, shared)
                r_class = metric_row(graph, prefix.name, "classifier_only_raw", track, classifier, raw_sequence, reference, truth, classifier_seconds, shared)
                r_class["inference_timing_note"] = "Measured fresh candidate generation plus batch classifier; uses final effective HMM candidate width for matched ablation, no DP or graph load"
                r_hmm["scorer_metrics_status"] = score.metrics_status
                for key in ("accuracy", "precision", "recall", "macro_f1"):
                    if not np.isclose(r_hmm[f"proxy_{key}"], score.metrics[key], rtol=1e-12, atol=1e-15):
                        raise AssertionError("Independent pointwise metrics disagree with backend scorer")
                    if not np.isclose(r_class[f"proxy_{key}"], score.metrics[f"classifier_{key}"], rtol=1e-12, atol=1e-15):
                        raise AssertionError("Classifier pointwise metrics disagree with backend scorer")
                current = [(r_class, classifier), (r_hmm, result.matched_edges)]
                if args.baseline_output:
                    b, bp = baseline_row(prefix, graph, track, reference, truth, Path(args.baseline_output))
                    current.append((b, bp))
                for row, prediction in current:
                    rows.append(row)
                    pooled[row["method"]].append((graph.canonical_edges[truth] + offset, graph.canonical_edges[prediction] + offset))
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                    progress({"stage": "student_ablation", **row})
                offset += len(graph.arcs)
            except Exception as exc:
                row = {"trip_id": prefix.name, "method": "classifier_plus_HMM", "status": "failed",
                       "elapsed_seconds": time.perf_counter() - trip_start,
                       "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                progress(row)
    if sha256(CONFIG_PATH) != frozen["config_file_sha256"] or sha256(args.model) != frozen["model_sha256"]:
        raise RuntimeError("Frozen artifact changed during ablation")
    aggregated = aggregate(rows, pooled, len(prefixes))
    output = {"completed_utc": utcnow(), "elapsed_seconds": time.perf_counter() - start,
              "expected_methods": expected_methods,
              "trip_count": len(prefixes), "failed_trip_ids": sorted({r["trip_id"] for r in rows if r["status"] != "ok"}),
              "all_ten_headline_eligible": (set(aggregated) == set(expected_methods)
                                            and all(v["all_ten_headline_eligible"] for v in aggregated.values())),
              "freeze": frozen, "aggregate": aggregated, "trips": rows,
              "failures": [r for r in rows if r["status"] != "ok"]}
    write_csv(REPORT / "student_ablation.csv", rows)
    write_json(REPORT / "student_ablation.json", output)
    progress({"stage": "student_ablation_complete", "aggregate": aggregated, "failures": len(output["failures"])})


def self_check():
    from mapmatch.data import Graph
    assert adjacent_collapse([1, 1, 2, 1, 1, 3]).tolist() == [1, 2, 1, 3]
    assert sample_indices(5, 256).tolist() == list(range(5))
    graph = Graph(np.array([[0., 0.], [.0001, 0.], [.0002, 0.]]), np.array([[0, 1], [1, 2], [2, 0]]))
    route = np.array([0, 1, 2, 0])
    occurrence = {0: [0, 3], 1: [1], 2: [2]}
    prefix = np.concatenate(([0.], np.cumsum(graph.edge_lengths[route])))
    breaks = np.array([0, 0, 0, 0])
    distance, reason = reference_span_distance(graph, route, occurrence, prefix, breaks, 0, .1, 1, .5)
    assert reason == "identified" and np.isclose(distance, 1.4 * graph.edge_lengths[0])
    distance, reason = reference_span_distance(graph, route, occurrence, prefix, breaks, 0, .5, 0, .4)
    assert distance == 0 and reason == "identified"
    print("decoder_experiments self-check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "search", "ablate", "self-check"))
    parser.add_argument("--train-root", type=Path, default=PROJECT / "Student folder/train")
    parser.add_argument("--student-root", type=Path, default=PROJECT / "Student folder/student_test")
    parser.add_argument("--model", type=Path, default=PROJECT / "artifacts/model.joblib")
    parser.add_argument("--baseline-output", type=Path, default=PROJECT / "outputs/baseline_student_test_v2")
    args = parser.parse_args()
    {"calibrate": calibrate, "search": search, "ablate": ablate, "self-check": lambda _: self_check()}[args.stage](args)


if __name__ == "__main__":
    main()
