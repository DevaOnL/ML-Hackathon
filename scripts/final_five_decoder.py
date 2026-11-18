#!/usr/bin/env python3
"""Final five-family decoder calibration (TRAIN-only search) and student ablation.

Scope (docs/FINAL_SCOPE.json, phase five-model-completion): all five model
families (Logistic Regression, distance-weighted KNN, Bagging, Gaussian Naive
Bayes, HistGradientBoosting) are complete and frozen. The emission model is
the final selection frozen at artifacts/final_five/model.joblib — the
TRAIN-selected HistGradientBoosting with the fixed tuned parameters recorded
in experiments/final_five/bayesian_optimization.json. This script never fits
or refits a classifier and runs no hyperparameter search; it only calibrates
decoder geometry weights on TRAIN and ablates the frozen decoder on the ten
student-development trips. Geometry scales (sigma/beta) are reused read-only
from the historical TRAIN-only fit in reports/decoder/ because they are
classifier-independent. Writes stay in reports/final_five/decoder/ and
artifacts/final_five/match_config.json. The historical phase-three decoder
freeze (artifacts/phase_three/match_config.json, reports/phase_three/decoder/)
and the earlier HistGB freeze (artifacts/match_config.json, reports/decoder/)
are preserved read-only and are never relabeled as this phase's result.

No inference starts until reports/final_five/all_five_model_acceptance.json
exists with passed=true and artifacts/final_five/clean_load_report.json
passed with the same frozen model hash. No evaluator input is ever opened,
read or listed by this script.
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
from sklearn.ensemble import HistGradientBoostingClassifier

from mapmatch.data import (discover_trips, load_graph, load_route, load_track,
                           validate_route)
from mapmatch.features import reference_observation_labels
from mapmatch.matcher import MapMatcher, MatchConfig
from mapmatch.metrics import score_routes

PHASE = "five-model-completion"
REPORT = PROJECT / "reports" / "final_five" / "decoder"
CONFIG_PATH = PROJECT / "artifacts" / "final_five" / "match_config.json"
DEFAULT_MODEL = PROJECT / "artifacts" / "final_five" / "model.joblib"
ACCEPTANCE_PATH = PROJECT / "reports" / "final_five" / "all_five_model_acceptance.json"
CLEAN_LOAD_PATH = PROJECT / "artifacts" / "final_five" / "clean_load_report.json"
MODEL_FREEZE_PATH = PROJECT / "artifacts" / "final_five" / "model_freeze.json"
SELECTION_EVIDENCE = ("experiments/final_five/selection.json",
                      "experiments/final_five/bayesian_optimization.json")
DEFAULT_SCALE_FIT = PROJECT / "reports" / "decoder" / "scale_fit.json"
SCALE_RESIDUALS = PROJECT / "reports" / "decoder" / "training_residual_samples.csv"
SCALE_TRANSITIONS = PROJECT / "reports" / "decoder" / "training_transition_samples.csv"
FINAL_SCOPE_PATH = PROJECT / "docs" / "FINAL_SCOPE.json"
REQUESTED_FAMILIES = ("logistic_regression", "knn_distance_weighted", "bagging",
                      "gaussian_nb", "hist_gradient_boosting")
EXPECTED_FINAL_FAMILY = "hist_gradient_boosting"
EXPECTED_FINAL_RUN_NAME = "hist_gradient_boosting_tuned"
# Fixed tuned parameters of the frozen final model; winner of the completed
# TRAIN-only GP-EI study, reproduced by the fresh tuned 3-fold CV. Never a
# decoder-stage search: the decoder must refuse any other configuration.
FIXED_HISTGB_PARAMETERS = {
    "max_iter": 104,
    "learning_rate": 0.15902391896503631,
    "max_leaf_nodes": 22,
    "min_samples_leaf": 38,
    "l2_regularization": 0.8604220152998319,
}
CALIBRATION_IDS = tuple(f"{i:08d}" for i in range(1, 72, 10))
SIGMA_BOUNDS = (3.0, 50.0)
BETA_BOUNDS = (2.0, 100.0)
EXPECTED_STUDENT_TRIPS = 10
EXPECTED_STUDENT_OBSERVATIONS = 19160
HISTORICAL_NOTE = ("Historical decoder freezes at artifacts/match_config.json "
                   "(reports/decoder/freeze.json, earlier HistGB) and "
                   "artifacts/phase_three/match_config.json "
                   "(reports/phase_three/decoder/freeze.json, requested trio) are "
                   "preserved read-only; neither is this phase's final result.")
CANDIDATE_WIDTH_NOTE = ("Candidate width starts at max_candidates=12 and doubles "
                        "per connected-transition retry (12->24->48), capped by "
                        "candidate_retry_limit=48; the limit caps the widened "
                        "width, not the initial width and not the attempt count.")
SOURCES = ("scripts/final_five_decoder.py", "mapmatch/data.py",
           "mapmatch/candidates.py", "mapmatch/matcher.py",
           "mapmatch/routing.py", "mapmatch/features.py",
           "mapmatch/learning.py", "mapmatch/metrics.py",
           "mapmatch/final_models.py", "backend/emissions.py",
           "backend/scoring.py")


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
    return {p: sha256(PROJECT / p) for p in SOURCES if (PROJECT / p).exists()}


def requested_families():
    try:
        scope = json.loads(FINAL_SCOPE_PATH.read_text(encoding="utf-8"))
        families = tuple(scope.get("model_families") or ())
        return families if families else REQUESTED_FAMILIES
    except (OSError, ValueError):
        return REQUESTED_FAMILIES


def display_path(path):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT))
    except ValueError:
        return str(resolved)


def assert_no_evaluator_paths(*paths):
    """No evaluator input may be opened, read or listed anywhere in this script."""
    for path in paths:
        text = str(Path(path).resolve()).lower()
        if "evaluator" in text:
            raise ValueError("Refusing an evaluator path as decoder input: " + text)


def load_final_model(path):
    """Load the frozen final emission model with the standard bundle loader."""
    from mapmatch.learning import load_emission_model
    return load_emission_model(path), "mapmatch.learning"


def require_final_acceptance(model_path):
    """Hard gate: no decoder inference until the final five-model acceptance
    exists, passed, and pins exactly the frozen model being loaded."""
    assert_no_evaluator_paths(model_path)
    missing = [str(p) for p in (ACCEPTANCE_PATH, CLEAN_LOAD_PATH, MODEL_FREEZE_PATH,
                                model_path) if not Path(p).exists()]
    if missing:
        raise ValueError("Final five-family model acceptance is not complete; "
                         "missing: " + ", ".join(missing))
    acceptance = read_json(ACCEPTANCE_PATH)
    clean = read_json(CLEAN_LOAD_PATH)
    freeze = read_json(MODEL_FREEZE_PATH)
    if acceptance.get("passed") is not True:
        raise ValueError("all_five_model_acceptance.json has not passed")
    if acceptance.get("exactly_five_families") is not True or tuple(acceptance.get("families") or ()) != REQUESTED_FAMILIES:
        raise ValueError("final acceptance does not cover exactly the five declared families")
    if any(entry.get("passed") is not True for entry in (acceptance.get("all_models") or {}).values()):
        raise ValueError("final acceptance contains a model that did not pass")
    if acceptance.get("all80_train_trips") is not True or acceptance.get("all10_student_trips") is not True:
        raise ValueError("final acceptance does not confirm all-80 TRAIN and all-10 student trips")
    if acceptance.get("student_used_for_selection") is not False:
        raise ValueError("final acceptance reports student use in selection")
    if acceptance.get("evaluator_accessed") is not False:
        raise ValueError("final acceptance reports evaluator access; decoder stays blind")
    if clean.get("passed") is not True:
        raise ValueError("clean-environment model load report has not passed")
    model_hash = sha256(model_path)
    for label, record in (("acceptance", acceptance), ("clean_load", clean), ("model_freeze", freeze)):
        if record.get("model_sha256", record.get("selected_model_sha256")) != model_hash:
            raise ValueError(f"{label} does not pin the current frozen final model")
    return {"acceptance_path": str(ACCEPTANCE_PATH), "acceptance_sha256": sha256(ACCEPTANCE_PATH),
            "acceptance_passed": True, "clean_load_path": str(CLEAN_LOAD_PATH),
            "clean_load_sha256": sha256(CLEAN_LOAD_PATH), "clean_load_passed": True,
            "model_freeze_sha256": sha256(MODEL_FREEZE_PATH),
            "model_sha256": model_hash,
            "gate_note": "Decoder inference started only after the five-family acceptance "
                         "and clean-environment load both passed with this exact model hash."}


def verify_final_model(model):
    """Strict check that the frozen model is the final TRAIN-selected HistGB
    with the fixed tuned parameters. The decoder refuses anything else."""
    metadata = getattr(model, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    estimator = getattr(model, "estimator", None)
    if not isinstance(estimator, HistGradientBoostingClassifier):
        raise ValueError("Final emission model is not a HistGradientBoostingClassifier")
    if metadata.get("model_name") != EXPECTED_FINAL_FAMILY:
        raise ValueError("Frozen final model family is not " + EXPECTED_FINAL_FAMILY)
    if metadata.get("run_name") != EXPECTED_FINAL_RUN_NAME:
        raise ValueError("Frozen final model run name is not " + EXPECTED_FINAL_RUN_NAME)
    if metadata.get("parameters") != FIXED_HISTGB_PARAMETERS:
        raise ValueError("Frozen final model parameters differ from the fixed tuned "
                         "HistGB configuration: " + json.dumps(metadata.get("parameters"), sort_keys=True))
    if metadata.get("phase") != PHASE:
        raise ValueError("Frozen final model was not produced for phase " + PHASE)
    if metadata.get("training_trip_count") != 80:
        raise ValueError("Frozen final model was not fitted on all 80 TRAIN trips")
    params = estimator.get_params(deep=False)
    mismatched = {k: params.get(k) for k, v in FIXED_HISTGB_PARAMETERS.items() if params.get(k) != v}
    if mismatched:
        raise ValueError("Final estimator hyperparameters differ from the fixed tuned "
                         "configuration: " + json.dumps(mismatched, sort_keys=True))
    return {"family": EXPECTED_FINAL_FAMILY, "run_name": EXPECTED_FINAL_RUN_NAME,
            "fixed_parameters": dict(FIXED_HISTGB_PARAMETERS),
            "parameters_match_fixed": True, "training_trip_count": 80,
            "enforcement": "strict; the decoder aborts on any other family or configuration"}


def family_check(model):
    """Informational consistency record over the five requested families."""
    metadata = getattr(model, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    requested = list(requested_families())
    fields = {k: metadata.get(k) for k in ("model_name", "run_name", "family")}
    text = " ".join(str(v) for v in fields.values() if v is not None).lower()
    tokens = {"logistic_regression": ("logistic",),
              "knn_distance_weighted": ("knn", "neighbor", "distance"),
              "bagging": ("bag", "bootstrap"),
              "gaussian_nb": ("naive bayes", "gaussian nb", "gaussiannb", "gaussian_nb", "gnb"),
              "hist_gradient_boosting": ("hist gradient", "histgradient", "hist_gradient", "gradient boosting", "hgb")}
    matched = sorted(f for f, keys in tokens.items() if any(k in text for k in keys))
    return {"requested_families": requested, "metadata_fields": fields,
            "matched_requested_families": matched,
            "consistent_with_requested_five": bool(matched) and all(m in requested for m in matched),
            "selected_family_matches_metadata": EXPECTED_FINAL_FAMILY in matched,
            "enforcement": "informational; the strict family/parameter gate is verify_final_model"}


def write_json(path, value):
    path = Path(path)
    if path != CONFIG_PATH and not path.resolve().is_relative_to(REPORT.resolve()):
        raise ValueError("Final experiment writes must stay in reports/final_five/decoder/ "
                         "or artifacts/final_five/match_config.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    path = Path(path)
    if not path.resolve().is_relative_to(REPORT.resolve()):
        raise ValueError("Final CSV must stay in reports/final_five/decoder/")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def progress(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def scale_evidence_summary(scale_fit_path, residuals_path, transitions_path):
    data = read_json(scale_fit_path)
    return {"path": display_path(scale_fit_path),
            "sha256": sha256(scale_fit_path),
            "fitted": data["fitted"],
            "sigma_unclamped_m": data["sigma_unclamped_m"],
            "beta_unclamped_m": data["beta_unclamped_m"],
            "sigma_clamped": data["sigma_clamped"],
            "beta_clamped": data["beta_clamped"],
            "sigma_sample_count": data["perpendicular_residual_m"]["count"],
            "sigma_observations_excluded": data["sigma_observations_excluded"],
            "beta_pair_status_counts": data["beta_pair_status_counts"],
            "training_trip_count": data["training_trip_count"],
            "training_reference_discontinuity_count": data["reference_discontinuity_count"],
            "residual_samples_path": display_path(residuals_path),
            "residual_samples_sha256": sha256(residuals_path),
            "transition_samples_path": display_path(transitions_path),
            "transition_samples_sha256": sha256(transitions_path),
            "original_plan_sha256": data["plan_sha256"],
            "reuse_rationale": ("Geometry-only sigma/beta come from TRAIN reference residuals and "
                                "transition spans and do not depend on the emission classifier "
                                "family, so the historical immutable fit is reused read-only "
                                "instead of recomputed.")}


def plan(scale):
    return {
        "declared_utc": utcnow(), "version": 1, "phase": PHASE,
        "requested_model_families": list(requested_families()),
        "training_root": "Student folder/train", "train_trip_count": 80,
        "final_model": {
            "family": EXPECTED_FINAL_FAMILY,
            "run_name": EXPECTED_FINAL_RUN_NAME,
            "fixed_parameters": dict(FIXED_HISTGB_PARAMETERS),
            "selection": "TRAIN-only out-of-fold trip-macro observation top-1 among exactly the "
                         "five declared families; tuned configuration is the completed TRAIN-only "
                         "GP-EI study winner reproduced by fresh tuned 3-fold CV",
            "no_decoder_stage_hpo": "The decoder performs no hyperparameter search and no classifier "
                                    "fit; the final model and its parameters are frozen upstream",
            "artifact": "artifacts/final_five/model.joblib"},
        "acceptance_gate": "Search and ablation refuse to run until "
                           "reports/final_five/all_five_model_acceptance.json passed=true and "
                           "artifacts/final_five/clean_load_report.json passed=true pin the same "
                           "frozen model hash",
        "evaluator_boundary": "No evaluator input is opened, read or listed at any stage; "
                              "evaluator_accessed stays false in every output",
        "scale_evidence": scale,
        "scale_reuse": "read-only reuse of the historical TRAIN-only geometry fit; no refitting and no copy",
        "calibration_trip_ids": list(CALIBRATION_IDS),
        "calibration_rule": "first eight sorted TRAIN IDs whose numeric ID modulo 10 is 1; unchanged preregistered TRAIN8 set",
        "objective": "mean exact-original-ID length-weighted repeated-arc route F1 across the eight TRAIN calibration trips; a failed trip contributes zero",
        "selection_tie_break": "predeclared order A, B, C for exactly equal objectives",
        "configurations": [
            {"name": "A_fitted", "beta_multiplier": 1.0, "classifier_weight": 1.0, "geometry_weight": 1.0},
            {"name": "B_twice_beta", "beta_multiplier": 2.0, "classifier_weight": 1.0, "geometry_weight": 1.0},
            {"name": "C_classifier_up", "beta_multiplier": 1.0, "classifier_weight": 2.0, "geometry_weight": 0.5},
        ],
        "candidate_width_policy": CANDIDATE_WIDTH_NOTE,
        "freeze": "Write the final MatchConfig artifact before student-development ablation; never revise it from those results",
        "search_input_boundary": "The search opens TRAIN tracks and TRAIN reference routes only; no student file is read during search and the plan cannot change from student outcomes",
        "decoder_script_sha256": sha256(Path(__file__)),
        "historical_comparator": {"baseline_output": "outputs/baseline_student_test_v2",
                                  "note": "Geometry-only all-ten baseline (macro lengthF1 0.9734481048408883, countF1 0.9261891312626117) reused as a historical comparator, not as a new learned model"},
        "limitations": ["Scale evidence is a historical TRAIN-only fit reused read-only; this phase never refits geometry scales.",
                        "The final emission model was fitted on all TRAIN80, so decoder TRAIN8 search is in-sample calibration after classifier fitting, not unseen generalization.",
                        "Historical student-development results (geometry-only baseline, phase-three trio and earlier HistGB runs) existed as context, so this is not a fully blind phase; the predeclared search still reads only TRAIN predictions and references and student scores cannot revise it.",
                        "The ten student-development trips were classified in earlier experiments before this calibration; that prior classification is disclosed and was not used for decoder configuration choice, and this calibration reads no student file until after the config freeze.",
                        "The ten student-development trips are real and used for ablation; they are not the reserved evaluator set, which remains unread.",
                        "Pointwise labels are route-derived proxies with a weak GPS-motion direction tie-break, not supplied observation-aligned ground truth."],
    }


def plan_matches(existing, fresh):
    keys = ("calibration_trip_ids", "configurations", "objective", "selection_tie_break",
            "final_model", "candidate_width_policy")
    if any(existing.get(k) != fresh.get(k) for k in keys):
        return False
    old, new = existing.get("scale_evidence", {}), fresh.get("scale_evidence", {})
    return (old.get("sha256") == new.get("sha256")
            and old.get("fitted") == new.get("fitted")
            and old.get("residual_samples_sha256") == new.get("residual_samples_sha256")
            and old.get("transition_samples_sha256") == new.get("transition_samples_sha256"))


def declare(args):
    assert_no_evaluator_paths(args.scale_fit, args.scale_residuals, args.scale_transitions)
    path = REPORT / "calibration_plan.json"
    fresh = plan(scale_evidence_summary(args.scale_fit, args.scale_residuals, args.scale_transitions))
    if path.exists():
        existing = read_json(path)
        if not plan_matches(existing, fresh):
            raise ValueError("Existing final declaration differs from the in-code protocol")
        progress({"stage": "declare", "status": "already_declared",
                  "plan_sha256": canonical_hash(existing), "plan_path": str(path)})
        return existing
    write_json(path, fresh)
    progress({"stage": "declare", "status": "declared", "plan_sha256": canonical_hash(fresh),
              "plan_path": str(path), "calibration_trip_ids": list(CALIBRATION_IDS),
              "scale_fit_sha256": fresh["scale_evidence"]["sha256"],
              "decoder_script_sha256": fresh["decoder_script_sha256"]})
    return fresh


def ensure_plan(args):
    path = REPORT / "calibration_plan.json"
    if not path.exists():
        raise ValueError("Run the declare stage before search; the protocol must be declared before inference")
    existing = read_json(path)
    fresh = plan(scale_evidence_summary(args.scale_fit, args.scale_residuals, args.scale_transitions))
    if not plan_matches(existing, fresh):
        raise ValueError("Final declaration differs from the in-code protocol; the preregistered plan is immutable")
    if existing.get("decoder_script_sha256") not in (None, sha256(Path(__file__))):
        raise ValueError("Decoder source changed after the preregistered declaration; redeclare before any inference")
    return existing


def verify_scale(args, declaration):
    recorded = declaration["scale_evidence"]
    if sha256(args.scale_fit) != recorded["sha256"]:
        raise ValueError("Reused scale evidence changed after final declaration")
    data = read_json(args.scale_fit)
    if data["fitted"] != recorded["fitted"]:
        raise ValueError("Reused scale evidence fitted values differ from the declaration")
    return data


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


def overlap_only(graph, prediction, reference):
    p, r = Counter(map(int, prediction)), Counter(map(int, reference))
    common = p & r
    common_count = sum(common.values())
    common_length = sum(float(graph.edge_lengths[e]) * n for e, n in common.items())
    plen, rlen = float(graph.edge_lengths[prediction].sum()), float(graph.edge_lengths[reference].sum())
    return {"edge_f1": 2 * common_count / max(len(prediction) + len(reference), 1),
            "length_f1": 2 * common_length / (plen + rlen) if plen + rlen else 0.0,
            "predicted_arc_count": len(prediction),
            "reference_arc_count": len(reference),
            "predicted_length_m": plen, "reference_length_m": rlen}


def selection_evidence():
    evidence = {}
    for name in SELECTION_EVIDENCE:
        path = PROJECT / name
        if not path.exists():
            raise ValueError("Final selection evidence missing: " + name)
        evidence[name] = sha256(path)
    return evidence


def search(args):
    assert_no_evaluator_paths(args.train_root, args.model, args.scale_fit,
                              args.scale_residuals, args.scale_transitions)
    declaration = ensure_plan(args)
    scale = verify_scale(args, declaration)
    acceptance = require_final_acceptance(args.model)
    if CONFIG_PATH.exists() or (REPORT / "freeze.json").exists() or (REPORT / "search_results.jsonl").exists():
        raise ValueError("Final config/search already exists; preserve frozen experiment")
    model_path = Path(args.model)
    model_hash = sha256(model_path)
    model, loader_module = load_final_model(model_path)
    model_check = verify_final_model(model)
    configs = []
    for item in declaration["configurations"]:
        config = MatchConfig(sigma_m=scale["fitted"]["sigma_m"],
                             beta_m=float(np.clip(scale["fitted"]["beta_m"] * item["beta_multiplier"], *BETA_BOUNDS)),
                             classifier_weight=item["classifier_weight"], geometry_weight=item["geometry_weight"])
        configs.append((item["name"], config))
    write_json(REPORT / "search_configs.json", {"created_utc": utcnow(), "phase": PHASE,
                                                "model_sha256": model_hash, "loader_module": loader_module,
                                                "family_check": family_check(model),
                                                "final_model_check": model_check,
                                                "acceptance": acceptance,
                                                "selection_evidence_sha256": selection_evidence(),
                                                "plan_sha256": canonical_hash(declaration),
                                                "scale_evidence": declaration["scale_evidence"],
                                                "configs": {n: asdict(c) for n, c in configs},
                                                "candidate_width_policy": CANDIDATE_WIDTH_NOTE,
                                                "source_sha256": source_hashes()})
    by_id = {p.name: p for p in discover_trips(args.train_root)}
    missing = [t for t in CALIBRATION_IDS if t not in by_id]
    if missing:
        raise ValueError(f"TRAIN calibration trips missing: {missing}")
    records = []
    start = time.perf_counter()
    with (REPORT / "search_results.jsonl").open("w", encoding="utf-8") as log:
        for trip_id in CALIBRATION_IDS:
            prefix = by_id[trip_id]
            graph = load_graph(prefix)
            track = load_track(prefix.with_suffix(".track"))
            # Calibration references are used only for scoring, never by MapMatcher.
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
                progress({"stage": "search", **{k: v for k, v in row.items() if k != "traceback"}})
    if sha256(model_path) != model_hash:
        raise RuntimeError("Final model changed during decoder selection; no config frozen")
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
    frozen = {"frozen_utc": utcnow(), "phase": PHASE,
              "selected_configuration": name,
              "config": asdict(config), "config_file_sha256": sha256(CONFIG_PATH),
              "effective_config_sha256": canonical_hash(asdict(config)),
              "model_sha256": model_hash, "model_path": display_path(model_path),
              "loader_module": loader_module, "family_check": family_check(model),
              "final_model_check": model_check,
              "acceptance": acceptance,
              "selection_evidence_sha256": selection_evidence(),
              "model_metadata": {k: (model.metadata.get(k) if isinstance(getattr(model, "metadata", None), dict) else None)
                                 for k in ("model_name", "run_name", "family", "parameters", "feature_version",
                                           "training_trip_count", "training_observations", "training_candidate_rows", "fit_data")},
              "plan_sha256": canonical_hash(declaration),
              "scale_evidence": declaration["scale_evidence"],
              "calibration_trip_ids": list(CALIBRATION_IDS), "student_results_seen_for_selection": False,
              "student_results_used_for_selection": False,
              "prior_student_context_available": True,
              "evaluator_accessed": False,
              "selection_boundary_note": ("Historical geometry-only, phase-three trio and earlier HistGB student "
                                          "results were available as context; this predeclared search reads only "
                                          "TRAIN predictions/references and never reads student scores. Final "
                                          "student ablation starts only after this freeze."),
              "historical_note": HISTORICAL_NOTE,
              "objective": declaration["objective"], "objectives": objectives,
              "elapsed_seconds": time.perf_counter() - start, "source_sha256": source_hashes()}
    write_csv(REPORT / "search_objectives.csv", objectives)
    write_json(REPORT / "freeze.json", frozen)
    progress({"stage": "FINAL_CONFIG_FROZEN", "phase": PHASE,
              "selected_configuration": name,
              "config_file_sha256": frozen["config_file_sha256"],
              "effective_config_sha256": frozen["effective_config_sha256"],
              "config": asdict(config), "model_sha256": model_hash,
              "loader_module": loader_module, "objectives": objectives})


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
    assert_no_evaluator_paths(args.student_root, args.model, args.baseline_output)
    acceptance = require_final_acceptance(args.model)
    frozen = read_json(REPORT / "freeze.json")
    config = MatchConfig(**read_json(CONFIG_PATH))
    if sha256(CONFIG_PATH) != frozen["config_file_sha256"] or sha256(args.model) != frozen["model_sha256"]:
        raise ValueError("Frozen final model/config changed")
    if (REPORT / "student_ablation.json").exists() or (REPORT / "student_ablation.jsonl").exists():
        raise ValueError("Final student ablation already exists; do not overwrite reported evidence")
    model, loader_module = load_final_model(args.model)
    verify_final_model(model)
    prefixes = discover_trips(args.student_root)
    if len(prefixes) != EXPECTED_STUDENT_TRIPS:
        raise ValueError("Ablation requires all ten student-development trips")
    expected_methods = ["classifier_only_raw", "classifier_plus_HMM"]
    if args.baseline_output:
        expected_methods.append("geometry_HMM_default_reused")
    write_json(REPORT / "ablation_run.json", {"started_utc": utcnow(), "phase": PHASE,
               "requested_model_families": list(requested_families()),
               "final_model_check": verify_final_model(model),
               "acceptance": acceptance,
               "loader_module": loader_module,
               "student_trip_ids": [p.name for p in prefixes],
               "freeze_file_sha256": sha256(REPORT / "freeze.json"),
               "model_sha256": frozen["model_sha256"], "config_sha256": frozen["config_file_sha256"],
               "config": asdict(config), "source_sha256": source_hashes(),
               "scale_evidence": frozen["scale_evidence"],
               "baseline_output": display_path(args.baseline_output) if args.baseline_output else None,
               "baseline_provenance_sha256": {n: sha256(Path(args.baseline_output) / n) for n in ("run_config.json", "index.csv")} if args.baseline_output else None,
               "baseline_role": "Historical geometry-only comparator reused with provenance; not a new learned model",
               "candidate_width_policy": CANDIDATE_WIDTH_NOTE,
               "pointwise_definition": "Canonical directed arc labels from reference geometry-and-motion proxy; all original observations, none excluded; misses and ambiguity retained",
               "sequence_f1_definition": "Directed-arc sequence F1 uses multiset occurrence matching (Counter intersection) of canonical arc IDs; repeated arcs count per occurrence",
               "classifier_only_sequence": "Per-observation argmax of the exact final effective HMM candidates/probabilities; collapse adjacent identical IDs only; no connectors inserted",
               "log_loss_definition": "Categorical NLL of normalized pre-HMM classifier emissions including candidate misses at 1e-7; same emission NLL for classifier and HMM, NOT HMM posterior NLL or binary candidate log loss; geometry baseline uses normalized geometric emissions at the same 1e-7 floor",
               "posterior_nll_note": "N/A: the Viterbi decoder computes only the best path and no observation posteriors, so an HMM posterior NLL does not exist for this system",
               "pooled_macro_f1": "Macro over union of (trip_id, canonical directed arc ID) truth/prediction labels; graph-local edge numbers are never conflated",
               "reference_discontinuity_note": "Reference discontinuities are measured live from the supplied references and reported per trip below",
               "expected_total_observations": EXPECTED_STUDENT_OBSERVATIONS,
               "inference_uses_reference": False,
               "reserved_evaluator_data_available": False,
               "evaluator_accessed": False,
               "student_results_used_to_revise_config": False})
    rows, pooled = [], defaultdict(list)
    checksums, reference_discontinuities = {}, {}
    total_observations = 0
    offset = 0
    start = time.perf_counter()
    with (REPORT / "student_ablation.jsonl").open("w", encoding="utf-8") as log:
        for prefix in prefixes:
            trip_start = time.perf_counter()
            try:
                graph = load_graph(prefix)
                track = load_track(prefix.with_suffix(".track"))
                total_observations += len(track)
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
                reference_discontinuities[prefix.name] = [int(d) for d in validate_route(graph, reference)["discontinuities"]]
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
                checksums[prefix.name] = {"observations": len(track)}
                for row, prediction in current:
                    rows.append(row)
                    pooled[row["method"]].append((graph.canonical_edges[truth] + offset, graph.canonical_edges[prediction] + offset))
                    checksums[prefix.name][row["method"]] = {
                        "status": row["status"], "connected": bool(row["prediction_connected"]),
                        "predicted_arc_count": row["predicted_arc_count"],
                        "route_file_sha256": row["route_file_sha256"],
                        "matched_edge_array_sha256": row["matched_edge_array_sha256"]}
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                    progress({"stage": "student_ablation", **row})
                offset += len(graph.arcs)
            except Exception as exc:
                row = {"trip_id": prefix.name, "method": "classifier_plus_HMM", "status": "failed",
                       "elapsed_seconds": time.perf_counter() - trip_start,
                       "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
                rows.append(row)
                checksums[prefix.name] = {"status": "failed", "error_type": type(exc).__name__}
                log.write(json.dumps(row) + "\n")
                log.flush()
                progress(row)
    if sha256(CONFIG_PATH) != frozen["config_file_sha256"] or sha256(args.model) != frozen["model_sha256"]:
        raise RuntimeError("Frozen final artifact changed during ablation")
    aggregated = aggregate(rows, pooled, len(prefixes))
    output = {"completed_utc": utcnow(), "phase": PHASE,
              "loader_module": loader_module,
              "acceptance": acceptance,
              "elapsed_seconds": time.perf_counter() - start,
              "expected_methods": expected_methods,
              "trip_count": len(prefixes), "total_observations": total_observations,
              "expected_total_observations": EXPECTED_STUDENT_OBSERVATIONS,
              "observation_total_matches_expected": total_observations == EXPECTED_STUDENT_OBSERVATIONS,
              "student_reference_discontinuities": reference_discontinuities,
              "failed_trip_ids": sorted({r["trip_id"] for r in rows if r["status"] != "ok"}),
              "all_ten_headline_eligible": (set(aggregated) == set(expected_methods)
                                            and all(v["all_ten_headline_eligible"] for v in aggregated.values())),
              "evaluator_accessed": False,
              "freeze": frozen, "aggregate": aggregated, "trips": rows,
              "failures": [r for r in rows if r["status"] != "ok"]}
    write_csv(REPORT / "student_ablation.csv", rows)
    write_json(REPORT / "student_ablation.json", output)
    write_json(REPORT / "student_route_checksums.json", {
        "purpose": "Per-trip route/state checksums for Lead CLI parity; no final route directory is written by the final decoder experiment",
        "phase": PHASE,
        "model_sha256": frozen["model_sha256"],
        "config_file_sha256": frozen["config_file_sha256"],
        "evaluator_accessed": False,
        "route_file_sha256_definition": "sha256 of the predicted .route text (one original directed arc ID per line, newline-terminated)",
        "matched_edge_array_sha256_definition": "sha256 of the int64 little-endian matched-edge array",
        "trips": checksums})
    progress({"stage": "FINAL_DECODER_RESULTS_READY", "phase": PHASE,
              "aggregate": aggregated, "failures": len(output["failures"]),
              "total_observations": total_observations,
              "student_reference_discontinuities": reference_discontinuities})


def self_check(args=None):
    from mapmatch.data import Graph
    assert adjacent_collapse([1, 1, 2, 1, 1, 3]).tolist() == [1, 2, 1, 3]
    assert list(CALIBRATION_IDS) == [f"{i:08d}" for i in range(1, 72, 10)]
    assert FIXED_HISTGB_PARAMETERS == {"max_iter": 104, "learning_rate": 0.15902391896503631,
                                       "max_leaf_nodes": 22, "min_samples_leaf": 38,
                                       "l2_regularization": 0.8604220152998319}
    graph = Graph(np.array([[0., 0.], [.0001, 0.], [.0002, 0.]]), np.array([[0, 1], [1, 2], [2, 0]]))
    route = np.array([0, 1, 2, 0])
    occurrence = {0: [0, 3], 1: [1], 2: [2]}
    prefix = np.concatenate(([0.], np.cumsum(graph.edge_lengths[route])))
    breaks = np.array([0, 0, 0, 0])
    distance, reason = reference_span_distance(graph, route, occurrence, prefix, breaks, 0, .1, 1, .5)
    assert reason == "identified" and np.isclose(distance, 1.4 * graph.edge_lengths[0])
    distance, reason = reference_span_distance(graph, route, occurrence, prefix, breaks, 0, .5, 0, .4)
    assert distance == 0 and reason == "identified"
    assert_no_evaluator_paths(PROJECT / "Student folder" / "train")
    try:
        assert_no_evaluator_paths(PROJECT / "evaluator_test" / "trip")
    except ValueError:
        pass
    else:
        raise AssertionError("evaluator path guard did not fire")
    print("final_five_decoder self-check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("declare", "search", "ablate", "self-check"))
    parser.add_argument("--train-root", type=Path, default=PROJECT / "Student folder/train")
    parser.add_argument("--student-root", type=Path, default=PROJECT / "Student folder/student_test")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--baseline-output", type=Path, default=PROJECT / "outputs/baseline_student_test_v2")
    parser.add_argument("--scale-fit", type=Path, default=DEFAULT_SCALE_FIT)
    parser.add_argument("--scale-residuals", type=Path, default=SCALE_RESIDUALS)
    parser.add_argument("--scale-transitions", type=Path, default=SCALE_TRANSITIONS)
    args = parser.parse_args()
    {"declare": declare, "search": search, "ablate": ablate,
     "self-check": lambda a: self_check(a)}[args.stage](args)


if __name__ == "__main__":
    main()
