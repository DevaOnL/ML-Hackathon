#!/usr/bin/env python3
"""Frozen TASK52 decoder menu and in-memory Lead integration helpers.

The CLI prints the menu only. It cannot discover/load raw trips, train a model,
read a target file, or run a route search. Lead owns authorized TRAIN selection
and postprediction scoring. Importing this module performs no file/data access.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from mapmatch.accuracy_matcher import (AccuracyEmissionModel, AccuracyMapMatcher,
                                       AccuracyTransitionPolicy)
from mapmatch.matcher import MatchConfig

ROOT = Path(__file__).resolve().parents[1]
MENU_PATH = ROOT / "experiments/accuracy_improvement/decoder/preregistered_menu.json"
SEED = 20261118


@dataclass(frozen=True)
class DecoderConfiguration:
    name: str
    match_config: MatchConfig
    transition_policy: AccuracyTransitionPolicy = AccuracyTransitionPolicy()

    @property
    def config(self):
        return self.match_config


def decoder_configurations() -> tuple[DecoderConfiguration, ...]:
    """Eight fixed-K12, dense-observation choices; no data-dependent construction."""
    base = replace(MatchConfig(), sigma_m=7.5, beta_m=5.0, max_candidates=12,
                   candidate_retry_limit=12, candidate_radius_m=80.0,
                   max_radius_m=1000.0, observation_distance_m=0.0,
                   observation_interval=5.0, classifier_weight=1.0, geometry_weight=1.0)
    slack = AccuracyTransitionPolicy(path_cost_slack_m=10.0)
    stationary = replace(slack, stationary_backtrack_weight=1.0, immediate_reverse_penalty=2.0)
    return (
        DecoderConfiguration("fixed_b5_c1_g1", base),
        DecoderConfiguration("fixed_b2_c1_g1", replace(base, beta_m=2.0)),
        DecoderConfiguration("fixed_b10_c1_g1", replace(base, beta_m=10.0)),
        DecoderConfiguration("fixed_b5_c2_g1", replace(base, classifier_weight=2.0)),
        DecoderConfiguration("fixed_b5_c1_g025", replace(base, geometry_weight=0.25)),
        DecoderConfiguration("fixed_b5_c2_g025", replace(base, classifier_weight=2.0, geometry_weight=0.25)),
        DecoderConfiguration("fixed_b5_slack", base, slack),
        DecoderConfiguration("fixed_b5_stationary_slack", base, stationary),
    )


def menu_manifest() -> dict:
    return {"schema": "accuracy-decoder-menu-v1", "seed": SEED,
            "configurations": [{"name": c.name, "match_config": asdict(c.match_config),
                                 "transition_policy": asdict(c.transition_policy)}
                                for c in decoder_configurations()],
            "selection": "equal-trip length_f1; tie edge_f1; then menu order",
            "scale_policy": "fixed sigma7.5/beta constants; no label-derived calibration",
            "selection_route_rule": "two SHA256(20261118:id)-smallest validation IDs per fold",
            "candidate_policy": "K12, radius80 with inherited empty-only radius fallback, dense allobs",
            "raw_train_decoding": "Lead only after frozen candidate manifest + explicit protocol gate",
            "historical_model_is_paired_baseline": False}


def verify_menu_registration(path: str | Path = MENU_PATH) -> str:
    """Check on-disk preregistration against code, returning its byte SHA256."""
    raw = Path(path).read_bytes()
    registered = json.loads(raw)
    base = decoder_configurations()[0].match_config
    expected = []
    for item in registered["configurations"]:
        config = replace(base, **item["match_overrides"])
        policy = AccuracyTransitionPolicy(**item["transition_policy"])
        expected.append(DecoderConfiguration(item["name"], config, policy))
    if tuple(expected) != decoder_configurations():
        raise ValueError("decoder menu code differs from on-disk preregistration")
    for name in ("max_candidates", "candidate_retry_limit", "candidate_radius_m",
                 "max_radius_m", "observation_distance_m", "observation_interval"):
        if registered["candidate_policy"].get(name) != getattr(base, name):
            raise ValueError("candidate policy differs from on-disk preregistration")
    if any(getattr(base, k) != v for k, v in registered["base_config_overrides"].items()):
        raise ValueError("fixed physical constants differ from preregistration")
    return hashlib.sha256(raw).hexdigest()


def _ids(values, name):
    if (not isinstance(values, (list, tuple)) or not values
            or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{name} must contain nonempty unique string trip IDs")
    return tuple(sorted(values))


def selection_plan(folds: Sequence[Mapping], *, tuning_trip_ids=None) -> tuple[dict, ...]:
    """Plan six routes from declared whole-trip partitions, without opening data.

    Pass accuracy_common.get_folds(load_train_dataset()). No outer IDs, candidate
    scores or reference metrics are used in the deterministic selection.
    """
    if len(folds) != 3:
        raise ValueError("three tuning64 whole-trip folds required")
    parsed, universe, seen_validation = {}, None, set()
    for fold in folds:
        index = fold.get("fold", fold.get("fold_id"))
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or index not in (0, 1, 2) or index in parsed:
            raise ValueError("fold IDs must be unique 0,1,2")
        train = _ids(fold.get("train_trip_ids"), "train_trip_ids")
        valid = _ids(fold.get("validation_trip_ids"), "validation_trip_ids")
        current = set(train) | set(valid)
        if (set(train) & set(valid) or len(valid) < 2 or len(current) != 64
                or (universe is not None and current != universe)
                or seen_validation & set(valid)):
            raise ValueError("fold partitions must be disjoint and cover exactly the same tuning64")
        universe = current
        seen_validation.update(valid)
        parsed[int(index)] = (train, valid)
    if seen_validation != universe:
        raise ValueError("every tuning64 trip must validate exactly once")
    if tuning_trip_ids is not None and set(_ids(tuning_trip_ids, "tuning_trip_ids")) != universe:
        raise ValueError("fold population differs from registered tuning64")
    result = []
    for index in range(3):
        train, valid = parsed[index]
        order = sorted(valid, key=lambda t: (hashlib.sha256(f"{SEED}:{t}".encode()).hexdigest(), t))
        for trip in order[:2]:
            result.append({"fold_id": index, "trip_id": trip,
                           "train_trip_ids": list(train), "validation_trip_ids": list(valid),
                           "selection_sha256": hashlib.sha256(f"{SEED}:{trip}".encode()).hexdigest()})
    return tuple(result)


def assert_paired_models(baseline: AccuracyEmissionModel, candidate: AccuracyEmissionModel):
    """A historical all80 estimator is not an eligible paired control."""
    if baseline.profile != "v1":
        raise ValueError("paired baseline must be a directly refitted v1 fold model")
    for name in ("fold_id", "train_trip_ids", "validation_trip_ids", "protocol_sha256"):
        if baseline.metadata[name] != candidate.metadata[name]:
            raise ValueError(f"paired model partitions/protocol differ: {name}")


def require_decoder_authorization(protocol: Mapping, *, candidate_manifest_sha256: str,
                                  decoder_menu_sha256: str) -> Mapping:
    """Separate raw-route gate; training_authorized alone is insufficient.

    Expected Lead protocol block:
    decoder_selection={authorized:true, model_menu_frozen:true,
      candidate_manifest_sha256:<pin>, decoder_menu_sha256:<pin>}.
    """
    gate = protocol.get("decoder_selection", {})
    if (protocol.get("training_authorized") is not True or not isinstance(gate, Mapping)
            or gate.get("authorized") is not True or gate.get("model_menu_frozen") is not True):
        raise ValueError("raw TRAIN decoding requires Lead authorization and a frozen model menu")
    for key, pin in (("candidate_manifest_sha256", candidate_manifest_sha256),
                     ("decoder_menu_sha256", decoder_menu_sha256)):
        if (not isinstance(pin, str) or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin)
                or gate.get(key) != pin):
            raise ValueError(f"decoder authorization pin mismatch: {key}")
    return gate


def decode_arrays(graph, track, model: AccuracyEmissionModel, configuration: DecoderConfiguration,
                  *, trip_id: str):
    """In-memory inference only; Lead must gate raw-data access before calling.

    Full raw track is required (not the old 350-observation cached sample). The
    fixed configuration retains every GPS row and extracts full-window features.
    There is intentionally no input-root or route-path argument.
    """
    if trip_id not in model.metadata["validation_trip_ids"] or trip_id in model.metadata["train_trip_ids"]:
        raise ValueError("selection route must be held out of its fold estimator")
    if configuration not in decoder_configurations():
        raise ValueError("selection configuration is not in the frozen eight-item menu")
    with threadpool_limits(limits=2):
        result = AccuracyMapMatcher(graph, configuration.match_config, model,
                                    configuration.transition_policy).match(track)
    if len(result.observation_indices) != len(track):
        raise ValueError("strict first-pass decoder must retain all raw GPS observations")
    result.diagnostics["accuracy_fold_id"] = model.metadata["fold_id"]
    result.diagnostics["accuracy_feature_profile"] = model.profile
    result.diagnostics["accuracy_configuration"] = configuration.name
    result.diagnostics["accuracy_probability_checks"] = dict(model.probability_checks)
    return result


def score_decoded_route(graph, track, result, target_route) -> dict:
    """Postprediction TRAIN-array scoring, never called by matching.

    Reuse unchanged route metrics and proxy-label/classification definitions; no
    files are opened and the prediction is never edited in response to targets.
    Lead owns supplying permitted TRAIN targets only after inference.
    """
    from mapmatch.features import reference_observation_labels
    from mapmatch.metrics import score_routes
    from backend.scoring import classification_metrics
    scores = score_routes(graph, result.route, target_route)
    proxy = reference_observation_labels(graph, track, target_route,
                                         observation_indices=result.observation_indices)
    truth = graph.canonical_edges[proxy.edges]
    prediction = graph.canonical_edges[result.matched_edges]
    classification = classification_metrics(truth, prediction)
    scores.update({"proxy_edge_accuracy": classification["accuracy"],
                   "proxy_edge_macro_f1": classification["macro_f1"],
                   "proxy_edge_precision": classification["precision"],
                   "proxy_edge_recall": classification["recall"],
                   "elapsed_seconds": float(result.diagnostics["elapsed_seconds"]),
                   "extra_arc_count": scores["predicted_arc_count"] - scores["common_arc_count"],
                   "proxy_label_policy": "unchanged route_projection_proxy_not_independent_per_GPS_ground_truth"})
    return scores


def aggregate_route_records(records: Sequence[Mapping], expected_trip_ids) -> dict:
    """Keep failed routes in the denominator; reject omitted/duplicate routes."""
    expected = set(_ids(expected_trip_ids, "expected selection trips"))
    observed = [r["trip_id"] for r in records]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError("every selected trip, including failures, requires exactly one record")
    success = [r for r in records if r.get("status") == "complete"]
    if any(r.get("status") not in ("complete", "failed") for r in records):
        raise ValueError("route record status must be complete or failed")
    def value(record, key):
        if record["status"] == "failed":
            return 0.0
        v = record["metrics"][key]
        if isinstance(v, (bool, np.bool_)) or not np.isfinite(v) or not 0.0 <= v <= 1.0:
            raise ValueError(f"invalid route metric {key}")
        return float(v)
    return {"routes": len(records), "successful_routes": len(success),
            "failed_routes": len(records) - len(success),
            "mean_length_f1": float(np.mean([value(r, "length_f1") for r in records])),
            "mean_edge_f1": float(np.mean([value(r, "edge_f1") for r in records])),
            "failure_policy": "zero contribution; all six selection routes remain in denominator"}


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-menu", action="store_true", help="Print fixed configurations without loading any data")
    args = parser.parse_args(argv)
    if not args.show_menu:
        parser.error("raw route runs are Lead-owned; only --show-menu is supported")
    print(json.dumps(menu_manifest(), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
