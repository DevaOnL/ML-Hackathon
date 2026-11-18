#!/usr/bin/env python3
"""Isolated TRAIN-only accuracy experiments; never imports or opens evaluator data.

The original release is immutable. Cached TRAIN proxy labels/candidates are reused
unchanged, with a prospective 64/16 trip partition for this iteration. These trips
were historically seen by the released model: this is not a new blind test set.
"""
from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import warnings

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from mapmatch.experiments import CandidateDataset, ExperimentConfig, build_group_folds
from mapmatch.features import FEATURE_NAMES, FEATURE_VERSION, PROXY_LABEL_POLICY
from mapmatch.learning import candidate_classification_metrics
from mapmatch.phase_models import bank_seed, knn_reference_bank

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "experiments/accuracy_improvement"
PRIVATE_ROOT = EXPERIMENT_ROOT / "private"
PROTOCOL_PATH = EXPERIMENT_ROOT / "PROTOCOL.json"
TRAIN_CACHE = ROOT / "experiments/cache/train_edee9a8f151dd25e.npz"
TRAIN_COVERAGE = TRAIN_CACHE.with_suffix(".json")
TRAIN_CACHE_SHA256 = "db523c112a6119f16b6f473d6221e3c56b6730ccb2db64dee55433aedbf3ba9e"
SEED = 20261118
THREADS = 2
FAMILIES = ("logistic_regression", "knn_distance_weighted", "bagging", "gaussian_nb", "hist_gradient_boosting")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x.relative_to(ROOT)) if x.is_relative_to(ROOT) else str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def write_new_json(path, value):
    path = Path(path)
    text = json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def split_trip_ids(trip_ids):
    ids = sorted(set(str(t) for t in trip_ids))
    if len(ids) < 5:
        raise ValueError("at least five distinct TRAIN trips required")
    ordered = sorted(ids, key=lambda t: (hashlib.sha256(f"{SEED}:{t}".encode()).hexdigest(), t))
    n_outer = max(1, len(ordered) // 5)
    return tuple(sorted(ordered[n_outer:])), tuple(sorted(ordered[:n_outer]))


def subset_dataset(dataset, observations):
    obs = np.asarray(observations)
    if obs.ndim != 1 or obs.dtype.kind not in "iu" or np.any(obs < 0) or np.any(obs >= len(dataset.observation_trip_ids)) or (len(obs) > 1 and np.any(np.diff(obs) <= 0)):
        raise ValueError("observation subset must be sorted, unique integer indices in bounds")
    lengths = np.diff(dataset.offsets)
    rows = np.concatenate([np.arange(dataset.offsets[i], dataset.offsets[i+1], dtype=np.int64) for i in obs]) if len(obs) else np.empty(0, np.int64)
    row_fields = {"X", "y", "candidate_edges", "candidate_fractions", "candidate_distances"}
    values = {}
    for field in fields(CandidateDataset):
        name = field.name
        if name == "offsets":
            values[name] = np.r_[0, np.cumsum(lengths[obs])]
        elif name == "coverage":
            values[name] = {"split": "train", "subset": "explicit_observation_subset", "trip_count": len(np.unique(dataset.observation_trip_ids[obs])), "observations": len(obs), "source_cache_sha256": TRAIN_CACHE_SHA256}
        else:
            values[name] = getattr(dataset, name)[rows if name in row_fields else obs].copy()
    return CandidateDataset(**values)


def load_full_train_dataset():
    """Load only the pinned TRAIN cache; never accepts an arbitrary input root."""
    if sha256_file(TRAIN_CACHE) != TRAIN_CACHE_SHA256:
        raise ValueError("original TRAIN cache byte identity changed")
    with np.load(TRAIN_CACHE, allow_pickle=False) as archive:
        values = {f.name: archive[f.name] for f in fields(CandidateDataset) if f.name != "coverage"}
    data = CandidateDataset(**values, coverage={"split": "train", "source_cache_sha256": TRAIN_CACHE_SHA256})
    if data.X.shape != (269463, 38) or len(data.observation_trip_ids) != 28000 or len(np.unique(data.observation_trip_ids)) != 80:
        raise ValueError("original TRAIN sampling/schema differs")
    if not np.isfinite(data.X).all() or not np.array_equal(np.unique(data.y), [0, 1]) or data.offsets[0] != 0 or data.offsets[-1] != len(data.y):
        raise ValueError("invalid TRAIN cache")
    return data


def load_train_dataset():
    """Return 64 tuning trips; the prospective 16 confirmation trips are excluded."""
    data = load_full_train_dataset()
    tune, outer = split_trip_ids(data.observation_trip_ids)
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if list(tune) != protocol["tuning_trip_ids"] or list(outer) != protocol["confirmation_trip_ids"]:
        raise ValueError("prospective TRAIN partition differs from preregistration")
    return subset_dataset(data, np.flatnonzero(np.isin(data.observation_trip_ids, tune)))


def get_folds(dataset):
    return build_group_folds(dataset, ExperimentConfig(seed=SEED, folds=3, n_jobs=THREADS))


def profile_features(X, offsets, profile):
    if profile in ("v1", "baseline38"):
        result = np.asarray(X, dtype=np.float64)
        if result.ndim != 2 or result.shape[1] != len(FEATURE_NAMES) or not np.isfinite(result).all():
            raise ValueError("invalid baseline38 feature matrix")
        return result
    from mapmatch.accuracy_features import transform_accuracy_features
    return np.asarray(transform_accuracy_features(X, offsets, profile=profile), dtype=np.float64)


def profile_names(profile):
    if profile in ("v1", "baseline38"):
        return tuple(FEATURE_NAMES)
    from mapmatch.accuracy_features import feature_names
    return tuple(feature_names(profile))


def strict_positive_probabilities(estimator, X):
    classes = np.asarray(estimator.classes_)
    if classes.ndim != 1 or len(classes) != 2 or set(classes.tolist()) != {0, 1}:
        raise ValueError("estimator must have exactly classes 0 and 1")
    raw = np.asarray(estimator.predict_proba(np.asarray(X, dtype=np.float64)), dtype=np.float64)
    if raw.shape != (len(X), 2) or not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1) or not np.allclose(raw.sum(axis=1), 1, rtol=0, atol=1e-10):
        raise ValueError("classifier probabilities invalid; tolerance is not relaxed")
    positive = raw[:, int(np.flatnonzero(classes == 1)[0])]
    return np.clip(positive, 1e-7, 1-1e-7), {"maximum_row_sum_error": float(np.max(np.abs(raw.sum(axis=1)-1))) if len(raw) else 0., "clipped_low_rows": int(np.sum(positive < 1e-7)), "clipped_high_rows": int(np.sum(positive > 1-1e-7))}


def metrics_for(dataset, probabilities):
    return candidate_classification_metrics(dataset.y, probabilities, dataset.offsets, dataset.observation_trip_ids, true_edges=dataset.reference_edges, candidate_edges=dataset.candidate_edges)


def rank_trial(result):
    m = result["oof_metrics"]
    return (m["trip_macro_top1_accuracy"], m["macro_f1"], -m["log_loss"])


def fit_estimator(estimator, X, y, weights, family, fold_index, fit_policy, bank_budget):
    fit_rows = np.arange(len(y), dtype=np.int64)
    kwargs = {}
    bank = None
    if fit_policy == "knn_bank":
        if family != "knn_distance_weighted":
            raise ValueError("KNN bank policy is exclusive to KNN")
        fit_rows = knn_reference_bank(y, weights, int(bank_budget), bank_seed(SEED, fold_index))
        bank = {"source_rows": len(y), "selected_rows": len(fit_rows), "budget": int(bank_budget), "seed": bank_seed(SEED, fold_index), "policy": "unchanged class-stratified weighted TRAIN-only bank, no replacement", "sample_weight_fit_supported": False}
    elif fit_policy == "weighted":
        kwargs = {estimator.steps[-1][0]+"__sample_weight": weights} if isinstance(estimator, Pipeline) else {"sample_weight": weights}
    else:
        raise ValueError("unknown fitting policy")
    start = time.perf_counter()
    with threadpool_limits(limits=THREADS), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X[fit_rows], y[fit_rows], **kwargs)
    return time.perf_counter()-start, sorted({str(w.message) for w in caught}), bank


def validate_folds(dataset, folds):
    expected = set(np.unique(dataset.observation_trip_ids).tolist())
    all_validation = []
    for f in folds:
        a, b = set(f["train_trip_ids"]), set(f["validation_trip_ids"])
        if a & b or a | b != expected:
            raise ValueError("whole-trip validation leakage or missing trips")
        for name, trips in (("train", a), ("validation", b)):
            obs = np.asarray(f[name+"_observations"], dtype=np.int64)
            rows = np.asarray(f[name+"_rows"], dtype=np.int64)
            expected_obs = np.flatnonzero(np.isin(dataset.observation_trip_ids, sorted(trips)))
            expected_rows = np.flatnonzero(np.repeat(np.isin(dataset.observation_trip_ids, sorted(trips)), np.diff(dataset.offsets)))
            if not np.array_equal(obs, expected_obs) or not np.array_equal(rows, expected_rows):
                raise ValueError("fold rows/observations do not exactly match declared whole trips")
        all_validation.extend(f["validation_observations"].tolist())
    if sorted(all_validation) != list(range(len(dataset.observation_trip_ids))):
        raise ValueError("validation observations must occur once across folds")


def validate_registered_trial(protocol, family, run_name, parameters, profile, folds):
    declared = protocol.get("folds")
    actual = [{k: f[k] for k in ("fold", "train_trip_ids", "validation_trip_ids")} for f in folds]
    if declared != actual:
        raise ValueError("regenerated whole-trip folds differ from preregistration")
    stem = run_name[:-7] if run_name.endswith("_screen") else run_name[:-5] if run_name.endswith("_full") else run_name
    entries = [c for c in protocol.get("family_menus", {}).get(family, []) if c["run_name"] == stem]
    if len(entries) != 1:
        raise ValueError("trial is not in the preregistered fixed menu")
    item = entries[0]
    registered_profile = "v1" if item["profile"] == "baseline38" else item["profile"]
    if item.get("family") != family or item["parameters"] != parameters or registered_profile != profile:
        raise ValueError("trial parameters, family or features differ from preregistration")
    if protocol.get("source_TRAIN_cache_sha256") != TRAIN_CACHE_SHA256:
        raise ValueError("preregistered TRAIN byte hash differs")


def run_cv_trial(family, run_name, parameters, profile, output, estimator_factory, fold_ids=None, bank_budget=60000, fit_policy="weighted"):
    if profile == "baseline38":
        profile = "v1"
    if family not in FAMILIES or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_name):
        raise ValueError("invalid experiment identity")
    output = Path(output).resolve()
    if not output.is_relative_to(EXPERIMENT_ROOT) or output == EXPERIMENT_ROOT or output.is_relative_to(PRIVATE_ROOT):
        raise ValueError("aggregate output must be an isolated accuracy experiment subdirectory")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol.get("training_authorized") is not True:
        raise ValueError("actual training not authorized in preregistered protocol")
    data = load_train_dataset()
    folds = get_folds(data)
    validate_folds(data, folds)
    validate_registered_trial(protocol, family, run_name, parameters, profile, folds)
    requested = tuple(range(3)) if fold_ids is None else tuple(int(x) for x in fold_ids)
    if not requested or len(set(requested)) != len(requested) or not set(requested).issubset({0, 1, 2}):
        raise ValueError("fold IDs must be a unique subset of 0,1,2")
    record_path = output / "cv" / (run_name+".json")
    private = PRIVATE_ROOT / output.name / run_name
    if record_path.exists() or private.exists():
        raise FileExistsError("refuse to overwrite or replay an existing trial")
    private.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    X = profile_features(data.X, data.offsets, profile)
    if X.shape != (len(data.y), len(profile_names(profile))) or not np.isfinite(X).all():
        raise ValueError("feature profile dimensions or values invalid")
    p = np.full(len(data.y), np.nan)
    fold_records = []
    source_pins = {"common": sha256_file(__file__), "factory_module": sha256_file(__import__(estimator_factory.__module__, fromlist=["__file__"]).__file__)}
    for relative in ("scripts/accuracy_linear_bayes.py", "scripts/accuracy_knn.py", "mapmatch/phase_models.py", "mapmatch/features.py", "mapmatch/learning.py", "mapmatch/experiments.py"):
        source = ROOT / relative
        if source.exists():
            source_pins[relative] = sha256_file(source)
    if profile != "v1":
        from mapmatch import accuracy_features
        source_pins["features"] = sha256_file(accuracy_features.__file__)
    try:
        for fold in folds:
            fi = fold["fold"]
            if fi not in requested:
                continue
            train_rows, val_rows = fold["train_rows"], fold["validation_rows"]
            estimator = estimator_factory(dict(parameters), SEED, THREADS)
            fit_seconds, fit_warnings, bank = fit_estimator(estimator, X[train_rows], data.y[train_rows], data.training_weights(train_rows), family, fi, fit_policy, bank_budget)
            t = time.perf_counter()
            with threadpool_limits(limits=THREADS):
                probabilities, checks = strict_positive_probabilities(estimator, X[val_rows])
            inference_seconds = time.perf_counter()-t
            p[val_rows] = probabilities
            valid = subset_dataset(data, np.asarray(fold["validation_observations"], np.int64))
            metrics = metrics_for(valid, probabilities)
            model_path = private / f"fold_{fi}.joblib"
            bundle = {"estimator": estimator, "profile": profile, "feature_names": profile_names(profile), "base_feature_names": FEATURE_NAMES, "base_feature_version": FEATURE_VERSION, "feature_version": FEATURE_VERSION if profile == "v1" else "pathsnap-candidate-v2", "family": family, "run_name": run_name, "parameters": parameters, "fold_id": fi, "fold": fi, "train_trip_ids": fold["train_trip_ids"], "validation_trip_ids": fold["validation_trip_ids"], "training_scope": "TRAIN tuning64 fold only; no confirmation/student/evaluator", "source_sha256": source_pins, "protocol_sha256": sha256_file(PROTOCOL_PATH)}
            joblib.dump(bundle, model_path, compress=3)
            fold_record = {"fold": fi, "train_trip_ids": fold["train_trip_ids"], "validation_trip_ids": fold["validation_trip_ids"], "training_seconds": fit_seconds, "inference_seconds": inference_seconds, "fit_warnings": fit_warnings, "bank": bank, "metrics": metrics, "probability_checks": checks, "private_model_sha256": sha256_file(model_path)}
            fold_records.append(fold_record)
            print(json.dumps({"event": "accuracy_cv_fold", "run_name": run_name, "family": family, "profile": profile, "fold": fi, "top1": metrics["trip_macro_top1_accuracy"], "binary_macro_f1": metrics["macro_f1"], "binary_log_loss": metrics["log_loss"], "fit_seconds": fit_seconds, "inference_seconds": inference_seconds}, allow_nan=False), flush=True)
        obs = np.sort(np.concatenate([f["validation_observations"] for f in folds if f["fold"] in requested]))
        selected = subset_dataset(data, obs)
        row_mask = np.isin(data.row_trip_ids, selected.observation_trip_ids)
        if not np.isfinite(p[row_mask]).all():
            raise ValueError("incomplete out-of-fold probabilities")
        values = {f.name: getattr(selected, f.name) for f in fields(CandidateDataset) if f.name != "coverage"}
        np.savez_compressed(private / "probabilities.npz", probabilities=p[row_mask], **{k:v for k,v in values.items() if k != "X"})
        aggregate = metrics_for(selected, p[row_mask])
        result = {"status": "complete", "created_at_utc": utc_now(), "family": family, "name": run_name, "parameters": parameters, "profile": profile, "feature_count": X.shape[1], "source_sha256": source_pins, "protocol_sha256": sha256_file(PROTOCOL_PATH), "TRAIN_source_sha256": TRAIN_CACHE_SHA256, "validation_role": "tuning64 whole-trip OOF; not outer confirmation", "fold_ids": list(requested), "full_three_fold": set(requested)=={0,1,2}, "objective_name": "out_of_fold_equal_trip_observation_top1_accuracy", "objective": aggregate["trip_macro_top1_accuracy"], "oof_metrics": aggregate, "folds": fold_records, "elapsed_seconds": time.perf_counter()-start, "training_seconds_total": sum(x["training_seconds"] for x in fold_records), "inference_seconds_total": sum(x["inference_seconds"] for x in fold_records), "private_output": private.relative_to(ROOT).as_posix(), "proxy_labels_changed": False, "student_accessed": False, "evaluator_accessed": False, "prospective_confirmation_used": False}
        for key, digest in source_pins.items():
            source_path = Path(__file__) if key == "common" else Path(__import__(estimator_factory.__module__, fromlist=["__file__"]).__file__) if key == "factory_module" else ROOT / "mapmatch/accuracy_features.py" if key == "features" else ROOT / key
            if sha256_file(source_path) != digest:
                raise ValueError("source changed during measured trial: " + key)
        if sha256_file(PROTOCOL_PATH) != result["protocol_sha256"]:
            raise ValueError("protocol changed during measured trial")
        write_new_json(record_path, result)
        return result
    except Exception as exc:
        write_new_json(private / "failure.json", {"time_utc": utc_now(), "run_name": run_name, "error_type": type(exc).__name__, "message": str(exc), "folds_completed": [f["fold"] for f in fold_records]})
        raise
