"""Reproducible PathSnap candidate experiments; no evaluator partition access.

Run with thread env vars set to 1::

    python -m mapmatch.experiments --data-root "Student folder"

All 80 train trips participate in trip-grouped CV and the final fit. CV/final
training use at most 350 deterministic observations PER TRIP, not a row-random
split. Student-test is development evaluation only, never an input to selection
or Bayesian optimization. Every saved curve is calculated from measured runs.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np
import scipy
from scipy.stats import norm
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .candidates import CandidateIndex
from .data import discover_trips, load_graph, load_route, load_track
from .features import (FEATURE_DESCRIPTIONS, FEATURE_NAMES, FEATURE_VERSION, PROXY_LABEL_POLICY,
                       CandidateFeatureExtractor, reference_observation_labels)
from .learning import candidate_classification_metrics, predict_positive_proba


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20261118
    train_observations_per_trip: int = 350
    student_observations_per_trip: int = 0  # zero explicitly means ALL
    folds: int = 3
    bayesian_trials: int = 12
    n_jobs: int = 2
    candidate_radius_m: float = 80.0
    max_radius_m: float = 1000.0
    max_candidates: int = 12
    sample_spacing_m: float = 40.0


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(value), allow_nan=False) + "\n")


def log(event, **kwargs):
    print(json.dumps(_jsonable({"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **kwargs}), allow_nan=False), flush=True)


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_observation_indices(n: int, cap: int) -> np.ndarray:
    if cap <= 0 or n <= cap:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, cap, dtype=np.int64)


@dataclass
class CandidateDataset:
    X: np.ndarray
    y: np.ndarray
    offsets: np.ndarray
    observation_trip_ids: np.ndarray
    observation_indices: np.ndarray
    reference_edges: np.ndarray
    candidate_edges: np.ndarray
    candidate_fractions: np.ndarray
    candidate_distances: np.ndarray
    proxy_distances: np.ndarray
    ambiguous: np.ndarray
    expanded: np.ndarray
    coverage: dict

    @property
    def row_trip_ids(self):
        return np.repeat(self.observation_trip_ids, np.diff(self.offsets))

    def training_weights(self, rows=None):
        # Each sampled observation has equal total fitting mass. Trips have the
        # same cap; shorter trips receive explicit inverse-count correction.
        counts = {trip: int(np.sum(self.observation_trip_ids == trip)) for trip in np.unique(self.observation_trip_ids)}
        lengths = np.diff(self.offsets)
        observation_mass = np.array([1.0 / counts[t] for t in self.observation_trip_ids])
        weights = np.repeat(observation_mass / np.maximum(lengths, 1), lengths)
        if rows is not None:
            weights = weights[rows]
        return weights / np.mean(weights)


def _input_manifest(prefixes):
    return [{"trip_id": p.name, "files": {suffix: {"bytes": p.with_suffix(suffix).stat().st_size, "sha256": _sha256(p.with_suffix(suffix))} for suffix in (".track", ".nodes", ".arcs", ".route")}} for p in prefixes]


def build_dataset(data_root: Path, split: str, config: ExperimentConfig, output: Path) -> CandidateDataset:
    if split not in ("train", "student_test"):
        raise ValueError("experiments may access train/student_test only, never evaluator data")
    start = time.perf_counter()
    prefixes = discover_trips(data_root / split)
    expected = 80 if split == "train" else 10
    if len(prefixes) != expected:
        raise ValueError(f"{split}: expected {expected} complete trips, found {len(prefixes)}")
    cap = config.train_observations_per_trip if split == "train" else config.student_observations_per_trip
    manifest = _input_manifest(prefixes)
    cache_policy = {"feature_version": FEATURE_VERSION, "proxy": PROXY_LABEL_POLICY, "config": asdict(config), "split": split, "manifest": manifest}
    fingerprint = hashlib.sha256(json.dumps(cache_policy, sort_keys=True).encode()).hexdigest()
    cache = output / "cache" / f"{split}_{fingerprint[:16]}.npz"
    coverage_path = cache.with_suffix(".json")
    if cache.is_file() and coverage_path.is_file():
        with np.load(cache, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in CandidateDataset.__dataclass_fields__ if key != "coverage"}
        coverage = json.loads(coverage_path.read_text())
        log("dataset_cache_loaded", split=split, observations=len(arrays["offsets"]) - 1, rows=len(arrays["y"]), seconds=time.perf_counter() - start)
        return CandidateDataset(**arrays, coverage=coverage)
    rows, targets, offsets = [], [], [0]
    obs_trips, obs_indices, refs, edges, fractions, distances = [], [], [], [], [], []
    proxy_distances, ambiguous, expanded_all, trips = [], [], [], []
    for prefix in prefixes:
        trip_start = time.perf_counter()
        graph = load_graph(prefix)
        track = load_track(prefix.with_suffix(".track"))
        indices = deterministic_observation_indices(len(track), cap)
        xy = graph.project(track[:, :2])
        index = CandidateIndex(graph, config.sample_spacing_m)
        candidates, expanded = [], []
        for i in indices:
            radius = config.candidate_radius_m
            current = index.query(xy[i], radius, config.max_candidates)
            was_expanded = False
            while not len(current) and radius < config.max_radius_m:
                radius = min(radius * 2, config.max_radius_m)
                was_expanded = True
                current = index.query(xy[i], radius, config.max_candidates)
            # Empty sets remain in offsets/metrics as honest misses.
            candidates.append(current)
            expanded.append(was_expanded)
        X, trip_offsets = CandidateFeatureExtractor(graph).transform(track, indices, candidates)
        # Reference is loaded AFTER feature creation and cannot influence candidates.
        label = reference_observation_labels(graph, track, load_route(prefix.with_suffix(".route")), indices)
        y = np.concatenate([(graph.canonical_edges[c.edges] == reference).astype(np.int8) for c, reference in zip(candidates, label.edges)])
        rows.append(X)
        targets.append(y)
        offsets.extend((trip_offsets[1:] + offsets[-1]).tolist())
        obs_trips.extend([prefix.name] * len(indices))
        obs_indices.extend(indices.tolist())
        refs.extend(label.edges.tolist())
        edges.extend(np.concatenate([c.edges for c in candidates]).tolist())
        fractions.extend(np.concatenate([c.fractions for c in candidates]).tolist())
        distances.extend(np.concatenate([c.distances for c in candidates]).tolist())
        proxy_distances.extend(label.distances.tolist())
        ambiguous.extend(label.ambiguous.tolist())
        expanded_all.extend(expanded)
        miss = sum(not np.any(graph.canonical_edges[c.edges] == r) for c, r in zip(candidates, label.edges))
        trip_info = {"trip_id": prefix.name, "total_observations": len(track), "selected_observations": len(indices), "selected_indices": indices.tolist(), "coverage_fraction": len(indices) / len(track), "candidate_rows": len(y), "candidate_miss_count": miss, "candidate_miss_rate": miss / len(indices), "ambiguous_proxy_observations": int(label.ambiguous.sum()), "proxy_distance_median_m": float(np.median(label.distances)), "proxy_distance_p95_m": float(np.quantile(label.distances, .95)), "expanded_candidate_observations": int(sum(expanded)), "seconds": time.perf_counter() - trip_start}
        trips.append(trip_info)
        log("trip_features", split=split, trip=prefix.name, observations=len(indices), rows=len(y), misses=miss, seconds=trip_info["seconds"])
    coverage = {"split": split, "trip_count": len(prefixes), "total_observations": sum(t["total_observations"] for t in trips), "selected_observations": len(obs_trips), "candidate_rows": sum(len(y) for y in targets), "sampling": "all observations" if cap <= 0 else f"deterministic np.linspace first-to-last, maximum {cap} per trip; no label-dependent sampling", "maximum_observations_per_trip": cap or None, "trips": trips, "input_manifest": manifest, "fingerprint": fingerprint, "seconds": time.perf_counter() - start}
    dataset = CandidateDataset(np.concatenate(rows), np.concatenate(targets), np.asarray(offsets, np.int64), np.asarray(obs_trips, dtype="U32"), np.asarray(obs_indices, np.int64), np.asarray(refs, np.int64), np.asarray(edges, np.int64), np.asarray(fractions), np.asarray(distances), np.asarray(proxy_distances), np.asarray(ambiguous, bool), np.asarray(expanded_all, bool), coverage)
    if dataset.offsets[-1] != len(dataset.X) or not np.array_equal(np.unique(dataset.y), [0, 1]):
        raise ValueError("invalid assembled supervised dataset")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{key: getattr(dataset, key) for key in CandidateDataset.__dataclass_fields__ if key != "coverage"})
    save_json(coverage_path, coverage)
    log("dataset_ready", split=split, observations=len(obs_trips), candidate_rows=len(dataset.y), seconds=time.perf_counter() - start)
    return dataset


MODEL_FAMILIES = {
    "logistic_regression": "scaled linear discriminative logistic regression",
    "gaussian_nb": "scaled conditional-independent Gaussian generative Bayes",
    "random_forest": "bagged randomized decision trees",
    "extra_trees": "extremely randomized decision-tree ensemble",
    "hist_gradient_boosting": "histogram gradient-boosted decision trees",
}


def default_parameters():
    return {
        "logistic_regression": {"C": 1.0, "max_iter": 500, "solver": "lbfgs"},
        "gaussian_nb": {"var_smoothing": 1e-7},
        "random_forest": {"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 4, "max_features": .7},
        "extra_trees": {"n_estimators": 100, "max_depth": 22, "min_samples_leaf": 3, "max_features": .8},
        "hist_gradient_boosting": {"max_iter": 120, "learning_rate": .09, "max_leaf_nodes": 23, "min_samples_leaf": 40, "l2_regularization": .5},
    }


def make_estimator(name: str, parameters: dict, config: ExperimentConfig):
    if name == "logistic_regression":
        estimator = LogisticRegression(**parameters, random_state=config.seed)
    elif name == "gaussian_nb":
        estimator = GaussianNB(**parameters)
    elif name == "random_forest":
        estimator = RandomForestClassifier(**parameters, random_state=config.seed, n_jobs=config.n_jobs)
    elif name == "extra_trees":
        estimator = ExtraTreesClassifier(**parameters, random_state=config.seed, n_jobs=config.n_jobs)
    elif name == "hist_gradient_boosting":
        # Disable sklearn random holdout early stopping: validation must remain
        # exclusively group-separated. Iterations are a searched hyperparameter.
        estimator = HistGradientBoostingClassifier(**parameters, random_state=config.seed, early_stopping=False)
    else:
        raise ValueError(f"unknown model {name}")
    # Learned scaling exists only for the models that need it, inside each CV fit.
    if name in ("logistic_regression", "gaussian_nb"):
        return Pipeline([("scale", StandardScaler()), ("classifier", estimator)])
    return estimator


def _fit(estimator, X, y, weights):
    kwargs = {"classifier__sample_weight": weights} if isinstance(estimator, Pipeline) else {"sample_weight": weights}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X, y, **kwargs)
    return [str(w.message) for w in caught]


def build_group_folds(dataset: CandidateDataset, config: ExperimentConfig):
    if config.folds not in (3, 5):
        raise ValueError("use 3 or 5 trip-grouped folds")
    splitter = GroupKFold(n_splits=config.folds)
    groups = dataset.observation_trip_ids
    folds = []
    for fold, (train_obs, validation_obs) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        train_trips, validation_trips = np.unique(groups[train_obs]), np.unique(groups[validation_obs])
        if set(train_trips) & set(validation_trips):
            raise AssertionError("trip leakage in cross-validation")
        train_mask = np.zeros(len(groups), bool)
        validation_mask = np.zeros(len(groups), bool)
        train_mask[train_obs], validation_mask[validation_obs] = True, True
        train_rows = np.flatnonzero(np.repeat(train_mask, np.diff(dataset.offsets)))
        validation_rows = np.flatnonzero(np.repeat(validation_mask, np.diff(dataset.offsets)))
        folds.append({"fold": fold, "train_observations": train_obs, "validation_observations": validation_obs, "train_rows": train_rows, "validation_rows": validation_rows, "train_trip_ids": train_trips.tolist(), "validation_trip_ids": validation_trips.tolist()})
    return folds


def _metrics(dataset: CandidateDataset, p, observations=None):
    if observations is None:
        return candidate_classification_metrics(dataset.y, p, dataset.offsets, dataset.observation_trip_ids)
    observations = np.asarray(observations, np.int64)
    mask = np.zeros(len(dataset.offsets) - 1, dtype=bool)
    mask[observations] = True
    rows = np.flatnonzero(np.repeat(mask, np.diff(dataset.offsets)))
    offsets = np.r_[0, np.cumsum(np.diff(dataset.offsets)[observations])]
    return candidate_classification_metrics(dataset.y[rows], p, offsets, dataset.observation_trip_ids[observations])


def cross_validate_model(name, parameters, dataset, folds, config, output, run_name, save_predictions=True):
    t0 = time.perf_counter()
    oof = np.full(len(dataset.y), np.nan)
    fold_results = []
    for fold in folds:
        model = make_estimator(name, parameters, config)
        train, val = fold["train_rows"], fold["validation_rows"]
        t = time.perf_counter()
        fit_warnings = _fit(model, dataset.X[train], dataset.y[train], dataset.training_weights(train))
        training_seconds = time.perf_counter() - t
        t = time.perf_counter()
        p = predict_positive_proba(model, dataset.X[val])
        inference_seconds = time.perf_counter() - t
        oof[val] = p
        metrics = _metrics(dataset, p, fold["validation_observations"])
        record = {"fold": fold["fold"], "train_trip_ids": fold["train_trip_ids"], "validation_trip_ids": fold["validation_trip_ids"], "train_rows": len(train), "validation_rows": len(val), "training_seconds": training_seconds, "inference_seconds": inference_seconds, "warnings": fit_warnings, "metrics": metrics}
        fold_results.append(record)
        log("cv_fold", run_name=run_name, fold=fold["fold"], training_seconds=training_seconds, top1=metrics["top1_accuracy"], macro_f1=metrics["macro_f1"], log_loss=metrics["log_loss"])
    if not np.isfinite(oof).all():
        raise AssertionError("missing out-of-fold probabilities")
    aggregate = _metrics(dataset, oof)
    # This is the selection objective; each entire held-out trip gets equal mass.
    objective = aggregate["trip_macro_top1_accuracy"]
    result = {"name": run_name, "model_name": name, "family": MODEL_FAMILIES[name], "parameters": parameters, "oof_metrics": aggregate, "folds": fold_results, "objective": objective, "objective_name": "out_of_fold_trip_macro_top1_accuracy", "macro_f1_fold_mean": float(np.mean([f["metrics"]["macro_f1"] for f in fold_results])), "macro_f1_fold_std": float(np.std([f["metrics"]["macro_f1"] for f in fold_results])), "top1_fold_mean": float(np.mean([f["metrics"]["top1_accuracy"] for f in fold_results])), "top1_fold_std": float(np.std([f["metrics"]["top1_accuracy"] for f in fold_results])), "training_seconds_total": sum(f["training_seconds"] for f in fold_results), "inference_seconds_total": sum(f["inference_seconds"] for f in fold_results), "elapsed_seconds": time.perf_counter() - t0}
    save_json(output / "cv" / f"{run_name}.json", result)
    if save_predictions:
        path = output / "predictions" / f"{run_name}_oof.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, probabilities=oof, y=dataset.y, offsets=dataset.offsets, observation_trip_ids=dataset.observation_trip_ids, observation_indices=dataset.observation_indices, candidate_edges=dataset.candidate_edges, reference_edges=dataset.reference_edges)
    return result


# kind, lower, upper; dimensions are normalized to [0,1] for GP modelling.
SEARCH_SPACES = {
    "logistic_regression": {"C": ("log", .001, 100.0)},
    "gaussian_nb": {"var_smoothing": ("log", 1e-11, 1e-3)},
    "random_forest": {"n_estimators": ("int", 60, 160), "max_depth": ("int", 10, 28), "min_samples_leaf": ("logint", 2, 30), "max_features": ("float", .35, 1.0)},
    "extra_trees": {"n_estimators": ("int", 60, 160), "max_depth": ("int", 10, 28), "min_samples_leaf": ("logint", 2, 30), "max_features": ("float", .35, 1.0)},
    "hist_gradient_boosting": {"learning_rate": ("log", .025, .2), "max_iter": ("int", 70, 200), "max_leaf_nodes": ("int", 7, 47), "min_samples_leaf": ("logint", 15, 160), "l2_regularization": ("log", .01, 20.0)},
}


def encode_parameters(parameters, space):
    values = []
    for key, (kind, lo, hi) in space.items():
        x = float(parameters[key])
        if kind in ("log", "logint"):
            values.append((np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo)))
        else:
            values.append((x - lo) / (hi - lo))
    return np.asarray(values)


def decode_parameters(vector, base, space):
    result = dict(base)
    for x, (key, (kind, lo, hi)) in zip(vector, space.items()):
        v = float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo)))) if kind in ("log", "logint") else float(lo + x * (hi - lo))
        result[key] = int(round(v)) if kind in ("int", "logint") else v
    return result


def expected_improvement(mean, std, incumbent, exploration=.0003):
    improvement = np.asarray(mean) - incumbent - exploration
    sigma = np.asarray(std)
    z = np.divide(improvement, sigma, out=np.zeros_like(improvement), where=sigma > 1e-12)
    return np.where(sigma > 1e-12, improvement * norm.cdf(z) + sigma * norm.pdf(z), 0.0)


def bayesian_optimize(name, baseline, dataset, folds, config, output):
    if config.bayesian_trials < 5:
        raise ValueError("Bayesian optimization needs >=5 trials including prior")
    start = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    space, base = SEARCH_SPACES[name], baseline["parameters"]
    vectors = [encode_parameters(base, space)]
    results = [baseline]
    records = [{"trial": 0, "proposal": "measured_baseline_prior", "parameters": base, "objective": baseline["objective"], "cv_macro_f1": baseline["oof_metrics"]["macro_f1"], "cv_log_loss": baseline["oof_metrics"]["log_loss"], "cv_file": "cv/" + baseline["name"] + ".json", "cv_seconds": baseline["elapsed_seconds"], "reused_actual_baseline": True}]
    trial_path = output / "bayesian_trials.jsonl"
    # A run is reproducible; reset only its own trial log, never append stale runs.
    trial_path.write_text("", encoding="utf-8")
    append_json(trial_path, records[0])
    seen = {json.dumps(base, sort_keys=True)}
    for trial in range(1, config.bayesian_trials):
        proposal_start = time.perf_counter()
        acquisition = {}
        if trial < 4:
            vector = rng.uniform(size=len(space))
            proposal = "seeded_initial_design"
        else:
            # Genuine model-based Bayesian optimization: a probabilistic GP
            # surrogate is conditioned on measured prior CV objectives, then
            # selects the candidate maximizing analytic expected improvement.
            dim = len(space)
            kernel = ConstantKernel(1.0, (.05, 20.0)) * Matern(length_scale=np.full(dim, .35), length_scale_bounds=(.03, 3.0), nu=2.5) + WhiteKernel(noise_level=.001, noise_level_bounds=(1e-6, .1))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=config.seed + trial, n_restarts_optimizer=1, alpha=1e-8)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                gp.fit(np.asarray(vectors), np.asarray([r["objective"] for r in results]))
            pool = np.vstack([rng.uniform(size=(2048, dim)), np.clip(vectors[int(np.argmax([r["objective"] for r in results]))] + rng.normal(scale=.12, size=(1024, dim)), 0.0, 1.0)])
            mean, std = gp.predict(pool, return_std=True)
            ei = expected_improvement(mean, std, max(r["objective"] for r in results))
            chosen = None
            for pos in np.argsort(ei)[::-1]:
                candidate_parameters = decode_parameters(pool[pos], base, space)
                if json.dumps(candidate_parameters, sort_keys=True) not in seen:
                    chosen = int(pos)
                    break
            if chosen is None:
                raise RuntimeError("Bayesian search exhausted unique parameter proposals")
            vector = pool[chosen]
            proposal = "gaussian_process_expected_improvement"
            acquisition = {"predicted_objective_mean": float(mean[chosen]), "predicted_objective_std": float(std[chosen]), "expected_improvement": float(ei[chosen]), "exploration_xi": .0003, "surrogate_kernel": str(gp.kernel_), "surrogate_training_points": len(results), "surrogate_warnings": [str(w.message) for w in caught], "proposal_pool_size": len(pool)}
        parameters = decode_parameters(vector, base, space)
        while json.dumps(parameters, sort_keys=True) in seen:
            vector = rng.uniform(size=len(space))
            parameters = decode_parameters(vector, base, space)
        seen.add(json.dumps(parameters, sort_keys=True))
        proposal_seconds = time.perf_counter() - proposal_start
        run_name = f"bayes_{name}_{trial:02d}"
        result = cross_validate_model(name, parameters, dataset, folds, config, output, run_name, save_predictions=False)
        vectors.append(encode_parameters(parameters, space))
        results.append(result)
        record = {"trial": trial, "proposal": proposal, "parameters": parameters, "objective": result["objective"], "cv_macro_f1": result["oof_metrics"]["macro_f1"], "cv_log_loss": result["oof_metrics"]["log_loss"], "cv_file": f"cv/{run_name}.json", "cv_seconds": result["elapsed_seconds"], "proposal_seconds": proposal_seconds, **acquisition}
        records.append(record)
        append_json(trial_path, record)
        log("bayesian_trial", **record)
    winner = max(results, key=lambda r: (r["objective"], r["oof_metrics"]["macro_f1"], -r["oof_metrics"]["log_loss"]))
    summary = {"method": "GaussianProcessRegressor Matern-5/2 surrogate + analytic expected improvement", "model_name": name, "search_space": space, "objective": "maximize out-of-fold equal-trip mean observation top1 proxy accuracy; fixed 3/5-fold GroupKFold on all 80 TRAIN trips", "selection_data": "TRAIN only; student_test metrics not loaded during HPO", "trials_including_measured_prior": len(records), "model_based_trials": sum(r["proposal"] == "gaussian_process_expected_improvement" for r in records), "best_cv_run_name": winner["name"], "best_parameters": winner["parameters"], "best_objective": winner["objective"], "baseline_objective": baseline["objective"], "seconds": time.perf_counter() - start, "trials": records}
    save_json(output / "bayesian_optimization.json", summary)
    return winner, summary


def fit_and_evaluate(name, parameters, train, student, config, output, artifacts, run_name):
    estimator = make_estimator(name, parameters, config)
    t = time.perf_counter()
    fit_warnings = _fit(estimator, train.X, train.y, train.training_weights())
    fit_seconds = time.perf_counter() - t
    t = time.perf_counter()
    train_p = predict_positive_proba(estimator, train.X)
    train_inference_seconds = time.perf_counter() - t
    t = time.perf_counter()
    student_p = predict_positive_proba(estimator, student.X)
    student_inference_seconds = time.perf_counter() - t
    result = {"run_name": run_name, "model_name": name, "parameters": parameters, "training_seconds": fit_seconds, "train_inference_seconds": train_inference_seconds, "student_inference_seconds": student_inference_seconds, "student_inference_microseconds_per_observation": 1e6 * student_inference_seconds / (len(student.offsets) - 1), "warnings": fit_warnings, "train_resubstitution_metrics": _metrics(train, train_p), "student_metrics": _metrics(student, student_p)}
    save_json(output / "evaluation" / f"{run_name}.json", result)
    prediction_path = output / "predictions" / f"{run_name}_student.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_path, probabilities=student_p, y=student.y, offsets=student.offsets, observation_trip_ids=student.observation_trip_ids, observation_indices=student.observation_indices, candidate_edges=student.candidate_edges, reference_edges=student.reference_edges)
    (artifacts / "baselines").mkdir(parents=True, exist_ok=True)
    # Actual fitted models retained, not only comparison numbers.
    joblib.dump({"estimator": estimator, "feature_names": list(FEATURE_NAMES), "feature_version": FEATURE_VERSION, "metadata": {"model_name": name, "run_name": run_name, "seed": config.seed, "parameters": parameters, "training_trip_ids": np.unique(train.observation_trip_ids).tolist()}}, artifacts / "baselines" / f"{run_name}.joblib", compress=3)
    log("full_fit_student_evaluation", run_name=run_name, training_seconds=fit_seconds, student_top1=result["student_metrics"]["top1_accuracy"], student_macro_f1=result["student_metrics"]["macro_f1"], student_log_loss=result["student_metrics"]["log_loss"])
    return estimator, result


def write_comparison(cvs, evaluations, output):
    rows = []
    for key, result in cvs.items():
        evaluation = evaluations[key]
        cv, student = result["oof_metrics"], evaluation["student_metrics"]
        rows.append({"model": key, "family": result["family"], "parameters": json.dumps(result["parameters"], sort_keys=True), "cv_macro_f1": cv["macro_f1"], "cv_macro_f1_fold_mean": result["macro_f1_fold_mean"], "cv_macro_f1_fold_std": result["macro_f1_fold_std"], "cv_top1_accuracy": cv["top1_accuracy"], "cv_trip_macro_top1_accuracy": cv["trip_macro_top1_accuracy"], "cv_log_loss": cv["log_loss"], "cv_training_seconds": result["training_seconds_total"], "cv_inference_seconds": result["inference_seconds_total"], "student_accuracy": student["accuracy"], "student_precision": student["precision"], "student_recall": student["recall"], "student_macro_f1": student["macro_f1"], "student_top1_accuracy": student["top1_accuracy"], "student_trip_macro_top1_accuracy": student["trip_macro_top1_accuracy"], "student_log_loss": student["log_loss"], "student_candidate_miss_rate": student["candidate_miss_rate"], "fit_all80_seconds": evaluation["training_seconds"], "student_inference_seconds": evaluation["student_inference_seconds"], "route_continuity": "N/A - classifier independent points; HMM ablation is evaluated separately by route matcher"})
    with (output / "model_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def make_plots(rows, evaluations, bayes, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    labels = [r["model"].replace("hist_gradient_boosting", "HistGB").replace("logistic_regression", "Logistic").replace("gaussian_nb", "GaussianNB").replace("random_forest", "RandomForest").replace("extra_trees", "ExtraTrees") for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, [r["cv_macro_f1_fold_mean"] for r in rows], yerr=[r["cv_macro_f1_fold_std"] for r in rows], capsize=4, color="#286b93")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set(ylabel="Candidate binary macro-F1 (mean +/- fold std)", title="Trip-grouped cross-validation: all 80 TRAIN trips", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(plots / "cv_macro_f1.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - .18, [r["student_top1_accuracy"] for r in rows], .36, label="Observation top-1 (misses included)")
    ax.bar(x + .18, [r["student_macro_f1"] for r in rows], .36, label="Candidate macro-F1")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set(ylabel="Score against route-projection proxy", title="Student-test10: development evaluation, not evaluator-test", ylim=(0, 1)); ax.legend()
    fig.tight_layout(); fig.savefig(plots / "student_top1_macro_f1.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - .18, [r["cv_log_loss"] for r in rows], .36, label="TRAIN out-of-fold")
    ax.bar(x + .18, [r["student_log_loss"] for r in rows], .36, label="Student-test development")
    ax.set_xticks(x, labels, rotation=20, ha="right"); ax.set(ylabel="Binary candidate log loss (lower is better)", title="Probability quality: unnormalized binary probabilities"); ax.legend()
    fig.tight_layout(); fig.savefig(plots / "log_loss.png", dpi=160); plt.close(fig)
    trial_x = [r["trial"] for r in bayes["trials"]]; scores = np.array([r["objective"] for r in bayes["trials"]])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(trial_x, scores, "o-", label="Measured grouped-CV objective"); ax.plot(trial_x, np.maximum.accumulate(scores), "--", label="Best measured so far")
    ax.set(xlabel="Trial (0 = measured prior)", ylabel="Equal-trip top-1 proxy accuracy", title="Actual Gaussian-process expected-improvement Bayesian search"); ax.legend()
    fig.tight_layout(); fig.savefig(plots / "bayesian_optimization.png", dpi=160); plt.close(fig)
    for key, evaluation in evaluations.items():
        cm = np.array(evaluation["student_metrics"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4)); image = ax.imshow(cm, cmap="Blues")
        for (i, j), count in np.ndenumerate(cm):
            ax.text(j, i, str(count), ha="center", va="center", color="white" if count > cm.max() * .5 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Other", "Proxy match"], yticklabels=["Other", "Proxy match"], xlabel="Predicted", ylabel="Route-projection proxy", title=key + " (student-test)"); fig.colorbar(image, ax=ax)
        fig.tight_layout(); fig.savefig(plots / f"confusion_{key}.png", dpi=150); plt.close(fig)


def run_experiments(data_root: Path, output: Path, artifacts: Path, config: ExperimentConfig):
    if not 1 <= config.n_jobs <= 4:
        raise ValueError("resource cap n_jobs must be between 1 and 4")
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True); artifacts.mkdir(parents=True, exist_ok=True)
    versions = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__, "platform": platform.platform()}
    save_json(output / "run_config.json", {"config": asdict(config), "data_root": data_root.resolve(), "output": output.resolve(), "versions": versions, "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}, "started_utc": datetime.now(timezone.utc).isoformat(), "selection_protocol": "baseline selection and HPO use TRAIN grouped CV only; student_test loaded after all selection"})
    with threadpool_limits(limits=1):
        train = build_dataset(data_root, "train", config, output)
        save_json(output / "coverage.json", {"train": train.coverage})
        folds = build_group_folds(train, config)
        save_json(output / "cv_folds.json", [{k: v for k, v in fold.items() if k in ("fold", "train_trip_ids", "validation_trip_ids")} | {"train_observations": len(fold["train_observations"]), "validation_observations": len(fold["validation_observations"]), "train_candidate_rows": len(fold["train_rows"]), "validation_candidate_rows": len(fold["validation_rows"])} for fold in folds])
        cvs = {}
        for name, parameters in default_parameters().items():
            cvs[name] = cross_validate_model(name, parameters, train, folds, config, output, name)
        chosen = max(cvs, key=lambda name: (cvs[name]["objective"], cvs[name]["oof_metrics"]["macro_f1"], -cvs[name]["oof_metrics"]["log_loss"]))
        save_json(output / "baseline_selection.json", {"selected_model": chosen, "objective": "out_of_fold_trip_macro_top1_accuracy", "scores": {name: r["objective"] for name, r in cvs.items()}, "student_test_consulted": False})
        log("baseline_selected", model=chosen, objective=cvs[chosen]["objective"])
        winner, bayes = bayesian_optimize(chosen, cvs[chosen], train, folds, config, output)
        tuned_name = chosen + "_tuned"
        cvs[tuned_name] = dict(winner, name=tuned_name)
        # Development labels are first loaded here, AFTER final selection/HPO.
        student = build_dataset(data_root, "student_test", config, output)
        save_json(output / "coverage.json", {"train": train.coverage, "student_test": student.coverage})
        evaluations = {}; final_estimator = None
        for run_name, result in cvs.items():
            estimator, evaluation = fit_and_evaluate(result["model_name"], result["parameters"], train, student, config, output, artifacts, run_name)
            evaluations[run_name] = evaluation
            if run_name == tuned_name:
                final_estimator = estimator
        rows = write_comparison(cvs, evaluations, output)
        metadata = {"artifact_format": "pathsnap-supervised-candidate-v1", "model_name": chosen, "run_name": tuned_name, "family": MODEL_FAMILIES[chosen], "seed": config.seed, "parameters": winner["parameters"], "feature_names": list(FEATURE_NAMES), "feature_version": FEATURE_VERSION, "training_trip_ids": np.unique(train.observation_trip_ids).tolist(), "training_trip_count": len(np.unique(train.observation_trip_ids)), "training_observations": len(train.offsets) - 1, "training_candidate_rows": len(train.y), "training_sampling": train.coverage["sampling"], "fit_data": "TRAIN only; all80 trips; no student_test fitting", "preprocessing": "StandardScaler inside fold-local Pipeline" if chosen in ("logistic_regression", "gaussian_nb") else "No learned preprocessing; numerical features constructed deterministically", "training_weighting": "inverse candidate count per observation and inverse sampled observation count per trip, normalized to mean1", "class_threshold": .5, "emission_policy": "clip binary positive probabilities to [1e-7,1-1e-7], normalize within candidate set; graph/GPS only", "candidate_config": {key: getattr(config, key) for key in ("candidate_radius_m", "max_radius_m", "max_candidates", "sample_spacing_m")}, "proxy_label_policy": PROXY_LABEL_POLICY, "metrics": {"cv": winner["oof_metrics"], "student_test_development": evaluations[tuned_name]["student_metrics"]}, "timing": {"final_training_seconds": evaluations[tuned_name]["training_seconds"], "student_inference_seconds": evaluations[tuned_name]["student_inference_seconds"], "bayesian_seconds": bayes["seconds"]}, "selection": {"data": "TRAIN GroupKFold only", "objective": "equal-trip mean out-of-fold observation top1 proxy accuracy", "cv_folds": config.folds, "bayesian_trials": config.bayesian_trials, "model_based_trials": bayes["model_based_trials"], "student_test_used_for_selection": False, "evaluator_data_used": False}, "versions": versions, "coverage_file": str(output / "coverage.json"), "experiment_results_file": str(output / "results.json"), "created_utc": datetime.now(timezone.utc).isoformat()}
        schema = {"version": FEATURE_VERSION, "dtype": "float32", "row_order": "observation order, then unchanged Candidates.edges/fractions/distances order; IDs are persisted separately, never numeric feature values", "reference_route_used_as_features": False, "features": [{"index": i, "name": name, "description": FEATURE_DESCRIPTIONS[name]} for i, name in enumerate(FEATURE_NAMES)]}
        save_json(artifacts / "feature_schema.json", schema)
        save_json(artifacts / "model_metadata.json", metadata)
        bundle = {"estimator": final_estimator, "feature_names": list(FEATURE_NAMES), "feature_version": FEATURE_VERSION, "metadata": metadata}
        model_path = artifacts / "model.joblib"
        joblib.dump(bundle, model_path, compress=3)
        save_json(artifacts / "artifact_manifest.json", {"model": str(model_path), "sha256": _sha256(model_path), "bytes": model_path.stat().st_size, "feature_schema": "feature_schema.json", "metadata": "model_metadata.json"})
        results = {"config": asdict(config), "versions": versions, "target_policy": PROXY_LABEL_POLICY, "selection_model": chosen, "final_run_name": tuned_name, "cv": cvs, "evaluations": evaluations, "comparison": rows, "bayesian": bayes, "coverage_summary": {"train_trips": train.coverage["trip_count"], "train_observations": train.coverage["selected_observations"], "train_total_observations": train.coverage["total_observations"], "student_trips": student.coverage["trip_count"], "student_observations": student.coverage["selected_observations"], "student_total_observations": student.coverage["total_observations"]}, "elapsed_seconds": time.perf_counter() - started, "evaluator_testing": "not present and never accessed; no final-evaluator metrics claimed", "route_continuity": "classifier-independent candidate metrics only here; classifier/HMM ablation belongs to route matcher experiment"}
        save_json(output / "results.json", results)
        make_plots(rows, evaluations, bayes, output)
    log("experiments_complete", model=chosen, artifact=str(model_path), seconds=time.perf_counter() - started, final_cv_top1=winner["oof_metrics"]["top1_accuracy"], final_student_top1=evaluations[tuned_name]["student_metrics"]["top1_accuracy"])
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Student folder"))
    parser.add_argument("--output", type=Path, default=Path("experiments"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--train-observations-per-trip", type=int, default=350)
    parser.add_argument("--student-observations-per-trip", type=int, default=0)
    parser.add_argument("--folds", type=int, choices=(3, 5), default=3)
    parser.add_argument("--bayesian-trials", type=int, default=12)
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args(argv)
    config = ExperimentConfig(seed=args.seed, train_observations_per_trip=args.train_observations_per_trip, student_observations_per_trip=args.student_observations_per_trip, folds=args.folds, bayesian_trials=args.bayesian_trials, n_jobs=args.n_jobs)
    if config.train_observations_per_trip < 1:
        parser.error("training requires an explicit positive trip-balanced sample cap")
    run_experiments(args.data_root, args.output, args.artifacts, config)


if __name__ == "__main__":
    main()
