#!/usr/bin/env python3
"""Exact distance-weighted KNN tools for a separate TRAIN-only experiment.

No training happens on import. The estimator is an ordinary sklearn pipeline;
only the repeated probability calculation is shared across a predeclared k grid.
A maximum-k exact neighbour query is made once per query batch. Truncated prefixes
are valid nearest-neighbour sets. Exact cutoff ties are counted; by default only
those exceptional query rows are re-queried at the smaller k, reproducing the
backend tie choice of standalone sklearn. Disabling this correction explicitly
uses the maximum-k ordering for the equally-near tied subset. Zero-distance
voting is identical to sklearn: only exact matches within that prefix vote.

The lead-owned central runner uses make_knn_pipeline with the shared TRAIN
protocol for all measured fits; this module never launches duplicate trials.
Released models, live state and held-out data are untouched.
"""
from __future__ import annotations

import os
import sys

# Bound BLAS/OpenMP even when launched outside the common runner.
for _thread_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_name] = "1"
sys.dont_write_bytecode = True

from collections.abc import Iterable
from dataclasses import dataclass
import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


DEFAULT_K_VALUES = (5, 15, 35)
MAX_THREADS = 2


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validated_k_values(k_values: Iterable[int], available: int) -> tuple[int, ...]:
    values = tuple(_positive_integer(k, "k") for k in k_values)
    if not values or len(set(values)) != len(values):
        raise ValueError("k_values must be nonempty and unique")
    if max(values) > available:
        raise ValueError("k exceeds the available neighbours")
    return values


def make_knn_pipeline(*, n_neighbors: int = 35, algorithm: str = "kd_tree",
                      p: int = 2, leaf_size: int = 40, n_jobs: int = 1) -> Pipeline:
    """Construct genuine sklearn standardized, exact inverse-distance KNN.

    Fit this pipeline on the existing class-stratified weighted reference bank,
    not on all rows then retrospectively subsample. Thus the scaler sees exactly
    the same fold-local rows as the released baseline bank/scaler policy. KNN
    accepts no sample_weight; the shared existing bank policy supplies weighting.
    """
    n_neighbors = _positive_integer(n_neighbors, "n_neighbors")
    leaf_size = _positive_integer(leaf_size, "leaf_size")
    n_jobs = _positive_integer(n_jobs, "n_jobs")
    if n_jobs > MAX_THREADS:
        raise ValueError("n_jobs cannot exceed the two-thread experiment limit")
    if algorithm not in ("kd_tree", "ball_tree", "brute"):
        raise ValueError("only exact kd_tree, ball_tree, and brute queries are allowed")
    if isinstance(p, (bool, np.bool_)) or p not in (1, 2):
        raise ValueError("only Minkowski p=1 or p=2 is predeclared")
    classifier = KNeighborsClassifier(
        n_neighbors=n_neighbors, weights="distance", algorithm=algorithm,
        leaf_size=leaf_size, metric="minkowski", p=int(p), n_jobs=n_jobs,
    )
    return Pipeline([("scale", StandardScaler()), ("classifier", classifier)])


def make_knn_estimator(parameters: dict, seed: int, n_jobs: int) -> Pipeline:
    """Common-runner factory; only the shared reference bank needs the seed.

    KD/Ball/brute exact KNN and StandardScaler are deterministic for fixed row
    order. The common runner supplies seed+7919*fold to its existing bank sampler,
    not to this estimator. Exceeding the CPU budget is rejected, not hidden.
    """
    return make_knn_pipeline(**dict(parameters), n_jobs=n_jobs)


def inverse_distance_k_votes(distances: np.ndarray, indices: np.ndarray,
                             bank_labels: np.ndarray,
                             k_values: Iterable[int]) -> dict[int, np.ndarray]:
    """Positive probabilities for binary sklearn distance weighting at each k.

    The input must be the sorted result of one exact sklearn kneighbors query.
    A row with any infinite reciprocal (normally distance==0) uses the sklearn
    infinity mask, so nonzero distances get exactly zero weight in that row.
    No epsilon, squared-distance vote, class prior correction, or calibration is
    introduced. Each k only uses its own prefix, including for duplicate points.
    """
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices)
    bank_labels = np.asarray(bank_labels)
    if distances.ndim != 2 or indices.shape != distances.shape:
        raise ValueError("distances and indices must be aligned two-dimensional arrays")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("indices must be integers")
    if bank_labels.ndim != 1 or not len(bank_labels) or not np.isin(bank_labels, (0, 1)).all():
        raise ValueError("bank_labels must be a nonempty binary vector")
    if np.any(indices < 0) or np.any(indices >= len(bank_labels)):
        raise ValueError("neighbour index outside the reference bank")
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("neighbour distances must be finite and nonnegative")
    if np.any(np.diff(distances, axis=1) < 0):
        raise ValueError("neighbour distances must be sorted")
    k_values = _validated_k_values(k_values, distances.shape[1])
    with np.errstate(divide="ignore", over="ignore"):
        weights = 1.0 / distances
    infinity_mask = np.isinf(weights)
    rows_with_infinity = np.any(infinity_mask, axis=1)
    weights[rows_with_infinity] = infinity_mask[rows_with_infinity]
    labels = bank_labels[indices]
    results = {}
    for k in k_values:
        prefix_weights, prefix_labels = weights[:, :k], labels[:, :k]
        # Two class sums match sklearn predict_proba normalization.
        negative = np.sum(prefix_weights * (prefix_labels == 0), axis=1)
        positive = np.sum(prefix_weights * (prefix_labels == 1), axis=1)
        results[k] = positive / (negative + positive)
    return results


@dataclass
class KGridPrediction:
    probabilities: dict[int, np.ndarray]
    query_seconds: float
    transform_seconds: float
    vote_seconds: float
    query_calls: int
    cutoff_tie_rows: dict[int, int]
    cutoff_tie_correction_calls: int = 0
    maximum_k_boundary_ties_observed: bool = False


def predict_k_grid(estimator: Pipeline, X: np.ndarray, k_values: Iterable[int],
                   *, batch_size: int = 8192, exact_cutoff_ties: bool = True) -> KGridPrediction:
    """Reuse one exact kmax query per batch for every predeclared neighbour k.

    This does not refit or change estimator.n_neighbors. A normal model with a
    fixed k can therefore be saved or independently queried after screening.
    Internal cutoff ties (k < kmax) are counted and, by default, only affected
    rows are re-queried at k for exact standalone-sklearn backend parity. Extra
    queries are explicitly counted. Ties beyond kmax are not queried, so that
    boundary remains unknown rather than misleadingly reported as zero ties.
    """
    if not isinstance(estimator, Pipeline) or tuple(estimator.named_steps) != ("scale", "classifier"):
        raise TypeError("expected a StandardScaler/KNeighborsClassifier pipeline")
    scale, classifier = estimator.named_steps["scale"], estimator.named_steps["classifier"]
    if not isinstance(scale, StandardScaler) or not isinstance(classifier, KNeighborsClassifier):
        raise TypeError("expected genuine sklearn StandardScaler and KNeighborsClassifier")
    if classifier.weights != "distance" or classifier.metric != "minkowski" or classifier.p not in (1, 2):
        raise ValueError("expected predeclared inverse-distance Minkowski KNN")
    if classifier.algorithm not in ("kd_tree", "ball_tree", "brute"):
        raise ValueError("only exact neighbour queries are supported")
    if classifier.n_jobs not in (1, 2):
        raise ValueError("query n_jobs must be one or two")
    if getattr(classifier, "outputs_2d_", True) or not np.isin(classifier.classes_, (0, 1)).all():
        raise ValueError("expected fitted single-output binary KNN")
    bank_labels = classifier.classes_[classifier._y]
    k_values = _validated_k_values(k_values, len(bank_labels))
    kmax = max(k_values)
    batch_size = _positive_integer(batch_size, "batch_size")
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != scale.n_features_in_ or not np.isfinite(X).all():
        raise ValueError("query features must be a finite aligned matrix")
    probabilities = {k: np.empty(len(X), dtype=np.float64) for k in k_values}
    cutoff_tie_rows = {k: 0 for k in k_values if k < kmax}
    query_seconds = transform_seconds = vote_seconds = 0.0
    query_calls = cutoff_tie_correction_calls = 0
    with threadpool_limits(limits=classifier.n_jobs):
        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            began = time.perf_counter()
            transformed = scale.transform(X[start:stop])
            transform_seconds += time.perf_counter() - began
            began = time.perf_counter()
            distances, indices = classifier.kneighbors(transformed, n_neighbors=kmax, return_distance=True)
            query_seconds += time.perf_counter() - began
            query_calls += 1
            began = time.perf_counter()
            voted = inverse_distance_k_votes(distances, indices, bank_labels, k_values)
            vote_seconds += time.perf_counter() - began
            for k, values in voted.items():
                if k < kmax:
                    tied = distances[:, k - 1] == distances[:, k]
                    cutoff_tie_rows[k] += int(np.count_nonzero(tied))
                    if exact_cutoff_ties and np.any(tied):
                        began = time.perf_counter()
                        tie_distances, tie_indices = classifier.kneighbors(
                            transformed[tied], n_neighbors=k, return_distance=True,
                        )
                        query_seconds += time.perf_counter() - began
                        query_calls += 1
                        cutoff_tie_correction_calls += 1
                        began = time.perf_counter()
                        values[tied] = inverse_distance_k_votes(
                            tie_distances, tie_indices, bank_labels, (k,),
                        )[k]
                        vote_seconds += time.perf_counter() - began
                probabilities[k][start:stop] = values
    return KGridPrediction(probabilities, query_seconds, transform_seconds, vote_seconds,
                           query_calls, cutoff_tie_rows, cutoff_tie_correction_calls)
