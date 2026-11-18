"""Supervised binary candidate probabilities, grouped metrics and artifact loading.

A positive row means the candidate matches a route-derived observation proxy.
The saved estimator only sees graph/GPS features. It cannot load route files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

from .candidates import Candidates
from .data import integer_array
from .features import (FEATURE_NAMES, FEATURE_VERSION, CandidateFeatureExtractor,
                       features_for_candidates, reference_observation_edges,
                       reference_observation_labels)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "phase_three" / "model.joblib"


def predict_positive_proba(model, X: np.ndarray) -> np.ndarray:
    """Binary positive-class probabilities; never assume a class-column order."""
    if not len(X):
        return np.empty(0, dtype=float)
    # GaussianNB joint likelihoods can accumulate in float32 when scaler inputs
    # are float32. Measured real rows then had sum(p)=1+1.52e-5. Promote only
    # GaussianNB scoring inputs, preserving strict guards and other estimators.
    classifier = model.steps[-1][1] if isinstance(model, Pipeline) else model
    inference_X = np.asarray(X, dtype=np.float64) if isinstance(classifier, GaussianNB) else X
    probabilities = np.asarray(model.predict_proba(inference_X), dtype=float)
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if classes.ndim != 1 or len(classes) != 2 or set(classes.tolist()) != {0, 1} or len(positive) != 1 or probabilities.shape != (len(X), 2):
        raise ValueError("candidate classifier must expose exactly binary classes 0 and 1")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or np.any(probabilities > 1.0) or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-7):
        raise ValueError("candidate classifier returned invalid binary probabilities")
    result = probabilities[:, int(positive[0])]
    return np.clip(result, 1e-7, 1.0 - 1e-7)


@dataclass
class EmissionModel:
    """Validated estimator wrapper used by the HMM and server.

    ``predict_proba`` retains ordinary binary probabilities for classification
    metrics; ``score_candidates`` normalizes positive probabilities within this
    observation to form decoder emission potentials. No Gaussian fallback.
    """

    estimator: object
    metadata: dict
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self):
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError("emission wrapper feature order differs from installed schema")
        classes = np.asarray(getattr(self.estimator, "classes_", []))
        if not hasattr(self.estimator, "predict_proba") or classes.ndim != 1 or len(classes) != 2 or set(classes.tolist()) != {0, 1}:
            raise ValueError("emission wrapper requires a fitted binary probabilistic estimator")

    @property
    def classes_(self):
        return self.estimator.classes_

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score_candidates(self, graph, track, xy, index: int, candidates: Candidates) -> np.ndarray:
        X = features_for_candidates(graph, track, xy, index, candidates)
        scores = predict_positive_proba(self.estimator, X)
        return scores / scores.sum() if len(scores) else scores

    def score_candidates_batch(self, graph, track, observation_indices, candidates: Sequence[Candidates]):
        """One concatenated predict call; batch API preferred by MapMatcher."""
        return self.score_sequence(graph, track, observation_indices, candidates)

    def score_sequence(self, graph, track, observation_indices, candidates: Sequence[Candidates]):
        X, offsets = CandidateFeatureExtractor(graph).transform(track, observation_indices, candidates)
        scores = predict_positive_proba(self.estimator, X)
        result = []
        for start, end in zip(offsets[:-1], offsets[1:]):
            row = scores[start:end]
            result.append(row / row.sum() if len(row) else row)
        return result


def _validate_bundle(bundle) -> dict:
    if not isinstance(bundle, dict) or "estimator" not in bundle:
        raise ValueError("not a PathSnap candidate model artifact")
    if tuple(bundle.get("feature_names", ())) != FEATURE_NAMES or bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError("model feature schema differs from installed feature extractor")
    if not hasattr(bundle["estimator"], "predict_proba") or not np.array_equal(np.asarray(getattr(bundle["estimator"], "classes_", [])), [0, 1]):
        raise ValueError("model must be a fitted binary probabilistic candidate classifier")
    return bundle


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """Load a *trusted local* joblib artifact (pickle files can execute code)."""
    return _validate_bundle(joblib.load(Path(path)))


def load_emission_model(path: str | Path = DEFAULT_MODEL_PATH) -> EmissionModel:
    bundle = load_model(path)
    return EmissionModel(bundle["estimator"], bundle.get("metadata", {}), tuple(bundle["feature_names"]))


def candidate_probabilities(bundle_or_estimator, graph, track, observation_indices, candidates):
    """Batched normalized emissions in the exact original candidate arc ordering."""
    if isinstance(bundle_or_estimator, EmissionModel):
        model = bundle_or_estimator
    elif isinstance(bundle_or_estimator, dict):
        bundle = _validate_bundle(bundle_or_estimator)
        model = EmissionModel(bundle["estimator"], bundle.get("metadata", {}), tuple(bundle["feature_names"]))
    else:
        model = EmissionModel(bundle_or_estimator, {})
    return model.score_sequence(graph, track, observation_indices, candidates)


def candidate_classification_metrics(y_true, probabilities, offsets, observation_trip_ids=None,
                                     true_edges=None, candidate_edges=None) -> dict:
    """Binary row metrics + honest observation top-1/misses (all observations).

    A missed target has all-zero candidate labels and counts as wrong in the
    unconditional top-1 denominator. Conditional top-1 is reported separately.
    Raw binary probabilities (not normalized emissions) supply binary log loss.
    """
    raw_y = np.asarray(y_true)
    if raw_y.ndim != 1 or raw_y.dtype.kind not in "biuf" or not np.isfinite(raw_y).all() or not np.all((raw_y == 0) | (raw_y == 1)):
        raise ValueError("candidate targets must be a binary 0/1 vector")
    y = raw_y.astype(np.int8)
    p = np.asarray(probabilities, dtype=float)
    offsets = integer_array(offsets, "offsets")
    if y.shape != p.shape or offsets.ndim != 1 or not len(offsets) or offsets[0] != 0 or offsets[-1] != len(y) or np.any(np.diff(offsets) < 0):
        raise ValueError("invalid candidate metric inputs or observation offsets")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("probabilities must be finite and in [0,1]")
    prediction = (p >= 0.5).astype(np.int8)
    n_obs = len(offsets) - 1
    hits, available, selected_rows = [], [], []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        present = bool(np.any(y[lo:hi] == 1))
        row = int(lo + np.argmax(p[lo:hi])) if hi > lo else -1
        available.append(present)
        hits.append(bool(row >= 0 and y[row] == 1))
        selected_rows.append(row)
    available = np.asarray(available, dtype=bool)
    hits = np.asarray(hits, dtype=bool)
    if len(y):
        metrics = {
            "accuracy": float(accuracy_score(y, prediction)),
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "f1": float(f1_score(y, prediction, zero_division=0)),
            "macro_precision": float(precision_score(y, prediction, average="macro", labels=[0, 1], zero_division=0)),
            "macro_recall": float(recall_score(y, prediction, average="macro", labels=[0, 1], zero_division=0)),
            "macro_f1": float(f1_score(y, prediction, average="macro", labels=[0, 1], zero_division=0)),
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
            "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1.0 - 1e-7), labels=[0, 1])),
        }
    else:
        metrics = {key: None for key in ["accuracy", "precision", "recall", "f1", "macro_precision", "macro_recall", "macro_f1", "log_loss"]}
        metrics["confusion_matrix"] = [[0, 0], [0, 0]]
    metrics.update({
        "candidate_rows": len(y), "positive_rows": int(y.sum()),
        "observations": n_obs, "candidate_present_observations": int(available.sum()),
        "candidate_miss_count": int((~available).sum()),
        "candidate_miss_rate": float(np.mean(~available)) if n_obs else None,
        "top1_accuracy": float(np.mean(hits)) if n_obs else None,
        "top1_accuracy_given_candidate": float(np.mean(hits[available])) if available.any() else None,
        "top1_correct_observations": int(hits.sum()),
        "metric_target": "route_projection_proxy_not_independent_per_GPS_ground_truth",
    })
    if observation_trip_ids is not None:
        trips = np.asarray(observation_trip_ids)
        if len(trips) != n_obs:
            raise ValueError("one trip ID required per observation")
        per_trip = []
        for trip in np.unique(trips):
            mask = trips == trip
            per_trip.append({"trip_id": str(trip), "observations": int(mask.sum()),
                             "top1_accuracy": float(np.mean(hits[mask])),
                             "candidate_miss_rate": float(np.mean(~available[mask]))})
        metrics["per_trip"] = per_trip
        metrics["trip_macro_top1_accuracy"] = float(np.mean([v["top1_accuracy"] for v in per_trip])) if per_trip else None
    if true_edges is not None and candidate_edges is not None:
        true_edges = np.asarray(true_edges, dtype=np.int64)
        candidate_edges = np.asarray(candidate_edges, dtype=np.int64)
        if true_edges.shape != (n_obs,) or candidate_edges.shape != y.shape:
            raise ValueError("reference edges need one value per observation and candidate edges one per row")
        selected_edges = np.array([candidate_edges[r] if r >= 0 else -1 for r in selected_rows], dtype=np.int64)
        if observation_trip_ids is not None:
            # Arc row IDs are GRAPH-LOCAL. Namespace by trip before pooling.
            trips = np.asarray(observation_trip_ids)
            true_classes = np.array([f"{trip}:{int(edge)}" for trip, edge in zip(trips, true_edges)])
            predicted_classes = np.array([f"{trip}:{int(edge)}" for trip, edge in zip(trips, selected_edges)])
            metrics["edge_macro_f1"] = float(f1_score(true_classes, predicted_classes, average="macro", zero_division=0)) if n_obs else None
            metrics["edge_macro_f1_class_namespace"] = "trip_id:canonical_directed_arc_id"
        else:
            metrics["edge_macro_f1"] = float(f1_score(true_edges, selected_edges, average="macro", zero_division=0)) if n_obs else None
            metrics["edge_macro_f1_class_namespace"] = "single_graph_canonical_directed_arc_id; caller must not pool graph-local IDs"
    return metrics
