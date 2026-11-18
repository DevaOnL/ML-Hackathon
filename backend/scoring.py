"""Postprediction, server-private evaluation with aggregate-only public results.

The supplied .route encodes a road sequence, not labels aligned to GPS times.
Observation accuracy/F1 therefore use explicit geometry-and-motion proxy labels.
No function here returns the reference edges, coordinates, or proxy label array.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from mapmatch.data import load_route
from .geometry import route_continuity

PROXY_NOTE = "Observation metrics use geometry-and-motion proxy alignment of the supplied route, not official timestamp-aligned ground truth."
PUBLIC_NOTES = frozenset({
    PROXY_NOTE,
    "No server-private reference is available; prediction is live but accuracy metrics are unavailable.",
    "Precision, recall and F1 are macro-averaged over canonical directed edge IDs; duplicate equivalent arc rows are canonicalized.",
    "No observation could be assigned a valid proxy reference label.",
    "Top-1 is classifier-only candidate argmax; log loss is candidate-normalized categorical negative log likelihood, with missing proxy candidates assigned probability 1e-7.",
    "Classifier-only scores are unavailable because the decoder supplied no candidate audit for scored observations.",
    "Server-side reference scoring failed; the returned prediction is live and unmodified. Accuracy metrics are unavailable.",
})
METRIC_KEYS = (
    "accuracy", "precision", "recall", "macro_f1", "top1_accuracy",
    "candidate_miss_rate", "log_loss", "route_continuity", "route_edge_f1",
    "route_length_f1", "sequence_similarity", "classifier_accuracy",
    "classifier_precision", "classifier_recall", "classifier_macro_f1",
    "scored_observation_fraction",
)


@dataclass
class ScoreSummary:
    metrics: dict[str, float | None]
    metrics_status: str
    metric_notes: list[str]


def empty_metrics(graph, result) -> dict[str, float | None]:
    metrics = dict.fromkeys(METRIC_KEYS)
    metrics["route_continuity"] = route_continuity(graph, np.asarray(result.route))
    return metrics


def classification_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if len(truth) != len(prediction) or not len(truth):
        raise ValueError("Scoring needs equal nonempty observation vectors.")
    labels = np.union1d(truth, prediction)
    true_counts = np.bincount(np.searchsorted(labels, truth), minlength=len(labels))
    predicted_counts = np.bincount(np.searchsorted(labels, prediction), minlength=len(labels))
    correct = truth == prediction
    hits = np.bincount(np.searchsorted(labels, truth[correct]), minlength=len(labels))
    precision = np.divide(hits, predicted_counts, out=np.zeros(len(labels)), where=predicted_counts > 0)
    recall = np.divide(hits, true_counts, out=np.zeros(len(labels)), where=true_counts > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(labels)), where=precision + recall > 0)
    return {"accuracy": float(np.mean(correct)), "precision": float(np.mean(precision)), "recall": float(np.mean(recall)), "macro_f1": float(np.mean(f1))}


def _scalar(mapping: dict, *names: str) -> float | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value)):
            return float(value)
    return None


def score_prediction(graph, track, result, model, config, reference_path: Path | None) -> ScoreSummary:
    """Invoked only after matching. model is the request-local emission audit."""
    metrics = empty_metrics(graph, result)
    if reference_path is None:
        return ScoreSummary(metrics, "unavailable", ["No server-private reference is available; prediction is live but accuracy metrics are unavailable.", PROXY_NOTE])
    # This is deliberately the first reference-file read in the request pipeline.
    reference = load_route(reference_path)
    if not len(reference) or np.any(reference < 0) or np.any(reference >= len(graph.arcs)):
        raise ValueError("The server-private reference route is empty or invalid.")
    from mapmatch.metrics import score_routes
    route_scores = score_routes(graph, result.route, reference)
    metrics["route_edge_f1"] = _scalar(route_scores, "edge_f1", "arc_f1", "f1", "count_f1", "directed_edge_f1")
    metrics["route_length_f1"] = _scalar(route_scores, "length_f1", "length_weighted_f1", "weighted_f1")
    metrics["sequence_similarity"] = _scalar(route_scores, "sequence_similarity", "edit_similarity", "normalized_edit_similarity")
    from mapmatch.features import reference_observation_edges
    indices = np.asarray(result.observation_indices, dtype=np.int64)
    truth = np.asarray(reference_observation_edges(graph, track, reference, observation_indices=indices), dtype=np.int64)
    if truth.shape != indices.shape:
        raise ValueError("Proxy reference alignment returned an invalid shape.")
    valid = (truth >= 0) & (truth < len(graph.arcs))
    metrics["scored_observation_fraction"] = float(np.mean(valid))
    notes = [PROXY_NOTE, "Precision, recall and F1 are macro-averaged over canonical directed edge IDs; duplicate equivalent arc rows are canonicalized."]
    if not np.any(valid):
        return ScoreSummary(metrics, "partial", notes + ["No observation could be assigned a valid proxy reference label."])
    truth = graph.canonical_edges[truth[valid]]
    matched = graph.canonical_edges[np.asarray(result.matched_edges, dtype=np.int64)[valid]]
    metrics.update(classification_metrics(truth, matched))
    records = getattr(model, "records", {})
    top1, probability, missed, covered_truth = [], [], [], []
    for index, label in zip(indices[valid], truth):
        record = records.get(int(index))
        if record is None:
            continue
        edges = graph.canonical_edges[record.edges]
        probabilities = record.probabilities
        top1.append(int(edges[int(np.argmax(probabilities))]))
        mass = float(np.sum(probabilities[edges == label]))
        probability.append(max(mass, 1e-7))
        missed.append(not bool(np.any(edges == label)))
        covered_truth.append(int(label))
    if covered_truth:
        classifier = classification_metrics(np.asarray(covered_truth), np.asarray(top1))
        metrics.update({f"classifier_{name}": value for name, value in classifier.items()})
        metrics["top1_accuracy"] = classifier["accuracy"]
        metrics["candidate_miss_rate"] = float(np.mean(missed))
        metrics["log_loss"] = float(-np.mean(np.log(probability)))
        notes.append("Top-1 is classifier-only candidate argmax; log loss is candidate-normalized categorical negative log likelihood, with missing proxy candidates assigned probability 1e-7.")
    else:
        notes.append("Classifier-only scores are unavailable because the decoder supplied no candidate audit for scored observations.")
    partial = len(covered_truth) != int(np.sum(valid)) or not np.all(valid) or any(metrics[key] is None for key in ("route_edge_f1", "route_length_f1", "sequence_similarity"))
    return ScoreSummary(metrics, "partial" if partial else "ok", notes)


def public_score(summary: ScoreSummary) -> dict:
    """Allowlist scalar metric keys instead of serializing arbitrary scorer output."""
    metrics = {}
    for key in METRIC_KEYS:
        value = summary.metrics.get(key)
        metrics[key] = float(value) if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value)) else None
    status = summary.metrics_status if summary.metrics_status in {"ok", "partial", "unavailable", "error"} else "error"
    notes = list(dict.fromkeys(note for note in summary.metric_notes if isinstance(note, str) and note in PUBLIC_NOTES))
    if PROXY_NOTE not in notes:
        notes.append(PROXY_NOTE)
    return {"metrics": metrics, "metrics_status": status, "metric_notes": notes, "proxy_reference": True}
