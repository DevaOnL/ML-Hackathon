"""Explicit local route metrics; these are not an organizer-provided score.

Multiset overlap preserves repeated traversals; edit similarity preserves order.
Reference topological defects are reported but do not invalidate overlap scoring.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .data import Graph, integer_array, validate_route


def _route(graph: Graph, value: np.ndarray | list[int], name: str) -> np.ndarray:
    result = integer_array(value, name)
    if np.any(result < 0) or np.any(result >= len(graph.arcs)):
        raise ValueError(f"{name} contains an invalid original zero-based arc ID")
    return result


def _overlap(prediction: np.ndarray, reference: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float, float, float, float, float]:
    p, r = Counter(prediction.tolist()), Counter(reference.tolist())
    common = p & r
    if weights is None:
        matched = float(sum(common.values()))
        p_total, r_total = float(len(prediction)), float(len(reference))
    else:
        matched = float(sum(weights[e] * count for e, count in common.items()))
        p_total = float(sum(weights[e] * count for e, count in p.items()))
        r_total = float(sum(weights[e] * count for e, count in r.items()))
    if p_total == 0 and r_total == 0:
        # Empty/zero-length geometry cannot provide a length denominator.
        # Use count multiplicities rather than declare arbitrary traversals equal.
        if weights is not None:
            precision, recall, f1, *_ = _overlap(prediction, reference)
            return precision, recall, f1, matched, p_total, r_total
        precision = recall = 1.0
    else:
        precision = matched / p_total if p_total else 0.0
        recall = matched / r_total if r_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, matched, p_total, r_total


def edit_distance(first: np.ndarray | list[int], second: np.ndarray | list[int]) -> int:
    """Levenshtein distance with linear auxiliary memory and exact integer costs."""
    first, second = list(first), list(second)
    if len(first) < len(second):
        first, second = second, first
    previous = np.arange(len(second) + 1, dtype=np.int64)
    for i, a in enumerate(first, 1):
        current = np.empty_like(previous)
        current[0] = i
        for j, b in enumerate(second, 1):
            current[j] = min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (a != b))
        previous = current
    return int(previous[-1])


def score_routes(graph: Graph, prediction: np.ndarray | list[int], reference: np.ndarray | list[int]) -> dict:
    prediction = _route(graph, prediction, "prediction")
    reference = _route(graph, reference, "reference")
    ep, er, ef, common_count, pred_count, ref_count = _overlap(prediction, reference)
    lp, lr, lf, common_length, pred_length, ref_length = _overlap(prediction, reference, graph.edge_lengths)
    canonical_prediction = graph.canonical_edges[prediction]
    canonical_reference = graph.canonical_edges[reference]
    _, _, canonical_f1, *_ = _overlap(canonical_prediction, canonical_reference, graph.edge_lengths)
    edit = edit_distance(prediction, reference)
    pvalid, rvalid = validate_route(graph, prediction), validate_route(graph, reference)
    joins = max(len(prediction) - 1, 0)
    return {
        "exact_match": bool(np.array_equal(prediction, reference)),
        "edge_precision": ep, "edge_recall": er, "edge_f1": ef,
        "length_precision": lp, "length_recall": lr, "length_f1": lf,
        "duplicate_canonical_length_f1": canonical_f1,
        "edit_distance": edit,
        "edit_similarity": 1.0 - edit / max(len(prediction), len(reference), 1),
        "predicted_arc_count": len(prediction), "reference_arc_count": len(reference),
        "common_arc_count": int(common_count),
        "predicted_length_m": float(np.sum(graph.edge_lengths[prediction])),
        "reference_length_m": float(np.sum(graph.edge_lengths[reference])),
        "common_length_m": common_length,
        "length_metric_basis": "arc_count_fallback_for_zero_length_routes" if pred_length == 0 and ref_length == 0 else "meters",
        "route_continuity": 1.0 - len(pvalid["discontinuities"]) / joins if joins else 1.0,
        "prediction_connected": pvalid["valid"],
        "reference_connected": rvalid["valid"],
        "reference_discontinuity_count": len(rvalid["discontinuities"]),
        "metric_note": "Directed arc multiset overlap; sequence edit separately. Not an official organizer score.",
    }
