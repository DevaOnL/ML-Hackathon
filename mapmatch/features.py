"""Leakage-safe candidate geometry/motion features and scoring-only proxy labels.

Feature extraction accepts a road graph and noisy observations, NEVER a reference
route. The separate ``reference_*`` functions are for supervised target creation
and post-inference scoring only: .route contains a route, not per-GPS truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .candidates import Candidates
from .data import Graph, integer_array, validate_track

FEATURE_VERSION = "pathsnap-candidate-v1"
# Ordering is an inference contract, persisted with every trained artifact.
FEATURE_DESCRIPTIONS = {
    "distance_m": "Exact observation-to-directed-segment distance in projected metres",
    "log1p_distance_m": "log(1 + lateral distance in metres)",
    "distance_squared_10m": "Squared lateral distance divided by 100, clipped at 10000",
    "distance_minus_nearest_m": "Lateral distance minus nearest candidate distance",
    "distance_ratio_nearest": "(distance+1)/(nearest distance+1), clipped at 100",
    "strict_distance_rank": "Fraction of other candidates strictly closer (ties share a rank)",
    "candidate_count": "Number of candidates at this observation",
    "nearest_distinct_gap_m": "Distance gap between nearest and next distinct lateral distance",
    "fraction": "Clipped projection fraction along the directed arc",
    "distance_to_endpoint_m": "Along-segment distance to the nearer endpoint",
    "log1p_edge_length_m": "log(1 + segment length in metres)",
    "projection_is_interior": "Projection is strictly inside the segment",
    "heading_cos_short": "Signed arc alignment with centred +/-1 observation GPS motion",
    "heading_cos_medium": "Signed arc alignment with centred +/-3 observation GPS motion",
    "heading_cos_long": "Signed arc alignment with centred +/-8 observation GPS motion",
    "heading_abs_cos_medium": "Undirected medium-window alignment",
    "heading_reliability_short": "Short-window displacement / 12m clipped to [0,1]",
    "heading_reliability_medium": "Medium-window displacement / 12m clipped to [0,1]",
    "heading_reliability_long": "Long-window displacement / 12m clipped to [0,1]",
    "reliable_direction_mismatch": "(1 - signed medium alignment) times heading reliability",
    "previous_lateral_distance_m": "Distance from GPS observation three rows earlier to this segment",
    "next_lateral_distance_m": "Distance from GPS observation three rows later to this segment",
    "neighbour_mean_lateral_distance_m": "Mean of previous and next lateral distances",
    "neighbour_max_lateral_distance_m": "Maximum of previous and next lateral distances",
    "neighbour_projected_speed_mps": "Signed speed of neighbouring GPS projections along this segment",
    "motion_along_arc_mps": "Medium GPS velocity dotted with unit directed arc vector; duration floored at 1s",
    "log1p_speed_short_mps": "log(1 + short centred GPS speed); duration floored at 1s",
    "log1p_speed_medium_mps": "log(1 + medium centred GPS speed); duration floored at 1s",
    "speed_change_mps": "Forward minus backward local GPS speed, clipped to +/-100m/s",
    "gps_turn_cos": "Cosine between incoming and outgoing +/-3 GPS motion vectors",
    "log1p_previous_gap_s": "log(1 + time since previous GPS observation)",
    "log1p_next_gap_s": "log(1 + time to next GPS observation)",
    "low_motion": "Medium-window GPS displacement below 3 metres",
    "log1p_tail_indegree": "Canonical directed incoming degree of arc tail",
    "log1p_tail_outdegree": "Canonical directed outgoing degree of arc tail",
    "log1p_head_indegree": "Canonical directed incoming degree of arc head",
    "log1p_head_outdegree": "Canonical directed outgoing degree of arc head",
    "edge_is_zero_length": "Segment length below one micrometre",
}
FEATURE_NAMES = tuple(FEATURE_DESCRIPTIONS)


def _window(track: np.ndarray, xy: np.ndarray, index: int, half_width: int):
    # Do not estimate heading across a dropout wider than 20 seconds.
    lo, hi = max(0, index - half_width), min(len(track) - 1, index + half_width)
    while lo < index and track[index, 2] - track[lo, 2] > 20.0:
        lo += 1
    while hi > index and track[hi, 2] - track[index, 2] > 20.0:
        hi -= 1
    vector = xy[hi] - xy[lo]
    displacement = float(np.linalg.norm(vector))
    duration = max(float(track[hi, 2] - track[lo, 2]), 1.0)
    direction = vector / displacement if displacement > 1e-9 else np.zeros(2)
    return lo, hi, vector, direction, displacement, duration


def _project_to_segments(point, starts, vectors, squared_lengths):
    dots = np.einsum("ij,ij->i", point - starts, vectors)
    fractions = np.clip(np.divide(dots, squared_lengths, out=np.zeros_like(dots), where=squared_lengths > 1e-12), 0.0, 1.0)
    distances = np.linalg.norm(starts + fractions[:, None] * vectors - point, axis=1)
    return fractions, distances


class CandidateFeatureExtractor:
    """Candidate rows remain in ``Candidates.edges`` order; no road ID is a feature."""

    feature_names = FEATURE_NAMES
    version = FEATURE_VERSION

    def __init__(self, graph: Graph):
        self.graph = graph
        self.vectors = graph.xy[graph.heads] - graph.xy[graph.tails]
        self.directions = np.divide(self.vectors, graph.edge_lengths[:, None], out=np.zeros_like(self.vectors), where=graph.edge_lengths[:, None] > 1e-9)
        unique = np.flatnonzero(graph.canonical_edges == np.arange(len(graph.arcs)))
        self.indegree = np.bincount(graph.heads[unique], minlength=len(graph.nodes))
        self.outdegree = np.bincount(graph.tails[unique], minlength=len(graph.nodes))

    def features_for_candidates(self, track, xy, index: int, candidates: Candidates) -> np.ndarray:
        n = len(candidates)
        if not n:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        if index < 0 or index >= len(track):
            raise IndexError("observation index outside track")
        edges = integer_array(candidates.edges, "candidate edges")
        if np.any(edges < 0) or np.any(edges >= len(self.graph.arcs)):
            raise ValueError("candidate contains invalid original arc IDs")
        distance = np.asarray(candidates.distances, dtype=float)
        fraction = np.asarray(candidates.fractions, dtype=float)
        if distance.shape != (n,) or fraction.shape != (n,) or not np.isfinite(distance).all() or not np.isfinite(fraction).all() or np.any(distance < 0) or np.any(fraction < 0) or np.any(fraction > 1):
            raise ValueError("candidate distances must be nonnegative finite K-vectors and fractions in [0,1]")
        graph = self.graph
        lengths = graph.edge_lengths[edges]
        directions = self.directions[edges]
        starts, vectors = graph.xy[graph.tails[edges]], self.vectors[edges]
        squared = lengths ** 2
        windows = [_window(track, xy, index, width) for width in (1, 3, 8)]
        aligns = [np.clip(directions @ w[3], -1.0, 1.0) for w in windows]
        reliability = [min(w[4] / 12.0, 1.0) for w in windows]
        medium = windows[1]
        lo, hi = medium[:2]
        previous_fraction, previous_distance = _project_to_segments(xy[lo], starts, vectors, squared)
        next_fraction, next_distance = _project_to_segments(xy[hi], starts, vectors, squared)
        incoming, outgoing = xy[index] - xy[lo], xy[hi] - xy[index]
        before_norm, after_norm = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        turn = float(np.dot(incoming, outgoing) / (before_norm * after_norm)) if before_norm * after_norm > 1e-8 else 1.0
        before_speed = before_norm / max(float(track[index, 2] - track[lo, 2]), 1.0)
        after_speed = after_norm / max(float(track[hi, 2] - track[index, 2]), 1.0)
        nearest = float(np.min(distance))
        distinct = np.unique(np.round(distance, 6))
        distinct_gap = float(distinct[1] - distinct[0]) if len(distinct) > 1 else 0.0
        rank = np.sum(distance[:, None] > distance[None, :] + 1e-7, axis=1) / max(n - 1, 1)
        previous_gap = float(track[index, 2] - track[max(index - 1, 0), 2])
        next_gap = float(track[min(index + 1, len(track) - 1), 2] - track[index, 2])
        const = lambda value: np.full(n, value, dtype=float)
        features = np.column_stack([
            distance, np.log1p(distance), np.minimum((distance / 10.0) ** 2, 10000.0),
            distance - nearest, np.minimum((distance + 1.0) / (nearest + 1.0), 100.0),
            rank, const(n), const(distinct_gap), fraction, np.minimum(fraction, 1.0 - fraction) * lengths,
            np.log1p(lengths), ((fraction > 1e-6) & (fraction < 1.0 - 1e-6)).astype(float),
            *aligns, np.abs(aligns[1]), *[const(r) for r in reliability],
            (1.0 - aligns[1]) * reliability[1],
            previous_distance, next_distance, (previous_distance + next_distance) / 2.0,
            np.maximum(previous_distance, next_distance),
            (next_fraction - previous_fraction) * lengths / medium[5],
            directions @ (medium[2] / medium[5]),
            const(np.log1p(windows[0][4] / windows[0][5])), const(np.log1p(medium[4] / medium[5])),
            const(np.clip(after_speed - before_speed, -100.0, 100.0)), const(np.clip(turn, -1.0, 1.0)),
            const(np.log1p(previous_gap)), const(np.log1p(next_gap)), const(medium[4] < 3.0),
            np.log1p(self.indegree[graph.tails[edges]]), np.log1p(self.outdegree[graph.tails[edges]]),
            np.log1p(self.indegree[graph.heads[edges]]), np.log1p(self.outdegree[graph.heads[edges]]),
            (lengths < 1e-6).astype(float),
        ])
        if features.shape != (n, len(FEATURE_NAMES)) or not np.isfinite(features).all():
            raise ValueError("invalid candidate feature matrix")
        return features.astype(np.float32)

    def transform(self, track, observation_indices, candidates: Sequence[Candidates]):
        """Return ``(X, offsets)``; observation j owns X[offsets[j]:offsets[j+1]]."""
        track = validate_track(track)
        indices = integer_array(observation_indices, "observation_indices")
        if len(indices) != len(candidates) or np.any(indices < 0) or np.any(indices >= len(track)):
            raise ValueError("observation_indices must match the candidate list and track bounds")
        xy = self.graph.project(track[:, :2])
        offsets = np.r_[0, np.cumsum([len(c) for c in candidates], dtype=np.int64)]
        rows = [self.features_for_candidates(track, xy, int(i), c) for i, c in zip(indices, candidates)]
        return (np.concatenate(rows) if rows else np.empty((0, len(FEATURE_NAMES)), np.float32)), offsets


def _extractor_for_graph(graph: Graph) -> CandidateFeatureExtractor:
    # Cache on the graph itself: no global cache retains all training-trip graphs.
    extractor = getattr(graph, "_pathsnap_candidate_features", None)
    if extractor is None:
        extractor = CandidateFeatureExtractor(graph)
        graph._pathsnap_candidate_features = extractor
    return extractor


def features_for_candidates(graph: Graph, track, xy, index: int, candidates: Candidates) -> np.ndarray:
    """Functional inference API; xy is graph.project(track[:, :2]), not reference XY."""
    return _extractor_for_graph(graph).features_for_candidates(track, xy, index, candidates)


@dataclass(frozen=True)
class ReferenceObservationLabels:
    edges: np.ndarray
    distances: np.ndarray
    score_margins: np.ndarray
    ambiguous: np.ndarray
    observation_indices: np.ndarray


PROXY_LABEL_POLICY = {
    "name": "nearest-reference-segment-with-motion-orientation-v1",
    "is_ground_truth": False,
    "description": "Project each GPS observation onto every distinct canonical directed reference-route segment; minimise distance squared plus 32m^2 times reliable signed heading mismatch. Motion uses at most +/-3 rows and never crosses a >20s local dropout. Route data is used ONLY to create labels or score completed predictions, never features.",
    "orientation_penalty_m2": 32.0,
    "ambiguous_score_margin_m2": 4.0,
    "limitations": "No per-observation labels were supplied. Intersections, stops, nearby parallel roads, repeated traversals and route defects can make these proxy labels uncertain; these metrics are not independently verified GPS ground truth. The geometry-derived target favours geometric predictors.",
}


def reference_observation_labels(graph: Graph, track, reference_route, observation_indices=None) -> ReferenceObservationLabels:
    """Construct transparent, candidate-independent route-projection proxy targets.

    This function must be called only during fitting or AFTER prediction. The
    reference search is over the entire supplied route, not generated candidates,
    so an absent reference segment remains an honest candidate miss.
    """
    track = validate_track(track)
    route = integer_array(reference_route, "reference_route")
    if not len(route) or np.any(route < 0) or np.any(route >= len(graph.arcs)):
        raise ValueError("reference route must contain valid original arc IDs")
    indices = np.arange(len(track), dtype=np.int64) if observation_indices is None else integer_array(observation_indices, "observation_indices")
    if np.any(indices < 0) or np.any(indices >= len(track)):
        raise ValueError("reference observation indices outside track")
    route_edges = np.unique(graph.canonical_edges[route])
    starts = graph.xy[graph.tails[route_edges]]
    vectors = graph.xy[graph.heads[route_edges]] - starts
    lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
    directions = np.divide(vectors, np.sqrt(lengths_squared)[:, None], out=np.zeros_like(vectors), where=lengths_squared[:, None] > 1e-12)
    xy = graph.project(track[:, :2])
    selected, distances, margins = [], [], []
    for index in indices:
        _, distance = _project_to_segments(xy[index], starts, vectors, lengths_squared)
        window = _window(track, xy, int(index), 3)
        reliability = min(window[4] / 12.0, 1.0)
        alignment = np.clip(directions @ window[3], -1.0, 1.0)
        score = distance ** 2 + 32.0 * reliability * (1.0 - alignment)
        winner = int(np.argmin(score))
        second = float(np.partition(score, 1)[1]) if len(score) > 1 else float("inf")
        selected.append(route_edges[winner])
        distances.append(distance[winner])
        margins.append(second - score[winner])
    margin_array = np.asarray(margins)
    return ReferenceObservationLabels(np.asarray(selected, np.int64), np.asarray(distances), margin_array, margin_array < 4.0, indices)


def reference_observation_edges(graph: Graph, track, reference_route, observation_indices=None) -> np.ndarray:
    """Return canonical directed proxy edges, NOT supplied per-GPS truth."""
    return reference_observation_labels(graph, track, reference_route, observation_indices).edges
