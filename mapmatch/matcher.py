"""Projected directed-edge HMM, Viterbi decoding, and full-route reconstruction."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time

import numpy as np

from .candidates import CandidateIndex, Candidates
from .data import Graph, validate_route, validate_track
from .routing import Router


@dataclass(frozen=True)
class MatchConfig:
    sigma_m: float = 10.0
    beta_m: float = 15.0
    max_candidates: int = 12
    candidate_radius_m: float = 80.0
    max_radius_m: float = 1000.0
    routing_slack_m: float = 300.0
    backward_tolerance_m: float = 20.0
    observation_distance_m: float = 0.0
    observation_interval: float = 5.0
    sample_spacing_m: float = 40.0
    cache_size: int = 2048
    classifier_weight: float = 1.0
    geometry_weight: float = 1.0
    candidate_retry_limit: int = 48

    def __post_init__(self) -> None:
        positive = ("sigma_m", "beta_m", "candidate_radius_m", "max_radius_m", "routing_slack_m", "observation_interval", "sample_spacing_m")
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("backward_tolerance_m", "observation_distance_m", "classifier_weight", "geometry_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_radius_m < self.candidate_radius_m:
            raise ValueError("max_radius_m must be at least candidate_radius_m")
        for name in ("max_candidates", "cache_size", "candidate_retry_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class MatchResult:
    route: np.ndarray
    observation_indices: np.ndarray
    matched_edges: np.ndarray
    matched_fractions: np.ndarray
    matched_distances: np.ndarray
    breaks: list[dict]
    diagnostics: dict


class MatchingError(RuntimeError):
    """No honest connected route can be recovered under the supplied settings."""


class MapMatcher:
    def __init__(self, graph: Graph, config: MatchConfig | None = None, emission_model=None):
        self.graph = graph
        self.config = config or MatchConfig()
        self.emission_model = emission_model
        self.index = CandidateIndex(graph, self.config.sample_spacing_m)
        self.router = Router(graph, self.config.cache_size)

    def _observations(self, xy: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        config = self.config
        if config.observation_distance_m <= 0 or len(xy) <= 2:
            return np.arange(len(xy), dtype=np.int64)
        retained = [0]
        for i in range(1, len(xy) - 1):
            last = retained[-1]
            if (np.linalg.norm(xy[i] - xy[last]) >= config.observation_distance_m
                    or timestamps[i] - timestamps[last] >= config.observation_interval):
                retained.append(i)
        retained.append(len(xy) - 1)
        return np.array(retained, dtype=np.int64)

    def _candidates(self, point: np.ndarray) -> tuple[Candidates, bool]:
        radius = self.config.candidate_radius_m
        expanded = False
        while True:
            candidates = self.index.query(point, radius, self.config.max_candidates)
            if len(candidates):
                return candidates, expanded
            if radius >= self.config.max_radius_m:
                raise MatchingError(f"no road candidate within {self.config.max_radius_m:g} m of a GPS observation")
            radius = min(radius * 2, self.config.max_radius_m)
            expanded = True

    def _transitions(self, previous: Candidates, current: Candidates, chord: float, horizon: float) -> np.ndarray:
        graph = self.graph
        edges1, edges2 = previous.edges, current.edges
        len1, len2 = graph.edge_lengths[edges1], graph.edge_lengths[edges2]
        remainder = (1.0 - previous.fractions) * len1
        initial = current.fractions * len2
        targets = graph.tails[edges2]
        matrix = np.full((len(previous), len(current)), np.inf)
        sources, inverse = np.unique(graph.heads[edges1], return_inverse=True)
        network = np.array([self.router.distances(int(source), targets, horizon) for source in sources])
        matrix[:] = remainder[:, None] + network[inverse] + initial[None, :]
        same = edges1[:, None] == edges2[None, :]
        along = initial[None, :] - (previous.fractions * len1)[:, None]
        # Small backward projections are measurement jitter, not a full road loop.
        direct = same & (along >= -self.config.backward_tolerance_m)
        matrix[direct] = np.maximum(along[direct], 0.0)
        matrix[matrix > horizon] = np.inf
        return np.abs(matrix - chord) / self.config.beta_m

    def match(self, track: np.ndarray) -> MatchResult:
        """Decode, widening candidate pruning only after a genuine reachability failure.

        An isolated nearby road can evict the reachable road from the top K.
        Bounded retries recompute the entire dynamic program with more states;
        they never reset a path or connect disconnected graph components.
        """
        start = time.perf_counter()
        original = self.config
        attempts = []
        width = original.max_candidates
        try:
            while True:
                self.config = replace(original, max_candidates=width)
                try:
                    result = self._match_once(track)
                except MatchingError as exc:
                    if "no connected directed transition" not in str(exc) or width >= original.candidate_retry_limit:
                        raise
                    attempts.append({"max_candidates": width, "failure": str(exc)})
                    width = min(max(width * 2, width + 1), original.candidate_retry_limit)
                    continue
                result.diagnostics["requested_config"] = asdict(original)
                result.diagnostics["candidate_retry_attempts"] = attempts
                result.diagnostics["effective_max_candidates"] = width
                result.diagnostics["elapsed_seconds"] = time.perf_counter() - start
                return result
        finally:
            self.config = original

    def _match_once(self, track: np.ndarray) -> MatchResult:
        start_time = time.perf_counter()
        track = validate_track(track)
        xy_all = self.graph.project(track[:, :2])
        retained = self._observations(xy_all, track[:, 2])
        xy = xy_all[retained]
        candidates = []
        expanded_count = 0
        for i, point in enumerate(xy):
            try:
                current, expanded = self._candidates(point)
            except MatchingError as exc:
                raise MatchingError(f"observation {int(retained[i])}: {exc}") from exc
            candidates.append(current)
            expanded_count += int(expanded)
        emissions = [self.config.geometry_weight * 0.5 * (c.distances / self.config.sigma_m) ** 2 for c in candidates]
        classifier_edges = None
        if self.emission_model is not None:
            if hasattr(self.emission_model, "score_candidates_batch"):
                probabilities = self.emission_model.score_candidates_batch(self.graph, track, retained, candidates)
            else:
                probabilities = [self.emission_model.score_candidates(self.graph, track, xy_all, int(i), c) for i, c in zip(retained, candidates)]
            if len(probabilities) != len(candidates):
                raise ValueError("emission model returned an incorrect observation count")
            classifier_edges = []
            for i, (c, probability) in enumerate(zip(candidates, probabilities)):
                probability = np.asarray(probability, dtype=float)
                if probability.shape != (len(c),) or not np.isfinite(probability).all() or np.any(probability < 0) or probability.sum() <= 0:
                    raise ValueError("emission model must return finite nonnegative candidate probabilities with positive sum")
                probability = probability / probability.sum()
                emissions[i] = emissions[i] - self.config.classifier_weight * np.log(np.maximum(probability, 1e-12))
                classifier_edges.append(int(c.edges[int(np.argmax(probability))]))
        cost = emissions[0].copy()
        backpointers = [np.full(len(candidates[0]), -1, dtype=np.int64)]
        offsets = float(np.min(cost))
        cost -= np.min(cost)
        retried = 0
        for i in range(1, len(candidates)):
            chord = float(np.linalg.norm(xy[i] - xy[i - 1]))
            horizon = chord + self.config.routing_slack_m
            transition = self._transitions(candidates[i - 1], candidates[i], chord, horizon)
            total = cost[:, None] + transition
            if not np.isfinite(total).any():
                # An unusual detour may exceed the efficient local routing bound.
                # Expand explicitly before concluding that the track is disconnected.
                horizon = max(5000.0, horizon * 10.0)
                transition = self._transitions(candidates[i - 1], candidates[i], chord, horizon)
                total = cost[:, None] + transition
                retried += 1
            if not np.isfinite(total).any():
                raise MatchingError(f"no connected directed transition between observations {int(retained[i - 1])} and {int(retained[i])}; no route was fabricated")
            back = np.argmin(total, axis=0)
            cost = total[back, np.arange(len(candidates[i]))] + emissions[i]
            minimum = float(np.min(cost))
            offsets += minimum
            cost -= minimum
            backpointers.append(back)
        state = int(np.argmin(cost))
        selection = np.empty(len(candidates), dtype=np.int64)
        for i in range(len(candidates) - 1, -1, -1):
            selection[i] = state
            state = int(backpointers[i][state])
        edges = np.array([c.edges[s] for c, s in zip(candidates, selection)], dtype=np.int64)
        fractions = np.array([c.fractions[s] for c, s in zip(candidates, selection)])
        distances = np.array([c.distances[s] for c, s in zip(candidates, selection)])
        route = [int(edges[0])]
        for i in range(1, len(edges)):
            prev, edge = int(edges[i - 1]), int(edges[i])
            along = (fractions[i] - fractions[i - 1]) * self.graph.edge_lengths[edge]
            if prev == edge and along >= -self.config.backward_tolerance_m:
                continue
            chord = float(np.linalg.norm(xy[i] - xy[i - 1]))
            horizon = max(5000.0, (chord + self.config.routing_slack_m) * 10.0)
            distance, connector = self.router.shortest_path(int(self.graph.heads[prev]), int(self.graph.tails[edge]), horizon)
            if not math.isfinite(distance):
                raise MatchingError("internal reconstruction failure: decoded states have no connected path")
            # Preserve real cycles/repeated traversals; do not globally deduplicate.
            route.extend(connector)
            route.append(edge)
        value = np.asarray(route, dtype=np.int64)
        validation = validate_route(self.graph, value)
        if not validation["valid"]:
            raise MatchingError("internal reconstruction failure: decoded route is not topologically connected")
        diagnostics = {
            "config": asdict(self.config),
            "observation_count": len(track),
            "matched_observation_count": len(retained),
            "expanded_candidate_observations": expanded_count,
            "expanded_routing_transitions": retried,
            "route_arc_count": len(value),
            "route_length_m": validation["length_m"],
            "negative_log_likelihood_relative": offsets + float(np.min(cost)),
            "lateral_distance_median_m": float(np.median(distances)),
            "lateral_distance_p95_m": float(np.quantile(distances, 0.95)),
            "lateral_distance_max_m": float(np.max(distances)),
            "routing_searches": self.router.searches,
            "routing_cache_hits": self.router.cache_hits,
            "elapsed_seconds": time.perf_counter() - start_time,
            "route_validation": validation,
            "breaks": [],
            "uses_reference_route": False,
            "emission_model": "learned_classifier_plus_geometry" if self.emission_model is not None else "geometry_only_baseline",
            "classifier_only_edges": classifier_edges,
        }
        return MatchResult(value, retained, edges, fractions, distances, [], diagnostics)
