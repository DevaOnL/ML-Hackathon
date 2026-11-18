"""Request-local evidence that the trained classifier participates in decoding.

These probabilities are held only until this request finishes. They contain no
reference information and are not reused for another inference request.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidatePrediction:
    edges: np.ndarray
    probabilities: np.ndarray


class AuditedEmissionModel:
    def __init__(self, model):
        self.model = model
        self.records: dict[int, CandidatePrediction] = {}
        self.call_count = 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def score_candidates(self, graph, track, xy, index: int, candidates) -> np.ndarray:
        probabilities = self.model.score_candidates(graph, track, xy, index, candidates)
        return self._record(graph, track, index, candidates, probabilities)

    def score_candidates_batch(self, graph, track, observation_indices, candidates):
        if len(observation_indices) != len(candidates):
            raise ValueError("The decoder supplied unequal observation and candidate counts.")
        batch = getattr(self.model, "score_candidates_batch", None)
        if callable(batch):
            probabilities = batch(graph, track, observation_indices, candidates)
        else:
            xy = graph.project(track[:, :2])
            probabilities = [self.model.score_candidates(graph, track, xy, int(i), c) for i, c in zip(observation_indices, candidates)]
        if len(probabilities) != len(candidates):
            raise ValueError("The trained model returned unequal batch probability counts.")
        return [self._record(graph, track, int(i), c, p) for i, c, p in zip(observation_indices, candidates, probabilities)]

    def _record(self, graph, track, index, candidates, probabilities) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=float)
        if (probabilities.shape != (len(candidates.edges),) or not len(probabilities)
                or not np.isfinite(probabilities).all() or np.any(probabilities <= 0)
                or np.any(probabilities > 1) or not np.isclose(probabilities.sum(), 1.0, rtol=1e-5, atol=1e-8)):
            raise ValueError("The trained model returned invalid candidate probabilities.")
        if not 0 <= int(index) < len(track):
            raise ValueError("The decoder supplied an invalid observation index to the trained model.")
        edges = np.asarray(candidates.edges, dtype=np.int64).copy()
        if edges.shape != probabilities.shape or np.any(edges < 0) or np.any(edges >= len(graph.arcs)):
            raise ValueError("The decoder supplied invalid candidate edge IDs.")
        saved = probabilities.copy()
        edges.setflags(write=False)
        saved.setflags(write=False)
        self.records[int(index)] = CandidatePrediction(edges, saved)
        self.call_count += 1
        return probabilities
