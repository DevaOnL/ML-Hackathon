"""Exact point-to-segment candidates backed by a conservative sampled KD tree."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .data import Graph


@dataclass(frozen=True)
class Candidates:
    edges: np.ndarray
    fractions: np.ndarray
    distances: np.ndarray

    def __len__(self) -> int:
        return len(self.edges)


class CandidateIndex:
    """Index segment interiors, not only nodes or midpoints.

    Samples are at most ``spacing_m`` apart and include both endpoints. Every
    point on a segment is within spacing/2 of a sample. Searching radius +
    spacing/2 therefore retrieves every segment within the requested radius.
    Distances and fractions are then computed against the full exact segment.
    """

    def __init__(self, graph: Graph, spacing_m: float = 40.0):
        if not np.isfinite(spacing_m) or spacing_m <= 0:
            raise ValueError("spacing_m must be finite and positive")
        self.graph = graph
        self.spacing_m = float(spacing_m)
        edge_ids = np.flatnonzero(graph.canonical_edges == np.arange(len(graph.arcs)))
        count = np.maximum(1, np.ceil(graph.edge_lengths[edge_ids] / spacing_m).astype(np.int64))
        self.sample_edges = np.repeat(edge_ids, count + 1)
        starts = np.repeat(np.cumsum(count + 1) - (count + 1), count + 1)
        fractions = (np.arange(len(self.sample_edges)) - starts) / np.repeat(count, count + 1)
        xy = graph.xy[graph.tails[self.sample_edges]] + fractions[:, None] * (graph.xy[graph.heads[self.sample_edges]] - graph.xy[graph.tails[self.sample_edges]])
        self.tree = cKDTree(xy)

    def query(self, point_xy: np.ndarray, radius_m: float = 80.0, max_candidates: int = 12) -> Candidates:
        point = np.asarray(point_xy, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("candidate query point must be a finite XY pair")
        if not np.isfinite(radius_m) or radius_m <= 0 or max_candidates < 1:
            raise ValueError("radius_m and max_candidates must be positive")
        samples = self.tree.query_ball_point(point, radius_m + self.spacing_m / 2.0 + 1e-8)
        edges = np.unique(self.sample_edges[samples])
        if not len(edges):
            return Candidates(np.empty(0, dtype=np.int64), np.empty(0), np.empty(0))
        tails = self.graph.xy[self.graph.tails[edges]]
        vectors = self.graph.xy[self.graph.heads[edges]] - tails
        lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
        dot = np.einsum("ij,ij->i", point - tails, vectors)
        fractions = np.clip(np.divide(dot, lengths_squared, out=np.zeros_like(dot), where=lengths_squared > 0), 0.0, 1.0)
        distances = np.linalg.norm(tails + fractions[:, None] * vectors - point, axis=1)
        keep = np.flatnonzero(distances <= radius_m + 1e-8)
        order = keep[np.lexsort((edges[keep], distances[keep]))[:max_candidates]]
        return Candidates(edges[order], fractions[order], distances[order])
