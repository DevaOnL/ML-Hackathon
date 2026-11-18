"""Directed, deterministic bounded Dijkstra with original arc-ID reconstruction."""
from __future__ import annotations

from collections import OrderedDict
import heapq
import math

import numpy as np

from .data import Graph


class Router:
    def __init__(self, graph: Graph, cache_size: int = 2048, distance_bucket_m: float = 250.0):
        if cache_size < 1 or distance_bucket_m <= 0 or not math.isfinite(distance_bucket_m):
            raise ValueError("invalid routing cache parameters")
        self.graph = graph
        self.cache_size = int(cache_size)
        self.distance_bucket_m = float(distance_bucket_m)
        self._cache: OrderedDict[int, tuple[float, dict[int, float], dict[int, int]]] = OrderedDict()
        unique = np.flatnonzero(graph.canonical_edges == np.arange(len(graph.arcs)))
        order = unique[np.argsort(graph.tails[unique], kind="stable")]
        self._edges = order
        self._offsets = np.concatenate(([0], np.cumsum(np.bincount(graph.tails[order], minlength=len(graph.nodes)))))
        self.searches = 0
        self.cache_hits = 0

    def _node(self, node: int) -> int:
        if isinstance(node, (bool, np.bool_)) or int(node) != node or node < 0 or node >= len(self.graph.nodes):
            raise ValueError("routing node must be a valid zero-based node ID")
        return int(node)

    def _search(self, source: int, limit: float) -> tuple[dict[int, float], dict[int, int]]:
        source = self._node(source)
        if math.isnan(limit) or limit < 0:
            raise ValueError("max_distance must be nonnegative")
        existing = self._cache.get(source)
        if existing is not None and existing[0] >= limit:
            self._cache.move_to_end(source)
            self.cache_hits += 1
            return existing[1], existing[2]
        horizon = math.inf if math.isinf(limit) else max(self.distance_bucket_m, math.ceil(limit / self.distance_bucket_m) * self.distance_bucket_m)
        distances = {source: 0.0}
        previous: dict[int, int] = {}
        queue = [(0.0, source)]
        heads, lengths = self.graph.heads, self.graph.edge_lengths
        self.searches += 1
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            for edge_value in self._edges[self._offsets[node]:self._offsets[node + 1]]:
                edge = int(edge_value)
                target = int(heads[edge])
                candidate = distance + float(lengths[edge])
                if candidate <= horizon and candidate < distances.get(target, math.inf):
                    distances[target] = candidate
                    previous[target] = edge
                    heapq.heappush(queue, (candidate, target))
        self._cache[source] = (horizon, distances, previous)
        self._cache.move_to_end(source)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return distances, previous

    def distances(self, source: int, targets: np.ndarray | list[int], max_distance: float = math.inf) -> np.ndarray:
        targets = [self._node(t) for t in targets]
        distances, _ = self._search(source, float(max_distance))
        return np.array([distances.get(t, math.inf) if distances.get(t, math.inf) <= max_distance else math.inf for t in targets])

    def shortest_path(self, source: int, target: int, max_distance: float = math.inf) -> tuple[float, list[int]]:
        source, target = self._node(source), self._node(target)
        distances, previous = self._search(source, float(max_distance))
        distance = distances.get(target, math.inf)
        if distance > max_distance or not math.isfinite(distance):
            return math.inf, []
        result = []
        node = target
        while node != source:
            edge = previous[node]
            result.append(edge)
            node = int(self.graph.tails[edge])
        result.reverse()
        return distance, result
