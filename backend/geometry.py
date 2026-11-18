"""Serialize predicted and observed geometry only; never reference routes."""
from __future__ import annotations

import numpy as np

from mapmatch.data import integer_array


def prediction_arrays(graph, track, result) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    route = integer_array(result.route, "predicted route")
    indices = integer_array(result.observation_indices, "predicted observation indices")
    edges = integer_array(result.matched_edges, "predicted edges")
    fractions = np.asarray(result.matched_fractions, dtype=float)
    distances = np.asarray(result.matched_distances, dtype=float)
    if not len(route) or np.any(route < 0) or np.any(route >= len(graph.arcs)):
        raise ValueError("The decoder returned an invalid or empty predicted route.")
    if (not len(indices) or np.any(indices < 0) or np.any(indices >= len(track))
            or np.any(np.diff(indices) <= 0) or len(edges) != len(indices)
            or fractions.shape != indices.shape or distances.shape != indices.shape
            or np.any(edges < 0) or np.any(edges >= len(graph.arcs))
            or not np.isfinite(fractions).all() or not np.isfinite(distances).all()
            or np.any(fractions < 0) or np.any(fractions > 1) or np.any(distances < 0)):
        raise ValueError("The decoder returned invalid predicted observation assignments.")
    return route, indices, edges, fractions, distances


def predicted_geojson(graph, route: np.ndarray) -> dict:
    """Do not draw an artificial connector if a decoder explicitly returns breaks."""
    split = np.flatnonzero(graph.heads[route[:-1]] != graph.tails[route[1:]]) + 1
    features = []
    for part_index, edges in enumerate(np.split(route, split)):
        coordinates = [graph.nodes[int(graph.tails[edges[0]])].tolist()]
        coordinates.extend(graph.nodes[graph.heads[edges]].tolist())
        features.append({
            "type": "Feature",
            "properties": {"kind": "predicted_route", "part_index": part_index, "arc_count": int(len(edges))},
            "geometry": {"type": "LineString", "coordinates": coordinates},
        })
    return {"type": "FeatureCollection", "features": features}


def raw_gps_geojson(track: np.ndarray) -> dict:
    geometry = {"type": "LineString", "coordinates": track[:, :2].tolist()}
    if len(track) == 1:
        geometry = {"type": "Point", "coordinates": track[0, :2].tolist()}
    return {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "properties": {"kind": "raw_gps", "observation_count": int(len(track)), "timestamps": track[:, 2].tolist()},
        "geometry": geometry,
    }]}


def observation_assignments(graph, indices, edges, fractions, distances) -> list[dict]:
    coordinates = graph.nodes[graph.tails[edges]] + fractions[:, None] * (graph.nodes[graph.heads[edges]] - graph.nodes[graph.tails[edges]])
    return [
        {"observation_index": int(i), "edge_id": int(e), "fraction": float(f), "distance_m": float(d), "coordinate": coordinate.tolist()}
        for i, e, f, d, coordinate in zip(indices, edges, fractions, distances, coordinates)
    ]


def route_continuity(graph, route: np.ndarray) -> float:
    if len(route) < 2:
        return 1.0
    return float(np.mean(graph.heads[route[:-1]] == graph.tails[route[1:]]))
