"""Strict input parsing and graph geometry; inference never loads route labels."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


def _coordinates(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 2 or not len(result):
        raise ValueError(f"{name} must be a nonempty (n, 2) coordinate array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    if (np.abs(result[:, 0]) > 180).any() or (np.abs(result[:, 1]) > 90).any():
        raise ValueError(f"{name} must contain longitude, latitude in decimal degrees")
    return result


def integer_array(value: np.ndarray | Iterable[int], name: str, columns: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite integers")
    if raw.size and (np.any(raw != np.floor(raw)) or np.any(raw >= 2**63) or np.any(raw < -(2**63))):
        raise ValueError(f"{name} must contain valid int64 integers")
    result = raw.astype(np.int64)
    if columns is not None and (result.ndim != 2 or result.shape[1] != columns):
        raise ValueError(f"{name} must have {columns} columns")
    if columns is None and result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


@dataclass
class Graph:
    """Rows of arcs retain their original zero-based IDs, including duplicates."""

    nodes: np.ndarray
    arcs: np.ndarray
    xy: np.ndarray = field(init=False, repr=False)
    edge_lengths: np.ndarray = field(init=False, repr=False)
    canonical_edges: np.ndarray = field(init=False, repr=False)
    origin: np.ndarray = field(init=False, repr=False)
    scale: np.ndarray = field(init=False, repr=False)
    tails: np.ndarray = field(init=False, repr=False)
    heads: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.nodes = _coordinates(self.nodes, "nodes").copy()
        self.arcs = integer_array(self.arcs, "arcs", columns=2).copy()
        if not len(self.arcs):
            raise ValueError("graph must contain at least one arc")
        if (self.arcs < 0).any() or (self.arcs >= len(self.nodes)).any():
            raise ValueError("arc endpoints must be valid zero-based node IDs")
        self.origin = np.mean(self.nodes, axis=0)
        self.scale = np.deg2rad(1.0) * EARTH_RADIUS_M * np.array([np.cos(np.deg2rad(self.origin[1])), 1.0])
        self.xy = self.project(self.nodes)
        self.tails, self.heads = self.arcs.T
        self.edge_lengths = np.linalg.norm(self.xy[self.heads] - self.xy[self.tails], axis=1)
        # Never sum parallel arc weights (the default sparse COO->CSR behavior).
        _, first, inverse = np.unique(self.arcs, axis=0, return_index=True, return_inverse=True)
        self.canonical_edges = first[inverse].astype(np.int64)

    def project(self, coordinates: np.ndarray) -> np.ndarray:
        return (np.asarray(coordinates, dtype=float) - self.origin) * self.scale

    def unproject(self, coordinates: np.ndarray) -> np.ndarray:
        if np.any(np.abs(self.scale) < 1e-12):
            raise ValueError("longitude is undefined at the projection pole")
        return np.asarray(coordinates, dtype=float) / self.scale + self.origin


def _read_numeric(path: str | Path, columns: int, *, allow_empty: bool = False) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not path.stat().st_size:
        if allow_empty:
            return np.empty((0, columns), dtype=float)
        raise ValueError(f"{path}: empty input")
    # Physical row numbers define graph IDs: silently skipping a blank line
    # would shift every following node/arc. Reject blank rows explicitly.
    with path.open("r", encoding="utf-8") as stream:
        for row, text in enumerate(stream, 1):
            if not text.strip():
                raise ValueError(f"{path}: blank physical row {row} would change row IDs")
    try:
        value = np.loadtxt(path, dtype=float, ndmin=2, comments=None)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid numeric input: {exc}") from exc
    if value.shape[1] != columns or (not len(value) and not allow_empty):
        raise ValueError(f"{path}: expected {columns} columns, got shape {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{path}: non-finite values")
    return value


def load_graph(prefix: str | Path) -> Graph:
    prefix = Path(prefix)
    return Graph(_read_numeric(prefix.with_suffix(".nodes"), 2), _read_numeric(prefix.with_suffix(".arcs"), 2))


def validate_track(track: np.ndarray) -> np.ndarray:
    value = np.asarray(track, dtype=float)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value):
        raise ValueError("track must be a nonempty (n, 3) lon, lat, timestamp array")
    _coordinates(value[:, :2], "track")
    if not np.isfinite(value[:, 2]).all():
        raise ValueError("track timestamps must be finite")
    if len(value) > 1 and np.any(np.diff(value[:, 2]) <= 0):
        raise ValueError("track timestamps must be strictly increasing; input is never silently sorted")
    return value


def load_track(path: str | Path) -> np.ndarray:
    return validate_track(_read_numeric(path, 3))


def load_route(path: str | Path) -> np.ndarray:
    return integer_array(_read_numeric(path, 1, allow_empty=True).ravel(), "route")


def validate_route(graph: Graph, route: np.ndarray | Iterable[int]) -> dict:
    value = integer_array(route, "route")
    invalid = np.flatnonzero((value < 0) | (value >= len(graph.arcs)))
    if len(invalid):
        return {"valid": False, "arc_count": len(value), "invalid_positions": invalid.tolist(), "discontinuities": [], "length_m": None}
    discontinuities = np.flatnonzero(graph.heads[value[:-1]] != graph.tails[value[1:]])
    return {"valid": bool(not len(discontinuities)), "arc_count": len(value), "invalid_positions": [], "discontinuities": discontinuities.tolist(), "length_m": float(np.sum(graph.edge_lengths[value]))}


def write_route(path: str | Path, route: np.ndarray | Iterable[int]) -> None:
    value = integer_array(route, "route")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    np.savetxt(temporary, value, fmt="%d")
    temporary.replace(path)


def discover_trips(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    tracks = sorted(root.glob("*/*.track"))
    if not tracks:
        tracks = sorted(root.glob("*.track"))
    if not tracks:
        raise ValueError(f"{root}: no .track files found")
    result = []
    for track in tracks:
        prefix = track.with_suffix("")
        for suffix in (".nodes", ".arcs"):
            if not prefix.with_suffix(suffix).is_file():
                raise FileNotFoundError(prefix.with_suffix(suffix))
        result.append(prefix)
    if len({p.name for p in result}) != len(result):
        raise ValueError("trip IDs must be unique within an input root")
    return result
