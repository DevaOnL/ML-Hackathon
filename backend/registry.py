"""Dynamic, immutable trip registration and bounded graph-only caching.

No route/reference file is opened or inspected by discovery or graph loading.
Only registered prefixes, never request-supplied paths, reach filesystem loaders.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import re
from threading import RLock

import numpy as np

from mapmatch.data import Graph, load_graph, load_track
from .errors import ServiceError

TRIP_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z", re.ASCII)


def contained_file(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ServiceError("dataset_file_invalid", "A registered dataset file is missing or outside its configured root.")
    return resolved


@dataclass(frozen=True)
class Trip:
    trip_id: str
    prefix: Path
    relative_prefix: Path
    track: np.ndarray

    def metadata(self) -> dict:
        return {
            "trip_id": self.trip_id,
            "observation_count": int(len(self.track)),
            "duration_s": float(self.track[-1, 2] - self.track[0, 2]),
        }


class TripRegistry:
    def __init__(self, root: Path, expected_count: int = 10):
        self.root = root.resolve()
        if not self.root.is_dir():
            raise ServiceError("dataset_missing", "The configured trip dataset directory is unavailable.")
        # Both original per-trip folders and flat evaluator packages are accepted.
        tracks = sorted(set(self.root.glob("*.track")) | set(self.root.glob("*/*.track")))
        if len(tracks) != expected_count:
            raise ServiceError("trip_count_mismatch", f"Expected exactly {expected_count} trips in the configured dataset; found {len(tracks)}.")
        self._trips: dict[str, Trip] = {}
        for track_path in tracks:
            trip_id = track_path.stem
            if not TRIP_ID_PATTERN.fullmatch(trip_id):
                raise ServiceError("dataset_id_invalid", "Dataset trip IDs must use 1-64 letters, digits, underscores or hyphens.")
            if trip_id in self._trips:
                raise ServiceError("dataset_id_duplicate", "The configured dataset contains duplicate trip IDs.")
            contained_file(track_path, self.root)
            prefix = track_path.with_suffix("")
            contained_file(prefix.with_suffix(".nodes"), self.root)
            contained_file(prefix.with_suffix(".arcs"), self.root)
            try:
                track = load_track(track_path)
            except Exception as exc:
                raise ServiceError("track_invalid", "A registered GPS track failed strict input validation.") from exc
            track.setflags(write=False)
            self._trips[trip_id] = Trip(trip_id, prefix, prefix.relative_to(self.root), track)

    def __len__(self) -> int:
        return len(self._trips)

    def get(self, trip_id: str) -> Trip:
        if not TRIP_ID_PATTERN.fullmatch(trip_id) or trip_id not in self._trips:
            raise ServiceError("invalid_trip_id", "Select a trip_id returned by GET /trips.", 422)
        return self._trips[trip_id]

    def metadata(self) -> list[dict]:
        return [self._trips[key].metadata() for key in sorted(self._trips)]

    def reference_path(self, trip: Trip, reference_dir: Path | None) -> Path | None:
        """Call ONLY after prediction; do not pass the returned path to inference."""
        root = (reference_dir or self.root).resolve()
        if reference_dir is None:
            candidates = [trip.prefix.with_suffix(".route")]
        else:
            candidates = [root / trip.relative_prefix.with_suffix(".route"), root / trip.trip_id / f"{trip.trip_id}.route", root / f"{trip.trip_id}.route"]
        found = []
        for candidate in dict.fromkeys(candidates):
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root):
                raise ServiceError("reference_path_invalid", "Server-private reference configuration is invalid.", 500)
            if resolved.is_file() and resolved not in found:
                found.append(resolved)
        if len(found) > 1:
            raise ServiceError("reference_ambiguous", "More than one server-private reference matches the selected trip.", 500)
        return found[0] if found else None


class GraphCache:
    """Caches immutable input graphs only; no matcher, route, label, or response cache."""
    def __init__(self, registry: TripRegistry, max_size: int):
        self.registry = registry
        self.max_size = max_size
        self._values: OrderedDict[str, Graph] = OrderedDict()
        self._lock = RLock()

    def get(self, trip: Trip) -> Graph:
        with self._lock:
            if trip.trip_id in self._values:
                self._values.move_to_end(trip.trip_id)
                return self._values[trip.trip_id]
            contained_file(trip.prefix.with_suffix(".nodes"), self.registry.root)
            contained_file(trip.prefix.with_suffix(".arcs"), self.registry.root)
            try:
                graph = load_graph(trip.prefix)
            except Exception as exc:
                raise ServiceError("graph_invalid", "The selected trip road network failed strict input validation.", 500) from exc
            self._values[trip.trip_id] = graph
            while len(self._values) > self.max_size:
                self._values.popitem(last=False)
            return graph

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
