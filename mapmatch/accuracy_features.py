"""Opt-in, stateless transforms of the immutable 38-column candidate matrix.

Only the supplied matrix and observation offsets are used. No fitting, labels,
files, graph identifiers, coordinates or absolute timestamps are required.
The context profile preserves the original columns verbatim as a prefix;
compact and physics are independent schemas, not baseline replacements.

Distances used by derived columns are clipped to [0, 1e6] metres and signed
velocities to +/-1000 m/s to keep transforms finite for extreme finite inputs.
Normal candidate distances retain the exact energy d**2 + 32 * mismatch.
All logarithms use log1p, including at zero-energy stationary observations.
"""
from __future__ import annotations

import numpy as np

from .features import FEATURE_NAMES as _BASE_FEATURE_NAMES

ACCURACY_FEATURE_VERSION = "pathsnap-candidate-v2"

__all__ = ["ACCURACY_FEATURE_VERSION", "feature_names", "transform_accuracy_features"]

_BASE_INDEX = {name: i for i, name in enumerate(_BASE_FEATURE_NAMES)}
_CONTEXT_EXTRA_NAMES = (
    "geometry_energy_m2",
    "log1p_geometry_energy_32m2",
    "geometry_energy_gap_m2",
    "log1p_geometry_energy_gap_32m2",
    "geometry_energy_strict_rank",
    "signed_log1p_geometry_energy_margin_32m2",
    "geometry_energy_distinct_gap_m2",
    "geometry_energy_soft_support",
    "multiwindow_direction_mismatch",
    "short_direction_mismatch",
    "long_direction_mismatch",
    "multiwindow_direction_disagreement",
    "reliable_heading_angle_cost",
    "candidate_nearest_angular_cost_m2",
    "distance_heading_interaction",
    "direction_energy_fraction",
    "multiwindow_geometry_energy_m2",
    "log1p_multiwindow_geometry_energy_32m2",
    "log1p_multiwindow_energy_gap_32m2",
    "multiwindow_energy_strict_rank",
    "signed_log1p_multiwindow_energy_margin_32m2",
    "log1p_neighbour_lateral_mean_10m",
    "log1p_neighbour_lateral_max_10m",
    "signed_log1p_neighbour_lateral_excess_10m",
    "signed_sqrt_projected_speed_10mps",
    "signed_sqrt_motion_along_arc_10mps",
    "reliable_reverse_motion",
    "projected_motion_disagreement",
    "reliable_turn_cost",
    "endpoint_exposure",
    "log1p_distance_gap_10m",
    "signed_log1p_distance_margin_10m",
)
_PHYSICS_NAMES = (
    "log1p_geometry_energy_32m2",
    "geometry_energy_strict_rank",
    "signed_log1p_geometry_energy_margin_32m2",
)
_COMPACT_NAMES = _PHYSICS_NAMES + (
    "multiwindow_direction_mismatch",
    "multiwindow_direction_disagreement",
    "signed_log1p_neighbour_lateral_excess_10m",
    "signed_sqrt_motion_along_arc_10mps",
    "projected_motion_disagreement",
    "reliable_turn_cost",
    "endpoint_exposure",
    "low_motion",
    "edge_is_zero_length",
)
_SCHEMAS = {
    "context": _BASE_FEATURE_NAMES + _CONTEXT_EXTRA_NAMES,
    "compact": _COMPACT_NAMES,
    "physics": _PHYSICS_NAMES,
}


def feature_names(profile: str = "context") -> tuple[str, ...]:
    """Return the deterministic, ordered schema: context=70, compact=12, physics=3."""
    if not isinstance(profile, str) or profile not in _SCHEMAS:
        raise ValueError("profile must be 'context', 'compact', or 'physics'")
    return _SCHEMAS[profile]


def _validated_inputs(X, offsets):
    try:
        matrix = np.asarray(X)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a finite real matrix with 38 columns") from exc
    if matrix.ndim != 2 or matrix.shape[1] != 38 or matrix.dtype.kind not in "fiu":
        raise ValueError("X must be a finite real matrix with 38 columns")
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("X must contain only finite float64-representable values")

    try:
        boundaries = np.asarray(offsets)
    except (TypeError, ValueError) as exc:
        raise ValueError("offsets must be a finite integer-valued vector") from exc
    if boundaries.ndim != 1 or not boundaries.size or boundaries.dtype.kind not in "fiu":
        raise ValueError("offsets must be a nonempty integer-valued vector")
    # Check bounds before converting to an index dtype (no truncation/overflow).
    n = len(matrix)
    if (not np.isfinite(boundaries).all()
            or np.any(boundaries < 0) or np.any(boundaries > n)
            or boundaries[0] != 0 or boundaries[-1] != n
            or np.any(boundaries[1:] < boundaries[:-1])
            or np.any(boundaries != np.floor(boundaries))):
        raise ValueError("offsets must be nondecreasing integers starting at 0 and ending at len(X)")
    return matrix, boundaries.astype(np.intp, copy=True)


def _signed_log1p(values):
    return np.sign(values) * np.log1p(np.abs(values))


def _relative_statistics(values, offsets):
    """Gap, strict fractional rank, best-rival margin, next-distinct gap.

    A positive margin means a unique winner; tied winners and singletons have
    zero margin. Ranks count strictly smaller candidates, never row positions.
    Sorted value/count reductions make ties and candidate permutations exact.
    """
    gap = np.zeros_like(values)
    rank = np.zeros_like(values)
    margin = np.zeros_like(values)
    distinct_gap = np.zeros_like(values)
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        if lo == hi:
            continue
        group = values[lo:hi]
        unique, inverse, counts = np.unique(group, return_inverse=True, return_counts=True)
        gap[lo:hi] = group - unique[0]
        strictly_smaller = np.r_[0, np.cumsum(counts[:-1])]
        rank[lo:hi] = strictly_smaller[inverse] / max(hi - lo - 1, 1)
        rival = np.full(len(group), unique[0])
        if len(unique) > 1:
            distinct_gap[lo:hi] = unique[1] - unique[0]
            if counts[0] == 1:
                rival[inverse == 0] = unique[1]
        margin[lo:hi] = rival - group
    return gap, rank, margin, distinct_gap


def transform_accuracy_features(X, offsets, profile: str = "context") -> np.ndarray:
    """Return a new finite float64 feature matrix, without mutating either input.

    X must be a finite real (n, 38) BASE38 matrix. Observation j owns rows
    offsets[j]:offsets[j+1]; offsets must monotonically cover every row. Empty
    groups, an empty matrix with offsets=[0], and integer-valued float offsets
    are accepted. Rows may be permuted within observations equivariantly.

    Physics uses log-scaled geometry energy, strict group rank, and signed
    best-rival margin. Compact adds a small set of motion/shape signals without
    duplicating the correlated raw distance and heading columns. Context adds
    all engineered columns after the unchanged baseline prefix. Fixed scales
    (32 m^2, 10 m, 10 m/s) are physical transforms, not learned calibration.
    """
    names = feature_names(profile)
    matrix, offsets = _validated_inputs(X, offsets)
    if not len(matrix):
        return np.empty((0, len(names)), dtype=np.float64)

    def column(name):
        return matrix[:, _BASE_INDEX[name]]

    distance = np.clip(column("distance_m"), 0.0, 1e6)
    mismatch = np.clip(column("reliable_direction_mismatch"), 0.0, 2.0)
    distance_squared = distance ** 2
    energy = distance_squared + 32.0 * mismatch
    energy_gap, energy_rank, energy_margin, distinct_gap = _relative_statistics(energy, offsets)
    derived = {
        "log1p_geometry_energy_32m2": np.log1p(energy / 32.0),
        "geometry_energy_strict_rank": energy_rank,
        "signed_log1p_geometry_energy_margin_32m2": _signed_log1p(energy_margin / 32.0),
    }
    if profile == "physics":
        return np.column_stack([derived[name] for name in names])

    cosines = np.clip(matrix[:, [_BASE_INDEX["heading_cos_" + window]
                                 for window in ("short", "medium", "long")]], -1.0, 1.0)
    reliability = np.clip(matrix[:, [_BASE_INDEX["heading_reliability_" + window]
                                     for window in ("short", "medium", "long")]], 0.0, 1.0)
    window_mismatch = reliability * (1.0 - cosines)
    multiwindow_mismatch = np.mean(window_mismatch, axis=1)
    # Unreliable windows do not create spurious inter-window disagreement.
    disagreement = np.mean(np.abs(np.diff(cosines, axis=1))
                           * np.minimum(reliability[:, :-1], reliability[:, 1:]), axis=1)
    neighbour_mean = np.clip(column("neighbour_mean_lateral_distance_m"), 0.0, 1e6)
    projected_speed = np.clip(column("neighbour_projected_speed_mps"), -1000.0, 1000.0)
    motion = np.clip(column("motion_along_arc_mps"), -1000.0, 1000.0)
    endpoint = np.clip(column("distance_to_endpoint_m"), 0.0, 1e6)
    derived.update({
        "multiwindow_direction_mismatch": multiwindow_mismatch,
        "multiwindow_direction_disagreement": disagreement,
        "signed_log1p_neighbour_lateral_excess_10m": _signed_log1p((neighbour_mean - distance) / 10.0),
        "signed_sqrt_motion_along_arc_10mps": np.sign(motion) * np.sqrt(np.abs(motion) / 10.0),
        "projected_motion_disagreement": _signed_log1p((projected_speed - motion) / 10.0),
        "reliable_turn_cost": reliability[:, 1] * (1.0 - np.clip(column("gps_turn_cos"), -1.0, 1.0)) / 2.0,
        "endpoint_exposure": np.exp(-np.minimum(endpoint / 8.0, 80.0)),
        "low_motion": np.clip(column("low_motion"), 0.0, 1.0),
        "edge_is_zero_length": np.clip(column("edge_is_zero_length"), 0.0, 1.0),
    })
    if profile == "compact":
        return np.column_stack([derived[name] for name in names])

    multiwindow_energy = distance_squared + 32.0 * multiwindow_mismatch
    multi_gap, multi_rank, multi_margin, _ = _relative_statistics(multiwindow_energy, offsets)
    distance_gap, _, distance_margin, _ = _relative_statistics(distance, offsets)
    nearest_angular_cost = np.zeros(len(matrix), dtype=np.float64)
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        if lo == hi:
            continue
        nearest = distance[lo:hi] == np.min(distance[lo:hi])
        # All equal-distance nearest candidates participate; never pick a row ID.
        nearest_mismatch = np.min(mismatch[lo:hi][nearest])
        nearest_angular_cost[lo:hi] = 32.0 * (mismatch[lo:hi] - nearest_mismatch)
    derived.update({
        "geometry_energy_m2": energy,
        "geometry_energy_gap_m2": energy_gap,
        "log1p_geometry_energy_gap_32m2": np.log1p(energy_gap / 32.0),
        "geometry_energy_distinct_gap_m2": distinct_gap,
        "geometry_energy_soft_support": np.exp(-np.minimum(energy_gap / 32.0, 80.0)),
        "short_direction_mismatch": window_mismatch[:, 0],
        "long_direction_mismatch": window_mismatch[:, 2],
        "reliable_heading_angle_cost": reliability[:, 1] * (np.arccos(cosines[:, 1]) / np.pi) ** 2,
        "candidate_nearest_angular_cost_m2": nearest_angular_cost,
        "distance_heading_interaction": mismatch / (1.0 + distance_squared / 32.0),
        "direction_energy_fraction": 32.0 * mismatch / (energy + 32.0),
        "multiwindow_geometry_energy_m2": multiwindow_energy,
        "log1p_multiwindow_geometry_energy_32m2": np.log1p(multiwindow_energy / 32.0),
        "log1p_multiwindow_energy_gap_32m2": np.log1p(multi_gap / 32.0),
        "multiwindow_energy_strict_rank": multi_rank,
        "signed_log1p_multiwindow_energy_margin_32m2": _signed_log1p(multi_margin / 32.0),
        "log1p_neighbour_lateral_mean_10m": np.log1p(neighbour_mean / 10.0),
        "log1p_neighbour_lateral_max_10m": np.log1p(np.clip(column("neighbour_max_lateral_distance_m"), 0.0, 1e6) / 10.0),
        "signed_sqrt_projected_speed_10mps": np.sign(projected_speed) * np.sqrt(np.abs(projected_speed) / 10.0),
        "reliable_reverse_motion": reliability[:, 1] * np.sqrt(np.maximum(-motion, 0.0) / 10.0),
        "log1p_distance_gap_10m": np.log1p(distance_gap / 10.0),
        "signed_log1p_distance_margin_10m": _signed_log1p(distance_margin / 10.0),
    })
    return np.column_stack([matrix] + [derived[name] for name in _CONTEXT_EXTRA_NAMES])
