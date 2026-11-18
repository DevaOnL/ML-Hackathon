"""Opt-in accuracy emissions and transition policies; the v1 matcher is unchanged.

Inference accepts graph/GPS inputs only. Fold artifacts are trusted local pickle
files and must be hash-pinned by the caller before loading. No fitting, dataset
discovery, target construction, or route-file loading happens in this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .candidates import Candidates
from .data import Graph, integer_array, validate_track
from .features import FEATURE_NAMES, FEATURE_VERSION, CandidateFeatureExtractor
from .matcher import MapMatcher, MatchConfig, MatchResult

ACCURACY_DECODER_VERSION = "pathsnap-accuracy-decoder-v2"
TRAINING_SCOPE = "TRAIN tuning64 fold only; no confirmation/student/evaluator"
PROBABILITY_ATOL = 1e-10
PROBABILITY_CLIP = 1e-7
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")


def accuracy_feature_names(profile: str) -> tuple[str, ...]:
    if profile == "v1":
        return FEATURE_NAMES
    from .accuracy_features import feature_names
    return tuple(feature_names(profile))


def _profile_version(profile: str) -> str:
    if profile == "v1":
        return FEATURE_VERSION
    from .accuracy_features import ACCURACY_FEATURE_VERSION
    accuracy_feature_names(profile)
    return ACCURACY_FEATURE_VERSION


def _trip_ids(values, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a nonempty list of unique trip IDs")
    if any(not isinstance(v, str) or not _ID.fullmatch(v) for v in values):
        raise ValueError(f"{name} contains invalid trip IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate trip IDs")
    return tuple(sorted(values))


def _binary_classes(estimator) -> np.ndarray:
    classes = np.asarray(getattr(estimator, "classes_", []))
    if (classes.ndim != 1 or classes.shape != (2,)
            or classes.dtype.kind not in "iuf"
            or set(classes.tolist()) != {0, 1}):
        raise ValueError("estimator must expose exactly fitted binary classes 0 and 1")
    return classes


def _feature_matrix(X, columns: int, name: str = "features") -> np.ndarray:
    raw = np.asarray(X)
    if raw.ndim != 2 or raw.shape[1] != columns or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real matrix with {columns} columns")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite float64 values")
    return result


def strict_binary_probabilities(estimator, X) -> np.ndarray:
    """No clipping/renormalization until *both* raw class columns pass validation.

    Every estimator sees float64. For a Pipeline, run its preprocessing once and
    promote again at the final boundary: a custom transformer must not silently
    downcast a GaussianNB (or any other final classifier) back to float32.
    """
    classes = _binary_classes(estimator)
    columns = getattr(estimator, "n_features_in_", None)
    if isinstance(columns, (bool, np.bool_)) or not isinstance(columns, (int, np.integer)) or columns < 1:
        raise ValueError("estimator is missing fitted input feature metadata")
    values = _feature_matrix(X, int(columns))
    if not len(values):
        return np.empty((0, 2), dtype=np.float64)
    classifier = estimator
    if isinstance(estimator, Pipeline):
        if len(estimator.steps) > 1:
            values = estimator[:-1].transform(values)
        classifier = estimator.steps[-1][1]
        final_columns = getattr(classifier, "n_features_in_", None)
        if not isinstance(final_columns, (int, np.integer)) or isinstance(final_columns, (bool, np.bool_)):
            raise ValueError("pipeline classifier is missing fitted feature metadata")
        values = _feature_matrix(values, int(final_columns), "pipeline output")
        if not np.array_equal(_binary_classes(classifier), classes):
            raise ValueError("pipeline and final classifier classes disagree")
    raw = np.asarray(classifier.predict_proba(values))
    if (raw.dtype != np.dtype(np.float64) or raw.shape != (len(values), 2)
            or not np.isfinite(raw).all() or np.any(raw < 0.0) or np.any(raw > 1.0)
            or not np.allclose(raw.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_ATOL)):
        raise ValueError("invalid float64 binary probabilities; both columns must sum to 1 within 1e-10")
    return raw


def validate_accuracy_metadata(estimator, metadata: Mapping) -> dict:
    """Reject legacy, wrong-profile, unfitted, or overlapping-fold envelopes."""
    if not isinstance(metadata, Mapping):
        raise ValueError("accuracy model requires fold metadata")
    result = dict(metadata)
    profile = result.get("profile")
    if profile not in ("v1", "context", "compact", "physics"):
        raise ValueError("accuracy model profile must be canonical v1/context/compact/physics")
    names = accuracy_feature_names(profile)
    base_names, profile_names = result.get("base_feature_names"), result.get("feature_names")
    if (not isinstance(base_names, (list, tuple)) or tuple(base_names) != FEATURE_NAMES
            or result.get("base_feature_version") != FEATURE_VERSION
            or not isinstance(profile_names, (list, tuple)) or tuple(profile_names) != names
            or result.get("feature_version") != _profile_version(profile)):
        raise ValueError("accuracy model feature schema/order/version differs from installed extractor")
    for name in ("family", "run_name"):
        if not isinstance(result.get(name), str) or not _ID.fullmatch(result[name]):
            raise ValueError(f"accuracy model requires valid {name}")
    if not isinstance(result.get("parameters"), Mapping) or result.get("training_scope") != TRAINING_SCOPE:
        raise ValueError("accuracy model parameters or TRAIN-fold fit scope are missing")
    fold = result.get("fold_id")
    if isinstance(fold, (bool, np.bool_)) or not isinstance(fold, (int, np.integer)) or int(fold) not in (0, 1, 2):
        raise ValueError("accuracy model requires fold_id 0, 1, or 2")
    if "fold" in result and result["fold"] != fold:
        raise ValueError("accuracy model fold aliases disagree")
    train = _trip_ids(result.get("train_trip_ids"), "train_trip_ids")
    valid = _trip_ids(result.get("validation_trip_ids"), "validation_trip_ids")
    if set(train) & set(valid):
        raise ValueError("accuracy model train/validation trip IDs overlap")
    if not isinstance(result.get("protocol_sha256"), str) or not _HASH.fullmatch(result["protocol_sha256"]):
        raise ValueError("accuracy model requires a training protocol SHA256 pin")
    sources = result.get("source_sha256")
    if (not isinstance(sources, Mapping) or not {"common", "factory_module"}.issubset(sources)
            or any(not isinstance(v, str) or not _HASH.fullmatch(v) for v in sources.values())
            or (profile != "v1" and "features" not in sources)):
        raise ValueError("accuracy model requires complete source SHA256 pins")
    if not callable(getattr(estimator, "predict_proba", None)):
        raise ValueError("accuracy model must be a fitted probabilistic estimator")
    try:
        check_is_fitted(estimator)
    except (TypeError, ValueError) as exc:
        raise ValueError("accuracy model estimator is not fitted") from exc
    _binary_classes(estimator)
    if getattr(estimator, "n_features_in_", None) != len(names):
        raise ValueError("accuracy estimator fitted feature count differs from profile schema")
    result["fold_id"] = int(fold)
    result["train_trip_ids"], result["validation_trip_ids"] = train, valid
    result["feature_names"] = names
    return result


@dataclass(frozen=True)
class CandidateScoreRecord:
    edges: np.ndarray
    probabilities: np.ndarray
    positive_probabilities: np.ndarray


@dataclass
class AccuracyEmissionModel:
    """One transform and predict call for a full retained observation sequence.

    Binary probabilities remain raw for audits. Emissions apply the existing
    1e-7 compatibility clip to the positive column, then normalize per group.
    Candidate order and original observation indices are never changed.
    """

    estimator: object
    metadata: Mapping
    records: dict[int, CandidateScoreRecord] = field(default_factory=dict, init=False)
    probability_checks: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.metadata = validate_accuracy_metadata(self.estimator, self.metadata)

    @property
    def profile(self):
        return self.metadata["profile"]

    @property
    def classes_(self):
        return self.estimator.classes_

    @property
    def feature_names(self):
        return self.metadata["feature_names"]

    def predict_proba(self, X):
        """X must already have this profile; returns validated raw binary values."""
        return strict_binary_probabilities(self.estimator, X)

    def score_candidates_batch(self, graph: Graph, track, observation_indices,
                               candidates: Sequence[Candidates]):
        self.records = {}
        self.probability_checks = {}
        track = validate_track(track)
        indices = integer_array(observation_indices, "observation_indices")
        if (len(indices) != len(candidates) or np.any(indices < 0)
                or np.any(indices >= len(track)) or np.any(np.diff(indices) <= 0)):
            raise ValueError("observation indices must be unique, increasing, and match candidate groups")
        for group in candidates:
            edges = integer_array(group.edges, "candidate edges")
            fractions = np.asarray(group.fractions)
            distances = np.asarray(group.distances)
            if (not len(edges) or len(set(edges.tolist())) != len(edges)
                    or np.any(edges < 0) or np.any(edges >= len(graph.arcs))
                    or fractions.shape != edges.shape or distances.shape != edges.shape
                    or fractions.dtype.kind not in "iuf" or distances.dtype.kind not in "iuf"
                    or not np.isfinite(fractions).all() or not np.isfinite(distances).all()
                    or np.any(fractions < 0) or np.any(fractions > 1) or np.any(distances < 0)):
                raise ValueError("candidate group must have unique valid edges and finite fractions/distances")
        X, offsets = CandidateFeatureExtractor(graph).transform(track, indices, candidates)
        if self.profile == "v1":
            values = _feature_matrix(X, len(FEATURE_NAMES))
        else:
            from .accuracy_features import transform_accuracy_features
            values = transform_accuracy_features(X, offsets, profile=self.profile)
        probabilities = self.predict_proba(values)
        positive = probabilities[:, int(np.flatnonzero(_binary_classes(self.estimator) == 1)[0])]
        clipped = np.clip(positive, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
        self.probability_checks = {
            "maximum_row_sum_error": float(np.max(np.abs(probabilities.sum(axis=1) - 1.0))) if len(probabilities) else 0.0,
            "clipped_low_rows": int(np.sum(positive < PROBABILITY_CLIP)),
            "clipped_high_rows": int(np.sum(positive > 1.0 - PROBABILITY_CLIP)),
            "feature_rows": len(values), "feature_columns": len(self.feature_names),
            "float64_input": True, "uses_reference_route": False,
        }
        result = []
        for index, group, lo, hi in zip(indices, candidates, offsets[:-1], offsets[1:]):
            row = clipped[lo:hi]
            total = float(np.sum(row, dtype=np.float64))
            if not math.isfinite(total) or total <= 0:
                raise ValueError("candidate positive probabilities require a finite positive group sum")
            normalized = row / total
            result.append(normalized)
            self.records[int(index)] = CandidateScoreRecord(group.edges.copy(), normalized.copy(), positive[lo:hi].copy())
        return result


def load_accuracy_emission_model(path: str | Path, *, expected_sha256: str,
                                 expected_fold_id: int, expected_train_trip_ids,
                                 expected_validation_trip_ids,
                                 expected_protocol_sha256: str | None = None) -> AccuracyEmissionModel:
    """Load only a trusted, hash-pinned common fold artifact, never a v1 bundle.

    Hash the exact bytes later unpickled through the same open descriptor.
    Expected trip partitions are mandatory; a matching filename is not evidence
    that a model was trained on the other folds.
    """
    if not isinstance(expected_sha256, str) or not _HASH.fullmatch(expected_sha256):
        raise ValueError("expected model SHA256 is required before loading pickle")
    if (isinstance(expected_fold_id, (bool, np.bool_))
            or not isinstance(expected_fold_id, (int, np.integer)) or expected_fold_id not in (0, 1, 2)):
        raise ValueError("expected fold ID must be 0, 1, or 2")
    train = _trip_ids(expected_train_trip_ids, "expected train trip IDs")
    valid = _trip_ids(expected_validation_trip_ids, "expected validation trip IDs")
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("accuracy fold model SHA256 differs from frozen manifest")
        stream.seek(0)
        bundle = joblib.load(stream)
    if not isinstance(bundle, dict) or "estimator" not in bundle:
        raise ValueError("not a common accuracy fold artifact")
    model = AccuracyEmissionModel(bundle["estimator"], {k: v for k, v in bundle.items() if k != "estimator"})
    if (model.metadata["fold_id"] != expected_fold_id
            or model.metadata["train_trip_ids"] != train
            or model.metadata["validation_trip_ids"] != valid):
        raise ValueError("accuracy fold identity/training/validation trips differ from expected partition")
    if expected_protocol_sha256 is not None and model.metadata["protocol_sha256"] != expected_protocol_sha256:
        raise ValueError("accuracy fold training protocol pin differs")
    return model


@dataclass(frozen=True)
class AccuracyTransitionPolicy:
    path_cost_slack_m: float = 0.0
    stationary_chord_m: float = 3.0
    stationary_backward_deadband_m: float = 1.0
    stationary_backtrack_weight: float = 0.0
    immediate_reverse_penalty: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def route_motion_diagnostics(graph: Graph, track, result: MatchResult, *, stationary_chord_m: float = 3.0) -> dict:
    """Graph/GPS-only counters; repeated traversals are reported, not removed."""
    xy = graph.project(np.asarray(track)[result.observation_indices, :2])
    chords = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    edges = result.matched_edges
    same = edges[:-1] == edges[1:]
    backward = np.maximum((result.matched_fractions[:-1] - result.matched_fractions[1:])
                          * graph.edge_lengths[edges[:-1]], 0.0)
    backward = np.where(same, backward, 0.0)
    stationary = chords <= stationary_chord_m
    def reverses(route):
        a, b = route[:-1], route[1:]
        return (a != b) & (graph.tails[a] == graph.heads[b]) & (graph.heads[a] == graph.tails[b])
    return {
        "stationary_matched_pairs": int(np.sum(stationary)),
        "same_edge_backward_pairs": int(np.sum(backward > 1e-9)),
        "same_edge_backward_distance_m": float(np.sum(backward)),
        "stationary_backward_distance_m": float(np.sum(backward[stationary])),
        "matched_immediate_reverse_pairs": int(np.sum(reverses(edges))),
        "route_immediate_reverse_pairs": int(np.sum(reverses(result.route))),
        "route_repeated_traversal_count": int(len(result.route) - len(np.unique(result.route))),
    }


class AccuracyMapMatcher(MapMatcher):
    """Reuse v1 DP, candidate search, retry semantics, and route reconstruction.

    Only transition costs are optionally changed. Reachability, original arc IDs
    and actual cycles are preserved. No postprediction route editing occurs.
    """

    def __init__(self, graph: Graph, config: MatchConfig | None = None, emission_model=None,
                 transition_policy: AccuracyTransitionPolicy | None = None):
        super().__init__(graph, config, emission_model)
        self.transition_policy = transition_policy or AccuracyTransitionPolicy()

    def _transitions(self, previous: Candidates, current: Candidates, chord: float, horizon: float):
        cost = super()._transitions(previous, current, chord, horizon)
        policy = self.transition_policy
        if policy.path_cost_slack_m:
            cost = np.maximum(cost - policy.path_cost_slack_m / self.config.beta_m, 0.0)
        if chord <= policy.stationary_chord_m:
            a, b = previous.edges[:, None], current.edges[None, :]
            if policy.stationary_backtrack_weight:
                backwards = ((previous.fractions[:, None] - current.fractions[None, :])
                             * self.graph.edge_lengths[a])
                penalty = np.maximum(backwards - policy.stationary_backward_deadband_m, 0.0)
                cost = cost + np.where(a == b, policy.stationary_backtrack_weight * penalty / self.config.beta_m, 0.0)
            if policy.immediate_reverse_penalty:
                reverse = ((a != b) & (self.graph.tails[a] == self.graph.heads[b])
                           & (self.graph.heads[a] == self.graph.tails[b]))
                cost = cost + policy.immediate_reverse_penalty * reverse
        return cost

    def match(self, track) -> MatchResult:
        result = super().match(track)
        result.diagnostics.update({
            "accuracy_decoder_version": ACCURACY_DECODER_VERSION,
            "accuracy_transition_policy": asdict(self.transition_policy),
            "accuracy_motion": route_motion_diagnostics(self.graph, track, result,
                stationary_chord_m=self.transition_policy.stationary_chord_m),
            "viterbi_and_reconstruction": "unchanged mapmatch.matcher.MapMatcher",
        })
        return result
