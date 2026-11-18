"""Hash-pinned non-OOF accuracy artifacts; no fitting or dataset discovery.

Confirmation is internal, historically exposed TRAIN evidence, never a new blind
test. Deployment is an all-TRAIN refit, never an OOF model. Pickle remains trusted
code: a caller must obtain its SHA256 from an independently trusted manifest.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path, PurePosixPath
import re
from typing import Mapping

import joblib
import numpy as np
from sklearn.utils.validation import check_is_fitted

from .accuracy_features import ACCURACY_FEATURE_VERSION
from .accuracy_matcher import AccuracyEmissionModel, _binary_classes, accuracy_feature_names
from .features import FEATURE_NAMES, FEATURE_VERSION

__all__ = [
    "AccuracyConfirmationModel", "AccuracyDeploymentModel",
    "load_accuracy_confirmation_model", "load_accuracy_deployment_model",
    "validate_accuracy_confirmation_metadata", "validate_accuracy_deployment_metadata",
]

CONFIRMATION_ROLE = "internal_confirmation16"
DEPLOYMENT_ROLE = "all_train_deployment"
CONFIRMATION_TRAINING_SCOPE = (
    "TRAIN64 fitting, reserved16 one-shot internal confirmation; never historicallyunseen"
)
DEPLOYMENT_TRAINING_SCOPE = (
    "all80 TRAIN final refit after frozen selection; no student/evaluator fitting"
)
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
# Repository-relative keys emitted by accuracy_confirmation.freeze_selection.
# Additional frozen dependencies are accepted, but OOF alias keys are not paths.
REQUIRED_SOURCE_PATHS = frozenset({
    "scripts/accuracy_common.py", "scripts/accuracy_trees.py",
    "scripts/accuracy_linear_bayes.py", "scripts/accuracy_knn.py",
    "scripts/accuracy_confirmation.py", "scripts/accuracy_decoder.py",
    "scripts/accuracy_route_validation.py", "mapmatch/accuracy_features.py",
    "mapmatch/accuracy_matcher.py", "mapmatch/features.py", "mapmatch/learning.py",
    "mapmatch/experiments.py", "mapmatch/phase_models.py", "mapmatch/data.py",
    "mapmatch/metrics.py", "mapmatch/matcher.py", "mapmatch/routing.py",
    "mapmatch/candidates.py",
})


def _sha256(value, name: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA256 pin")
    return value


def _trip_ids(values, name: str, count: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"{name} must contain exactly {count} unique trip IDs")
    if any(not isinstance(value, str) or not _ID.fullmatch(value) for value in values):
        raise ValueError(f"{name} contains invalid trip IDs")
    if len(set(values)) != count:
        raise ValueError(f"{name} contains duplicate trip IDs")
    return tuple(sorted(values))


def _confirmation_partition(train, valid, prefix: str = ""):
    train = _trip_ids(train, prefix + "train_trip_ids", 64)
    valid = _trip_ids(valid, prefix + "validation_trip_ids", 16)
    if set(train) & set(valid):
        raise ValueError("confirmation train/validation trip IDs overlap")
    return train, valid


def _source_path(value) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        return False
    path = PurePosixPath(value)
    return (not path.is_absolute() and len(path.parts) >= 2
            and ".." not in path.parts and path.as_posix() == value)


def _common_metadata(estimator, metadata: Mapping) -> dict:
    if not isinstance(metadata, Mapping):
        raise ValueError("accuracy role model requires metadata")
    result = dict(metadata)
    profile = result.get("profile")
    if not isinstance(profile, str) or profile not in ("v1", "context", "compact", "physics"):
        raise ValueError("accuracy model profile must be canonical v1/context/compact/physics")
    names = accuracy_feature_names(profile)
    base_names, profile_names = result.get("base_feature_names"), result.get("feature_names")
    version = FEATURE_VERSION if profile == "v1" else ACCURACY_FEATURE_VERSION
    if (not isinstance(base_names, (list, tuple)) or tuple(base_names) != FEATURE_NAMES
            or result.get("base_feature_version") != FEATURE_VERSION
            or not isinstance(profile_names, (list, tuple)) or tuple(profile_names) != names
            or result.get("feature_version") != version):
        raise ValueError("accuracy model feature schema/order/version differs from installed extractor")
    for name in ("family", "run_name"):
        if not isinstance(result.get(name), str) or not _ID.fullmatch(result[name]):
            raise ValueError(f"accuracy model requires valid {name}")
    if not isinstance(result.get("parameters"), Mapping):
        raise ValueError("accuracy model requires parameters metadata")
    _sha256(result.get("protocol_sha256"), "protocol_sha256")
    sources = result.get("source_sha256")
    if (not isinstance(sources, Mapping) or not REQUIRED_SOURCE_PATHS.issubset(sources)
            or any(not _source_path(path) for path in sources)):
        raise ValueError("accuracy role model requires complete repository-relative source_sha256 pins")
    for digest in sources.values():
        _sha256(digest, "source_sha256 value")
    if not callable(getattr(estimator, "predict_proba", None)):
        raise ValueError("accuracy model must be a fitted probabilistic estimator")
    try:
        check_is_fitted(estimator)
    except (TypeError, ValueError) as exc:
        raise ValueError("accuracy model estimator is not fitted") from exc
    _binary_classes(estimator)
    columns = getattr(estimator, "n_features_in_", None)
    if (isinstance(columns, (bool, np.bool_)) or not isinstance(columns, (int, np.integer))
            or columns != len(names)):
        raise ValueError("accuracy estimator fitted feature count differs from profile schema")
    result["feature_names"] = names
    result["source_sha256"] = dict(sources)
    return result


def validate_accuracy_confirmation_metadata(estimator, metadata: Mapping) -> dict:
    """Validate the existing fold-3 TRAIN64/reserved16 envelope without relabeling."""
    result = _common_metadata(estimator, metadata)
    if (result.get("training_scope") != CONFIRMATION_TRAINING_SCOPE
            or ("fit_role" in result and result["fit_role"] != CONFIRMATION_ROLE)):
        raise ValueError("accuracy model is not an internal_confirmation16 fit")
    for name in ("fold_id", "fold"):
        if name == "fold" and name not in result:
            continue
        fold = result.get(name)
        if (isinstance(fold, (bool, np.bool_)) or not isinstance(fold, (int, np.integer))
                or fold != 3):
            raise ValueError("internal confirmation requires fold_id 3 and matching fold alias")
    train, valid = _confirmation_partition(result.get("train_trip_ids"), result.get("validation_trip_ids"))
    result["train_trip_ids"], result["validation_trip_ids"] = train, valid
    return result


def validate_accuracy_deployment_metadata(estimator, metadata: Mapping) -> dict:
    """Validate a final all80 TRAIN refit, never an OOF/confirmation substitute."""
    result = _common_metadata(estimator, metadata)
    if (result.get("fit_role") != DEPLOYMENT_ROLE
            or result.get("training_scope") != DEPLOYMENT_TRAINING_SCOPE):
        raise ValueError("accuracy model is not an all_train_deployment fit")
    if any(result.get(name) is not None for name in ("fold_id", "fold")):
        raise ValueError("all-TRAIN deployment must not declare a fold")
    result["train_trip_ids"] = _trip_ids(result.get("train_trip_ids"), "train_trip_ids", 80)
    result["validation_trip_ids"] = _trip_ids(result.get("validation_trip_ids"), "validation_trip_ids", 0)
    return result


class _FullTrackSingleCandidateAdapter:
    def score_candidates(self, graph, track, xy, index: int, candidates):
        """One group, unchanged fulltrack motion windows; API compatibility only."""
        return self.score_candidates_batch(graph, track, np.array([index], dtype=np.int64), [candidates])[0]


class AccuracyConfirmationModel(_FullTrackSingleCandidateAdapter, AccuracyEmissionModel):
    """Internal TRAIN16 confirmation; inference is inherited unchanged."""

    def __post_init__(self):
        self.metadata = validate_accuracy_confirmation_metadata(self.estimator, self.metadata)


class AccuracyDeploymentModel(_FullTrackSingleCandidateAdapter, AccuracyEmissionModel):
    """Final all-TRAIN deployment; inference is inherited unchanged."""

    def __post_init__(self):
        self.metadata = validate_accuracy_deployment_metadata(self.estimator, self.metadata)


def _load_pinned_bundle(path: str | Path, expected_sha256: str) -> dict:
    _sha256(expected_sha256, "expected model SHA256")
    # Read once, then hash and unpickle that immutable byte snapshot. Reopening,
    # seeking a mutable on-disk file, or substituting a filename cannot swap it.
    with Path(path).open("rb") as stream:
        payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("accuracy role model SHA256 differs from frozen manifest")
    bundle = joblib.load(io.BytesIO(payload))
    if not isinstance(bundle, dict) or "estimator" not in bundle:
        raise ValueError("not an accuracy role artifact dictionary with estimator")
    return bundle


def load_accuracy_confirmation_model(
    path: str | Path, *, expected_sha256: str, expected_train_trip_ids,
    expected_validation_trip_ids, expected_protocol_sha256: str,
) -> AccuracyConfirmationModel:
    """Load a trusted TRAIN64/reserved16 model pinned to its exact partition.

    The protocol pin and both expected populations are mandatory. This does not
    imply historical non-exposure or authorize another confirmation/tuning pass.
    """
    _sha256(expected_sha256, "expected model SHA256")
    protocol = _sha256(expected_protocol_sha256, "expected protocol SHA256")
    train, valid = _confirmation_partition(
        expected_train_trip_ids, expected_validation_trip_ids, "expected ")
    bundle = _load_pinned_bundle(path, expected_sha256)
    model = AccuracyConfirmationModel(
        bundle["estimator"], {key: value for key, value in bundle.items() if key != "estimator"})
    if (model.metadata["train_trip_ids"] != train
            or model.metadata["validation_trip_ids"] != valid):
        raise ValueError("confirmation training/validation trips differ from expected partition")
    if model.metadata["protocol_sha256"] != protocol:
        raise ValueError("confirmation training protocol pin differs")
    return model


def load_accuracy_deployment_model(
    path: str | Path, *, expected_sha256: str, expected_protocol_sha256: str | None = None,
) -> AccuracyDeploymentModel:
    """Load a trusted all80 final refit; protocol pinning is optional, not inferred.

    The artifact must always contain a valid protocol pin. No TRAIN manifest,
    data, cache, student inputs, evaluator inputs, or source files are opened.
    """
    _sha256(expected_sha256, "expected model SHA256")
    if expected_protocol_sha256 is not None:
        _sha256(expected_protocol_sha256, "expected protocol SHA256")
    bundle = _load_pinned_bundle(path, expected_sha256)
    model = AccuracyDeploymentModel(
        bundle["estimator"], {key: value for key, value in bundle.items() if key != "estimator"})
    if (expected_protocol_sha256 is not None
            and model.metadata["protocol_sha256"] != expected_protocol_sha256):
        raise ValueError("deployment training protocol pin differs")
    return model
