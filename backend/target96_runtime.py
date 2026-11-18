"""Explicit, acceptance-gated target96 runtime; never a production entrypoint.

Importing this module opens no artifacts or datasets. Receipts describe exposed
development evidence, not a blind/official 96% result or deployment permission.
The existing service owns request-local score audits and reference access.
"""
from __future__ import annotations

from copy import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from .app import create_app
from .config import Settings
from .service import InferenceRuntime
from mapmatch.candidates import CandidateIndex
from mapmatch.matcher import MatchConfig
from mapmatch.target96 import MemoIndex
from mapmatch.target96_arc import ArcPolicy, ArcPriorMatcher, HopRouter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts" / "target96" / "release_manifest.json"
_FILES = frozenset({"model.joblib", "training_metadata.json", "match_config.json", "arc_policy.json"})
_SOURCES = frozenset({
    "scripts/target96_release.py", "mapmatch/target96_arc.py", "mapmatch/target96.py",
    "mapmatch/matcher.py", "mapmatch/metrics.py", "mapmatch/features.py",
    "mapmatch/accuracy_features.py", "mapmatch/accuracy_artifacts.py",
    "mapmatch/accuracy_matcher.py", "mapmatch/learning.py", "backend/emissions.py",
})
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pin(value):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError("target96 requires lowercase SHA256 pins")
    return value


def _object(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate target96 JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite target96 JSON number: " + value)

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("target96 JSON must be an object")
    return value


def _local_file(root, name):
    path = root / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
        raise ValueError("target96 artifact must be a regular local file: " + name)
    return path


def _unit_number(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _acceptance(path, manifest_hash):
    receipt = _object(_local_file(path.parent, "acceptance.json"))
    if (receipt.get("accepted") is not True or receipt.get("target_met") is not True
            or receipt.get("release_manifest_sha256") != manifest_hash
            or not _unit_number(receipt.get("observed_edge_f1"))
            or receipt["observed_edge_f1"] < .96
            or type(receipt.get("failed_trips")) is not int or receipt["failed_trips"] != 0
            or not _unit_number(receipt.get("route_continuity")) or receipt["route_continuity"] != 1
            or not _unit_number(receipt.get("mean_length_f1"))
            or receipt.get("not_official_or_blind") is not True):
        raise ValueError("target96 release not accepted")
    _pin(receipt.get("comparison_sha256"))
    return sha256(path.parent / "acceptance.json")


def _model_role(metadata, profile, protocol):
    if not isinstance(metadata, dict):
        raise ValueError("target96 requires deployment training metadata")
    if profile == "context":
        ids = metadata.get("train_trip_ids")
        if (metadata.get("fit_role") != "all_train_deployment"
                or metadata.get("training_scope") != "all80 TRAIN final refit after frozen selection; no student/evaluator fitting"
                or metadata.get("profile") != profile or metadata.get("protocol_sha256") != protocol
                or metadata.get("validation_trip_ids") not in ([], ())):
            raise ValueError("target96 context model is not an all-TRAIN deployment role")
    else:
        ids = metadata.get("training_trip_ids")
        if type(metadata.get("training_trip_count")) is not int or metadata["training_trip_count"] != 80:
            raise ValueError("target96 v1 model must be the all80 TRAIN deployment")
    if (not isinstance(ids, (list, tuple)) or len(ids) != 80
            or any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", x) for x in ids)
            or len(set(ids)) != 80 or any(metadata.get(x) is not None for x in ("fold", "fold_id"))):
        raise ValueError("target96 deployment requires eighty unique TRAIN trips and no fold")


class _StatelessContextEmitter:
    """Reuse the validated estimator, never shared graph/track score records.

    AccuracyDeploymentModel writes .records and .probability_checks. A shallow
    per-call scoring clone keeps that state local without copying/refitting the
    estimator. The service's AuditedEmissionModel remains the sole request audit.
    """

    def __init__(self, model):
        self._template = copy(model)
        self._template.records = {}
        self._template.probability_checks = {}
        self.estimator = model.estimator
        self.metadata = dict(model.metadata)
        self.feature_names = tuple(model.feature_names)

    @property
    def classes_(self):
        return self.estimator.classes_

    @property
    def profile(self):
        return self.metadata["profile"]

    def predict_proba(self, values):
        return self._template.predict_proba(values)

    def score_candidates_batch(self, graph, track, observation_indices, candidates):
        local = copy(self._template)
        local.records = {}
        local.probability_checks = {}
        return local.score_candidates_batch(graph, track, observation_indices, candidates)

    def score_candidates(self, graph, track, xy, index, candidates):
        return self.score_candidates_batch(graph, track, [index], [candidates])[0]


def release_components(manifest_path=DEFAULT_MANIFEST):
    """Verify accepted release before returning lazy loader and service factory.

    There is deliberately no acceptance bypass and no fallback to v1 defaults.
    Artifacts/manifests are trusted administrator inputs; pickle is executable.
    """
    supplied = Path(manifest_path)
    if supplied.is_symlink():
        raise ValueError("target96 manifest must not be a symlink")
    path = supplied.resolve()
    root = path.parent
    manifest_hash = sha256(path)
    receipt_hash = _acceptance(path, manifest_hash)
    manifest = _object(path)
    files = manifest.get("files")
    if (manifest.get("schema") != "pathsnap-route96-frozen-release-v1"
            or not isinstance(files, dict) or not _FILES <= files.keys()
            or set(files) - _FILES - {"training_metrics.json"}):
        raise ValueError("target96 manifest file allowlist invalid")
    profile = manifest.get("profile")
    if (profile not in ("context", "v1") or manifest.get("training_fit_reused") is not True
            or manifest.get("student_role") != "historicallyexposeddevelopment,notblind"
            or manifest.get("evaluator_role") != "never used for training/selection"
            or manifest.get("retraining_or_selection_after_student_allowed") is not False):
        raise ValueError("target96 release profile/data roles invalid")
    for key in ("model_sha256", "training_protocol_sha256", "route_protocol_sha256",
                "frozen_selection_sha256", "baseline_model_sha256", "baseline_config_sha256"):
        _pin(manifest.get(key))
    if manifest["model_sha256"] != files["model.joblib"]:
        raise ValueError("target96 model hash aliases disagree")
    if profile == "v1" and manifest["model_sha256"] != manifest["baseline_model_sha256"]:
        raise ValueError("target96 v1 artifact is not the released baseline deployment")
    sources = manifest.get("source_sha256")
    if not isinstance(sources, dict) or not _SOURCES <= sources.keys():
        raise ValueError("target96 inference source pins missing")
    pinned = {}
    for name, digest in files.items():
        pinned[_local_file(root, name)] = _pin(digest)
    for name, digest in sources.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or relative.as_posix() != name or ".." in relative.parts
                or "\\" in name or relative.suffix != ".py"
                or relative.parts[0] not in {"mapmatch", "backend", "scripts"}):
            raise ValueError("target96 inference source path invalid")
        source = ROOT / name
        if not source.resolve().is_relative_to(ROOT.resolve()) or source.is_symlink() or not source.is_file():
            raise ValueError("target96 inference source path escapes checkout")
        pinned[source] = _pin(digest)

    def verify():
        if sha256(path) != manifest_hash or _acceptance(path, manifest_hash) != receipt_hash:
            raise ValueError("target96 accepted release changed")
        for artifact, digest in pinned.items():
            if artifact.is_symlink() or sha256(artifact) != digest:
                raise ValueError("target96 release artifact/source hash differs: " + artifact.name)

    verify()
    config_values = _object(root / "match_config.json")
    policy_values = _object(root / "arc_policy.json")
    choice = manifest.get("configuration", {})
    if (not isinstance(choice, dict) or choice.get("profile") != profile
            or choice.get("match_config") != config_values or choice.get("arc_policy") != policy_values):
        raise ValueError("target96 configuration aliases disagree")
    config = MatchConfig(**config_values)
    policy = ArcPolicy(**policy_values)
    if config.classifier_weight <= 0 or config.observation_distance_m != 0:
        raise ValueError("target96 requires learned emissions for every observation")
    _model_role(_object(root / "training_metadata.json"), profile, manifest["training_protocol_sha256"])

    def loader(model_path):
        if Path(model_path).resolve() != root / "model.joblib":
            raise ValueError("configured model differs from target96 manifest")
        verify()
        if profile == "context":
            from mapmatch.accuracy_artifacts import load_accuracy_deployment_model
            model = load_accuracy_deployment_model(
                model_path, expected_sha256=files["model.joblib"],
                expected_protocol_sha256=manifest["training_protocol_sha256"])
        else:
            from mapmatch.learning import load_emission_model
            model = load_emission_model(model_path)
        verify()
        _model_role(getattr(model, "metadata", None), profile, manifest["training_protocol_sha256"])
        if not callable(getattr(model, "score_candidates", None)) or not callable(getattr(model, "score_candidates_batch", None)):
            raise ValueError("target96 model lacks learned emission interfaces")
        return _StatelessContextEmitter(model) if profile == "context" else model

    def matcher(graph, actual_config, *, emission_model):
        if asdict(actual_config) != asdict(config):
            raise ValueError("configured decoder differs from target96 manifest")
        if not callable(getattr(emission_model, "score_candidates", None)):
            raise ValueError("target96 matcher requires the request emission audit")
        return ArcPriorMatcher(
            graph, actual_config, emission_model, policy,
            shared_index=MemoIndex(CandidateIndex(graph, actual_config.sample_spacing_m)),
            shared_router=HopRouter(graph, actual_config.cache_size),
        )

    return manifest, loader, matcher


def create_target96_app(settings=None, *, manifest_path=DEFAULT_MANIFEST):
    """Construct the existing API explicitly; never start a server or switch it."""
    path = Path(manifest_path).resolve()
    _, loader, matcher = release_components(manifest_path)
    if settings is None:
        settings = replace(
            Settings.from_env(), model_path=path.parent / "model.joblib",
            metadata_path=path.parent / "training_metadata.json",
            match_config_path=path.parent / "match_config.json", match_config_required=True,
        )
    if (settings.model_path.resolve() != path.parent / "model.joblib"
            or settings.metadata_path is None or settings.metadata_path.resolve() != path.parent / "training_metadata.json"
            or settings.match_config_path is None or settings.match_config_path.resolve() != path.parent / "match_config.json"
            or settings.match_config_required is not True):
        raise ValueError("configured artifact paths differ from target96 release")
    runtime = InferenceRuntime(settings, model_loader=loader, matcher_factory=matcher)
    return create_app(settings, runtime=runtime)
