"""Opt-in, hash-bound accuracy release integration; default v1 remains unchanged.

Create the existing API with this module only after the separately recorded
release acceptance gate. No server starts or private data is opened on import.
"""
from __future__ import annotations
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path

from .app import create_app
from .service import InferenceRuntime
from .config import Settings
from mapmatch.accuracy_matcher import AccuracyMapMatcher,AccuracyTransitionPolicy
from mapmatch.matcher import MatchConfig

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST=ROOT/"artifacts"/"accuracy_improvement"/"release_manifest.json"


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as s:
        for b in iter(lambda:s.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def release_components(manifest_path=DEFAULT_MANIFEST):
    path=Path(manifest_path).resolve();root=path.parent;manifest=json.loads(path.read_text())
    if manifest.get("schema")!="pathsnap-accuracy-export-v1" or set(manifest.get("files",{}))!={"model.joblib","match_config.json","transition_policy.json","model_metadata.json"}:raise ValueError("accuracy manifest file allowlist invalid")
    for name,digest in manifest["files"].items():
        if sha256(root/name)!=digest:raise ValueError("accuracy release artifact hash differs: "+name)
    if manifest["model_sha256"]!=manifest["files"]["model.joblib"]:raise ValueError("model hash aliases disagree")
    for relative,digest in manifest["source_sha256"].items():
        source=ROOT/relative
        if not source.resolve().is_relative_to(ROOT) or sha256(source)!=digest:raise ValueError("accuracy inference source drift")
    config=MatchConfig(**json.loads((root/"match_config.json").read_text()));policy=AccuracyTransitionPolicy(**json.loads((root/"transition_policy.json").read_text()))
    def loader(model_path):
        if Path(model_path).resolve()!=root/"model.joblib":raise ValueError("configured model differs from accuracy release manifest")
        from mapmatch.accuracy_artifacts import load_accuracy_deployment_model
        model=load_accuracy_deployment_model(model_path,expected_sha256=manifest["model_sha256"],expected_protocol_sha256=manifest["protocol_sha256"])
        if not callable(getattr(model,"score_candidates",None)):raise ValueError("accuracy artifact requires fulltrack one-observation adapter")
        return model
    def matcher(graph,actual_config,*,emission_model):
        if asdict(actual_config)!=asdict(config):raise ValueError("configured decoder differs from accuracy release manifest")
        return AccuracyMapMatcher(graph,actual_config,emission_model,policy)
    return manifest,loader,matcher


def create_accuracy_app(settings=None,*,manifest_path=DEFAULT_MANIFEST,require_acceptance=True):
    path=Path(manifest_path).resolve()
    if settings is None:
        settings=replace(Settings.from_env(),model_path=path.parent/"model.joblib",metadata_path=path.parent/"model_metadata.json",match_config_path=path.parent/"match_config.json",match_config_required=True)
    if require_acceptance:
        receipt=path.parent/"acceptance.json"
        a=json.loads(receipt.read_text())
        if a.get("accepted") is not True or a.get("release_manifest_sha256")!=sha256(path) or a.get("student_development_used_for_tuning") is not False or a.get("evaluator_used_for_tuning") is not False:raise ValueError("accuracy production release not accepted")
    _,loader,matcher=release_components(path)
    runtime=InferenceRuntime(settings,model_loader=loader,matcher_factory=matcher)
    return create_app(settings,runtime=runtime)
