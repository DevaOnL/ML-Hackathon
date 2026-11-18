#!/usr/bin/env python3
"""Prepare the git-excluded inputs that experiments/phase_three/run_phase_three.py
requires, without running the out-of-scope shared five-family pipeline.

Scoped reproduction tool (tasktask-21). It covers the shared inputs of the
selected phase families only; no out-of-phase model is ever fitted and
mapmatch.experiments.run_experiments is never called.

Stages
------
check    Read-only inventory and integrity verification. Builds nothing, fits
         nothing, writes nothing. Hashes raw trip files BYTES for the cache
         fingerprint (.route label content is never parsed or interpreted);
         student raw content is hashed only with --hash-student-raw, kept for
         use after the phase freeze.
train    Rebuild the TRAIN dataset cache, the shared group folds and the
         logistic-regression grouped-CV OOF record when those root files are
         missing. Refuses to overwrite anything that already exists.
student  Rebuild the student-test cache and the LR student predictions. Gated
         on experiments/phase_three/selection.json AND a valid
         artifacts/phase_three/model_freeze.json; no student data is touched
         before both gates pass. LR predictions come from a read-only load of
         artifacts/baselines/logistic_regression.joblib; --allow-lr-fit opts in
         to a fresh LR-only all-80 fit when that baseline is absent.

Exit codes: 0 ready / nothing to do, 2 required inputs missing,
1 verification failure or refusal. Failures are reported honestly with
per-array digests; a mismatch is never silently accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from mapmatch.data import discover_trips
from mapmatch.experiments import (
    CandidateDataset,
    ExperimentConfig,
    _fit,
    _input_manifest,
    _metrics,
    build_dataset,
    build_group_folds,
    make_estimator,
    save_json,
)
from mapmatch.features import FEATURE_NAMES, FEATURE_VERSION, PROXY_LABEL_POLICY
from mapmatch.learning import load_model, predict_positive_proba

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_MISSING = 2

# The cache fingerprint was produced with the shared run_experiments defaults.
# Its HPO budget entry is bayesian_trials=12 -- the saved shared-run value, not
# the phase runner's HPO budget of 8. Changing it changes the fingerprint.
SHARED_CONFIG = ExperimentConfig(
    seed=20261118,
    train_observations_per_trip=350,
    student_observations_per_trip=0,
    folds=3,
    bayesian_trials=12,
    n_jobs=2,
    candidate_radius_m=80.0,
    max_radius_m=1000.0,
    max_candidates=12,
    sample_spacing_m=40.0,
)

LR_PARAMETERS = {"C": 1.0, "max_iter": 500, "solver": "lbfgs"}
METRIC_TOLERANCE = 1e-9

# Phase scope (docs/PHASE_SCOPE.json): the three selected families.
PHASE_FAMILIES = ("logistic_regression", "knn_distance_weighted", "bagging")

# Original shared run runtime (experiments/run_config.json). The current
# environment may differ (numpy was 2.5.2 then, requirements.txt pins 2.5.5);
# every report below prints both so SHA deviations stay attributable.
ORIGINAL_RUN_VERSIONS = {
    "python": "3.14.6",
    "numpy": "2.5.2",
    "scipy": "1.18.0",
    "sklearn": "1.8.0",
    "joblib": "1.5.3",
}

# Per-array sha256 of the two frozen shared caches, measured read-only on the
# historical files (2026-11-18). Used to diagnose container-level SHA
# mismatches honestly: identical arrays under a different numpy zip layout
# still fail the runner's pinned-SHA assertion, and that is reported, never
# hidden.
TRAIN_ARRAY_SHA256 = {
    "X": {"shape": [269463, 38], "dtype": "float32", "sha256": "23050a48c3a32c7277e9756d4b97837836f16b64130d5762c422cd5e92c88981"},
    "y": {"shape": [269463], "dtype": "int8", "sha256": "ffd0673ff3ca20efc06e3a13a565a3ecb7ed5a43a6b9209558a6a2f9574472d3"},
    "offsets": {"shape": [28001], "dtype": "int64", "sha256": "cfefeb1e669736e097b472e2da831d34a1cf0e85907b6781a6d6653a11894809"},
    "observation_trip_ids": {"shape": [28000], "dtype": "<U32", "sha256": "5f2476e59cb544fc562ea64527dbb815058bd7dbbcf891ec8967ea02d073c651"},
    "observation_indices": {"shape": [28000], "dtype": "int64", "sha256": "b855de75513611a1d6088564229ef44c664b4728ff53849270c762ccd5407e10"},
    "reference_edges": {"shape": [28000], "dtype": "int64", "sha256": "f671261ef18e7b9a0ad53ea307819dea9feae2315a13e5e967866ccf5bdc2323"},
    "candidate_edges": {"shape": [269463], "dtype": "int64", "sha256": "3b1f00a07eac0ae9578fedb23bce73fb4ce255c54e9d03ad52bb76709b71645c"},
    "candidate_fractions": {"shape": [269463], "dtype": "float64", "sha256": "63047a115036d75a03462eddfe714daa9cdfd68d6f7541c48f27937fa62446ba"},
    "candidate_distances": {"shape": [269463], "dtype": "float64", "sha256": "d69da451cafacfd131571a70b02e09d8d1fe1cf7181136c3f406605920466a56"},
    "proxy_distances": {"shape": [28000], "dtype": "float64", "sha256": "30aed1b00e668fadeb3745265c0bfdeab77c9cdf79cca662ee57e11ff6febe7c"},
    "ambiguous": {"shape": [28000], "dtype": "bool", "sha256": "9d91e517f74182ed79727cf519912b27870e206a7ef755f418e256c66a71fa8b"},
    "expanded": {"shape": [28000], "dtype": "bool", "sha256": "229744afae78c459aec2d3d441a4bd83264295a64efc8458fbb00ec1bea8af26"},
}

STUDENT_ARRAY_SHA256 = {
    "X": {"shape": [175778, 38], "dtype": "float32", "sha256": "839c019acc595fe4416a207f2d5f041bc5577b0407991bd4e3d3829d30cc8dc8"},
    "y": {"shape": [175778], "dtype": "int8", "sha256": "f887251b4bfa3fbe54ae7a057ef0e516c25f7f248c812b5d4d56188208f86d99"},
    "offsets": {"shape": [19161], "dtype": "int64", "sha256": "9b3b12a1bc7aa8f3b7617f3ad37eb1740668b50861823e1815e1916cf115e16a"},
    "observation_trip_ids": {"shape": [19160], "dtype": "<U32", "sha256": "de021ea33537a78c7fc2644d40c179d146847efb458c8897b3bb93e39c5b310b"},
    "observation_indices": {"shape": [19160], "dtype": "int64", "sha256": "b6b18533863fb50eb33f0a84f0f0acdc99f6c2f37c633dad3b20dd4b84deb740"},
    "reference_edges": {"shape": [19160], "dtype": "int64", "sha256": "44670e1099c6fc9c7271120d178df0520e5574fe4e212cf3cf1312374e8deb1b"},
    "candidate_edges": {"shape": [175778], "dtype": "int64", "sha256": "9b0204d6b60cd64e011e070c9fa9d69808f6a35b4b61988501069c13a422cbfb"},
    "candidate_fractions": {"shape": [175778], "dtype": "float64", "sha256": "ef89d7bf2d18c03c1cab89687dafbb175b3a2343e2c027f3fe3c374209e836d6"},
    "candidate_distances": {"shape": [175778], "dtype": "float64", "sha256": "cd7fdb3a8923e7716894a5599c318aeeb85e0d96e769a8b51b207fe24dba519f"},
    "proxy_distances": {"shape": [19160], "dtype": "float64", "sha256": "7e7f678267512c0102b79f115b5a05c881b85f41ee8214713d64f360abcd1e5a"},
    "ambiguous": {"shape": [19160], "dtype": "bool", "sha256": "dae3d0aa51dfc31f56593dd31bc5840a88aa6dc34ea90072fbb19b52746b1639"},
    "expanded": {"shape": [19160], "dtype": "bool", "sha256": "87c42a054d2f98a7f3e9dd9080babe76a9e1588a54edfcfc9da180297bc9046a"},
}

TRAIN_SPEC = {
    "split": "train",
    "trips": 80,
    "observations": 28000,
    "rows": 269463,
    "misses": 578,
    "cache": "train_edee9a8f151dd25e",
    "fingerprint": "edee9a8f151dd25e816cb180eba44cf82916f8e9ef3da7126f6f63f93d47c278",
    "sha256": "db523c112a6119f16b6f473d6221e3c56b6730ccb2db64dee55433aedbf3ba9e",
    "array_sha256": TRAIN_ARRAY_SHA256,
}

STUDENT_SPEC = {
    "split": "student_test",
    "trips": 10,
    "observations": 19160,
    "rows": 175778,
    "misses": 19,
    "cache": "student_test_240a8fbfc79205b3",
    "fingerprint": "240a8fbfc79205b39b781c97aa0c5281a56ecadce3042516071db33e8728bb0e",
    "sha256": "ae96abb4784501b2b086c9b4aa2f5668e1c9d7f29abda907b7f402118af81b73",
    "array_sha256": STUDENT_ARRAY_SHA256,
}

# Reference sha256 of the committed shared records, measured 2026-11-18 on the
# current project tree. Informational only: the phase runner pins the two cache
# SHAs and validates these records semantically, so a difference here is
# reported as a warning rather than a failure.
REFERENCE_SHAS = {
    "experiments/cv_folds.json": "175e9476604746bc2d801c4d2e0eb5a2a5876ecf4f95d730443cd5841d5fd5fa",
    "experiments/cv/logistic_regression.json": "ae4bb5a57b1ef3c22d388d189260c7e93353e6a0887d5c9767434b08fdd4815b",
    "experiments/evaluation/logistic_regression.json": "9c385f4ea675dce282a887d4c43ead3b0b777fad649456a2917fd7febcc8cb61",
    "experiments/predictions/logistic_regression_oof.npz": "937a54e0139b90c999edfb8675eb2afe71a1ffd7d75a61f343c104b7d9bc6432",
    "experiments/predictions/logistic_regression_student.npz": "31562ba3540e4d7bab0c07d7b5a30dbb05f524ccde4fbb5967b8d041201d3c9b",
    "experiments/results.json": "4f308d422263e313b13253e2bca2267dba6a445303fbc7a404d9c44d9d1c1198",
    "artifacts/baselines/logistic_regression.joblib": "29157349d52a993d33cae78ed244805a9f087e79b6dc914aaa3b4461c02a58c9",
}


class PrepError(RuntimeError):
    """Honest failure: a gate, guard or verification did not pass."""


def note(message: str) -> None:
    print(f"[phase-prep] {message}", flush=True)


def runtime_versions() -> dict:
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def report_versions() -> dict:
    current = runtime_versions()
    note(f"runtime now: {json.dumps(current, sort_keys=True)}")
    note(f"runtime of the original shared run: {json.dumps(ORIGINAL_RUN_VERSIONS, sort_keys=True)}")
    drifted = {k: (ORIGINAL_RUN_VERSIONS.get(k), v) for k, v in current.items() if ORIGINAL_RUN_VERSIONS.get(k) != v}
    if drifted:
        note(f"version drift vs original run: {json.dumps(drifted, sort_keys=True)} "
             "(npz container SHAs may differ even when array content matches; see docs/PHASE_REPRODUCTION.md)")
    return current


def paths(root: Path) -> dict:
    experiments = root / "experiments"
    return {
        "cache_dir": experiments / "cache",
        "cv_folds": experiments / "cv_folds.json",
        "cv_lr": experiments / "cv" / "logistic_regression.json",
        "oof_lr": experiments / "predictions" / "logistic_regression_oof.npz",
        "eval_lr": experiments / "evaluation" / "logistic_regression.json",
        "student_lr": experiments / "predictions" / "logistic_regression_student.npz",
        "results": experiments / "results.json",
        "baseline": root / "artifacts" / "baselines" / "logistic_regression.joblib",
        "selection": experiments / "phase_three" / "selection.json",
        "freeze": root / "artifacts" / "phase_three" / "model_freeze.json",
        "frozen_model": root / "artifacts" / "phase_three" / "model.joblib",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def npz_array_digests(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: {
                "shape": list(archive[key].shape),
                "dtype": str(archive[key].dtype),
                "sha256": hashlib.sha256(np.ascontiguousarray(archive[key]).tobytes()).hexdigest(),
            }
            for key in archive.files
        }


def recompute_fingerprint(data_root: Path, spec: dict, config: ExperimentConfig | None = None) -> str:
    """Recompute the build_dataset cache fingerprint from the raw trip files.

    Reads each trip's .track/.nodes/.arcs/.route BYTES for hashing only. Route
    label content is never parsed or interpreted, so the TRAIN check never uses
    raw-data labels.
    """
    if config is None:
        config = SHARED_CONFIG
    split_dir = Path(data_root) / spec["split"]
    try:
        prefixes = discover_trips(split_dir)
    except (NotADirectoryError, ValueError) as exc:
        raise PrepError(f"raw data unusable at {split_dir}: {exc}") from exc
    if len(prefixes) != spec["trips"]:
        raise PrepError(f"{split_dir}: expected {spec['trips']} complete trips, found {len(prefixes)}")
    manifest = _input_manifest(prefixes)
    policy = {
        "feature_version": FEATURE_VERSION,
        "proxy": PROXY_LABEL_POLICY,
        "config": asdict(config),
        "split": spec["split"],
        "manifest": manifest,
    }
    payload = json.dumps(policy, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def observation_target_misses(y, offsets) -> int:
    """Observations whose candidate set lacks the reference target arc.

    Mirrors the phase runner's post-D002 check (cumsum over per-row positives),
    which counts target-arc misses, not empty candidate sets.
    """
    y = np.asarray(y)
    offsets = np.asarray(offsets)
    positive = np.concatenate([[0], np.cumsum(y == 1)])
    segment_positives = positive[offsets[1:]] - positive[offsets[:-1]]
    return int((segment_positives == 0).sum())


def load_verified_dataset(root: Path, spec: dict) -> CandidateDataset:
    """Read-only load of a shared cache with the runner's pinned assertions."""
    cache = paths(root)["cache_dir"] / f"{spec['cache']}.npz"
    if not cache.exists():
        raise PrepError(f"shared cache {cache.name} not found")
    observed = sha256_file(cache)
    if observed != spec["sha256"]:
        raise PrepError(f"shared cache {cache.name} sha256 {observed} != pinned {spec['sha256']}")
    coverage = json.loads(cache.with_suffix(".json").read_text())
    fields = [f for f in CandidateDataset.__dataclass_fields__ if f != "coverage"]
    with np.load(cache, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in fields}
    return CandidateDataset(**arrays, coverage=coverage)


def verify_dataset_cache(npz_path: Path, spec: dict, *, deep: bool = True) -> tuple[bool, list[str]]:
    """Verify a dataset cache against its pinned spec.

    Returns (ok, report lines). Every mismatch is reported; nothing is silently
    accepted. With deep=True the arrays are loaded and per-array digests are
    compared against the historical reference so a container-level SHA mismatch
    can be diagnosed.
    """
    ok = True
    lines = []
    observed = sha256_file(npz_path)
    if observed == spec["sha256"]:
        lines.append(f"sha256 {npz_path.name}: {observed} (pinned match)")
    else:
        ok = False
        lines.append(f"sha256 {npz_path.name}: {observed} != pinned {spec['sha256']}")
    coverage = json.loads(npz_path.with_suffix(".json").read_text())
    if coverage.get("fingerprint") != spec["fingerprint"]:
        ok = False
        lines.append(f"fingerprint {coverage.get('fingerprint')!r} != pinned {spec['fingerprint']!r}")
    counts = [
        ("split", coverage.get("split"), spec["split"]),
        ("trip_count", coverage.get("trip_count"), spec["trips"]),
        ("selected_observations", coverage.get("selected_observations"), spec["observations"]),
        ("candidate_rows", coverage.get("candidate_rows"), spec["rows"]),
    ]
    for label, observed_value, expected in counts:
        if observed_value != expected:
            ok = False
            lines.append(f"coverage {label}: {observed_value!r} != {expected!r}")
    if not deep:
        return ok, lines
    fields = [f for f in CandidateDataset.__dataclass_fields__ if f != "coverage"]
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != set(fields):
            ok = False
            lines.append(f"npz members {sorted(archive.files)} != {sorted(fields)}")
            return ok, lines
        arrays = {key: archive[key] for key in fields}
    dataset = CandidateDataset(**arrays, coverage=coverage)
    if len(dataset.offsets) - 1 != spec["observations"] or len(dataset.y) != spec["rows"]:
        ok = False
        lines.append(f"shape mismatch: observations {len(dataset.offsets) - 1}, rows {len(dataset.y)}")
    if len(np.unique(dataset.observation_trip_ids)) != spec["trips"]:
        ok = False
        lines.append(f"trip count mismatch: {len(np.unique(dataset.observation_trip_ids))}")
    misses = observation_target_misses(dataset.y, dataset.offsets)
    if misses != spec["misses"]:
        ok = False
        lines.append(f"candidate-miss count {misses} != {spec['misses']}")
    if dataset.X.shape[1] != len(FEATURE_NAMES) or str(dataset.X.dtype) != "float32":
        ok = False
        lines.append("feature schema mismatch (columns or dtype)")
    digests = npz_array_digests(npz_path)
    for key, expected in spec["array_sha256"].items():
        observed_digest = digests.get(key)
        if observed_digest is None:
            ok = False
            lines.append(f"array {key}: missing from cache")
        elif observed_digest["sha256"] != expected["sha256"]:
            ok = False
            lines.append(
                f"array {key}: sha256 {observed_digest['sha256']} != reference "
                f"{expected['sha256']} (shape {observed_digest['shape']}, dtype {observed_digest['dtype']})"
            )
    for key in digests:
        if key not in spec["array_sha256"]:
            lines.append(f"array {key}: no historical reference digest (informational)")
    return ok, lines


def prediction_arrays(dataset: CandidateDataset, p) -> dict:
    """The exact npz member set of the shared prediction records."""
    return {
        "probabilities": p,
        "y": dataset.y,
        "offsets": dataset.offsets,
        "observation_trip_ids": dataset.observation_trip_ids,
        "observation_indices": dataset.observation_indices,
        "candidate_edges": dataset.candidate_edges,
        "reference_edges": dataset.reference_edges,
    }


def compare_metric_records(label: str, computed: dict, recorded: dict, tolerance: float = METRIC_TOLERANCE) -> list[str]:
    """Compare recomputed metrics with a committed record.

    Floats must agree within tolerance, other scalars exactly. Nested structures
    (per_trip, confusion_matrix) are skipped. Returns a list of mismatch lines;
    empty means agreement.
    """
    mismatches = []
    for key, value in recorded.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            continue
        if key not in computed:
            mismatches.append(f"{label}.{key}: missing from recomputation")
            continue
        if isinstance(value, float):
            if abs(float(computed[key]) - float(value)) > tolerance:
                mismatches.append(f"{label}.{key}: recomputed {computed[key]!r} vs recorded {value!r} "
                                  f"(delta {abs(float(computed[key]) - float(value)):.3e} > {tolerance:.0e})")
        elif computed[key] != value:
            mismatches.append(f"{label}.{key}: recomputed {computed[key]!r} vs recorded {value!r}")
    return mismatches


def require_missing(targets: list[Path], action: str) -> None:
    """Guard: this tool only creates files that do not exist yet."""
    existing = [target for target in targets if target.exists()]
    if existing:
        listing = ", ".join(str(target) for target in existing)
        raise PrepError(
            f"refusing to {action}: already present ({listing}). Shared root records are "
            "immutable history; this stage only fills gaps in a fresh clone. Delete "
            "deliberately and re-run if you own that decision."
        )


def read_phase_selection(root: Path) -> dict:
    path = paths(root)["selection"]
    if not path.exists():
        raise PrepError(
            "phase gate failed: experiments/phase_three/selection.json is missing. "
            "The phase select stage must record the TRAIN-only family selection before "
            "any student data is prepared."
        )
    selection = json.loads(path.read_text())
    selected = selection.get("selected_model")
    if selected not in PHASE_FAMILIES:
        raise PrepError(f"phase gate failed: selection.json selected_model {selected!r} is outside the phase families {PHASE_FAMILIES}")
    return selection


def read_valid_freeze(root: Path, selected_model: str) -> dict:
    path = paths(root)["freeze"]
    if not path.exists():
        raise PrepError(
            "phase gate failed: artifacts/phase_three/model_freeze.json is missing. "
            "The phase freeze stage must complete (and its HPO inputs exist) before "
            "any student data is prepared."
        )
    freeze = json.loads(path.read_text())
    frozen_model = paths(root)["frozen_model"]
    if freeze.get("model") != "artifacts/phase_three/model.joblib" or not frozen_model.exists():
        raise PrepError("phase gate failed: freeze record does not name an existing artifacts/phase_three/model.joblib")
    if freeze.get("model_name") not in PHASE_FAMILIES:
        raise PrepError(f"phase gate failed: frozen model_name {freeze.get('model_name')!r} is outside the phase families")
    if freeze.get("model_name") != selected_model:
        raise PrepError(
            f"phase gate failed: freeze model_name {freeze.get('model_name')!r} != selection "
            f"selected_model {selected_model!r}"
        )
    observed = sha256_file(frozen_model)
    if observed != freeze.get("sha256"):
        raise PrepError(
            f"phase gate failed: artifacts/phase_three/model.joblib sha256 {observed} != freeze "
            f"record {freeze.get('sha256')!r}"
        )
    return freeze


# ---------------------------------------------------------------------------
# Stage: check (read-only)
# ---------------------------------------------------------------------------

def check_stage(root: Path, data_root: Path, *, train_spec: dict = TRAIN_SPEC,
                student_spec: dict = STUDENT_SPEC, hash_student_raw: bool = False) -> int:
    """Read-only inventory and verification. Writes nothing, fits nothing."""
    root, data_root = Path(root), Path(data_root)
    p = paths(root)
    missing: list[str] = []
    failures: list[str] = []

    note("STAGE check: read-only verification; no cache is built and no model is fitted")
    report_versions()
    note(f"shared config for fingerprinting: {json.dumps(asdict(SHARED_CONFIG), sort_keys=True)}")
    note("fingerprint uses bayesian_trials=12 (saved shared run), not the phase HPO budget of 8")

    def reference_line(relative: str) -> None:
        path = root / relative
        if not path.exists():
            missing.append(relative)
            return
        observed = sha256_file(path)
        expected = REFERENCE_SHAS.get(relative)
        marker = "matches reference" if observed == expected else f"DIFFERS from reference {expected}"
        note(f"{relative}: present, sha256 {observed} ({marker})")

    # --- raw TRAIN data (bytes hashed for the fingerprint; labels never parsed)
    try:
        fingerprint = recompute_fingerprint(data_root, train_spec)
        if fingerprint == train_spec["fingerprint"]:
            note(f"raw train: {train_spec['trips']} complete trips, fingerprint {fingerprint[:16]} (pinned match)")
        else:
            failures.append(f"raw train fingerprint {fingerprint} != pinned {train_spec['fingerprint']}")
    except PrepError as exc:
        missing.append(f"raw train data under {data_root / train_spec['split']} ({exc})")

    # --- raw student data: presence only by default; content hashing is
    # post-freeze (--hash-student-raw), matching the no-student-access-before-
    # freeze discipline.
    student_dir = data_root / student_spec["split"]
    if student_dir.is_dir():
        try:
            prefixes = discover_trips(student_dir)
        except ValueError:
            prefixes = []
        if len(prefixes) != student_spec["trips"]:
            missing.append(f"raw student data: {len(prefixes)} complete trips, expected {student_spec['trips']}")
        else:
            note(f"raw student: {student_spec['trips']} complete trips (presence only; "
                 f"content {'hashed' if hash_student_raw else 'not hashed'} in this mode)")
            if hash_student_raw:
                try:
                    fingerprint = recompute_fingerprint(data_root, student_spec)
                    if fingerprint == student_spec["fingerprint"]:
                        note(f"raw student fingerprint {fingerprint[:16]} (pinned match)")
                    else:
                        failures.append(f"raw student fingerprint {fingerprint} != pinned {student_spec['fingerprint']}")
                except PrepError as exc:
                    failures.append(f"raw student data unusable: {exc}")
    else:
        missing.append(f"raw student data directory {student_dir}")

    # --- dataset caches
    train_cache = p["cache_dir"] / f"{train_spec['cache']}.npz"
    if train_cache.exists():
        ok, lines = verify_dataset_cache(train_cache, train_spec, deep=True)
        for line in lines:
            note(f"train cache: {line}")
        if not ok:
            failures.append("train cache failed pinned verification (see report above)")
    else:
        missing.append(f"experiments/cache/{train_spec['cache']}.npz (+ .json)")
    student_cache = p["cache_dir"] / f"{student_spec['cache']}.npz"
    if student_cache.exists():
        ok, lines = verify_dataset_cache(student_cache, student_spec, deep=hash_student_raw)
        for line in lines:
            note(f"student cache: {line}")
        if not ok:
            failures.append("student cache failed pinned verification (see report above)")
    else:
        missing.append(f"experiments/cache/{student_spec['cache']}.npz (+ .json)")

    # --- shared records
    if p["cv_folds"].exists():
        try:
            shared = json.loads(p["cv_folds"].read_text())
            entries = shared["folds"] if isinstance(shared, dict) and "folds" in shared else shared
            problems = validate_fold_entries(entries, train_spec)
            for line in problems:
                note(f"cv_folds: {line}")
            if problems:
                failures.append("experiments/cv_folds.json failed structure validation")
            else:
                note(f"cv_folds: {len(entries)} group folds, structure valid")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            failures.append(f"experiments/cv_folds.json unreadable: {exc}")
        reference_line("experiments/cv_folds.json")
    else:
        missing.append("experiments/cv_folds.json (train stage creates it when missing)")

    cv_valid = False
    if p["cv_lr"].exists():
        try:
            record = json.loads(p["cv_lr"].read_text())
            if record.get("parameters") != LR_PARAMETERS:
                failures.append(f"experiments/cv/logistic_regression.json parameters {record.get('parameters')!r} != phase defaults {LR_PARAMETERS!r}")
            else:
                cv_valid = True
                note("cv LR record: parameters match the phase defaults; "
                     f"objective {record.get('objective')!r}")
        except json.JSONDecodeError as exc:
            failures.append(f"experiments/cv/logistic_regression.json unreadable: {exc}")
        reference_line("experiments/cv/logistic_regression.json")
    else:
        missing.append("experiments/cv/logistic_regression.json (train stage creates it via LR-only CV when missing)")

    if p["oof_lr"].exists():
        reference_line("experiments/predictions/logistic_regression_oof.npz")
        if cv_valid and train_cache.exists() and sha256_file(train_cache) == train_spec["sha256"]:
            dataset = load_verified_dataset(root, train_spec)
            with np.load(p["oof_lr"], allow_pickle=False) as archive:
                members = set(archive.files)
                oof = archive["probabilities"] if "probabilities" in members else None
            expected_members = set(prediction_arrays(dataset, np.zeros(1)))
            if members != expected_members:
                failures.append(f"OOF npz members {sorted(members)} != {sorted(expected_members)}")
            elif oof is None or len(oof) != len(dataset.y):
                failures.append("OOF probabilities length does not match the train cache")
            else:
                mismatches = compare_metric_records("oof_metrics", _metrics(dataset, oof), json.loads(p["cv_lr"].read_text())["oof_metrics"])
                if mismatches:
                    failures.extend(mismatches)
                else:
                    note(f"OOF record: recomputed metrics agree with the committed record within {METRIC_TOLERANCE:.0e}")
    else:
        missing.append("experiments/predictions/logistic_regression_oof.npz (train stage rebuilds it via LR-only grouped CV)")

    if p["eval_lr"].exists():
        try:
            record = json.loads(p["eval_lr"].read_text())
            if record.get("parameters") != LR_PARAMETERS:
                failures.append(f"experiments/evaluation/logistic_regression.json parameters {record.get('parameters')!r} != {LR_PARAMETERS!r}")
            else:
                note("evaluation LR record: parameters match; "
                     f"student top1 {record['student_metrics']['top1_accuracy']!r}")
        except (json.JSONDecodeError, KeyError) as exc:
            failures.append(f"experiments/evaluation/logistic_regression.json unreadable: {exc}")
        reference_line("experiments/evaluation/logistic_regression.json")
    else:
        missing.append("experiments/evaluation/logistic_regression.json (committed historical record; the preparer never fabricates it)")

    if p["student_lr"].exists():
        reference_line("experiments/predictions/logistic_regression_student.npz")
    else:
        missing.append("experiments/predictions/logistic_regression_student.npz (student stage rebuilds it after the phase freeze)")

    reference_line("artifacts/baselines/logistic_regression.joblib")

    if p["results"].exists():
        try:
            results = json.loads(p["results"].read_text())
            comparison = results.get("comparison")
            if not isinstance(comparison, list) or not comparison or "target_policy" not in results:
                failures.append("experiments/results.json is missing its comparison rows or target_policy")
            else:
                note(f"results.json: committed historical record, {len(comparison)} comparison rows, "
                     f"{len(comparison[0])} columns; schema present (never created or modified by this tool)")
        except json.JSONDecodeError as exc:
            failures.append(f"experiments/results.json unreadable: {exc}")
    else:
        missing.append("experiments/results.json (committed historical record; this tool refuses to fabricate measured results)")

    # --- phase gates (informational)
    note(f"phase gate selection.json: {'present' if p['selection'].exists() else 'absent'}")
    note(f"phase gate model_freeze.json: {'present' if p['freeze'].exists() else 'absent'}")

    for line in missing:
        note(f"MISSING {line}")
    for line in failures:
        note(f"FAILURE {line}")
    if failures:
        note("check verdict: verification failures present; nothing was written and nothing was silently accepted")
        return EXIT_FAILURE
    if missing:
        note(f"check verdict: {len(missing)} required input(s) missing; run the matching stage in a fresh clone")
        return EXIT_MISSING
    note("check verdict: every phase-required shared input is present and verified; nothing to prepare")
    return EXIT_OK


def validate_fold_entries(entries, train_spec: dict) -> list[str]:
    problems = []
    if not isinstance(entries, list) or not entries:
        return ["fold entries are not a non-empty list"]
    seen_validation: set[str] = set()
    for index, entry in enumerate(entries):
        train_trips = set(entry.get("train_trip_ids") or [])
        validation_trips = set(entry.get("validation_trip_ids") or [])
        if not train_trips or not validation_trips:
            problems.append(f"fold {index}: empty trip side")
        if train_trips & validation_trips:
            problems.append(f"fold {index}: trip leakage between train and validation")
        if validation_trips & seen_validation:
            problems.append(f"fold {index}: validation trips reused across folds")
        seen_validation |= validation_trips
    if train_spec is not None and entries and not problems:
        total = set()
        for entry in entries:
            total |= set(entry.get("train_trip_ids") or [])
            total |= set(entry.get("validation_trip_ids") or [])
        if len(total) != train_spec["trips"]:
            problems.append(f"folds cover {len(total)} trips, expected {train_spec['trips']}")
    return problems


# ---------------------------------------------------------------------------
# Stage: train
# ---------------------------------------------------------------------------

def lr_grouped_oof(dataset: CandidateDataset, folds: list, parameters: dict, config: ExperimentConfig):
    """Grouped-CV mirror of cross_validate_model for logistic regression only.

    Fits no other family and writes no record. Used when the committed shared
    CV record exists and only the git-excluded OOF array file is missing.
    """
    started = time.perf_counter()
    oof = np.full(len(dataset.y), np.nan)
    fold_records = []
    for fold in folds:
        model = make_estimator("logistic_regression", parameters, config)
        train_rows, validation_rows = fold["train_rows"], fold["validation_rows"]
        t = time.perf_counter()
        fit_warnings = _fit(model, dataset.X[train_rows], dataset.y[train_rows], dataset.training_weights(train_rows))
        training_seconds = time.perf_counter() - t
        t = time.perf_counter()
        probabilities = predict_positive_proba(model, dataset.X[validation_rows])
        inference_seconds = time.perf_counter() - t
        oof[validation_rows] = probabilities
        fold_records.append({
            "fold": fold["fold"],
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "warnings": fit_warnings,
            "metrics": _metrics(dataset, probabilities, fold["validation_observations"]),
        })
        note(f"LR CV fold {fold['fold']}: top1 {fold_records[-1]['metrics']['top1_accuracy']:.6f}, "
             f"macro_f1 {fold_records[-1]['metrics']['macro_f1']:.6f}, fit {training_seconds:.1f}s")
    if not np.isfinite(oof).all():
        raise AssertionError("missing out-of-fold probabilities")
    aggregate = _metrics(dataset, oof)
    return oof, aggregate, fold_records, time.perf_counter() - started


def train_stage(root: Path, data_root: Path, *, train_spec: dict = TRAIN_SPEC,
                student_spec: dict = STUDENT_SPEC) -> int:
    root, data_root = Path(root), Path(data_root)
    p = paths(root)
    cache_npz = p["cache_dir"] / f"{train_spec['cache']}.npz"
    cache_json = p["cache_dir"] / f"{train_spec['cache']}.json"

    note("STAGE train: rebuild the TRAIN cache, shared folds and LR grouped-CV OOF record")
    report_versions()

    # Guard: only fill gaps; never overwrite shared history.
    require_missing([cache_npz, cache_json, p["oof_lr"]], "rebuild TRAIN inputs")

    # Gate: raw data must reproduce the pinned cache fingerprint.
    fingerprint = recompute_fingerprint(data_root, train_spec)
    if fingerprint != train_spec["fingerprint"]:
        raise PrepError(
            f"raw TRAIN data fingerprint {fingerprint} != pinned {train_spec['fingerprint']}. "
            "The raw trip files do not match the data that produced the frozen shared cache; "
            "this is reported, not worked around."
        )
    note(f"raw TRAIN fingerprint {fingerprint[:16]} (pinned match; bytes hashed, labels never parsed)")

    # Build the cache with the shared builder. No model is involved.
    note("building TRAIN dataset cache via mapmatch.experiments.build_dataset (features only, no models)")
    dataset = build_dataset(data_root, train_spec["split"], SHARED_CONFIG, root / "experiments")
    if not cache_npz.exists():
        raise PrepError(f"build_dataset did not produce the expected cache file {cache_npz.name}")
    ok, lines = verify_dataset_cache(cache_npz, train_spec, deep=True)
    for line in lines:
        note(f"rebuilt train cache: {line}")
    if not ok:
        raise PrepError(
            "rebuilt TRAIN cache failed pinned verification. Nothing was silently accepted. "
            "If every per-array digest matches and only the container sha256 differs, the array "
            "content is faithful and only the npz zip layout changed (original run used numpy "
            f"{ORIGINAL_RUN_VERSIONS['numpy']}; see docs/PHASE_REPRODUCTION.md). The phase runner "
            "still hard-asserts the pinned sha256, so decide deliberately: rebuild under the "
            "original runtime, or have the maintainer re-pin. The rebuilt files were left in place "
            "for diagnosis; delete them by hand before any retry."
        )
    note("TRAIN cache rebuilt and verified against the pinned spec")

    # Shared group folds.
    folds = build_group_folds(dataset, SHARED_CONFIG)
    if p["cv_folds"].exists():
        shared = json.loads(p["cv_folds"].read_text())
        entries = shared["folds"] if isinstance(shared, dict) and "folds" in shared else shared
        problems = validate_fold_entries(entries, train_spec)
        if problems:
            raise PrepError("existing experiments/cv_folds.json failed validation: " + "; ".join(problems))
        for mine, ref in zip(folds, entries):
            if set(mine["train_trip_ids"]) != set(ref["train_trip_ids"]) or set(mine["validation_trip_ids"]) != set(ref["validation_trip_ids"]):
                raise PrepError(
                    f"recomputed fold {mine['fold']} trip sets differ from the committed "
                    "experiments/cv_folds.json; refusing to proceed with a divergent fold structure"
                )
        note("group folds recomputed and equal to the committed experiments/cv_folds.json")
    else:
        entries = [
            {
                "fold": fold["fold"],
                "train_trip_ids": fold["train_trip_ids"],
                "validation_trip_ids": fold["validation_trip_ids"],
                "train_observations": len(fold["train_observations"]),
                "validation_observations": len(fold["validation_observations"]),
                "train_candidate_rows": len(fold["train_rows"]),
                "validation_candidate_rows": len(fold["validation_rows"]),
            }
            for fold in folds
        ]
        save_json(p["cv_folds"], entries)
        note("wrote experiments/cv_folds.json (was missing; same schema as the shared run)")

    # LR grouped-CV record: reuse the committed record when present, create it
    # through the shared LR-only path when absent. No other family is fitted.
    if p["cv_lr"].exists():
        record = json.loads(p["cv_lr"].read_text())
        if record.get("parameters") != LR_PARAMETERS:
            raise PrepError(
                f"committed LR CV record parameters {record.get('parameters')!r} != phase defaults "
                f"{LR_PARAMETERS!r}; refusing to mirror a divergent record"
            )
        note("committed LR CV record found: rebuilding only the git-excluded OOF arrays via the LR-only mirror")
        oof, aggregate, fold_records, elapsed = lr_grouped_oof(dataset, folds, LR_PARAMETERS, SHARED_CONFIG)
        mismatches = compare_metric_records("oof_metrics", aggregate, record["oof_metrics"])
        if mismatches:
            raise PrepError(
                "rebuilt LR OOF metrics disagree with the committed record:\n  "
                + "\n  ".join(mismatches)
                + "\nNothing was written. A regenerated OOF file that contradicts the "
                "committed record would be silently accepted history; that is not done here."
            )
        p["oof_lr"].parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p["oof_lr"], **prediction_arrays(dataset, oof))
        note(f"wrote experiments/predictions/logistic_regression_oof.npz; metrics agree with the "
             f"committed record within {METRIC_TOLERANCE:.0e} (LR-only mirror, {elapsed:.1f}s)")
        # Round-trip: reload what was written and re-verify.
        with np.load(p["oof_lr"], allow_pickle=False) as archive:
            reloaded = archive["probabilities"]
        mismatches = compare_metric_records("oof_metrics", _metrics(dataset, reloaded), record["oof_metrics"])
        if mismatches:
            raise PrepError("reloaded OOF file disagrees with the committed record: " + "; ".join(mismatches))
    else:
        note("committed LR CV record missing: running logistic-regression-only grouped CV via the "
             "shared cross_validate_model (writes cv and predictions records; no other family)")
        from mapmatch.experiments import cross_validate_model

        result = cross_validate_model("logistic_regression", LR_PARAMETERS, dataset, folds,
                                      SHARED_CONFIG, root / "experiments", "logistic_regression")
        note(f"LR CV record created: objective {result['objective']!r}")

    note("STAGE train complete: TRAIN cache, folds and LR OOF record are in place")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Stage: student (gated on selection + freeze)
# ---------------------------------------------------------------------------

def student_stage(root: Path, data_root: Path, *, allow_lr_fit: bool = False,
                  train_spec: dict = TRAIN_SPEC, student_spec: dict = STUDENT_SPEC) -> int:
    root, data_root = Path(root), Path(data_root)
    p = paths(root)

    note("STAGE student: rebuild the student-test cache and LR student predictions")
    report_versions()

    # Gates FIRST: no student data is touched before selection and freeze are valid.
    selection = read_phase_selection(root)
    selected = selection["selected_model"]
    freeze = read_valid_freeze(root, selected)
    note(f"phase gates passed: selection.json selects {selected!r}; freeze record valid "
         f"(model.joblib sha256 {freeze['sha256'][:16]})")

    # Guards: only fill gaps.
    cache_npz = p["cache_dir"] / f"{student_spec['cache']}.npz"
    cache_json = p["cache_dir"] / f"{student_spec['cache']}.json"
    require_missing([cache_npz, cache_json, p["student_lr"]], "rebuild student inputs")

    # Fingerprint gate on the raw student data.
    fingerprint = recompute_fingerprint(data_root, student_spec)
    if fingerprint != student_spec["fingerprint"]:
        raise PrepError(
            f"raw student data fingerprint {fingerprint} != pinned {student_spec['fingerprint']}; "
            "reported, not worked around"
        )
    note(f"raw student fingerprint {fingerprint[:16]} (pinned match)")

    # LR student predictions. The committed all-80 baseline is loaded read-only
    # so no fit happens by default.
    if p["eval_lr"].exists():
        evaluation = json.loads(p["eval_lr"].read_text())
        if evaluation.get("parameters") != LR_PARAMETERS:
            raise PrepError(
                f"committed LR evaluation record parameters {evaluation.get('parameters')!r} != "
                f"phase defaults {LR_PARAMETERS!r}"
            )
    else:
        raise PrepError(
            "experiments/evaluation/logistic_regression.json is missing. It is a committed "
            "historical measured record that the phase runner reuses wholesale; this tool does "
            "not fabricate measured records. Restore it from the published repository."
        )

    note("building student-test dataset cache via mapmatch.experiments.build_dataset (features only)")
    student = build_dataset(data_root, student_spec["split"], SHARED_CONFIG, root / "experiments")
    if not cache_npz.exists():
        raise PrepError(f"build_dataset did not produce the expected cache file {cache_npz.name}")
    ok, lines = verify_dataset_cache(cache_npz, student_spec, deep=True)
    for line in lines:
        note(f"rebuilt student cache: {line}")
    if not ok:
        raise PrepError(
            "rebuilt student cache failed pinned verification (same honesty policy as the TRAIN "
            "cache; see docs/PHASE_REPRODUCTION.md). The rebuilt files were left in place for "
            "diagnosis; delete them by hand before any retry."
        )

    if p["baseline"].exists():
        bundle = load_model(p["baseline"])  # validates feature schema and binary classes
        if bundle.get("metadata", {}).get("parameters") != LR_PARAMETERS:
            raise PrepError(f"baseline joblib parameters {bundle.get('metadata', {}).get('parameters')!r} != {LR_PARAMETERS!r}")
        note("LR student predictions: read-only load of the committed all-80 baseline "
             "artifacts/baselines/logistic_regression.joblib (no fit)")
        estimator = bundle["estimator"]
        fit_note = "committed baseline load"
    elif allow_lr_fit:
        note("--allow-lr-fit: performing a FRESH logistic-regression-only all-80 fit. This is a "
             "deliberate regeneration, recorded as such; it is not the historical baseline.")
        train = load_verified_dataset(root, train_spec)
        estimator = make_estimator("logistic_regression", LR_PARAMETERS, SHARED_CONFIG)
        t = time.perf_counter()
        fit_warnings = _fit(estimator, train.X, train.y, train.training_weights(np.arange(len(train.y))))
        fit_seconds = time.perf_counter() - t
        note(f"fresh LR all-80 fit completed in {fit_seconds:.1f}s (warnings: {len(fit_warnings)})")
        bundle = {
            "estimator": estimator,
            "feature_names": list(FEATURE_NAMES),
            "feature_version": FEATURE_VERSION,
            "metadata": {
                "model_name": "logistic_regression",
                "run_name": "logistic_regression",
                "seed": SHARED_CONFIG.seed,
                "parameters": LR_PARAMETERS,
                "training_trip_ids": np.unique(train.observation_trip_ids).tolist(),
                "regenerated_by": "scripts/prepare_phase_three_inputs.py --stage student --allow-lr-fit",
                "regenerated_versions": runtime_versions(),
            },
        }
        p["baseline"].parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, p["baseline"], compress=3)
        note("wrote artifacts/baselines/logistic_regression.joblib (was missing; fresh LR-only fit)")
        fit_note = f"fresh LR-only fit ({fit_seconds:.1f}s)"
    else:
        raise PrepError(
            "artifacts/baselines/logistic_regression.joblib is missing. The default path loads it "
            "read-only and fits nothing. Restore it from the published repository, or pass "
            "--allow-lr-fit for an explicitly recorded fresh LR-only fit."
        )

    t = time.perf_counter()
    probabilities = predict_positive_proba(estimator, student.X)
    inference_seconds = time.perf_counter() - t
    student_metrics = _metrics(student, probabilities)
    mismatches = compare_metric_records("student_metrics", student_metrics, evaluation["student_metrics"])
    if mismatches:
        raise PrepError(
            "recomputed LR student metrics disagree with the committed evaluation "
            "record:\n  " + "\n  ".join(mismatches)
            + "\nNothing was written; a predictions file contradicting the committed "
            "record would be silently accepted history."
        )
    p["student_lr"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p["student_lr"], **prediction_arrays(student, probabilities))
    note(f"wrote experiments/predictions/logistic_regression_student.npz; recomputed metrics agree "
         f"with the committed record within {METRIC_TOLERANCE:.0e} "
         f"({fit_note}; inference {inference_seconds:.1f}s, top1 {student_metrics['top1_accuracy']!r})")

    note("STAGE student complete: student cache and LR student predictions are in place")
    return EXIT_OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare git-excluded phase-three inputs (LR/KNN-distance/bagging scope) "
                    "without running the out-of-scope shared pipeline.",
    )
    parser.add_argument("--stage", required=True, choices=("check", "train", "student"))
    parser.add_argument("--data-root", default="Student folder",
                        help="directory containing train/ and student_test/ trip folders")
    parser.add_argument("--root", default=str(ROOT), help="repository root (default: this script's repo)")
    parser.add_argument("--hash-student-raw", action="store_true",
                        help="check stage: also hash raw student bytes for its fingerprint (post-freeze use)")
    parser.add_argument("--allow-lr-fit", action="store_true",
                        help="student stage: allow an explicitly recorded fresh LR-only all-80 fit "
                             "when the committed baseline joblib is missing")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root

    try:
        if args.stage == "check":
            return check_stage(root, data_root, hash_student_raw=args.hash_student_raw)
        if args.stage == "train":
            return train_stage(root, data_root)
        return student_stage(root, data_root, allow_lr_fit=args.allow_lr_fit)
    except PrepError as exc:
        print(f"[phase-prep] FAILED (nothing silently accepted): {exc}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
