#!/usr/bin/env python3
"""Validate, or explicitly restore missing private inputs for the final five families.

The default check is read-only. TRAIN restores only missing KNN/Bagging OOF
archives; student restores only missing KNN/Bagging student archives from the
published estimators after the final model and decoder freezes have passed.
Historical JSON, weights, models, and existing archives are never overwritten.
No raw-data builder, hyperparameter search, evaluator, or service is invoked.
"""
from __future__ import annotations

import sys
# The environment variable alone only affects future Python processes.
sys.dont_write_bytecode = True

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.final_five import run_final_five as final
from experiments.phase_three import run_phase_three as phase
from mapmatch.experiments import _metrics
from mapmatch.features import FEATURE_NAMES, FEATURE_VERSION
from mapmatch.learning import load_model
from mapmatch.phase_models import KNN_FIXED_PARAMETERS, KNN_REFERENCE_BANK_BUDGET, bank_seed
from scripts import prepare_phase_three_inputs as rawprep

FAMILIES = tuple(final.FINAL_FAMILIES[:3])
RESTORABLE = ("knn_distance_weighted", "bagging")
CACHE_SPECS = {"train": rawprep.TRAIN_SPEC, "student_test": rawprep.STUDENT_SPEC}
BASE_PATHS = {name: str(path.relative_to(final.ROOT)) for name, path in final.BASE_ARTIFACTS.items()}
CHECK_REPORT = "reports/final_five/reproduction_inputs_check.json"
REPLAY_PARENT = "experiments/final_five_replay_inputs"
METRIC_ATOL = 1e-12
PUBLISHED_RUNTIME = {"python": "3.14.6", "numpy": "2.5.5", "scipy": "1.18.0", "sklearn": "1.8.0", "joblib": "1.5.3"}
RUNTIME_NOTE = ("Matching package versions do not guarantee matching BLAS builds or CPU reduction order. "
                "All metric comparisons use atol=1e-12, rtol=0; a difference fails restoration.")
SOURCE_FILES = ("scripts/prepare_final_five_inputs.py", "scripts/prepare_phase_three_inputs.py",
                "experiments/phase_three/run_phase_three.py", "mapmatch/phase_models.py",
                "experiments/final_five/run_final_five.py", "mapmatch/experiments.py", "mapmatch/learning.py")


class PrepError(RuntimeError):
    pass


class MissingInput(PrepError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return rawprep.sha256_file(path)


def safe_path(relative):
    """Accept repository-relative fixed paths, rejecting symlinks at every level."""
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise PrepError("destination must be a fixed repository-relative path")
    root = ROOT.resolve()
    path = root / relative
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise PrepError("symlink path refused: " + str(relative))
        if current != path and current.exists() and not current.is_dir():
            raise PrepError("destination parent is not a directory: " + str(relative))
    if path.resolve() != path:
        raise PrepError("path escapes the fixed repository root: " + str(relative))
    return path


@contextmanager
def directory_fd(relative, *, create=False):
    """Walk from the repository descriptor with O_NOFOLLOW, including on writes."""
    safe_path(relative)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(ROOT.resolve(), flags)
    try:
        for part in Path(relative).parts:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def require_file(relative, instruction="restore the committed input from the public clone"):
    path = safe_path(relative)
    if not path.is_file():
        raise MissingInput(str(relative) + " is missing; " + instruction)
    return path


def read_json(relative):
    return json.loads(require_file(relative).read_text())


def exclusive_json(relative, value):
    """O_EXCL closes the exists-check race; an existing file is never truncated."""
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    path = safe_path(relative)
    with directory_fd(Path(relative).parent, create=True) as parent:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())


def prediction_path(name, split):
    if name not in FAMILIES or split not in ("train", "student_test"):
        raise PrepError("unknown fixed prediction destination")
    parent = "experiments/predictions" if name == "logistic_regression" else "experiments/phase_three/predictions"
    return parent + "/" + name + ("_oof.npz" if split == "train" else "_student.npz")


def replay_path(run_id):
    if not isinstance(run_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", run_id) is None:
        raise PrepError("run-id must be 1-48 lowercase letters, digits, underscores or hyphens")
    relative = REPLAY_PARENT + "/" + run_id
    safe_path(relative)
    return relative


def allocate_replay(relative, stage, missing):
    # mkdir is exclusive, so concurrent invocations cannot share a diagnostic run.
    with directory_fd(Path(relative).parent, create=True) as parent:
        os.mkdir(Path(relative).name, mode=0o700, dir_fd=parent)
    exclusive_json(relative + "/started.json", {
        "started_at_utc": utc_now(), "stage": stage, "status": "STARTED", "missing": missing,
        "purpose": "New restoration costs and diagnostics; historical timings are unchanged.",
        "versions": final.versions(), "runtime_note": RUNTIME_NOTE,
        "thread_caps": {key: os.environ[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "source_sha256": {name: sha256(safe_path(name)) for name in SOURCE_FILES},
        "evaluator_accessed": False,
    })


def original_records(split):
    """Bind original records to their immutable final-run reuse provenance."""
    records, pins = {}, {}
    for name in FAMILIES:
        kind = "cv" if split == "train" else "evaluation"
        source = "experiments/phase_three/" + kind + "/" + name + ".json"
        published = "experiments/final_five/current/" + kind + "/" + name + ".json"
        record, pin = read_json(source), read_json(published)
        if record.get("model_name") != name or record.get("parameters") != final.FINAL_DEFAULTS[name]:
            raise PrepError("original family/parameters mismatch: " + source)
        provenance = pin["reuse_provenance"]
        file_key, hash_key = ("cv_file", "cv_sha256") if split == "train" else ("evaluation_file", "evaluation_sha256")
        if provenance[file_key] != source or sha256(safe_path(source)) != provenance[hash_key]:
            raise PrepError("original record differs from published reuse hash: " + source)
        if pin.get("parameters") != record["parameters"] or pin.get("model_name") != name:
            raise PrepError("published reuse family/parameters mismatch: " + published)
        expected = prediction_path(name, split)
        pointer = record.get("oof_predictions_file") if split == "train" else record.get("student_predictions_file", record.get("provenance", {}).get("student_predictions_file"))
        if pointer != expected:
            raise PrepError("original record has an unexpected prediction path: " + source)
        if split == "train":
            strict_metrics(record["oof_metrics"], pin["oof_metrics"])
        else:
            strict_metrics(record["student_metrics"], pin["student_metrics"])
            if pin.get("model_path") != BASE_PATHS[name]:
                raise PrepError("published baseline artifact path mismatch: " + published)
        records[name], pins[name] = record, pin
    return records, pins


def strict_metrics(actual, expected):
    """Use the final-runner check plus every saved metric, including per-trip data."""
    final.compare_metrics(actual, expected, tolerance=METRIC_ATOL)

    def compare(got, ref, key):
        if isinstance(ref, dict):
            if not isinstance(got, dict) or set(got) != set(ref):
                raise PrepError("metric keys mismatch: " + key)
            for child in ref:
                compare(got[child], ref[child], key + "." + child)
        elif isinstance(ref, list):
            if not isinstance(got, list) or len(got) != len(ref):
                raise PrepError("metric list mismatch: " + key)
            for index, (a, b) in enumerate(zip(got, ref)):
                compare(a, b, key + "." + str(index))
        elif isinstance(ref, (int, float)) and not isinstance(ref, bool):
            if (not isinstance(got, (int, float)) or isinstance(got, bool)
                    or not np.isfinite(got) or not np.isfinite(ref)
                    or not np.isclose(got, ref, atol=METRIC_ATOL, rtol=0)):
                raise PrepError("metric value mismatch (atol=1e-12, rtol=0): " + key)
        elif got != ref:
            raise PrepError("metric value mismatch: " + key)
    compare(actual, expected, "metrics")


def validate_original_folds(records, folds):
    for name, record in records.items():
        final.check_folds(record, folds)
        if len(folds) != 3:
            raise PrepError("exactly three original folds are required")
        seen = set()
        for index, (got, ref) in enumerate(zip(record["folds"], folds)):
            if got["fold"] != index or ref["fold"] != index:
                raise PrepError("original CV fold index mismatch: " + name)
            for key in ("train_trip_ids", "validation_trip_ids"):
                if len(got[key]) != len(set(got[key])) or set(got[key]) != set(ref[key]):
                    raise PrepError("original CV trip set mismatch: " + name)
            if seen & set(ref["validation_trip_ids"]):
                raise PrepError("validation trip reused across folds")
            seen.update(ref["validation_trip_ids"])
            for key in ("train_rows", "validation_rows"):
                expected = len(ref[key]) if not isinstance(ref[key], int) else ref[key]
                if got[key] != expected:
                    raise PrepError("original CV row count mismatch: " + name)
            if name == "knn_distance_weighted":
                bank = got["knn_reference_bank"]
                source_rows = got["train_rows"]
                if (bank["seed"] != bank_seed(phase.phase_config().seed, index)
                        or bank["budget"] != KNN_REFERENCE_BANK_BUDGET
                        or bank["selected_rows"] != min(KNN_REFERENCE_BANK_BUDGET, source_rows)
                        or bank["source_rows"] != source_rows
                        or sum(bank["class_counts"].values()) != bank["selected_rows"]):
                    raise PrepError("original KNN fold-bank policy mismatch")


def shared_folds_before_data(records):
    saved = read_json("experiments/cv_folds.json")
    entries = saved["folds"] if isinstance(saved, dict) else saved
    problems = rawprep.validate_fold_entries(entries, CACHE_SPECS["train"])
    if problems:
        raise PrepError("shared folds failed validation: " + "; ".join(problems))
    folds = [{**entry, "train_rows": entry["train_candidate_rows"],
              "validation_rows": entry["validation_candidate_rows"]} for entry in entries]
    validate_original_folds(records, folds)


def verify_cache(split):
    spec = CACHE_SPECS[split]
    path = "experiments/cache/" + spec["cache"] + ".npz"
    require_file(path, "run the existing LR/raw-cache preparation stage documented in docs/FINAL_REPRODUCTION.md")
    require_file(str(Path(path).with_suffix(".json")))
    ok, notes = rawprep.verify_dataset_cache(safe_path(path), spec, deep=True)
    if not ok:
        raise PrepError("pinned cache verification failed: " + "; ".join(notes))
    return {"kind": "dataset_cache", "path": path, "sha256": sha256(safe_path(path)),
            "array_sha256_verified": True, "trips": spec["trips"], "observations": spec["observations"],
            "candidate_rows": spec["rows"], "candidate_misses": spec["misses"]}


def validate_freezes():
    """No student cache, reference array, or estimator is read until these pass."""
    base = "artifacts/final_five/"
    freeze = read_json(base + "model_freeze.json")
    if (freeze.get("phase") != "five-model-completion" or freeze.get("model_name") != "hist_gradient_boosting"
            or freeze.get("run_name") != "hist_gradient_boosting_tuned" or freeze.get("model") != base + "model.joblib"
            or freeze.get("student_used_for_selection") is not False or freeze.get("evaluator_used_for_selection") is not False):
        raise PrepError("current final model freeze is not the TRAIN-selected HistGB freeze")
    for filename, key in (("model.joblib", "model_sha256"), ("model_metadata.json", "metadata_sha256"),
                          ("feature_schema.json", "feature_schema_sha256")):
        if sha256(require_file(base + filename)) != freeze[key]:
            raise PrepError("current final model freeze hash mismatch: " + filename)
    schema = read_json(base + "feature_schema.json")
    if schema.get("feature_names") != list(FEATURE_NAMES) or schema.get("feature_version") != FEATURE_VERSION or schema.get("count") != len(FEATURE_NAMES):
        raise PrepError("current final feature schema mismatch")
    decoder = read_json("reports/final_five/decoder/freeze.json")
    config_path = base + "match_config.json"
    config = read_json(config_path)
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (decoder.get("phase") != "five-model-completion" or decoder.get("model_path") != freeze["model"]
            or decoder.get("model_sha256") != freeze["model_sha256"]
            or any(decoder.get(key) is not False for key in ("student_results_seen_for_selection", "student_results_used_for_selection", "evaluator_accessed"))
            or decoder.get("config_file_sha256") != sha256(safe_path(config_path))
            or decoder.get("config") != config or decoder.get("effective_config_sha256") != config_digest):
        raise PrepError("current final decoder freeze/config/model hash mismatch")
    return {"kind": "final_freezes", "model_sha256": freeze["model_sha256"],
            "model_freeze_sha256": sha256(safe_path(base + "model_freeze.json")),
            "decoder_freeze_sha256": sha256(safe_path("reports/final_five/decoder/freeze.json")),
            "config_sha256": decoder["config_file_sha256"]}


def validate_probabilities(dataset, prediction):
    phase.assert_prediction_identity(dataset, prediction)
    p = np.asarray(prediction["probabilities"])
    if p.dtype.kind != "f" or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise PrepError("probabilities must be finite floating-point values in [0,1]")
    return p


def verify_prediction(name, split, dataset, record, folds=None, path=None):
    prediction = phase.load_prediction_npz(require_file(path or prediction_path(name, split)))
    p = validate_probabilities(dataset, prediction)
    expected = record["oof_metrics" if split == "train" else "student_metrics"]
    strict_metrics(_metrics(dataset, p), expected)
    if folds is not None:
        for fold, original in zip(folds, record["folds"]):
            strict_metrics(_metrics(dataset, p[fold["validation_rows"]], fold["validation_observations"]), original["metrics"])
    return {"kind": "prediction_archive", "path": path or prediction_path(name, split),
            "sha256": sha256(safe_path(path or prediction_path(name, split))), "identity_verified": True,
            "all_saved_metrics_verified": True, "fold_metrics_verified": len(folds or []),
            "candidate_rows": len(p), "observations": len(dataset.offsets) - 1}


def verify_baseline(name, pin, train):
    path = require_file(BASE_PATHS[name])
    if sha256(path) != pin["model_sha256"]:
        raise PrepError("published baseline artifact SHA mismatch: " + name)
    bundle = load_model(path)  # trusted published joblib only; validates schema/classes
    md, estimator = bundle["metadata"], bundle["estimator"]
    trips = md.get("training_trip_ids", [])
    if (md.get("model_name") != name or md.get("run_name") != name or md.get("seed") != phase.phase_config().seed
            or md.get("parameters") != final.FINAL_DEFAULTS[name]
            or len(trips) != CACHE_SPECS["train"]["trips"] or len(set(trips)) != len(trips)
            or set(trips) != set(np.unique(train.observation_trip_ids))
            or estimator.n_features_in_ != len(FEATURE_NAMES)):
        raise PrepError("baseline metadata must match exactly all TRAIN trips, parameters and schema: " + name)
    classifier = estimator.named_steps.get("classifier") if isinstance(estimator, Pipeline) else estimator
    if name in ("logistic_regression", "knn_distance_weighted"):
        if not isinstance(estimator, Pipeline) or list(estimator.named_steps) != ["scale", "classifier"] or not isinstance(estimator.named_steps["scale"], StandardScaler):
            raise PrepError("baseline scaler pipeline mismatch: " + name)
    if name == "logistic_regression":
        if not isinstance(classifier, LogisticRegression):
            raise PrepError("baseline must contain an actual LogisticRegression")
        expected = {**final.FINAL_DEFAULTS[name], "random_state": phase.phase_config().seed}
    elif name == "knn_distance_weighted":
        if not isinstance(classifier, KNeighborsClassifier):
            raise PrepError("baseline must contain an actual KNeighborsClassifier")
        expected = {**final.FINAL_DEFAULTS[name], **KNN_FIXED_PARAMETERS}
        bank_rows = min(KNN_REFERENCE_BANK_BUDGET, len(train.y))
        if len(classifier._fit_X) != bank_rows or not np.all(estimator.named_steps["scale"].n_samples_seen_ == bank_rows):
            raise PrepError("historical KNN scaler and classifier must be fitted on the reference bank")
    else:
        if not isinstance(classifier, BaggingClassifier):
            raise PrepError("baseline must contain an actual BaggingClassifier")
        params = final.FINAL_DEFAULTS[name]
        expected = {"n_estimators": params["n_estimators"], "max_samples": params["max_samples"],
                    "bootstrap": True, "random_state": phase.phase_config().seed,
                    "estimator__max_depth": params["max_depth"], "estimator__min_samples_leaf": params["min_samples_leaf"]}
        if len(classifier.estimators_) != params["n_estimators"]:
            raise PrepError("baseline bagging fitted tree count mismatch")
    if any(classifier.get_params()[key] != value for key, value in expected.items()):
        raise PrepError("actual estimator parameters differ from the published defaults: " + name)
    return bundle, {"kind": "published_baseline", "path": BASE_PATHS[name], "sha256": pin["model_sha256"],
                    "training_trips": len(trips), "features": len(FEATURE_NAMES), "schema_classes_parameters_verified": True}


def replay_oof(name, record, dataset, folds, config, work, state):
    """Mirror the original phase CV loop; scaler/bank semantics stay in phase_fit."""
    oof = np.full(len(dataset.y), np.nan)
    seen = np.zeros(len(dataset.y), dtype=np.int8)
    new_folds = []
    for fold, original in zip(folds, record["folds"]):
        state["fits_performed"] += 1
        estimator, fit_seconds, warnings, bank = phase.phase_fit(name, record["parameters"], config, dataset, fold["train_rows"], fold["fold"])
        state["prediction_calls"] += 1
        probabilities, inference_seconds = phase.phase_predict(estimator, dataset.X[fold["validation_rows"]])
        if np.asarray(probabilities).shape != (len(fold["validation_rows"]),) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
            raise PrepError("invalid recomputed fold probabilities: " + name)
        oof[fold["validation_rows"]] = probabilities
        seen[fold["validation_rows"]] += 1
        metrics = _metrics(dataset, probabilities, fold["validation_observations"])
        measured = {"fold": fold["fold"], "train_rows": len(fold["train_rows"]), "validation_rows": len(fold["validation_rows"]),
                    "training_seconds": fit_seconds, "inference_seconds": inference_seconds,
                    "warnings": warnings, "knn_reference_bank": bank, "metrics": metrics,
                    "measurement_role": "fresh input-restoration measurement; not the historical timing"}
        exclusive_json(work + "/" + name + "_fold_" + str(fold["fold"]) + ".json", measured)
        strict_metrics(metrics, original["metrics"])
        if name == "knn_distance_weighted" and bank != original["knn_reference_bank"]:
            raise PrepError("recomputed KNN reference-bank policy differs from original fold")
        new_folds.append(measured)
    if not np.isfinite(oof).all() or not np.all(seen == 1):
        raise PrepError("OOF rows must be covered exactly once without NaNs")
    strict_metrics(_metrics(dataset, oof), record["oof_metrics"])
    return oof, new_folds


def publish_prediction(name, split, dataset, probabilities, record, folds, work):
    """Validate a new private archive, then hard-link it without replacing anything."""
    destination = prediction_path(name, split)
    temporary = work + "/" + name + ("_oof.candidate.npz" if split == "train" else "_student.candidate.npz")
    candidate = safe_path(temporary)
    with directory_fd(Path(temporary).parent) as parent:
        fd = os.open(candidate.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **phase.prediction_arrays(dataset, probabilities))
            stream.flush()
            os.fsync(stream.fileno())
    verify_prediction(name, split, dataset, record, folds, path=temporary)
    with directory_fd(Path(temporary).parent) as source, directory_fd(Path(destination).parent, create=True) as target:
        os.link(candidate.name, Path(destination).name, src_dir_fd=source, dst_dir_fd=target, follow_symlinks=False)
    return verify_prediction(name, split, dataset, record, folds)


def preflight_predictions(split):
    missing = []
    for name in FAMILIES:
        relative = prediction_path(name, split)
        path = safe_path(relative)
        if path.exists() and not path.is_file():
            raise PrepError("prediction destination is not a regular file: " + relative)
        if not path.exists():
            if name == "logistic_regression":
                raise MissingInput(relative + "; use scripts/prepare_phase_three_inputs.py --stage "
                                   + ("train" if split == "train" else "student") + " first; this tool never duplicates LR fits")
            missing.append(name)
    return missing


def initial_report(stage):
    return {"stage": stage, "checked_at_utc": utc_now(), "status": "PASSED", "fits_performed": 0,
            "prediction_calls": 0, "dataset_builds": 0, "evaluator_accessed": False,
            "runtime": final.versions(), "published_runtime": PUBLISHED_RUNTIME, "runtime_note": RUNTIME_NOTE,
            "metric_atol": METRIC_ATOL, "metric_rtol": 0, "checks": [], "missing": [], "failures": []}


def check_stage(report):
    # Inventory existence even when a clean clone has no datasets to validate.
    inventory = ["experiments/cv_folds.json"]
    for split, spec in CACHE_SPECS.items():
        inventory.extend(["experiments/cache/" + spec["cache"] + extension for extension in (".npz", ".json")])
        inventory.extend(prediction_path(name, split) for name in FAMILIES)
    report["inventory"] = [{"path": relative, "present": safe_path(relative).is_file()} for relative in inventory]
    for item in report["inventory"]:
        if not item["present"]:
            instruction = ""
            if "logistic_regression" in item["path"]:
                instruction = "; prepare LR first with scripts/prepare_phase_three_inputs.py (see docs/FINAL_REPRODUCTION.md)"
            report["missing"].append(item["path"] + instruction)
    # Check TRAIN first; student content is never read before the final freezes.
    train_records = None
    train = None
    try:
        train_records, _ = original_records("train")
        shared_folds_before_data(train_records)
        report["checks"].append(verify_cache("train"))
        train, folds = phase.load_train_and_folds(phase.phase_config())
        validate_original_folds(train_records, folds)
        report["checks"].append({"kind": "group_folds", "path": "experiments/cv_folds.json",
                                  "sha256": sha256(safe_path("experiments/cv_folds.json")), "folds": len(folds)})
        for name in FAMILIES:
            try:
                report["checks"].append(verify_prediction(name, "train", train, train_records[name], folds))
            except MissingInput as exc:
                report["missing"].append(str(exc))
    except MissingInput as exc:
        report["missing"].append(str(exc))
    except (PrepError, AssertionError, KeyError, ValueError, OSError) as exc:
        report["failures"].append(str(exc))
    try:
        report["checks"].append(validate_freezes())
        student_records, pins = original_records("student_test")
        report["checks"].append(verify_cache("student_test"))
        student = phase.load_cached_dataset("student_test")
        for name in FAMILIES:
            try:
                report["checks"].append(verify_prediction(name, "student_test", student, student_records[name]))
                if train is not None:
                    _, baseline = verify_baseline(name, pins[name], train)
                    report["checks"].append(baseline)
            except MissingInput as exc:
                report["missing"].append(str(exc))
    except MissingInput as exc:
        report["missing"].append(str(exc))
    except (PrepError, AssertionError, KeyError, ValueError, OSError) as exc:
        report["failures"].append(str(exc))


def prepare_stage(stage, run_id, report):
    split = "train" if stage == "train" else "student_test"
    if stage == "student":
        report["checks"].append(validate_freezes())
    missing = preflight_predictions(split)
    work = replay_path(run_id) if run_id is not None else None
    if missing:
        if work is None:
            raise PrepError("missing KNN/Bagging inputs require an explicit new --run-id for diagnostic timings")
        if safe_path(work).exists():
            raise PrepError("refuse to reuse an existing input-restoration run-id")
        if final.versions() != PUBLISHED_RUNTIME:
            raise PrepError("restoration requires the pinned published runtime; use the original .venv")
    train_records, _ = original_records("train")
    shared_folds_before_data(train_records)
    if stage == "student":
        records, pins = original_records(split)
        for name in FAMILIES:
            require_file(BASE_PATHS[name])
    else:
        records = train_records
    report["checks"].append(verify_cache("train"))
    config = phase.phase_config()
    train, folds = phase.load_train_and_folds(config)
    validate_original_folds(train_records, folds)
    bundles = {}
    if stage == "student":
        for name in FAMILIES:
            bundles[name], check = verify_baseline(name, pins[name], train)
            report["checks"].append(check)
        report["checks"].append(verify_cache(split))
        dataset = phase.load_cached_dataset(split)
    else:
        dataset = train
    # All existing inputs are checked before the first fit or prediction.
    for name in FAMILIES:
        if name not in missing:
            report["checks"].append(verify_prediction(name, split, dataset, records[name], folds if stage == "train" else None))
    if not missing:
        report["action"] = "all requested archives exist; validated without fitting, inference, or writes"
        return
    for name in missing:
        if safe_path(prediction_path(name, split)).exists():
            raise PrepError("input appeared during preflight; refusing duplicate fitting or inference")
    allocate_replay(work, stage, missing)
    started = time.perf_counter()
    try:
        pending = []
        for name in missing:
            if stage == "train":
                probabilities, _ = replay_oof(name, records[name], dataset, folds, config, work, report)
            else:
                report["prediction_calls"] += 1
                probabilities, seconds = phase.phase_predict(bundles[name]["estimator"], dataset.X)
                prediction = phase.prediction_arrays(dataset, probabilities)
                validate_probabilities(dataset, prediction)
                metrics = _metrics(dataset, probabilities)
                exclusive_json(work + "/" + name + "_inference.json", {"inference_seconds": seconds,
                    "metrics": metrics, "artifact": BASE_PATHS[name], "artifact_sha256": pins[name]["model_sha256"],
                    "measurement_role": "fresh full-student inference; no model fitted", "fits_performed": 0})
                strict_metrics(metrics, records[name]["student_metrics"])
            pending.append((name, probabilities))
        # Do not publish either newly computed archive until both families pass.
        for name, probabilities in pending:
            report["checks"].append(publish_prediction(name, split, dataset, probabilities, records[name],
                                                       folds if stage == "train" else None, work))
        exclusive_json(work + "/complete.json", {"status": "PASSED", "completed_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started, "fits_performed": report["fits_performed"],
            "prediction_calls": report["prediction_calls"], "restored": [prediction_path(n, split) for n in missing],
            "historical_records_modified": False, "evaluator_accessed": False})
        report["diagnostics"] = work
    except Exception as exc:
        exclusive_json(work + "/failed.json", {"status": "FAILED", "failed_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started, "fits_performed": report["fits_performed"],
            "prediction_calls": report["prediction_calls"], "error": str(exc), "evaluator_accessed": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "train", "student"), default="check")
    parser.add_argument("--run-id", help="new private diagnostic directory; required only when filling missing KNN/Bagging inputs")
    parser.add_argument("--report", help="check only: explicitly create reports/final_five/reproduction_inputs_check.json; never overwrite")
    args = parser.parse_args(argv)
    report = initial_report(args.stage)
    report_path = None
    try:
        if args.report:
            if args.stage != "check" or args.report != CHECK_REPORT:
                raise PrepError("--report is restricted to --stage check --report " + CHECK_REPORT)
            report_path = safe_path(args.report)
            if report_path.exists():
                raise PrepError("refuse to overwrite an existing check report")
        if args.stage == "check" and args.run_id is not None:
            raise PrepError("check does not accept --run-id and never creates a diagnostic directory")
        with threadpool_limits(limits=1):
            if args.stage == "check":
                check_stage(report)
            else:
                prepare_stage(args.stage, args.run_id, report)
    except MissingInput as exc:
        report["missing"].append(str(exc))
    except (PrepError, AssertionError, KeyError, ValueError, OSError) as exc:
        report["failures"].append(str(exc))
    report["status"] = "FAILED" if report["failures"] else "MISSING" if report["missing"] else "PASSED"
    # The public report contains only verification summaries, counts and hashes.
    # Per-trip metrics and new timings remain inside the excluded diagnostic run.
    if report_path is not None and not report_path.exists():
        exclusive_json(args.report, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 1 if report["status"] == "FAILED" else 2 if report["status"] == "MISSING" else 0


if __name__ == "__main__":
    raise SystemExit(main())
