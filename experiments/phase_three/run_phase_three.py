"""Phase-three staged experiment runner (user-selected three model families).

Stages, strictly ordered and gated on recorded outputs so no student-test
array can be read before the family selection is on disk and no student
diagnostic can run before the winner is frozen:

    cv      grouped CV of knn_distance_weighted + bagging; logistic_regression
            CV is reused read-only from the valid shared record
    select  record the TRAIN-only family selection (selection.json)
    hpo     8-trial Bayesian optimization of the selected family
            (1 measured prior + 2 seeded + 5 GP-EI)
    freeze  final all80 fit of the winner, atomic artifact export, clean-env
            smoke load, model_freeze.json, PHASE_MODEL_FROZEN signal
    student all-10 student diagnostics for all three families + tuned winner,
            results.json / model_comparison.csv (shared schema)
    all     cv -> select -> hpo -> freeze -> student

Writes only under experiments/phase_three/ and artifacts/phase_three/ (plus
this file itself). Never mutates shared experiments/ root files, the deployed
artifacts/model.joblib, or any mapmatch source module; shared primitives are
imported read-only. See experiments/phase_three/PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import subprocess
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.observation_metrics import observation_metrics
from mapmatch.data import discover_trips, load_graph
from mapmatch.experiments import (CandidateDataset, ExperimentConfig, _metrics, _sha256, append_json,
    build_group_folds, decode_parameters, encode_parameters, expected_improvement, log, save_json)
from mapmatch.features import FEATURE_DESCRIPTIONS, FEATURE_NAMES, FEATURE_VERSION, PROXY_LABEL_POLICY
from mapmatch.learning import predict_positive_proba
from mapmatch.phase_models import (KNN_FINAL_FIT_INDEX, KNN_REFERENCE_BANK_BUDGET, KNN_FIXED_PARAMETERS,
    PHASE_DEFAULT_PARAMETERS, PHASE_FAMILIES, PHASE_MODEL_COUNT, PHASE_NAME, PHASE_SEARCH_SPACES,
    bank_seed, knn_reference_bank, make_phase_estimator, phase_preprocessing_description)

SHARED = ROOT / "experiments"
PHASE_DIR = SHARED / "phase_three"
ARTIFACTS = ROOT / "artifacts" / "phase_three"
DATA_ROOT = ROOT / "Student folder"
TRAIN_CACHE = "train_edee9a8f151dd25e"
STUDENT_CACHE = "student_test_240a8fbfc79205b3"
TRAIN_CACHE_SHA256 = "db523c112a6119f16b6f473d6221e3c56b6730ccb2db64dee55433aedbf3ba9e"
STUDENT_CACHE_SHA256 = "ae96abb4784501b2b086c9b4aa2f5668e1c9d7f29abda907b7f402118af81b73"
EXPECTED = {
    "train": {"observations": 28000, "rows": 269463, "trips": 80, "misses": 578,
              "cache": TRAIN_CACHE, "sha256": TRAIN_CACHE_SHA256},
    "student_test": {"observations": 19160, "rows": 175778, "trips": 10, "misses": 19,
                     "cache": STUDENT_CACHE, "sha256": STUDENT_CACHE_SHA256},
}
FAMILIES = ["logistic_regression", "knn_distance_weighted", "bagging"]
NEW_FAMILIES = ["knn_distance_weighted", "bagging"]
HPO_TRIALS = 8
HPO_SEEDED = 2
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "NUMEXPR_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def versions() -> dict:
    import platform
    import scipy
    import sklearn
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "joblib": joblib.__version__,
            "platform": platform.platform(), "machine": platform.machine()}


def hardware() -> dict:
    import platform
    return {"platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "thread_caps": THREAD_ENV}


def phase_config() -> ExperimentConfig:
    # Identical to the shared saved run configuration except the phase-three
    # HPO budget of 8 trials. n_jobs follows the saved config (2).
    return ExperimentConfig(seed=20261118, train_observations_per_trip=350,
                            student_observations_per_trip=0, folds=3,
                            bayesian_trials=HPO_TRIALS, n_jobs=2)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def require(path: Path, message: str) -> dict:
    if not path.exists():
        raise SystemExit(f"MISSING {path.relative_to(ROOT)}: {message}")
    return load_json(path)


def refuse_existing_stage(stage: str) -> None:
    """Fail before loading data/fitting or changing any completed/partial outputs."""
    downstream = [PHASE_DIR / "results.json", ARTIFACTS / "model_freeze.json"]
    if stage == "cv":
        outputs = [PHASE_DIR / "cv" / f"{name}.json" for name in FAMILIES]
        outputs += [PHASE_DIR / "cv" / "stage_summary.json", PHASE_DIR / "selection.json",
                    PHASE_DIR / "bayesian_trials.jsonl", *downstream]
        outputs += list((PHASE_DIR / "predictions").glob("*_oof.npz"))
    elif stage == "select":
        outputs = [PHASE_DIR / "selection.json", PHASE_DIR / "bayesian_trials.jsonl", *downstream]
    elif stage == "hpo":
        outputs = [PHASE_DIR / "bayesian_trials.jsonl", PHASE_DIR / "bayesian_optimization.json", *downstream]
        outputs += list((PHASE_DIR / "cv").glob("bayes_*.json"))
    elif stage == "freeze":
        outputs = [ARTIFACTS / name for name in ("model.joblib", "model.joblib.tmp", "model_metadata.json",
                   "feature_schema.json", "clean_load_report.json", "artifact_manifest.json", "model_freeze.json")]
        outputs += list((PHASE_DIR / "cv").glob("*_tuned.json"))
        outputs += list((PHASE_DIR / "predictions").glob("*_tuned_oof.npz"))
    elif stage == "student":
        outputs = [PHASE_DIR / name for name in ("results.json", "sequence_classification.json", "model_comparison.csv")]
        outputs += list((PHASE_DIR / "evaluation").glob("*.json"))
        outputs += list((PHASE_DIR / "predictions").glob("*_student.npz"))
        outputs += list((ARTIFACTS / "baselines").glob("*.joblib"))
        outputs += list((PHASE_DIR / "sparse_confusion").glob("*.csv"))
    else:
        raise ValueError("unknown phase stage")
    existing = sorted({str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
                       for path in outputs if path.exists() or path.is_symlink()})
    if existing:
        raise FileExistsError(f"refuse to overwrite/refit phase stage {stage}; existing outputs: " + ", ".join(existing[:8]))


def assert_prediction_identity(dataset, prediction: dict) -> None:
    for key in ("y", "offsets", "observation_trip_ids", "observation_indices", "candidate_edges", "reference_edges"):
        if key not in prediction or not np.array_equal(prediction[key], getattr(dataset, key)):
            raise AssertionError("saved prediction cache identity mismatch: " + key)
    if np.asarray(prediction["probabilities"]).shape != dataset.y.shape:
        raise AssertionError("saved prediction cache probability shape mismatch")


def load_cached_dataset(split: str) -> CandidateDataset:
    """Read-only load of the already-frozen shared cache; never rebuilds it."""
    spec = EXPECTED[split]
    cache = SHARED / "cache" / f"{spec['cache']}.npz"
    if not cache.exists():
        raise SystemExit(f"shared cache {cache.name} not found; phase three must reuse it")
    if _sha256(cache) != spec["sha256"]:
        raise AssertionError(f"shared cache {cache.name} sha256 mismatch")
    coverage = load_json(cache.with_suffix(".json"))
    if coverage.get("split") != split:
        raise AssertionError(f"coverage split {coverage.get('split')!r} != {split!r}")
    if cache.name != f"{split}_{coverage['fingerprint'][:16]}.npz":
        raise AssertionError("cache filename does not match recorded fingerprint")
    fields = [f for f in CandidateDataset.__dataclass_fields__ if f != "coverage"]
    with np.load(cache, allow_pickle=False) as z:
        arrays = {k: z[k] for k in fields}
    dataset = CandidateDataset(**arrays, coverage=coverage)
    if len(dataset.offsets) - 1 != spec["observations"] or len(dataset.y) != spec["rows"]:
        raise AssertionError(f"{split} cache shape mismatch")
    if len(np.unique(dataset.observation_trip_ids)) != spec["trips"]:
        raise AssertionError(f"{split} cache trip count mismatch")
    positive = np.concatenate([[0], np.cumsum(dataset.y == 1)])
    segment_positives = positive[dataset.offsets[1:]] - positive[dataset.offsets[:-1]]
    if int((segment_positives == 0).sum()) != spec["misses"]:
        raise AssertionError(f"{split} cache candidate-miss count mismatch")
    if dataset.X.shape[1] != len(FEATURE_NAMES) or str(dataset.X.dtype) != "float32":
        raise AssertionError(f"{split} cache feature schema mismatch")
    if dataset.offsets[-1] != len(dataset.X) or not np.array_equal(np.unique(dataset.y), [0, 1]):
        raise AssertionError("invalid cached dataset")
    return dataset


def load_train_and_folds(config: ExperimentConfig):
    """Cached TRAIN dataset plus group folds asserted equal to the shared cv_folds.json."""
    train = load_cached_dataset("train")
    folds = build_group_folds(train, config)
    shared = load_json(SHARED / "cv_folds.json")
    entries = shared["folds"] if isinstance(shared, dict) and "folds" in shared else shared
    if len(entries) != len(folds):
        raise AssertionError("fold count mismatch vs shared cv_folds.json")
    for mine, ref in zip(folds, entries):
        if set(mine["train_trip_ids"]) != set(ref["train_trip_ids"]) or set(mine["validation_trip_ids"]) != set(ref["validation_trip_ids"]):
            raise AssertionError(f"fold {mine['fold']} trip split differs from shared cv_folds.json")
    return train, folds


def phase_fit(name, parameters, config, dataset, rows, fit_index):
    """Fit one phase family on the given TRAIN rows (KNN: on its reference bank)."""
    estimator = make_phase_estimator(name, parameters, config.seed, config.n_jobs)
    rows = np.asarray(rows, np.int64)
    X, y = dataset.X[rows], dataset.y[rows]
    weights = dataset.training_weights(rows)
    bank_info = None
    fit_rows = rows
    kwargs = {}
    if name == "knn_distance_weighted":
        seed = bank_seed(config.seed, fit_index)
        bank = knn_reference_bank(y, weights, KNN_REFERENCE_BANK_BUDGET, seed)
        classes, counts = np.unique(y[bank], return_counts=True)
        bank_info = {
            "policy": "class-stratified weighted sampling without replacement over THIS fit's TRAIN rows only; inclusion probability proportional to saved training weights; deterministic seed",
            "budget": KNN_REFERENCE_BANK_BUDGET, "selected_rows": int(len(bank)),
            "source_rows": int(len(rows)), "seed": int(seed),
            "class_counts": {str(int(c)): int(n) for c, n in zip(classes, counts)},
            "sample_weight_note": "KNeighborsClassifier.fit accepts no sample_weight; the bank policy is the disclosed substitute",
        }
        fit_rows = rows[bank]
    elif isinstance(estimator, Pipeline):
        kwargs = {"classifier__sample_weight": weights}
    else:
        kwargs = {"sample_weight": weights}
    t = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(dataset.X[fit_rows], dataset.y[fit_rows], **kwargs)
    fit_seconds = time.perf_counter() - t
    fit_warnings = sorted({str(w.message) for w in caught})
    return estimator, fit_seconds, fit_warnings, bank_info


def phase_predict(estimator, X):
    t = time.perf_counter()
    p = predict_positive_proba(estimator, X)
    return p, time.perf_counter() - t


def prediction_arrays(dataset, p) -> dict:
    return {"probabilities": np.asarray(p, dtype=np.float64), "y": dataset.y,
            "offsets": dataset.offsets, "observation_trip_ids": dataset.observation_trip_ids,
            "observation_indices": dataset.observation_indices, "candidate_edges": dataset.candidate_edges,
            "reference_edges": dataset.reference_edges}


def save_prediction_npz(path: Path, dataset, p) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **prediction_arrays(dataset, p))
    return str(path.relative_to(ROOT))


def load_prediction_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def cross_validate_phase(name, parameters, run_name, dataset, folds, config, save_predictions=True):
    """Grouped CV mirroring mapmatch.experiments.cross_validate_model for phase families."""
    outputs = [PHASE_DIR / "cv" / f"{run_name}.json", PHASE_DIR / "predictions" / f"{run_name}_oof.npz"]
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FileExistsError("refuse to refit an existing measured CV run: " + run_name)
    t0 = time.perf_counter()
    oof = np.full(len(dataset.y), np.nan)
    fold_results = []
    for fold in folds:
        train, val = fold["train_rows"], fold["validation_rows"]
        model, fit_seconds, fit_warnings, bank_info = phase_fit(name, parameters, config, dataset, train, fold["fold"])
        p, inference_seconds = phase_predict(model, dataset.X[val])
        oof[val] = p
        metrics = _metrics(dataset, p, fold["validation_observations"])
        record = {"fold": fold["fold"], "train_trip_ids": fold["train_trip_ids"],
                  "validation_trip_ids": fold["validation_trip_ids"], "train_rows": len(train),
                  "validation_rows": len(val), "training_seconds": fit_seconds,
                  "inference_seconds": inference_seconds, "warnings": fit_warnings,
                  "knn_reference_bank": bank_info, "metrics": metrics}
        fold_results.append(record)
        log("phase_cv_fold", phase=PHASE_NAME, run_name=run_name, fold=fold["fold"],
            training_seconds=fit_seconds, top1=metrics["top1_accuracy"],
            macro_f1=metrics["macro_f1"], log_loss=metrics["log_loss"])
    if not np.isfinite(oof).all():
        raise AssertionError("missing out-of-fold probabilities")
    aggregate = _metrics(dataset, oof)
    objective = aggregate["trip_macro_top1_accuracy"]
    result = {"name": run_name, "model_name": name, "family": PHASE_FAMILIES[name],
              "phase": PHASE_NAME, "parameters": parameters, "oof_metrics": aggregate,
              "folds": fold_results, "objective": objective,
              "objective_name": "out_of_fold_trip_macro_top1_accuracy",
              "macro_f1_fold_mean": float(np.mean([f["metrics"]["macro_f1"] for f in fold_results])),
              "macro_f1_fold_std": float(np.std([f["metrics"]["macro_f1"] for f in fold_results])),
              "top1_fold_mean": float(np.mean([f["metrics"]["top1_accuracy"] for f in fold_results])),
              "top1_fold_std": float(np.std([f["metrics"]["top1_accuracy"] for f in fold_results])),
              "training_seconds_total": sum(f["training_seconds"] for f in fold_results),
              "inference_seconds_total": sum(f["inference_seconds"] for f in fold_results),
              "elapsed_seconds": time.perf_counter() - t0}
    if save_predictions:
        result["oof_predictions_file"] = save_prediction_npz(
            PHASE_DIR / "predictions" / f"{run_name}_oof.npz", dataset, oof)
    save_json(PHASE_DIR / "cv" / f"{run_name}.json", result)
    log("phase_cv_complete", phase=PHASE_NAME, run_name=run_name, objective=objective,
        macro_f1=aggregate["macro_f1"], log_loss=aggregate["log_loss"],
        elapsed_seconds=result["elapsed_seconds"])
    return result


# ---------------------------------------------------------------- stages ----

def stage_cv(config):
    refuse_existing_stage("cv")
    train, folds = load_train_and_folds(config)
    # Logistic regression: reuse the valid shared CV record without refitting.
    shared_cv = SHARED / "cv" / "logistic_regression.json"
    shared_oof = SHARED / "predictions" / "logistic_regression_oof.npz"
    lr = load_json(shared_cv)
    if lr["parameters"] != PHASE_DEFAULT_PARAMETERS["logistic_regression"]:
        raise AssertionError("shared LR parameters differ from the phase declared defaults")
    stored = load_prediction_npz(shared_oof)
    assert_prediction_identity(train, stored)
    recomputed = _metrics(train, stored["probabilities"])
    for key in ("macro_f1", "log_loss", "top1_accuracy", "trip_macro_top1_accuracy"):
        if abs(recomputed[key] - lr["oof_metrics"][key]) > 1e-9:
            raise AssertionError(f"LR reuse validation failed on {key}")
    lr_phase = dict(lr)
    lr_phase.update({
        "phase": PHASE_NAME, "reused": True,
        "oof_predictions_file": str(shared_oof.relative_to(ROOT)),
        "reuse_provenance": {
            "cv_file": str(shared_cv.relative_to(ROOT)), "cv_sha256": _sha256(shared_cv),
            "oof_file": str(shared_oof.relative_to(ROOT)), "oof_sha256": _sha256(shared_oof),
            "validation": "parameters equal to phase defaults; fold trip sets equal to shared cv_folds.json; OOF metrics recomputed from the saved array within 1e-9; dataset cache sha256 verified",
        }})
    save_json(PHASE_DIR / "cv" / "logistic_regression.json", lr_phase)
    log("phase_cv_reuse", phase=PHASE_NAME, model="logistic_regression",
        objective=lr_phase["objective"], reused=True)
    results = {"logistic_regression": lr_phase}
    for name in NEW_FAMILIES:
        results[name] = cross_validate_phase(name, PHASE_DEFAULT_PARAMETERS[name], name,
                                             train, folds, config)
    summary = {"stage": "cv", "phase": PHASE_NAME, "student_test_consulted": False,
               "completed_at_utc": utc_now(),
               "models": {n: {"objective": r["objective"],
                              "macro_f1": r["oof_metrics"]["macro_f1"],
                              "log_loss": r["oof_metrics"]["log_loss"],
                              "reused": bool(r.get("reused"))} for n, r in results.items()}}
    save_json(PHASE_DIR / "cv" / "stage_summary.json", summary)
    print(json.dumps({"signal": "PHASE_CV_READY", **summary["models"]}, allow_nan=False))
    return results


def stage_select(config):
    refuse_existing_stage("select")
    results = {n: require(PHASE_DIR / "cv" / f"{n}.json", "run --stage cv first")
               for n in FAMILIES}

    def rank(r):
        return (r["objective"], r["oof_metrics"]["macro_f1"], -r["oof_metrics"]["log_loss"])

    chosen = max(FAMILIES, key=lambda n: rank(results[n]))
    selection = {
        "phase": PHASE_NAME, "selected_model": chosen,
        "objective_name": "out_of_fold_trip_macro_top1_accuracy",
        "selection_data": "TRAIN grouped CV only; the student development set was not loaded",
        "student_test_consulted": False,
        "families_considered": FAMILIES,
        "scores": {n: {"objective": r["objective"], "macro_f1": r["oof_metrics"]["macro_f1"],
                       "log_loss": r["oof_metrics"]["log_loss"],
                       "parameters": r["parameters"],
                       "cv_file": f"cv/{n}.json", "reused": bool(r.get("reused"))}
                   for n, r in results.items()},
        "tie_break": "objective, then binary macro-F1, then lower log loss",
        "selected_at_utc": utc_now(),
        "protocol": "experiments/phase_three/PROTOCOL.md",
    }
    save_json(PHASE_DIR / "selection.json", selection)
    log("phase_selection", phase=PHASE_NAME, selected_model=chosen,
        objective=results[chosen]["objective"])
    print(json.dumps({"signal": "PHASE_SELECTION_RECORDED", "selected_model": chosen,
                      "objective": results[chosen]["objective"]}, allow_nan=False))
    return selection


def stage_hpo(config):
    refuse_existing_stage("hpo")
    selection = require(PHASE_DIR / "selection.json", "run --stage select first")
    name = selection["selected_model"]
    baseline = require(PHASE_DIR / "cv" / f"{name}.json", "baseline CV record missing")
    train, folds = load_train_and_folds(config)
    start = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    space, base = PHASE_SEARCH_SPACES[name], dict(baseline["parameters"])
    vectors = [encode_parameters(base, space)]
    results = [baseline]
    records = [{"trial": 0, "proposal": "measured_baseline_prior", "parameters": base,
                "objective": baseline["objective"], "cv_macro_f1": baseline["oof_metrics"]["macro_f1"],
                "cv_log_loss": baseline["oof_metrics"]["log_loss"],
                "cv_file": f"cv/{baseline['name']}.json",
                "cv_seconds": baseline["elapsed_seconds"], "reused_actual_baseline": True}]
    trial_path = PHASE_DIR / "bayesian_trials.jsonl"
    with trial_path.open("x", encoding="utf-8"):
        pass  # Exclusive creation: never truncate actual measured HPO records.
    append_json(trial_path, records[0])
    seen = {json.dumps(base, sort_keys=True)}
    for trial in range(1, HPO_TRIALS):
        proposal_start = time.perf_counter()
        acquisition = {}
        if trial <= HPO_SEEDED:
            vector = rng.uniform(size=len(space))
            proposal = "seeded_initial_design"
        else:
            # Genuine model-based Bayesian optimization: GP surrogate conditioned
            # on measured CV objectives, analytic expected-improvement selection.
            dim = len(space)
            kernel = ConstantKernel(1.0, (.05, 20.0)) * Matern(length_scale=np.full(dim, .35),
                length_scale_bounds=(.03, 3.0), nu=2.5) + WhiteKernel(noise_level=.001,
                noise_level_bounds=(1e-6, .1))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                random_state=config.seed + trial, n_restarts_optimizer=1, alpha=1e-8)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                gp.fit(np.asarray(vectors), np.asarray([r["objective"] for r in results]))
            pool = np.vstack([rng.uniform(size=(2048, dim)),
                              np.clip(vectors[int(np.argmax([r["objective"] for r in results]))]
                                      + rng.normal(scale=.12, size=(1024, dim)), 0.0, 1.0)])
            mean, std = gp.predict(pool, return_std=True)
            ei = expected_improvement(mean, std, max(r["objective"] for r in results))
            chosen_pos = None
            for pos in np.argsort(ei)[::-1]:
                if json.dumps(decode_parameters(pool[pos], base, space), sort_keys=True) not in seen:
                    chosen_pos = int(pos)
                    break
            if chosen_pos is None:
                raise RuntimeError("Bayesian search exhausted unique parameter proposals")
            vector = pool[chosen_pos]
            proposal = "gaussian_process_expected_improvement"
            acquisition = {"predicted_objective_mean": float(mean[chosen_pos]),
                "predicted_objective_std": float(std[chosen_pos]),
                "expected_improvement": float(ei[chosen_pos]), "exploration_xi": .0003,
                "surrogate_kernel": str(gp.kernel_), "surrogate_training_points": len(results),
                "surrogate_warnings": [str(w.message) for w in caught],
                "proposal_pool_size": len(pool)}
        parameters = decode_parameters(vector, base, space)
        while json.dumps(parameters, sort_keys=True) in seen:
            proposal = "seeded_collision_fallback"
            acquisition = {"reason": "decoded parameter collision; actual seeded-uniform replacement"}
            vector = rng.uniform(size=len(space))
            parameters = decode_parameters(vector, base, space)
        seen.add(json.dumps(parameters, sort_keys=True))
        proposal_seconds = time.perf_counter() - proposal_start
        run_name = f"bayes_{name}_{trial:02d}"
        result = cross_validate_phase(name, parameters, run_name, train, folds, config,
                                      save_predictions=False)
        vectors.append(encode_parameters(parameters, space))
        results.append(result)
        record = {"trial": trial, "proposal": proposal, "parameters": parameters,
                  "objective": result["objective"], "cv_macro_f1": result["oof_metrics"]["macro_f1"],
                  "cv_log_loss": result["oof_metrics"]["log_loss"],
                  "cv_file": f"cv/{run_name}.json", "cv_seconds": result["elapsed_seconds"],
                  "proposal_seconds": proposal_seconds, **acquisition}
        records.append(record)
        append_json(trial_path, record)
        log("phase_bayesian_trial", phase=PHASE_NAME, model_name=name, **record)
    winner = max(results, key=lambda r: (r["objective"], r["oof_metrics"]["macro_f1"],
                                         -r["oof_metrics"]["log_loss"]))
    summary = {"method": "GaussianProcessRegressor Matern-5/2 surrogate + analytic expected improvement",
        "phase": PHASE_NAME, "model_name": name, "search_space": space,
        "objective": "maximize out-of-fold equal-trip mean observation top1 proxy accuracy; fixed 3-fold GroupKFold on all 80 TRAIN trips",
        "selection_data": "TRAIN only; student_test metrics not loaded during HPO",
        "trials_including_measured_prior": len(records),
        "seeded_initial_design_trials": HPO_SEEDED,
        "model_based_trials": sum(r["proposal"] == "gaussian_process_expected_improvement" for r in records),
        "best_cv_run_name": winner["name"], "best_parameters": winner["parameters"],
        "best_objective": winner["objective"], "baseline_objective": baseline["objective"],
        "improvement": winner["objective"] - baseline["objective"],
        "seconds": time.perf_counter() - start, "trials": records}
    save_json(PHASE_DIR / "bayesian_optimization.json", summary)
    print(json.dumps({"signal": "PHASE_HPO_READY", "model_name": name,
                      "best_parameters": winner["parameters"],
                      "best_objective": winner["objective"],
                      "improvement": summary["improvement"]}, allow_nan=False))
    return summary


def clean_environment_smoke_load(*, output_path: Path | None = None) -> dict:
    """Compare the actual clean child's versions and probabilities to this runtime.

    Historical freeze records are immutable. An explicit new output path permits
    an additional no-fit audit without rewriting the freeze's original evidence.
    """
    output_path = output_path or ARTIFACTS / "clean_load_report.json"
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("refuse to overwrite clean-environment evidence")
    from hashlib import sha256
    from mapmatch.learning import load_emission_model
    model_path = ARTIFACTS / "model.joblib"
    train = load_cached_dataset("train")
    X = np.ascontiguousarray(train.X[:2000])
    expected = predict_positive_proba(load_emission_model(model_path).estimator, X)
    fixture_hash = sha256(X.tobytes()).hexdigest()
    script = f"""
import json, sys, hashlib
from pathlib import Path
root = Path({str(ROOT)!r})
sys.path.insert(0, str(root))
import numpy as np
from experiments.phase_three.run_phase_three import versions
from mapmatch.features import FEATURE_NAMES, FEATURE_VERSION
from mapmatch.learning import load_emission_model, predict_positive_proba
model = load_emission_model(Path({str(model_path)!r}))
assert tuple(model.feature_names) == tuple(FEATURE_NAMES)
assert model.metadata.get("feature_version") == FEATURE_VERSION
with np.load(Path({str(SHARED / 'cache' / (TRAIN_CACHE + '.npz'))!r}), allow_pickle=False) as z:
    X = np.ascontiguousarray(z["X"][:2000])
p = predict_positive_proba(model.estimator, X)
print(json.dumps({{"versions": versions(), "fixture_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
                  "probabilities": p.tolist(), "classes": [int(c) for c in model.classes_]}}))
"""
    env = dict(os.environ)
    env.update(THREAD_ENV)
    proc = subprocess.run([str(ROOT / ".venv-clean" / "bin" / "python"), "-c", script],
                          capture_output=True, text=True, cwd=str(ROOT), env=env, timeout=3600)
    child = json.loads(proc.stdout) if proc.returncode == 0 else {}
    observed = np.asarray(child.get("probabilities", []), dtype=float)
    parent_versions = versions()
    version_match = all(child.get("versions", {}).get(key) == parent_versions[key]
                        for key in ("python", "numpy", "scipy", "sklearn", "joblib"))
    fixture_match = child.get("fixture_sha256") == fixture_hash
    valid = observed.shape == expected.shape and bool(np.isfinite(observed).all())
    delta = float(np.max(np.abs(expected - observed))) if valid else None
    report = {"python": ".venv-clean/bin/python", "versions": child.get("versions", {}),
              "parent_versions": parent_versions, "versions_match": version_match,
              "returncode": proc.returncode, "stderr": proc.stderr.strip()[-4000:],
              "fixture": "first 2000 TRAIN candidate rows; no labels used for inference",
              "fixture_sha256": fixture_hash, "fixture_identity_match": fixture_match,
              "model_sha256": _sha256(model_path), "predictions": len(expected),
              "max_abs_probability_delta": delta, "probability_tolerance_absolute": 1e-12,
              "passed": proc.returncode == 0 and version_match and fixture_match and valid
                        and delta is not None and delta <= 1e-12}
    save_json(output_path, report)
    if not report["passed"]:
        raise AssertionError("clean-environment version/fixture/probability comparison failed")
    return report


def stage_freeze(config):
    refuse_existing_stage("freeze")
    selection = require(PHASE_DIR / "selection.json", "run --stage select first")
    bayes = require(PHASE_DIR / "bayesian_optimization.json", "run --stage hpo first")
    name = selection["selected_model"]
    parameters = bayes["best_parameters"]
    train, folds = load_train_and_folds(config)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    baseline = load_json(PHASE_DIR / "cv" / f"{name}.json")
    same_as_baseline = json.dumps(parameters, sort_keys=True) == json.dumps(baseline["parameters"], sort_keys=True)
    # 1. Retain the winner's OOF arrays (reuse when the prior won; else one
    #    confirmatory replay on the identical folds, never an extra HPO trial).
    if same_as_baseline:
        winner_cv = dict(baseline)
        winner_cv["name"] = f"{name}_tuned"
        winner_cv["parameters"] = parameters
        winner_cv["final_oof_export"] = {"replayed": False, "reused_baseline_oof": True,
            "reason": "winning HPO configuration is exactly the measured prior",
            "oof_predictions_file": baseline.get("oof_predictions_file")}
        save_json(PHASE_DIR / "cv" / f"{name}_tuned.json", winner_cv)
        replay = dict(winner_cv["final_oof_export"])
    else:
        winner_cv = cross_validate_phase(name, parameters, f"{name}_tuned", train, folds, config)
        trial_record = next(r for r in bayes["trials"]
                            if json.dumps(r["parameters"], sort_keys=True) == json.dumps(parameters, sort_keys=True))
        agreement = {"objective": {"hpo_trial": trial_record["objective"], "replay": winner_cv["objective"]},
                     "macro_f1": {"hpo_trial": trial_record["cv_macro_f1"], "replay": winner_cv["oof_metrics"]["macro_f1"]},
                     "log_loss": {"hpo_trial": trial_record["cv_log_loss"], "replay": winner_cv["oof_metrics"]["log_loss"]}}
        for key, a in agreement.items():
            values = [a["hpo_trial"], a["replay"]]
            if abs(values[0] - values[1]) > 1e-6:
                raise AssertionError(f"winner replay disagrees with HPO trial on {key}: {a}")
        replay = {"replayed": True, "winning_trial": trial_record["trial"], "agreement": agreement,
                  "oof_predictions_file": winner_cv.get("oof_predictions_file")}
    # 2. Final fit on the EXACT cached all80 TRAIN matrix.
    all_rows = np.arange(len(train.y))
    estimator, fit_seconds, fit_warnings, bank_info = phase_fit(name, parameters, config,
                                                                train, all_rows, KNN_FINAL_FIT_INDEX)
    p_train, train_inference_seconds = phase_predict(estimator, train.X)
    resubstitution = _metrics(train, p_train)
    run_name = f"{name}_tuned"
    metadata = {
        "artifact_format": "pathsnap-supervised-candidate-v1",
        "model_name": name, "run_name": run_name, "family": PHASE_FAMILIES[name],
        "phase": PHASE_NAME, "seed": config.seed,
        "parameters": parameters,
        "fixed_mechanics": ({"knn": KNN_FIXED_PARAMETERS} if name == "knn_distance_weighted" else None),
        "feature_names": list(FEATURE_NAMES), "feature_version": FEATURE_VERSION,
        "training_seconds": fit_seconds, "training_observations": len(train.offsets) - 1,
        "training_trip_count": len(np.unique(train.observation_trip_ids)),
        "training_trip_ids": np.unique(train.observation_trip_ids).tolist(),
        "training_candidate_rows": len(train.y),
        "training_sampling": "deterministic np.linspace first-to-last, maximum 350 observations per trip; no label-dependent sampling",
        "training_weighting": ("inverse candidate count per observation and inverse sampled observation count per trip, "
                               "normalized to mean1" + ("; applied through the KNN reference-bank inclusion probabilities because "
                               "KNeighborsClassifier.fit accepts no sample_weight" if name == "knn_distance_weighted" else "")),
        "preprocessing": phase_preprocessing_description(name),
        "class_threshold": 0.5,
        "emission_policy": "clip binary positive probabilities to [1e-7,1-1e-7], normalize within candidate set; graph/GPS only",
        "candidate_config": {"candidate_radius_m": config.candidate_radius_m, "max_radius_m": config.max_radius_m,
                             "max_candidates": config.max_candidates, "sample_spacing_m": config.sample_spacing_m},
        "proxy_label_policy": PROXY_LABEL_POLICY,
        "metrics": {"train_resubstitution": resubstitution, "cv_objective": winner_cv["objective"],
                    "cv_oof_metrics": winner_cv["oof_metrics"]},
        "selection": {"phase": PHASE_NAME, "selected_model": name,
                      "selection_objective": winner_cv["objective"],
                      "hpo_trials_including_measured_prior": HPO_TRIALS,
                      "selection_file": "experiments/phase_three/selection.json",
                      "hpo_summary_file": "experiments/phase_three/bayesian_optimization.json",
                      "student_test_consulted_before_freeze": False},
        "timing": {"fit_all80_seconds": fit_seconds, "train_inference_seconds": train_inference_seconds},
        "versions": versions(), "hardware": hardware(),
        "final_oof_export": replay,
        "coverage_file": f"experiments/cache/{TRAIN_CACHE}.json",
        "experiment_results_file": "experiments/phase_three/results.json",
        "knn_reference_bank": bank_info,
        "fit_warnings": fit_warnings,
    }
    bundle = {"estimator": estimator, "feature_names": list(FEATURE_NAMES),
              "feature_version": FEATURE_VERSION, "metadata": metadata}
    tmp = ARTIFACTS / "model.joblib.tmp"
    joblib.dump(bundle, tmp, compress=3)
    os.replace(tmp, ARTIFACTS / "model.joblib")  # atomic; never rewritten after freeze
    schema = {"version": FEATURE_VERSION, "dtype": "float32",
              "row_order": "observation order, then unchanged Candidates.edges/fractions/distances order; IDs persisted separately, never numerical features",
              "reference_route_used_as_features": False,
              "features": [{"index": i, "name": n, "description": FEATURE_DESCRIPTIONS[n]}
                           for i, n in enumerate(FEATURE_NAMES)]}
    save_json(ARTIFACTS / "feature_schema.json", schema)
    save_json(ARTIFACTS / "model_metadata.json", metadata)
    sha = _sha256(ARTIFACTS / "model.joblib")
    size = (ARTIFACTS / "model.joblib").stat().st_size
    manifest = {"phase": PHASE_NAME, "created_at_utc": utc_now(),
                "files": {"model.joblib": {"sha256": sha, "bytes": size},
                          "model_metadata.json": {"sha256": _sha256(ARTIFACTS / "model_metadata.json")},
                          "feature_schema.json": {"sha256": _sha256(ARTIFACTS / "feature_schema.json")},
                          "clean_load_report.json": None}}
    smoke = clean_environment_smoke_load()
    manifest["files"]["clean_load_report.json"] = {"sha256": _sha256(ARTIFACTS / "clean_load_report.json")}
    save_json(ARTIFACTS / "artifact_manifest.json", manifest)
    freeze = {"model": "artifacts/phase_three/model.joblib", "sha256": sha, "bytes": size,
              "model_name": name, "run_name": run_name, "parameters": parameters,
              "objective": winner_cv["objective"],
              "feature_schema": "artifacts/phase_three/feature_schema.json",
              "metadata": "artifacts/phase_three/model_metadata.json",
              "clean_environment_load": {"passed": smoke["passed"], "python": smoke["python"],
                                         "report": "artifacts/phase_three/clean_load_report.json"},
              "frozen_at_utc": utc_now(),
              "extras": {"phase": PHASE_NAME, "immutable_after_freeze": True,
                         "selection_file": "experiments/phase_three/selection.json",
                         "historical_artifact_untouched": "artifacts/model.joblib (prior freeze) is not modified by phase three",
                         "consumer_note": "phase-three winner for decoder/routes/live phase_three scopes; routes must be re-derived by their owners"}}
    save_json(ARTIFACTS / "model_freeze.json", freeze)
    log("phase_model_frozen", phase=PHASE_NAME, model_name=name, sha256=sha, bytes=size)
    signal = {"signal": "PHASE_MODEL_FROZEN", "model": freeze["model"], "sha256": sha,
              "bytes": size, "model_name": name, "run_name": run_name,
              "parameters": parameters, "objective": winner_cv["objective"],
              "frozen_at_utc": freeze["frozen_at_utc"]}
    print(json.dumps(signal, allow_nan=False))
    return signal


def ambiguity_strata(dataset, prediction) -> dict:
    """Top-1 and clipped categorical NLL by ambiguity / candidate-miss strata."""
    p, offsets = prediction["probabilities"], prediction["offsets"]
    y = prediction["y"]
    n = len(offsets) - 1
    ambiguous = np.asarray(dataset.ambiguous, bool)
    correct = np.zeros(n, bool)
    present = np.zeros(n, bool)
    target = np.zeros(n)
    for i, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        if hi > lo:
            local = np.clip(p[lo:hi], 1e-7, 1 - 1e-7)
            normalized = local / local.sum()
            row = int(np.argmax(normalized))
            correct[i] = bool(y[lo + row])
            present[i] = bool(np.any(y[lo:hi]))
            target[i] = float(normalized[y[lo:hi] == 1].sum()) if present[i] else 0.0
    nll = -np.log(np.maximum(target, 1e-7))

    def block(mask):
        return {"observations": int(mask.sum()),
                "top1_accuracy": float(correct[mask].mean()) if mask.any() else None,
                "categorical_log_loss_clipped": float(nll[mask].mean()) if mask.any() else None}

    return {"strata_definition": "ambiguous = proxy within the disclosed ambiguity margin; candidate_miss = no candidate matches the proxy edge; strata overlap and each keeps misses in its denominator",
            "ambiguous_proxy": block(ambiguous), "non_ambiguous_proxy": block(~ambiguous),
            "candidate_miss": block(~present), "candidates_present": block(present)}


def save_sparse_confusion(path: Path, sparse: Counter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trip_id", "reference_edge", "predicted_edge", "count"])
        for (trip, ref, pred), count in sorted(sparse.items(), key=lambda kv: (-kv[1], kv[0])):
            writer.writerow([trip, ref, pred, count])


def base_family(model_name: str) -> str:
    return model_name[:-len("_tuned")] if model_name.endswith("_tuned") else model_name


def stage_student(config):
    refuse_existing_stage("student")
    selection = require(PHASE_DIR / "selection.json", "run --stage select first")
    freeze = require(ARTIFACTS / "model_freeze.json", "run --stage freeze first")
    bayes = require(PHASE_DIR / "bayesian_optimization.json", "run --stage hpo first")
    selected = selection["selected_model"]
    tuned_name = f"{selected}_tuned"
    train, _ = load_train_and_folds(config)
    student = load_cached_dataset("student_test")
    graphs = {p.name: load_graph(p) for p in discover_trips(DATA_ROOT / "student_test")}
    if set(graphs) != set(np.unique(student.observation_trip_ids).tolist()):
        raise AssertionError("student graph set does not match cached trip ids")
    all_rows = np.arange(len(train.y))
    evaluations, sequence, strata = {}, {}, {}
    student_predictions = {}
    for name in FAMILIES:
        if name == "logistic_regression":
            shared_eval = SHARED / "evaluation" / "logistic_regression.json"
            ev = load_json(shared_eval)
            prediction = load_prediction_npz(SHARED / "predictions" / "logistic_regression_student.npz")
            assert_prediction_identity(student, prediction)
            evaluation = dict(ev)
            evaluation.update({"phase": PHASE_NAME, "reused": True, "fitted": False,
                "provenance": {"evaluation_file": str(shared_eval.relative_to(ROOT)),
                               "evaluation_sha256": _sha256(shared_eval),
                               "student_predictions_file": "experiments/predictions/logistic_regression_student.npz",
                               "note": "valid saved all80 fit of the identical configuration; numbers reused without refitting"}})
            oof_prediction = load_prediction_npz(SHARED / "predictions" / "logistic_regression_oof.npz")
        else:
            parameters = PHASE_DEFAULT_PARAMETERS[name]
            tuned_is_default = json.dumps(bayes["best_parameters"], sort_keys=True) == json.dumps(parameters, sort_keys=True)
            if name == selected and tuned_is_default:
                # The frozen artifact IS this baseline configuration.
                bundle = joblib.load(ARTIFACTS / "model.joblib")
                estimator = bundle["estimator"]
                md = bundle["metadata"]
                fit_seconds = md["timing"]["fit_all80_seconds"]
                train_resub = md["metrics"]["train_resubstitution"]
                train_seconds_infer = md["timing"]["train_inference_seconds"]
                source = "frozen_artifact"
            else:
                estimator, fit_seconds, fit_warnings, bank_info = phase_fit(
                    name, parameters, config, train, all_rows, KNN_FINAL_FIT_INDEX)
                p_train, train_seconds_infer = phase_predict(estimator, train.X)
                train_resub = _metrics(train, p_train)
                baselines = ARTIFACTS / "baselines"
                baselines.mkdir(parents=True, exist_ok=True)
                joblib.dump({"estimator": estimator, "feature_names": list(FEATURE_NAMES),
                             "feature_version": FEATURE_VERSION,
                             "metadata": {"model_name": name, "run_name": name, "phase": PHASE_NAME,
                                          "seed": config.seed, "parameters": parameters,
                                          "training_trip_ids": np.unique(train.observation_trip_ids).tolist()}},
                            baselines / f"{name}.joblib", compress=3)
                source = "fresh_default_parameter_fit"
            p_student, student_seconds = phase_predict(estimator, student.X)
            student_metrics = _metrics(student, p_student)
            student_file = save_prediction_npz(PHASE_DIR / "predictions" / f"{name}_student.npz", student, p_student)
            prediction = load_prediction_npz(PHASE_DIR / "predictions" / f"{name}_student.npz")
            evaluation = {"run_name": name, "model_name": name, "family": PHASE_FAMILIES[name],
                          "phase": PHASE_NAME, "parameters": parameters,
                          "training_seconds": fit_seconds,
                          "train_inference_seconds": train_seconds_infer,
                          "student_inference_seconds": student_seconds,
                          "warnings": [], "train_resubstitution_metrics": train_resub,
                          "student_metrics": student_metrics, "fitted": True,
                          "from_frozen_artifact": source == "frozen_artifact", "fit_source": source,
                          "student_predictions_file": student_file}
            oof_record = load_json(PHASE_DIR / "cv" / f"{name}.json")
            oof_prediction = load_prediction_npz(ROOT / oof_record["oof_predictions_file"])
        save_json(PHASE_DIR / "evaluation" / f"{name}.json", evaluation)
        evaluations[name] = evaluation
        student_predictions[name] = prediction
        metrics_student, sparse = observation_metrics(prediction, graphs)
        save_sparse_confusion(PHASE_DIR / "sparse_confusion" / f"{name}_student.csv", sparse)
        metrics_oof, _ = observation_metrics(oof_prediction, None)
        sequence[name] = {"student_test": metrics_student, "train_out_of_fold": metrics_oof}
        strata[name] = ambiguity_strata(student, prediction)
        log("phase_student_evaluation", phase=PHASE_NAME, model_name=name,
            student_top1=metrics_student["top1_accuracy"],
            student_observation_edge_macro_f1=metrics_student["observation_edge_macro_f1"])
    # Tuned winner: predictions must come from the frozen artifact bytes.
    frozen_bundle = joblib.load(ARTIFACTS / "model.joblib")
    if selected in NEW_FAMILIES and json.dumps(bayes["best_parameters"], sort_keys=True) == json.dumps(PHASE_DEFAULT_PARAMETERS[selected], sort_keys=True):
        tuned_prediction = student_predictions[selected]
        save_prediction_npz(PHASE_DIR / "predictions" / f"{tuned_name}_student.npz",
                            student, tuned_prediction["probabilities"])
        tuned_train_inference = evaluations[selected]["train_inference_seconds"]
        tuned_student_seconds = evaluations[selected]["student_inference_seconds"]
    else:
        p_tuned, tuned_student_seconds = phase_predict(frozen_bundle["estimator"], student.X)
        tuned_file = save_prediction_npz(PHASE_DIR / "predictions" / f"{tuned_name}_student.npz", student, p_tuned)
        tuned_prediction = load_prediction_npz(PHASE_DIR / "predictions" / f"{tuned_name}_student.npz")
        tuned_train_inference = frozen_bundle["metadata"]["timing"]["train_inference_seconds"]
    md = frozen_bundle["metadata"]
    if selected == "logistic_regression" and bayes["best_parameters"] == PHASE_DEFAULT_PARAMETERS[selected]:
        np.testing.assert_allclose(tuned_prediction["probabilities"], student_predictions[selected]["probabilities"],
                                   rtol=0.0, atol=1e-12, err_msg="same-parameter LR saved/fresh fits disagree")
    tuned_student_metrics = _metrics(student, tuned_prediction["probabilities"])
    tuned_eval = {"run_name": tuned_name, "model_name": selected, "family": PHASE_FAMILIES[selected],
                  "phase": PHASE_NAME, "parameters": bayes["best_parameters"],
                  "training_seconds": md["timing"]["fit_all80_seconds"],
                  "train_inference_seconds": tuned_train_inference,
                  "student_inference_seconds": tuned_student_seconds,
                  "warnings": md.get("fit_warnings", []),
                  "train_resubstitution_metrics": md["metrics"]["train_resubstitution"],
                  "student_metrics": tuned_student_metrics, "fitted": False,
                  "from_frozen_artifact": True, "fit_source": "frozen_artifact",
                  "student_predictions_file": str((PHASE_DIR / "predictions" / f"{tuned_name}_student.npz").relative_to(ROOT))}
    save_json(PHASE_DIR / "evaluation" / f"{tuned_name}.json", tuned_eval)
    evaluations[tuned_name] = tuned_eval
    metrics_student, sparse = observation_metrics(tuned_prediction, graphs)
    save_sparse_confusion(PHASE_DIR / "sparse_confusion" / f"{tuned_name}_student.csv", sparse)
    tuned_cv_record = load_json(PHASE_DIR / "cv" / f"{tuned_name}.json")
    metrics_oof, _ = observation_metrics(load_prediction_npz(ROOT / tuned_cv_record["oof_predictions_file"]), None)
    sequence[tuned_name] = {"student_test": metrics_student, "train_out_of_fold": metrics_oof}
    strata[tuned_name] = ambiguity_strata(student, tuned_prediction)
    # Comparison table in the exact shared 30-column schema.
    shared_results = load_json(SHARED / "results.json")
    columns = list(shared_results["comparison"][0].keys())
    cv_records = {n: load_json(PHASE_DIR / "cv" / f"{n}.json") for n in FAMILIES}
    cv_records[tuned_name] = tuned_cv_record

    def row(model_name):
        cvr, ev = cv_records[model_name], evaluations[model_name]
        om, oof_m = sequence[model_name]["student_test"], sequence[model_name]["train_out_of_fold"]
        sm = ev["student_metrics"]
        values = {"model": model_name, "family": PHASE_FAMILIES[base_family(model_name)],
                  "parameters": json.dumps(cvr["parameters"], sort_keys=True),
                  "cv_macro_f1": cvr["oof_metrics"]["macro_f1"],
                  "cv_macro_f1_fold_mean": cvr.get("macro_f1_fold_mean"),
                  "cv_macro_f1_fold_std": cvr.get("macro_f1_fold_std"),
                  "cv_top1_accuracy": cvr["oof_metrics"]["top1_accuracy"],
                  "cv_trip_macro_top1_accuracy": cvr["objective"],
                  "cv_log_loss": cvr["oof_metrics"]["log_loss"],
                  "cv_training_seconds": cvr.get("training_seconds_total"),
                  "cv_inference_seconds": cvr.get("inference_seconds_total"),
                  "student_accuracy": sm["accuracy"], "student_precision": sm["precision"],
                  "student_recall": sm["recall"], "student_macro_f1": sm["macro_f1"],
                  "student_top1_accuracy": sm["top1_accuracy"],
                  "student_trip_macro_top1_accuracy": sm["trip_macro_top1_accuracy"],
                  "student_log_loss": sm["log_loss"],
                  "student_candidate_miss_rate": sm["candidate_miss_rate"],
                  "fit_all80_seconds": ev.get("training_seconds"),
                  "student_inference_seconds": ev.get("student_inference_seconds"),
                  "route_continuity": om.get("raw_classifier_continuity"),
                  "student_candidate_binary_macro_f1": sm["macro_f1"],
                  "student_observation_edge_macro_f1": om["observation_edge_macro_f1"],
                  "student_trip_macro_observation_edge_f1": om["trip_macro_observation_edge_f1"],
                  "student_observation_categorical_log_loss_clipped": om["observation_categorical_log_loss_clipped"],
                  "student_raw_classifier_continuity": om.get("raw_classifier_continuity"),
                  "student_edge_change_continuity": om.get("edge_change_continuity"),
                  "cv_observation_edge_macro_f1": oof_m["observation_edge_macro_f1"],
                  "cv_observation_categorical_log_loss_clipped": oof_m["observation_categorical_log_loss_clipped"]}
        for key in ("student_accuracy", "student_macro_f1", "student_top1_accuracy", "student_log_loss"):
            if values[key] is None:
                raise AssertionError(f"missing essential metric {key} for {model_name}")
        return {c: values.get(c) for c in columns}

    rows = [row(n) for n in FAMILIES] + [row(tuned_name)]
    with (PHASE_DIR / "model_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in r.items()})
    save_json(PHASE_DIR / "sequence_classification.json", {
        "phase": PHASE_NAME,
        "target": "route-projection proxy, not independently supplied per-GPS ground truth",
        "models": sequence})
    stage_seconds = {entry["stage"]: entry["seconds"] for entry in
                     (json.loads(line) for line in (PHASE_DIR / "stage_log.jsonl").read_text().splitlines())} \
        if (PHASE_DIR / "stage_log.jsonl").exists() else {}
    results = {
        "phase": {"name": PHASE_NAME, "model_count": PHASE_MODEL_COUNT,
                  "families": FAMILIES, "user_selected_families": True,
                  "protocol": "experiments/phase_three/PROTOCOL.md",
                  "write_scope": ["experiments/phase_three/", "artifacts/phase_three/", "mapmatch/phase_models.py"],
                  "organizer_five_model_note": "deferred by explicit user instruction; the historical five-model CV/HPO evidence and HistGB freeze remain untouched in the experiments/ root and artifacts/"},
        "config": {**dataclasses.asdict(config), "thread_environment": THREAD_ENV,
                   "hpo_trials_including_measured_prior": HPO_TRIALS,
                   "hpo_seeded_initial_design_trials": HPO_SEEDED},
        "versions": versions(),
        "target_policy": shared_results.get("target_policy"),
        "proxy_label_policy": PROXY_LABEL_POLICY,
        "selection_model": selected, "final_run_name": tuned_name,
        "student_gating": {"selection_recorded_before_student_load": True,
                           "model_frozen_before_student_load": True,
                           "note": "the student cache is loaded only by this stage, after selection.json and model_freeze.json exist"},
        "cv": {n: cv_records[n] for n in list(FAMILIES) + [tuned_name]},
        "evaluations": evaluations,
        "comparison": rows,
        "bayesian": bayes,
        "lr_reuse": load_json(PHASE_DIR / "cv" / "logistic_regression.json").get("reuse_provenance"),
        "knn_bank_policy": {"budget": KNN_REFERENCE_BANK_BUDGET,
                            "description": "class-stratified, training-weight-proportional sampling without replacement over each fit's own TRAIN rows; deterministic seed; fixed before any measurement"},
        "coverage_summary": {
            "train": {"trips": EXPECTED["train"]["trips"], "observations": EXPECTED["train"]["observations"],
                      "candidate_rows": EXPECTED["train"]["rows"], "candidate_miss_count": EXPECTED["train"]["misses"],
                      "ambiguous_proxy_observations": int(np.asarray(train.ambiguous).sum()),
                      "cache": f"experiments/cache/{TRAIN_CACHE}.npz", "cache_sha256": TRAIN_CACHE_SHA256},
            "student_test": {"trips": EXPECTED["student_test"]["trips"], "observations": EXPECTED["student_test"]["observations"],
                             "candidate_rows": EXPECTED["student_test"]["rows"], "candidate_miss_count": EXPECTED["student_test"]["misses"],
                             "ambiguous_proxy_observations": int(np.asarray(student.ambiguous).sum()),
                             "cache": f"experiments/cache/{STUDENT_CACHE}.npz", "cache_sha256": STUDENT_CACHE_SHA256}},
        "ambiguity_strata": strata,
        "frozen_artifact": {"model": "artifacts/phase_three/model.joblib", "sha256": freeze["sha256"],
                            "bytes": freeze["bytes"], "freeze_record": "artifacts/phase_three/model_freeze.json"},
        "elapsed_seconds": stage_seconds,
        "elapsed_seconds_note": "per-stage measured seconds from stage_log.jsonl; invocation mode and process ID are recorded with each new stage entry",
        "observation_metrics_supplement": "sequence_classification.json",
    }
    save_json(PHASE_DIR / "results.json", results)
    log("phase_results_ready", phase=PHASE_NAME, selected_model=selected,
        final_run_name=tuned_name)
    print(json.dumps({"signal": "PHASE_RESULTS_READY", "selected_model": selected,
                      "final_run_name": tuned_name,
                      "results_file": "experiments/phase_three/results.json",
                      "comparison_file": "experiments/phase_three/model_comparison.csv"}, allow_nan=False))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["cv", "select", "hpo", "freeze", "student", "all"],
                        required=True)
    args = parser.parse_args(argv)
    stages = ["cv", "select", "hpo", "freeze", "student"] if args.stage == "all" else [args.stage]
    for stage in stages:
        refuse_existing_stage(stage)  # Before directories or run metadata are touched.
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    (PHASE_DIR / "cv").mkdir(exist_ok=True)
    (PHASE_DIR / "predictions").mkdir(exist_ok=True)
    (PHASE_DIR / "evaluation").mkdir(exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    config = phase_config()
    run_config = {
        "phase": PHASE_NAME, "config": dataclasses.asdict(config),
        "versions": versions(), "thread_environment": THREAD_ENV,
        "caches": {split: {"file": f"experiments/cache/{spec['cache']}.npz", "sha256": spec["sha256"]}
                   for split, spec in EXPECTED.items()},
        "hpo_budget": {"trials_including_measured_prior": HPO_TRIALS,
                       "seeded_initial_design": HPO_SEEDED,
                       "gp_ei_trials": HPO_TRIALS - 1 - HPO_SEEDED},
        "stages": ["cv", "select", "hpo", "freeze", "student"],
        "protocol": "experiments/phase_three/PROTOCOL.md",
        "written_at_utc": utc_now()}
    run_config_path = PHASE_DIR / "run_config.json"
    if run_config_path.exists():
        prior = load_json(run_config_path)
        for key in ("phase", "config", "caches", "hpo_budget", "protocol"):
            if prior[key] != run_config[key]:
                raise AssertionError("saved run configuration differs: " + key)
    else:
        save_json(run_config_path, run_config)
    handlers = {"cv": stage_cv, "select": stage_select, "hpo": stage_hpo,
                "freeze": stage_freeze, "student": stage_student}
    for stage in stages:
        started = utc_now()
        t = time.perf_counter()
        with threadpool_limits(1):
            handlers[stage](config)
        append_json(PHASE_DIR / "stage_log.jsonl",
                    {"stage": stage, "invocation_mode": args.stage, "process_id": os.getpid(),
                     "started_at_utc": started, "seconds": time.perf_counter() - t,
                     "completed_at_utc": utc_now(), "versions": versions()})


if __name__ == "__main__":
    main()
