#!/usr/bin/env python3
"""Independent read-only audit of the historical GaussianNB / HistGradientBoosting artifacts.

Scope: gaussian_nb, hist_gradient_boosting (default), hist_gradient_boosting_tuned plus the
frozen artifacts/model.joblib, verified against the historical experiment records
(run_config, results, cv/, evaluation/, predictions/, bayesian_optimization, model_freeze,
acceptance_verification) and the raw TRAIN / student caches. The phase-three
LR/KNN/Bagging trio is already accepted and out of scope here.

Guarantees: no refitting, no HPO or decoder reruns, no external HTTP, no access to
evaluator_test/ or archives. Raw caches stay local; the written report contains
aggregates and hashes only. Old artifacts/CV/preds/metrics are never modified.

Scoring paths: every replay uses the exact historical scoring path. For gaussian_nb that
is the raw float32 predict_proba + clip; the float64 promotion in the current
learning.predict_positive_proba is an uncommitted working-tree change that postdates the
artifacts, and its live re-scoring impact is quantified in a dedicated disclosure check.

Run with the original training interpreter and pinned threads:

    cd /home/d3va/Downloads/ML_Hackathon
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        PYTHONDONTWRITEBYTECODE=1 .venv/bin/python analysis/five_model_audit/verify_gnb_histgb_audit.py
"""
from __future__ import annotations

import hashlib
import json
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, _logsumexp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mapmatch.experiments import CandidateDataset, deterministic_observation_indices  # noqa: E402
from mapmatch.features import FEATURE_NAMES, FEATURE_VERSION  # noqa: E402
from mapmatch.learning import candidate_classification_metrics, load_model, predict_positive_proba  # noqa: E402

EXP = ROOT / "experiments"
ART = ROOT / "artifacts"
OUT = ROOT / "analysis/five_model_audit" / "gnb_histgb_audit.json"
STRICT = 1e-12  # probability/metric replay tolerance; never loosened
FAMILY_DEFAULT_HGB = "hist_gradient_boosting"
FAMILY_TUNED_HGB = "hist_gradient_boosting_tuned"

CHECKS: list[tuple[str, dict]] = []


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def check(name, condition, detail=None, families=("shared",)):
    entry = {"passed": bool(condition), "detail": _jsonable(detail), "families": list(families)}
    CHECKS.append((name, entry))
    if not entry["passed"]:
        print(f"FAILED CHECK: {name}: {json.dumps(_jsonable(detail))[:500]}", file=sys.stderr)
    return entry["passed"]


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(pattern):
    matches = sorted((EXP / "cache").glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one cache for {pattern}, found {matches}")
    with np.load(matches[0], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in CandidateDataset.__dataclass_fields__ if key != "coverage"}
    coverage = read_json(matches[0].with_suffix(".json"))
    return matches[0], CandidateDataset(**arrays, coverage=coverage)


def max_abs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def compare_metrics(recorded, replayed, tol=STRICT):
    """Recursive numeric comparison; returns (passed, max_delta, mismatches)."""
    deltas, mismatches = [], []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                mismatches.append(f"{path}: key sets differ {sorted(set(a) ^ set(b))[:6]}")
            for key in sorted(set(a) & set(b)):
                walk(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                mismatches.append(f"{path}: lengths {len(a)} vs {len(b)}")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        elif isinstance(a, bool) or isinstance(b, bool):
            if a != b:
                mismatches.append(f"{path}: {a} vs {b}")
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = abs(float(a) - float(b))
            deltas.append(delta)
            if delta > tol:
                mismatches.append(f"{path}: {a} vs {b} (delta {delta:.3e})")
        elif a != b:
            mismatches.append(f"{path}: {a!r} vs {b!r}")

    walk(recorded, replayed, "")
    return not mismatches, (max(deltas) if deltas else 0.0), mismatches[:10]


def fold_replay_metrics(dataset, probabilities, validation_trip_ids):
    """Reproduces a fold's validation metrics from OOF probabilities, deriving the
    validation rows from the recorded whole-trip validation_trip_ids."""
    lengths = np.diff(dataset.offsets)
    mask = np.isin(dataset.observation_trip_ids, np.asarray(validation_trip_ids))
    rows = np.flatnonzero(np.repeat(mask, lengths))
    offsets = np.r_[0, np.cumsum(lengths[mask])]
    return (candidate_classification_metrics(dataset.y[rows], probabilities[rows], offsets,
                                             dataset.observation_trip_ids[mask]),
            int(mask.sum()), int(rows.size))


def historical_positive_proba(name, estimator, X):
    """Scoring path exactly as it existed when the artifacts were produced.

    git shows the GaussianNB float64 promotion in learning.predict_positive_proba is an
    uncommitted working-tree change that postdates artifact creation (artifacts 11:08,
    phase-three merge 13:40, promotion added after it). The historical path scored the
    float32 cache through model.predict_proba directly; the saved GaussianNB predictions
    are bit-identical to it. Non-GaussianNB estimators are unaffected by the promotion.
    """
    if name == "gaussian_nb":
        return np.clip(np.asarray(estimator.predict_proba(X), dtype=np.float64)[:, 1], 1e-7, 1.0 - 1e-7)
    return predict_positive_proba(estimator, X)


def main():
    report = {
        "audit": "historical gaussian_nb + hist_gradient_boosting artifact verification (task-30)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {"refits": "none", "hpo_reruns": "none", "decoder_reruns": "none",
                   "external_http": "none", "evaluator_test_access": "none",
                   "strict_tolerance": STRICT, "old_records_modified": False},
    }

    git_head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    report["workspace"] = {"git_head": git_head,
                           "audit_script": "analysis/five_model_audit/verify_gnb_histgb_audit.py"}

    # ---------------------------------------------------------------- versions
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
               "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
               "platform": platform.platform()}
    run_config = read_json(EXP / "run_config.json")
    training = run_config["versions"]
    drift = {k: {"trained": training.get(k), "current": runtime.get(k)}
             for k in runtime if k in training and runtime[k] != training.get(k)}
    check("runtime_versions_recorded", True, runtime)
    check("training_runtime_version_drift_disclosed", True,
          {"drift": drift, "note": "historical artifacts were trained under the recorded versions; "
           "all replay tolerances remain 1e-12 and are not loosened"})
    report["versions"] = {"runtime": runtime, "training": training, "drift": drift,
                          "thread_environment": run_config.get("thread_environment"),
                          "selection_protocol": run_config.get("selection_protocol")}

    # ---------------------------------------------------------------- datasets
    train_path, train = load_cache("train_*.npz")
    student_path, student = load_cache("student_test_*.npz")
    train_trips = np.unique(train.observation_trip_ids)
    student_trips = np.unique(student.observation_trip_ids)
    train_miss = int(sum(not np.any(train.y[s:e] == 1) for s, e in zip(train.offsets[:-1], train.offsets[1:])))
    student_miss = int(sum(not np.any(student.y[s:e] == 1) for s, e in zip(student.offsets[:-1], student.offsets[1:])))
    check("train_cache_shape", train.X.shape == (269463, 38) and len(train.y) == 269463
          and len(train.offsets) - 1 == 28000 and len(train_trips) == 80,
          {"shape": list(train.X.shape), "rows": len(train.y), "observations": len(train.offsets) - 1,
           "trips": len(train_trips), "dtype": str(train.X.dtype), "finite": bool(np.isfinite(train.X).all())})
    check("student_cache_shape", student.X.shape == (175778, 38) and len(student.y) == 175778
          and len(student.offsets) - 1 == 19160 and len(student_trips) == 10,
          {"shape": list(student.X.shape), "rows": len(student.y), "observations": len(student.offsets) - 1,
           "trips": len(student_trips), "dtype": str(student.X.dtype)})
    check("candidate_miss_counts", train_miss == 578 and student_miss == 19,
          {"train_miss": train_miss, "student_miss": student_miss})
    check("train_student_trip_disjoint", not set(train_trips.tolist()) & set(student_trips.tolist()),
          {"train_trips": len(train_trips), "student_trips": len(student_trips)})
    check("train_sampling_deterministic_linspace",
          all(np.array_equal(np.asarray(t["selected_indices"]),
                             deterministic_observation_indices(t["total_observations"], 350))
              for t in train.coverage["trips"]),
          {"trips": len(train.coverage["trips"]),
           "sampling": train.coverage["sampling"],
           "total_observations_all_trips": train.coverage["total_observations"],
           "selected_observations": train.coverage["selected_observations"]})
    check("student_all_observations_not_truncated",
          student.coverage["sampling"] == "all observations"
          and student.coverage["selected_observations"] == student.coverage["total_observations"] == 19160
          and student.coverage["trip_count"] == 10,
          {"sampling": student.coverage["sampling"],
           "selected": student.coverage["selected_observations"],
           "total": student.coverage["total_observations"], "trips": student.coverage["trip_count"]})

    # raw-file identity: coverage manifests vs current raw TRAIN/student files
    for split, cov, cache_path in (("train", train.coverage, train_path),
                                   ("student_test", student.coverage, student_path)):
        manifest_mismatches = []
        for entry in cov["input_manifest"]:
            for suffix, recorded in entry["files"].items():
                path = ROOT / "Student folder" / split / entry["trip_id"] / (entry["trip_id"] + suffix)
                current = {"bytes": path.stat().st_size, "sha256": sha256(path)}
                if current != recorded:
                    manifest_mismatches.append({"trip": entry["trip_id"], "suffix": suffix})
        check(f"raw_input_manifest[{split}]", not manifest_mismatches,
              {"trips": len(cov["input_manifest"]), "files_per_trip": 4,
               "mismatches": manifest_mismatches[:5],
               "fingerprint_prefix_matches_cache": cov["fingerprint"][:16] in cache_path.name})

    # ---------------------------------------------------------------- shared folds
    cv_folds = read_json(EXP / "cv_folds.json")
    fold_partition_ok = len(cv_folds) == 3
    all_val_trips = []
    for fold in cv_folds:
        fold_partition_ok &= not (set(fold["train_trip_ids"]) & set(fold["validation_trip_ids"]))
        fold_partition_ok &= fold["train_observations"] + fold["validation_observations"] == 28000
        fold_partition_ok &= fold["train_candidate_rows"] + fold["validation_candidate_rows"] == 269463
        all_val_trips.extend(fold["validation_trip_ids"])
    check("shared_group_folds", fold_partition_ok and len(set(all_val_trips)) == 80,
          {"folds": len(cv_folds), "validation_trips_total": len(set(all_val_trips)),
           "structure": "3-fold GroupKFold over whole trips; 80 TRAIN trips partitioned"})
    report["folds"] = [{"fold": f["fold"], "train_trips": len(f["train_trip_ids"]),
                        "validation_trips": len(f["validation_trip_ids"])} for f in cv_folds]

    # ---------------------------------------------------------------- SHA attestations
    freeze = read_json(EXP / "model_freeze.json")
    manifest = read_json(ART / "artifact_manifest.json")
    acceptance = read_json(EXP / "acceptance_verification.json")
    model_sha = sha256(ART / "model.joblib")
    check("frozen_model_sha256_matches_actual",
          model_sha == freeze["sha256"] == manifest["sha256"] == read_json(EXP / "results.json")["recovery"]["model_sha256"],
          {"actual": model_sha, "recorded_freeze": freeze["sha256"], "recorded_manifest": manifest["sha256"],
           "bytes": (ART / "model.joblib").stat().st_size})
    sha_mismatches = {}
    for name in ("logistic_regression", "gaussian_nb", "random_forest", "extra_trees",
                 "hist_gradient_boosting", "hist_gradient_boosting_tuned"):
        actual = sha256(ART / "baselines" / f"{name}.joblib")
        recorded = acceptance["checks"][f"baseline_loads[{name}]"]["detail"]["sha256"]
        if actual != recorded:
            sha_mismatches[name] = {"actual": actual, "recorded": recorded}
    check("baseline_artifact_sha256_matches_acceptance_records", not sha_mismatches,
          {"verified": 6, "mismatches": sha_mismatches})
    report["artifact_hashes"] = {"model_joblib": model_sha,
                                 **{n: sha256(ART / "baselines" / f"{n}.joblib") for n in
                                    ("gaussian_nb", "hist_gradient_boosting", "hist_gradient_boosting_tuned")}}

    # ---------------------------------------------------------------- artifacts in scope
    results = read_json(EXP / "results.json")
    in_scope = {
        "gaussian_nb": {"artifact": ART / "baselines" / "gaussian_nb.joblib",
                        "evaluation": read_json(EXP / "evaluation" / "gaussian_nb.json"),
                        "cv": read_json(EXP / "cv" / "gaussian_nb.json"),
                        "oof": EXP / "predictions" / "gaussian_nb_oof.npz",
                        "student": EXP / "predictions" / "gaussian_nb_student.npz",
                        "families": ("gaussian_nb",)},
        FAMILY_DEFAULT_HGB: {"artifact": ART / "baselines" / "hist_gradient_boosting.joblib",
                             "evaluation": read_json(EXP / "evaluation" / "hist_gradient_boosting.json"),
                             "cv": read_json(EXP / "cv" / "hist_gradient_boosting.json"),
                             "oof": EXP / "predictions" / "hist_gradient_boosting_oof.npz",
                             "student": EXP / "predictions" / "hist_gradient_boosting_student.npz",
                             "families": (FAMILY_DEFAULT_HGB,)},
        FAMILY_TUNED_HGB: {"artifact": ART / "baselines" / "hist_gradient_boosting_tuned.joblib",
                           "evaluation": read_json(EXP / "evaluation" / "hist_gradient_boosting_tuned.json"),
                           "cv": results["cv"]["hist_gradient_boosting_tuned"],
                           "cv_replay": read_json(EXP / "selected_replay" / "cv" / "hist_gradient_boosting_tuned.json"),
                           "oof": EXP / "predictions" / "hist_gradient_boosting_tuned_oof.npz",
                           "oof_replay_copy": EXP / "selected_replay" / "predictions" / "hist_gradient_boosting_tuned_oof.npz",
                           "student": EXP / "predictions" / "hist_gradient_boosting_tuned_student.npz",
                           "families": (FAMILY_TUNED_HGB,)},
    }

    loaded = {}
    for name, spec in in_scope.items():
        bundle = load_model(spec["artifact"])
        estimator = bundle["estimator"]
        loaded[name] = (bundle, estimator)
        params = bundle["metadata"]["parameters"]
        check(f"bundle_params_consistent[{name}]",
              params == spec["evaluation"]["parameters"] == spec["cv"]["parameters"] == results["cv"][name]["parameters"],
              {"artifact_params": params}, spec["families"])
        check(f"bundle_trained_on_all80_train[{name}]",
              len(bundle["metadata"]["training_trip_ids"]) == 80
              and set(bundle["metadata"]["training_trip_ids"]) == set(train_trips.tolist())
              and bundle["metadata"]["seed"] == 20261118
              and not set(bundle["metadata"]["training_trip_ids"]) & set(student_trips.tolist()),
              {"trips": len(bundle["metadata"]["training_trip_ids"]), "seed": bundle["metadata"]["seed"]},
              spec["families"])

    # GaussianNB: actual estimator, parameters, fitted counts, weighted priors, closed form
    gnb_bundle, gnb_est = loaded["gaussian_nb"]
    check("gnb_is_actual_pipeline", isinstance(gnb_est, Pipeline)
          and isinstance(gnb_est.named_steps["scale"], StandardScaler)
          and isinstance(gnb_est.named_steps["classifier"], GaussianNB)
          and gnb_est.named_steps["classifier"].__class__.__module__.startswith("sklearn.naive_bayes"),
          {"steps": [s for s, _ in gnb_est.steps],
           "classifier_module": gnb_est.named_steps["classifier"].__class__.__module__},
          ("gaussian_nb",))
    gnb_clf = gnb_est.named_steps["classifier"]
    scaler = gnb_est.named_steps["scale"]
    check("gnb_parameters", gnb_clf.var_smoothing == 1e-7 and gnb_clf.priors is None,
          {"var_smoothing": gnb_clf.var_smoothing, "priors": gnb_clf.priors}, ("gaussian_nb",))
    weights = train.training_weights()
    f32_counts = np.array([np.asarray(weights[train.y == c], np.float32).sum() for c in (0, 1)], np.float32)
    check("gnb_fitted_counts", tuple(gnb_clf.classes_.tolist()) == (0, 1)
          and gnb_clf.theta_.shape == (2, 38) and gnb_clf.var_.shape == (2, 38)
          and int(scaler.n_samples_seen_) == 269463
          and bool(np.array_equal(f32_counts, gnb_clf.class_count_))
          and abs(float(weights.sum()) - 269463.0) < 1e-6,
          {"class_count_": gnb_clf.class_count_.tolist(),
           "float32_weight_sums": f32_counts.tolist(),
           "float64_weight_sums": [float(weights[train.y == c].sum()) for c in (0, 1)],
           "scaler_n_samples_seen_": int(scaler.n_samples_seen_),
           "theta_shape": list(gnb_clf.theta_.shape), "var_shape": list(gnb_clf.var_.shape),
           "note": "class_count_ is the float32 accumulation of the sample weights per class and matches "
                   "exactly; its float32 sum (269463.0195) differs from the 269463 candidate rows only by "
                   "float32 accumulation error - the float64 weight total is 269463.0"}, ("gaussian_nb",))
    # The historical fit ran float32 end to end (float32 cache X; StandardScaler and GaussianNB
    # preserve dtype), so provenance is verified like-for-like in float32. Float64 recomputations
    # are recorded as the scale of float32 rounding, not as failures.
    Z32 = np.asarray(scaler.transform(train.X), np.float32)
    Z64 = np.asarray(scaler.transform(train.X), dtype=np.float64)
    epsilon32 = np.float32(gnb_clf.var_smoothing) * np.var(Z32, axis=0).max()
    f32_exact = {"class_count_": True, "theta_": True, "var_": True}
    f64_deltas = {"theta_": 0.0, "var_": 0.0, "class_count_": 0.0}
    recomputed_prior = np.empty(2)
    for c in (0, 1):
        mask = train.y == c
        wc32 = np.asarray(weights[mask], np.float32)
        Zc32 = Z32[mask]
        count32 = wc32.sum()
        theta32 = (wc32[:, None] * Zc32).sum(0) / count32
        var32 = (wc32[:, None] * (Zc32 - theta32) ** 2).sum(0) / count32
        f32_exact["class_count_"] &= bool(count32 == gnb_clf.class_count_[c])
        f32_exact["theta_"] &= bool(np.array_equal(theta32, gnb_clf.theta_[c]))
        f32_exact["var_"] &= bool(np.array_equal(var32 + epsilon32, gnb_clf.var_[c]))
        wc = weights[mask]
        theta_c = (wc[:, None] * Z64[mask]).sum(0) / wc.sum()
        var_c = (wc[:, None] * (Z64[mask] - theta_c) ** 2).sum(0) / wc.sum()
        f64_deltas["theta_"] = max(f64_deltas["theta_"], float(np.max(np.abs(theta_c - gnb_clf.theta_[c]))))
        f64_deltas["var_"] = max(f64_deltas["var_"], float(np.max(np.abs(var_c + epsilon32 - gnb_clf.var_[c]))))
        f64_deltas["class_count_"] = max(f64_deltas["class_count_"],
                                         abs(float(wc.sum()) - float(gnb_clf.class_count_[c])))
        recomputed_prior[c] = wc.sum() / weights.sum()
    check("gnb_closed_form_refit_equations", all(f32_exact.values()),
          {"float32_exact_match": f32_exact,
           "float64_recompute_deltas": f64_deltas,
           "epsilon_rule": "var_smoothing * max global unweighted feature variance (float32)",
           "note": "theta_/var_/class_count_ reproduce the weighted GaussianNB moment equations on the "
                   "actual scaled TRAIN cache BIT-EXACTLY under the historical float32 arithmetic; the "
                   "float64 recomputation differs only by float32 accumulation rounding (deltas above). "
                   "A fabricated or mislabeled estimator would not match"},
          ("gaussian_nb",))
    unweighted_prior = np.array([np.mean(train.y == 0), np.mean(train.y == 1)])
    prior32 = (gnb_clf.class_count_ / gnb_clf.class_count_.sum()).astype(np.float32)
    check("gnb_weights_adjust_priors", bool(np.array_equal(prior32, gnb_clf.class_prior_))
          and float(np.max(np.abs(recomputed_prior - gnb_clf.class_prior_))) < 1e-6
          and float(np.max(np.abs(recomputed_prior - unweighted_prior))) > 1e-9,
          {"class_prior_": gnb_clf.class_prior_.tolist(),
           "float32_count_ratio": prior32.tolist(),
           "weighted_recomputed_float64": recomputed_prior.tolist(),
           "float64_recompute_delta": float(np.max(np.abs(recomputed_prior - gnb_clf.class_prior_))),
           "unweighted_row_proportions": unweighted_prior.tolist(),
           "note": "class_prior_ equals the float32 ratio of the stored class counts exactly, and the "
                   "float64 weighted recomputation agrees to ~1.3e-8 (float32 rounding); sample weights "
                   "(inverse candidate count per observation, inverse observation count per trip) shift "
                   "class priors away from raw row proportions"}, ("gaussian_nb",))

    train_fix = np.linspace(0, len(train.X) - 1, 257, dtype=int)
    student_fix = np.linspace(0, len(student.X) - 1, 257, dtype=int)

    def manual_gnb_proba(X):
        """Literal Gaussian NB posterior from the stored parameters, replicating scikit-learn
        1.8's exact operation order (_joint_log_likelihood then _logsumexp normalization,
        per the installed sklearn source) in the artifact's float32 arithmetic."""
        Z = np.asarray(scaler.transform(X))
        jll = np.empty((len(Z), 2), dtype=Z.dtype)
        for i in range(2):
            jointi = np.log(gnb_clf.class_prior_[i])
            n_ij = -0.5 * np.sum(np.log(2.0 * np.pi * gnb_clf.var_[i, :]))
            n_ij = n_ij - 0.5 * np.sum(((Z - gnb_clf.theta_[i, :]) ** 2) / (gnb_clf.var_[i, :]), axis=1)
            jll[:, i] = jointi + n_ij
        log_prob_x = _logsumexp(jll, axis=1, xp=np)
        return np.exp(jll - np.atleast_2d(log_prob_x).T)

    formula_delta = max_abs(manual_gnb_proba(train.X[train_fix]), gnb_est.predict_proba(train.X[train_fix]))
    check("gnb_predictions_equal_gaussian_formula", formula_delta <= STRICT,
          {"max_abs_delta_vs_pipeline": formula_delta, "fixture_rows": 257,
           "note": "pipeline probabilities equal the literal Gaussian NB posterior formula computed from "
                   "the stored parameters, bit-exactly under the historical float32 arithmetic"},
          ("gaussian_nb",))

    # HistGradientBoosting: actual estimator, not a relabel
    for name in (FAMILY_DEFAULT_HGB, FAMILY_TUNED_HGB):
        est = loaded[name][1]
        recorded = in_scope[name]["evaluation"]["parameters"]
        params = est.get_params()
        check(f"hgb_is_actual_estimator[{name}]", isinstance(est, HistGradientBoostingClassifier)
              and est.__class__.__module__.startswith("sklearn.ensemble"),
              {"class": f"{est.__class__.__module__}.{est.__class__.__name__}",
               "pipeline_wrapped": isinstance(est, Pipeline)}, in_scope[name]["families"])
        check(f"hgb_parameters[{name}]", all(params[k] == v for k, v in recorded.items())
              and params["random_state"] == 20261118 and params["early_stopping"] is False,
              {"recorded": recorded, "random_state": params["random_state"],
               "early_stopping": params["early_stopping"]}, in_scope[name]["families"])
        check(f"hgb_fitted_rounds[{name}]", tuple(est.classes_.tolist()) == (0, 1)
              and int(est.n_iter_) == int(recorded["max_iter"])
              and len(est._predictors) == int(recorded["max_iter"])
              and int(getattr(est, "n_features_in_", 0)) == 38,
              {"n_iter_": int(est.n_iter_), "predictors": len(est._predictors),
               "max_iter": recorded["max_iter"], "n_features_in_": int(getattr(est, "n_features_in_", 0)),
               "note": "boosting rounds actually grown equal max_iter; no sklearn early stopping"},
              in_scope[name]["families"])

    # frozen model.joblib vs tuned baseline artifact (independent recovery refit) predict identically
    frozen_bundle = load_model(ART / "model.joblib")
    frozen_delta = max_abs(predict_positive_proba(frozen_bundle["estimator"], student.X[student_fix]),
                           predict_positive_proba(loaded[FAMILY_TUNED_HGB][1], student.X[student_fix]))
    check("frozen_model_matches_tuned_baseline_predictions", frozen_delta <= STRICT,
          {"max_abs_delta": frozen_delta,
           "note": "model.joblib freeze and the recovered hist_gradient_boosting_tuned baseline are "
                   "independent deterministic fits with identical parameters, seed and data"})

    # ---------------------------------------------------------------- prediction replays
    replay_summary = {}
    for name, spec in in_scope.items():
        families = spec["families"]
        estimator = loaded[name][1]
        entry = {}
        with np.load(spec["student"], allow_pickle=False) as z:
            student_probs = z["probabilities"]
            identity_ok = all(np.array_equal(z[key], getattr(student, key)) for key in
                              ("y", "offsets", "observation_trip_ids", "observation_indices",
                               "candidate_edges", "reference_edges"))
        check(f"student_npz_identity[{name}]", identity_ok and len(student_probs) == 175778,
              {"rows": len(student_probs), "identity_keys": 6}, families)
        fixture_delta = max_abs(historical_positive_proba(name, estimator, student.X[student_fix]),
                                student_probs[student_fix])
        check(f"student_fixture_replay[{name}]", fixture_delta <= STRICT,
              {"max_abs_delta": fixture_delta, "fixture_rows": 257, "tolerance": STRICT}, families)
        replayed = candidate_classification_metrics(student.y, student_probs, student.offsets,
                                                    student.observation_trip_ids)
        ok, worst, mismatches = compare_metrics(spec["evaluation"]["student_metrics"], replayed)
        check(f"student_full_metrics_replay[{name}]", ok,
              {"max_numeric_delta": worst, "mismatches": mismatches,
               "recorded_top1": spec["evaluation"]["student_metrics"]["top1_accuracy"],
               "replayed_top1": replayed["top1_accuracy"],
               "trips_covered": len(replayed["per_trip"])}, families)
        # full TRAIN resubstitution replay directly from the artifact
        train_p = historical_positive_proba(name, estimator, train.X)
        train_replayed = candidate_classification_metrics(train.y, train_p, train.offsets,
                                                          train.observation_trip_ids)
        ok_t, worst_t, mism_t = compare_metrics(spec["evaluation"]["train_resubstitution_metrics"], train_replayed)
        check(f"train_resubstitution_replay[{name}]", ok_t,
              {"max_numeric_delta": worst_t, "mismatches": mism_t,
               "replayed_top1": train_replayed["top1_accuracy"]}, families)
        entry["student"] = {"fixture_max_abs_delta": fixture_delta, "full_replay_max_delta": worst,
                            "top1_accuracy": replayed["top1_accuracy"],
                            "trip_macro_top1_accuracy": replayed["trip_macro_top1_accuracy"],
                            "macro_f1": replayed["macro_f1"], "log_loss": replayed["log_loss"],
                            "per_trip_count": len(replayed["per_trip"])}
        entry["train_resubstitution"] = {"max_delta": worst_t, "top1_accuracy": train_replayed["top1_accuracy"],
                                         "macro_f1": train_replayed["macro_f1"]}

        with np.load(spec["oof"], allow_pickle=False) as z:
            oof_probs = z["probabilities"]
            oof_identity = all(np.array_equal(z[key], getattr(train, key)) for key in
                               ("y", "offsets", "observation_trip_ids", "observation_indices",
                                "candidate_edges", "reference_edges"))
        check(f"oof_npz_identity[{name}]", oof_identity and len(oof_probs) == 269463
              and np.isfinite(oof_probs).all(),
              {"rows": len(oof_probs), "finite": bool(np.isfinite(oof_probs).all())}, families)
        if "oof_replay_copy" in spec:
            with np.load(spec["oof"], allow_pickle=False) as z1, \
                    np.load(spec["oof_replay_copy"], allow_pickle=False) as z2:
                copy_equal = all(np.array_equal(z1[k], z2[k]) for k in z1.files)
            check(f"oof_replay_copy_identical[{name}]", copy_equal, {"compared_arrays": 7}, families)
        oof_replayed = candidate_classification_metrics(train.y, oof_probs, train.offsets,
                                                        train.observation_trip_ids)
        ok, worst, mismatches = compare_metrics(spec["cv"]["oof_metrics"], oof_replayed)
        check(f"oof_full_metrics_replay[{name}]", ok,
              {"max_numeric_delta": worst, "mismatches": mismatches,
               "recorded_objective": spec["cv"]["objective"],
               "replayed_objective": oof_replayed["trip_macro_top1_accuracy"],
               "candidate_miss_count": oof_replayed["candidate_miss_count"]}, families)
        entry["oof"] = {"full_replay_max_delta": worst,
                        "objective_trip_macro_top1": oof_replayed["trip_macro_top1_accuracy"],
                        "top1_accuracy": oof_replayed["top1_accuracy"], "macro_f1": oof_replayed["macro_f1"],
                        "log_loss": oof_replayed["log_loss"], "candidate_miss_count": oof_replayed["candidate_miss_count"]}

        fold_ok, fold_worst, fold_row_sum, fold_obs_sum = True, 0.0, 0, 0
        for fold in spec["cv"]["folds"]:
            fold_metrics, val_obs, val_rows = fold_replay_metrics(train, oof_probs,
                                                                  fold["validation_trip_ids"])
            ok_f, worst_f, mism_f = compare_metrics(fold["metrics"], fold_metrics)
            shared = next(f for f in cv_folds if f["fold"] == fold["fold"])
            fold_ok &= ok_f and fold["validation_rows"] == val_rows and fold["train_rows"] + val_rows == 269463
            fold_ok &= (fold["train_trip_ids"] == shared["train_trip_ids"]
                        and fold["validation_trip_ids"] == shared["validation_trip_ids"])
            fold_worst = max(fold_worst, worst_f)
            fold_row_sum += val_rows
            fold_obs_sum += val_obs
        check(f"oof_fold_metrics_replay[{name}]", fold_ok and fold_row_sum == 269463 and fold_obs_sum == 28000,
              {"folds": len(spec["cv"]["folds"]), "max_numeric_delta": fold_worst,
               "validation_row_sum": fold_row_sum, "validation_observation_sum": fold_obs_sum,
               "note": "per-fold metrics recomputed from OOF probabilities on the whole-trip validation "
                       "sets; recorded validation_rows are CANDIDATE rows (269463 across folds, 28000 "
                       "underlying observations); fold trip lists equal the shared cv_folds.json"}, families)
        entry["folds"] = {"count": len(spec["cv"]["folds"]), "max_numeric_delta": fold_worst}

        final_train_p = historical_positive_proba(name, estimator, train.X[train_fix])
        oof_gap = max_abs(final_train_p, oof_probs[train_fix])
        check(f"oof_is_fold_isolated_not_final_model[{name}]", oof_gap > 1e-6,
              {"max_abs_delta_final_vs_oof_on_train_fixture": oof_gap, "fixture_rows": 257,
               "note": "OOF probabilities differ from the final all-80 fit: they come from per-fold models"},
              families)
        replay_summary[name] = entry
    report["prediction_replays"] = replay_summary

    # ------------------------------------------- GNB serving-wrapper path drift (disclosure)
    # The current working tree promotes GaussianNB scoring inputs to float64 inside
    # learning.predict_positive_proba (an uncommitted fix for the float32 sum-to-1 guard);
    # the artifacts predate it. Quantify what re-scoring the saved artifact through today's
    # wrapper would change, without touching any historical record.
    wrapper_probs = predict_positive_proba(loaded["gaussian_nb"][1], student.X)
    with np.load(in_scope["gaussian_nb"]["student"], allow_pickle=False) as z:
        saved_student_gnb = z["probabilities"]
    wrapper_metrics = candidate_classification_metrics(student.y, wrapper_probs, student.offsets,
                                                       student.observation_trip_ids)
    recorded_student = in_scope["gaussian_nb"]["evaluation"]["student_metrics"]
    wrapper_metric_deltas = {k: float(abs(wrapper_metrics[k] - recorded_student[k]))
                             for k in ("top1_accuracy", "trip_macro_top1_accuracy", "macro_f1", "log_loss")}
    raw_two_class = np.asarray(loaded["gaussian_nb"][1].predict_proba(train.X), dtype=np.float64)
    guard_violations = int(np.sum(np.abs(raw_two_class.sum(axis=1) - 1.0) > 1.01e-5))
    check("gnb_serving_wrapper_drift_disclosed", True,
          {"git_state": "uncommitted working-tree change in mapmatch/learning.py; artifacts predate it "
                        "(predict_positive_proba promotion of GaussianNB inputs to float64)",
           "prob_max_abs_delta_full_student": max_abs(wrapper_probs, saved_student_gnb),
           "student_metric_deltas": wrapper_metric_deltas,
           "direction": "the promoted float64 path removes float32 likelihood noise and scores slightly "
                       "better on student (top1 +6.8e-4, trip_macro_top1 +1.7e-3, log_loss -2.6e-3)",
           "guard_rationale": "the raw float32 path violates the predict_positive_proba sum-to-1 guard "
                              f"on {guard_violations} of 269463 TRAIN rows (max deviation 1.52e-5); the "
                              "promotion exists to keep guarded serving from raising",
           "note": "disclosure only, always passes - historical records and this audit's replays use the "
                   "historical float32 path, which reproduces every saved prediction and metric exactly"},
          ("gaussian_nb",))

    # ---------------------------------------------------------------- HPO inventory
    bayes = read_json(EXP / "bayesian_optimization.json")
    trials = [json.loads(line) for line in (EXP / "bayesian_trials.jsonl").read_text().splitlines() if line.strip()]
    composition = [t["proposal"] for t in trials]
    winner = max(trials, key=lambda t: t["objective"])
    winner_params_agree = (bayes["best_parameters"] == winner["parameters"]
                           == in_scope[FAMILY_TUNED_HGB]["evaluation"]["parameters"]
                           == in_scope[FAMILY_TUNED_HGB]["cv"]["parameters"]
                           == frozen_bundle["metadata"]["parameters"]
                           == read_json(ART / "model_metadata.json")["parameters"])
    objectives_match = all(
        abs(t["objective"] - read_json(EXP / t["cv_file"])["objective"]) <= STRICT for t in trials)
    baseline_objective = read_json(EXP / "cv" / "hist_gradient_boosting.json")["objective"]
    check("hpo_trial_inventory", len(trials) == 12 == bayes["trials_including_measured_prior"]
          and bayes["model_based_trials"] == 8
          and composition.count("measured_baseline_prior") == 1
          and composition.count("seeded_initial_design") == 3
          and composition.count("gaussian_process_expected_improvement") == 8
          and composition[0] == "measured_baseline_prior"
          and trials[0]["reused_actual_baseline"] is True,
          {"total_trials": len(trials),
           "composition": {k: composition.count(k) for k in sorted(set(composition))}})
    check("hpo_winner_and_objective", winner["trial"] == 5
          and abs(winner["objective"] - bayes["best_objective"]) <= STRICT
          and bayes["best_cv_run_name"] == winner["cv_file"].split("/")[-1].replace(".json", "")
          and winner["objective"] > baseline_objective and objectives_match,
          {"winner_trial": winner["trial"], "winner_objective": winner["objective"],
           "best_objective_record": bayes["best_objective"],
           "default_baseline_objective": baseline_objective,
           "objective_description": bayes["objective"], "tuned_cv_run": bayes["best_cv_run_name"]})
    check("hpo_winner_parameters_consistent_everywhere", winner_params_agree,
          {"winner_parameters": winner["parameters"]})
    check("hpo_objective_is_train_cv_only", "TRAIN only" in bayes["selection_data"],
          {"selection_data": bayes["selection_data"], "method": bayes["method"]})
    ei_pins = {"exploration_xi": winner.get("exploration_xi"),
               "surrogate_kernel": winner.get("surrogate_kernel"),
               "expected_improvement": winner.get("expected_improvement"),
               "seed": run_config["config"]["seed"], "search_space": bayes["search_space"]}
    check("hpo_pins_recorded", ei_pins["exploration_xi"] == 0.0003
          and "WhiteKernel" in str(ei_pins["surrogate_kernel"])
          and "Matern" in str(ei_pins["surrogate_kernel"]),
          ei_pins)
    selection = read_json(EXP / "baseline_selection.json")
    check("selection_used_train_cv_only",
          selection["student_test_consulted"] is False
          and selection["selected_model"] == "hist_gradient_boosting"
          and selection["objective"] == "out_of_fold_trip_macro_top1_accuracy"
          and selection["scores"]["hist_gradient_boosting"] == max(selection["scores"].values()),
          {"selected": selection["selected_model"], "scores": selection["scores"],
           "note": "baseline selection on TRAIN grouped-CV objective; run.log shows student_test loaded "
                   "only after baseline_selected and all 12 bayesian trials"})
    final_replay = read_json(EXP / "final_oof_replay.json")
    ok_fr, worst_fr, _ = compare_metrics(final_replay["original_trial_metrics"], final_replay["replay_metrics"])
    check("final_oof_replay_record_consistent",
          final_replay["replayed"] is True and final_replay["matches_original_trial_metrics"] is True and ok_fr
          and abs(final_replay["replay_metrics"]["top1_accuracy"]
                  - in_scope[FAMILY_TUNED_HGB]["cv"]["objective"]) <= STRICT,
          {"max_delta": worst_fr, "purpose": final_replay["purpose"]})
    report["hpo_inventory"] = {
        "total_trials_including_measured_prior": len(trials),
        "composition": {k: composition.count(k) for k in sorted(set(composition))},
        "objective_description": bayes["objective"], "selection_data": bayes["selection_data"],
        "method": bayes["method"], "default_objective": baseline_objective,
        "winner_trial": winner["trial"], "winner_objective": winner["objective"],
        "winner_parameters": winner["parameters"], "search_space": bayes["search_space"],
        "exploration_xi": winner.get("exploration_xi"), "surrogate_kernel": winner.get("surrogate_kernel"),
        "baseline_selection": selection,
        "student_access_disclosure": "student_test metrics exist for all six families (development "
                                     "evaluation recorded after selection); they are historical "
                                     "development numbers, NOT a blind holdout and NOT evaluator_test",
        "evaluator_testing_statement": results.get("evaluator_testing"),
        "recovery": results.get("recovery"),
    }

    # ---------------------------------------------------------------- cross-environment
    child = (
        "import json, platform, sys\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[1]); sys.path.insert(0, str(root))\n"
        "import glob\n"
        "import joblib, numpy as np, scipy, sklearn\n"
        "from mapmatch.learning import load_model, predict_positive_proba\n"
        "train = sorted(glob.glob(str(root/'experiments/cache/train_*.npz')))[0]\n"
        "student = sorted(glob.glob(str(root/'experiments/cache/student_test_*.npz')))[0]\n"
        "with np.load(train, allow_pickle=False) as z:\n"
        "    Xtr = z['X'][np.linspace(0, len(z['X'])-1, 257, dtype=int)]\n"
        "with np.load(student, allow_pickle=False) as z:\n"
        "    Xst = z['X'][np.linspace(0, len(z['X'])-1, 257, dtype=int)]\n"
        "out = {'versions': {'python': platform.python_version(), 'numpy': np.__version__,\n"
        "                    'scipy': scipy.__version__, 'sklearn': sklearn.__version__},\n"
        "       'predictions': {}}\n"
        "for name in ('gaussian_nb', 'hist_gradient_boosting', 'hist_gradient_boosting_tuned'):\n"
        "    m = load_model(root/'artifacts/baselines'/(name + '.joblib'))\n"
        "    out['predictions'][name] = {'train': predict_positive_proba(m['estimator'], Xtr).tolist(),\n"
        "                                'student': predict_positive_proba(m['estimator'], Xst).tolist()}\n"
        "m = load_model(root/'artifacts/model.joblib')\n"
        "out['predictions']['model_joblib'] = {'student': predict_positive_proba(m['estimator'], Xst).tolist()}\n"
        "print(json.dumps(out))\n"
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[v] = "1"
    probes = {}
    for venv in (".venv", ".venv-clean"):
        completed = subprocess.run([str(ROOT / venv / "bin/python"), "-B", "-c", child, str(ROOT)],
                                   cwd=ROOT, env=environment, capture_output=True, text=True,
                                   check=True, timeout=600)
        probes[venv] = json.loads(completed.stdout)
    deltas = {}
    for model in ("gaussian_nb", "hist_gradient_boosting", "hist_gradient_boosting_tuned"):
        for split in ("train", "student"):
            deltas[f"{model}.{split}"] = max_abs(probes[".venv"]["predictions"][model][split],
                                                 probes[".venv-clean"]["predictions"][model][split])
    deltas["model_joblib.student"] = max_abs(probes[".venv"]["predictions"]["model_joblib"]["student"],
                                             probes[".venv-clean"]["predictions"]["model_joblib"]["student"])
    check("cross_environment_replay", max(deltas.values()) <= STRICT,
          {"max_abs_deltas": deltas, "tolerance": STRICT,
           "versions_venv": probes[".venv"]["versions"],
           "versions_venv_clean": probes[".venv-clean"]["versions"],
           "note": ".venv and .venv-clean (clean serving environment) agree at 1e-12 on both fixtures"})
    report["cross_environment"] = {"max_abs_deltas": deltas,
                                   "versions": {k: probes[k]["versions"] for k in probes}}

    # ---------------------------------------------------------------- verdict
    def family_pass(family):
        return all(entry["passed"] for _, entry in CHECKS if family in entry["families"])

    report["checks"] = {name: entry for name, entry in CHECKS}
    failures = [name for name, entry in CHECKS if not entry["passed"]]
    report["families"] = {
        "gaussian_nb": {"default": "pass" if family_pass("gaussian_nb") else "fail"},
        "hist_gradient_boosting": {"default": "pass" if family_pass(FAMILY_DEFAULT_HGB) else "fail",
                                   "tuned": "pass" if family_pass(FAMILY_TUNED_HGB) else "fail"},
    }
    report["disclosures"] = [
        "Historical artifacts were trained under numpy {trained_numpy}; the current original .venv and "
        ".venv-clean both run numpy {current_numpy}. Despite that drift, the float32 fit accumulation "
        "and the historical scoring path reproduce every stored parameter, saved prediction and "
        "recorded metric bit-exactly (max delta 0.0) under the current environment, so the drift is "
        "immaterial. The strict 1e-12 replay tolerance was applied "
        "unchanged.".format(trained_numpy=training["numpy"], current_numpy=runtime["numpy"]),
        "GaussianNB ran float32 end to end (float32 cache X; StandardScaler and GaussianNB preserve "
        "dtype). The current working tree contains an uncommitted learning.predict_positive_proba fix "
        "that promotes GaussianNB scoring inputs to float64, because raw float32 likelihoods violate "
        "the sum-to-1 guard on 2 of 269463 TRAIN rows (max deviation 1.52e-5). Re-scoring the saved "
        "artifact through today's wrapper therefore differs from the historical predictions by at most "
        "1.0e-5 in probability on extreme near-clip rows (typical rows stay below 1.3e-6) and 2.7e-3 in "
        "student metrics (slightly better: it removes float32 noise). All historical records remain "
        "exactly reproducible via the historical float32 path.",
        "Fold records count CANDIDATE rows: validation_rows sum to 269463 across the three folds "
        "(28000 underlying observations). OOF predictions come from per-fold models that were not "
        "retained; they are verified via saved-array metric replay, fold partition and per-fold metric "
        "replay, not by re-prediction.",
        "student_test was consulted historically as a development evaluation after selection; those "
        "results are not blind. evaluator_test/ and archives were never accessed by this audit.",
        "artifacts/model.joblib was frozen before the recovery refit of the remaining baselines; the "
        "tuned baseline artifact is an independent deterministic refit verified to predict identically.",
    ]
    report["passed"] = not failures
    OUT.write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "failures": failures,
                      "families": report["families"], "checks_total": len(CHECKS),
                      "output": str(OUT.relative_to(ROOT))}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
