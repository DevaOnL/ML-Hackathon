#!/usr/bin/env python3
"""Verify actual trained LR/KNN/Bagging artifacts and all-ten classifier tests.

No fitting, decoder, HTTP, screenshots, or report prerequisites. Replays saved
probabilities against cache identities and the actual saved estimators on a small
deterministic fixture. Raw caches and per-observation labels remain local.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mapmatch.experiments import ExperimentConfig
from mapmatch.learning import candidate_classification_metrics, load_emission_model, predict_positive_proba
from mapmatch.phase_models import PHASE_FAMILIES, KNN_REFERENCE_BANK_BUDGET, KNN_FINAL_FIT_INDEX, bank_seed, knn_reference_bank
from experiments.phase_three.run_phase_three import load_cached_dataset
from experiments.observation_metrics import observation_metrics

PHASE = ROOT / "experiments/phase_three"
ARTIFACTS = ROOT / "artifacts/phase_three"


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def prediction_file(name, evaluation):
    if name == "logistic_regression":
        return ROOT / "experiments/predictions/logistic_regression_student.npz"
    return ROOT / evaluation["student_predictions_file"]


def model_file(name, selected, parameters, best):
    if name.endswith("_tuned") or (name == selected and parameters == best):
        return ARTIFACTS / "model.joblib"
    if name == "logistic_regression":
        return ROOT / "artifacts/baselines/logistic_regression.joblib"
    return ARTIFACTS / "baselines" / (name + ".joblib")


def cross_environment_probe(model):
    """Both interpreters actually load and predict, reporting their own versions."""
    child = """
import hashlib,json,platform,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
import joblib,numpy as np,scipy,sklearn
from mapmatch.features import FEATURE_NAMES
from mapmatch.learning import load_emission_model,predict_positive_proba
with np.load(root/'experiments/cache/train_edee9a8f151dd25e.npz',allow_pickle=False) as z:
 X=z['X'][np.linspace(0,len(z['X'])-1,257,dtype=int)]
m=load_emission_model(Path(sys.argv[2]));p=predict_positive_proba(m.estimator,X)
print(json.dumps({'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__},'fixture_sha256':hashlib.sha256(X.tobytes()).hexdigest(),'feature_count':len(FEATURE_NAMES),'probabilities':p.tolist()}))
"""
    environment = dict(os.environ)
    environment.update({k:"1" for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")})
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    probes = {}
    for name in (".venv", ".venv-clean"):
        completed = subprocess.run([str(ROOT/name/"bin/python"), "-B", "-c", child, str(ROOT), str(model)],
                                   cwd=ROOT, env=environment, capture_output=True, text=True, check=True, timeout=90)
        probes[name] = json.loads(completed.stdout)
    a, b = probes[".venv"], probes[".venv-clean"]
    require(a["fixture_sha256"] == b["fixture_sha256"], "cross-env feature fixture identity")
    require(a["versions"] == b["versions"], "cross-env actual runtime versions")
    delta = float(np.max(np.abs(np.asarray(a["probabilities"])-np.asarray(b["probabilities"]))))
    require(delta <= 1e-12, "cross-env exact deterministic probabilities")
    return {"passed":True,"fixture_rows":257,"fixture":"deterministic linspace TRAIN candidate rows; no student data",
            "fixture_sha256":a["fixture_sha256"], "max_abs_probability_delta":delta,
            "interpreters":{name:{"versions":probe["versions"],"feature_count":probe["feature_count"]}
                            for name,probe in probes.items()}}


def verify():
    results = read_json(PHASE / "results.json")
    frozen = read_json(ARTIFACTS / "model_freeze.json")
    selection = read_json(PHASE / "selection.json")
    hpo = read_json(PHASE / "bayesian_optimization.json")
    train = load_cached_dataset("train")
    student = load_cached_dataset("student_test")
    train_trips = set(np.unique(train.observation_trip_ids))
    student_trips = set(np.unique(student.observation_trip_ids))
    require(len(train_trips) == 80 and len(student_trips) == 10 and not train_trips & student_trips, "TRAIN/student trip boundaries")
    require(sha256(ARTIFACTS / "model.joblib") == frozen["sha256"], "frozen model bytes")
    selected = selection["selected_model"]
    require(selected in PHASE_FAMILIES and selection["student_test_consulted"] is False, "TRAIN-only three-family selection")
    expected_names = set(PHASE_FAMILIES) | {selected + "_tuned"}
    require(set(results["evaluations"]) == expected_names, "exactly three families plus tuned comparison")
    fixture = np.linspace(0, len(student.X)-1, 257, dtype=int)
    identity_keys = ("y", "offsets", "observation_trip_ids", "observation_indices", "candidate_edges", "reference_edges")
    model_rows = []
    for name, evaluation in results["evaluations"].items():
        family = name.removesuffix("_tuned")
        path = model_file(name, selected, evaluation["parameters"], hpo["best_parameters"])
        wrapper = load_emission_model(path)
        estimator = wrapper.estimator
        classifier = estimator[-1] if isinstance(estimator, Pipeline) else estimator
        require(set(wrapper.metadata["training_trip_ids"]) == train_trips, name + " all80 TRAIN fit")
        if family == "logistic_regression":
            require(isinstance(classifier, LogisticRegression) and isinstance(estimator, Pipeline), "actual LR implementation")
            require(int(estimator[0].n_samples_seen_) == len(train.X), "LR scaler TRAIN rows")
        elif family == "knn_distance_weighted":
            require(isinstance(classifier, KNeighborsClassifier) and isinstance(estimator, Pipeline), "actual KNN implementation")
            require(classifier.weights == "distance" and classifier.n_samples_fit_ == KNN_REFERENCE_BANK_BUDGET, "genuine bounded distance KNN")
            require(int(estimator[0].n_samples_seen_) == KNN_REFERENCE_BANK_BUDGET, "KNN scaler TRAIN bank rows")
            bank = knn_reference_bank(train.y, train.training_weights(), KNN_REFERENCE_BANK_BUDGET,
                                      bank_seed(ExperimentConfig().seed, KNN_FINAL_FIT_INDEX))
            require(np.allclose(classifier._fit_X, estimator[0].transform(train.X[bank]), rtol=1e-6, atol=1e-6), "KNN references are exact predeclared all80 TRAIN bank")
            require(np.allclose(estimator[0].mean_, train.X[bank].mean(axis=0, dtype=np.float64), rtol=1e-7, atol=1e-7), "KNN scaler fitted only on TRAIN bank")
        elif family == "bagging":
            require(isinstance(classifier, BaggingClassifier) and classifier.bootstrap and isinstance(classifier.estimator, DecisionTreeClassifier), "actual bootstrap Bagging implementation")
            require(len(classifier.estimators_) == evaluation["parameters"]["n_estimators"], "fitted Bagging tree count")
            require(classifier._n_samples == len(train.X), "Bagging fitted on all TRAIN rows")
        prediction_path = prediction_file(name, evaluation)
        with np.load(prediction_path, allow_pickle=False) as stored:
            prediction = {key: stored[key] for key in stored.files}
        for key in identity_keys:
            require(np.array_equal(prediction[key], getattr(student, key)), name + " student identity " + key)
        p = prediction["probabilities"]
        require(np.isfinite(p).all() and np.all((p >= 1e-7) & (p <= 1-1e-7)), name + " probability domain")
        actual = predict_positive_proba(estimator, student.X[fixture])
        delta = float(np.max(np.abs(actual-p[fixture])))
        require(delta <= 1e-12, name + " saved probabilities differ from artifact fixture")
        metric = candidate_classification_metrics(student.y, p, student.offsets, student.observation_trip_ids)
        for key in ("accuracy", "macro_f1", "log_loss", "top1_accuracy", "trip_macro_top1_accuracy", "candidate_miss_rate"):
            require(abs(metric[key]-evaluation["student_metrics"][key]) <= 1e-12, name + " recorded metric " + key)
        observation, _ = observation_metrics(prediction)
        model_rows.append({"model": name, "family": family, "artifact": str(path.relative_to(ROOT)),
                           "artifact_sha256": sha256(path), "actual_classifier": type(classifier).__name__,
                           "training_trip_count": 80, "tested_trip_count": 10, "observations": len(student.offsets)-1,
                           "candidate_rows": len(student.y), "fixture_probability_max_abs_delta": delta,
                           "top1_accuracy": metric["top1_accuracy"], "trip_macro_top1_accuracy": metric["trip_macro_top1_accuracy"],
                           "binary_candidate_macro_f1": metric["macro_f1"], "binary_log_loss": metric["log_loss"],
                           "observation_edge_macro_f1": observation["observation_edge_macro_f1"],
                           "categorical_emission_nll": observation["observation_categorical_log_loss_clipped"],
                           "candidate_miss_count": metric["candidate_miss_count"]})
    return {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
            "scope": "actual implementations, trained artifacts and all-ten classifier results only; no screenshot/report dependency",
            "metric_target": "route-derived observation proxy labels, not supplied point-aligned ground truth or reserved evaluator results",
            "selected_family": selected, "frozen_model_sha256": frozen["sha256"], "models": model_rows,
            "probability_replay_runtime": {"executable": str(Path(sys.executable).relative_to(ROOT)) if Path(sys.executable).is_relative_to(ROOT) else Path(sys.executable).name,
                                           "numpy": np.__version__, "tolerance_absolute": 1e-12},
            "knn_cross_build_diagnostic": "reports/phase_three/knn_runtime_diagnostic.json",
            "selected_model_clean_environment": cross_environment_probe(ARTIFACTS / "model.joblib")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase_three/model_acceptance.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refuse to overwrite verified model acceptance")
    result = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
