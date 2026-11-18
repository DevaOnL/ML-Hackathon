"""Fresh GNB/HistGB experiments and immutable consolidation of exactly five families.

TRAIN-only selection precedes student evaluation. No evaluator path is accessed.
Existing historical/phase-three measured files are only read, never modified.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from experiments.observation_metrics import observation_metrics
from experiments.phase_three.run_phase_three import (
    assert_prediction_identity, load_cached_dataset, load_prediction_npz,
    load_train_and_folds, prediction_arrays,
)
from mapmatch.data import discover_trips, load_graph
from mapmatch.experiments import (
    ExperimentConfig, _fit, _metrics, _sha256, cross_validate_model, save_json,
)
from mapmatch.features import FEATURE_NAMES, FEATURE_VERSION, PROXY_LABEL_POLICY
from mapmatch.final_models import FINAL_DEFAULTS, FINAL_FAMILIES, FINAL_NEW_FAMILIES, make_final_estimator
from mapmatch.learning import load_emission_model, predict_positive_proba

OUT = ROOT / "experiments/final_five/current"
ART = ROOT / "artifacts/final_five"
REPORTS = ROOT / "reports/final_five"
PHASE = ROOT / "experiments/phase_three"
DATA = ROOT / "Student folder"
BASE_ARTIFACTS = {
    "logistic_regression": ROOT / "artifacts/baselines/logistic_regression.joblib",
    "knn_distance_weighted": ROOT / "artifacts/phase_three/baselines/knn_distance_weighted.joblib",
    "bagging": ROOT / "artifacts/phase_three/baselines/bagging.joblib",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def relative(path):
    return str(Path(path).relative_to(ROOT))


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "joblib": joblib.__version__}


def refuse_existing(paths):
    for path in paths:
        if Path(path).exists() or Path(path).is_symlink():
            raise FileExistsError("refuse to overwrite measured output: " + str(path))


def compare_metrics(actual, expected, *, tolerance=1e-12):
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "log_loss", "top1_accuracy",
                "trip_macro_top1_accuracy", "candidate_miss_count", "candidate_rows", "observations"):
        if abs(actual[key] - expected[key]) > tolerance:
            raise AssertionError("saved metric replay mismatch: " + key)
    if actual["confusion_matrix"] != expected["confusion_matrix"]:
        raise AssertionError("saved confusion matrix replay mismatch")


def select_family(cvs):
    if set(cvs) != set(FINAL_FAMILIES):
        raise ValueError("selection requires exactly the five declared families")
    return max(FINAL_FAMILIES, key=lambda n: (cvs[n]["objective"], cvs[n]["oof_metrics"]["macro_f1"], -cvs[n]["oof_metrics"]["log_loss"]))


def check_folds(cv, folds):
    if len(cv["folds"]) != len(folds):
        raise AssertionError("whole-trip fold count mismatch")
    for got, ref in zip(cv["folds"], folds):
        for key in ("train_trip_ids", "validation_trip_ids"):
            if set(got[key]) != set(ref[key]):
                raise AssertionError("whole-trip fold mismatch: " + key)
        if set(got["train_trip_ids"]) & set(got["validation_trip_ids"]):
            raise AssertionError("trip leakage across CV fit/validation")


def validate_historical_hpo(selected, baseline, folds):
    """No fits and no student-data loading; validate the complete measured TRAIN study."""
    hpo_path = ROOT / "experiments/bayesian_optimization.json"
    log_path = ROOT / "experiments/bayesian_trials.jsonl"
    hpo = read_json(hpo_path)
    trials = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if selected != "hist_gradient_boosting" or hpo["model_name"] != selected:
        raise AssertionError("historical study does not match the TRAIN-selected family")
    if trials != hpo["trials"] or len(trials) != 12 or [t["trial"] for t in trials] != list(range(12)):
        raise AssertionError("historical trial log/summary mismatch")
    expected = ["measured_baseline_prior"] + ["seeded_initial_design"] * 3 + ["gaussian_process_expected_improvement"] * 8
    if [t["proposal"] for t in trials] != expected:
        raise AssertionError("historical Bayesian proposal schedule mismatch")
    if abs(baseline["objective"] - hpo["baseline_objective"]) > 1e-12 or baseline["parameters"] != trials[0]["parameters"]:
        raise AssertionError("new baseline differs from historical HPO prior")
    seen = set()
    pins = {}
    for trial in trials:
        encoded = json.dumps(trial["parameters"], sort_keys=True)
        if encoded in seen:
            raise AssertionError("duplicate historical HPO configuration")
        seen.add(encoded)
        cv_path = ROOT / "experiments" / trial["cv_file"]
        if not cv_path.resolve().is_relative_to((ROOT / "experiments/cv").resolve()):
            raise AssertionError("unsafe HPO evidence pointer")
        cv = read_json(cv_path)
        check_folds(cv, folds)
        if cv["model_name"] != selected or cv["parameters"] != trial["parameters"] or abs(cv["objective"] - trial["objective"]) > 1e-12:
            raise AssertionError("historical HPO trial evidence mismatch")
        if trial["proposal"] == "gaussian_process_expected_improvement":
            if trial["surrogate_training_points"] != trial["trial"] or trial["proposal_pool_size"] != 3072 or trial["predicted_objective_std"] <= 0 or trial["expected_improvement"] < 0:
                raise AssertionError("invalid historical GP-EI metadata")
        pins[relative(cv_path)] = _sha256(cv_path)
    winner = max(trials, key=lambda t: (t["objective"], t["cv_macro_f1"], -t["cv_log_loss"]))
    if winner["parameters"] != hpo["best_parameters"] or abs(winner["objective"] - hpo["best_objective"]) > 1e-12:
        raise AssertionError("historical HPO winner mismatch")
    result = dict(hpo)
    result.update({"phase": "five-model-completion", "reused": True, "fresh_hpo_trials": 0,
        "reuse_validation_at_utc": now(), "seeded_initial_design_trials": 3,
        "reuse_provenance": {"summary_file": relative(hpo_path), "summary_sha256": _sha256(hpo_path),
            "trial_log_file": relative(log_path), "trial_log_sha256": _sha256(log_path), "trial_cv_sha256": pins,
            "note": "Actual completed historical TRAIN-only GP-EI study; not newly executed trials. Fresh default CV reproduces the prior objective."}})
    return result


def freeze_bundle(estimator, metadata, target=ART):
    target = Path(target)
    outputs = [target / name for name in ("model.joblib", "model_metadata.json", "feature_schema.json", "model_freeze.json")]
    refuse_existing(outputs)
    target.mkdir(parents=True, exist_ok=True)
    bundle = {"estimator": estimator, "feature_names": list(FEATURE_NAMES), "feature_version": FEATURE_VERSION, "metadata": metadata}
    joblib.dump(bundle, outputs[0], compress=3)
    save_json(outputs[1], metadata)
    save_json(outputs[2], {"feature_names": list(FEATURE_NAMES), "feature_version": FEATURE_VERSION, "count": len(FEATURE_NAMES)})
    freeze = {"frozen_at_utc": now(), "phase": "five-model-completion", "model_name": metadata["model_name"],
        "run_name": metadata["run_name"], "model": relative(outputs[0]) if outputs[0].is_relative_to(ROOT) else str(outputs[0]),
        "model_sha256": _sha256(outputs[0]), "bytes": outputs[0].stat().st_size,
        "metadata_sha256": _sha256(outputs[1]), "feature_schema_sha256": _sha256(outputs[2]),
        "student_used_for_selection": False, "evaluator_used_for_selection": False}
    save_json(outputs[3], freeze)
    return freeze


def stage_train():
    outputs = [OUT / "selection.json", OUT / "bayesian_optimization.json", ART / "model_freeze.json", ART / "model.joblib", OUT / "train_complete.json"]
    outputs += [OUT / "cv" / (n + ".json") for n in (*FINAL_FAMILIES, "hist_gradient_boosting_tuned")]
    outputs += [ART / "baselines" / (n + ".joblib") for n in (*FINAL_NEW_FAMILIES, "hist_gradient_boosting_tuned")]
    outputs += list((OUT / "predictions").glob("*_oof.npz"))
    outputs += list((OUT / "training").glob("*.json"))
    refuse_existing(outputs)
    config = ExperimentConfig()
    train, folds = load_train_and_folds(config)
    cvs = {}
    for name in FINAL_FAMILIES[:3]:
        source = PHASE / "cv" / (name + ".json")
        cv = read_json(source)
        check_folds(cv, folds)
        if cv["parameters"] != FINAL_DEFAULTS[name]:
            raise AssertionError("reused baseline configuration mismatch")
        oof_path = ROOT / cv["oof_predictions_file"]
        pred = load_prediction_npz(oof_path)
        assert_prediction_identity(train, pred)
        compare_metrics(_metrics(train, pred["probabilities"]), cv["oof_metrics"])
        cvs[name] = {**cv, "phase": "five-model-completion", "reused": True,
            "reuse_provenance": {"cv_file": relative(source), "cv_sha256": _sha256(source), "oof_file": relative(oof_path), "oof_sha256": _sha256(oof_path)}}
        save_json(OUT / "cv" / (name + ".json"), cvs[name])
    for name in FINAL_NEW_FAMILIES:
        cvs[name] = cross_validate_model(name, FINAL_DEFAULTS[name], train, folds, config, OUT, name)
    selected = select_family(cvs)
    selection = {"selected_at_utc": now(), "selected_model": selected, "families": list(FINAL_FAMILIES),
        "criterion": "TRAIN out_of_fold_trip_macro_top1_accuracy; binary macro F1 then lower binary log loss as tie-breakers",
        "objectives": {n: cvs[n]["objective"] for n in FINAL_FAMILIES},
        "student_used": False, "evaluator_used": False, "historical_student_context_available": True}
    save_json(OUT / "selection.json", selection)
    hpo = validate_historical_hpo(selected, cvs[selected], folds)
    save_json(OUT / "bayesian_optimization.json", hpo)
    tuned_name = selected + "_tuned"
    tuned_cv = cross_validate_model(selected, hpo["best_parameters"], train, folds, config, OUT, tuned_name)
    if abs(tuned_cv["objective"] - hpo["best_objective"]) > 1e-12:
        raise AssertionError("fresh tuned CV does not reproduce the frozen historical winner objective")
    fresh = {}
    for name in (*FINAL_NEW_FAMILIES, tuned_name):
        family = selected if name == tuned_name else name
        parameters = hpo["best_parameters"] if name == tuned_name else FINAL_DEFAULTS[name]
        model = make_final_estimator(family, parameters, seed=config.seed, n_jobs=config.n_jobs)
        t0 = time.perf_counter()
        caught = _fit(model, train.X, train.y, train.training_weights())
        fit_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        p_train = predict_positive_proba(model, train.X)
        train_inference_seconds = time.perf_counter() - t0
        md = {"model_name": family, "run_name": name, "phase": "five-model-completion", "seed": config.seed,
            "parameters": parameters, "feature_version": FEATURE_VERSION,
            "training_trip_ids": np.unique(train.observation_trip_ids).tolist(), "training_trip_count": 80,
            "training_observations": len(train.offsets) - 1, "training_candidate_rows": len(train.y),
            "training_sampling": "350 deterministic observations per each of all 80 TRAIN trips; 38 graph/GPS-only features",
            "training_weighting": "inverse candidates per observation and inverse observations per trip, normalized to unit mean",
            "preprocessing": "TRAIN-fold-local StandardScaler" if family == "gaussian_nb" else "deterministic graph/GPS features; no learned scaling",
            "proxy_label_policy": PROXY_LABEL_POLICY, "versions": versions(),
            "metrics": {"train_resubstitution": _metrics(train, p_train), "grouped_cv": (tuned_cv if name == tuned_name else cvs[name])["oof_metrics"]},
            "timing": {"fit_all80_seconds": fit_seconds, "train_inference_seconds": train_inference_seconds},
            "selection": selection, "fit_warnings": caught,
            "candidate_config": {"candidate_radius_m": 80., "max_radius_m": 1000., "max_candidates": 12, "sample_spacing_m": 40.},
            "fit_data": "TRAIN only; student and evaluator not read by this stage", "fresh_fit": True,
            "hpo_provenance": hpo["reuse_provenance"] if name == tuned_name else None}
        art_path = ART / "baselines" / (name + ".joblib")
        art_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": model, "metadata": md, "feature_names": list(FEATURE_NAMES), "feature_version": FEATURE_VERSION}, art_path, compress=3)
        fresh[name] = {"artifact": relative(art_path), "sha256": _sha256(art_path), "metadata": md}
        save_json(OUT / "training" / (name + ".json"), fresh[name])
        print(json.dumps({"event": "fresh_all80_fit", "model": name, "seconds": fit_seconds}), flush=True)
        if name == tuned_name:
            freeze_bundle(model, md)
    save_json(OUT / "train_complete.json", {"completed_at_utc": now(), "selected_model": selected,
        "families": list(FINAL_FAMILIES), "fresh_fits": list(fresh), "config": asdict(config), "versions": versions(),
        "source_sha256": {p: _sha256(ROOT / p) for p in ("experiments/final_five/run_final_five.py", "mapmatch/final_models.py", "experiments/final_five/PROTOCOL.md", "mapmatch/learning.py")},
        "model_freeze": relative(ART / "model_freeze.json"), "student_loaded": False, "evaluator_loaded": False})


def student_predictions_path(name):
    if name in FINAL_FAMILIES[:3]:
        if name == "logistic_regression":
            return ROOT / "experiments/predictions/logistic_regression_student.npz"
        return PHASE / "predictions" / (name + "_student.npz")
    return OUT / "predictions" / (name + "_student.npz")


def stage_student():
    refuse_existing([OUT / "results.json", OUT / "model_comparison.csv", OUT / "sequence_classification.json",
        *[OUT / "evaluation" / (n + ".json") for n in (*FINAL_FAMILIES, "hist_gradient_boosting_tuned")],
        *list((OUT / "predictions").glob("*_student.npz"))])
    read_json(OUT / "train_complete.json")
    freeze = read_json(ART / "model_freeze.json")
    if _sha256(ART / "model.joblib") != freeze["model_sha256"]:
        raise AssertionError("frozen model changed before student evaluation")
    train = load_cached_dataset("train")
    student = load_cached_dataset("student_test")
    graphs = {p.name: load_graph(p) for p in discover_trips(DATA / "student_test")}
    if set(graphs) != set(np.unique(student.observation_trip_ids)):
        raise AssertionError("student graph identity mismatch")
    evaluations, sequence, comparison, acceptance = {}, {}, [], {}
    names = (*FINAL_FAMILIES, "hist_gradient_boosting_tuned")
    for name in names:
        if name in FINAL_FAMILIES[:3]:
            ev_path = PHASE / "evaluation" / (name + ".json")
            ev = read_json(ev_path)
            prediction = load_prediction_npz(student_predictions_path(name))
            assert_prediction_identity(student, prediction)
            compare_metrics(_metrics(student, prediction["probabilities"]), ev["student_metrics"])
            artifact = BASE_ARTIFACTS[name]
            ev = {**ev, "phase": "five-model-completion", "reused": True,
                "reuse_provenance": {"evaluation_file": relative(ev_path), "evaluation_sha256": _sha256(ev_path)}}
        else:
            fit = read_json(OUT / "training" / (name + ".json"))
            artifact = ROOT / fit["artifact"]
            if _sha256(artifact) != fit["sha256"]:
                raise AssertionError("fresh baseline model identity changed")
            model = load_emission_model(artifact)
            t0 = time.perf_counter()
            probabilities = predict_positive_proba(model.estimator, student.X)
            inference_seconds = time.perf_counter() - t0
            prediction = prediction_arrays(student, probabilities)
            out = student_predictions_path(name)
            refuse_existing([out])
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, **prediction)
            md = fit["metadata"]
            ev = {"run_name": name, "model_name": md["model_name"], "parameters": md["parameters"],
                "phase": "five-model-completion", "reused": False, "fresh_fit": True,
                "training_seconds": md["timing"]["fit_all80_seconds"],
                "train_inference_seconds": md["timing"]["train_inference_seconds"],
                "student_inference_seconds": inference_seconds, "warnings": md["fit_warnings"],
                "train_resubstitution_metrics": md["metrics"]["train_resubstitution"],
                "student_metrics": _metrics(student, probabilities)}
        assert_prediction_identity(student, prediction)
        model = load_emission_model(artifact)
        if set(model.metadata["training_trip_ids"]) != set(np.unique(train.observation_trip_ids)):
            raise AssertionError("saved model was not fitted on exactly all 80 TRAIN trips")
        if name == "gaussian_nb" and not (isinstance(model.estimator, Pipeline) and isinstance(model.estimator.named_steps["classifier"], GaussianNB)):
            raise AssertionError("GaussianNB artifact has wrong actual estimator class")
        if name.startswith("hist_gradient_boosting") and not isinstance(model.estimator, HistGradientBoostingClassifier):
            raise AssertionError("HistGB artifact has wrong actual estimator class")
        fixture = np.unique(np.linspace(0, len(student.X) - 1, 257).astype(int))
        got = predict_positive_proba(model.estimator, student.X[fixture])
        delta = float(np.max(np.abs(got - prediction["probabilities"][fixture])))
        if not np.allclose(got, prediction["probabilities"][fixture], atol=1e-12, rtol=0):
            raise AssertionError("actual saved-model strict fixture mismatch: " + name)
        metrics, _ = observation_metrics(prediction, graphs)
        sequence[name] = metrics
        ev.update({"model_path": relative(artifact), "model_sha256": _sha256(artifact),
            "student_predictions_file": relative(student_predictions_path(name)),
            "student_predictions_sha256": _sha256(student_predictions_path(name)),
            "fixture_max_absolute_delta": delta, "fixture_rows": len(fixture)})
        evaluations[name] = ev
        save_json(OUT / "evaluation" / (name + ".json"), ev)
        cv = read_json(OUT / "cv" / (name + ".json"))
        comparison.append({"model": name, "family": ev["model_name"], "cv_trip_macro_top1_accuracy": cv["objective"],
            "cv_macro_f1": cv["oof_metrics"]["macro_f1"], "cv_log_loss": cv["oof_metrics"]["log_loss"],
            "fit_all80_seconds": ev["training_seconds"], "student_top1_accuracy": ev["student_metrics"]["top1_accuracy"],
            "student_macro_f1": ev["student_metrics"]["macro_f1"], "student_precision": ev["student_metrics"]["precision"],
            "student_recall": ev["student_metrics"]["recall"], "student_log_loss": ev["student_metrics"]["log_loss"],
            "student_observation_edge_macro_f1": metrics["observation_edge_macro_f1"],
            "student_inference_seconds": ev["student_inference_seconds"], "reused_fit": name in FINAL_FAMILIES[:3]})
        acceptance[name] = {"passed": True, "artifact": relative(artifact), "sha256": _sha256(artifact),
            "estimator_class": type(model.estimator).__module__ + "." + type(model.estimator).__name__,
            "training_trip_count": len(model.metadata["training_trip_ids"]), "student_trip_count": len(metrics["per_trip"]),
            "student_observations": len(student.offsets) - 1, "student_rows": len(student.y),
            "strict_fixture_rows": len(fixture), "max_probability_delta": delta, "reused_fit": name in FINAL_FAMILIES[:3]}
        print(json.dumps({"event": "all_ten_student_test", "model": name, "top1": ev["student_metrics"]["top1_accuracy"], "strict_fixture_delta": delta}), flush=True)
    import csv
    with (OUT / "model_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    save_json(OUT / "sequence_classification.json", sequence)
    results = {"phase": "five-model-completion", "completed_at_utc": now(), "families": list(FINAL_FAMILIES),
        "family_count": len(FINAL_FAMILIES), "selected_model": read_json(OUT / "selection.json")["selected_model"],
        "final_run_name": "hist_gradient_boosting_tuned", "cv": {n: read_json(OUT / "cv" / (n + ".json")) for n in names},
        "evaluations": evaluations, "comparison": comparison, "bayesian": read_json(OUT / "bayesian_optimization.json"),
        "model_freeze": freeze, "versions": versions(), "student_role": "historically used student-development set; not globally blind",
        "evaluator_status": "not accessed; final decoder must freeze before evaluator work"}
    save_json(OUT / "results.json", results)
    save_json(REPORTS / "all_five_model_acceptance.json", {"checked_at_utc": now(), "passed": True,
        "families": list(FINAL_FAMILIES), "exactly_five_families": True, "all_models": acceptance,
        "fresh_fits": ["gaussian_nb", "hist_gradient_boosting", "hist_gradient_boosting_tuned"],
        "all80_train_trips": True, "all10_student_trips": True, "student_used_for_selection": False,
        "evaluator_accessed": False, "selected_model_sha256": freeze["model_sha256"],
        "results_sha256": _sha256(OUT / "results.json")})


def stage_clean():
    out = ART / "clean_load_report.json"
    refuse_existing([out])
    model = load_emission_model(ART / "model.joblib")
    train = load_cached_dataset("train")
    X = np.ascontiguousarray(train.X[:2000])
    expected = predict_positive_proba(model.estimator, X)
    script = """import json,hashlib,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import numpy as np
from experiments.final_five.run_final_five import versions
from experiments.phase_three.run_phase_three import load_cached_dataset
from mapmatch.learning import load_emission_model,predict_positive_proba
m=load_emission_model(Path(sys.argv[2]))
X=np.ascontiguousarray(load_cached_dataset('train').X[:2000])
print(json.dumps({'versions':versions(),'fixture_sha256':hashlib.sha256(X.tobytes()).hexdigest(),'probabilities':predict_positive_proba(m.estimator,X).tolist()}))
"""
    p = subprocess.run([str(ROOT / ".venv-clean/bin/python"), "-B", "-c", script, str(ROOT), str(ART / "model.joblib")], capture_output=True, text=True, timeout=180)
    if p.returncode:
        raise AssertionError("clean-child model load failed: " + p.stderr[-2000:])
    child = json.loads(p.stdout)
    got = np.asarray(child["probabilities"])
    if child["versions"] != versions() or child["fixture_sha256"] != hashlib.sha256(X.tobytes()).hexdigest():
        raise AssertionError("clean-child versions or TRAIN fixture identity mismatch")
    if got.shape != expected.shape or not np.allclose(got, expected, atol=1e-12, rtol=0):
        raise AssertionError("clean-child strict probability mismatch")
    report = {"checked_at_utc": now(), "passed": True, "versions": child["versions"], "parent_versions": versions(),
        "fixture": "first 2000 TRAIN candidate rows; labels not used by prediction", "fixture_sha256": child["fixture_sha256"],
        "max_probability_delta": float(np.max(np.abs(got - expected))), "atol": 1e-12, "rtol": 0,
        "model_sha256": _sha256(ART / "model.joblib")}
    save_json(out, report)
    print(json.dumps(report), flush=True)


def isolated_run_paths(run_id):
    if not isinstance(run_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", run_id) is None:
        raise ValueError("run-id must be 1-48 lowercase letters, digits, underscores or hyphens")
    values = []
    for category in ("experiments", "artifacts", "reports"):
        parent = ROOT / category / "final_five_runs"
        destination = parent / run_id
        if parent.resolve() != ROOT / category / "final_five_runs" or destination.resolve() != parent / run_id:
            raise ValueError("run destination must not traverse symlinks")
        values.append(destination)
    return tuple(values)


def main(argv=None):
    global OUT, ART, REPORTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "student", "clean"), required=True)
    parser.add_argument("--run-id", help="explicit isolated experiment, artifact and report paths; existing outputs still refused")
    args = parser.parse_args(argv)
    previous = OUT, ART, REPORTS
    try:
        if args.run_id:
            OUT, ART, REPORTS = isolated_run_paths(args.run_id)
        with threadpool_limits(limits=1):
            {"train": stage_train, "student": stage_student, "clean": stage_clean}[args.stage]()
    finally:
        OUT, ART, REPORTS = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
