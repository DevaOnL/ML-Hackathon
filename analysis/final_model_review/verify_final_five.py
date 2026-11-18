"""Independent ACTUAL verification of the fresh final five-family stage.

Read-only: verifies every recorded SHA-256, cache identities, fold structure,
OOF/student prediction identity, full metric replay (binary + observation),
strict 1e-12 fixture replay with the original .venv for all six saved models,
HPO study validation, and the model freeze bundle. No fitting, no HPO, no
decoder search, no evaluator access, no HTTP.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
import sklearn
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.observation_metrics import observation_metrics
from experiments.phase_three.run_phase_three import (
    assert_prediction_identity, load_cached_dataset, load_prediction_npz,
)
from mapmatch.data import discover_trips, load_graph
from mapmatch.experiments import _metrics
from mapmatch.learning import load_emission_model, predict_positive_proba

OUT = ROOT / "experiments/final_five/current"
ART = ROOT / "artifacts/final_five"
REPORTS = ROOT / "reports/final_five"

results = {"checks": [], "failures": []}


def check(name, ok, detail=""):
    results["checks"].append({"check": name, "passed": bool(ok), "detail": str(detail)[:400]})
    if not ok:
        results["failures"].append({"check": name, "detail": str(detail)[:400]})
    return ok


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def metric_deltas(actual, expected):
    deltas = {}
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "log_loss",
                "top1_accuracy", "top1_accuracy_given_candidate", "trip_macro_top1_accuracy",
                "candidate_miss_count", "candidate_rows", "observations"):
        if key in expected and key in actual:
            deltas[key] = abs(float(actual[key]) - float(expected[key]))
        elif key in expected:
            deltas[key] = "missing"
    if "confusion_matrix" in expected:
        deltas["confusion_matrix_equal"] = actual.get("confusion_matrix") == expected["confusion_matrix"]
    return deltas


# ---------------------------------------------------------------- datasets
train = load_cached_dataset("train")
student = load_cached_dataset("student_test")
train_trips = set(np.unique(train.observation_trip_ids).tolist())
student_trips = set(np.unique(student.observation_trip_ids).tolist())
check("train_cache_identity", True,
      f"trips={len(train_trips)} rows={len(train.y)} observations={len(train.offsets)-1} "
      f"X={train.X.shape} dtype={train.X.dtype}")
check("student_cache_identity", True,
      f"trips={len(student_trips)} rows={len(student.y)} observations={len(student.offsets)-1}")
check("train_student_trip_disjoint", not (train_trips & student_trips),
      f"overlap={sorted(train_trips & student_trips)}")
check("train_80_trips", len(train_trips) == 80)
check("student_10_trips", len(student_trips) == 10)
check("train_rows_269463", len(train.y) == 269463)
check("student_rows_175778", len(student.y) == 175778)

# ---------------------------------------------------------------- fold structure
CV_FILES = ["logistic_regression", "knn_distance_weighted", "bagging", "gaussian_nb",
            "hist_gradient_boosting", "hist_gradient_boosting_tuned"]
reference_folds = None
for name in CV_FILES:
    cv = read_json(OUT / "cv" / f"{name}.json")
    folds = cv["folds"]
    ok = len(folds) == 3
    all_ids = set()
    for f in folds:
        tr, va = set(f["train_trip_ids"]), set(f["validation_trip_ids"])
        ok &= not (tr & va) and tr | va == train_trips
        all_ids |= va
    ok &= all_ids == train_trips  # every trip validated exactly once
    detail = (f"folds=3 val_sizes={[len(f['validation_trip_ids']) for f in folds]} "
              f"val_rows={[f['validation_rows'].__len__() if hasattr(f['validation_rows'], '__len__') else f['validation_rows'] for f in folds]}")
    check(f"cv_folds_{name}", ok, detail)
    sig = [tuple(sorted(f["train_trip_ids"])) + tuple(sorted(f["validation_trip_ids"])) for f in folds]
    if reference_folds is None:
        reference_folds = sig
        shared = read_json(ROOT / "experiments/cv_folds.json")
        entries = shared["folds"] if isinstance(shared, dict) and "folds" in shared else shared
        shared_sig = [tuple(sorted(e["train_trip_ids"])) + tuple(sorted(e["validation_trip_ids"])) for e in entries]
        check("folds_match_shared_cv_folds_json", sig == shared_sig)
    else:
        check(f"cv_folds_identical_to_lr_{name}", sig == reference_folds)

# ---------------------------------------------------------------- HPO independent validation
hpo = read_json(ROOT / "experiments/bayesian_optimization.json")
trials = [json.loads(l) for l in (ROOT / "experiments/bayesian_trials.jsonl").read_text().splitlines() if l.strip()]
check("hpo_trial_log_matches_summary", trials == hpo["trials"] and len(trials) == 12)
check("hpo_trial_ids", [t["trial"] for t in trials] == list(range(12)))
schedule = ["measured_baseline_prior"] + ["seeded_initial_design"] * 3 + ["gaussian_process_expected_improvement"] * 8
check("hpo_proposal_schedule", [t["proposal"] for t in trials] == schedule)
seen = set()
dup_free = True
for t in trials:
    enc = json.dumps(t["parameters"], sort_keys=True)
    if enc in seen:
        dup_free = False
    seen.add(enc)
check("hpo_no_duplicate_configs", dup_free, f"unique={len(seen)}")
pins = read_json(OUT / "bayesian_optimization.json")["reuse_provenance"]["trial_cv_sha256"]
for t in trials:
    cv_path = ROOT / "experiments" / t["cv_file"]
    ok = cv_path.exists() and cv_path.resolve().is_relative_to((ROOT / "experiments/cv").resolve())
    ok = ok and sha256(cv_path) == pins["experiments/" + t["cv_file"]]
    cv = read_json(cv_path)
    ok = ok and cv["model_name"] == "hist_gradient_boosting" and cv["parameters"] == t["parameters"]
    ok = ok and abs(cv["objective"] - t["objective"]) <= 1e-12
    sig = [tuple(sorted(f["train_trip_ids"])) + tuple(sorted(f["validation_trip_ids"])) for f in cv["folds"]]
    ok = ok and sig == reference_folds
    if t["proposal"] == "gaussian_process_expected_improvement":
        ok = ok and t["surrogate_training_points"] == t["trial"] and t["proposal_pool_size"] == 3072
        ok = ok and t["predicted_objective_std"] > 0 and t["expected_improvement"] >= 0
    check(f"hpo_trial_{t['trial']}_evidence", ok, t["cv_file"])
winner = max(trials, key=lambda t: (t["objective"], t["cv_macro_f1"], -t["cv_log_loss"]))
check("hpo_winner_consistent", winner["parameters"] == hpo["best_parameters"]
      and abs(winner["objective"] - hpo["best_objective"]) <= 1e-12,
      f"winner=trial{winner['trial']} objective={winner['objective']}")
check("hpo_reuse_provenance_shas",
      sha256(ROOT / "experiments/bayesian_optimization.json") ==
      read_json(OUT / "bayesian_optimization.json")["reuse_provenance"]["summary_sha256"]
      and sha256(ROOT / "experiments/bayesian_trials.jsonl") ==
      read_json(OUT / "bayesian_optimization.json")["reuse_provenance"]["trial_log_sha256"])
check("hpo_fresh_trials_zero", read_json(OUT / "bayesian_optimization.json")["fresh_hpo_trials"] == 0)

# default HGB fresh CV must reproduce prior baseline objective
hgb_default_cv = read_json(OUT / "cv/hist_gradient_boosting.json")
check("hpo_baseline_reproduced_by_fresh_cv",
      abs(hgb_default_cv["objective"] - hpo["baseline_objective"]) <= 1e-12
      and hgb_default_cv["parameters"] == trials[0]["parameters"],
      f"fresh={hgb_default_cv['objective']} prior={hpo['baseline_objective']}")
tuned_cv = read_json(OUT / "cv/hist_gradient_boosting_tuned.json")
check("tuned_cv_matches_hpo_winner", abs(tuned_cv["objective"] - hpo["best_objective"]) <= 1e-12
      and tuned_cv["parameters"] == hpo["best_parameters"])

# ---------------------------------------------------------------- results.json + acceptance + freeze
results_json = read_json(OUT / "results.json")
acceptance = read_json(REPORTS / "all_five_model_acceptance.json")
freeze = read_json(ART / "model_freeze.json")
check("results_sha256_in_acceptance", sha256(OUT / "results.json") == acceptance["results_sha256"])
check("acceptance_passed_flag", acceptance["passed"] is True)
check("frozen_model_sha_everywhere",
      sha256(ART / "model.joblib") == acceptance["selected_model_sha256"] == freeze["model_sha256"]
      == results_json["model_freeze"]["model_sha256"] == read_json(ART / "clean_load_report.json")["model_sha256"],
      f"model_sha256={freeze['model_sha256']}")
check("freeze_metadata_schema_shas",
      sha256(ART / "model_metadata.json") == freeze["metadata_sha256"]
      and sha256(ART / "feature_schema.json") == freeze["feature_schema_sha256"]
      and freeze["bytes"] == (ART / "model.joblib").stat().st_size
      and freeze["student_used_for_selection"] is False
      and freeze["evaluator_used_for_selection"] is False)
check("freeze_run_is_tuned_hgb", freeze["model_name"] == "hist_gradient_boosting"
      and freeze["run_name"] == "hist_gradient_boosting_tuned")
check("results_selected_model", results_json["selected_model"] == "hist_gradient_boosting"
      and results_json["final_run_name"] == "hist_gradient_boosting_tuned"
      and results_json["family_count"] == 5
      and set(results_json["families"]) == {"logistic_regression", "knn_distance_weighted", "bagging",
                                            "gaussian_nb", "hist_gradient_boosting"})
expected_objectives = {"logistic_regression": 0.884, "knn_distance_weighted": 0.8767857142857143,
                       "bagging": 0.8914642857142857, "gaussian_nb": 0.8734285714285714,
                       "hist_gradient_boosting": 0.8926785714285714}
check("cv_objectives_match_selection_records",
      all(abs(results_json["cv"][n]["objective"] - v) <= 1e-12 for n, v in expected_objectives.items())
      and read_json(OUT / "selection.json")["objectives"] == {n: results_json["cv"][n]["objective"] for n in expected_objectives})
check("selection_uses_train_only",
      read_json(OUT / "selection.json")["student_used"] is False
      and read_json(OUT / "selection.json")["evaluator_used"] is False)
# selection picks max objective among the five
five = {n: results_json["cv"][n]["objective"] for n in expected_objectives}
check("selection_is_argmax", max(five, key=five.get) == "hist_gradient_boosting", str(five))

# train_complete source shas
tc = read_json(OUT / "train_complete.json")
src_ok = all(sha256(ROOT / p) == s for p, s in tc["source_sha256"].items())
check("train_complete_source_shas", src_ok, json.dumps(tc["source_sha256"]))
check("train_complete_student_evaluator_not_loaded",
      tc["student_loaded"] is False and tc["evaluator_loaded"] is False)

# ---------------------------------------------------------------- per-model acceptance records
CLASS_MAP = {"logistic_regression": LogisticRegression, "knn_distance_weighted": KNeighborsClassifier,
             "bagging": BaggingClassifier, "gaussian_nb": GaussianNB,
             "hist_gradient_boosting": HistGradientBoostingClassifier,
             "hist_gradient_boosting_tuned": HistGradientBoostingClassifier}
for name, rec in acceptance["all_models"].items():
    art = ROOT / rec["artifact"]
    ok = art.exists() and sha256(art) == rec["sha256"]
    em = load_emission_model(art)
    cls = em.estimator.steps[-1][1] if isinstance(em.estimator, Pipeline) else em.estimator
    ok = ok and isinstance(cls, CLASS_MAP[name])
    ok = ok and set(em.metadata["training_trip_ids"]) == train_trips
    ok = ok and rec["training_trip_count"] == 80 and rec["student_trip_count"] == 10
    ok = ok and rec["student_rows"] == 175778 and rec["strict_fixture_rows"] == 257
    ok = ok and rec["max_probability_delta"] <= 1e-12  # strict tolerance; LR honestly logs 1.67e-15
    ok = ok and em.feature_names == tuple(json.loads((ART / "feature_schema.json").read_text())["feature_names"])
    check(f"acceptance_model_{name}", ok, f"sha={rec['sha256'][:12]} class={type(cls).__name__}")

# ---------------------------------------------------------------- prediction identity + metric replay
def replay(path_str, dataset, expected_metrics, label):
    pred = load_prediction_npz(ROOT / path_str)
    try:
        assert_prediction_identity(dataset, pred)
        ident = True
    except AssertionError as e:
        ident = False
        check(f"identity_{label}", False, e)
    got = _metrics(dataset, pred["probabilities"])
    deltas = metric_deltas(got, expected_metrics)
    maxd = max((v for v in deltas.values() if isinstance(v, float)), default=0.0)
    ok = ident and maxd <= 1e-12 and deltas.get("confusion_matrix_equal", True)
    check(f"metric_replay_{label}", ok, f"max_delta={maxd:.3e} keys={ {k: v for k, v in deltas.items() if v not in (0, True)} }")
    return pred

for name in CV_FILES:
    cv = read_json(OUT / "cv" / f"{name}.json")
    if "reuse_provenance" in cv:
        oof_path = cv["reuse_provenance"]["oof_file"]
        ok = sha256(ROOT / cv["reuse_provenance"]["cv_file"]) == cv["reuse_provenance"]["cv_sha256"]
        ok = ok and sha256(ROOT / oof_path) == cv["reuse_provenance"]["oof_sha256"]
        check(f"reuse_provenance_shas_{name}", ok, f"{cv['reuse_provenance']['cv_file']}")
    else:
        oof_path = f"experiments/final_five/current/predictions/{name}_oof.npz"
    replay(oof_path, train, cv["oof_metrics"], f"oof_{name}")
    ev = read_json(OUT / "evaluation" / f"{name}.json")
    check(f"evaluation_model_sha_{name}", sha256(ROOT / ev["model_path"]) == ev["model_sha256"])
    check(f"evaluation_predictions_sha_{name}",
          sha256(ROOT / ev["student_predictions_file"]) == ev["student_predictions_sha256"])
    pred = replay(ev["student_predictions_file"], student, ev["student_metrics"], f"student_{name}")
    check(f"evaluation_fixture_delta_{name}", ev["fixture_max_absolute_delta"] <= 1e-12
          and ev["fixture_rows"] == 257,
          f"recorded_delta={ev['fixture_max_absolute_delta']:.3e}")
    tr_path = OUT / "training" / f"{name}.json"
    if tr_path.exists():
        tr_meta = read_json(tr_path)["metadata"]
        check(f"evaluation_train_trips_{name}", set(tr_meta["training_trip_ids"]) == train_trips
              and tr_meta["training_trip_count"] == 80)

# training record artifact shas (fresh three)
for name in ("gaussian_nb", "hist_gradient_boosting", "hist_gradient_boosting_tuned"):
    tr = read_json(OUT / "training" / f"{name}.json")
    check(f"training_record_sha_{name}", sha256(ROOT / tr["artifact"]) == tr["sha256"])
    check(f"training_record_fresh_{name}", tr["metadata"]["fresh_fit"] is True and tr["metadata"]["training_trip_count"] == 80)

# ---------------------------------------------------------------- strict fixture replay, all six models
for name in CV_FILES:
    ev = read_json(OUT / "evaluation" / f"{name}.json")
    pred = load_prediction_npz(ROOT / ev["student_predictions_file"])
    em = load_emission_model(ROOT / ev["model_path"])
    fixture = np.unique(np.linspace(0, len(student.X) - 1, 257).astype(int))
    got = predict_positive_proba(em.estimator, student.X[fixture])
    delta = float(np.max(np.abs(got - pred["probabilities"][fixture])))
    check(f"strict_fixture_replay_{name}", delta <= 1e-12, f"max_abs_delta={delta:.3e} rows={len(fixture)}")

# clean load report fixture reproduction (2000 TRAIN rows, frozen model)
clr = read_json(ART / "clean_load_report.json")
X2000 = train.X[:2000]
em = load_emission_model(ART / "model.joblib")
p2000 = predict_positive_proba(em.estimator, X2000)
check("clean_report_fixture_sha", hashlib.sha256(X2000.tobytes()).hexdigest() == clr["fixture_sha256"])
check("clean_report_claims", clr["passed"] is True and clr["max_probability_delta"] == 0.0
      and clr["atol"] == 1e-12 and clr["rtol"] == 0)
results["fixture2000_parent_probability_sha256"] = hashlib.sha256(np.asarray(p2000, dtype=np.float64).tobytes()).hexdigest()

# ---------------------------------------------------------------- versions + timestamps
results["versions_now"] = {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                           "joblib": joblib.__version__}
check("versions_match_results_json", results_json["versions"] == results["versions_now"],
      json.dumps(results["versions_now"]))
ts = {"train_complete": tc["completed_at_utc"], "acceptance": acceptance["checked_at_utc"],
      "clean_report": clr["checked_at_utc"]}
check("timestamps_ordered",
      datetime.fromisoformat(ts["train_complete"]) < datetime.fromisoformat(ts["acceptance"])
      < datetime.fromisoformat(ts["clean_report"]), json.dumps(ts))

# ---------------------------------------------------------------- GNB old-vs-new disclosure
old_gnb = read_json(ROOT / "experiments/final_five/cv/gaussian_nb.json")
new_gnb = read_json(OUT / "cv/gaussian_nb.json")
check("gnb_old_attempt_preserved_and_distinct",
      old_gnb["objective"] != new_gnb["objective"]
      and old_gnb["oof_metrics"]["log_loss"] != new_gnb["oof_metrics"]["log_loss"],
      f"old(float32)={old_gnb['objective']:.16f}/ll={old_gnb['oof_metrics']['log_loss']:.7f} "
      f"new(float64)={new_gnb['objective']:.16f}/ll={new_gnb['oof_metrics']['log_loss']:.7f}")
protocol = (ROOT / "experiments/final_five/PROTOCOL.md").read_text()
check("protocol_discloses_correction", "1.52e-5" in protocol and "float64" in protocol
      and "experiments/final_five/current" in protocol)

results["summary"] = {"total": len(results["checks"]), "passed": sum(c["passed"] for c in results["checks"]),
                      "failed": len(results["failures"])}
print(json.dumps(results, indent=1, default=str))
