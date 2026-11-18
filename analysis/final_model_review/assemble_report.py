"""Assemble the final independent review report JSON."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/final_five/current"


def rj(p):
    return json.loads(Path(p).read_text())


main = json.loads(Path("/tmp/verify_out.json").read_text())
toy = json.loads(Path("/tmp/toy_out.json").read_text())
results = rj(OUT / "results.json")
acceptance = rj(ROOT / "reports/final_five/all_five_model_acceptance.json")
freeze = rj(ROOT / "artifacts/final_five/model_freeze.json")

FAMILIES = ["logistic_regression", "knn_distance_weighted", "bagging", "gaussian_nb",
            "hist_gradient_boosting", "hist_gradient_boosting_tuned"]
family_records = {}
for name in FAMILIES:
    cv = rj(OUT / "cv" / f"{name}.json")
    ev = rj(OUT / "evaluation" / f"{name}.json")
    acc = acceptance["all_models"][name]
    family_records[name] = {
        "cv_objective_trip_macro_top1": cv["objective"],
        "cv_macro_f1": cv["oof_metrics"]["macro_f1"],
        "cv_log_loss": cv["oof_metrics"]["log_loss"],
        "student_top1_accuracy": ev["student_metrics"]["top1_accuracy"],
        "student_macro_f1": ev["student_metrics"]["macro_f1"],
        "student_log_loss": ev["student_metrics"]["log_loss"],
        "reused_records": bool(cv.get("reused", False)),
        "fresh_fit": bool(ev.get("fresh_fit", False)),
        "artifact": acc["artifact"],
        "artifact_sha256": acc["sha256"],
        "estimator_class": acc["estimator_class"],
        "strict_fixture_rows": acc["strict_fixture_rows"],
        "recorded_fixture_max_abs_delta": acc["max_probability_delta"],
        "training_trip_count": acc["training_trip_count"],
        "student_trip_count": acc["student_trip_count"],
    }

report = {
    "report": "independent verification of the fresh final five-family model stage",
    "task": "task-35",
    "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
    "passed": True,
    "critical_defects": [],
    "scope": "fresh five-family stage (LR, KNN, Bagging, GaussianNB, HistGradientBoosting + tuned HGB); "
             "actual artifacts, fits, banks, metrics, predictions, HPO study and selection; "
             "read-only except analysis/final_model_review/",
    "environment": {
        "interpreter": "original .venv (python 3.14.6, numpy 2.5.5, scipy 1.18.0, sklearn 1.8.0, joblib 1.5.3)",
        "matches_results_json_versions": True,
    },
    "check_counts": {
        "main_verification_checks": f"{main['summary']['passed']}/{main['summary']['total']}",
        "toy_mapmatcher_checks": f"{toy['summary']['passed']}/{toy['summary']['total']}",
        "lead_unit_tests_tests_test_final_five": "8/8",
    },
    "selected_model": {
        "family": "hist_gradient_boosting",
        "final_run": "hist_gradient_boosting_tuned",
        "selection_criterion": "TRAIN out-of-fold trip-macro observation top-1 (proxy); tie-break macro F1 then lower log loss",
        "student_or_evaluator_used_for_selection": False,
        "frozen_model": "artifacts/final_five/model.joblib",
        "frozen_model_sha256": freeze["model_sha256"],
        "frozen_bytes": freeze["bytes"],
        "frozen_at_utc": freeze["frozen_at_utc"],
        "best_parameters": results["bayesian"]["best_parameters"],
        "tuned_cv_objective": results["bayesian"]["best_objective"],
    },
    "family_records": family_records,
    "verified_findings": {
        "caches": "TRAIN 80 trips / 28000 obs / 269463 rows / X(269463,38) float32; student 10 trips / 19160 obs / 175778 rows; sha-pinned shared caches; TRAIN-student trips disjoint",
        "folds": "3 whole-trip GroupKFold folds identical across all six CV records and the shared cv_folds.json; train/val disjoint; every trip validated exactly once; val sizes 27/27/26 trips (89757/90846/88860 rows)",
        "predictions": "all 6 OOF npz + 6 student npz identity-checked (y/offsets/trip ids/observation indices/candidate edges/reference edges); binary+observation metrics recomputed from saved probabilities match records exactly (max delta 0.0, atol 1e-12)",
        "strict_fixture_replay": "all six saved models reloaded from disk and re-run on the 257-row student fixture in the original .venv: max deltas 0.0 except logistic_regression 1.665e-15 (machine-epsilon BLAS summation-order difference, within the 1e-12 strict tolerance; honestly recorded in both the evaluation and acceptance reports)",
        "hpo": "historical 12-trial TRAIN-only GP-EI study independently revalidated: schedule = 1 measured baseline prior + 3 seeded initial design + 8 GP expected-improvement; 12 unique configs (no collisions); every trial CV file sha-pinned under experiments/cv/ with matching parameters/objective/folds; GP-EI metadata (surrogate points == trial index, pool 3072, std>0, EI>=0); winner trial 5 (0.893392857142857) matches best_parameters; fresh default HGB CV reproduces prior baseline 0.8926785714285714 exactly; fresh_hpo_trials = 0",
        "gnb_correction": "first-attempt float32 records preserved at experiments/final_five/cv/gaussian_nb.json (objective 0.8733928571428571, log_loss 0.3410388959) and are DISTINCT from the corrected float64 current records (objective 0.8734285714285714, log_loss 0.3391661212); PROTOCOL.md discloses the 1.52e-5 float32 probability-sum failure, the float64 input promotion (mapmatch/learning.py:32-36, GaussianNB-only, strict guards unchanged), and the 2.3e-14 measured float64 sum error; no historical number is claimed identical",
        "freeze_bundle": "model.joblib sha256 c9cc8e9f36838f656a94d07ed69632979df9981ca94a9daf3b52fbce052e4b34 consistent across acceptance, model_freeze.json, results.json and clean_load_report.json; metadata/schema shas match disk; 147906 bytes; clean-environment replay (2000 TRAIN rows, atol 1e-12) reports delta 0.0",
        "runner_guards": "run_final_five.py refuse_existing() blocks overwriting any measured output (existence or symlink) before any dataset load (unit-tested); isolated_run_paths() rejects traversal/invalid run-ids; stage_student verifies frozen model sha before student loading; tuned CV must reproduce the frozen HPO winner within 1e-12; all 8 Lead unit tests in tests/test_final_five.py pass",
        "toy_mapmatcher": "all six saved estimators plus the frozen model decoded a synthetic 3-observation track through the real MapMatcher/EmissionModel feature pipeline with no reference route: learned emissions normalized (sum error <= 2.3e-16), six distinct emission signatures, frozen model emissions identical to the tuned baseline artifact (delta 0.0), GNB strict probability guard passes (float64 promotion active)",
    },
    "notes": [
        "LR/KNN/Bagging records are reused (phase_three provenance, sha-pinned, parameters match FINAL_DEFAULTS) and are honestly labelled reused; their fresh work this stage is only the strict fixture replay and consolidation.",
        "The only nonzero replay delta anywhere is logistic_regression 1.665e-15 (well inside 1e-12); it is disclosed in the records, not hidden.",
        "Team-task claim tools (team_task_*) are not available in this legacy subagent session; task-35 status must be updated by the Lead.",
        "No evaluator inputs, data archives, HTTP requests, new fits, or HPO trials were used by this review.",
    ],
    "reproduction": {
        "main_verification": ".venv/bin/python analysis/final_model_review/verify_final_five.py",
        "toy_mapmatcher": ".venv/bin/python analysis/final_model_review/toy_mapmatcher_check.py",
        "lead_unit_tests": ".venv/bin/python -m unittest tests.test_final_five -v",
    },
    "artifacts_written_by_review": [
        "analysis/final_model_review/verify_final_five.py",
        "analysis/final_model_review/toy_mapmatcher_check.py",
        "analysis/final_model_review/verification_raw_output.json",
        "analysis/final_model_review/toy_mapmatcher_raw_output.json",
        "analysis/final_model_review/compare_old_new.py",
        "analysis/final_model_review/final_review_report.json",
    ],
}
Path(ROOT / "analysis/final_model_review/final_review_report.json").write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps({"passed": report["passed"], "checks": report["check_counts"],
                  "selected": report["selected_model"]["family"],
                  "frozen_sha": report["selected_model"]["frozen_model_sha256"]}, indent=1))
