#!/usr/bin/env python3
"""Final-five (five-model-completion) figure generator.

Scope: writes ONLY under visualizations/final_five/. The completed phase-one
EDA plots (visualizations/plots/...), the phase-three figures
(visualizations/phase_three/...) and every other path in the repository are
read-only for this generator; existing EDA assets are re-pinned by hash.

Families (docs/FINAL_REQUIREMENTS.md, five-model rule):
  logistic_regression, knn_distance_weighted, bagging, gaussian_nb,
  hist_gradient_boosting - plus the tuned variant of the selected family
  (hist_gradient_boosting_tuned), six measured configurations in total.

Inputs (owned elsewhere; nothing is trained or refitted here):
  experiments/final_five/current/results.json          (learning pipeline)
  experiments/final_five/current/model_comparison.csv
  experiments/final_five/current/selection.json
  experiments/final_five/current/sequence_classification.json
  experiments/final_five/current/cv/*.json             (six run files)
  experiments/final_five/current/evaluation/*.json     (six run files)
  experiments/final_five/current/bayesian_optimization.json
  experiments/bayesian_trials.jsonl                    (historical HPO log)
  reports/final_five/all_five_model_acceptance.json
  artifacts/final_five/clean_load_report.json
  reports/final_five/decoder/student_ablation.json     (decoder mode, optional)

Every plotted number is read from those measured files. Nothing is invented:
no epoch-wise loss curves are fabricated for families that keep no epoch-wise
training history (sklearn estimators here keep none; stated as N/A), chart
scales are labelled and never cropped silently, and truncated y-axes say so
in the caption. HPO parameter plots show observed samples only and make no
causal importance claim.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # single-thread numeric policy; explicit env still wins
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import argparse
import csv
import hashlib
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIVE_DIR = Path(__file__).resolve().parent
ROOT = FIVE_DIR.parents[1]
PLOTS = FIVE_DIR / "plots"
MANIFEST = FIVE_DIR / "figure_manifest.json"
sys.path.insert(0, str(ROOT))
from visualizations.final_five.figure_integrity import (  # noqa: E402
    EXPECTED_BY_STAGE, safe_output as guarded_output, verify_input_pins, verify_record_integrity,
)

# ------------------------------------------------------------------ constants

FAMILIES = ("logistic_regression", "knn_distance_weighted", "bagging",
            "gaussian_nb", "hist_gradient_boosting")
FINAL_RUN = "hist_gradient_boosting_tuned"
RUNS = FAMILIES + (FINAL_RUN,)  # five families + tuned variant of the winner

RUN_LABELS = {
    "logistic_regression": "Logistic\nregression",
    "knn_distance_weighted": "KNN\n(distance-wtd)",
    "bagging": "Bagging\n(decision trees)",
    "gaussian_nb": "Gaussian\nNB",
    "hist_gradient_boosting": "HistGB\n(default)",
    "hist_gradient_boosting_tuned": "HistGB\n(tuned)",
}
RUN_COLORS = {
    "logistic_regression": "#4477aa",
    "knn_distance_weighted": "#66ccee",
    "bagging": "#228833",
    "gaussian_nb": "#ccbb44",
    "hist_gradient_boosting": "#ee6677",
    "hist_gradient_boosting_tuned": "#aa3377",
}
SELECTED_FAMILY = "hist_gradient_boosting"
SELECTED_SHA256 = "c9cc8e9f36838f656a94d07ed69632979df9981ca94a9daf3b52fbce052e4b34"

HPO_PARAM_KEYS = ("max_iter", "learning_rate", "max_leaf_nodes",
                  "min_samples_leaf", "l2_regularization")

PROXY_NOTE = ("Proxy labels: nearest reference-route segment plus weak GPS-motion "
              "orientation; not supplied pointwise ground truth.")
BINARY_NOTE = ("Binary candidate-row metric: one row per (observation, candidate "
               "directed arc) at threshold 0.5; distinct from positive-class F1 and "
               "from observation-level edge F1.")
EPOCH_NA_NOTE = ("None of the five families keeps an epoch-wise training history under "
                 "this sklearn pipeline; per-epoch loss curves are stated as N/A in "
                 "docs/FINAL_REQUIREMENTS.md instead of being fabricated.")

# Expected denominators of the real final-five run (validated before charting).
REAL_EXPECTED = {
    "families": FAMILIES,
    "final_run": FINAL_RUN,
    "selected_model": "hist_gradient_boosting",
    "cv_folds": 3,
    "train_trips": 80,
    "train_observations": 28000,
    "train_observations_per_trip": 350,
    "train_candidate_rows": 269463,
    "train_positive_rows": 27422,
    "train_candidate_miss_count": 578,
    "student_trips": 10,
    "student_observations": 19160,
    "student_candidate_rows": 175778,
    "student_positive_rows": 19141,
    "student_candidate_miss_count": 19,
    "selected_model_sha256": SELECTED_SHA256,
    "hpo_trials": 12,
    "hpo_measured_prior": 1,
    "hpo_seeded_initial_design": 3,
    "hpo_gp_expected_improvement": 8,
}

FIG_DPI = 300
A4_HALF = (8.27, 5.85)   # inches, landscape half page
A4_FULL = (8.27, 11.69)  # inches, portrait full page

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.facecolor": "white", "axes.facecolor": "#f8fafc",
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#dbe3ec", "grid.alpha": 0.65,
    "savefig.facecolor": "white", "legend.frameon": False,
})

REGISTRY: list[dict] = []


# ------------------------------------------------------------------ utilities

def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_file_path(path)


def sha256_file_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_with_hash(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), sha256_bytes(raw)


def safe_output(path: Path) -> Path:
    return guarded_output(path, FIVE_DIR)


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def wrap_caption(text: str, width: int = 138) -> str:
    return "\n".join(textwrap.wrap(" ".join(text.split()), width=width))


def save_figure(fig, filename: str, title: str, caption: str, *, units: str,
                source: str, stage: str = "models", ocr_expect: list[str] | None = None) -> dict:
    """Save one figure as 300 dpi PNG plus PDF and register its manifest record."""
    png = safe_output(PLOTS / filename)
    pdf = safe_output(png.with_suffix(".pdf"))
    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.985)
    wrapped = wrap_caption(caption)
    fig.text(0.02, 0.015, wrapped, fontsize=10, color="#334155", va="bottom")
    fig.tight_layout(rect=(0.015, 0.015 + 0.018 * wrapped.count("\n") + 0.035,
                           0.985, 0.925), h_pad=2.2, w_pad=2.0)
    fig.savefig(png, dpi=FIG_DPI)
    fig.savefig(pdf)
    plt.close(fig)
    record = {
        "path": rel(png), "pdf_path": rel(pdf),
        "bytes": png.stat().st_size, "sha256": sha256_file_path(png),
        "pdf_sha256": sha256_file_path(pdf),
        "dpi": FIG_DPI,
        "pixels": {"width": int(fig.get_size_inches()[0] * FIG_DPI),
                   "height": int(fig.get_size_inches()[1] * FIG_DPI)},
        "page": "A4 half (landscape)" if max(fig.get_size_inches()) < 9 else "A4 full (portrait)",
        "title": title, "caption": caption, "units": units,
        "source": source, "stage": stage, "status": "actual_measured",
        "created_utc": utc_now(),
        "ocr_expect": ocr_expect or [title.split(":")[0].split()[0].lower()],
    }
    REGISTRY.append(record)
    return record


def label_bars(ax, bars, values=None, fmt="{:.4f}", fontsize=11, offset=4):
    for i, bar in enumerate(bars):
        v = bar.get_height() if values is None else values[i]
        ax.annotate(fmt.format(v), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, offset), textcoords="offset points", ha="center",
                    fontsize=fontsize, rotation=0)


# ------------------------------------------------------------------ validation

def validate_family_sets(results: dict, expected: dict) -> list[str]:
    families = tuple(results.get("families") or ())
    check(list(families) == list(expected["families"]),
          f"family set mismatch: {families} != {expected['families']}")
    check(results.get("family_count") == len(families), "family_count disagrees with families list")
    selected = results.get("selected_model")
    check(selected == expected["selected_model"],
          f"selected_model {selected!r} != {expected['selected_model']!r}")
    final_run = results.get("final_run_name")
    check(final_run == expected["final_run"],
          f"final_run_name {final_run!r} != {expected['final_run']!r}")
    check(final_run != selected and final_run.startswith(selected),
          "final run must be a tuned variant of the selected family, not the bare family name")
    runs = list(families) + [final_run]
    check(set(results["cv"]) == set(runs),
          f"cv keys {sorted(results['cv'])} != the six expected runs {sorted(runs)}")
    check(set(results["evaluations"]) == set(runs),
          f"evaluation keys {sorted(results['evaluations'])} != the six expected runs")
    return runs


def _check_fold_partition(folds: list[dict], expected_trips: int, run: str) -> None:
    val_ids: list[str] = []
    for fold in folds:
        val_ids.extend(fold["validation_trip_ids"])
        check(len(set(fold["train_trip_ids"]) & set(fold["validation_trip_ids"])) == 0,
              f"{run}: fold {fold.get('fold')} train/validation trips overlap")
    check(len(val_ids) == len(set(val_ids)),
          f"{run}: validation trips repeat across folds")
    check(len(set(val_ids)) == expected_trips,
          f"{run}: folds cover {len(set(val_ids))} validation trips, expected {expected_trips}")


def validate_cv_block(run: str, block: dict, expected: dict) -> dict:
    oof = block["oof_metrics"]
    check(len(block["folds"]) == expected["cv_folds"],
          f"{run}: {len(block['folds'])} folds, expected {expected['cv_folds']}")
    _check_fold_partition(block["folds"], expected["train_trips"], run)
    per_trip = oof["per_trip"]
    check(len(per_trip) == expected["train_trips"],
          f"{run}: {len(per_trip)} per-trip CV rows, expected {expected['train_trips']}")
    check(all(t["observations"] == expected["train_observations_per_trip"] for t in per_trip),
          f"{run}: CV per-trip observation counts differ from {expected['train_observations_per_trip']}")
    check(oof["observations"] == expected["train_observations"],
          f"{run}: CV observations {oof['observations']} != {expected['train_observations']}")
    check(oof["candidate_rows"] == expected["train_candidate_rows"],
          f"{run}: CV candidate rows {oof['candidate_rows']} != {expected['train_candidate_rows']}")
    check(oof["positive_rows"] == expected["train_positive_rows"],
          f"{run}: CV positive rows {oof['positive_rows']} != {expected['train_positive_rows']}")
    check(oof["candidate_miss_count"] == expected["train_candidate_miss_count"],
          f"{run}: CV proxy-target miss count {oof['candidate_miss_count']} != "
          f"{expected['train_candidate_miss_count']}")
    cm = oof["confusion_matrix"]
    check(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1] == oof["candidate_rows"],
          f"{run}: CV confusion matrix does not sum to candidate rows")
    check(cm[1][0] + cm[1][1] == oof["positive_rows"],
          f"{run}: CV confusion positive row does not match positive rows")
    check(oof["observations"] == oof["candidate_present_observations"] + oof["candidate_miss_count"],
          f"{run}: CV observation identity observations != present + miss")
    pooled = oof["top1_correct_observations"] / oof["observations"]
    check(abs(pooled - oof["top1_accuracy"]) < 1e-9,
          f"{run}: CV pooled top1 != correct/observations")
    trip_macro = sum(t["top1_accuracy"] for t in per_trip) / len(per_trip)
    check(abs(trip_macro - oof["trip_macro_top1_accuracy"]) < 1e-9,
          f"{run}: CV trip-macro top1 != mean of per-trip rates")
    # equal 350-observation denominators: pooled and trip-macro must coincide
    check(abs(oof["top1_accuracy"] - oof["trip_macro_top1_accuracy"]) < 1e-9,
          f"{run}: CV pooled and trip-macro top1 differ despite equal trip denominators")
    for fold in block["folds"]:
        m = fold["metrics"]
        for key in ("macro_f1", "top1_accuracy", "log_loss"):
            check(key in m and 0.0 <= m[key] <= 1.0, f"{run}: fold metric {key} missing or out of range")
    return {"oof": oof, "folds": block["folds"], "parameters": block.get("parameters", {}),
            "training_seconds_total": block.get("training_seconds_total", 0.0),
            "inference_seconds_total": block.get("inference_seconds_total", 0.0),
            "elapsed_seconds": block.get("elapsed_seconds", 0.0),
            "reused": block.get("reused")}


def validate_eval_block(run: str, block: dict, expected: dict) -> dict:
    st = block["student_metrics"]
    check(st["observations"] == expected["student_observations"],
          f"{run}: student observations {st['observations']} != {expected['student_observations']}")
    check(st["candidate_rows"] == expected["student_candidate_rows"],
          f"{run}: student candidate rows {st['candidate_rows']} != {expected['student_candidate_rows']}")
    check(st["positive_rows"] == expected["student_positive_rows"],
          f"{run}: student positive rows {st['positive_rows']} != {expected['student_positive_rows']}")
    check(st["candidate_miss_count"] == expected["student_candidate_miss_count"],
          f"{run}: student miss count {st['candidate_miss_count']} != {expected['student_candidate_miss_count']}")
    check(len(st["per_trip"]) == expected["student_trips"],
          f"{run}: {len(st['per_trip'])} student trips, expected {expected['student_trips']}")
    check(sum(t["observations"] for t in st["per_trip"]) == st["observations"],
          f"{run}: student per-trip observations do not sum to the total")
    pooled = st["top1_correct_observations"] / st["observations"]
    check(abs(pooled - st["top1_accuracy"]) < 1e-9,
          f"{run}: student pooled top1 != correct/observations")
    trip_macro = sum(t["top1_accuracy"] for t in st["per_trip"]) / len(st["per_trip"])
    check(abs(trip_macro - st["trip_macro_top1_accuracy"]) < 1e-9,
          f"{run}: student trip-macro top1 != unweighted mean of per-trip rates")
    cm = st["confusion_matrix"]
    check(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1] == st["candidate_rows"],
          f"{run}: student confusion matrix does not sum to candidate rows")
    check(cm[1][0] + cm[1][1] == st["positive_rows"],
          f"{run}: student confusion positive row does not match positive rows")
    return {"student": st, "resub": block["train_resubstitution_metrics"],
            "training_seconds": block.get("training_seconds", 0.0),
            "student_inference_seconds": block.get("student_inference_seconds", 0.0),
            "reused": block.get("reused")}


def validate_hpo(bayes: dict, cv_view: dict, results: dict, expected: dict) -> dict:
    trials = bayes["trials"]
    check(len(trials) == expected["hpo_trials"],
          f"HPO lists {len(trials)} trials, expected {expected['hpo_trials']}")
    check(bayes.get("trials_including_measured_prior") == len(trials),
          "HPO trial-count field disagrees with the trial list")
    categories: dict[str, int] = {}
    for t in trials:
        categories[t["proposal"]] = categories.get(t["proposal"], 0) + 1
        check(0.0 <= t["objective"] <= 1.0, f"HPO trial {t['trial']} objective out of range")
        check(set(t["parameters"]) == set(HPO_PARAM_KEYS),
              f"HPO trial {t['trial']} parameters != the five search-space keys")
    check(categories.get("measured_baseline_prior") == expected["hpo_measured_prior"],
          f"HPO measured prior count {categories.get('measured_baseline_prior')} != {expected['hpo_measured_prior']}")
    check(categories.get("seeded_initial_design") == expected["hpo_seeded_initial_design"],
          f"HPO seeded count {categories.get('seeded_initial_design')} != {expected['hpo_seeded_initial_design']}")
    check(categories.get("gaussian_process_expected_improvement") == expected["hpo_gp_expected_improvement"],
          f"HPO GP-EI count {categories.get('gaussian_process_expected_improvement')} != "
          f"{expected['hpo_gp_expected_improvement']}")
    best = max(trials, key=lambda t: t["objective"])
    check(bayes["best_objective"] == best["objective"], "HPO best_objective != max trial objective")
    tuned_params = cv_view[expected["final_run"]]["parameters"]
    check(bayes["best_parameters"] == tuned_params,
          "HPO best parameters differ from the tuned run's CV parameters")
    check("TRAIN" in bayes["selection_data"] and "student" in bayes["selection_data"].lower(),
          "HPO selection_data must state TRAIN-only usage")
    # selection objectives must equal the five default runs' CV trip-macro top1
    sel = results["selection_objectives"] if "selection_objectives" in results else None
    if sel is not None:
        for name, value in sel.items():
            check(abs(value - cv_view[name]["oof"]["trip_macro_top1_accuracy"]) < 1e-9,
                  f"selection objective for {name} != its CV trip-macro top1")
    return {"trials": trials, "categories": categories, "best": best,
            "baseline_objective": bayes["baseline_objective"],
            "seconds": bayes["seconds"], "method": bayes["method"],
            "search_space": bayes["search_space"], "objective": bayes["objective"]}


def validate_sequence(seq: dict, eval_view: dict, runs: list[str], expected: dict) -> dict:
    check(list(seq.keys()) == runs or set(seq.keys()) == set(runs),
          f"sequence_classification keys {sorted(seq)} != the six runs")
    for run in runs:
        v = seq[run]
        check(v["observations"] == expected["student_observations"],
              f"{run}: sequence observations != {expected['student_observations']}")
        check(v["candidate_miss_count"] == expected["student_candidate_miss_count"],
              f"{run}: sequence miss count != {expected['student_candidate_miss_count']}")
        check(abs(v["top1_accuracy"] - eval_view[run]["student"]["top1_accuracy"]) < 1e-9,
              f"{run}: sequence top1 disagrees with the evaluation file")
    return seq


def prepare(results: dict, cv_files: dict[str, dict], eval_files: dict[str, dict],
            seq: dict, selection: dict, expected: dict) -> dict:
    """Strong validation of every source file; returns the chart-ready view."""
    runs = validate_family_sets(results, expected)
    cv_view: dict[str, dict] = {}
    for run in runs:
        # cross-check results.json against the per-run cv file
        check(results["cv"][run]["oof_metrics"]["macro_f1"] == cv_files[run]["oof_metrics"]["macro_f1"],
              f"{run}: results.json and cv/{run}.json macro F1 disagree")
        cv_view[run] = validate_cv_block(run, cv_files[run], expected)
    eval_view: dict[str, dict] = {}
    for run in runs:
        check(results["evaluations"][run]["student_metrics"]["macro_f1"]
              == eval_files[run]["student_metrics"]["macro_f1"],
              f"{run}: results.json and evaluation/{run}.json student macro F1 disagree")
        eval_view[run] = validate_eval_block(run, eval_files[run], expected)
    seq_view = validate_sequence(seq, eval_view, runs, expected)
    bayes_view = validate_hpo(results["bayesian"], cv_view,
                             {"selection_objectives": selection.get("objectives")}, expected)
    check(selection.get("student_used") is False, "selection.json must record student_used = false")
    check(selection.get("evaluator_used") is False, "selection.json must record evaluator_used = false")
    check(selection["selected_model"] == expected["selected_model"],
          "selection.json selected_model mismatch")
    freeze = results["model_freeze"]
    check(freeze["model_sha256"] == expected.get("selected_model_sha256", SELECTED_SHA256),
          f"model freeze sha {freeze['model_sha256']} != the selected artifact sha")
    check(freeze["student_used_for_selection"] is False and freeze["evaluator_used_for_selection"] is False,
          "model freeze must record that neither student nor evaluator data selected the model")
    return {"runs": runs, "cv": cv_view, "eval": eval_view, "seq": seq_view,
            "hpo": bayes_view, "selection": selection, "freeze": freeze}


# ------------------------------------------------------------------ charts

def _run_axes(ax, runs, rotation=0):
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([RUN_LABELS[r] for r in runs], rotation=rotation)
    ax.grid(axis="y", alpha=0.5)


def chart_cv_macro_f1(view, meta):
    runs = view["runs"]
    fig, ax = plt.subplots(figsize=A4_HALF)
    means = [view["cv"][r]["oof"]["macro_f1"] for r in runs]
    stds = [np.std([f["metrics"]["macro_f1"] for f in view["cv"][r]["folds"]]) for r in runs]
    bars = ax.bar(range(len(runs)), means, 0.62, color=[RUN_COLORS[r] for r in runs],
                  yerr=stds, capsize=5, error_kw={"elinewidth": 1.4})
    label_bars(ax, bars)
    _run_axes(ax, runs)
    ax.set_ylim(0.90, 0.96)
    ax.set_ylabel("Binary macro F1 (0-1)")
    ax.set_title("Out-of-fold binary macro F1, pooled over the 3 grouped folds")
    sel = runs.index("hist_gradient_boosting")
    ax.get_xticklabels()[sel].set_fontweight("bold")
    save_figure(fig, "final_five_cv_macro_f1.png",
                "Cross-validated binary macro F1: five families plus the tuned HistGB run",
                f"Pooled out-of-fold binary macro F1 from 3-fold grouped CV over all 80 TRAIN trips "
                f"(350 observations per trip, 269,463 candidate rows, 27,422 positive). Error bars show "
                f"the standard deviation across the three folds. {BINARY_NOTE} This metric did not drive "
                f"selection; the recorded objective was TRAIN trip-macro top-1 (see the primary-objective "
                f"figure). Y-axis starts at 0.90; the metric range is 0 to 1. {PROXY_NOTE} "
                f"Source: experiments/final_five/current/cv/*.json, results.json.",
                units="fraction 0-1 (binary macro F1 over candidate rows)",
                source="experiments/final_five/current/cv/*.json; experiments/final_five/current/results.json",
                ocr_expect=["macro", "f1", "fold"])


def chart_primary_objective(view, meta):
    runs = view["runs"]
    fig, ax = plt.subplots(figsize=A4_HALF)
    objectives = [view["cv"][r]["oof"]["trip_macro_top1_accuracy"] for r in runs]
    colors = [RUN_COLORS[r] for r in runs]
    bars = ax.bar(range(len(runs)), objectives, 0.62, color=colors)
    label_bars(ax, bars)
    _run_axes(ax, runs)
    ax.set_ylim(0.85, 0.90)
    ax.set_ylabel("Trip-macro top-1 accuracy (0-1)")
    ax.set_title("Selection objective: TRAIN out-of-fold trip-macro top-1")
    ax.annotate("selected family", (runs.index(SELECTED_FAMILY), objectives[runs.index(SELECTED_FAMILY)]),
                xytext=(0, 26), textcoords="offset points", ha="center", fontsize=11,
                fontweight="bold", color="#7a1020")
    ax.annotate("final artifact\n(refit, all 80 trips)", (runs.index(FINAL_RUN), objectives[runs.index(FINAL_RUN)]),
                xytext=(0, 26), textcoords="offset points", ha="center", fontsize=10, color="#4b1a66")
    save_figure(fig, "final_five_primary_objective.png",
                "Primary selection objective: TRAIN out-of-fold trip-macro top-1 accuracy",
                f"Criterion recorded in selection.json: out-of-fold trip-macro top-1 accuracy on the 80 TRAIN "
                f"trips, measured before any student or evaluator data was loaded. Every TRAIN trip contributes "
                f"350 observations, so pooled and trip-macro top-1 coincide exactly on CV. Among the five family "
                f"defaults HistGradientBoosting is highest (0.8927); its tuned variant reaches 0.8934 and was "
                f"refitted on all 80 trips as the final frozen artifact. Binary macro F1 and log loss were "
                f"tie-breakers only. Y-axis starts at 0.85; the metric range is 0 to 1. "
                f"Source: selection.json, cv/*.json, bayesian_optimization.json.",
                units="fraction 0-1 (trip-macro top-1 accuracy)",
                source="experiments/final_five/current/selection.json; cv/*.json",
                ocr_expect=["objective", "top-1", "trip"])


def chart_student_metrics(view, meta):
    runs = view["runs"]
    fig, ax = plt.subplots(figsize=A4_HALF)
    x = np.arange(len(runs))
    pooled = [view["eval"][r]["student"]["top1_accuracy"] for r in runs]
    macro = [view["eval"][r]["student"]["macro_f1"] for r in runs]
    trip = [view["eval"][r]["student"]["trip_macro_top1_accuracy"] for r in runs]
    w = 0.36
    b1 = ax.bar(x - w / 2, pooled, w, color=[RUN_COLORS[r] for r in runs], alpha=0.95,
                label="Pooled top-1 (19,160 obs)")
    b2 = ax.bar(x + w / 2, macro, w, color=[RUN_COLORS[r] for r in runs], alpha=0.45,
                hatch="//", label="Binary macro F1 (candidate rows)")
    ax.scatter(x, trip, marker="D", s=70, color="#111827", zorder=5,
               label="Trip-macro top-1 (mean of 10 trips)")
    label_bars(ax, b1, fmt="{:.4f}", fontsize=10)
    label_bars(ax, b2, fmt="{:.4f}", fontsize=10, offset=16)
    _run_axes(ax, runs)
    ax.set_ylim(0.85, 0.97)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Student-development partition: top-1 accuracy and binary macro F1")
    ax.legend(loc="upper left", fontsize=10)
    save_figure(fig, "final_five_student_top1_macro_f1.png",
                "Student-development results: pooled top-1, trip-macro top-1, and binary macro F1",
                f"All 10 student-development trips: 19,160 observations, 175,778 candidate rows, 19,141 positive. "
                f"Pooled top-1 weighs every observation equally, so longer trips count more; the diamond markers "
                f"show the unweighted mean of the 10 per-trip rates, which is higher for every configuration "
                f"because the below-average trips carry more observations than the above-average ones do. On CV "
                f"the two coincide (equal 350-observation trips); on this partition they are distinct "
                f"denominators. {BINARY_NOTE} The student partition was "
                f"measured after selection and never used to choose or tune a model. Y-axis starts at 0.85. "
                f"Source: evaluation/*.json.",
                units="fractions 0-1 (top-1 accuracy; binary macro F1)",
                source="experiments/final_five/current/evaluation/*.json",
                ocr_expect=["student", "top-1", "macro"])


def chart_log_loss(view, meta):
    runs = view["runs"]
    fig, ax = plt.subplots(figsize=A4_HALF)
    x = np.arange(len(runs))
    cv_ll = [view["cv"][r]["oof"]["log_loss"] for r in runs]
    st_ll = [view["eval"][r]["student"]["log_loss"] for r in runs]
    w = 0.36
    b1 = ax.bar(x - w / 2, cv_ll, w, color=[RUN_COLORS[r] for r in runs], alpha=0.95,
                label="CV out-of-fold (TRAIN)")
    b2 = ax.bar(x + w / 2, st_ll, w, color=[RUN_COLORS[r] for r in runs], alpha=0.45,
                hatch="//", label="Student-development")
    label_bars(ax, b1, fmt="{:.4f}", fontsize=10)
    label_bars(ax, b2, fmt="{:.4f}", fontsize=10, offset=16)
    _run_axes(ax, runs)
    ax.set_ylim(0, 0.42)
    ax.set_ylabel("Binary log loss (nats, lower is better)")
    ax.set_title("Binary candidate-row log loss: CV and student partitions")
    ax.legend(loc="upper left", fontsize=10)
    gnb = runs.index("gaussian_nb")
    ax.annotate("GaussianNB: F1 competitive,\nprobabilities poorly calibrated",
                (gnb, st_ll[gnb]), xytext=(gnb - 1.1, 0.315), fontsize=10,
                arrowprops={"arrowstyle": "->", "color": "#334155"})
    save_figure(fig, "final_five_binary_log_loss.png",
                "Binary log loss: CV out-of-fold and student-development",
                f"Lower is better. GaussianNB is competitive on binary macro F1 yet clearly worst on log loss "
                f"(CV 0.3392, student 0.3591): its probability outputs are poorly calibrated on this "
                f"candidate-row task. The student value comes from the fresh float64 refit of this phase, not "
                f"from the historical artifact (see reports/final_five/gnb_probability_diagnostic.json: "
                f"probabilities finite, within [0,1], row sums within 1.6e-5 of 1). Every other family stays "
                f"below 0.10 on both partitions. {BINARY_NOTE} Source: cv/*.json, evaluation/*.json.",
                units="nats per candidate row (natural-log loss)",
                source="experiments/final_five/current/cv/*.json; evaluation/*.json",
                ocr_expect=["log", "loss", "gaussian"])


def chart_log_loss_partitions(view, meta):
    runs = view["runs"]
    fig, ax = plt.subplots(figsize=A4_HALF)
    x = np.arange(len(runs))
    resub = [view["eval"][r]["resub"]["log_loss"] for r in runs]
    cv_ll = [view["cv"][r]["oof"]["log_loss"] for r in runs]
    st_ll = [view["eval"][r]["student"]["log_loss"] for r in runs]
    w = 0.26
    ax.bar(x - w, resub, w, color="#94a3b8", label="Train resubstitution (all 80 trips)")
    b2 = ax.bar(x, cv_ll, w, color="#2266b2", label="CV out-of-fold (TRAIN)")
    b3 = ax.bar(x + w, st_ll, w, color="#d77926", label="Student-development")
    for bars in (b2, b3):
        label_bars(ax, bars, fmt="{:.4f}", fontsize=9, offset=3)
    _run_axes(ax, runs)
    ax.set_ylim(0, 0.42)
    ax.set_ylabel("Binary log loss (nats, lower is better)")
    ax.set_title("Measured log loss by partition; no epoch-wise curves exist to plot")
    ax.legend(loc="upper left", fontsize=10)
    save_figure(fig, "final_five_log_loss_partitions.png",
                "Binary log loss on three measured partitions (epoch-wise history: N/A)",
                f"Train resubstitution (refit on all 80 TRAIN trips, rescored on TRAIN), CV out-of-fold, and "
                f"student log loss per configuration. {EPOCH_NA_NOTE} These measured partition losses replace "
                f"epoch plots; resubstitution loss is optimistic by construction because the same rows are "
                f"scored that trained the fit. GaussianNB's calibration gap dominates both partitions. "
                f"Source: evaluation/*.json, cv/*.json.",
                units="nats per candidate row (natural-log loss)",
                source="experiments/final_five/current/evaluation/*.json; cv/*.json",
                ocr_expect=["partition", "log", "loss"])


def chart_selection_comparison(view, meta):
    runs = view["runs"]
    fig, axs = plt.subplots(2, 2, figsize=A4_HALF)
    specs = [
        ("TRAIN CV trip-macro top-1\n(selection objective)",
         [view["cv"][r]["oof"]["trip_macro_top1_accuracy"] for r in runs], (0.85, 0.90), "{:.4f}"),
        ("CV binary macro F1\n(tie-breaker)",
         [view["cv"][r]["oof"]["macro_f1"] for r in runs], (0.90, 0.96), "{:.4f}"),
        ("CV binary log loss\n(tie-breaker, lower is better)",
         [view["cv"][r]["oof"]["log_loss"] for r in runs], (0, 0.12), "{:.4f}"),
        ("Student inference seconds\n(log scale)",
         [view["eval"][r]["student_inference_seconds"] for r in runs], None, "{:.3g} s"),
    ]
    for ax, (title, values, ylim, fmt) in zip(axs.flat, specs):
        colors = [RUN_COLORS[r] for r in runs]
        bars = ax.bar(range(len(runs)), values, 0.62, color=colors)
        label_bars(ax, bars, fmt=fmt, fontsize=9, offset=3)
        _run_axes(ax, runs, rotation=0)
        ax.set_title(title, fontsize=11.5)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_yscale("log")
            ax.set_ylim(0.01, 600)
        ax.tick_params(axis="x", labelsize=8.5)
    axs[0][0].get_xticklabels()[runs.index(SELECTED_FAMILY)].set_fontweight("bold")
    fig.suptitle("")
    save_figure(fig, "final_five_selection_comparison.png",
                "Why HistGradientBoosting was selected: four measured axes",
                f"The recorded decision (selection.json) ranks the five family defaults by TRAIN out-of-fold "
                f"trip-macro top-1, with binary macro F1 and log loss as tie-breakers. HistGradientBoosting "
                f"leads the objective (0.8927 vs Bagging 0.8915) and also holds the best CV macro F1 "
                f"(0.9429 vs 0.9426); GaussianNB loses the log-loss tie-breaker, and KNN pays 187 s of "
                f"student inference where the selected tuned model pays 0.50 s. No student or evaluator "
                f"number entered the decision. Y-axes for the "
                f"rate panels start at the stated values; the inference panel is log-scaled across four orders "
                f"of magnitude. Source: results.json, model_comparison.csv, evaluation/*.json.",
                units="fractions 0-1 for rates; seconds for inference (log panel)",
                source="experiments/final_five/current/results.json; model_comparison.csv",
                ocr_expect=["objective", "macro", "inference"])


def _confusion_panel(ax, cm, title, counts_note):
    cm = np.asarray(cm, dtype=float)
    rates = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    labels = ["negative", "positive"]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j]):,}\n{rates[i, j] * 100:.2f}%",
                    ha="center", va="center", fontsize=12,
                    color="white" if rates[i, j] > 0.5 else "#111827")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([f"predicted\n{labels[0]}", f"predicted\n{labels[1]}"], fontsize=11)
    ax.set_yticklabels([f"true\n{labels[0]}", f"true\n{labels[1]}"], fontsize=11)
    ax.set_title(title, fontsize=12.5)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
    return im


def chart_confusions(view, meta):
    runs = view["runs"]
    caption = (f"Binary candidate-row confusion over all 10 student-development trips: 175,778 rows, 19,141 "
               f"positive, threshold 0.5. Percentages are row-normalized (per true class). Class imbalance keeps "
               f"raw row accuracy near 0.97 to 0.98 for every configuration, which is why accuracy alone is not "
               f"the headline number. The 19 observations whose proxy target road is absent from the candidate "
               f"set contribute no positive row and are counted separately as candidate misses. "
               f"{BINARY_NOTE} Source: evaluation/<run>.json.")
    for run in runs:
        fig, ax = plt.subplots(figsize=A4_HALF)
        _confusion_panel(ax, view["eval"][run]["student"]["confusion_matrix"],
                         f"Student confusion: {RUN_LABELS[run].replace(chr(10), ' ')}", "")
        save_figure(fig, f"final_five_confusion_{run}.png",
                    f"Student-development confusion matrix: {run} (binary candidate rows)",
                    caption, units="candidate-row counts; row-normalized percentages",
                    source=f"experiments/final_five/current/evaluation/{run}.json",
                    ocr_expect=["confusion", "predicted", "true"])
    fig, axs = plt.subplots(3, 2, figsize=A4_FULL)
    for ax, run in zip(axs.flat, runs):
        _confusion_panel(ax, view["eval"][run]["student"]["confusion_matrix"],
                         RUN_LABELS[run].replace(chr(10), " "), "")
    save_figure(fig, "final_five_confusion_matrices_all.png",
                "All six configurations: student-development confusion matrices (3x2 grid)",
                f"Five model families plus the tuned HistGradientBoosting variant make six measured "
                f"configurations; this grid shows all six so the tuned run is never confused with a sixth "
                f"family. {caption}",
                units="candidate-row counts; row-normalized percentages",
                source="experiments/final_five/current/evaluation/*.json",
                ocr_expect=["confusion", "predicted"])


def chart_fold_stability(view, meta):
    runs = view["runs"]
    fig, axs = plt.subplots(2, 1, figsize=A4_HALF, sharex=True)
    for ax, key, label in ((axs[0], "macro_f1", "Binary macro F1"),
                           (axs[1], "top1_accuracy", "Top-1 accuracy")):
        for i, run in enumerate(runs):
            fold_vals = [f["metrics"][key] for f in view["cv"][run]["folds"]]
            mean = float(np.mean(fold_vals))
            ax.scatter([i] * len(fold_vals), fold_vals, s=52, color=RUN_COLORS[run], zorder=4,
                       edgecolor="white", linewidth=0.8)
            ax.hlines(mean, i - 0.28, i + 0.28, color="#111827", linewidth=2.2, zorder=5)
            ax.annotate(f"{mean:.4f}", (i, mean), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=10)
        ax.set_ylim(0.80, 0.97)
        ax.set_ylabel(f"{label} (0-1)")
        ax.set_title(f"Per-fold {label.lower()} across the 3 grouped folds", fontsize=12)
        ax.grid(axis="y", alpha=0.5)
    _run_axes(axs[1], runs)
    save_figure(fig, "final_five_fold_stability.png",
                "Fold stability: every fold of the 3-fold grouped CV, per configuration",
                f"Each dot is one GroupKFold fold (27, 27 and 26 validation trips). Black bars join the fold "
                f"mean. Fold 2 is the hardest fold for every configuration, which moves the fold spread more "
                f"than the choice of model does; fold standard deviations of macro F1 range from 0.007 to "
                f"0.009. The tuned HistGB run is not more stable than the default (macro F1 fold SD 0.0094 vs "
                f"0.0075). Y-axes start at 0.80; the metric range is 0 to 1. Source: cv/*.json folds.",
                units="fractions 0-1 (per-fold binary macro F1; per-fold top-1 accuracy)",
                source="experiments/final_five/current/cv/*.json",
                ocr_expect=["fold", "stability"])


def chart_generalization_gap(view, meta):
    runs = view["runs"]
    fig, ax = plt.subplots(figsize=A4_HALF)
    x = np.arange(len(runs))
    resub = [view["eval"][r]["resub"]["macro_f1"] for r in runs]
    cvm = [view["cv"][r]["oof"]["macro_f1"] for r in runs]
    stum = [view["eval"][r]["student"]["macro_f1"] for r in runs]
    w = 0.26
    ax.bar(x - w, resub, w, color="#94a3b8", label="Train resubstitution")
    b2 = ax.bar(x, cvm, w, color="#2266b2", label="CV out-of-fold (TRAIN)")
    b3 = ax.bar(x + w, stum, w, color="#d77926", label="Student-development")
    for bars in (b2, b3):
        label_bars(ax, bars, fmt="{:.4f}", fontsize=9, offset=3)
    _run_axes(ax, runs)
    ax.set_ylim(0.88, 1.00)
    ax.set_ylabel("Binary macro F1 (0-1)")
    ax.set_title("Generalization: resubstitution, CV, and student macro F1")
    ax.legend(loc="lower left", fontsize=10)
    save_figure(fig, "final_five_generalization_gap.png",
                "Generalization gap: train resubstitution vs CV vs student macro F1",
                f"Binary macro F1 on three partitions: refit on all 80 TRAIN trips and rescored on TRAIN "
                f"(optimistic by construction), out-of-fold CV on the same 80 trips, and the 10 "
                f"student-development trips scored after selection. Resubstitution sits above CV by between "
                f"0.000 (GaussianNB, little capacity to overfit) and 0.029 (Bagging, 40 deep trees). Student "
                f"stays within 0.004 of CV for every configuration, so no configuration collapses on the "
                f"student partition. {BINARY_NOTE} Y-axis starts at 0.88. Source: evaluation/*.json, cv/*.json.",
                units="fraction 0-1 (binary macro F1)",
                source="experiments/final_five/current/evaluation/*.json; cv/*.json",
                ocr_expect=["generalization", "resubstitution", "student"])


def chart_timing(view, meta):
    runs = view["runs"]
    fig, axs = plt.subplots(2, 2, figsize=A4_HALF)
    panels = [
        ("CV training, 3 folds (s)", [view["cv"][r]["training_seconds_total"] for r in runs]),
        ("CV inference, 3 folds (s)", [view["cv"][r]["inference_seconds_total"] for r in runs]),
        ("Final fit, all 80 trips (s)", [view["eval"][r]["training_seconds"] for r in runs]),
        ("Student inference, 10 trips (s)", [view["eval"][r]["student_inference_seconds"] for r in runs]),
    ]
    for ax, (title, values) in zip(axs.flat, panels):
        bars = ax.bar(range(len(runs)), [max(v, 0.001) for v in values], 0.62,
                      color=[RUN_COLORS[r] for r in runs])
        label_bars(ax, bars, values=values, fmt="{:.4g} s", fontsize=9, offset=3)
        _run_axes(ax, runs)
        ax.set_yscale("log")
        ax.set_ylim(0.01, 700)
        ax.set_title(title, fontsize=11.5)
        ax.tick_params(axis="x", labelsize=8.5)
    save_figure(fig, "final_five_timing.png",
                "Measured compute cost per configuration (log scale, CPU only)",
                f"Wall-clock seconds as measured by the pipeline on CPU; no GPU was used or claimed. KNN spends "
                f"almost nothing on fitting (it stores a reference bank) but needs 299 s of CV inference and "
                f"187.1 s on the 10 student trips, against 0.50 s for the selected tuned HistGB (roughly 375x "
                f"faster on student inference). The tuned final artifact was refit on all 80 trips in 4.40 s. "
                f"Log scale spans five orders of magnitude; every bar is labelled with its measured value. "
                f"Source: cv/*.json, evaluation/*.json, model_comparison.csv.",
                units="seconds (wall-clock, log-scaled panels)",
                source="experiments/final_five/current/cv/*.json; evaluation/*.json; model_comparison.csv",
                ocr_expect=["seconds", "inference", "training"])


def chart_tuned_vs_untuned(view, meta):
    default, tuned = "hist_gradient_boosting", FINAL_RUN
    panels = [
        ("CV trip-macro top-1\n(objective)", view["cv"][default]["oof"]["trip_macro_top1_accuracy"],
         view["cv"][tuned]["oof"]["trip_macro_top1_accuracy"], "{:.4f}", "tuned wins"),
        ("CV binary macro F1", view["cv"][default]["oof"]["macro_f1"],
         view["cv"][tuned]["oof"]["macro_f1"], "{:.4f}", "default wins"),
        ("CV log loss (lower better)", view["cv"][default]["oof"]["log_loss"],
         view["cv"][tuned]["oof"]["log_loss"], "{:.5f}", "default wins"),
        ("Student top-1 (pooled)", view["eval"][default]["student"]["top1_accuracy"],
         view["eval"][tuned]["student"]["top1_accuracy"], "{:.4f}", "tuned wins"),
        ("Student binary macro F1", view["eval"][default]["student"]["macro_f1"],
         view["eval"][tuned]["student"]["macro_f1"], "{:.4f}", "tuned wins"),
        ("Student log loss (lower better)", view["eval"][default]["student"]["log_loss"],
         view["eval"][tuned]["student"]["log_loss"], "{:.5f}", "default wins"),
    ]
    fig, axs = plt.subplots(2, 3, figsize=A4_HALF)
    for ax, (title, d, t, fmt, verdict) in zip(axs.flat, panels):
        bars = ax.bar([0, 1], [d, t], 0.55, color=["#ee6677", "#aa3377"])
        label_bars(ax, bars, fmt=fmt, fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["default", "tuned"], fontsize=11)
        lo, hi = min(d, t), max(d, t)
        pad = (hi - lo) * 3 + 1e-4
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{title}\n{verdict}", fontsize=10.5)
        ax.grid(axis="y", alpha=0.5)
    save_figure(fig, "final_five_tuned_vs_untuned.png",
                "Tuned vs default HistGradientBoosting on identical partitions: different metrics, different winners",
                f"Same 3-fold CV and the same student partition; only the parameters differ (tuned: max_iter 104, "
                f"learning_rate 0.159, max_leaf_nodes 22, min_samples_leaf 38, l2_regularization 0.86; from "
                f"Bayesian trial 5). The tuned run wins the recorded selection objective (CV trip-macro top-1 "
                f"0.8934 vs 0.8927) and both student accuracy panels, but CV binary macro F1 declines (0.9416 vs "
                f"0.9429) and student log loss is marginally worse (0.05550 vs 0.05522). The tuned run was kept "
                f"because the recorded objective is top-1; these small contrary movements are reported rather "
                f"than hidden. Panel y-axes are zoomed to the pair of values; read the printed numbers. "
                f"Source: cv/hist_gradient_boosting*.json, evaluation/hist_gradient_boosting*.json.",
                units="fractions 0-1 (accuracies, F1); nats (log loss)",
                source="experiments/final_five/current/cv/hist_gradient_boosting*.json; "
                       "evaluation/hist_gradient_boosting*.json",
                ocr_expect=["tuned", "default", "wins"])


def chart_hpo_progress(view, meta):
    trials = view["hpo"]["trials"]
    fig, ax = plt.subplots(figsize=A4_HALF)
    cat_style = {
        "measured_baseline_prior": ("#111827", "s", "measured baseline prior (trial 0)"),
        "seeded_initial_design": ("#4477aa", "^", "seeded initial design (trials 1-3)"),
        "gaussian_process_expected_improvement": ("#d77926", "o", "GP Matern-5/2 + expected improvement (trials 4-11)"),
    }
    for cat, (color, marker, label) in cat_style.items():
        pts = [t for t in trials if t["proposal"] == cat]
        if pts:
            ax.scatter([t["trial"] for t in pts], [t["objective"] for t in pts],
                       s=90, color=color, marker=marker, zorder=4, label=label)
    best = view["hpo"]["best"]
    ax.scatter([best["trial"]], [best["objective"]], s=260, facecolor="none",
               edgecolor="#7a1020", linewidth=2.4, zorder=5, label=f"best trial ({best['trial']})")
    ax.axhline(view["hpo"]["baseline_objective"], color="#111827", linestyle="--",
               linewidth=1.2, label=f"baseline objective {view['hpo']['baseline_objective']:.4f}")
    for t in trials:
        ax.annotate(f"{t['objective']:.4f}", (t["trial"], t["objective"]), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=8.5, rotation=45)
    ax.set_xlabel("Trial index")
    ax.set_ylabel("Observed objective: CV trip-macro top-1 (0-1)")
    ax.set_title("Bayesian HPO progress: 12 measured trials on the TRAIN objective")
    ax.set_ylim(0.8885, 0.8945)
    ax.set_xticks(range(len(trials)))
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.5)
    save_figure(fig, "final_five_hpo_progress.png",
                "Bayesian HPO progress: observed objective across 12 trials",
                f"Objective: out-of-fold trip-macro top-1 on the 80 TRAIN trips; the search never loaded "
                f"student or evaluator data. Trial 0 reuses the measured default-parameter baseline; trials "
                f"1-3 form the seeded initial design; trials 4-11 are Gaussian-process (Matern-5/2) proposals "
                f"chosen by expected improvement. The best observed objective is trial 5 (0.8934, ringed), "
                f"0.0007 above the baseline; the whole search spans just over 0.004, so the tuning gain is "
                f"real but small. These are observed objective values from the retained trial log, not smoothed "
                f"or predicted curves. Y-axis spans 0.006 of the 0-1 range to keep differences legible. "
                f"Source: bayesian_optimization.json, experiments/bayesian_trials.jsonl.",
                units="fraction 0-1 (CV trip-macro top-1 objective)",
                source="experiments/final_five/current/bayesian_optimization.json; "
                       "experiments/bayesian_trials.jsonl",
                ocr_expect=["trial", "objective", "baseline"])


def chart_hpo_parameters(view, meta):
    trials = view["hpo"]["trials"]
    best = view["hpo"]["best"]
    params = [("learning_rate", "learning rate (log scale)", True),
              ("max_iter", "max_iter (trees)", False),
              ("max_leaf_nodes", "max_leaf_nodes", False),
              ("min_samples_leaf", "min_samples_leaf (log scale)", True),
              ("l2_regularization", "l2_regularization (log scale)", True)]
    fig, axs = plt.subplots(1, 5, figsize=(A4_HALF[0] * 1.9, A4_HALF[1] * 0.92))
    objs = np.array([t["objective"] for t in trials])
    norm = (objs - objs.min()) / (objs.max() - objs.min() + 1e-12)
    cmap = plt.get_cmap("viridis")
    for ax, (key, label, logy) in zip(axs, params):
        vals = [t["parameters"][key] for t in trials]
        sc = ax.scatter(range(len(trials)), vals, c=norm, cmap=cmap, s=95, zorder=4,
                        vmin=0, vmax=1, edgecolor="white", linewidth=0.8)
        ax.scatter([best["trial"]], [best["parameters"][key]], s=260, facecolor="none",
                   edgecolor="#7a1020", linewidth=2.2, zorder=5)
        ax.set_title(label, fontsize=10.5)
        ax.set_xlabel("trial", fontsize=10)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.5)
        ax.tick_params(labelsize=9)
    cbar = fig.colorbar(sc, ax=axs, orientation="horizontal", fraction=0.045, pad=0.16)
    cbar.set_label("observed objective (dark = better)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    save_figure(fig, "final_five_hpo_parameters.png",
                "Observed HPO parameter values per trial (no causal importance claimed)",
                f"Sampled hyperparameter values for each of the 12 Bayesian trials, coloured by the measured "
                f"objective (darker is better); the best trial is ringed. These are observed samples from the "
                f"retained search record. They are not a causal importance or sensitivity analysis: with 12 "
                f"correlated trials chosen by the acquisition rule, parameter-objective patterns cannot support "
                f"effect claims, and none are made here. Search space: learning_rate log 0.025-0.2, max_iter "
                f"70-200, max_leaf_nodes 7-47, min_samples_leaf log-int 15-160, l2_regularization log 0.01-20. "
                f"Source: bayesian_optimization.json.",
                units="parameter values in native units per panel; trial index 0-11",
                source="experiments/final_five/current/bayesian_optimization.json",
                ocr_expect=["trial", "learning", "parameter"])


def chart_observation_metrics(view, meta):
    runs = view["runs"]
    seq = view["seq"]
    fig, axs = plt.subplots(1, 2, figsize=(A4_HALF[0], A4_HALF[1] * 0.92))
    x = np.arange(len(runs))
    pooled = [seq[r]["observation_edge_macro_f1"] for r in runs]
    trip = [seq[r]["trip_macro_observation_edge_f1"] for r in runs]
    w = 0.36
    axs[0].bar(x - w / 2, pooled, w, color=[RUN_COLORS[r] for r in runs], label="pooled")
    b2 = axs[0].bar(x + w / 2, trip, w, color=[RUN_COLORS[r] for r in runs], alpha=0.45,
                    hatch="//", label="trip-macro (mean of 10)")
    label_bars(axs[0], b2, fmt="{:.4f}", fontsize=9, offset=3)
    _run_axes(axs[0], runs)
    axs[0].tick_params(axis="x", labelsize=8)
    axs[0].set_ylim(0.75, 0.90)
    axs[0].set_ylabel("Observation edge macro F1 (0-1)")
    axs[0].set_title("Directed-edge macro F1 per observation", fontsize=12)
    axs[0].legend(fontsize=9)
    nll = [seq[r]["observation_categorical_log_loss_clipped"] for r in runs]
    bars = axs[1].bar(x, nll, 0.62, color=[RUN_COLORS[r] for r in runs])
    label_bars(axs[1], bars, fmt="{:.4f}", fontsize=9)
    _run_axes(axs[1], runs)
    axs[1].tick_params(axis="x", labelsize=8)
    axs[1].set_ylim(0, 1.05)
    axs[1].set_ylabel("Categorical NLL per observation (nats)")
    axs[1].set_title("Normalized matching-candidate NLL", fontsize=12)
    save_figure(fig, "final_five_observation_edge_metrics.png",
                "Observation-level edge metrics on the student partition",
                f"Left: graph-namespaced directed-edge macro F1; classes are trip:arc identifiers including "
                f"reference-only and prediction-only roads, macro-averaged over observed classes. Right: "
                f"per-observation normalized categorical negative log-likelihood of the candidate matching the "
                f"proxy target (probabilities normalized within each observation, floor 1e-7, the 19 candidate "
                f"misses retained with zero target mass). Both are observation-level metrics and are "
                f"deliberately distinct from the binary candidate-row F1 shown elsewhere. The tuned HistGB run "
                f"has the highest edge F1 and the lowest categorical NLL of the six configurations. Left "
                f"y-axis starts at 0.75. Source: sequence_classification.json.",
                units="fractions 0-1 (edge macro F1); nats per observation (NLL)",
                source="experiments/final_five/current/sequence_classification.json",
                ocr_expect=["edge", "observation", "nll"])


def chart_continuity(view, meta):
    runs = view["runs"]
    seq = view["seq"]
    fig, axs = plt.subplots(1, 2, figsize=(A4_HALF[0], A4_HALF[1] * 0.92))
    x = np.arange(len(runs))
    raw = [seq[r]["raw_classifier_continuity"] for r in runs]
    ec = [seq[r]["edge_change_continuity"] for r in runs]
    for ax, values, title, ylab in (
            (axs[0], raw, "All adjacent pairs", "Raw classifier continuity (0-1)"),
            (axs[1], ec, "Pairs where the edge changes", "Edge-change continuity (0-1)")):
        bars = ax.bar(x, values, 0.62, color=[RUN_COLORS[r] for r in runs])
        label_bars(ax, values=values, bars=bars, fmt="{:.4f}", fontsize=9)
        _run_axes(ax, runs)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylim(0.60, 1.00)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=12)
    save_figure(fig, "final_five_continuity.png",
                "Raw classifier continuity on the student partition (no decoder applied)",
                f"Continuity is scored on adjacent available observation assignments of the raw classifier "
                f"argmax: a pair is continuous if it stays on the same canonical edge or the head of the "
                f"previous edge equals the tail of the next; no routed connectors are inserted and observed "
                f"time gaps are included. Right panel restricts to the 4,620 to 5,077 pairs where the edge "
                f"changes, the stricter test. This is the classifier-only baseline that the frozen decoder "
                f"ablation (classifier + HMM) will be compared against once its report exists. Y-axes start "
                f"at 0.60. Source: sequence_classification.json.",
                units="fractions 0-1 (continuity rates over adjacent observation pairs)",
                source="experiments/final_five/current/sequence_classification.json",
                ocr_expect=["continuity", "edge"])


def chart_denominators(view, meta):
    e = REAL_EXPECTED
    fig, axs = plt.subplots(1, 2, figsize=(A4_HALF[0], A4_HALF[1] * 0.92))
    groups = ["observations", "candidate rows", "positive rows"]
    cv_vals = [e["train_observations"], e["train_candidate_rows"], e["train_positive_rows"]]
    st_vals = [e["student_observations"], e["student_candidate_rows"], e["student_positive_rows"]]
    x = np.arange(3)
    w = 0.36
    b1 = axs[0].bar(x - w / 2, cv_vals, w, color="#2266b2", label="TRAIN CV (80 trips)")
    b2 = axs[0].bar(x + w / 2, st_vals, w, color="#d77926", label="Student (10 trips)")
    for bars, vals in ((b1, cv_vals), (b2, st_vals)):
        label_bars(axs[0], bars, values=vals, fmt="{:,.0f}", fontsize=9, offset=3)
    axs[0].set_yscale("log")
    axs[0].set_ylim(1000, 1.5e6)
    axs[0].set_xticks(x); axs[0].set_xticklabels(groups)
    axs[0].set_ylabel("Count (log scale)")
    axs[0].set_title("Counting denominators per partition", fontsize=12)
    axs[0].legend(fontsize=9)
    miss_cv, miss_st = e["train_candidate_miss_count"], e["student_candidate_miss_count"]
    bars = axs[1].bar([0, 1], [miss_cv, miss_st], 0.5, color=["#2266b2", "#d77926"])
    axs[1].set_xticks([0, 1])
    axs[1].set_xticklabels([f"TRAIN CV\n{miss_cv}/{e['train_observations']} obs\n(2.06%)",
                            f"Student\n{miss_st}/{e['student_observations']} obs\n(0.099%)"])
    for bar, v in zip(bars, (miss_cv, miss_st)):
        axs[1].annotate(f"{v}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=12)
    axs[1].set_ylim(0, 700)
    axs[1].set_ylabel("Proxy-target candidate misses (count)")
    axs[1].set_title("Observations whose target road\nwas absent from the candidate set", fontsize=12)
    save_figure(fig, "final_five_denominators.png",
                "What each metric divides: observations, candidate rows, and proxy-target misses",
                f"Left: observations, candidate rows and positive rows per partition, log-scaled because "
                f"candidate rows outnumber positives about 9 to 1. Right: proxy-target candidate misses. "
                f"Candidate sets were never empty on either partition (zero missing candidate sets); in 578 of "
                f"28,000 TRAIN observations (2.06%) and 19 of 19,160 student observations (0.099%) the proxy "
                f"target road was not among the candidates, so those observations cannot be scored top-1 "
                f"correct no matter what the model does. Pooled top-1 divides by observations; trip-macro "
                f"divides equally by trips; on TRAIN every trip has exactly 350 observations so the two "
                f"coincide, on the student partition they do not. Source: cv/*.json oof_metrics, "
                f"evaluation/*.json student_metrics.",
                units="counts (rows, observations); miss rates in percent",
                source="experiments/final_five/current/cv/*.json; evaluation/*.json",
                ocr_expect=["denominators", "candidate", "misses"])


MODEL_CHARTS = (chart_cv_macro_f1, chart_primary_objective, chart_student_metrics,
                chart_log_loss, chart_log_loss_partitions, chart_selection_comparison,
                chart_confusions, chart_fold_stability, chart_generalization_gap,
                chart_timing, chart_tuned_vs_untuned, chart_hpo_progress,
                chart_hpo_parameters, chart_observation_metrics, chart_continuity,
                chart_denominators)


# ------------------------------------------------------------------ decoder stage

def generate_decoder_chart(ablation_path: Path) -> list[dict]:
    """Route/HMM ablation chart from the decoder child's report, when it exists."""
    if not ablation_path.exists():
        return []
    ab, digest = load_json_with_hash(ablation_path)
    check(not ab.get("failed_trip_ids"),
          f"decoder ablation reports failed trips {ab.get('failed_trip_ids')}; refusing to chart incomplete evidence")
    check(ab.get("trip_count") == REAL_EXPECTED["student_trips"],
          f"decoder ablation covers {ab.get('trip_count')} trips, expected 10")
    aggregate = ab["aggregate"]
    methods = list(aggregate.keys())
    titles = {"classifier_only_raw": "Classifier only\n(raw argmax)",
              "classifier_plus_HMM": "Classifier + HMM\n(Viterbi)",
              "geometry_HMM_default_reused": "Geometry-only HMM\n(historical, reused)"}
    metrics = [("route_edge_f1", "Route arc-multiset F1"), ("route_length_f1", "Route length F1 (m)"),
               ("route_edit_similarity", "Ordered edit similarity"), ("route_continuity", "Route continuity (1/jumps)")]
    fig, axs = plt.subplots(1, 4, figsize=(A4_HALF[0] * 1.9, A4_HALF[1] * 0.85))
    x = np.arange(len(methods))
    for ax, (metric, title) in zip(axs, metrics):
        values = [aggregate[m]["trip_macro"][metric] for m in methods]
        bars = ax.bar([titles.get(m, m) for m in methods], values, 0.6,
                      color=["#94a3b8", "#2266b2", "#d77926"][:len(methods)])
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=11)
        label_bars(ax, bars, values=values, fmt="{:.4f}", fontsize=10)
        ax.tick_params(axis="x", labelsize=8.5)
    denominator = aggregate[methods[0]].get("trip_macro_denominator", methods and "n/a")
    save_figure(fig, "final_five_route_ablation.png",
                "Final-five decoder ablation: classifier-only vs classifier+HMM",
                f"Decoder evidence from {rel(ablation_path)} on all 10 student-development trips; trip-macro "
                f"denominator {denominator} trips. Metrics are computed over chronological arc-ID traversals "
                f"(repeats and loops preserved) directly against the supplied reference arc traversals. "
                f"The paired decoder experiment uses the same effective candidates for classifier-only "
                f"and Viterbi decoding; observation-level proxy labels are a separate measure. "
                f"Routes and metrics come from the frozen decoder ablation, not from this plotter. "
                f"Source: {rel(ablation_path)} "
                f"(sha256 {digest[:12]}...).",
                units="fractions 0-1 (route-level F1, similarity, continuity)",
                source=rel(ablation_path), stage="decoder",
                ocr_expect=["ablation", "route"])
    return REGISTRY[-1:]


# ------------------------------------------------------------------ manifest

def reused_eda_records() -> tuple[list[dict], list[str]]:
    """Re-pin the 12 completed EDA plots read-only; never edit or regenerate them."""
    source_manifest = ROOT / "visualizations" / "phase_three" / "figure_manifest.json"
    records, problems = [], []
    data = json.loads(source_manifest.read_text(encoding="utf-8"))
    for entry in data.get("eda_reused_read_only", []):
        path = ROOT / entry["path"]
        if not path.exists():
            problems.append(f"missing EDA asset {entry['path']}")
            continue
        current = sha256_file_path(path)
        if current != entry.get("sha256"):
            problems.append(f"EDA asset hash drift for {entry['path']}")
            continue
        rec = dict(entry)
        rec["reused_read_only"] = True
        rec["origin_manifest"] = rel(source_manifest)
        rec["note"] = ("completed phase-one EDA figure, reused read-only for the final report; "
                       "path and source hash unchanged, no edits")
        records.append(rec)
    return records, problems


def preserved_records(mode: str, inputs: dict, previous: dict | None) -> tuple[list[dict], dict, list[str]]:
    """Preserve an untouched stage only after file/DPI/input checks succeed."""
    if mode == "all" or not previous:
        return [], {}, []
    stage = "models" if mode == "decoder" else "decoder"
    records = [r for r in previous.get("figures", []) if r.get("stage") == stage]
    if not records:
        return [], {}, []
    try:
        names = [Path(r["path"]).name for r in records]
        check(len(names) == len(set(names)) and set(names) == EXPECTED_BY_STAGE[stage],
              f"incomplete preserved {stage} figure set")
        old_inputs = previous.get("inputs", {})
        check(bool(old_inputs), "preserved figures have no input pins")
        verify_input_pins(old_inputs, ROOT)
        for key, value in inputs.items():
            if not key.startswith("decoder_"):
                check(old_inputs.get(key) == value, f"preserved figure input changed: {key}")
        if stage == "decoder":
            check("decoder_report" in old_inputs and "decoder_source" in old_inputs,
                  "preserved decoder chart lacks report/source pins")
            for key in ("decoder_report", "decoder_source"):
                check(inputs.get(key) == old_inputs[key], f"preserved decoder input changed: {key}")
        for record in records:
            verify_record_integrity(record, ROOT)
        return records, {k: v for k, v in old_inputs.items() if k.startswith("decoder_")}, []
    except (KeyError, OSError, ValueError) as exc:
        message = f"Cannot preserve {stage} figures: {exc}; regenerate with --mode all"
        if stage == "models":
            raise ValueError(message) from exc
        return [], {}, [message]


def build_manifest(mode: str, inputs: dict, meta: dict, decoder_status: dict,
                   preserved: list[dict] = ()) -> dict:
    figures = list(preserved) + list(REGISTRY)
    check(len({r["path"] for r in figures}) == len(figures), "duplicate figure paths")
    model_figures = [r for r in figures if r["stage"] == "models"]
    decoder_figures = [r for r in figures if r["stage"] == "decoder"]
    complete = all({Path(r["path"]).name for r in figures if r["stage"] == stage} == names
                   for stage, names in EXPECTED_BY_STAGE.items())
    if decoder_figures:
        decoder_status = {"status": "charted", "report": inputs["decoder_report"]["path"],
                          "report_sha256": inputs["decoder_report"]["sha256"],
                          "figures": [r["path"] for r in decoder_figures]}
    eda_records, eda_problems = reused_eda_records()
    if eda_problems:
        raise ValueError("EDA read-only pin check failed: " + "; ".join(eda_problems))
    manifest = {
        "phase": "five-model-completion",
        "generated_utc": utc_now(),
        "generator": rel(FIVE_DIR / "generate_final_five.py"),
        "generator_sha256": sha256_file_path(FIVE_DIR / "generate_final_five.py"),
        "mode": mode,
        "scope_guard": "all writes confined to visualizations/final_five/",
        "families": list(FAMILIES),
        "runs": list(RUNS),
        "run_note": ("five model families plus the tuned variant of the selected family "
                     "(hist_gradient_boosting_tuned) = six measured configurations"),
        "selected_model": {
            "family": SELECTED_FAMILY,
            "final_run": FINAL_RUN,
            "model_sha256": SELECTED_SHA256,
            "selected_by": "TRAIN out-of-fold trip-macro top-1 (selection.json); student/evaluator never used",
            "final_fit": "tuned run refitted last on all 80 TRAIN trips, fresh 4.40 s, CPU only (no GPU claim)",
        },
        "inputs": inputs,
        "figures": figures,
        "figure_count": len(figures),
        "model_figure_count": len(model_figures),
        "decoder_figure_count": len(decoder_figures),
        "expected_figure_counts": {"models": 22, "decoder": 1, "total": 23},
        "eda_reused_read_only": eda_records,
        "epoch_wise_loss": {
            "status": "N/A for all five families",
            "detail": EPOCH_NA_NOTE,
            "per_family": {f: "N/A - no epoch-wise training history retained by this sklearn estimator; "
                              "no curves fabricated (docs/FINAL_REQUIREMENTS.md)" for f in FAMILIES},
            "substitute_figure": "final_five_log_loss_partitions.png",
        },
        "decoder": decoder_status,
        "partial": not complete,
        "verification": "run visualizations/final_five/verify_final_figures.py (OCR + integrity)",
        "notes": [
            "Every plotted value is read from the measured result files listed under inputs.",
            "Binary macro F1, positive-class F1, and observation-level edge F1 are distinct metrics; "
            "each figure states which one it shows.",
            "HPO parameter figures show observed samples only; no causal importance claim is made.",
            BINARY_NOTE,
            PROXY_NOTE,
        ],
    }
    manifest.update({k: v for k, v in meta.items() if k not in manifest})
    path = safe_output(MANIFEST)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    return manifest


def load_inputs(results_dir: Path) -> tuple[dict, dict]:
    """Load and hash every measured input; returns (data, manifest inputs block)."""
    inputs: dict = {}
    paths = {
        "results": results_dir / "results.json",
        "model_comparison_csv": results_dir / "model_comparison.csv",
        "selection": results_dir / "selection.json",
        "sequence_classification": results_dir / "sequence_classification.json",
        "bayesian_optimization": results_dir / "bayesian_optimization.json",
    }
    data: dict = {}
    for key, p in paths.items():
        check(p.exists(), f"missing required input {rel(p)}")
        raw = p.read_bytes()
        data[key] = (json.loads(raw.decode("utf-8")) if p.suffix == ".json"
                     else list(csv.DictReader(raw.decode("utf-8").splitlines())))
        inputs[key] = {"path": rel(p), "sha256": sha256_bytes(raw)}
    cv_files, eval_files = {}, {}
    for run in RUNS:
        p = results_dir / "cv" / f"{run}.json"
        check(p.exists(), f"missing CV file {rel(p)}")
        raw = p.read_bytes()
        cv_files[run] = json.loads(raw.decode("utf-8"))
        inputs.setdefault("cv", {})[run] = {"path": rel(p), "sha256": sha256_bytes(raw)}
        p = results_dir / "evaluation" / f"{run}.json"
        check(p.exists(), f"missing evaluation file {rel(p)}")
        raw = p.read_bytes()
        eval_files[run] = json.loads(raw.decode("utf-8"))
        inputs.setdefault("evaluation", {})[run] = {"path": rel(p), "sha256": sha256_bytes(raw)}
    trial_log = ROOT / "experiments" / "bayesian_trials.jsonl"
    check(trial_log.exists(), "missing experiments/bayesian_trials.jsonl")
    inputs["hpo_trial_log"] = {"path": rel(trial_log), "sha256": sha256_file_path(trial_log)}
    acceptance = ROOT / "reports" / "final_five" / "all_five_model_acceptance.json"
    check(acceptance.exists(), "missing all_five_model_acceptance.json")
    acc_raw = acceptance.read_bytes()
    inputs["model_acceptance"] = {"path": rel(acceptance), "sha256": sha256_bytes(acc_raw)}
    acceptance = json.loads(acc_raw.decode("utf-8"))
    check(acceptance.get("passed") is True, "all_five_model_acceptance.json did not pass")
    check(acceptance.get("selected_model_sha256") == SELECTED_SHA256,
          "acceptance report model sha does not match the frozen artifact")
    clean_load = ROOT / "artifacts" / "final_five" / "clean_load_report.json"
    if clean_load.exists():
        raw = clean_load.read_bytes()
        inputs["clean_load_report"] = {"path": rel(clean_load), "sha256": sha256_bytes(raw)}
        clean = json.loads(raw.decode("utf-8"))
        check(clean.get("passed") is True, "clean_load_report.json did not pass")
        check(clean.get("model_sha256") == SELECTED_SHA256,
              "clean-load report model sha does not match the frozen artifact")
    eda_manifest = ROOT / "visualizations" / "phase_three" / "figure_manifest.json"
    inputs["eda_pin_source"] = {"path": rel(eda_manifest),
                                "sha256": sha256_file_path(eda_manifest)}
    for key, path in (("figure_source", FIVE_DIR / "generate_final_five.py"),
                      ("figure_integrity_source", FIVE_DIR / "figure_integrity.py")):
        inputs[key] = {"path": rel(path), "sha256": sha256_file_path(path)}
    return {"results": data["results"], "cv_files": cv_files, "eval_files": eval_files,
            "seq": data["sequence_classification"], "selection": data["selection"]}, inputs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("models", "decoder", "all"), default="models",
                        help="models: the 22 model figures (default); decoder: route/HMM ablation "
                             "figures if the decoder report exists; all: both")
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "experiments" / "final_five" / "current")
    parser.add_argument("--decoder-report", type=Path,
                        default=ROOT / "reports" / "final_five" / "decoder" / "student_ablation.json")
    args = parser.parse_args(argv)

    REGISTRY.clear()
    loaded, inputs = load_inputs(args.results_dir.resolve())
    if args.decoder_report.is_file():
        for key, path in (("decoder_report", args.decoder_report),
                          ("decoder_source", ROOT / "scripts" / "final_five_decoder.py")):
            inputs[key] = {"path": rel(path), "sha256": sha256_file_path(path)}
    previous = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else None
    preserved, preserved_inputs, warnings = preserved_records(args.mode, inputs, previous)
    inputs.update(preserved_inputs)
    for warning in warnings:
        print(warning)
    view = prepare(loaded["results"], loaded["cv_files"], loaded["eval_files"],
                   loaded["seq"], loaded["selection"], REAL_EXPECTED)
    print(f"validation passed: {len(view['runs'])} runs, "
          f"{len(view['hpo']['trials'])} HPO trials, selected {view['freeze']['run_name']}")

    if args.mode in ("models", "all"):
        for chart in MODEL_CHARTS:
            chart(view, inputs)
            print(f"  charted {REGISTRY[-1]['path']}")

    decoder_status = {"status": "await_route",
                      "note": ("decoder child's route/HMM ablation report is not present yet; "
                               "the Lead should rerun with --mode decoder (or all) once "
                               "reports/final_five/decoder/student_ablation.json exists"),
                      "expected_report": rel(args.decoder_report)}
    if args.mode in ("decoder", "all"):
        if args.decoder_report.exists():
            generate_decoder_chart(args.decoder_report.resolve())
            decoder_status = {"status": "charted",
                              "report": rel(args.decoder_report),
                              "report_sha256": sha256_file_path(args.decoder_report),
                              "figures": [r["path"] for r in REGISTRY if r["stage"] == "decoder"]}
        else:
            print(f"decoder report absent ({rel(args.decoder_report)}); manifest records await_route")

    meta = {"completed_at_utc_source": view["results_completed_at"] if "results_completed_at" in view
            else loaded["results"].get("completed_at_utc"),
            "versions": loaded["results"].get("versions", {}),
            "student_role": loaded["results"].get("student_role", ""),
            "evaluator_status": loaded["results"].get("evaluator_status", "")}
    manifest = build_manifest(args.mode, inputs, meta, decoder_status, preserved)
    print(f"wrote {manifest['figure_count']} figure records "
          f"({manifest['model_figure_count']} model + {manifest['decoder_figure_count']} decoder) + "
          f"{len(manifest['eda_reused_read_only'])} re-pinned EDA records to {rel(MANIFEST)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
