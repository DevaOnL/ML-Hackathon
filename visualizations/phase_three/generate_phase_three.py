#!/usr/bin/env python3
"""Phase-three (three-model-first) chart and map generator.

Scope: writes ONLY under visualizations/phase_three/. Historical
visualizations/plots, visualizations/maps and the root
visualizations/figure_manifest.json are complete and are never touched;
existing assets are reused read-only.

Phase families (docs/PHASE_SCOPE.json, user-selected 18 Nov 2026):
logistic_regression, knn_distance_weighted, bagging — plus the tuned
variant of the selected family. GNB/HistGB/RF/ET are NOT this phase.

Inputs (produced by other owners; this generator never trains anything):
  - experiments/phase_three/results.json            (learning agent, task-16)
  - reports/phase_three/decoder/student_ablation.json (+ ablation_run.json)
                                                    (decoder agent, task-17)
  - outputs/phase_three_student_test/routes/00000009.route (Lead final CLI)

Every plotted number is read from those stable measured files; nothing is
invented, no epoch curves are fabricated for tree/NN-free families, and
chart scales are never manipulated to exaggerate tiny differences.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # single-thread numeric policy; explicit env still wins
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import argparse
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parent
ROOT = PHASE_DIR.parents[1]
sys.path.insert(0, str(PHASE_DIR.parent))

# Read-only reuse of the stable shared style/helpers. Importing common only
# configures matplotlib style; its write helpers are NOT used because they
# permit writes anywhere under visualizations/ (too broad for this phase).
from common import COLORS, PROXY_NOTE, SPLIT_LABELS, label_bars, read_json, sha256  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PHASE_OUT = PHASE_DIR
PHASE_PLOTS = PHASE_OUT / "plots"
PHASE_MAPS = PHASE_OUT / "maps"
PHASE_OCR = PHASE_OUT / "ocr"
PHASE_MANIFEST = PHASE_OUT / "figure_manifest.json"
PHASE_VERIFICATION = PHASE_OUT / "asset_verification.json"
PHASE_REGISTRY = PHASE_OUT / ".figure_registry.json"

DEFAULT_RESULTS = ROOT / "experiments" / "phase_three" / "results.json"
DEFAULT_ABLATION = ROOT / "reports" / "phase_three" / "decoder" / "student_ablation.json"
DEFAULT_ABLATION_RUN = ROOT / "reports" / "phase_three" / "decoder" / "ablation_run.json"
DEFAULT_PREDICTION = ROOT / "outputs" / "phase_three_student_test" / "routes" / "00000009.route"
DEFAULT_GRAPH_PREFIX = ROOT / "Student folder" / "student_test" / "00000009" / "00000009"
FROZEN_ARTIFACT = ROOT / "artifacts" / "phase_three" / "model.joblib"
VENDOR_LEAFLET = PHASE_DIR.parent / "vendor" / "leaflet"
ROOT_FIGURE_MANIFEST = PHASE_DIR.parent / "figure_manifest.json"
HISTORICAL_BASELINE_MAP = PHASE_DIR.parent / "maps" / "development_trip_00000009_baseline.png"
HISTORICAL_PRIOR_FINAL_MAP = PHASE_DIR.parent / "maps" / "development_trip_00000009_final.png"

PHASE_FAMILIES = ("logistic_regression", "knn_distance_weighted", "bagging")
REQUIRED_RESULT_KEYS = ("config", "versions", "target_policy", "selection_model", "final_run_name",
                        "cv", "evaluations", "comparison", "bayesian", "coverage_summary")
FAMILY_LABELS = {"logistic_regression": "Logistic\nregression",
                 "knn_distance_weighted": "KNN\n(distance-weighted)",
                 "bagging": "Bagging\n(decision trees)"}
BINARY_METRIC_NOTE = ("Binary candidate-row metric: one row per (observation, candidate) "
                      "directed arc at threshold 0.5 — not an observation or route metric.")
EDGE_METRIC_NOTE = ("Graph-namespaced directed-edge observation metric: classes are "
                     "trip_id:canonical_arc_id strings, macro-averaged over observed classes.")
NLL_NOTE = ("Per-observation normalized categorical NLL of the matching-candidate mass "
            "(floor 1e-7 on absent target mass; misses retained; vector not re-clipped).")


# ---------------------------------------------------------------- scope guard

def phase_safe_output(path: Path) -> Path:
    """Writes are forbidden outside visualizations/phase_three/."""
    path = path.resolve()
    if not path.is_relative_to(PHASE_OUT.resolve()):
        raise ValueError(f"Phase-three writes must stay in visualizations/phase_three/: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def phase_write_json(path: Path, data) -> None:
    phase_safe_output(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def phase_save_plot(fig, filename: str, title: str, note: str) -> Path:
    """Same visual layout contract as the historical save_plot, phase-scoped."""
    path = phase_safe_output(PHASE_PLOTS / filename)
    fig.suptitle(title, fontsize=16, fontweight="bold", x=.045, ha="left", y=.985)
    fig.text(.045, .018, note, fontsize=8, color="#475569", va="bottom")
    fig.tight_layout(rect=(.025, .075, .985, .935), h_pad=2.3, w_pad=2.0)
    fig.savefig(path, dpi=150, pil_kwargs={"compress_level": 8})
    plt.close(fig)
    register_figure({"file": filename, "title": title, "caption": note, "stage": "charts"})
    return path


def phase_write_text(path: Path, text: str) -> Path:
    path = phase_safe_output(path)
    path.write_text(text, encoding="utf-8")
    return path


def load_registry() -> list[dict]:
    """Persistent figure registry: survives separate stage invocations."""
    if PHASE_REGISTRY.exists():
        return json.loads(PHASE_REGISTRY.read_text(encoding="utf-8"))
    return []


def register_figure(entry: dict) -> None:
    """Idempotent upsert by filename so reruns replace rather than duplicate."""
    registry = [e for e in load_registry() if e["file"] != entry["file"]]
    registry.append(entry)
    phase_write_json(PHASE_REGISTRY, registry)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------- result loading

def load_atomic_json(path: Path):
    """Reject a file that changes while being read (wait for stable publication)."""
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise RuntimeError(f"{path} changed during read; wait for atomic final publication")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def attach_observation_supplement(r: dict, results_path: Path) -> dict:
    """Merge the published observation-metrics supplement into the in-memory view.

    The learning agent records observation-level metrics in a sibling file named
    by results.json["observation_metrics_supplement"] (sequence_classification.json):
    models[run]["student_test"] for the development split and
    models[run]["train_out_of_fold"] for the CV folds. The merge is in-memory
    only; results.json itself is never rewritten.
    """
    name = r.get("observation_metrics_supplement")
    if not name:
        raise ValueError("results.json lacks observation_metrics_supplement; wait for complete publication")
    path = (results_path.parent / name).resolve()
    if not path.is_file():
        raise ValueError(f"Observation-metrics supplement {name!r} not published yet")
    supplement, digest = load_atomic_json(path)
    models = supplement.get("models")
    if not isinstance(models, dict):
        raise ValueError("Observation-metrics supplement has no models block")
    expected = set(r["cv"]) | set(r["evaluations"])
    if not expected <= set(models):
        raise ValueError(f"Supplement missing runs {sorted(expected - set(models))}")
    for run, block in models.items():
        for split, target in (("student_test", "student_observation_metrics"),
                              ("train_out_of_fold", "observation_metrics")):
            if split not in block:
                raise ValueError(f"Supplement block for {run} missing {split}")
        if run in r["evaluations"]:
            r["evaluations"][run]["student_observation_metrics"] = block["student_test"]
        if run in r["cv"]:
            r["cv"][run]["observation_metrics"] = block["train_out_of_fold"]
    return {"path": str(path.relative_to(ROOT)), "sha256": digest,
            "target": supplement.get("target", "")}


def validate_phase_results(r: dict) -> dict:
    for key in REQUIRED_RESULT_KEYS:
        if key not in r:
            raise ValueError(f"Incomplete phase-three results: missing {key!r}; wait for PHASE_RESULTS_READY")
    cv, evaluations = r["cv"], r["evaluations"]
    missing = [f for f in PHASE_FAMILIES if f not in cv]
    if missing:
        raise ValueError(f"Phase-three results missing requested families {missing}; found {sorted(cv)}")
    extra = set(cv) - set(PHASE_FAMILIES)
    final = r["final_run_name"]
    if final not in cv:
        raise ValueError(f"final_run_name {final!r} has no CV record")
    if extra and extra != {final}:
        raise ValueError(f"Unexpected non-family CV entries {sorted(extra)}; phase covers only the three selected families plus the tuned variant")
    if set(cv) != set(evaluations):
        raise ValueError(f"CV and evaluation runs disagree: {sorted(cv)} vs {sorted(evaluations)}")
    trials = r["bayesian"].get("trials", [])
    if len(trials) != r["config"]["bayesian_trials"]:
        raise ValueError(f"Bayesian trial log incomplete: {len(trials)} of {r['config']['bayesian_trials']}")
    for name, evaluation in evaluations.items():
        if "student_observation_metrics" not in evaluation:
            raise ValueError(f"Observation-level supplement missing for {name}; wait for complete publication")
        for metric in ("observation_edge_macro_f1", "observation_categorical_log_loss_clipped"):
            if metric not in evaluation["student_observation_metrics"]:
                raise ValueError(f"Observation-level supplement for {name} missing {metric}")
        for block in ("student_metrics", "train_resubstitution_metrics"):
            if block not in evaluation:
                raise ValueError(f"{block} missing for {name}")
    for name in cv:
        for block in ("oof_metrics", "folds", "observation_metrics"):
            if block not in cv[name]:
                raise ValueError(f"CV record for {name} missing {block}")
    return {"runs": list(cv), "final_run_name": final, "selection_model": r["selection_model"]}


def run_label(name: str, final_run_name: str) -> str:
    base = name.removesuffix("_tuned")
    label = FAMILY_LABELS.get(base, base.replace("_", "\n"))
    return label + ("\n(tuned)" if name == final_run_name and name != base else "")


def ordered_runs(r: dict) -> list[str]:
    """Three families in declared order, then the tuned variant last."""
    runs = [f for f in PHASE_FAMILIES if f in r["cv"]]
    tuned = [n for n in r["cv"] if n not in PHASE_FAMILIES]
    return runs + sorted(tuned)


# ------------------------------------------------------------------- charts

def train_observations(cov: dict) -> int:
    """Observation count for the TRAIN split; coverage_summary nests it under 'train'."""
    block = cov.get("train") or {}
    return int(block.get("observations", cov.get("train_observations", 0)))


def student_observations(cov: dict) -> int:
    """Observation count for the student split; coverage_summary nests it under 'student_test'."""
    block = cov.get("student_test") or {}
    return int(block.get("observations", cov.get("student_observations", 0)))


def chart_cv_macro_f1(r: dict, meta: dict) -> None:
    runs = ordered_runs(r)
    means = [r["cv"][n]["macro_f1_fold_mean"] for n in runs]
    stds = [r["cv"][n]["macro_f1_fold_std"] for n in runs]
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    x = np.arange(len(runs))
    bars = ax.bar(x, means, .58, yerr=stds, capsize=6, color="#2266b2", ecolor="#31536e",
                  error_kw={"elinewidth": 1.4})
    for i, n in enumerate(runs):
        fold_f1 = [f["metrics"]["macro_f1"] for f in r["cv"][n]["folds"]]
        ax.scatter(np.full(len(fold_f1), i) + .34, fold_f1, s=26, color="#d77926", zorder=4,
                   label="Per-fold grouped-CV macro-F1" if i == 0 else None)
    ax.axhline(r["cv"][meta["final_run_name"]]["macro_f1_fold_mean"], color="#9a3412", lw=1, ls=":",
               label=f"Final tuned run ({meta['final_run_name']})")
    ax.set_xticks(x, [run_label(n, meta["final_run_name"]) for n in runs])
    ax.set_ylabel("Binary candidate-row macro-F1 (OOF)")
    ax.set_ylim(.5, 1.0)
    label_bars(ax, bars, 4)
    ax.legend(loc="lower right")
    oof = r["cv"][runs[0]]["oof_metrics"]
    cov = r["coverage_summary"]
    phase_save_plot(
        fig, "phase_three_cv_macro_f1_comparison.png",
        "Phase-three model comparison: grouped-CV binary candidate macro-F1",
        f"Three user-selected families + tuned variant; TRAIN trip-grouped {r['config']['folds']}-fold CV, "
        f"{train_observations(cov):,} sampled observations / {oof['candidate_rows']:,} OOF candidate rows. "
        f"Error bars: fold std; orange dots: individual folds. {BINARY_METRIC_NOTE} {PROXY_NOTE}")


def chart_binary_log_loss(r: dict, meta: dict) -> None:
    runs = ordered_runs(r)
    values = [r["cv"][n]["oof_metrics"]["log_loss"] for n in runs]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bars = ax.bar([run_label(n, meta["final_run_name"]) for n in runs], values, .55, color="#64748b")
    ax.set_ylabel("Binary candidate-row log loss (OOF, lower is better)")
    best = min(values)
    ax.set_ylim(0, max(values) * 1.25)
    label_bars(ax, bars, 4)
    ax.annotate("best", (values.index(best), best), xytext=(0, 18), textcoords="offset points",
                ha="center", fontsize=9, color="#166534")
    phase_save_plot(
        fig, "phase_three_binary_log_loss.png",
        "Phase-three out-of-fold binary candidate log loss",
        f"Same OOF candidate rows as the macro-F1 chart (threshold 0.5 probabilities). "
        f"{BINARY_METRIC_NOTE} Lower is better; axis starts at zero — no scale manipulation. {PROXY_NOTE}")


def chart_timing(r: dict, meta: dict) -> None:
    runs = ordered_runs(r)
    cv_train = [r["cv"][n]["training_seconds_total"] for n in runs]
    cv_infer = [r["cv"][n]["inference_seconds_total"] for n in runs]
    final_fit = [r["evaluations"][n].get("training_seconds", float("nan")) for n in runs]
    # microseconds per observation: recorded directly where available, otherwise derived
    # from the measured student_inference_seconds over the scored observation count.
    student_us, student_total_s = [], []
    for n in runs:
        ev = r["evaluations"][n]
        obs = ev["student_metrics"].get("observations")
        total = ev.get("student_inference_seconds")
        recorded = ev.get("student_inference_microseconds_per_observation")
        if recorded is None and total is not None and obs:
            recorded = total / obs * 1_000_000.0
        student_us.append(recorded)
        student_total_s.append(total)
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8.6))
    x = np.arange(len(runs)); labels = [run_label(n, meta["final_run_name"]) for n in runs]
    b1 = axs[0][0].bar(x, cv_train, .55, color="#2266b2")
    axs[0][0].set_xticks(x, labels); axs[0][0].set_ylabel("Seconds (measured)")
    axs[0][0].set_title("Grouped-CV fit time (3 folds, summed)")
    label_bars(axs[0][0], b1, 1)
    b2 = axs[0][1].bar(x, cv_infer, .55, color="#d77926")
    axs[0][1].set_xticks(x, labels); axs[0][1].set_ylabel("Seconds (measured)")
    axs[0][1].set_title("Grouped-CV inference time (OOF rows)")
    label_bars(axs[0][1], b2, 1)
    b3 = axs[1][0].bar(x, final_fit, .55, color="#2266b2")
    axs[1][0].set_xticks(x, labels); axs[1][0].set_ylabel("Seconds (measured)")
    axs[1][0].set_title("Final all-80-trip fit time")
    label_bars(axs[1][0], b3, 1)
    b4 = axs[1][1].bar(x, student_us, .55, color="#d77926")
    axs[1][1].set_yscale("log")
    axs[1][1].set_xticks(x, labels); axs[1][1].set_ylabel("µs per observation (log scale)")
    axs[1][1].set_title("Student split inference per observation")
    for bar, us, total in zip(b4, student_us, student_total_s):
        note = f"{us:,.1f} µs" + (f"\n{total:,.2f} s total" if total is not None else "")
        axs[1][1].annotate(note, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    bank = r.get("knn_bank_policy") or {}
    bank_note = (f"; KNN uses its predeclared {bank['budget']:,}-row TRAIN-only reference bank, so its "
                 f"student timing is bank search, not generalization" if bank.get("budget") else
                 "; KNN timing includes its predeclared TRAIN-only reference bank")
    obs = r["evaluations"][runs[0]]["student_metrics"]["observations"]
    phase_save_plot(
        fig, "phase_three_training_inference_times.png",
        "Phase-three measured fit and inference timing",
        "Single-machine wall-clock timings from the phase-three run logs; BLAS/OpenMP threads capped at 1, "
        "model n_jobs follow the saved config" + bank_note + ". Bottom-right: per-observation student "
        f"inference over {obs:,} observations; the log axis is used only because the values span four orders "
        "of magnitude (logistic regression ~1 µs vs KNN ~10 ms per observation); total seconds are annotated "
        "per bar. Each panel shows one quantity in one unit; no axis is stretched to exaggerate differences.")


def chart_generalization(r: dict, meta: dict) -> None:
    runs = ordered_runs(r)
    train = [r["evaluations"][n]["train_resubstitution_metrics"]["macro_f1"] for n in runs]
    cvm = [r["cv"][n]["macro_f1_fold_mean"] for n in runs]
    student = [r["evaluations"][n]["student_metrics"]["macro_f1"] for n in runs]
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    x = np.arange(len(runs)); w = .26
    b1 = ax.bar(x - w, train, w, color="#94a3b8", label="TRAIN resubstitution (all-80 fit)")
    b2 = ax.bar(x, cvm, w, color="#2266b2", label="TRAIN grouped-CV OOF")
    b3 = ax.bar(x + w, student, w, color="#d77926", label="Student development (10 trips)")
    ax.set_xticks(x, [run_label(n, meta["final_run_name"]) for n in runs])
    ax.set_ylabel("Binary candidate-row macro-F1")
    ax.set_ylim(.5, 1.0)
    for bars in (b1, b2, b3):
        label_bars(ax, bars, 3)
    for i in range(len(runs)):
        gap = student[i] - cvm[i]
        ax.annotate(f"CV→student {gap:+.4f}", (i + w, max(student[i], cvm[i])), xytext=(0, 26),
                    textcoords="offset points", ha="center", fontsize=8, color="#7c2d12")
    ax.legend(loc="lower right")
    cov = r["coverage_summary"]
    phase_save_plot(
        fig, "phase_three_generalization_macro_f1.png",
        "Phase-three training vs CV vs student-development gap",
        f"Same binary candidate metric across splits; TRAIN fit scored on its own training rows (resubstitution, "
        f"optimistic), CV is out-of-fold, student is the 10-trip development set ({student_observations(cov):,} "
        f"observations, all rows, no selection). CV→student deltas annotated as measured, including any negative "
        f"gap. {BINARY_METRIC_NOTE} {PROXY_NOTE}")


def chart_bayesian_trace(r: dict, meta: dict) -> None:
    trials = r["bayesian"]["trials"]
    objs = [t["objective"] for t in trials]
    running = list(np.maximum.accumulate(objs))
    best_i = int(np.argmax(objs))
    baseline = r["bayesian"].get("baseline_objective", objs[0])
    gain = objs[best_i] - baseline
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    x = np.arange(len(trials))
    ax.plot(x, objs, "o-", color="#2266b2", lw=1.6, ms=7, label="Trial objective (equal-trip mean OOF top-1 proxy accuracy)")
    ax.plot(x, running, "s--", color="#d77926", lw=1.4, ms=5, label="Running best")
    ax.scatter([0], [objs[0]], s=190, facecolors="none", edgecolors="#16884a", linewidths=2,
               label="Measured prior (selected family grouped-CV baseline)", zorder=5)
    ax.scatter([best_i], [objs[best_i]], s=190, facecolors="none", edgecolors="#9a3412", linewidths=2,
               label=f"Best trial ({best_i})", zorder=5)
    ax.set_xlabel(f"Bayesian trial (1 measured prior + {len(trials) - 1} search trials; "
                  f"{r['config']['bayesian_trials']} total as configured)")
    ax.set_ylabel("Objective (higher is better)")
    lo, hi = min(objs), max(objs)
    pad = max((hi - lo) * .35, 1e-4)
    ax.set_ylim(lo - pad, hi + pad)
    ax.annotate(f"gain vs prior: {gain:+.6f}", (best_i, objs[best_i]), xytext=(12, 14),
                textcoords="offset points", fontsize=9, color="#9a3412")
    ax.legend(loc="lower right", fontsize=8.5)
    cvf = r["cv"][meta["final_run_name"]]["oof_metrics"]
    phase_save_plot(
        fig, "phase_three_bayesian_trial_trace.png",
        "Phase-three Bayesian HPO: actual trial objectives and running best",
        f"Genuine {len(trials)}-trial Gaussian-process EI search on the selected family "
        f"({r['bayesian'].get('model_name', meta['selection_model'])}) using TRAIN grouped CV only; every point is a "
        f"measured CV run, never an interpolated or invented curve. The measured gain over the prior is "
        f"{gain:+.6f} objective ({baseline:.6f} → {objs[best_i]:.6f}) — reported exactly, however small; "
        f"accompanying macro-F1/log-loss changes are charted separately without scale manipulation. "
        f"Objective definition: equal-trip mean OOF top-1 proxy accuracy, misses included. {PROXY_NOTE}")


def chart_tuned_vs_untuned(r: dict, meta: dict) -> None:
    final = meta["final_run_name"]
    base = final.removesuffix("_tuned")
    if base not in r["cv"]:
        base = meta["selection_model"]
    pairs = [("Objective (OOF top-1)", r["cv"][base]["objective"], r["cv"][final]["objective"], False),
             ("Binary macro-F1 (OOF)", r["cv"][base]["macro_f1_fold_mean"], r["cv"][final]["macro_f1_fold_mean"], False),
             ("Binary log loss (OOF)", r["cv"][base]["oof_metrics"]["log_loss"], r["cv"][final]["oof_metrics"]["log_loss"], True),
             ("Student macro-F1", r["evaluations"][base]["student_metrics"]["macro_f1"],
              r["evaluations"][final]["student_metrics"]["macro_f1"], False)]
    fig, axs = plt.subplots(1, 4, figsize=(13.6, 5.0))
    for ax, (title, a, b, lower_better) in zip(axs, pairs):
        bars = ax.bar(["Default", "Tuned"], [a, b], .55, color=["#94a3b8", "#2266b2"])
        span = max(a, b) - min(a, b)
        ax.set_ylim(min(a, b) - max(span * 4, abs(min(a, b)) * .02), max(a, b) + max(span * 4, abs(max(a, b)) * .02))
        label_bars(ax, bars, 4)
        delta = b - a
        ax.set_title(title + (" (lower better)" if lower_better else ""), fontsize=10)
        ax.annotate(f"{delta:+.4f}", (1, b), xytext=(0, 16), textcoords="offset points", ha="center",
                    fontsize=9, color="#166534" if (delta < 0) == lower_better else "#9a3412")
    delta_note = "; ".join(f"{t.split(' (')[0]} {b - a:+.4f}" for t, a, b, _ in pairs)
    phase_save_plot(
        fig, "phase_three_tuned_vs_untuned.png",
        f"Phase-three tuned vs default: {base}",
        f"Same family before/after Bayesian tuning, identical folds and candidate rows. Measured deltas — "
        f"{delta_note}. A tiny objective gain alongside worse values on other metrics is reported as measured; "
        f"panel y-ranges are widened around the two values so honest small differences stay visibly small. "
        f"{BINARY_METRIC_NOTE} {PROXY_NOTE}")


def _confusion_panel(ax, matrix, title, candidate_rows):
    matrix = np.asarray(matrix, dtype=float)
    ax.imshow(matrix, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            share = matrix[i, j] / matrix[i].sum() * 100 if matrix[i].sum() else 0
            ax.text(j, i, f"{int(matrix[i, j]):,}\n{share:.1f}%", ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > matrix.max() * .6 else "#142033")
    ax.set_xticks([0, 1], ["predicted negative", "predicted positive"])
    ax.set_yticks([0, 1], ["true negative", "true positive"])
    ax.set_title(title, fontsize=10)


def chart_confusions(r: dict, meta: dict) -> None:
    runs = ordered_runs(r)
    for n in runs:
        m = r["evaluations"][n]["student_metrics"]
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        _confusion_panel(ax, m["confusion_matrix"], run_label(n, meta["final_run_name"]).replace("\n", " "),
                         m["candidate_rows"])
        phase_save_plot(
            fig, f"phase_three_confusion_{n}.png",
            f"Phase-three student-development confusion · {n}",
            f"Binary candidate rows at threshold 0.5 on the 10 development trips: {m['candidate_rows']:,} rows, "
            f"{m['observations']:,} observations ({m['candidate_miss_count']:,} observations with no true candidate "
            f"retained in other denominators). {BINARY_METRIC_NOTE} {PROXY_NOTE}")
    fig, axs = plt.subplots(2, 2, figsize=(11.2, 9.6))
    for ax, n in zip(axs.flat, runs):
        m = r["evaluations"][n]["student_metrics"]
        _confusion_panel(ax, m["confusion_matrix"], n, m["candidate_rows"])
    if len(runs) < 4:
        for ax in axs.flat[len(runs):]:
            ax.axis("off")
    phase_save_plot(
        fig, "phase_three_confusion_matrices_all.png",
        "Phase-three confusion matrices: all requested runs (student development)",
        f"Threshold-0.5 binary candidate confusion for the three selected families plus the tuned variant on the "
        f"same 10 development trips. {BINARY_METRIC_NOTE} {PROXY_NOTE}")


def chart_observation_metrics(r: dict, meta: dict) -> None:
    runs = ordered_runs(r)
    binary_cv = [r["cv"][n]["macro_f1_fold_mean"] for n in runs]
    edge_cv = [r["cv"][n]["observation_metrics"].get("observation_edge_macro_f1", float("nan")) for n in runs]
    edge_student = [r["evaluations"][n]["student_observation_metrics"].get("observation_edge_macro_f1", float("nan")) for n in runs]
    nll_student = [r["evaluations"][n]["student_observation_metrics"].get("observation_categorical_log_loss_clipped", float("nan")) for n in runs]
    nll_cv = [r["cv"][n]["observation_metrics"].get("observation_categorical_log_loss_clipped", float("nan")) for n in runs]
    fig, axs = plt.subplots(1, 3, figsize=(13.8, 5.4))
    x = np.arange(len(runs)); labels = [run_label(n, meta["final_run_name"]) for n in runs]
    b1 = axs[0].bar(x - .19, binary_cv, .36, color="#94a3b8", label="Binary candidate macro-F1 (CV OOF)")
    b2 = axs[0].bar(x + .19, edge_cv, .36, color="#2266b2", label="Directed-edge observation macro-F1 (CV OOF)")
    axs[0].set_xticks(x, labels); axs[0].set_ylabel("Macro-F1")
    axs[0].set_ylim(0, 1)
    axs[0].set_title("Two distinct macro-F1 definitions", fontsize=11)
    label_bars(axs[0], b1, 3); label_bars(axs[0], b2, 3)
    axs[0].legend(fontsize=8)
    b3 = axs[1].bar(x - .19, edge_cv, .36, color="#2266b2", label="CV OOF (TRAIN folds)")
    b4 = axs[1].bar(x + .19, edge_student, .36, color="#d77926", label="Student development")
    axs[1].set_xticks(x, labels); axs[1].set_ylabel("Directed-edge observation macro-F1")
    axs[1].set_ylim(0, 1)
    axs[1].set_title("Edge macro-F1 by split", fontsize=11)
    label_bars(axs[1], b3, 3); label_bars(axs[1], b4, 3)
    axs[1].legend(fontsize=8)
    b5 = axs[2].bar(x - .19, nll_cv, .36, color="#2266b2", label="CV OOF (TRAIN folds)")
    b6 = axs[2].bar(x + .19, nll_student, .36, color="#d77926", label="Student development")
    axs[2].set_xticks(x, labels); axs[2].set_ylabel("Categorical NLL (lower is better)")
    axs[2].set_title("Per-observation categorical NLL", fontsize=11)
    label_bars(axs[2], b5, 3); label_bars(axs[2], b6, 3)
    axs[2].legend(fontsize=8)
    ns = r["evaluations"][runs[0]]["student_observation_metrics"].get("observations")
    namespace = r["evaluations"][runs[0]]["student_observation_metrics"].get("edge_class_namespace", "trip_id:canonical_arc_id")
    policy = r["evaluations"][runs[0]]["student_observation_metrics"].get("categorical_log_loss_policy", "")
    phase_save_plot(
        fig, "phase_three_observation_level_metrics.png",
        "Phase-three observation-level metrics: edge macro-F1 and categorical NLL (distinct from binary candidate rows)",
        f"Left: binary candidate-row macro-F1 vs graph-namespaced directed-edge observation macro-F1 are DIFFERENT "
        f"metrics with different denominators and are never mixed. Middle/right: edge macro-F1 and normalized "
        f"categorical NLL on CV folds and the {ns:,} student observations; classes are {namespace}. "
        f"{NLL_NOTE} {policy} {PROXY_NOTE}")


def chart_continuity(r: dict, meta: dict) -> None:
    runs = ordered_runs(r)
    raw = [r["evaluations"][n]["student_observation_metrics"].get("raw_classifier_continuity", float("nan")) for n in runs]
    edge_change = [r["evaluations"][n]["student_observation_metrics"].get("edge_change_continuity", float("nan")) for n in runs]
    pairs = [r["evaluations"][n]["student_observation_metrics"].get("adjacent_observation_pairs", float("nan")) for n in runs]
    same = [r["evaluations"][n]["student_observation_metrics"].get("same_edge_pairs", float("nan")) for n in runs]
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.4))
    x = np.arange(len(runs)); labels = [run_label(n, meta["final_run_name"]) for n in runs]
    b1 = axs[0].bar(x - .19, raw, .36, color="#2266b2", label="Raw classifier continuity")
    b2 = axs[0].bar(x + .19, edge_change, .36, color="#d77926", label="Continuity across edge changes")
    axs[0].set_xticks(x, labels); axs[0].set_ylim(0, 1)
    axs[0].set_ylabel("Fraction of adjacent observation pairs")
    label_bars(axs[0], b1, 3); label_bars(axs[0], b2, 3)
    axs[0].legend(fontsize=8.5)
    axs[1].bar(x - .19, same, .36, color="#2266b2", label="Same-edge pairs")
    axs[1].bar(x + .19, [p - s for p, s in zip(pairs, same)], .36, color="#d77926", label="Edge-change pairs")
    axs[1].set_xticks(x, labels); axs[1].set_ylabel("Adjacent observation pairs")
    axs[1].set_title("Pair counts", fontsize=11)
    axs[1].legend(fontsize=8.5)
    defn = r["evaluations"][runs[0]]["student_observation_metrics"].get("continuity_definition", "")
    phase_save_plot(
        fig, "phase_three_raw_classifier_continuity.png",
        "Phase-three raw-classifier continuity (no HMM, no routed connectors)",
        f"Continuity of the raw per-observation argmax edge assignment on the student development split, before any "
        f"Viterbi decoding: {defn} Counts use all adjacent pairs ({max(pairs):,.0f} for the widest run). "
        f"This is a diagnostic, not a route metric. {PROXY_NOTE}")


def _find_ambiguity_metrics(r: dict, name: str):
    """Locate ambiguity-stratified metrics wherever the learning agent records them.

    Actual layout: top-level results.json["ambiguity_strata"][run] with strata
    ambiguous_proxy / non_ambiguous_proxy / candidate_miss / candidates_present.
    Older layouts kept a dict under an 'ambig*' key inside the evaluation record.
    """
    block = (r.get("ambiguity_strata") or {}).get(name)
    if isinstance(block, dict) and "strata_definition" in block or (
            isinstance(block, dict) and any("ambig" in k for k in block)):
        return "ambiguity_strata", block
    evaluation = r["evaluations"][name]
    for container in (evaluation.get("student_observation_metrics", {}), evaluation):
        for key, value in container.items():
            if "ambig" in key.lower() and isinstance(value, dict):
                return key, value
    return None, None


def chart_ambiguity_strata(r: dict, meta: dict) -> bool:
    runs = ordered_runs(r)
    found = {n: _find_ambiguity_metrics(r, n) for n in runs}
    if not any(v[0] for v in found.values()):
        return False
    sample = found[runs[0]][1] or {}
    available = [s for s in ("ambiguous_proxy", "non_ambiguous_proxy", "ambiguous", "non_ambiguous")
                 if s in sample]
    if len(available) < 2:
        return False
    strata_labels = available[:2]
    metric_keys = {"top1_accuracy": "top1_accuracy",
                   "categorical_log_loss": next(k for k in ("categorical_log_loss_clipped", "categorical_log_loss")
                                                if any(k in (found[n][1] or {}).get(s, {}) for n in runs
                                                       for s in strata_labels))}
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.4))
    x = np.arange(len(runs)); labels = [run_label(n, meta["final_run_name"]) for n in runs]
    width = .8 / len(strata_labels)
    for ax, metric, ylabel, fmt in ((axs[0], "top1_accuracy", "Top-1 proxy accuracy", 3),
                                    (axs[1], metric_keys["categorical_log_loss"], "Categorical NLL (lower better)", 3)):
        for j, stratum in enumerate(strata_labels):
            values = []
            for n in runs:
                key, block = found[n]
                cell = (block or {}).get(stratum) or {}
                values.append(cell.get(metric, float("nan")))
            bars = ax.bar(x + (j - .5) * width, values, width,
                          color="#2266b2" if j == 0 else "#d77926",
                          label=stratum.replace("_", " ").replace(" proxy", "-proxy"))
            label_bars(ax, bars, fmt)
        ax.set_xticks(x, labels); ax.set_ylabel(ylabel); ax.legend(fontsize=8.5)
        if metric != "top1_accuracy":
            ax.set_ylim(0, max((cell.get(metric, 0) for n in runs for s in strata_labels
                                for cell in [(found[n][1] or {}).get(s) or {}]), default=1) * 1.15)
    definition = sample.get("strata_definition", "the disclosed proxy-ambiguity flag")
    miss = sample.get("candidate_miss") or {}
    miss_note = (f" The separate candidate_miss stratum ({miss.get('observations', 0):,} observations, "
                 f"top-1 = 0 by construction, NLL {miss.get('categorical_log_loss_clipped', float('nan')):.2f}) "
                 f"is not charted; both charted strata keep those misses in their denominators."
                 if miss else "")
    counts = " / ".join(f"{s.replace('_', ' ')}: {(sample.get(s) or {}).get('observations', 0):,}"
                        for s in strata_labels)
    phase_save_plot(
        fig, "phase_three_ambiguity_strata.png",
        "Phase-three ambiguity strata: proxy-ambiguous vs non-ambiguous observations",
        f"Student development observations split per the recorded definition: {definition}. "
        f"Observation counts per stratum ({counts}) differ across strata, so accuracy differences partly reflect "
        f"difficulty, not just model quality; axes start at zero and are never rescaled to inflate gaps."
        f"{miss_note} Strata come from the learning agent's recorded diagnostics — no metric is invented, and "
        f"they are proxy-derived, not supplied ground truth. {PROXY_NOTE}")
    return True


def chart_feature_importance(r: dict, meta: dict) -> bool:
    """Actual feature importance read from the frozen winning artifact (no refit).

    The frozen model is a bagged ensemble of decision trees; impurity-based
    (Gini) importances are averaged over the fitted estimators exactly as
    stored in artifacts/phase_three/model.joblib. Read-only: the artifact hash
    is verified against results.json["frozen_artifact"] before loading.
    """
    frozen = r.get("frozen_artifact") or {}
    model_path = ROOT / frozen.get("model", "artifacts/phase_three/model.joblib")
    if not model_path.is_file():
        return False
    current = sha256(model_path)
    if frozen.get("sha256") and current != frozen["sha256"]:
        raise ValueError(f"Frozen artifact hash drift: results.json pins {frozen['sha256'][:12]}…, "
                         f"file is {current[:12]}…; refusing to chart a different artifact")
    import joblib  # lazy: only needed for this chart
    bundle = joblib.load(model_path)
    estimator = bundle["estimator"]
    trees = getattr(estimator, "estimators_", None) or []
    trees = [t for t in trees if hasattr(t, "feature_importances_")]
    if not trees:
        return False
    names = list(bundle.get("feature_names") or [f"feature_{i}" for i in range(trees[0].n_features_in_)])
    importances = np.vstack([t.feature_importances_ for t in trees])
    mean = importances.mean(axis=0)
    std = importances.std(axis=0)
    if mean.shape[0] != len(names):
        raise ValueError(f"Feature-name/importance mismatch: {len(names)} names vs {mean.shape[0]} importances")
    total = float(mean.sum())
    if not 0.9 <= total <= 1.1:  # sanity: importances are tree-normalized; anything else is an absurd scale
        raise ValueError(f"Feature importances sum to {total:.4f}; refusing to chart a broken scale")
    shown = 15
    order = np.argsort(mean)[::-1][:shown]
    fig, ax = plt.subplots(figsize=(10.6, 6.8))
    ypos = np.arange(shown)[::-1]
    ax.barh(ypos, mean[order], .62, xerr=std[order], color="#2266b2", ecolor="#31536e",
            error_kw={"elinewidth": 1.2})
    ax.set_yticks(ypos, [names[i] for i in order], fontsize=9)
    ax.set_xlabel("Mean impurity-based importance (share of total, sums to 1.0 across all features)")
    ax.set_xlim(0, max((mean + std)[order]) * 1.18)
    for y, i in zip(ypos, order):
        ax.annotate(f"{mean[i]:.3f}", (mean[i], y), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8)
    run_name = (bundle.get("metadata") or {}).get("run_name", meta["final_run_name"])
    phase_save_plot(
        fig, "phase_three_feature_importance.png",
        f"Phase-three feature importance: frozen {run_name} ensemble (measured, read-only)",
        f"Mean impurity-based (Gini) importances of the frozen artifact pinned in results.json "
        f"(sha256 {current[:12]}…, verified before load): {len(trees)} fitted decision-tree estimators, "
        f"{len(names)} candidate features, importances sum to {total:.4f} over all features; top {shown} shown. "
        f"Error bars: std across estimators. Impurity importance is biased toward continuous / high-cardinality "
        f"features, is measured on TRAIN-fitted trees, and is a diagnostic of the frozen model — not a causal "
        f"claim and no refit was performed. {PROXY_NOTE}")
    return True


def generate_charts(results_path: Path) -> dict:
    results_path = results_path.resolve()
    r, digest = load_atomic_json(results_path)
    supplement = attach_observation_supplement(r, results_path)
    meta = validate_phase_results(r)
    chart_cv_macro_f1(r, meta)
    chart_binary_log_loss(r, meta)
    chart_timing(r, meta)
    chart_generalization(r, meta)
    chart_bayesian_trace(r, meta)
    chart_tuned_vs_untuned(r, meta)
    chart_confusions(r, meta)
    chart_observation_metrics(r, meta)
    chart_continuity(r, meta)
    ambiguity_available = chart_ambiguity_strata(r, meta)
    importance_available = chart_feature_importance(r, meta)
    inputs = {"results_json": {"path": str(results_path.relative_to(ROOT)), "sha256": digest,
                               "selection_model": meta["selection_model"], "final_run_name": meta["final_run_name"],
                               "runs": meta["runs"], "bayesian_trials": r["config"]["bayesian_trials"]},
              "observation_metrics_supplement": supplement}
    frozen = r.get("frozen_artifact") or {}
    if frozen:
        inputs["frozen_artifact"] = {**frozen, "sha256_verified": frozen.get("sha256") == sha256(FROZEN_ARTIFACT)
                                     if FROZEN_ARTIFACT.is_file() else False}
    if not ambiguity_available:
        print("NOTE: ambiguity-stratified metrics not present in results.json; ambiguity figure skipped "
              "(not fabricated)")
    if not importance_available:
        print("NOTE: frozen artifact not available/readable; feature-importance figure skipped (not fabricated)")
    chart_figures = [e for e in load_registry() if e["stage"] == "charts"]
    print(f"Phase-three charts: {len(chart_figures)} figures registered from {results_path.relative_to(ROOT)} "
          f"(sha256 {digest[:12]}…)")
    return {"inputs": inputs, "ambiguity_available": ambiguity_available, "importance_available": importance_available}


# ------------------------------------------------------- decoder ablation

def generate_decoder_chart(ablation_path: Path) -> dict:
    ablation_path = ablation_path.resolve()
    ab, digest = load_atomic_json(ablation_path)
    run_path = ablation_path.parent / "ablation_run.json"
    run_meta = read_json(run_path) if run_path.exists() else {}
    if run_meta.get("phase") not in (None, "three-model-first"):
        raise ValueError(f"{ablation_path} is not phase-three evidence (phase={run_meta.get('phase')!r})")
    if ab.get("failed_trip_ids"):
        raise ValueError(f"Decoder ablation reports failed trips {ab['failed_trip_ids']}; do not chart incomplete evidence")
    aggregate = ab["aggregate"]
    methods = list(aggregate.keys())
    titles = {"classifier_only_raw": "Classifier only\n(raw argmax sequence)",
              "classifier_plus_HMM": "Classifier + HMM\n(Viterbi decode)",
              "geometry_HMM_default_reused": "Geometry-only HMM\n(historical TRAIN fit, reused read-only)"}
    metrics = [("route_edge_f1", "Route count F1 (arc multiset)"), ("route_length_f1", "Route length F1 (metres)"),
               ("route_edit_similarity", "Ordered edit similarity"), ("route_continuity", "Route continuity (1/jumps)")]
    fig, axs = plt.subplots(1, 4, figsize=(14.6, 5.4))
    x = np.arange(len(methods))
    for ax, (metric, title) in zip(axs, metrics):
        values = [aggregate[m]["trip_macro"][metric] for m in methods]
        bars = ax.bar([titles.get(m, m) for m in methods], values, .6,
                      color=["#94a3b8", "#2266b2", "#d77926"][:len(methods)])
        ax.set_ylim(0, 1); ax.set_title(title, fontsize=10.5)
        label_bars(ax, bars, 3)
        ax.tick_params(axis="x", labelsize=8)
    trip_macro_denominator = aggregate[methods[0]].get("trip_macro_denominator")
    pooled = aggregate.get(methods[0], {}).get("pooled", {})
    discontinuities = [float(aggregate[m]["trip_macro"].get("route_discontinuities") or 0.0) for m in methods]
    reference_discontinuities = float(aggregate[methods[0]]["trip_macro"].get("reference_discontinuities") or 0.0)
    phase_save_plot(
        fig, "phase_three_route_ablation.png",
        "Phase-three final route ablation: classifier-only vs classifier+HMM vs geometry-only",
        f"New phase decoder evidence ({ablation_path.parent.relative_to(ROOT)}) on all 10 student development trips; "
        f"trip-macro denominator {trip_macro_denominator} trips, failed trips counted as zero (none failed). Pooled "
        f"route count F1 shown in report tables. Route discontinuities per trip (mean): "
        f"{' / '.join(f'{m}={d:.1f}' for m, d in zip(methods, discontinuities))}; reference itself averages "
        f"{reference_discontinuities:.1f} broken join(s)/trip ({'present' if reference_discontinuities > 0 else 'none recorded'}) "
        f"and is never 'fixed' with invented connectors. "
        f"Geometry-only method reuses the historical TRAIN-fitted scales read-only and is context, not a phase model. "
        f"Routes are chronological arc-ID traversals; repeats/loops preserved; metric = multiset/edit over arc IDs.")
    print(f"Phase-three decoder chart from {ablation_path.relative_to(ROOT)} (sha256 {digest[:12]}…)")
    return {"inputs": {"decoder_ablation": {"path": str(ablation_path.relative_to(ROOT)), "sha256": digest,
                                            "ablation_run": str(run_path.relative_to(ROOT)) if run_path.exists() else None,
                                            "methods": methods}}}


# ------------------------------------------------------------- final map

def _load_geometry(graph_prefix: Path, prediction_path: Path, reference_path: Path):
    prefix = graph_prefix.resolve(); prediction = prediction_path.resolve(); reference = reference_path.resolve()
    allowed = [ROOT / "Student folder" / s for s in ("train", "student_test")]
    if not any(prefix.is_relative_to(p.resolve()) for p in allowed):
        raise ValueError("Reference maps permit only original TRAIN or student_test development geometry")
    if reference != prefix.with_suffix(".route"):
        raise ValueError("Reference must be the original development route for this exact graph prefix")
    if prediction == reference:
        raise ValueError("A reference route cannot be presented as a prediction")
    nodes = np.loadtxt(prefix.with_suffix(".nodes"), ndmin=2)
    arcs = np.loadtxt(prefix.with_suffix(".arcs"), dtype=np.int64, ndmin=2)
    gps = np.loadtxt(prefix.with_suffix(".track"), ndmin=2)
    pred = np.loadtxt(prediction, dtype=np.int64, ndmin=1).reshape(-1)
    ref = np.loadtxt(reference, dtype=np.int64, ndmin=1).reshape(-1)
    if nodes.shape[1] != 2 or arcs.shape[1] != 2 or gps.shape[1] != 3:
        raise ValueError("Invalid graph/GPS geometry")
    if not np.isfinite(nodes).all() or not np.isfinite(gps).all():
        raise ValueError("Non-finite geometry")
    if arcs.min() < 0 or arcs.max() >= len(nodes):
        raise ValueError("Out-of-range arc endpoints")
    for name, route in (("prediction", pred), ("reference", ref)):
        if not len(route) or route.min() < 0 or route.max() >= len(arcs):
            raise ValueError(f"Out-of-range {name} arc IDs")
    return prefix, nodes, arcs, gps, pred, ref


def _route_integrity(arcs: np.ndarray, route: np.ndarray) -> dict:
    heads, tails = arcs[route, 0], arcs[route, 1]
    broken = int(np.sum(heads[1:] != tails[:-1]))
    unique, counts = np.unique(route, return_counts=True)
    return {"arc_rows": int(len(route)), "distinct_arcs": int(len(unique)),
            "repeated_traversal_arcs": int(np.sum(counts > 1)),
            "repeated_traversal_rows": int(np.sum(counts[counts > 1])),
            "broken_joins": broken,
            "connectors_inserted": 0}


def generate_final_map(graph_prefix: Path, prediction_path: Path, prediction_label: str,
                       screenshot: bool = True, results_path: Path | None = None) -> dict:
    import folium
    from branca.element import Element, MacroElement
    from jinja2 import Template

    if not VENDOR_LEAFLET.is_dir() or not (VENDOR_LEAFLET / "leaflet.js").is_file():
        raise FileNotFoundError(f"Offline Leaflet assets missing at {VENDOR_LEAFLET} (historical vendor copy)")
    prefix, nodes, arcs, gps, pred, ref = _load_geometry(graph_prefix, prediction_path, prefix_ref := graph_prefix.with_suffix(".route"))
    output_path = phase_safe_output(PHASE_MAPS / "phase_three_development_trip_00000009_final.html")
    rel = Path(os.path.relpath(VENDOR_LEAFLET, output_path.parent)).as_posix()
    # Relevant roads: dedupe reciprocal/duplicate geometry for display only.
    low = gps[:, :2].min(axis=0); high = gps[:, :2].max(axis=0)
    pad = np.maximum((high - low) * .10, .001); geom = nodes[arcs]
    mask = np.all(geom.max(axis=1) >= low - pad, axis=1) & np.all(geom.min(axis=1) <= high + pad, axis=1)
    ids = np.flatnonzero(mask)
    _, positions = np.unique(np.sort(arcs[ids], axis=1), axis=0, return_index=True)
    road_ids = ids[np.sort(positions)]
    roads = nodes[arcs[road_ids]]
    center = gps[:, :2].mean(axis=0)
    m = folium.Map(location=[center[1], center[0]], zoom_start=13, tiles=None, prefer_canvas=True,
                   control_scale=True, zoom_control=True, attribution_control=True)
    m.default_js = [("leaflet", f"{rel}/leaflet.js")]
    m.default_css = [("leaflet_css", f"{rel}/leaflet.css")]
    m.get_root().header.add_child(Element('<meta name="robots" content="noindex,nofollow">'))
    m.get_root().header.add_child(Element(
        '<style>.leaflet-container{background:#f4f7fa;font-family:Arial,sans-serif}'
        '.leaflet-control-layers{font-size:12px;max-width:420px}'
        '.leaflet-control-layers label{white-space:normal}'
        '.leaflet-control-attribution{font-size:10px}</style>'))
    layer = folium.FeatureGroup(name=f"Relevant road network · gray · {len(road_ids):,} chords", show=True)
    folium.PolyLine(roads[:, :, ::-1].tolist(), color=COLORS["roads"], weight=1.2, opacity=.50).add_to(layer)
    layer.add_to(m)
    layer = folium.FeatureGroup(name="Development reference traversal · green (not evaluator data)", show=True)
    folium.PolyLine(nodes[arcs[ref]][:, :, ::-1].tolist(), color=COLORS["reference"], weight=7, opacity=.70).add_to(layer)
    layer.add_to(m)
    layer = folium.FeatureGroup(name=f"Phase-three learned prediction · orange dashed · {prediction_label}", show=True)
    folium.PolyLine(nodes[arcs[pred]][:, :, ::-1].tolist(), color=COLORS["prediction"], weight=3.4, opacity=1,
                    dash_array="8,5").add_to(layer)
    prediction_layer = layer
    layer = folium.FeatureGroup(name=f"Raw GPS · blue · {len(gps):,} observations", show=True)
    folium.PolyLine(gps[:, [1, 0]].tolist(), color=COLORS["gps"], weight=1, opacity=.40, dash_array="2,5").add_to(layer)
    for lon, lat, _ in gps:
        folium.CircleMarker([lat, lon], radius=2.0, weight=0, fill=True, fill_color=COLORS["gps"],
                            fill_opacity=.75).add_to(layer)
    layer.add_to(m)
    layer = folium.FeatureGroup(name="First / last raw GPS position", show=True)
    for i, text in ((0, "First raw GPS row"), (-1, "Last raw GPS row")):
        folium.CircleMarker(gps[i, [1, 0]].tolist(), radius=6, weight=2,
                            color="#ffffff" if i == 0 else "#142033", fill=True, fill_color=COLORS["gps"],
                            fill_opacity=1, tooltip=text).add_to(layer)
    layer.add_to(m)
    prediction_layer.add_to(m)  # dashed prediction on top, green reference still a halo
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    label = html.escape(prediction_label)
    partition = prefix.parent.parent.name
    card = f'''<div id="development-map-title" style="position:fixed;top:14px;left:60px;z-index:9999;background:#ffffffee;border:1px solid #cbd5e1;border-radius:10px;padding:13px 18px;box-shadow:0 3px 14px #152d4a22;max-width:650px">
    <div style="font:700 21px Arial;color:#142033">PathSnap phase three · trip {prefix.name} · {partition} development only</div>
    <div style="font:13px Arial;color:#475569;margin-top:7px">{len(gps):,} GPS points · {len(nodes):,} nodes · {len(arcs):,} directed arcs</div>
    <div style="font:700 12px Arial;color:#9a3412;margin-top:7px">INTERNAL DEVELOPMENT REPORT MAP — includes reference geometry; never serve in evaluator UI/API</div></div>
    <div id="development-map-legend" style="position:fixed;bottom:28px;left:20px;z-index:9999;background:#ffffffee;border:1px solid #cbd5e1;border-radius:10px;padding:15px 18px;box-shadow:0 3px 14px #152d4a22;max-width:690px;font:14px Arial;line-height:1.8;color:#142033">
    <div style="font-weight:700;font-size:15px;margin-bottom:5px">Actual observations and traversed-arc geometry (phase-three final)</div>
    <div><span style="color:{COLORS['gps']};font-weight:900">● ┄</span> Raw GPS observations (blue); dotted line shows observation order</div>
    <div><span style="color:{COLORS['roads']};font-weight:900">━━</span> Relevant road network (gray endpoint chords)</div>
    <div><span style="color:{COLORS['reference']};font-weight:900">━━</span> Supplied development reference traversal (green)</div>
    <div><span style="color:{COLORS['prediction']};font-weight:900">┅┅</span> {label} — new learned prediction (orange dashed)</div>
    <div style="font-size:12px;color:#475569;margin-top:6px">Offline Folium / Leaflet · no web tiles · layer controls at upper right<br>Reference traversal is not independently verified pointwise ground truth.</div></div>'''
    m.get_root().html.add_child(Element(card))
    m.fit_bounds([[gps[:, 1].min(), gps[:, 0].min()], [gps[:, 1].max(), gps[:, 0].max()]],
                 padding_top_left=(55, 115), padding_bottom_right=(30, 245))
    ready = MacroElement()
    ready._template = Template("{% macro script(this, kwargs) %}window.__PATHSNAP_DEVELOPMENT_MAP_READY__ = true;{% endmacro %}")
    m.add_child(ready)
    m.save(str(output_path))
    text = output_path.read_text(encoding="utf-8")
    if re.search(r'''(?:src|href)=["']https?://''', text):
        raise RuntimeError("Unexpected remote dependency in offline Folium HTML")
    pred_integrity = _route_integrity(arcs, pred)
    ref_integrity = _route_integrity(arcs, ref)
    inputs = [prefix.with_suffix(".nodes"), prefix.with_suffix(".arcs"), prefix.with_suffix(".track"),
              prefix.with_suffix(".route"), prediction_path.resolve()]
    model_sha = None
    model_path = ROOT / "artifacts" / "phase_three" / "model.joblib"
    if model_path.is_file():
        model_sha = sha256(model_path)
    index_row = None
    index_path = prediction_path.parent / "index.csv"
    if index_path.is_file():
        for line in index_path.read_text(encoding="utf-8").splitlines()[1:]:
            if prediction_path.name in line or "00000009" in line:
                index_row = line.strip()
                if prediction_path.name in line and line.rstrip().endswith(sha256(prediction_path)):
                    break
    metadata = {"map_path": str(output_path.relative_to(ROOT)), "phase": "three-model-first",
                "development_only": True, "contains_reference_geometry": True, "do_not_serve_to_evaluator": True,
                "phase_status": "phase_three_final_learned",
                "historical_baseline_map_preserved": str(HISTORICAL_BASELINE_MAP.relative_to(ROOT)) if HISTORICAL_BASELINE_MAP.exists() else None,
                "historical_prior_final_map_preserved": str(HISTORICAL_PRIOR_FINAL_MAP.relative_to(ROOT)) if HISTORICAL_PRIOR_FINAL_MAP.exists() else None,
                "trip_id": prefix.name, "partition": partition, "prediction_label": prediction_label,
                "graph_prefix": str(prefix.relative_to(ROOT)),
                "prediction_path": str(prediction_path.resolve().relative_to(ROOT)),
                "development_reference_path": str(prefix.with_suffix(".route").relative_to(ROOT)),
                "gps_points": len(gps), "nodes": len(nodes), "directed_arcs": len(arcs),
                "displayed_road_chords": int(len(road_ids)),
                "prediction_route_integrity": pred_integrity, "reference_route_integrity": ref_integrity,
                "tiles": None, "leaflet_assets": str(VENDOR_LEAFLET.relative_to(ROOT)),
                "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
                "model_artifact_sha256": model_sha, "routes_index_row": index_row,
                "folium_version": folium.__version__, "created_at_utc": utcnow()}
    phase_write_json(output_path.with_suffix(".metadata.json"), metadata)
    register_figure({"file": output_path.name, "title": f"Phase-three final development map · trip {prefix.name}",
                     "caption": f"{prediction_label}; raw GPS blue, road graph gray, development reference green, "
                                f"learned prediction orange dashed. Original arc IDs preserved with repeats/loops; "
                                f"reference broken joins ({ref_integrity['broken_joins']}) shown as-is, no invented "
                                f"connectors. Internal report map, never evaluator-facing.",
                     "stage": "map"})
    if screenshot:
        capture_map_screenshot(output_path)
    print(f"Phase-three final map: {output_path.relative_to(ROOT)} "
          f"(prediction {pred_integrity['arc_rows']} arc rows, {pred_integrity['repeated_traversal_arcs']} repeated arcs, "
          f"{pred_integrity['broken_joins']} broken joins shown as-is)")
    return metadata


def capture_map_screenshot(html_path: Path) -> dict:
    from playwright.sync_api import sync_playwright
    screenshot_path = phase_safe_output(html_path.with_suffix(".png"))
    errors, requests, blocked = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda req: requests.append(req.url))
        def reject_network(route):
            blocked.append(route.request.url)
            route.abort()
        page.route(re.compile(r"^https?://"), reject_network)
        page.goto(html_path.resolve().as_uri(), wait_until="load")
        page.wait_for_function("window.__PATHSNAP_DEVELOPMENT_MAP_READY__ === true && typeof L !== 'undefined'")
        page.wait_for_function("document.querySelector('.leaflet-control-layers-overlays') !== null")
        page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
        proof = page.evaluate("""() => {const name=Object.keys(window).find(k => /^map_[a-f0-9]+$/.test(k) && window[k] instanceof L.Map); const m=window[name]; return {leaflet_version:L.version,map_variable:name,layer_count:Object.keys(m._layers).length,bounds:m.getBounds(),canvas_count:document.querySelectorAll('canvas').length,legend:document.querySelector('#development-map-legend').innerText,title:document.querySelector('#development-map-title').innerText,checked_layers:document.querySelectorAll('.leaflet-control-layers-selector:checked').length};}""")
        proof["layout_and_canvas"] = page.evaluate("""() => {
            const rect = selector => {const r=document.querySelector(selector).getBoundingClientRect();return {left:r.left,right:r.right,top:r.top,bottom:r.bottom,width:r.width,height:r.height};};
            const title=rect('#development-map-title'), control=rect('.leaflet-control-layers'), legend=rect('#development-map-legend');
            const canvas=document.querySelector('canvas'), pixels=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
            const colors={gps_blue:[33,102,232],road_gray:[146,155,167],reference_green:[22,136,74],prediction_orange:[244,123,19]}, counts={};
            for (const [name,c] of Object.entries(colors)) {let n=0;for(let i=0;i<pixels.length;i+=4){if(pixels[i+3]>40 && Math.abs(pixels[i]-c[0])<28 && Math.abs(pixels[i+1]-c[1])<28 && Math.abs(pixels[i+2]-c[2])<28)n++;}counts[name]=n;}
            return {title,control,legend,title_control_overlap:!(title.right<=control.left || control.right<=title.left || title.bottom<=control.top || control.bottom<=title.top),canvas_shape:[canvas.width,canvas.height],visible_layer_color_pixels:counts};
        }""")
        if proof["layout_and_canvas"]["title_control_overlap"]:
            raise RuntimeError("Map title and layer controls overlap")
        mandatory = ("gps_blue", "road_gray", "reference_green", "prediction_orange")
        pixel_counts = proof["layout_and_canvas"]["visible_layer_color_pixels"]
        if not all(pixel_counts.get(k, 0) > 5 for k in mandatory):
            raise RuntimeError(f"One mandatory actual map geometry colour is not visibly rendered: {pixel_counts}")
        legend_text = proof.get("legend", "")
        for token in ("Raw GPS", "road network", "reference", "learned prediction"):
            if token not in legend_text:
                raise RuntimeError(f"Legend is missing expected layer description {token!r}")
        page.screenshot(path=str(screenshot_path), full_page=True)
        proof.update({"browser_version": browser.version, "source_html": str(html_path.relative_to(ROOT)),
                      "screenshot_path": str(screenshot_path.relative_to(ROOT)),
                      "viewport": {"width": 1440, "height": 1000}, "page_errors": errors,
                      "requests": requests, "blocked_network_requests": blocked,
                      "capture_method": "actual installed Playwright Chromium, local file:// HTML, HTTP(S) blocked",
                      "created_at_utc": utcnow()})
        browser.close()
    if errors or blocked or proof["layer_count"] < 20:
        raise RuntimeError(f"Map failed browser verification: {proof}")
    if shutil.which("tesseract"):
        result = subprocess.run(["tesseract", str(screenshot_path), "stdout", "--psm", "11"],
                                text=True, capture_output=True, check=True)
        phase_write_text(screenshot_path.with_suffix(".ocr.txt"), result.stdout)
        proof["ocr_text_path"] = str(screenshot_path.with_suffix(".ocr.txt").relative_to(ROOT))
    phase_write_json(screenshot_path.with_suffix(".browser.json"), proof)
    print(f"Actual Chromium screenshot: {screenshot_path.relative_to(ROOT)}; {proof['layer_count']:,} live Leaflet "
          f"layers; {len(errors)} JS errors; colour pixels {proof['layout_and_canvas']['visible_layer_color_pixels']}")
    return proof


# ------------------------------------------------------------- manifest/verify

EDA_FILES = ("dataset_inventory.png", "missing_invalid_and_anomalies.png", "trajectory_distributions.png",
             "timestamp_gaps.png", "road_network_sizes_connectivity.png", "gps_geographic_coverage.png",
             "gps_trip_gallery_train_01.png", "gps_trip_gallery_train_02.png", "gps_trip_gallery_train_03.png",
             "gps_trip_gallery_train_04.png", "gps_trip_gallery_student_test_01.png", "representative_road_route.png")


def reused_eda_records() -> list[dict]:
    """Link the 12 existing EDA assets read-only, verifying their current hashes."""
    if not ROOT_FIGURE_MANIFEST.exists():
        raise FileNotFoundError("Root figure_manifest.json missing; EDA reuse links cannot be verified")
    root_manifest = read_json(ROOT_FIGURE_MANIFEST)
    by_name = {Path(row["path"]).name: row for row in root_manifest}
    records = []
    for name in EDA_FILES:
        row = by_name.get(name)
        if row is None:
            raise ValueError(f"EDA asset {name} not present in the completed root manifest")
        path = ROOT / row["path"]
        if not path.is_file():
            raise FileNotFoundError(f"EDA asset missing: {path}")
        current = sha256(path)
        if current != row.get("sha256"):
            raise ValueError(f"EDA asset hash drift for {name}: manifest {row.get('sha256')[:12]}… vs current {current[:12]}…")
        records.append({"path": row["path"], "sha256": current, "bytes": path.stat().st_size,
                        "title": row.get("title"), "caption": row.get("caption"),
                        "partition": row.get("partition"), "source": row.get("source"),
                        "status": row.get("status"), "reused_read_only": True,
                        "phase_role": "eda_context_historical_complete"})
    return records


def build_manifest(stage_inputs: dict | None = None) -> dict:
    records = []
    for entry in load_registry():
        filename = entry["file"]
        path = PHASE_MAPS / filename if entry["stage"] == "map" else PHASE_PLOTS / filename
        if not path.exists():
            raise FileNotFoundError(f"Figure registry references missing file: {path}")
        records.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size,
                        "title": entry["title"], "caption": entry["caption"],
                        "partition": "TRAIN grouped CV + student_test development; phase three (LR/KNN/Bagging + tuned)",
                        "source": (stage_inputs or {}).get("source_note", "phase-three stable measured outputs"),
                        "status": "stable_actual", "stage": entry["stage"]})
        if entry["stage"] == "map":
            # Map companions: the report links the PNG, and browser.json / metadata.json /
            # ocr.txt are the programmatic render proofs; record them with verified hashes.
            companion_notes = {".png": "actual headless-Chromium screenshot of the map HTML",
                               ".browser.json": "programmatic render proof (DOM layout, canvas pixel counts, "
                                                "blocked network, Leaflet layer count)",
                               ".metadata.json": "map provenance (input sha256s, route integrity, artifact hash)",
                               ".ocr.txt": "tesseract OCR text of the rendered screenshot"}
            for suffix, note in companion_notes.items():
                companion = path.with_suffix(suffix) if suffix != ".png" else path.with_suffix(".png")
                if companion != path and companion.is_file():
                    records.append({"path": str(companion.relative_to(ROOT)), "sha256": sha256(companion),
                                    "bytes": companion.stat().st_size, "title": f"{entry['title']} · {suffix[1:]}",
                                    "caption": note, "stage": "map_companion", "status": "stable_actual",
                                    "companion_of": str(path.relative_to(ROOT))})
    manifest = {"phase": "three-model-first",
                "generated_utc": utcnow(),
                "scope": "visualizations/phase_three/ only; historical visualizations/ root assets untouched",
                "families": list(PHASE_FAMILIES),
                "excluded_families_note": "GNB/HistGB/RF/ET are historical prior work, not this phase",
                "proxy_label_caveat": PROXY_NOTE,
                "inputs": (stage_inputs or {}).get("inputs", {}),
                "figures": records,
                "eda_reused_read_only": reused_eda_records(),
                "maps_retained_historical": [str(p.relative_to(ROOT)) for p in (HISTORICAL_BASELINE_MAP, HISTORICAL_PRIOR_FINAL_MAP) if p.exists()]}
    phase_write_json(PHASE_MANIFEST, manifest)
    print(f"Phase-three manifest: {PHASE_MANIFEST.relative_to(ROOT)} "
          f"({len(records)} phase figures, {len(manifest['eda_reused_read_only'])} reused EDA links)")
    return manifest


def verify_assets() -> int:
    problems = []
    if not PHASE_MANIFEST.exists():
        print("No phase manifest yet; nothing to verify")
        return 1
    manifest = read_json(PHASE_MANIFEST)
    checks = []
    for row in manifest["figures"]:
        path = ROOT / row["path"]
        record = {"path": row["path"], "exists": path.is_file()}
        if not record["exists"]:
            problems.append(f"missing figure {row['path']}")
            checks.append(record)
            continue
        current = sha256(path)
        record["sha256_matches"] = current == row["sha256"]
        record["bytes"] = path.stat().st_size
        if not record["sha256_matches"]:
            problems.append(f"hash drift for {row['path']}")
        record["inside_phase_scope"] = path.resolve().is_relative_to(PHASE_OUT.resolve())
        if not record["inside_phase_scope"]:
            problems.append(f"figure outside phase scope: {row['path']}")
        checks.append(record)
    eda_checks = []
    for row in manifest.get("eda_reused_read_only", []):
        path = ROOT / row["path"]
        record = {"path": row["path"], "exists": path.is_file(),
                  "sha256_matches": path.is_file() and sha256(path) == row["sha256"],
                  "read_only_link": True}
        if not record["exists"] or not record["sha256_matches"]:
            problems.append(f"EDA reuse link broken: {row['path']}")
        eda_checks.append(record)
    # Leakage guard: no phase map/figure may be referenced by the evaluator-facing app.
    leakage_hits = []
    for area in ("frontend", "backend", "static", "public"):
        area_path = ROOT / area
        if not area_path.is_dir():
            continue
        for hit in area_path.rglob("*phase_three*"):
            leakage_hits.append(str(hit.relative_to(ROOT)))
    if leakage_hits:
        problems.append(f"evaluator-facing areas reference phase-three assets: {leakage_hits[:5]}")
    # Optional OCR readability capture for phase plots (evidence, not visual inspection claims).
    ocr = []
    if shutil.which("tesseract"):
        for row in manifest["figures"]:
            if not row["path"].endswith(".png"):
                continue
            path = ROOT / row["path"]
            result = subprocess.run(["tesseract", str(path), "stdout", "--psm", "11"],
                                    text=True, capture_output=True, check=True)
            out = phase_write_text(PHASE_OCR / (Path(row["path"]).stem + ".txt"), result.stdout)
            ocr.append({"path": str(out.relative_to(ROOT)), "characters": len(result.stdout)})
    report = {"status": "passed" if not problems else "failed", "problems": problems,
              "generated_utc": utcnow(), "figures": checks, "eda_reused_read_only": eda_checks,
              "evaluator_leakage_scan": {"hits": leakage_hits}, "ocr": ocr,
              "no_evaluator_data_accessed": True,
              "visual_inspection_claim": "OCR/DOM/pixel evidence only; no image-tool visual inspection claimed"}
    phase_write_json(PHASE_VERIFICATION, report)
    if problems:
        print(f"Phase-three verification FAILED: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"Phase-three verification passed: {len(checks)} phase figures, {len(eda_checks)} read-only EDA links, "
          f"0 evaluator leakage hits, {len(ocr)} OCR captures")
    return 0


# ---------------------------------------------------------------- readiness

def status() -> dict:
    def describe(path: Path, validator=None):
        if not path.exists():
            return {"ready": False, "reason": "missing"}
        try:
            if validator:
                validator(path)
            return {"ready": True, "path": str(path.relative_to(ROOT))}
        except Exception as exc:  # noqa: BLE001 - status must report, not crash
            return {"ready": False, "reason": str(exc)[:200]}
    def results_ok(path):
        r, _ = load_atomic_json(path)
        attach_observation_supplement(r, path)
        validate_phase_results(r)
    def ablation_ok(path):
        ab, _ = load_atomic_json(path)
        if ab.get("failed_trip_ids"):
            raise ValueError("failed trips present")
    info = {"results_json": describe(DEFAULT_RESULTS, results_ok),
            "decoder_ablation": describe(DEFAULT_ABLATION, ablation_ok),
            "final_route": describe(DEFAULT_PREDICTION),
            "vendor_leaflet": {"ready": (VENDOR_LEAFLET / "leaflet.js").is_file()},
            "tesseract": {"available": bool(shutil.which("tesseract"))},
            "historical_eda_manifest": {"ready": ROOT_FIGURE_MANIFEST.is_file()}}
    print(json.dumps(info, indent=2))
    return info


# ---------------------------------------------------------------------- CLI

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase-three (LR / KNN / Bagging) chart and map generator. Writes only under visualizations/phase_three/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("stage", choices=("status", "charts", "decoder", "map", "manifest", "verify", "all"))
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--graph-prefix", type=Path, default=DEFAULT_GRAPH_PREFIX)
    parser.add_argument("--prediction-label", default=None,
                        help="Prediction layer label; defaults to the phase final run name from results.json")
    parser.add_argument("--no-screenshot", action="store_true", help="Skip the Chromium map capture")
    args = parser.parse_args()

    if args.stage == "status":
        status()
        return 0

    inputs: dict = {}

    if args.stage in ("charts", "all"):
        if not args.results.exists():
            if args.stage == "all":
                print(f"SKIP charts: {args.results} not present yet (waiting for PHASE_RESULTS_READY)")
            else:
                print(f"ERROR: {args.results} not present yet (waiting for PHASE_RESULTS_READY)", file=sys.stderr)
                return 2
        else:
            try:
                outcome = generate_charts(args.results)
            except (ValueError, RuntimeError) as exc:
                print(f"ERROR: charts not ready: {exc}", file=sys.stderr)
                return 2
            inputs.update(outcome["inputs"])

    if args.stage in ("decoder", "all"):
        if not args.ablation.exists():
            if args.stage == "all":
                print(f"SKIP decoder chart: {args.ablation} not present yet (waiting for phase decoder ablation)")
            else:
                print(f"ERROR: {args.ablation} not present yet", file=sys.stderr)
                return 2
        else:
            try:
                outcome = generate_decoder_chart(args.ablation)
            except (ValueError, RuntimeError) as exc:
                print(f"ERROR: decoder chart not ready: {exc}", file=sys.stderr)
                return 2
            inputs.update(outcome["inputs"])

    if args.stage in ("map", "all"):
        if not args.prediction.exists():
            if args.stage == "all":
                print(f"SKIP map: {args.prediction} not present yet (waiting for PHASE_FINAL_ROUTES_READY)")
            else:
                print(f"ERROR: {args.prediction} not present yet (waiting for PHASE_FINAL_ROUTES_READY)", file=sys.stderr)
                return 2
        else:
            label = args.prediction_label
            if label is None:
                label = "Phase-three learned prediction"
                if args.results.exists():
                    try:
                        r, _ = load_atomic_json(args.results)
                        label = f"Phase-three learned prediction ({r['final_run_name']})"
                    except Exception:  # noqa: BLE001 - label fallback is cosmetic
                        pass
            generate_final_map(args.graph_prefix, args.prediction, label, screenshot=not args.no_screenshot)

    if args.stage in ("manifest", "all", "charts", "decoder", "map"):
        build_manifest({"inputs": inputs,
                        "source_note": "experiments/phase_three/results.json, reports/phase_three/decoder/, "
                                       "outputs/phase_three_student_test/ (stable measured phase outputs)"})

    if args.stage in ("verify", "all"):
        return verify_assets()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
