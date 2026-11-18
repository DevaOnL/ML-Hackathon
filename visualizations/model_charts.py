"""Plots from explicitly final/stable, nonfabricated experiment JSON only."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from common import ROOT, OUT, PROXY_NOTE, read_json, write_json, save_plot, label_bars
from observation_charts import plot_observation_metrics, plot_continuity, plot_ambiguity_strata


def model_label(name):
    mapping = {"logistic_regression": "Logistic\nregression", "gaussian_nb": "Gaussian\nNaive Bayes", "decision_tree": "Decision\ntree", "random_forest": "Random\nforest", "hist_gradient_boosting": "Histogram\ngradient boost"}
    base = name.removesuffix("_tuned")
    return mapping.get(base, base.replace("_", "\n")) + ("\n(tuned)" if name.endswith("_tuned") else "")


def load_complete_experiments(path: Path):
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size): raise RuntimeError("Experiment output changed during read; wait for atomic final publication")
    r = json.loads(raw)
    for key in ("cv", "evaluations", "comparison", "bayesian", "coverage_summary", "final_run_name", "selection_model"):
        if key not in r: raise ValueError(f"Incomplete experiment output: {key}")
    if len(r["cv"]) < 6 or set(r["cv"]) != set(r["evaluations"]): raise ValueError("Need five completed baselines + tuned evaluation, not partial results")
    if len(r["bayesian"]["trials"]) != r["config"]["bayesian_trials"]: raise ValueError("Bayesian trial log is incomplete")
    for name, evaluation in r["evaluations"].items():
        if "student_observation_metrics" not in evaluation: raise ValueError(f"Observation-level supplement missing for {name}; wait for complete publication")
    return r, hashlib.sha256(raw).hexdigest()


def generate_experiment_charts(path: Path):
    r, fingerprint = load_complete_experiments(path)
    names = list(r["cv"]); labels = [model_label(n) for n in names]; x = np.arange(len(names))
    colors = ["#d77926" if n == r["final_run_name"] else "#2266b2" for n in names]
    c = r["coverage_summary"]
    sample_note = f"TRAIN CV: {c['train_observations']:,}/{c['train_total_observations']:,} observations across {c['train_trips']} trips. Student-development: {c['student_observations']:,}/{c['student_total_observations']:,} observations across {c['student_trips']} trips."
    foot = sample_note+"\n"+PROXY_NOTE
    fig, ax = plt.subplots(figsize=(12.6, 5.9))
    means = [r["cv"][n]["macro_f1_fold_mean"] for n in names]; std = [r["cv"][n]["macro_f1_fold_std"] for n in names]
    bars = ax.bar(x, means, yerr=std, capsize=5, color=colors, width=.65, alpha=.85); label_bars(ax, bars, 4)
    for i, n in enumerate(names):
        vals = [f["metrics"]["macro_f1"] for f in r["cv"][n]["folds"]]; ax.scatter(i+np.linspace(-.13, .13, len(vals)), vals, color="#132d46", s=24, zorder=4)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, min(1.08, max(means)+max(std)+.14)), ylabel="Binary-candidate Macro F1", title="Bars = trip-grouped fold mean; error bars = population SD; dots = actual folds")
    ax.grid(axis="y"); ax.set_axisbelow(True)
    save_plot(fig, "cv_macro_f1_comparison.png", "Cross-validation Macro F1: five baselines and selected tuned model", foot)
    fig, ax = plt.subplots(figsize=(12.6, 6.1))
    for i, (key, label, color) in enumerate([("top1_accuracy", "Observation Top-1 proxy accuracy", "#2266b2"), ("macro_f1", "Binary-candidate Macro F1", "#d77926")]):
        vals = [r["evaluations"][n]["student_metrics"][key] for n in names]; bars = ax.bar(x+(i-.5)*.35, vals, .35, label=label, color=color); label_bars(ax, bars, 4)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.14), ylabel="Agreement / F1 (0–1)", title="One winning arc per observation vs binary classification of candidate rows")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="y"); ax.set_axisbelow(True)
    save_plot(fig, "student_development_top1_macro_f1.png", "Student-development comparison (not final evaluator accuracy)", foot)
    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    for i, (source, key, label, color) in enumerate([("cv", "oof_metrics", "Training OOF", "#2266b2"), ("evaluations", "student_metrics", "Student-development", "#d77926")]):
        vals = [r[source][n][key]["log_loss"] for n in names]; bars = ax.bar(x+(i-.5)*.35, vals, .35, label=label, color=color); label_bars(ax, bars, 4)
    ax.set(xticks=x, xticklabels=labels, ylabel="Binary-candidate log loss (lower is better)"); ax.margins(y=.2); ax.grid(axis="y"); ax.set_axisbelow(True); ax.legend()
    save_plot(fig, "log_loss_comparison.png", "Probability quality: measured binary-candidate log loss", foot)
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    for ax, key, label in [(axs[0], "training_seconds", "Final refit on all 80 TRAIN trips, seconds"), (axs[1], "student_inference_seconds", "Classifier probability inference, seconds")]:
        vals = [r["evaluations"][n][key] for n in names]; bars = ax.bar(x, vals, color=colors); label_bars(ax, bars, 4)
        ax.set(xticks=x, xticklabels=labels, ylabel=label); ax.tick_params(axis="x", labelsize=8)
        if min(vals) > 0 and max(vals)/min(vals) > 20: ax.set_yscale("log")
        ax.margins(y=.3); ax.grid(axis="y"); ax.set_axisbelow(True)
    axs[0].set_title("One full TRAIN refit (not summed CV cost)"); axs[1].set_title("Same student candidate matrix for every model")
    save_plot(fig, "training_inference_times.png", "Measured model cost: training and probability inference", sample_note+"\nActual wall-clock timing; classifier-only inference excludes graph loading, candidate generation and HMM routing. Other concurrent work can affect timing.")
    plot_confusions(r, names, foot); plot_bayesian(r, foot); plot_tuned(r, foot)
    plot_observation_metrics(r, names, labels, x, colors, foot)
    plot_continuity(r, names, labels, x, colors, foot)
    strata_source = plot_ambiguity_strata(r, foot)
    fig, ax = plt.subplots(figsize=(12.6, 6))
    keys = [("evaluations", "train_resubstitution_metrics", "Training resubstitution", "#8eaac7"), ("cv", "oof_metrics", "Training out-of-fold", "#2266b2"), ("evaluations", "student_metrics", "Student-development", "#d77926")]
    for i, (source, key, label, color) in enumerate(keys): ax.bar(x+(i-1)*.25, [r[source][n][key]["macro_f1"] for n in names], .25, color=color, label=label)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.12), ylabel="Binary-candidate Macro F1"); ax.grid(axis="y"); ax.set_axisbelow(True); ax.legend(fontsize=9, loc="lower right")
    save_plot(fig, "generalization_macro_f1.png", "Generalisation: in-sample vs held-out proxy agreement", foot+"\nTraining resubstitution is not an unbiased estimate; model/search selection used trip-grouped TRAIN CV only.")
    write_json(OUT/"experiment_chart_provenance.json", {"source": str(path.relative_to(ROOT)), "sha256": fingerprint, "models": names, "selected_model": r["selection_model"], "final_run_name": r["final_run_name"], "coverage": c, "target_policy": r.get("target_policy"), "ambiguity_strata_source": strata_source, "observation_metrics": "per-model student observation metrics and TRAIN OOF observation metrics merged inside the results file by experiments/observation_metrics.py", "artificial_curves": False, "route_continuity": "Not inferred from candidate confusion; use separate direct continuity + route/HMM experiments"})
    print(f"Experiment charts: {len(names)} measured runs; source SHA256 {fingerprint}")


def draw_cm(ax, cm, title):
    cm = np.asarray(cm, dtype=int)
    if cm.shape != (2, 2): raise ValueError("Expected fixed-label [0,1] binary-candidate confusion")
    vmax = max(1, cm.max()); ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    for (i, j), value in np.ndenumerate(cm): ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=12, fontweight="bold", color="white" if value > vmax*.55 else "#102a43")
    ax.set(xticks=[0, 1], xticklabels=["Other (0)", "Proxy match (1)"], yticks=[0, 1], yticklabels=["Other (0)", "Proxy match (1)"], xlabel="Predicted candidate class", ylabel="True proxy candidate class", title=title); ax.tick_params(length=0)


def plot_confusions(r, names, foot):
    fig, axs = plt.subplots(2, math.ceil(len(names)/2), figsize=(13.2, 8), squeeze=False)
    for ax, name in zip(axs.flat, names):
        cm = r["evaluations"][name]["student_metrics"]["confusion_matrix"]; draw_cm(ax, cm, model_label(name).replace("\n", " "))
        one, axis = plt.subplots(figsize=(6.4, 5.9)); draw_cm(axis, cm, "Actual candidate counts; threshold = 0.5")
        save_plot(one, f"confusion_{name}.png", model_label(name).replace("\n", " ")+" · development confusion", "Student-development only. Rows = true proxy class; columns = predicted class.\n"+PROXY_NOTE)
    for ax in axs.flat[len(names):]: ax.set_visible(False)
    save_plot(fig, "confusion_matrices_all_models.png", "Binary-candidate confusion matrices: student-development", foot+"\nRows = true proxy; columns = prediction; raw counts, per-panel colour ranges. Negative candidates outnumber positive candidates.")


def plot_bayesian(r, foot):
    trials = r["bayesian"]["trials"]; x = np.array([t["trial"] for t in trials]); objectives = np.array([t["objective"] for t in trials]); fig, axs = plt.subplots(2, 1, figsize=(12.3, 8.2), sharex=True)
    axs[0].plot(x, objectives, color="#9aaec2", lw=1, zorder=1)
    symbols = {"measured_baseline_prior": ("s", "#64748b", "Measured untuned prior (reused CV)"), "seeded_initial_design": ("o", "#2266b2", "Seeded initial design"), "gaussian_process_expected_improvement": ("D", "#d77926", "GP expected-improvement proposal")}
    for proposal, (marker, color, label) in symbols.items():
        mask = np.array([t["proposal"] == proposal for t in trials]); axs[0].scatter(x[mask], objectives[mask], marker=marker, color=color, s=48, zorder=3, label=label)
    axs[0].plot(x, np.maximum.accumulate(objectives), color="#16884a", drawstyle="steps-post", lw=2, label="Best measured objective so far")
    axs[0].set(ylabel="Equal-trip OOF Top-1 proxy accuracy", title="Actual TRAIN selection objective; not student-development performance"); axs[0].legend(fontsize=8, loc="best"); axs[0].margins(y=.18)
    axs[1].plot(x, [t["cv_macro_f1"] for t in trials], "o-", color="#2266b2", label="Pooled OOF binary-candidate Macro F1")
    other = axs[1].twinx(); other.plot(x, [t["cv_log_loss"] for t in trials], "s--", color="#d77926", label="Pooled OOF candidate log loss")
    axs[1].set(xlabel="Trial index (0 = previously measured baseline prior)", ylabel="Macro F1", xticks=x); other.set_ylabel("Log loss (lower is better)")
    lines = axs[1].get_lines()+other.get_lines(); axs[1].legend(lines, [l.get_label() for l in lines], fontsize=8, loc="best")
    for ax in axs: ax.grid(); ax.set_axisbelow(True)
    save_plot(fig, "bayesian_trial_trace.png", f"Bayesian search: {len(trials)} recorded trials · {r['selection_model'].replace('_', ' ')}", foot+"\nSame trip-grouped TRAIN folds throughout. Trial 0 is measured/reused, not a new fit. Genuine GP surrogate + expected improvement.")


def plot_tuned(r, foot):
    names = [r["selection_model"], r["final_run_name"]]
    metrics = [("CV fold-mean candidate Macro F1", "cv", "macro_f1_fold_mean", None, True), ("Development observation Top-1", "evaluations", "top1_accuracy", "student_metrics", True), ("Development candidate Macro F1", "evaluations", "macro_f1", "student_metrics", True), ("Development candidate log loss", "evaluations", "log_loss", "student_metrics", False), ("Development inference seconds", "evaluations", "student_inference_seconds", None, False), ("TRAIN equal-trip OOF Top-1 objective", "cv", "objective", None, True)]
    fig, axs = plt.subplots(2, 3, figsize=(12.8, 8))
    improved = worsened = 0
    for ax, (title, source, key, nested, higher_better) in zip(axs.flat, metrics):
        vals = [(r[source][n][nested] if nested else r[source][n])[key] for n in names]; bars = ax.bar([0, 1], vals, color=["#2266b2", "#d77926"], width=.58); label_bars(ax, bars, 5)
        if vals[1] != vals[0]:
            if (vals[1] > vals[0]) == higher_better: improved += 1
            else: worsened += 1
        ax.set(xticks=[0, 1], xticklabels=["Untuned", "Tuned"], title=title, ylim=(0, max(vals)*1.22 if max(vals) > 0 else 1)); ax.grid(axis="y"); ax.set_axisbelow(True)
    gain = r["cv"][r["final_run_name"]]["objective"] - r["cv"][r["selection_model"]]["objective"]
    save_plot(fig, "tuned_vs_untuned.png", "Untuned vs Bayesian-tuned: identical data and definitions", foot+f"\nMeasured objective gain from tuning: {gain:+.6f} Top-1 accuracy ({gain*100:+.2f} pp), a small change. Across these six panels {improved} improved and {worsened} got worse for the tuned model; bars are unadjusted measurements and the y-axes start at zero. Untuned is the same algorithm before Bayesian search; log loss and seconds are lower-is-better.")
