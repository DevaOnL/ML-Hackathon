"""Observation-level, continuity and ambiguity-strata charts from measured experiment outputs only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from common import ROOT, save_plot, label_bars

PROXY_DIAGNOSTICS = ROOT/"experiments"/"proxy_diagnostics.json"


def stable_read_json(path: Path):
    """Read a finished upstream JSON once, refusing a file still being written."""
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size): raise RuntimeError(f"Upstream output changed during read; wait for its final publication: {path}")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def plot_observation_metrics(r, names, labels, x, colors, foot):
    student = [r["evaluations"][n]["student_observation_metrics"] for n in names]
    oof = []
    for n in names:
        m = r["cv"][n].get("observation_metrics")
        oof.append(m if m and "observation_edge_macro_f1" in m else None)
    fig, axs = plt.subplots(1, 3, figsize=(14.6, 5.7))
    w = .36
    bars = axs[0].bar(x-w/2, [s["observation_edge_macro_f1"] for s in student], w, color=colors, label="Student-development")
    label_bars(axs[0], bars, 4)
    kept = [i for i, m in enumerate(oof) if m is not None]
    if kept:
        bars2 = axs[0].bar([x[i]+w/2 for i in kept], [oof[i]["observation_edge_macro_f1"] for i in kept], w, color="#8eaac7", label="TRAIN out-of-fold")
        label_bars(axs[0], bars2, 4)
    for i, m in enumerate(oof):
        if m is None: axs[0].text(x[i]+w/2, .015, "OOF not retained", rotation=90, fontsize=7, ha="center", va="bottom", color="#64748b")
    axs[0].set(xticks=x, xticklabels=labels, ylabel="Observation edge Macro F1", ylim=(0, 1.08), title="Directed-edge classes, graph-namespaced")
    axs[0].legend(fontsize=8, loc="lower right"); axs[0].grid(axis="y"); axs[0].set_axisbelow(True)
    bars = axs[1].bar(x-w/2, [s["observation_categorical_log_loss_clipped"] for s in student], w, color="#2266b2", label="All observations, misses floored")
    bars2 = axs[1].bar(x+w/2, [s["observation_categorical_log_loss_given_candidate"] for s in student], w, color="#d77926", label="Reference candidate present")
    for b in (bars, bars2): label_bars(axs[1], b, 4)
    axs[1].set(xticks=x, xticklabels=labels, ylabel="Per-observation categorical log score", title="Categorical log score (lower is better)")
    axs[1].legend(fontsize=8); axs[1].grid(axis="y"); axs[1].set_axisbelow(True); axs[1].margins(y=.22)
    bars = axs[2].bar(x, [s["candidate_miss_rate"] for s in student], color=colors)
    label_bars(axs[2], bars, 4)
    axs[2].set(xticks=x, xticklabels=labels, ylabel="Miss rate", title="Reference arc missing from candidates")
    axs[2].grid(axis="y"); axs[2].set_axisbelow(True); axs[2].margins(y=.22)
    s = r["evaluations"][r["final_run_name"]]["student_observation_metrics"]
    save_plot(fig, "observation_level_metrics.png", "Observation-level metrics: separate label space and denominators", foot+f"\nThese scores are not comparable to the binary candidate-row charts. Edge Macro F1 classes are trip-qualified canonical directed arcs (namespace {s['edge_class_namespace']}) over the union of reference and predicted arcs. The categorical log score normalizes positive candidate probabilities within each observation; absent targets get zero mass and the floored variant includes candidate misses ({s['candidate_miss_count']:,} of {s['observations']:,} student observations for the final model, floor 1e-7). Emission log score, not a calibration claim.")


def plot_continuity(r, names, labels, x, colors, foot):
    student = [r["evaluations"][n]["student_observation_metrics"] for n in names]
    final = r["final_run_name"]
    per_trip = sorted(r["evaluations"][final]["student_observation_metrics"]["per_trip"], key=lambda t: int(t["trip_id"]))
    fig, axs = plt.subplots(1, 2, figsize=(13.8, 5.8), gridspec_kw={"width_ratios": [1.55, 1]})
    w = .36
    bars = axs[0].bar(x-w/2, [s["raw_classifier_continuity"] or 0 for s in student], w, color=colors, label="Adjacent pairs on same or connected arc")
    bars2 = axs[0].bar(x+w/2, [s["edge_change_continuity"] or 0 for s in student], w, color="#8eaac7", label="Edge changes that stay connected")
    for b in (bars, bars2): label_bars(axs[0], b, 4)
    axs[0].set(xticks=x, xticklabels=labels, ylabel="Fraction of adjacent pairs", ylim=(0, 1.1), title="Pointwise argmax assignments, no routing")
    axs[0].legend(fontsize=8, loc="lower right"); axs[0].grid(axis="y"); axs[0].set_axisbelow(True)
    tx = np.arange(len(per_trip))
    bars = axs[1].bar(tx, [t["raw_classifier_continuity"] or 0 for t in per_trip], color="#d77926")
    label_bars(axs[1], bars, 4)
    axs[1].set(xticks=tx, xticklabels=["trip "+t["trip_id"] for t in per_trip], ylabel="Raw classifier continuity", ylim=(0, 1.1), title=labels[names.index(final)].replace("\n", " ")+" per trip")
    axs[1].grid(axis="y"); axs[1].set_axisbelow(True); axs[1].tick_params(axis="x", labelsize=8, rotation=30)
    s = r["evaluations"][final]["student_observation_metrics"]
    save_plot(fig, "raw_classifier_continuity.png", "Raw classifier continuity before any routing", foot+f"\nAn adjacent observation pair is continuous when the canonical directed arc is identical or the previous head equals the next tail; no routed connectors are drawn, so discontinuities stay visible. Denominator for the final model: {s['adjacent_observation_pairs']:,} adjacent pairs and {s['edge_change_pairs']:,} edge changes across the 10 student-development trips. Route continuity after HMM/Viterbi routing is measured separately by the route matcher and is not shown here.")


def plot_ambiguity_strata(r, foot):
    diagnostics, digest = stable_read_json(PROXY_DIAGNOSTICS)
    strata = [("all", "All"), ("nonambiguous", "Non-\nambiguous"), ("ambiguous", "Ambiguous"), ("proxy_distance_le_30m", "Proxy dist\n<= 30 m"), ("proxy_distance_gt_30m", "Proxy dist\n> 30 m")]
    fig, axs = plt.subplots(2, 2, figsize=(13.4, 9.2), sharex="col")
    for col, (split, title) in enumerate([("train", "TRAIN out-of-fold"), ("student_test", "Student-development")]):
        d = diagnostics["splits"][split]; sens = d["sensitivity"]; xx = np.arange(len(strata))
        for row, (metric, ylab) in enumerate([("macro_f1", "Binary-candidate Macro F1"), ("top1_accuracy", "Observation Top-1 (misses included)")]):
            ax = axs[row][col]
            bars = ax.bar(xx, [sens[k][metric] for k, _ in strata], .6, color="#2266b2" if row == 0 else "#d77926")
            label_bars(ax, bars, 4)
            ax.set(ylabel=ylab, ylim=(0, 1.08), title=title if row == 0 else None)
            if row == 1:
                base = d["nearest_geometry_baseline"]
                ax.axhline(base["top1_accuracy"], color="#64748b", ls="--", lw=1.4, label="Nearest-geometry baseline (all)")
                if base.get("top1_nonambiguous") is not None: ax.axhline(base["top1_nonambiguous"], color="#94a3b8", ls=":", lw=1.4, label="Nearest-geometry (non-ambiguous)")
                ax.legend(fontsize=7, loc="lower right")
            ax.grid(axis="y"); ax.set_axisbelow(True)
            if row == 1: ax.set(xticks=xx, xticklabels=[lab for _, lab in strata])
    train, student = diagnostics["splits"]["train"], diagnostics["splits"]["student_test"]
    n_train = "/".join(f"{train['sensitivity'][k]['observations']:,}" for k, _ in strata)
    n_student = "/".join(f"{student['sensitivity'][k]['observations']:,}" for k, _ in strata)
    save_plot(fig, "ambiguity_strata_sensitivity.png", "Ambiguity and proxy-distance strata: same measured predictions, sliced", foot+f"\nTRAIN panel scores {train['model_run']} out-of-fold on all 80 trips; student panel scores the final all-80 fit on the 10 development trips. Stratum observation counts left to right: TRAIN {n_train}; student {n_student}. Dashed lines are the distance-only nearest-geometry ranking baseline, shown for context. "+diagnostics["target_warning"])
    return {"path": "experiments/proxy_diagnostics.json", "sha256": digest}
