"""Shared plotting style, provenance, and strict output-scope enforcement."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "visualizations"
PLOTS = OUT / "plots"
MAPS = OUT / "maps"
SPLITS = ("train", "student_test")
SPLIT_LABELS = {"train": "Training (80 trips)", "student_test": "Student-development (10 trips)"}
COLORS = {"train": "#2266b2", "student_test": "#d77926", "gps": "#2166e8", "roads": "#929ba7", "reference": "#16884a", "prediction": "#f47b13"}
PROXY_NOTE = "Proxy labels: nearest reference-route segment + weak GPS-motion orientation; NOT supplied pointwise ground truth."
SAVED = []
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                     "figure.facecolor": "white", "axes.facecolor": "#f8fafc", "axes.spines.top": False,
                     "axes.spines.right": False, "grid.color": "#dbe3ec", "grid.alpha": .65,
                     "savefig.facecolor": "white", "legend.frameon": False})


def safe_output(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise ValueError(f"Writes outside visualizations/ are forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    safe_output(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def save_plot(fig, filename: str, title: str, note: str) -> Path:
    path = safe_output(PLOTS / filename)
    fig.suptitle(title, fontsize=16, fontweight="bold", x=.045, ha="left", y=.985)
    fig.text(.045, .018, note, fontsize=8, color="#475569", va="bottom")
    fig.tight_layout(rect=(.025, .075, .985, .935), h_pad=2.3, w_pad=2.0)
    fig.savefig(path, dpi=150, pil_kwargs={"compress_level": 8})
    plt.close(fig)
    SAVED.append({"path": str(path.relative_to(ROOT)), "title": title, "caption": note})
    return path


def label_bars(ax, bars, precision=0, *, horizontal=False):
    for bar in bars:
        value = bar.get_width() if horizontal else bar.get_height()
        label = f"{value:,.{precision}f}"
        if horizontal:
            ax.annotate(label, (value, bar.get_y() + bar.get_height()/2), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
        else:
            ax.annotate(label, (bar.get_x() + bar.get_width()/2, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)


def log_bins(values, count=20):
    a = np.asarray(values, dtype=float)
    a = a[a > 0]
    return np.geomspace(a.min()*.95, a.max()*1.05, count)


def local_xy(coords):
    ref = np.median(coords, axis=0)
    return np.column_stack(((coords[:, 0]-ref[0])*math.cos(math.radians(ref[1]))*111.195, (coords[:, 1]-ref[1])*111.195))


def update_figure_manifest():
    path = OUT / "figure_manifest.json"
    previous = read_json(path) if path.exists() else []
    figures = {row["path"]: row for row in previous}
    figures.update({row["path"]: row for row in SAVED})
    for row in figures.values():
        p = ROOT/row["path"]; row["bytes"] = p.stat().st_size; row["sha256"] = sha256(p)
        name = p.name
        if name == "representative_road_route.png":
            source = "Original student_test trip 00000009 .nodes/.arcs/.track/.route plus explicitly labelled prediction (see map metadata)"
            partition = "student_test development trip 00000009 only"
        elif "gallery_train" in name:
            source = "Every original TRAIN .track observation, SHA256-checked against analysis/data_audit/manifest.json"
            partition = "train: all 80 trips, pages 1–4 together"
        elif "gallery_student_test" in name:
            source = "Every original student_test .track observation, SHA256-checked against analysis/data_audit/manifest.json"
            partition = "student_test development: all 10 trips"
        elif name in {"dataset_inventory.png", "missing_invalid_and_anomalies.png", "trajectory_distributions.png", "timestamp_gaps.png", "road_network_sizes_connectivity.png", "gps_geographic_coverage.png"}:
            source = "analysis/data_audit/{summary,per_trip,manifest}.json plus all original .track files; visualizations/eda_summary.json hashes"
            partition = "all 80 train + 10 student_test development trips; no evaluator data"
        else:
            source = "Complete stable experiments/results.json; visualizations/experiment_chart_provenance.json records SHA256 and exact sampling"
            partition = "TRAIN trip-grouped CV + student_test development; model chart captions specify sampling"
        row.setdefault("source", source); row.setdefault("partition", partition)
        row["status"] = "stable_interim_geometry_baseline_not_final" if "interim" in row["caption"].lower() else "stable_actual"
    write_json(path, list(figures.values()))
    print(f"Verified {len(figures)} figure records; generated assets remain under visualizations/")
