"""All-trip, nonfabricated development EDA from the completed audit + raw tracks."""
from __future__ import annotations
import hashlib
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from common import ROOT, OUT, SPLITS, SPLIT_LABELS, COLORS, read_json, sha256, write_json, save_plot, label_bars, log_bins, local_xy


def load_audit():
    directory = ROOT / "analysis" / "data_audit"
    summary = read_json(directory / "summary.json")
    trips = read_json(directory / "per_trip.json")
    manifest = read_json(directory / "manifest.json")
    if not summary["all_360_read_complete"] or not summary["all_files_parse_ok"]:
        raise ValueError("Complete, successful source audit is required")
    for split, count in {"train": 80, "student_test": 10}.items():
        if len([t for t in trips if t["split"] == split]) != count:
            raise ValueError(f"Incomplete EDA scope for {split}")
    if set(t["split"] for t in trips) != set(SPLITS):
        raise ValueError("Do not include evaluator data in development EDA")
    return summary, trips, manifest


def load_all_tracks(manifest):
    tracks, provenance = {}, []
    for file in manifest:
        if file["extension"] != "track" or file["split"] not in SPLITS: continue
        path = ROOT / file["relative_path"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != file["sha256"]:
            raise ValueError(f"Track no longer matches audited input: {path}")
        a = np.loadtxt(path, ndmin=2)
        if a.shape != (file["data_row_count"], 3) or not np.isfinite(a).all():
            raise ValueError(f"Invalid/incomplete track: {path}")
        tracks[(file["split"], file["trip_id"])] = a
        provenance.append({"path": file["relative_path"], "sha256": file["sha256"], "rows": len(a)})
    if len(tracks) != 90: raise ValueError("EDA must load every one of the 90 permitted tracks")
    return tracks, provenance


def plot_inventory(summary, manifest):
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.6))
    x = np.arange(len(SPLITS))
    labels = [SPLIT_LABELS[s].replace(" (", "\n(") for s in SPLITS]
    for ax, key, title in [(axs[0, 0], "trip_count", "Trips"), (axs[0, 1], "file_count", "Audited source files")]:
        vals = [summary["splits"][s][key] for s in SPLITS]
        bars = ax.bar(x, vals, color=[COLORS[s] for s in SPLITS], width=.58); label_bars(ax, bars)
        ax.set(xticks=x, xticklabels=labels, title=title, ylim=(0, max(vals)*1.2)); ax.grid(axis="y"); ax.set_axisbelow(True)
    ax = axs[1, 0]; types = ["track", "nodes", "arcs", "route"]; xx = np.arange(len(types))
    for i, split in enumerate(SPLITS):
        bars = ax.bar(xx + (i-.5)*.34, [summary["splits"][split]["row_totals"][t] for t in types], .34, label=SPLIT_LABELS[split], color=COLORS[split]); label_bars(ax, bars)
    ax.set(xticks=xx, xticklabels=["GPS points", "Road nodes", "Directed arcs", "Reference arcs"], yscale="log", ylabel="Rows (log scale)", title="Rows by actual file type")
    ax.margins(y=.3); ax.grid(axis="y"); ax.set_axisbelow(True); ax.legend(fontsize=8)
    ax = axs[1, 1]; bottom = np.zeros(2)
    for kind, color in zip(types, ["#2166b2", "#6d93bb", "#a7bed5", "#d77926"]):
        values = np.array([sum(m["bytes"] for m in manifest if m["split"] == s and m["extension"] == kind)/1e6 for s in SPLITS])
        ax.bar(x, values, bottom=bottom, label="."+kind, color=color, width=.58); bottom += values
    for i, value in enumerate(bottom): ax.text(i, value + max(bottom)*.025, f"{value:.2f} MB", ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=labels, ylabel="Uncompressed decimal MB", title="Complete dataset storage", ylim=(0, max(bottom)*1.2)); ax.grid(axis="y"); ax.set_axisbelow(True); ax.legend(ncols=2, fontsize=8)
    save_plot(fig, "dataset_inventory.png", "Dataset inventory: all 80 training + 10 student-development trips", "Source: completed byte/row audit, 360 original files. Student-development is not the reserved evaluator partition.")


def plot_quality(trips, manifest):
    fig, axs = plt.subplots(1, 2, figsize=(13.6, 7))
    categories = ["Missing required files", "Blank rows / missing fields", "Numeric / structural parse errors", "Invalid GPS coordinates", "Invalid road-node coordinates", "Invalid arc endpoints", "Invalid reference arc IDs", "Duplicate GPS timestamps", "Nonpositive timestamp steps"]
    anomaly_labels = ["Repeated GPS coordinates", "Repeated directed road pairs", "Zero-length road arc chords", "Road self-loops", "Broken reference-route joins", "GPS steps >50 m/s (if seconds)"]
    values, anomalies = {}, {}
    for s in SPLITS:
        tt = [t for t in trips if t["split"] == s]; mm = [m for m in manifest if m["split"] == s]
        values[s] = [len(tt)*4-len(mm), sum(m["blank_rows"] for m in mm), sum(m["parse_error_count"] for m in mm),
                     sum(t["track"]["invalid_geographic_rows"] for t in tt), sum(t["nodes"]["invalid_geographic_rows"] for t in tt),
                     sum(t["arcs"]["invalid_endpoint_rows"] for t in tt), sum(t["route_validity"]["invalid_zero_based_arc_ids"] for t in tt),
                     sum(t["track"]["duplicate_timestamp_extra_rows"] for t in tt), sum(t["track"]["nonpositive_time_deltas"] for t in tt)]
        anomalies[s] = [sum(t["track"]["duplicate_coordinate_extra_rows"] for t in tt), sum(t["arcs"]["duplicate_directed_pair_extra_arcs"] for t in tt),
                        sum(t["arcs"]["zero_chord_length_arcs"] for t in tt), sum(t["arcs"]["self_loops"] for t in tt),
                        sum(t["route_validity"]["arc_index_offset_0"]["broken_joins"] for t in tt), sum(t["track"]["steps_over_50mps_if_seconds"] for t in tt)]
    for ax, names, source in [(axs[0], categories, values), (axs[1], anomaly_labels, anomalies)]:
        y = np.arange(len(names))
        for i, s in enumerate(SPLITS):
            bars = ax.barh(y + (i-.5)*.34, source[s], .34, color=COLORS[s], label=SPLIT_LABELS[s]); label_bars(ax, bars, horizontal=True)
        ax.set(yticks=y, yticklabels=names, xlabel="Actual count (zeros explicitly printed)"); ax.invert_yaxis(); ax.grid(axis="x"); ax.set_axisbelow(True)
        ax.set_xlim(0, max(1, max(v for vv in source.values() for v in vv))*1.25); ax.legend(loc="lower right", fontsize=8)
    axs[0].set_title("Missing / invalid values: audited checks"); axs[1].set_title("Non-missing anomalies: retained, not silently repaired")
    save_plot(fig, "missing_invalid_and_anomalies.png", "Data quality across every permitted trip", "Source: complete file and cross-file audit. Blank/malformed rows are counted; missing values are not inferred from absent timestamps.\nRepeated coordinates can represent stops. Apparent speed uses GPS displacement / supplied seconds, not independently measured vehicle speed.")
    return {s: {"missing_invalid_counts": dict(zip(categories, values[s])), "other_anomalies": dict(zip(anomaly_labels, anomalies[s]))} for s in SPLITS}


def plot_trajectory_distributions(trips):
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8))
    points = [t["track"]["count"] for t in trips]; durations = [t["track"]["time_duration"]/60 for t in trips]
    for s in SPLITS:
        tt = [t for t in trips if t["split"] == s]; color = COLORS[s]
        axs[0, 0].hist([t["track"]["count"] for t in tt], bins=log_bins(points), histtype="step", lw=2, color=color, label=SPLIT_LABELS[s])
        axs[0, 1].hist([t["track"]["time_duration"]/60 for t in tt], bins=log_bins(durations), histtype="step", lw=2, color=color, label=SPLIT_LABELS[s])
        axs[1, 0].scatter([t["track"]["count"] for t in tt], [t["route_validity"]["count"] for t in tt], s=28, color=color, alpha=.75, label=SPLIT_LABELS[s])
        axs[1, 1].scatter([t["route_validity"]["path_endpoint_chord_length_m"]/1000 for t in tt], [t["track"]["total_point_to_point_distance_m"]/1000 for t in tt], s=28, color=color, alpha=.75, label=SPLIT_LABELS[s])
    axs[0, 0].set(xscale="log", xlabel="GPS observations per trip (log)", ylabel="Trips", title="Observed trajectory sizes")
    axs[0, 1].set(xscale="log", xlabel="Timestamp span, minutes (log)", ylabel="Trips", title="Duration includes observed timestamp gaps")
    axs[1, 0].set(xscale="log", yscale="log", xlabel="GPS observations (log)", ylabel="Reference traversal rows (log)", title="Route rows are not per-GPS labels")
    max_km = max(t["track"]["total_point_to_point_distance_m"] for t in trips)/1000*1.04
    axs[1, 1].plot([0, max_km], [0, max_km], ls="--", color="#64748b", lw=1, label="Equal chord sums")
    axs[1, 1].set(xlabel="Reference arc endpoint-chord sum, km", ylabel="Raw GPS point-to-point sum, km", title="Different sampling / noise / representations")
    for ax in axs.flat: ax.grid(); ax.set_axisbelow(True); ax.legend(fontsize=8)
    save_plot(fig, "trajectory_distributions.png", "Trajectory size, duration and reference-route structure", "All 80 training and 10 student-development trips; no subsampling. Chord sums do not claim exact physical road length.\nReference routes are chronological arc traversals, not an independently supplied label for every GPS observation.")


def plot_timestamps(trips, tracks):
    pooled = {s: np.concatenate([np.diff(a[:, 2]) for (split, _), a in tracks.items() if split == s]) for s in SPLITS}
    labels = ["≈1 s", "Other ≤2 s", ">2–10 s", ">10–60 s", ">60 s", "≤0 s"]; stats, counts = {}, {}
    for s, dt in pooled.items():
        regular = np.isclose(dt, 1, rtol=0, atol=1e-6)
        categories = [regular, (~regular)&(dt > 0)&(dt <= 2), (dt > 2)&(dt <= 10), (dt > 10)&(dt <= 60), dt > 60, dt <= 0]
        counts[s] = [int(mask.sum()) for mask in categories]; assert sum(counts[s]) == len(dt)
        stats[s] = {"steps": len(dt), "approximately_one_second_steps": int(regular.sum()), "regular_fraction": float(regular.mean()), "maximum_gap_seconds": float(dt.max()), "minimum_gap_seconds": float(dt.min()), "nonpositive_steps": int((dt <= 0).sum()), "approximately_one_second_tolerance_seconds": 1e-6, "exact_not_one_audit_count": sum(t["track"]["time_delta_not_one"] for t in trips if t["split"] == s), "category_counts": dict(zip(labels, counts[s]))}
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.3)); x = np.arange(len(labels))
    for i, s in enumerate(SPLITS):
        bars = axs[0].bar(x+(i-.5)*.36, counts[s], .36, color=COLORS[s], label=SPLIT_LABELS[s]); label_bars(axs[0], bars)
        tt = sorted([t for t in trips if t["split"] == s], key=lambda t: t["track"]["time_delta"]["p100"]); y = [t["track"]["time_delta"]["p100"] for t in tt]
        axs[1].scatter(np.arange(1, len(tt)+1), y, color=COLORS[s], s=28, label=SPLIT_LABELS[s])
        for j in np.argsort(y)[-3:]: axs[1].annotate(tt[j]["trip_id"], (j+1, y[j]), xytext=(-4, 7), textcoords="offset points", ha="right", fontsize=7, color=COLORS[s])
    axs[0].set(xticks=x, xticklabels=labels, yscale="symlog", ylabel="Observed consecutive GPS steps (symlog counts)", title="Actual time-step categories")
    axs[1].set(yscale="log", xlabel="Trip rank by maximum gap (within partition)", ylabel="Maximum timestamp step, seconds (log)", title="Large gaps are concentrated in a few trips")
    for ax in axs: ax.grid(axis="y"); ax.set_axisbelow(True); ax.legend(fontsize=8)
    note = "; ".join(f"{SPLIT_LABELS[s]}: {stats[s]['regular_fraction']:.2%} ≈1 s, max {stats[s]['maximum_gap_seconds']:,.0f} s" for s in SPLITS)
    save_plot(fig, "timestamp_gaps.png", "Timestamp ordering and sampling gaps: every consecutive pair", note+".\n≈1 s uses absolute tolerance 1e−6 s; exact audit counts are retained in eda_summary.json. No gaps filled or timestamps invented.")
    return stats


def plot_network(trips):
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8)); comps = [t["arcs"][k] for t in trips for k in ("weak_components", "strong_components")]; bins = log_bins(comps, 16)
    for s in SPLITS:
        tt = [t for t in trips if t["split"] == s]; color = COLORS[s]
        axs[0, 0].scatter([t["nodes"]["count"] for t in tt], [t["arcs"]["count"] for t in tt], s=28, alpha=.75, color=color, label=SPLIT_LABELS[s])
        axs[0, 1].hist([t["arcs"]["weak_components"] for t in tt], bins=bins, histtype="step", lw=2, color=color, label=f"{s}: weak")
        axs[0, 1].hist([t["arcs"]["strong_components"] for t in tt], bins=bins, histtype="step", lw=1.7, ls="--", color=color, label=f"{s}: strong")
        axs[1, 0].scatter([t["arcs"]["largest_weak_fraction"] for t in tt], [t["arcs"]["largest_strong_fraction"] for t in tt], s=28, alpha=.75, color=color, label=SPLIT_LABELS[s])
    axs[0, 0].set(xscale="log", yscale="log", xlabel="Road nodes per trip (log)", ylabel="Directed arcs per trip (log)", title="Network size varies substantially")
    axs[0, 1].set(xscale="log", xlabel="Components per trip (log)", ylabel="Trips", title="Weak vs strong graph connectivity")
    axs[1, 0].plot([0, 1], [0, 1], "--", color="#64748b", lw=1)
    axs[1, 0].set(xlabel="Node fraction in largest weak component", ylabel="Node fraction in largest strong component", title="One large component can dominate")
    axs[1, 0].xaxis.set_major_formatter(PercentFormatter(1)); axs[1, 0].yaxis.set_major_formatter(PercentFormatter(1)); x = np.arange(3)
    for i, s in enumerate(SPLITS):
        tt = [t for t in trips if t["split"] == s]; values = [sum(t["arcs"][k] for t in tt) for k in ("isolated_nodes", "nodes_with_zero_indegree", "nodes_with_zero_outdegree")]
        bars = axs[1, 1].bar(x+(i-.5)*.36, values, .36, color=COLORS[s], label=SPLIT_LABELS[s]); label_bars(axs[1, 1], bars)
    axs[1, 1].set(xticks=x, xticklabels=["Isolated", "Zero in-degree", "Zero out-degree"], ylabel="Nodes summed over trip-specific graphs", title="Directed dead ends are not isolated nodes"); axs[1, 1].margins(y=.18)
    for ax in axs.flat: ax.grid(); ax.set_axisbelow(True); ax.legend(fontsize=8)
    save_plot(fig, "road_network_sizes_connectivity.png", "Road-network sizes and connectivity: all 90 trip graphs", "Complete directed-graph audit, computed per trip. Networks can overlap geographically; totals are not a unique global road network.\nWeak connectivity ignores direction; strong connectivity requires directed paths both ways. Zero isolated nodes does not mean connected.")


def plot_geographic_coverage(tracks):
    fig, axs = plt.subplots(2, 1, figsize=(12.5, 8.3), sharex=True, sharey=True); all_coords = np.vstack([a[:, :2] for a in tracks.values()])
    for ax, split in zip(axs, SPLITS):
        total = 0
        for (s, _), a in sorted(tracks.items()):
            if s != split: continue
            ax.plot(a[:, 0], a[:, 1], color=COLORS[s], lw=.65, alpha=.6); total += len(a)
        ax.set_title(f"{SPLIT_LABELS[split]} · {total:,} actual GPS observations", loc="left"); ax.set(ylabel="Latitude (degrees)"); ax.grid(); ax.set_axisbelow(True)
    axs[-1].set_xlabel("Longitude (degrees; not a distance axis)"); margin = (all_coords.max(axis=0) - all_coords.min(axis=0))*.04
    axs[0].set(xlim=(all_coords[:, 0].min()-margin[0], all_coords[:, 0].max()+margin[0]), ylim=(all_coords[:, 1].min()-margin[1], all_coords[:, 1].max()+margin[1]))
    save_plot(fig, "gps_geographic_coverage.png", "Geographic GPS coverage: every observed trajectory", "All raw positions from 80 training and 10 student-development tracks; shared longitude/latitude axes; no web tiles.\nAt this multi-region scale a trip can appear as a compact point; the local-scale galleries below show every trajectory separately.")


def plot_galleries(tracks):
    for split in SPLITS:
        items = [(tid, a) for (s, tid), a in sorted(tracks.items()) if s == split]; page_size = 20 if split == "train" else 10; cols = 4 if split == "train" else 2
        for page, start in enumerate(range(0, len(items), page_size), 1):
            subset = items[start:start+page_size]; rows = math.ceil(len(subset)/cols); fig, axs = plt.subplots(rows, cols, figsize=(12.6, rows*2.25+1.1), squeeze=False)
            for ax, (tid, a) in zip(axs.flat, subset):
                xy = local_xy(a[:, :2]); ax.plot(xy[:, 0], xy[:, 1], color=COLORS[split], lw=.7, alpha=.8)
                ax.scatter(xy[0, 0], xy[0, 1], s=13, color="#148453", zorder=3); ax.scatter(xy[-1, 0], xy[-1, 1], s=16, marker="x", color="#ad3434", zorder=3)
                ax.set_title(f"{tid} · {len(a):,} GPS rows", fontsize=9, loc="left"); ax.set_aspect("equal", adjustable="datalim"); ax.tick_params(labelsize=7); ax.grid()
            for ax in axs.flat[len(subset):]: ax.set_visible(False)
            save_plot(fig, f"gps_trip_gallery_{split}_{page:02d}.png", f"Every raw GPS trajectory · {SPLIT_LABELS[split]} · page {page}", "Every observation included. Green dot = first GPS row; red × = last. Independent local-km panel axes, centered at median coordinate.\nEquirectangular projection uses cos(median latitude). Panels have different scales; timestamps/gaps are not interpolated.")


def generate_eda(*, gallery=True):
    summary, trips, manifest = load_audit(); tracks, provenance = load_all_tracks(manifest)
    plot_inventory(summary, manifest); quality = plot_quality(trips, manifest); plot_trajectory_distributions(trips)
    timestamps = plot_timestamps(trips, tracks); plot_network(trips); plot_geographic_coverage(tracks)
    if gallery: plot_galleries(tracks)
    evidence = {"scope": "All 80 training + 10 student-development trips; no evaluator files", "source_audit": {p: sha256(ROOT/p) for p in ["analysis/data_audit/summary.json", "analysis/data_audit/per_trip.json", "analysis/data_audit/manifest.json"]}, "trip_counts": {s: sum(t["split"] == s for t in trips) for s in SPLITS}, "gps_rows": {s: sum(len(a) for (ss, _), a in tracks.items() if ss == s) for s in SPLITS}, "quality": quality, "timestamps": timestamps, "track_input_hashes": provenance, "road_network_totals": {s: {key: sum(t["arcs"][key] for t in trips if t["split"] == s) for key in ["count", "weak_components", "strong_components", "isolated_nodes"]} for s in SPLITS}, "reference_warning": "Chronological arc traversals, not pointwise labels; validity checks do not imply independently verified physical ground truth."}
    write_json(OUT/"eda_summary.json", evidence)
    print(f"EDA generated from {len(tracks)} tracks / {sum(len(a) for a in tracks.values()):,} GPS rows")
