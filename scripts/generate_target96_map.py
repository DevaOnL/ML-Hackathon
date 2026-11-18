#!/usr/bin/env python3
"""Gated, report-only Target96 map of the already frozen development trip 9.

No model deserialization, prediction, training, tuning, HTTP, or evaluator access.
Real-data modes require both a Lead accepted notice and the exact accepted-file
SHA. They validate acceptance before touching any release, comparison, or raw
trip input. --self-test uses only synthetic in-memory fixtures. Existing maps and
all prediction/artifact bytes are read-only; outputs are create-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from mapmatch.data import load_graph, load_route, load_track, validate_route
from mapmatch.metrics import score_routes

TRIP_ID = "00000009"
ART = ROOT / "artifacts/target96"
EXP = ROOT / "experiments/target96"
ACCEPTANCE = ART / "acceptance.json"
RELEASE = ART / "release_manifest.json"
SUMMARY = EXP / "student_comparison/summary.json"
COMPARISON = EXP / f"student_comparison/{TRIP_ID}.json"
SELECTION = EXP / "precision/frozen_selection.json"
ROUTE_PROTOCOL = EXP / "precision/PROTOCOL.json"
TRAIN_PROTOCOL = ROOT / "experiments/accuracy_improvement/PROTOCOL.json"
PREFIX = ROOT / "Student folder/student_test" / TRIP_ID / TRIP_ID
PREDICTION = ROOT / "outputs/target96/candidate" / TRIP_ID / f"{TRIP_ID}.route"
MAP_DIR = ROOT / "visualizations/target96"
HTML_PATH = MAP_DIR / f"accepted_development_trip_{TRIP_ID}.html"
PNG_PATH = HTML_PATH.with_suffix(".png")
EVIDENCE_PATH = ROOT / "reports/target96/map_evidence.json"
VENDOR = ROOT / "frontend/vendor/leaflet"
ASSET_NAMES = ("leaflet.js", "leaflet.css", "images/layers.png", "images/layers-2x.png")
RELEASE_FILES = {"model.joblib", "training_metadata.json", "match_config.json", "arc_policy.json"}
RELEASE_SOURCES = {
    "scripts/target96_release.py", "mapmatch/target96_arc.py", "mapmatch/target96.py",
    "mapmatch/matcher.py", "mapmatch/metrics.py", "mapmatch/features.py",
    "mapmatch/accuracy_features.py", "mapmatch/accuracy_artifacts.py",
    "mapmatch/accuracy_matcher.py", "mapmatch/learning.py", "backend/emissions.py",
}
COLORS = {"network": "#929ba7", "gps": "#2166e8", "reference": "#16884a", "prediction": "#e46f0b"}
VIEWPORT = {"width": 1920, "height": 1080}
HEX_SHA = re.compile(r"[0-9a-f]{64}\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def sha_value(value: object) -> bool:
    return isinstance(value, str) and bool(HEX_SHA.fullmatch(value))


def relative(path: Path) -> str:
    return path.absolute().relative_to(ROOT).as_posix()


def repo_path(value: str) -> Path:
    require(isinstance(value, str) and bool(value) and "\\" not in value, "invalid repository path")
    path = Path(value)
    require(not path.is_absolute() and path.as_posix() == value
            and not any(part in (".", "..") for part in path.parts), "noncanonical repository path")
    return ROOT / path


def regular_input(path: Path) -> Path:
    path = path.absolute()
    require(path.is_relative_to(ROOT) and path.resolve() == path, "input outside canonical repository")
    require(not any(part.is_symlink() for part in (path, *path.parents)), "symlink input forbidden")
    require(path.is_file(), f"required input missing: {relative(path)}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with regular_input(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path: Path, pins: dict, expected: str | None = None) -> str:
    digest = sha256_file(path)
    require(expected is None or (sha_value(expected) and digest == expected),
            f"input hash mismatch: {relative(path)}")
    key = relative(path)
    require(key not in pins or pins[key] == digest, f"input changed during read: {key}")
    pins[key] = digest
    return digest


def pinned_json(path: Path, pins: dict, expected: str | None = None) -> dict:
    pin(path, pins, expected)
    with regular_input(path).open(encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {token}")))
    require(isinstance(value, dict), f"JSON must be an object: {relative(path)}")
    pin(path, pins, pins[relative(path)])
    return value


def object_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def integer_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def safe_output(path: Path) -> Path:
    path = path.absolute()
    require(path.resolve() == path and (path.is_relative_to(MAP_DIR) or path == EVIDENCE_PATH),
            "output outside Target96 map scope")
    require(not any(part.is_symlink() for part in (path, *path.parents)), "symlink output forbidden")
    require(not path.exists(), f"create-only output already exists: {relative(path)}")
    return path


def write_new(path: Path, value: str | bytes) -> None:
    path = safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        with path.open("xb") as stream:
            stream.write(value)
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(value)


def validate_acceptance(acceptance: dict) -> None:
    require(acceptance.get("accepted") is True and acceptance.get("target_met") is True,
            "Target96 has not been accepted")
    score = acceptance.get("observed_edge_f1")
    require(finite_number(score) and .96 <= score <= 1., "accepted observed_edge_f1 must be finite and >= 0.96")
    require(type(acceptance.get("failed_trips")) is int and acceptance["failed_trips"] == 0, "accepted comparison has failures")
    require(finite_number(acceptance.get("route_continuity")) and acceptance["route_continuity"] == 1.,
            "accepted routes must have continuity one")
    require(finite_number(acceptance.get("mean_length_f1")) and 0 <= acceptance["mean_length_f1"] <= 1,
            "accepted length F1 missing or invalid")
    require(acceptance.get("not_official_or_blind") is True, "development/nonofficial disclosure missing")
    require(acceptance.get("production_switched") is False, "acceptance snapshot is not an offline candidate")
    for key in ("release_manifest_sha256", "comparison_sha256"):
        require(sha_value(acceptance.get(key)), f"acceptance missing valid {key}")


def compare_metrics(actual: dict, expected: dict, label: str) -> None:
    require(bool(expected), f"empty {label}")
    for key, value in expected.items():
        observed = actual.get(key)
        equal = (finite_number(observed) and math.isclose(observed, value, rel_tol=1e-12, abs_tol=1e-12)
                 if finite_number(value) else type(observed) is type(value) and observed == value)
        require(equal, f"{label} differs: {key}")


def validate_summary(acceptance: dict, summary: dict, release_sha: str) -> dict:
    require(summary.get("release_manifest_sha256") == release_sha == acceptance["release_manifest_sha256"],
            "acceptance/summary release binding differs")
    require(type(summary.get("trips")) is int and summary["trips"] == 10
            and type(summary.get("observations")) is int and summary["observations"] == 19160,
            "accepted comparison must cover all ten trips and all 19160 observations")
    require(summary.get("target_met") is True and summary.get("target_threshold") == .96, "summary target gate failed")
    require(summary.get("selection_frozen_before_student") is True and summary.get("evaluator_used") is False
            and summary.get("reference_after_all_predictions") is True, "comparison separation flags failed")
    candidate, baseline = summary.get("candidate"), summary.get("baseline")
    require(isinstance(candidate, dict) and isinstance(baseline, dict), "candidate/baseline summary missing")
    compare_metrics(candidate, {"edge_f1": acceptance["observed_edge_f1"], "failures": 0,
                               "route_continuity": 1., "length_f1": acceptance["mean_length_f1"]}, "acceptance/summary metrics")
    compare_metrics(baseline, {"edge_f1": acceptance.get("baseline_edge_f1")}, "baseline acceptance metric")
    require(finite_number(baseline.get("length_f1")) and candidate["length_f1"] >= baseline["length_f1"] - .002,
            "accepted candidate violates the length-F1 gate")
    bindings = summary.get("trip_bindings")
    require(isinstance(bindings, list) and len(bindings) == 10, "ten immutable trip bindings required")
    seen, selected = set(), None
    for row in bindings:
        require(isinstance(row, dict), "malformed trip binding")
        trip = row.get("trip_id")
        require(isinstance(trip, str) and re.fullmatch(r"[0-9]{8}", trip) and trip not in seen, "duplicate/invalid bound trip")
        require(row.get("path") == f"experiments/target96/student_comparison/{trip}.json"
                and sha_value(row.get("sha256")), "noncanonical/unhashed trip binding")
        seen.add(trip)
        if trip == TRIP_ID:
            selected = row
    require(selected is not None, "trip 00000009 missing from accepted comparison")
    return selected


def validate_trip_record(comparison: dict, release_sha: str, release_source_sha: str) -> dict:
    require(comparison.get("trip_id") == TRIP_ID and comparison.get("release_manifest_sha256") == release_sha,
            "map trip/release identity mismatch")
    require(comparison.get("source_sha256") == release_source_sha
            and comparison.get("reference_loaded_after_both_predictions") is True
            and comparison.get("evaluator_used") is False, "trip prediction/source separation failed")
    records = comparison.get("records")
    require(isinstance(records, dict) and set(records) == {"candidate", "baseline"}, "paired trip comparison missing")
    record = records["candidate"]
    require(isinstance(record, dict) and record.get("success") is True, "candidate trip prediction failed")
    require(type(comparison.get("observations")) is int and comparison["observations"] > 0
            and record.get("observations") == comparison["observations"], "trip observation counts differ")
    require(sha_value(record.get("route_sha256")) and sha_value(record.get("route_file_sha256")),
            "candidate needs separate route sequence and actual file-byte SHA256")
    require(isinstance(record.get("metrics"), dict) and record["metrics"].get("prediction_connected") is True
            and record["metrics"].get("route_continuity") == 1., "candidate trip is not connected")
    hashes = comparison.get("input_sha256")
    require(isinstance(hashes, dict) and set(hashes) == {".nodes", ".arcs", ".track", ".route"}
            and all(sha_value(value) for value in hashes.values()), "trip raw input hash bindings incomplete")
    return record


def check_route_identity(route: np.ndarray, file_sha: str, record: dict) -> None:
    require(file_sha == record.get("route_file_sha256"), "frozen route file-byte hash mismatch")
    require(integer_hash(route) == record.get("route_sha256"), "frozen route int64 sequence hash mismatch")


def traversal_records(graph, route: np.ndarray) -> tuple[list[dict], dict]:
    validation = validate_route(graph, route)
    require(len(route) > 0 and not validation["invalid_positions"], "empty or invalid route")
    counts, totals, records = Counter(), Counter(map(int, route)), []
    for position, value in enumerate(route):
        arc = int(value)
        counts[arc] += 1
        coordinates = graph.nodes[graph.arcs[arc]][:, ::-1].tolist()
        require(np.isfinite(coordinates).all() and coordinates[0] != coordinates[1], "nonfinite or zero-length traversal geometry")
        records.append({"position": position, "arc_id": arc, "tail_node_id": int(graph.tails[arc]),
                        "head_node_id": int(graph.heads[arc]), "occurrence": counts[arc],
                        "coordinates_latlon": coordinates})
    return records, {"arc_rows": len(records), "distinct_arcs": len(totals),
                     "repeated_traversal_arcs": sum(value > 1 for value in totals.values()),
                     "repeated_traversal_rows": sum(value for value in totals.values() if value > 1),
                     "duplicate_occurrences_beyond_first": len(records) - len(totals),
                     "broken_joins": len(validation["discontinuities"]),
                     "broken_join_positions": validation["discontinuities"],
                     "connectors_inserted": 0, "empty_geometry_count": 0}


def timestamp(value: str) -> datetime:
    require(isinstance(value, str), "missing UTC evidence timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0, "evidence timestamp is not UTC")
    return result


def load_evidence(lead_notice: str, expected_acceptance_sha: str) -> dict:
    # This first gate deliberately precedes every non-acceptance input read.
    require(isinstance(lead_notice, str) and len(lead_notice.strip()) >= 12, "explicit Lead accepted notice required before any real-data read")
    require(sha_value(expected_acceptance_sha), "explicit exact accepted-file SHA256 required")
    pins: dict[str, str] = {}
    acceptance = pinned_json(ACCEPTANCE, pins, expected_acceptance_sha)
    validate_acceptance(acceptance)
    release = pinned_json(RELEASE, pins, acceptance["release_manifest_sha256"])
    summary = pinned_json(SUMMARY, pins, acceptance["comparison_sha256"])
    binding = validate_summary(acceptance, summary, pins[relative(RELEASE)])
    require(release.get("schema") == "pathsnap-route96-frozen-release-v1", "unknown frozen release schema")
    require(release.get("deployment_authorized") is False
            and release.get("retraining_or_selection_after_student_allowed") is False, "offline/frozen release policy differs")
    require(release.get("profile") in ("context", "v1"), "unknown frozen feature profile")
    expected_files = RELEASE_FILES | ({"training_metrics.json"} if release["profile"] == "context" else set())
    require(set(release.get("files", {})) == expected_files, "release artifact file set differs")
    for name, digest in release["files"].items():
        pin(ART / name, pins, digest)
    require(release.get("model_sha256") == pins[relative(ART / "model.joblib")], "frozen model hash mismatch")
    require(set(release.get("source_sha256", {})) == RELEASE_SOURCES, "frozen release source set differs")
    for name, digest in release["source_sha256"].items():
        pin(repo_path(name), pins, digest)
    for path, key in ((TRAIN_PROTOCOL, "training_protocol_sha256"), (ROUTE_PROTOCOL, "route_protocol_sha256")):
        pin(path, pins, release.get(key, ""))
    selection = pinned_json(SELECTION, pins, release.get("frozen_selection_sha256", ""))
    require(selection.get("student_used_for_selection") is False and selection.get("evaluator_used") is False,
            "route selection used student/evaluator feedback")
    choice = release.get("configuration")
    require(isinstance(choice, dict) and choice == selection.get("selected", {}).get("configuration"), "frozen route choice differs")
    config = pinned_json(ART / "match_config.json", pins)
    policy = pinned_json(ART / "arc_policy.json", pins)
    require(config == choice.get("match_config") and policy == choice.get("arc_policy")
            and choice.get("profile") == release["profile"], "model/config/policy profile mismatch")
    metadata = pinned_json(ART / "training_metadata.json", pins)
    if "model_sha256" in metadata:
        require(metadata["model_sha256"] == release["model_sha256"], "training metadata model hash mismatch")
    # Pin all accepted comparison files, but parse only the chosen trip.
    for row in summary["trip_bindings"]:
        pin(repo_path(row["path"]), pins, row["sha256"])
    comparison = pinned_json(COMPARISON, pins, binding["sha256"])
    record = validate_trip_record(comparison, pins[relative(RELEASE)], release["source_sha256"]["scripts/target96_release.py"])
    chronology = [timestamp(item.get("created_at_utc")) for item in (selection, release, comparison, summary, acceptance)]
    require(chronology == sorted(chronology), "freeze/prediction/comparison/acceptance chronology differs")
    # No graph, GPS, reference, or predicted route was opened before all gates above.
    for extension, digest in comparison["input_sha256"].items():
        pin(PREFIX.with_suffix(extension), pins, digest)
    route_file_sha = pin(PREDICTION, pins, record["route_file_sha256"])
    graph, gps = load_graph(PREFIX), load_track(PREFIX.with_suffix(".track"))
    predicted, reference = load_route(PREDICTION), load_route(PREFIX.with_suffix(".route"))
    check_route_identity(predicted, route_file_sha, record)
    require(len(gps) == comparison["observations"], "raw GPS/comparison observation count mismatch")
    predicted_records, predicted_integrity = traversal_records(graph, predicted)
    reference_records, reference_integrity = traversal_records(graph, reference)
    require(predicted_integrity["broken_joins"] == 0, "frozen candidate has broken directed joins")
    metrics = score_routes(graph, predicted, reference)
    compare_metrics(metrics, record["metrics"], "frozen candidate route metrics")
    for name in ASSET_NAMES:
        pin(VENDOR / name, pins)
    for path in (Path(__file__), ROOT / "mapmatch/data.py", ROOT / "mapmatch/metrics.py"):
        pin(path, pins)
    return {"acceptance": acceptance, "release": release, "summary": summary, "comparison": comparison,
            "lead_accepted_notice": lead_notice.strip(), "expected_acceptance_sha256": expected_acceptance_sha,
            "graph": graph, "gps": gps, "predicted": predicted, "ref": reference,
            "prediction_records": predicted_records, "reference_records": reference_records,
            "prediction_integrity": predicted_integrity, "reference_integrity": reference_integrity,
            "metrics": metrics, "model_metadata": metadata, "input_sha256": pins, "synthetic": False}


def source_ids(data: dict) -> dict:
    return {"network_arc_ids": list(range(len(data["graph"].arcs))),
            "prediction_arc_ids": data["predicted"].tolist(), "reference_arc_ids": data["ref"].tolist(),
            "gps_observation_indices": list(range(len(data["gps"]))),
            "gps_timestamps_s": data["gps"][:, 2].tolist()}


def build_html(data: dict, path: Path = HTML_PATH) -> str:
    import folium
    from branca.element import Element, MacroElement
    from jinja2 import Template

    graph, gps = data["graph"], data["gps"]
    assets = Path(os.path.relpath(VENDOR, path.parent)).as_posix()
    m = folium.Map(location=gps[:, :2].mean(axis=0)[::-1].tolist(), tiles=None, prefer_canvas=True,
                   control_scale=True, zoom_control=True, attribution_control=False,
                   zoom_animation=False, fade_animation=False, marker_zoom_animation=False)
    m.default_js = [("leaflet", f"{assets}/leaflet.js")]
    m.default_css = [("leaflet_css", f"{assets}/leaflet.css")]
    m.get_root().header.add_child(Element(
        '<meta name="robots" content="noindex,nofollow">'
        '<meta http-equiv="Content-Security-Policy" content="default-src &apos;none&apos;; '
        'script-src &apos;unsafe-inline&apos; file:; style-src &apos;unsafe-inline&apos; file:; '
        'img-src file: data:; connect-src &apos;none&apos;; object-src &apos;none&apos;; '
        'base-uri &apos;none&apos;; form-action &apos;none&apos;">'))
    m.get_root().header.add_child(Element("""<style>
html,body{margin:0;width:100%;height:100%;overflow:hidden}
.leaflet-container{background:#f4f7fa;font-family:Arial,sans-serif}
.leaflet-control-layers{font-size:19px;line-height:1.5;width:360px;padding:14px!important}
.leaflet-control-layers label{white-space:normal}
.map-card{position:fixed;z-index:9999;background:#fffffff5;border:1px solid #bcc7d2;border-radius:6px;color:#142033;padding:16px 22px;box-shadow:0 2px 12px #14203320;font-family:Arial,sans-serif;box-sizing:border-box}
#target96-title{top:16px;left:64px;right:440px;font-size:21px;line-height:1.45}
#target96-title h1{font-size:32px;line-height:1.2;margin:0 0 8px;font-weight:700}
#target96-legend{bottom:28px;left:24px;width:1000px;font-size:21px;line-height:1.48}
#target96-legend .sample{display:inline-block;width:44px;margin-right:9px;vertical-align:middle;border-top:4px solid}
#target96-legend .small{font-size:18px;margin-top:8px;color:#475569}
#target96-audit{right:24px;bottom:28px;width:560px;font-size:21px;line-height:1.5}
.leaflet-control-scale-line{font-size:17px}
</style>"""))
    groups = {}
    groups["network"] = folium.FeatureGroup(name=f"Road graph (gray): {len(graph.arcs):,} directed arcs", show=True).add_to(m)
    folium.PolyLine(graph.nodes[graph.arcs][:, :, ::-1].tolist(), color=COLORS["network"], weight=1.2,
                    opacity=.55, interactive=False, smooth_factor=0).add_to(groups["network"])
    groups["reference"] = folium.FeatureGroup(name=f"Internal reference (green): {len(data['ref']):,} traversals", show=True).add_to(m)
    folium.PolyLine([row["coordinates_latlon"] for row in data["reference_records"]],
                    color=COLORS["reference"], weight=9, opacity=.78, interactive=False, smooth_factor=0).add_to(groups["reference"])
    groups["gps"] = folium.FeatureGroup(name=f"Raw GPS (blue): {len(gps):,} timed points", show=True).add_to(m)
    folium.PolyLine(gps[:, [1, 0]].tolist(), color=COLORS["gps"], weight=1.8, opacity=.72,
                    dash_array="2,5", interactive=False, smooth_factor=0).add_to(groups["gps"])
    for index, (lon, lat, time_s) in enumerate(gps):
        folium.CircleMarker([lat, lon], radius=2.5, weight=0, color=COLORS["gps"], fill=True,
                            fill_opacity=.95, tooltip=f"GPS row {index}; supplied timestamp {time_s:g} s").add_to(groups["gps"])
    groups["prediction"] = folium.FeatureGroup(name=f"Frozen prediction (orange): {len(data['predicted']):,} ordered traversals", show=True).add_to(m)
    for row in data["prediction_records"]:
        tooltip = (f"Traversal position {row['position']} (zero-based); original arc ID {row['arc_id']}; "
                   f"nodes {row['tail_node_id']} to {row['head_node_id']}; occurrence {row['occurrence']}. "
                   "No per-traversal timestamp inferred.")
        folium.PolyLine(row["coordinates_latlon"], color=COLORS["prediction"], weight=3.8,
                        opacity=1, dash_array="10,5", smooth_factor=0, tooltip=html.escape(tooltip)).add_to(groups["prediction"])
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    pi, ri = data["prediction_integrity"], data["reference_integrity"]
    model_name = html.escape(str(data["model_metadata"].get("model_name", data["model_metadata"].get("family", "frozen classifier"))))
    choice_name = html.escape(str(data["release"]["configuration"]["name"]))
    synthetic = data.get("synthetic") is True
    title = "SYNTHETIC helper fixture - not a result" if synthetic else f"Target96 accepted route - trip {TRIP_ID}"
    model_sha = data["release"]["model_sha256"]
    all_trip_score = data["acceptance"]["observed_edge_f1"]
    cards = f'''<section class="map-card" id="target96-title">
<h1>{title}</h1><div>Student development | {len(gps):,} raw GPS observations | frozen prediction, not a new inference run</div>
<div>{model_name} | {choice_name} | model SHA-256 {model_sha[:12]}</div>
<div style="font-size:18px;color:#92400e;font-weight:700">INTERNAL REPORT ONLY - supplied reference included - never serve in the application</div></section>
<section class="map-card" id="target96-legend"><div style="font-weight:700;margin-bottom:5px">Original observations and directed arc traversals</div>
<div><span class="sample" style="color:{COLORS['network']}"></span>Full supplied road graph: gray endpoint chords</div>
<div><span class="sample" style="color:{COLORS['gps']};border-top-style:dotted"></span>Raw GPS: blue points and chronological polyline</div>
<div><span class="sample" style="color:{COLORS['reference']};border-top-width:7px"></span>Supplied development reference: green (toggle above)</div>
<div><span class="sample" style="color:{COLORS['prediction']};border-top-style:dashed"></span>Frozen learned prediction: orange, {pi['arc_rows']} traversals</div>
<div class="small">Repeated traversals retained at original coordinates; no joins inserted by this renderer.<br>
Historically exposed development data, not blind or official evaluation.<br>Route F1 compares original directed arc multisets, not observation proxy labels.</div></section>
<section class="map-card" id="target96-audit"><div style="font-weight:700">Frozen prediction audit</div>
<div>{pi['distinct_arcs']} distinct arcs | {pi['duplicate_occurrences_beyond_first']} repeat occurrences</div>
<div>Broken joins: prediction {pi['broken_joins']} | reference {ri['broken_joins']}</div>
<div>Trip route F1 {data['metrics']['edge_f1']:.4f} | length F1 {data['metrics']['length_f1']:.4f}</div>
<div>Accepted all-ten mean route F1 {all_trip_score:.4f}</div>
<div style="font-size:18px;color:#475569;margin-top:7px">Local Folium / Leaflet | no web tiles or remote assets<br>Exact file-byte and route-sequence hashes verified</div></section>'''
    m.get_root().html.add_child(Element(cards))
    extent = np.concatenate((gps[:, :2], graph.nodes[graph.arcs[data["predicted"]]].reshape(-1, 2),
                             graph.nodes[graph.arcs[data["ref"]]].reshape(-1, 2)))
    m.fit_bounds([[float(extent[:, 1].min()), float(extent[:, 0].min())],
                  [float(extent[:, 1].max()), float(extent[:, 0].max())]],
                 padding_top_left=(100, 220), padding_bottom_right=(450, 350), max_zoom=17)
    ready = MacroElement()
    assignments = ",".join(f"{json.dumps(key)}:{group.get_name()}" for key, group in groups.items())
    ready._template = Template("{% macro script(this, kwargs) %}"
                               f"window.__TARGET96_MAP__={m.get_name()};window.__TARGET96_GROUPS__={{{assignments}}};"
                               f"window.__TARGET96_SOURCE_IDS__={json.dumps(source_ids(data), allow_nan=False)};"
                               "window.__TARGET96_READY__=true;{% endmacro %}")
    m.add_child(ready)
    text = m.get_root().render()
    # Folium 1.x injects an unused Glyphicons font-face even with local default assets.
    # This map uses no icon markers; remove that optional remote font declaration.
    text = re.sub(r"@font-face\s*\{[^}]*https?://[^}]*\}", "", text, flags=re.I)
    require(not re.search(r'(?:src|href)\s*=\s*[\'"](?:https?:|//)', text, re.I)
            and not re.search(r'url\(\s*[\'"]?(?:https?:|//)', text, re.I), "remote HTML dependency forbidden")
    require("L.tileLayer(" not in text, "map tiles forbidden")
    return text


def verify_browser_geometry(data: dict, layers: dict, ids: dict) -> dict:
    require(set(layers) == {"network", "reference", "gps", "prediction"}, "browser map layer set differs")
    predicted = [row["coordinates"] for row in layers["prediction"]]
    require(predicted == [row["coordinates_latlon"] for row in data["prediction_records"]],
            "actual Leaflet prediction traversal order/geometry differs")
    require(len(layers["reference"]) == 1 and layers["reference"][0]["coordinates"]
            == [row["coordinates_latlon"] for row in data["reference_records"]], "actual Leaflet reference geometry differs")
    require(len(layers["network"]) == 1 and layers["network"][0]["coordinates"]
            == data["graph"].nodes[data["graph"].arcs][:, :, ::-1].tolist(), "actual Leaflet complete network geometry differs")
    gps = data["gps"][:, [1, 0]].tolist()
    require(layers["gps"] and layers["gps"][0]["coordinates"] == gps
            and [row["coordinates"] for row in layers["gps"][1:]] == gps, "actual Leaflet GPS point/order geometry differs")
    require(ids == source_ids(data), "actual browser original source IDs/timestamps differ")
    return {"actual_layer_counts": {key: len(value) for key, value in layers.items()},
            "prediction_geometry_latlon_sha256": object_hash(predicted),
            "all_original_source_ids_and_timestamps_match": True,
            "every_predicted_traversal_order_and_geometry_matches": True,
            "full_network_reference_and_raw_gps_geometry_match": True}


def capture_html(path: Path, data: dict) -> tuple[bytes, dict]:
    from playwright.sync_api import sync_playwright

    page_errors, console_errors, requests, blocked, failures = [], [], [], [], []
    allowed = {path.resolve().as_uri(), *((VENDOR / name).resolve().as_uri() for name in ASSET_NAMES)}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport=VIEWPORT, device_scale_factor=1, service_workers="block")
        page = context.new_page()
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.on("request", lambda request: requests.append(request.url))
        page.on("requestfailed", lambda request: failures.append({"url": request.url, "failure": request.failure}))
        def only_local(route):
            if route.request.url in allowed:
                route.continue_()
            else:
                blocked.append(route.request.url)
                route.abort()
        context.route(re.compile(r".*"), only_local)
        page.goto(path.resolve().as_uri(), wait_until="load")
        require(page.evaluate("() => window.__TARGET96_READY__ === true && typeof L !== 'undefined'") is True, "local Leaflet map not ready after load")
        require(page.locator(".leaflet-control-layers-overlays").count() == 1, "map layer controls not ready")
        def settle():
            # Browser requestAnimationFrame, not wait_for_function string eval: retain CSP without unsafe-eval.
            page.evaluate("""() => new Promise((resolve,reject) => {
 const started=performance.now();const tick=()=>{const m=window.__TARGET96_MAP__;
 if(!m._animatingZoom&&!m._panAnim?._inProgress&&!m._renderer?._redrawRequest)
   requestAnimationFrame(()=>requestAnimationFrame(resolve));
 else if(performance.now()-started>10000)reject(new Error('Leaflet did not settle'));
 else requestAnimationFrame(tick);};tick();})""")
        settle()
        proof = page.evaluate("""() => {
 const m=window.__TARGET96_MAP__, groups=window.__TARGET96_GROUPS__;
 const flat=points=>Array.isArray(points)&&points.length&&points[0] instanceof L.LatLng
   ? points.map(p=>[p.lat,p.lng]):points.map(flat);
 const layers=Object.fromEntries(Object.entries(groups).map(([key,g])=>[key,g.getLayers().map(l=>({
   coordinates:l.getLatLngs?flat(l.getLatLngs()):[l.getLatLng().lat,l.getLatLng().lng]}))]));
 const box=s=>{const e=document.querySelector(s),r=e.getBoundingClientRect();return {left:r.left,top:r.top,right:r.right,bottom:r.bottom,scrollWidth:e.scrollWidth,clientWidth:e.clientWidth,scrollHeight:e.scrollHeight,clientHeight:e.clientHeight}};
 const boxes=Object.fromEntries(['#target96-title','#target96-legend','#target96-audit','.leaflet-control-layers'].map(s=>[s,box(s)]));
 const inside=Object.values(boxes).every(r=>r.left>=0&&r.top>=0&&r.right<=innerWidth&&r.bottom<=innerHeight&&r.scrollWidth<=r.clientWidth&&r.scrollHeight<=r.clientHeight);
 const overlap=(a,b)=>!(a.right<=b.left||b.right<=a.left||a.bottom<=b.top||b.bottom<=a.top);
 const values=Object.values(boxes);let noOverlap=true;for(let i=0;i<values.length;i++)for(let j=i+1;j<values.length;j++)if(overlap(values[i],values[j]))noOverlap=false;
 const canvases=Array.from(document.querySelectorAll('canvas'));
 const colors={network:[146,155,167],gps:[33,102,232],reference:[22,136,74],prediction:[228,111,11]},counts={};
 for(const [name,color] of Object.entries(colors)){let n=0;for(const c of canvases){const pixels=c.getContext('2d').getImageData(0,0,c.width,c.height).data;for(let i=0;i<pixels.length;i+=4)if(pixels[i+3]>40&&Math.abs(pixels[i]-color[0])<32&&Math.abs(pixels[i+1]-color[1])<32&&Math.abs(pixels[i+2]-color[2])<32)n++;}counts[name]=n;}
 return {leaflet_version:L.version,layers,source_ids:window.__TARGET96_SOURCE_IDS__,
   all_groups_visible:Object.values(groups).every(g=>m.hasLayer(g)),
   checked_overlays:document.querySelectorAll('.leaflet-control-layers-selector:checked').length,
   tile_layer_count:Object.values(m._layers).filter(l=>l instanceof L.TileLayer).length,
   title:document.querySelector('#target96-title').innerText,legend:document.querySelector('#target96-legend').innerText,
   all_cards_inside_viewport:inside,no_card_overlap:noOverlap,boxes,
   body_scrolls:document.body.scrollWidth>innerWidth||document.body.scrollHeight>innerHeight,
   visible_layer_color_pixels:counts,document_url:location.href};
}""")
        proof.update(verify_browser_geometry(data, proof.pop("layers"), proof.pop("source_ids")))
        require(proof["all_groups_visible"] and proof["checked_overlays"] == 4 and proof["tile_layer_count"] == 0,
                "mandatory layer visibility or no-tiles check failed")
        require(proof["all_cards_inside_viewport"] and proof["no_card_overlap"] and not proof["body_scrolls"],
                "map title/legend/audit/controls clip, scroll, or overlap")
        require(all(count > 5 for count in proof["visible_layer_color_pixels"].values()),
                f"mandatory layer did not render visibly: {proof['visible_layer_color_pixels']}")
        require(TRIP_ID in proof["title"] and "Target96 accepted route" in proof["title"], "wrong map title")
        controls = page.locator(".leaflet-control-layers-selector")
        toggles = {}
        for index, name in enumerate(("network", "reference", "gps", "prediction")):
            controls.nth(index).uncheck()
            require(page.evaluate("name => window.__TARGET96_MAP__.hasLayer(window.__TARGET96_GROUPS__[name])", name) is False,
                    f"{name} toggle did not remove actual layer")
            controls.nth(index).check()
            require(page.evaluate("name => window.__TARGET96_MAP__.hasLayer(window.__TARGET96_GROUPS__[name])", name) is True,
                    f"{name} toggle did not restore actual layer")
            toggles[name] = True
        settle()
        # Capture only the actual file:// Folium page after all checks pass.
        png = page.screenshot(type="png", full_page=False, animations="disabled")
        proof.update({"browser_version": browser.version, "all_four_layer_toggles_verified": toggles,
                      "viewport": VIEWPORT, "device_scale_factor": 1, "capture_method": "Playwright Chromium page.screenshot",
                      "human_visual_inspection_claimed": False})
        context.close()
        browser.close()
    require(not page_errors and not console_errors and not blocked and not failures
            and all(url in allowed for url in requests),
            f"browser errors or nonlocal requests: {page_errors}, {console_errors}, {blocked}, {failures}")
    require(png[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", png[16:24]) == (VIEWPORT["width"], VIEWPORT["height"]),
            "actual screenshot PNG dimensions invalid")
    proof.update({"requests": requests, "blocked_requests": blocked, "request_failures": failures,
                  "page_errors": page_errors, "console_errors": console_errors,
                  "external_http_requests": 0, "file_only_requests_verified": True})
    return png, proof


def recheck_inputs(pins: dict) -> None:
    for name, digest in pins.items():
        require(sha256_file(repo_path(name)) == digest, f"immutable input drift during rendering: {name}")


def generate(data: dict, capture_existing_html_sha256: str | None = None) -> dict:
    for path in (PNG_PATH, EVIDENCE_PATH):
        safe_output(path)
    if capture_existing_html_sha256 is None:
        safe_output(HTML_PATH)
        write_new(HTML_PATH, build_html(data))
    else:
        require(sha_value(capture_existing_html_sha256) and sha256_file(HTML_PATH) == capture_existing_html_sha256,
                "capture-existing HTML must be explicitly byte-hash pinned; no overwrites allowed")
    png, browser = capture_html(HTML_PATH, data)
    recheck_inputs(data["input_sha256"])
    write_new(PNG_PATH, png)
    graph, gps, pins = data["graph"], data["gps"], data["input_sha256"]
    result = {
        "schema": "pathsnap-target96-accepted-report-map-v1", "passed": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "trip_id": TRIP_ID,
        "role": "Internal report only; historically exposed Student development; not official or blind",
        "reference_included": True, "frontend_asset": False, "inference_training_tuning_performed": False,
        "lead_accepted_notice": data["lead_accepted_notice"],
        "acceptance_path": relative(ACCEPTANCE), "acceptance_sha256": data["expected_acceptance_sha256"],
        "release_manifest_path": relative(RELEASE), "release_manifest_sha256": pins[relative(RELEASE)],
        "comparison_path": relative(COMPARISON), "comparison_sha256": pins[relative(COMPARISON)],
        "accepted_summary_path": relative(SUMMARY), "accepted_summary_sha256": pins[relative(SUMMARY)],
        "model_sha256": data["release"]["model_sha256"],
        "config_file_sha256": pins[relative(ART / "match_config.json")],
        "policy_file_sha256": pins[relative(ART / "arc_policy.json")],
        "frozen_configuration": data["release"]["configuration"],
        "model_label_from_frozen_metadata": data["model_metadata"].get("model_name", data["model_metadata"].get("family")),
        "prediction_path": relative(PREDICTION), "route_file_sha256": pins[relative(PREDICTION)],
        "route_int64_sequence_sha256": integer_hash(data["predicted"]),
        "actual_route_file_bytes_match_accepted_comparison": True, "actual_route_sequence_matches_accepted_comparison": True,
        "observations": len(gps), "network_nodes": len(graph.nodes), "network_directed_arcs": len(graph.arcs),
        "prediction_integrity": data["prediction_integrity"], "reference_integrity": data["reference_integrity"],
        "prediction_traversals": data["prediction_records"], "reference_traversals": data["reference_records"],
        "geometry_note": "Original graph endpoint chords in latitude/longitude; one separate segment per directed arc. Repeated traversals overlap spatially but remain distinct ordered layers. No connectors inserted or route arcs removed by this renderer.",
        "source_ids": source_ids(data),
        "network_geometry_latlon_sha256": object_hash(graph.nodes[graph.arcs][:, :, ::-1].tolist()),
        "raw_gps_lon_lat_time_sha256": object_hash(gps.tolist()),
        "route_metrics": data["metrics"], "route_metrics_match_frozen_comparison": True,
        "accepted_all_ten_mean_route_edge_f1": data["acceptance"]["observed_edge_f1"],
        "layer_colors": COLORS, "local_leaflet_assets": [relative(VENDOR / name) for name in ASSET_NAMES],
        "html_path": relative(HTML_PATH), "html_sha256": sha256_file(HTML_PATH),
        "screenshot_path": relative(PNG_PATH), "screenshot_sha256": sha256_file(PNG_PATH),
        "screenshot_pixels": VIEWPORT, "browser": browser,
        "library_versions": {name: importlib.metadata.version(name) for name in ("folium", "branca", "numpy", "playwright")},
        "input_sha256": pins, "all_input_hashes_unchanged_after_rendering": True,
    }
    recheck_inputs(pins)
    write_new(EVIDENCE_PATH, json.dumps(result, indent=2, allow_nan=False) + "\n")
    return {"passed": True, "html": relative(HTML_PATH), "screenshot": relative(PNG_PATH),
            "evidence": relative(EVIDENCE_PATH), "screenshot_sha256": result["screenshot_sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true", help="synthetic fixtures only; no real-data gate reads")
    mode.add_argument("--check", action="store_true", help="read-only readiness AFTER explicit Lead accepted notice")
    mode.add_argument("--generate", action="store_true", help="create-only actual Folium HTML, screenshot, evidence AFTER notice")
    parser.add_argument("--lead-accepted-notice", help="quote the actual Lead accepted notice; never infer it from files")
    parser.add_argument("--acceptance-sha256", help="exact acceptance.json SHA256 from the accepted handoff")
    parser.add_argument("--capture-existing-html-sha256", help="resume a browser-only failure from this exact existing HTML hash, without overwrite")
    args = parser.parse_args(argv)
    if args.self_test:
        suite = unittest.defaultTestLoader.discover(str(MAP_DIR), pattern="test_generate_target96_map.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        print(json.dumps({"synthetic_only": True, "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors)}))
        return 0 if result.wasSuccessful() and result.testsRun > 0 else 1
    data = load_evidence(args.lead_accepted_notice, args.acceptance_sha256)
    if args.check:
        recheck_inputs(data["input_sha256"])
        print(json.dumps({"ready": True, "read_only": True, "trip_id": TRIP_ID,
                          "acceptance_sha256": data["expected_acceptance_sha256"],
                          "route_file_sha256": data["input_sha256"][relative(PREDICTION)],
                          "outputs_created": False}, indent=2))
    else:
        print(json.dumps(generate(data, args.capture_existing_html_sha256), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
