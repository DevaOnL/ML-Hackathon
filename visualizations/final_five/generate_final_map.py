#!/usr/bin/env python3
"""Render one actual final-model Student development map, with local Leaflet only.

This internal report artifact includes the supplied development reference. It is
never a frontend/backend asset. The generator reads frozen predictions; it does
not load a classifier, run inference, contact a service, or read evaluator data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

import numpy as np

FIVE_DIR = Path(__file__).resolve().parent
ROOT = FIVE_DIR.parents[1]
sys.path.insert(0, str(ROOT))
from backend.geometry import predicted_geojson
from mapmatch.data import load_graph, load_route, load_track, validate_route
from mapmatch.metrics import score_routes
from visualizations.final_five.figure_integrity import safe_output as guarded_output, sha256_file

TRIP_ID = "00000009"
MODEL_SHA256 = "c9cc8e9f36838f656a94d07ed69632979df9981ca94a9daf3b52fbce052e4b34"
CONFIG_SHA256 = "10405aa6b6f370e41d4ddcb01207fd1e8f58900fd0395e541ee6bc78146168c5"
FAMILY = "hist_gradient_boosting"
RUN = "hist_gradient_boosting_tuned"
MAPS = FIVE_DIR / "maps"
OUTPUT = MAPS / f"final_development_trip_{TRIP_ID}.html"
MANIFEST = FIVE_DIR / "map_manifest.json"
COLORS = {"roads": "#929ba7", "gps": "#c87908", "reference": "#16884a", "prediction": "#2166e8"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def output(path: Path) -> Path:
    return guarded_output(path, FIVE_DIR)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    output(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def object_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def integer_hash(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def verify_identity(metadata: dict, freeze: dict, decoder: dict, run: dict,
                    model_sha: str, config_sha: str, metadata_sha: str, config: dict) -> str:
    require(model_sha == MODEL_SHA256 == freeze.get("model_sha256") == decoder.get("model_sha256")
            == run.get("model_sha256"), "wrong final model hash")
    require(config_sha == CONFIG_SHA256 == decoder.get("config_file_sha256"), "wrong frozen decoder hash")
    require(metadata_sha == freeze.get("metadata_sha256"), "model metadata hash drift")
    require(metadata.get("model_name") == freeze.get("model_name") == FAMILY
            and metadata.get("run_name") == freeze.get("run_name") == RUN,
            "wrong model family or tuned run")
    require(metadata.get("training_trip_count") == 80, "final model must be fitted on all 80 TRAIN trips")
    require(run.get("config") == decoder.get("config") == config, "CLI/config freeze mismatch")
    require(run.get("baseline") is False and run.get("uses_reference_routes") is False,
            "map prediction must be learned and reference-free")
    require(TRIP_ID in run.get("trip_ids", []), "chosen trip is missing from frozen CLI run")
    return "HistGradientBoosting (tuned)"


def allowed_inputs(prefix: Path, prediction: Path) -> None:
    expected = ROOT / "Student folder" / "student_test" / TRIP_ID / TRIP_ID
    require(prefix.absolute() == expected.absolute() and prefix.resolve() == expected.absolute(),
            "map accepts only the original Student development trip 00000009")
    target = ROOT / "outputs" / "final_student_test" / "routes" / f"{TRIP_ID}.route"
    require(prediction.absolute() == target.absolute() and prediction.resolve() == target.absolute(),
            "map prediction must be the frozen final Student CLI route")
    for path in [*(prefix.with_suffix(ext) for ext in (".nodes", ".arcs", ".track", ".route")), prediction]:
        require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
                "input must be a regular, unlinked development file")


def traversal_records(graph, route: np.ndarray) -> tuple[list[dict], dict]:
    validation = validate_route(graph, route)
    require(len(route) > 0 and not validation["invalid_positions"], "empty or invalid route")
    counts, total = Counter(), Counter(map(int, route))
    records = []
    for position, arc in enumerate(route):
        arc = int(arc)
        counts[arc] += 1
        coordinates = graph.nodes[graph.arcs[arc]][:, ::-1].tolist()
        require(np.isfinite(coordinates).all() and len(coordinates) == 2
                and coordinates[0] != coordinates[1], "empty or zero-length route geometry")
        records.append({"position": position, "arc_id": arc, "occurrence": counts[arc],
                        "coordinates_latlon": coordinates})
    return records, {"arc_rows": len(records), "distinct_arcs": len(total),
                     "repeated_traversal_arcs": sum(v > 1 for v in total.values()),
                     "repeated_traversal_rows": sum(v for v in total.values() if v > 1),
                     "duplicate_occurrences_beyond_first": len(records) - len(total),
                     "broken_joins": len(validation["discontinuities"]),
                     "broken_join_positions": validation["discontinuities"],
                     "empty_geometry_count": 0, "connectors_inserted": 0}


def load_evidence() -> dict:
    prefix = ROOT / "Student folder" / "student_test" / TRIP_ID / TRIP_ID
    prediction = ROOT / "outputs" / "final_student_test" / "routes" / f"{TRIP_ID}.route"
    allowed_inputs(prefix, prediction)
    artifacts = ROOT / "artifacts" / "final_five"
    outputs = ROOT / "outputs" / "final_student_test"
    delivery_path = ROOT / "reports" / "final_five" / "offline_delivery_verification.json"
    paths = [*(prefix.with_suffix(ext) for ext in (".nodes", ".arcs", ".track", ".route")), prediction,
             *(artifacts / name for name in ("model.joblib", "model_metadata.json", "model_freeze.json", "match_config.json")),
             ROOT / "reports" / "final_five" / "decoder" / "freeze.json", delivery_path,
             outputs / "run_config.json", outputs / "index.csv", outputs / "diagnostics" / f"{TRIP_ID}.json",
             outputs / "observations" / f"{TRIP_ID}.csv", Path(__file__), FIVE_DIR / "figure_integrity.py",
             ROOT / "mapmatch" / "data.py", ROOT / "mapmatch" / "metrics.py", ROOT / "backend" / "geometry.py"]
    vendor = ROOT / "frontend" / "vendor" / "leaflet"
    paths += [vendor / name for name in ("leaflet.js", "leaflet.css", "images/layers.png", "images/layers-2x.png")]
    before = {rel(p): sha256_file(p) for p in paths}
    model = read_json(artifacts / "model_metadata.json")
    freeze = read_json(artifacts / "model_freeze.json")
    decoder = read_json(ROOT / "reports" / "final_five" / "decoder" / "freeze.json")
    run = read_json(outputs / "run_config.json")
    config = read_json(artifacts / "match_config.json")
    label = verify_identity(model, freeze, decoder, run, before[rel(artifacts / "model.joblib")],
                            before[rel(artifacts / "match_config.json")], before[rel(artifacts / "model_metadata.json")], config)
    delivery = read_json(delivery_path)
    require(delivery.get("passed") is True and delivery.get("model_sha256") == MODEL_SHA256
            and delivery.get("config_file_sha256") == CONFIG_SHA256, "offline delivery gate failed")
    expected = delivery["trips"][TRIP_ID]
    require(all(expected["checks"].values()), "trip9 delivery checks failed")
    graph, gps = load_graph(prefix), load_track(prefix.with_suffix(".track"))
    pred, reference = load_route(prediction), load_route(prefix.with_suffix(".route"))
    predicted, pred_integrity = traversal_records(graph, pred)
    refs, ref_integrity = traversal_records(graph, reference)
    require(pred_integrity["broken_joins"] == 0, "final prediction has broken directed joins")
    with (outputs / "index.csv").open() as stream:
        index = [row for row in csv.DictReader(stream) if row["trip_id"] == TRIP_ID]
    require(len(index) == 1 and index[0]["status"] == "ok", "CLI route index is not uniquely successful")
    require(index[0]["sha256"] == expected["route_sha256"] == before[rel(prediction)]
            and int(index[0]["arc_count"]) == len(pred) == expected["predicted_arc_count"], "CLI route identity/count mismatch")
    with (outputs / "observations" / f"{TRIP_ID}.csv").open() as stream:
        observations = list(csv.DictReader(stream))
    require(len(observations) == len(gps) == expected["observations"], "GPS/assignment count mismatch")
    require([int(row["observation_index"]) for row in observations] == list(range(len(gps)))
            and np.array_equal([float(row["timestamp_s"]) for row in observations], gps[:, 2]),
            "GPS order or timestamps changed")
    states = [int(row["predicted_arc_id"]) for row in observations]
    require(integer_hash(states) == expected["matched_edge_array_sha256"], "CLI matched-state hash mismatch")
    require(object_hash(predicted_geojson(graph, pred)) == expected["predicted_geometry_sha256"],
            "predicted geometry differs from verified CLI geometry")
    metrics = score_routes(graph, pred, reference)
    for key, value in expected["metrics"].items():
        actual = metrics.get(key)
        require(math.isclose(actual, value, abs_tol=1e-10, rel_tol=1e-12) if isinstance(value, (int, float))
                and not isinstance(value, bool) else actual == value, f"recorded route metric mismatch: {key}")
    diagnostics = read_json(outputs / "diagnostics" / f"{TRIP_ID}.json")
    require(diagnostics.get("route_arc_count") == len(pred) and diagnostics.get("uses_reference_route") is False,
            "CLI diagnostics mismatch")
    times = defaultdict(list)
    for row in observations:
        times[int(row["predicted_arc_id"])].append(float(row["timestamp_s"]))
    for record in predicted:
        values = times.get(record["arc_id"], [])
        record["matched_observation_time_range_s"] = [min(values), max(values)] if values else None
    return {"prefix": prefix, "prediction": prediction, "graph": graph, "gps": gps,
            "pred": pred, "ref": reference, "predicted": predicted, "reference": refs,
            "prediction_integrity": pred_integrity, "reference_integrity": ref_integrity,
            "label": label, "model_metadata": model, "metrics": metrics, "input_sha256": before,
            "input_paths": paths, "vendor": vendor}


def render_html(data: dict, path: Path) -> dict:
    import folium
    from branca.element import Element, MacroElement
    from jinja2 import Template

    graph, gps = data["graph"], data["gps"]
    path = output(path)
    relative_vendor = Path(os.path.relpath(data["vendor"], path.parent)).as_posix()
    center = gps[:, :2].mean(axis=0)
    m = folium.Map(location=center[::-1].tolist(), tiles=None, prefer_canvas=True,
                   control_scale=True, zoom_control=True, attribution_control=False,
                   zoom_animation=False, fade_animation=False, marker_zoom_animation=False)
    m.default_js = [("leaflet", f"{relative_vendor}/leaflet.js")]
    m.default_css = [("leaflet_css", f"{relative_vendor}/leaflet.css")]
    m.get_root().header.add_child(Element('<meta name="robots" content="noindex,nofollow">'))
    m.get_root().header.add_child(Element('''<style>
html,body{margin:0;width:100%;height:100%;overflow:hidden}
.leaflet-container{background:#f4f7fa;font-family:Arial,sans-serif}
.leaflet-control-layers{font-size:19px;line-height:1.55;width:360px;padding:14px!important}
.leaflet-control-layers label{white-space:normal}
.report-card{position:fixed;z-index:9999;background:#fffffff5;border:1px solid #bcc7d2;border-radius:6px;color:#142033;padding:16px 22px;box-shadow:0 2px 12px #14203320;font-family:Arial,sans-serif}
#final-map-title{top:16px;left:60px;right:440px;font-size:22px;line-height:1.45}
#final-map-title h1{font-size:32px;line-height:1.2;margin:0 0 8px;font-weight:700}
#final-map-legend{bottom:28px;left:24px;width:910px;font-size:22px;line-height:1.48}
#final-map-legend .sample{display:inline-block;width:44px;margin-right:9px;vertical-align:middle;border-top:4px solid}
#final-map-legend .small{font-size:19px;margin-top:8px;color:#475569}
#final-map-badge{right:24px;bottom:28px;width:500px;font-size:21px;line-height:1.5}
.leaflet-control-scale-line{font-size:17px}
</style>'''))
    groups = {}
    groups["network"] = folium.FeatureGroup(name=f"Full road graph (gray): {len(graph.arcs):,} directed arcs", show=True).add_to(m)
    folium.PolyLine(graph.nodes[graph.arcs][:, :, ::-1].tolist(), color=COLORS["roads"], weight=1.2,
                    opacity=.55, interactive=False, smooth_factor=0).add_to(groups["network"])
    groups["reference"] = folium.FeatureGroup(name=f"Internal reference (green): {len(data['ref']):,} traversals", show=True).add_to(m)
    folium.PolyLine([r["coordinates_latlon"] for r in data["reference"]], color=COLORS["reference"],
                    weight=9, opacity=.78, interactive=False, smooth_factor=0).add_to(groups["reference"])
    groups["gps"] = folium.FeatureGroup(name=f"Raw GPS (amber): {len(gps):,} timed points", show=True).add_to(m)
    folium.PolyLine(gps[:, [1, 0]].tolist(), color=COLORS["gps"], weight=1.8, opacity=.70,
                    dash_array="2,5", interactive=False, smooth_factor=0).add_to(groups["gps"])
    for i, (lon, lat, timestamp) in enumerate(gps):
        folium.CircleMarker([lat, lon], radius=2.4, weight=0, color=COLORS["gps"], fill=True,
                            fill_opacity=.90, tooltip=f"GPS row {i}; supplied time {timestamp:g} s").add_to(groups["gps"])
    groups["prediction"] = folium.FeatureGroup(name=f"Learned Viterbi (blue): {len(data['pred']):,} ordered traversals", show=True).add_to(m)
    for record in data["predicted"]:
        time_range = record["matched_observation_time_range_s"]
        time_label = (f"Matched GPS times on this arc ID: {time_range[0]:g} to {time_range[1]:g} s "
                      "(all occurrences combined; no per-traversal time inferred)" if time_range else
                      "Connector traversal; no direct matched GPS timestamp")
        tooltip = (f"Traversal position {record['position']} (zero-based); arc ID {record['arc_id']}; "
                   f"occurrence {record['occurrence']}. {time_label}")
        folium.PolyLine(record["coordinates_latlon"], color=COLORS["prediction"], weight=3.8,
                        opacity=1, dash_array="10,5", smooth_factor=0,
                        tooltip=html.escape(tooltip)).add_to(groups["prediction"])
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    pi, ri = data["prediction_integrity"], data["reference_integrity"]
    model = data["model_metadata"]
    label = html.escape(data["label"])
    title = f'''<section class="report-card" id="final-map-title">
<h1>Final HistGradientBoosting · trip {TRIP_ID}</h1>
<div>Student development · learned Viterbi route · {len(gps):,} raw GPS observations</div>
<div>{label} · TRAIN {model['training_trip_count']} trips · model SHA-256 {MODEL_SHA256[:12]}</div>
<div style="font-size:19px;color:#92400e;font-weight:700">INTERNAL REPORT ONLY · supplied reference included · never serve in the application</div></section>
<section class="report-card" id="final-map-legend">
<div style="font-weight:700;margin-bottom:5px">Actual geographic observations and arc traversals</div>
<div><span class="sample" style="color:{COLORS['roads']}"></span>Full supplied road graph: gray endpoint chords</div>
<div><span class="sample" style="color:{COLORS['gps']};border-top-style:dotted"></span>Raw GPS: amber points and chronological polyline</div>
<div><span class="sample" style="color:{COLORS['reference']};border-top-width:7px"></span>Supplied development reference: green (toggle above)</div>
<div><span class="sample" style="color:{COLORS['prediction']};border-top-style:dashed"></span>Frozen learned Viterbi prediction: blue, {pi['arc_rows']} traversals</div>
<div class="small">All repeated traversals retained; original graph coordinates; no inserted joins.<br>
Observation accuracy uses geometry-and-motion proxy labels, not timestamp-aligned truth.<br>
Route F1 compares supplied arc traversals directly; it is not an official organizer score.</div></section>
<section class="report-card" id="final-map-badge">
<div style="font-weight:700">Frozen prediction audit</div>
<div>{pi['distinct_arcs']} distinct arcs · {pi['duplicate_occurrences_beyond_first']} repeat occurrences</div>
<div>Broken joins: prediction {pi['broken_joins']} · reference {ri['broken_joins']}</div>
<div>Trip route arc F1 {data['metrics']['edge_f1']:.4f} · length F1 {data['metrics']['length_f1']:.4f}</div>
<div style="font-size:18px;color:#475569;margin-top:7px">Local Folium / Leaflet · no web tiles or remote assets<br>
Decoder SHA-256 {CONFIG_SHA256[:12]}</div></section>'''
    m.get_root().html.add_child(Element(title))
    extent = np.concatenate((gps[:, :2], graph.nodes[graph.arcs[data["pred"]]].reshape(-1, 2),
                             graph.nodes[graph.arcs[data["ref"]]].reshape(-1, 2)))
    m.fit_bounds([[float(extent[:, 1].min()), float(extent[:, 0].min())],
                  [float(extent[:, 1].max()), float(extent[:, 0].max())]],
                 padding_top_left=(80, 205), padding_bottom_right=(450, 350), max_zoom=17)
    ready = MacroElement()
    group_names = {key: value.get_name() for key, value in groups.items()}
    assignments = ",".join(f"{json.dumps(key)}:{value}" for key, value in group_names.items())
    ready._template = Template("{% macro script(this, kwargs) %}"
                               f"window.__FINAL_MAP__={m.get_name()};window.__FINAL_GROUPS__={{{assignments}}};"
                               "window.__FINAL_MAP_READY__=true;{% endmacro %}")
    m.add_child(ready)
    m.save(str(path))
    text = path.read_text(encoding="utf-8")
    require(not re.search(r'''(?:src|href)\s*=\s*["'](?:https?:)?//''', text, flags=re.I),
            "remote asset found in offline HTML")
    return {"folium_version": folium.__version__, "leaflet_assets": rel(data["vendor"]),
            "layers": {key: {"variable": value.get_name(), "child_count": len(value._children)}
                       for key, value in groups.items()}, "full_network_arc_geometries": len(graph.arcs)}


def capture(path: Path, data: dict) -> dict:
    from playwright.sync_api import sync_playwright

    png, pdf = output(path.with_suffix(".png")), output(path.with_suffix(".pdf"))
    page_errors, console_errors, requests, blocked = [], [], [], []
    failed_requests = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.on("request", lambda request: requests.append(request.url))
        page.on("requestfailed", lambda request: failed_requests.append({"url": request.url, "failure": request.failure}))
        def block_network(route):
            blocked.append(route.request.url)
            route.abort()
        page.route(re.compile(r"^(?:https?|wss?|ftp):", re.I), block_network)
        page.goto(path.resolve().as_uri(), wait_until="load")
        page.wait_for_function("window.__FINAL_MAP_READY__ === true && typeof L !== 'undefined'")
        page.wait_for_function("document.querySelector('.leaflet-control-layers-overlays') !== null")
        page.wait_for_function("!window.__FINAL_MAP__._animatingZoom && !window.__FINAL_MAP__._panAnim?._inProgress && !window.__FINAL_MAP__._renderer?._redrawRequest")
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        proof = page.evaluate("""() => {
 const m=window.__FINAL_MAP__, groups=window.__FINAL_GROUPS__;
 const flatten=points=>Array.isArray(points)&&points.length&&points[0] instanceof L.LatLng
      ? points.map(p=>[p.lat,p.lng]):points.map(flatten);
 const layers=Object.fromEntries(Object.entries(groups).map(([name,g])=>[name,g.getLayers().map(layer=>({
      type:layer instanceof L.CircleMarker?'point':'polyline',
      coordinates:layer.getLatLngs?flatten(layer.getLatLngs()):[layer.getLatLng().lat,layer.getLatLng().lng]
 }))]));
 const rect=s=>{const e=document.querySelector(s),r=e.getBoundingClientRect();return {left:r.left,top:r.top,right:r.right,bottom:r.bottom,width:r.width,height:r.height,scrollWidth:e.scrollWidth,clientWidth:e.clientWidth,scrollHeight:e.scrollHeight,clientHeight:e.clientHeight}};
 const boxes=Object.fromEntries(['#final-map-title','#final-map-legend','#final-map-badge','.leaflet-control-layers'].map(s=>[s,rect(s)]));
 const within=Object.values(boxes).every(r=>r.left>=0&&r.top>=0&&r.right<=innerWidth&&r.bottom<=innerHeight&&r.scrollWidth<=r.clientWidth&&r.scrollHeight<=r.clientHeight);
 const overlap=(a,b)=>!(a.right<=b.left||b.right<=a.left||a.bottom<=b.top||b.bottom<=a.top);
 const pairs=Object.values(boxes);let noOverlap=true;for(let i=0;i<pairs.length;i++)for(let j=i+1;j<pairs.length;j++)if(overlap(pairs[i],pairs[j]))noOverlap=false;
 const canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d'),pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;
 const colors={gps_amber:[200,121,8],road_gray:[146,155,167],reference_green:[22,136,74],prediction_blue:[33,102,232]},counts={};
 for(const [name,color] of Object.entries(colors)){let n=0;for(let i=0;i<pixels.length;i+=4)if(pixels[i+3]>40&&Math.abs(pixels[i]-color[0])<32&&Math.abs(pixels[i+1]-color[1])<32&&Math.abs(pixels[i+2]-color[2])<32)n++;counts[name]=n;}
 return {leaflet_version:L.version,layer_count:Object.keys(m._layers).length,
   checked_layers:document.querySelectorAll('.leaflet-control-layers-selector:checked').length,
   title:document.querySelector('#final-map-title').innerText,legend:document.querySelector('#final-map-legend').innerText,
   boxes,all_overlays_inside_viewport:within,no_overlay_overlap:noOverlap,body_scrolls:document.body.scrollWidth>innerWidth||document.body.scrollHeight>innerHeight,
   visible_layer_color_pixels:counts,canvas_shape:[canvas.width,canvas.height],layers};
}""")
        layers = proof.pop("layers")
        expected_prediction = [r["coordinates_latlon"] for r in data["predicted"]]
        expected_ref = [r["coordinates_latlon"] for r in data["reference"]]
        predicted = [row["coordinates"] for row in layers["prediction"]]
        require(predicted == expected_prediction, "actual Leaflet prediction order/geometry mismatch")
        require(layers["reference"][0]["coordinates"] == expected_ref, "actual reference geometry mismatch")
        require(np.array_equal(layers["network"][0]["coordinates"], data["graph"].nodes[data["graph"].arcs][:, :, ::-1]),
                "actual Leaflet full graph geometry mismatch")
        require(np.array_equal(layers["gps"][0]["coordinates"], data["gps"][:, [1, 0]])
                and np.array_equal([row["coordinates"] for row in layers["gps"][1:]], data["gps"][:, [1, 0]]),
                "actual raw GPS geometry mismatch")
        proof["actual_layer_counts"] = {key: len(rows) for key, rows in layers.items()}
        proof["prediction_geometry_latlon_sha256"] = object_hash(predicted)
        proof["prediction_geometry_matches_every_cli_arc"] = True
        proof["all_full_graph_and_gps_coordinates_match"] = True
        require(proof["all_overlays_inside_viewport"] and proof["no_overlay_overlap"] and not proof["body_scrolls"],
                "map title/legend/controls are clipped or overlap")
        require(all(count > 5 for count in proof["visible_layer_color_pixels"].values()),
                f"one mandatory geometry layer did not render visibly: {proof['visible_layer_color_pixels']}")
        require(TRIP_ID in proof["title"] and "HistGradientBoosting" in proof["title"], "wrong map title")
        controls = page.locator('.leaflet-control-layers-selector')
        controls.nth(1).uncheck()
        require(page.evaluate("window.__FINAL_MAP__.hasLayer(window.__FINAL_GROUPS__.reference)") is False,
                "reference toggle did not remove the actual reference layer")
        controls.nth(1).check()
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        proof["reference_toggle_verified"] = True
        page.screenshot(path=str(png), full_page=False)
        page.pdf(path=str(pdf), format="A4", landscape=True, print_background=True,
                 margin={"top": "0", "bottom": "0", "left": "0", "right": "0"}, scale=.585)
        proof["browser_version"] = browser.version
        browser.close()
    local_prefixes = (path.resolve().as_uri(), data["vendor"].resolve().as_uri() + "/")
    unexpected = [url for url in requests if not url.startswith(local_prefixes)]
    require(not page_errors and not console_errors and not blocked and not failed_requests and not unexpected,
            f"browser errors/remote requests: {page_errors}, {console_errors}, {blocked}, {failed_requests}, {unexpected}")
    ocr = subprocess.run(["tesseract", str(png), "stdout", "--psm", "11", "tsv"],
                         text=True, capture_output=True, check=True, timeout=180)
    rows = list(csv.DictReader(ocr.stdout.splitlines(), delimiter="\t"))
    words = [row for row in rows if row.get("text", "").strip() and float(row.get("conf", -1)) >= 0]
    text = " ".join(row["text"] for row in words)
    require("HistGradientBoosting" in text and TRIP_ID in text, "map title OCR missing required tokens")
    confidence = statistics.mean(float(row["conf"]) for row in words)
    require(confidence >= 55, "map OCR confidence below 55")
    for row in words:
        require(int(row["left"]) > 0 and int(row["top"]) > 0
                and int(row["left"]) + int(row["width"]) < 1920
                and int(row["top"]) + int(row["height"]) < 1080, "OCR text touches screenshot edge")
    output(path.with_suffix(".ocr.txt")).write_text(text + "\n", encoding="utf-8")
    from pypdf import PdfReader
    pdf_reader = PdfReader(pdf)
    require(len(pdf_reader.pages) == 1, "map PDF must fit one landscape A4 page")
    pdf_page = pdf_reader.pages[0]
    require(float(pdf_page.mediabox.width) > float(pdf_page.mediabox.height), "map PDF is not landscape")
    pdf_text = pdf_page.extract_text()
    require("HistGradientBoosting" in pdf_text and TRIP_ID in pdf_text, "map PDF lost title text")
    proof.update({"source_html": rel(path), "screenshot_path": rel(png), "pdf_path": rel(pdf),
                  "viewport": {"width": 1920, "height": 1080}, "page_errors": page_errors,
                  "console_errors": console_errors, "requests": requests, "blocked_network_requests": blocked,
                  "failed_requests": failed_requests, "external_requests": unexpected,
                  "no_external_requests": True, "no_console_errors": True,
                  "ocr": {"text_path": rel(path.with_suffix(".ocr.txt")), "words": len(words),
                          "mean_confidence": round(confidence, 2), "required_title_tokens_found": True},
                  "pdf": {"pages": 1, "width_pt": float(pdf_page.mediabox.width),
                          "height_pt": float(pdf_page.mediabox.height), "title_text_present": True},
                  "capture_method": "Actual installed Playwright Chromium; local file URL; all network blocked",
                  "inspection": "DOM, exact Leaflet coordinates, canvas color counts, OCR and PDF text; no human visual inspection claimed",
                  "created_at_utc": utcnow()})
    write_json(path.with_suffix(".browser.json"), proof)
    return proof


def verify_map_manifest() -> dict:
    manifest = read_json(MANIFEST)
    require(manifest.get("passed") is True, "map manifest did not pass")
    for record in manifest["files"] + manifest["inputs"]:
        path = ROOT / record["path"]
        require(path.is_file() and sha256_file(path) == record["sha256"], f"map manifest hash drift: {path}")
    metadata = read_json(ROOT / manifest["metadata_path"])
    require(metadata["model_artifact_sha256"] == MODEL_SHA256 and metadata["decoder_config_sha256"] == CONFIG_SHA256,
            "map manifest model/config mismatch")
    require(metadata["prediction_route_integrity"]["broken_joins"] == 0
            and metadata["prediction_arc_geometry_count"] == metadata["prediction_route_integrity"]["arc_rows"],
            "map geometry/traversal count mismatch")
    browser = read_json(ROOT / manifest["browser_path"])
    require(browser["no_external_requests"] and browser["no_console_errors"]
            and browser["prediction_geometry_matches_every_cli_arc"], "map browser proof failed")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true", help="Read-only verification of the completed map manifest")
    args = parser.parse_args(argv)
    if args.verify:
        manifest = verify_map_manifest()
        print(f"map manifest verified: {len(manifest['files'])} files, {len(manifest['inputs'])} input pins")
        return 0
    data = load_evidence()
    render = render_html(data, OUTPUT)
    browser = capture(OUTPUT, data)
    after = {rel(path): sha256_file(path) for path in data["input_paths"]}
    require(after == data["input_sha256"], "frozen inputs changed during map generation")
    metadata = {"phase": "five-model-completion", "trip_id": TRIP_ID, "partition": "Student development (student_test)",
                "development_only": True, "contains_reference_geometry": True, "do_not_serve_in_application": True,
                "prediction_source": rel(data["prediction"]), "model_family": data["model_metadata"]["model_name"],
                "model_run": data["model_metadata"]["run_name"], "prediction_label": data["label"],
                "model_artifact_sha256": MODEL_SHA256, "decoder_config_sha256": CONFIG_SHA256,
                "training_trip_count": data["model_metadata"]["training_trip_count"],
                "nodes": len(data["graph"].nodes), "directed_arcs": len(data["graph"].arcs),
                "gps_points": len(data["gps"]), "prediction_arc_geometry_count": len(data["predicted"]),
                "prediction_route_integrity": data["prediction_integrity"], "reference_route_integrity": data["reference_integrity"],
                "prediction_geometry_latlon_sha256": object_hash([r["coordinates_latlon"] for r in data["predicted"]]),
                "route_metrics_recomputed_equal_delivery": data["metrics"], "tiles": None,
                "geometry_note": "Every predicted arc is drawn in CLI order, including duplicate occurrences, using its supplied endpoints. No road shape or connector is invented.",
                "proxy_note": "Observation labels use geometry-and-motion proxies; route overlap compares supplied arc traversals directly.",
                "time_note": "Traversal tooltips report observed time ranges pooled by arc ID; per-traversal times are never inferred.",
                "input_sha256": data["input_sha256"], "frozen_input_hashes_unchanged_before_after": True,
                "render": render, "created_at_utc": utcnow()}
    write_json(OUTPUT.with_suffix(".metadata.json"), metadata)
    files = [OUTPUT.with_suffix(suffix) for suffix in (".html", ".png", ".pdf", ".metadata.json", ".browser.json", ".ocr.txt")]
    manifest = {"phase": "five-model-completion", "passed": True, "development_only": True,
                "contains_reference_geometry": True, "do_not_serve_in_application": True,
                "map_path": rel(OUTPUT), "metadata_path": rel(OUTPUT.with_suffix(".metadata.json")),
                "browser_path": rel(OUTPUT.with_suffix(".browser.json")), "figure_count": 1,
                "files": [{"path": rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in files],
                "inputs": [{"path": path, "sha256": digest} for path, digest in after.items()],
                "model_artifact_sha256": MODEL_SHA256, "decoder_config_sha256": CONFIG_SHA256,
                "created_at_utc": utcnow()}
    write_json(MANIFEST, manifest)
    verify_map_manifest()
    print(f"map: {rel(OUTPUT)}; prediction {len(data['pred'])} arcs; {data['prediction_integrity']['duplicate_occurrences_beyond_first']} repeat occurrences; {browser['layer_count']} live layers")
    print(f"map manifest: {rel(MANIFEST)}; SHA-256 {sha256_file(MANIFEST)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
