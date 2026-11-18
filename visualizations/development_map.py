"""Private report-only Folium rendering; NEVER expose reference maps via evaluator UI/API."""
from __future__ import annotations
from datetime import datetime, timezone
import html
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from common import ROOT, OUT, MAPS, SPLITS, COLORS, safe_output, sha256, write_json, save_plot


def load_development_geometry(graph_prefix, prediction_path, development_reference_path):
    prefix = graph_prefix.resolve(); reference = development_reference_path.resolve(); prediction = prediction_path.resolve()
    allowed = [ROOT/"Student folder"/s for s in SPLITS]
    if not any(prefix.is_relative_to(p.resolve()) for p in allowed):
        raise ValueError("Reference maps permit only original TRAIN or student_test development geometry")
    if reference != prefix.with_suffix(".route"):
        raise ValueError("Pass the chosen original development reference separately, for this exact graph prefix")
    if prediction == reference: raise ValueError("A reference route cannot be presented as a prediction")
    nodes = np.loadtxt(prefix.with_suffix(".nodes"), ndmin=2); arcs = np.loadtxt(prefix.with_suffix(".arcs"), dtype=np.int64, ndmin=2)
    gps = np.loadtxt(prefix.with_suffix(".track"), ndmin=2); pred = np.loadtxt(prediction, dtype=np.int64, ndmin=1).reshape(-1); ref = np.loadtxt(reference, dtype=np.int64, ndmin=1).reshape(-1)
    if nodes.shape[1] != 2 or arcs.shape[1] != 2 or gps.shape[1] != 3 or not np.isfinite(nodes).all() or not np.isfinite(gps).all(): raise ValueError("Invalid graph/GPS geometry")
    if arcs.min() < 0 or arcs.max() >= len(nodes): raise ValueError("Out-of-range arc endpoints")
    for route in (pred, ref):
        if not len(route) or route.min() < 0 or route.max() >= len(arcs): raise ValueError("Out-of-range route arc IDs")
    return prefix, nodes, arcs, gps, pred, ref


def geometry_lines(nodes, arcs, route):
    # Independent arc geometries: never invent connectors over broken joins.
    return nodes[arcs[route]]


def relevant_roads(nodes, arcs, gps):
    low = gps[:, :2].min(axis=0); high = gps[:, :2].max(axis=0); pad = np.maximum((high-low)*.10, .001); geom = nodes[arcs]
    mask = np.all(geom.max(axis=1) >= low-pad, axis=1)&np.all(geom.min(axis=1) <= high+pad, axis=1); ids = np.flatnonzero(mask)
    # Show reciprocal/duplicate geometry once; the underlying graph is directed.
    _, positions = np.unique(np.sort(arcs[ids], axis=1), axis=0, return_index=True); ids = ids[np.sort(positions)]
    return nodes[arcs[ids]], len(ids)


def local_leaflet_assets():
    destination = safe_output(OUT/"vendor"/"leaflet"/"leaflet.js").parent; source = ROOT/"frontend"/"vendor"/"leaflet"
    for name in ("leaflet.js", "leaflet.css", "LICENSE"):
        target = destination/name
        if target.exists(): continue
        if not (source/name).is_file(): raise FileNotFoundError(f"Offline Leaflet asset missing: {source/name}")
        shutil.copyfile(source/name, safe_output(target))
    for name in ("layers.png", "layers-2x.png", "marker-icon.png", "marker-icon-2x.png", "marker-shadow.png"):
        target = destination/"images"/name
        if not target.exists(): shutil.copyfile(source/"images"/name, safe_output(target))
    return destination


def generate_development_map(graph_prefix: Path, prediction_path: Path, development_reference_path: Path, output_path: Path, *, prediction_label: str):
    """Reference is an explicit separate input; only permitted development partitions.

    Output includes reference geometry. Keep this module and its HTML out of the
    evaluator service/static allowlist, even if the repository is public.
    """
    import folium
    from branca.element import Element, MacroElement
    from jinja2 import Template
    prefix, nodes, arcs, gps, pred, ref = load_development_geometry(graph_prefix, prediction_path, development_reference_path)
    output_path = safe_output(output_path); vendor = local_leaflet_assets(); rel = Path(os.path.relpath(vendor, output_path.parent)).as_posix()
    roads, road_count = relevant_roads(nodes, arcs, gps); center = gps[:, :2].mean(axis=0)
    m = folium.Map(location=[center[1], center[0]], zoom_start=13, tiles=None, prefer_canvas=True, control_scale=True, zoom_control=True, attribution_control=True)
    # No jQuery/Bootstrap/icons needed: replace every Folium CDN dependency.
    m.default_js = [("leaflet", f"{rel}/leaflet.js")]; m.default_css = [("leaflet_css", f"{rel}/leaflet.css")]
    m.get_root().header.add_child(Element('<meta name="robots" content="noindex,nofollow">'))
    m.get_root().header.add_child(Element('<style>.leaflet-container{background:#f4f7fa;font-family:Arial,sans-serif}.leaflet-control-layers{font-size:12px;max-width:420px}.leaflet-control-layers label{white-space:normal}.leaflet-control-attribution{font-size:10px}</style>'))
    layer = folium.FeatureGroup(name=f"Relevant road network · gray · {road_count:,} chords", show=True)
    folium.PolyLine(roads[:, :, ::-1].tolist(), color=COLORS["roads"], weight=1.2, opacity=.50).add_to(layer); layer.add_to(m)
    layer = folium.FeatureGroup(name="Development reference traversal · green (not evaluator data)", show=True)
    folium.PolyLine(geometry_lines(nodes, arcs, ref)[:, :, ::-1].tolist(), color=COLORS["reference"], weight=7, opacity=.70).add_to(layer); layer.add_to(m)
    layer = folium.FeatureGroup(name=f"Prediction · orange dashed · {prediction_label}", show=True)
    folium.PolyLine(geometry_lines(nodes, arcs, pred)[:, :, ::-1].tolist(), color=COLORS["prediction"], weight=3.4, opacity=1, dash_array="8,5").add_to(layer); prediction_layer = layer
    layer = folium.FeatureGroup(name=f"Raw GPS · blue · {len(gps):,} observations", show=True)
    folium.PolyLine(gps[:, [1, 0]].tolist(), color=COLORS["gps"], weight=1, opacity=.40, dash_array="2,5").add_to(layer)
    for lon, lat, timestamp in gps:
        folium.CircleMarker([lat, lon], radius=2.0, weight=0, fill=True, fill_color=COLORS["gps"], fill_opacity=.75).add_to(layer)
    layer.add_to(m)
    layer = folium.FeatureGroup(name="First / last raw GPS position", show=True)
    for i, text in ((0, "First raw GPS row"), (-1, "Last raw GPS row")):
        folium.CircleMarker(gps[i, [1, 0]].tolist(), radius=6, weight=2, color="#ffffff" if i == 0 else "#142033", fill=True, fill_color=COLORS["gps"], fill_opacity=1, tooltip=text).add_to(layer)
    layer.add_to(m)
    # Draw the thin dashed prediction above dense raw points, with the wider
    # green reference still visible as a halo. No geometry is shifted.
    prediction_layer.add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    label = html.escape(prediction_label); partition = prefix.parent.parent.name
    card = f'''<div id="development-map-title" style="position:fixed;top:14px;left:60px;z-index:9999;background:#ffffffee;border:1px solid #cbd5e1;border-radius:10px;padding:13px 18px;box-shadow:0 3px 14px #152d4a22;max-width:650px">
    <div style="font:700 21px Arial;color:#142033">PathSnap · trip {prefix.name} · {partition} development only</div>
    <div style="font:13px Arial;color:#475569;margin-top:7px">{len(gps):,} GPS points · {len(nodes):,} nodes · {len(arcs):,} directed arcs</div>
    <div style="font:700 12px Arial;color:#9a3412;margin-top:7px">INTERNAL DEVELOPMENT REPORT MAP — includes reference geometry; never serve in evaluator UI/API</div></div>
    <div id="development-map-legend" style="position:fixed;bottom:28px;left:20px;z-index:9999;background:#ffffffee;border:1px solid #cbd5e1;border-radius:10px;padding:15px 18px;box-shadow:0 3px 14px #152d4a22;max-width:690px;font:14px Arial;line-height:1.8;color:#142033">
    <div style="font-weight:700;font-size:15px;margin-bottom:5px">Actual observations and traversed-arc geometry</div>
    <div><span style="color:{COLORS['gps']};font-weight:900">● ┄</span> Raw GPS observations (blue); dotted line shows observation order</div>
    <div><span style="color:{COLORS['roads']};font-weight:900">━━</span> Relevant road network (gray endpoint chords)</div>
    <div><span style="color:{COLORS['reference']};font-weight:900">━━</span> Supplied development reference traversal (green)</div>
    <div><span style="color:{COLORS['prediction']};font-weight:900">┅┅</span> {label} (orange dashed)</div>
    <div style="font-size:12px;color:#475569;margin-top:6px">Offline Folium / Leaflet · no web tiles · layer controls at upper right<br>Reference traversal is not independently verified pointwise ground truth.</div></div>'''
    m.get_root().html.add_child(Element(card))
    m.fit_bounds([[gps[:, 1].min(), gps[:, 0].min()], [gps[:, 1].max(), gps[:, 0].max()]], padding_top_left=(55, 115), padding_bottom_right=(30, 245))
    ready = MacroElement(); ready._template = Template("{% macro script(this, kwargs) %}window.__PATHSNAP_DEVELOPMENT_MAP_READY__ = true;{% endmacro %}"); m.add_child(ready)
    m.save(str(output_path)); text = output_path.read_text(encoding="utf-8")
    if re.search(r'''(?:src|href)=["']https?://''', text): raise RuntimeError("Unexpected remote dependency in offline Folium HTML")
    inputs = [prefix.with_suffix(".nodes"), prefix.with_suffix(".arcs"), prefix.with_suffix(".track"), development_reference_path.resolve(), prediction_path.resolve()]
    metadata = {"map_path": str(output_path.relative_to(ROOT)), "development_only": True, "contains_reference_geometry": True, "do_not_serve_to_evaluator": True, "trip_id": prefix.name, "partition": partition, "prediction_label": prediction_label, "graph_prefix": str(prefix.relative_to(ROOT)), "prediction_path": str(prediction_path.resolve().relative_to(ROOT)), "development_reference_path": str(development_reference_path.resolve().relative_to(ROOT)), "gps_points": len(gps), "nodes": len(nodes), "directed_arcs": len(arcs), "displayed_road_chords": road_count, "prediction_arc_rows": len(pred), "reference_arc_rows": len(ref), "tiles": None, "leaflet_assets": str(vendor.relative_to(ROOT)), "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in inputs}, "folium_version": folium.__version__}
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    plot_representative(prefix, nodes, arcs, gps, pred, ref, roads, prediction_label)
    print(f"Offline development Folium map: {output_path.relative_to(ROOT)} ({output_path.stat().st_size:,} bytes)")
    return output_path


def plot_representative(prefix, nodes, arcs, gps, pred, ref, roads, prediction_label):
    fig, axs = plt.subplots(1, 2, figsize=(13.4, 6.5))
    for ax in axs:
        ax.add_collection(LineCollection(roads, colors=COLORS["roads"], lw=.65, alpha=.45, zorder=1)); ax.add_collection(LineCollection(geometry_lines(nodes, arcs, ref), colors=COLORS["reference"], lw=4.5, alpha=.75, zorder=2))
        ax.add_collection(LineCollection(geometry_lines(nodes, arcs, pred), colors=COLORS["prediction"], lw=2.0, linestyles="dashed", zorder=5)); ax.scatter(gps[:, 0], gps[:, 1], s=3.2, color=COLORS["gps"], alpha=.8, zorder=4)
        ax.set_aspect(1/math.cos(math.radians(float(np.median(gps[:, 1])))), adjustable="box"); ax.set(xlabel="Longitude (degrees)", ylabel="Latitude (degrees)"); ax.grid(alpha=.4)
    low = gps[:, :2].min(axis=0); high = gps[:, :2].max(axis=0); delta = high-low
    axs[0].set(xlim=(low[0]-delta[0]*.05, high[0]+delta[0]*.05), ylim=(low[1]-delta[1]*.06, high[1]+delta[1]*.06), title=f"Full observed extent · {len(gps):,} GPS points")
    center = gps[len(gps)//2, :2]; width = np.maximum(delta*.22, .002)
    axs[1].set(xlim=(center[0]-width[0]/2, center[0]+width[0]/2), ylim=(center[1]-width[1]/2, center[1]+width[1]/2), title="Detail at temporal midpoint (not best-match selection)")
    handles = [Line2D([0], [0], color=COLORS["gps"], marker="o", lw=0, markersize=4, label="Raw GPS (blue)"), Line2D([0], [0], color=COLORS["roads"], lw=1.5, label="Road network (gray)"), Line2D([0], [0], color=COLORS["reference"], lw=4, label="Development reference (green)"), Line2D([0], [0], color=COLORS["prediction"], ls="--", lw=2, label="Prediction (orange)")]
    fig.legend(handles=handles, ncols=4, loc="lower center", bbox_to_anchor=(.5, .065), fontsize=9)
    save_plot(fig, "representative_road_route.png", f"Illustrative development road graph / route · trip {prefix.name}", f"{prediction_label}. Development reference geometry is internal to this report, never evaluator-facing.\nArcs use supplied node-endpoint chords; repeated traversal arcs retained; broken joins are not filled with invented connectors.")


def capture_map_screenshot(html_path: Path, *, screenshot_path: Path | None = None):
    from playwright.sync_api import sync_playwright
    screenshot_path = safe_output(screenshot_path or html_path.with_suffix(".png")); errors, requests, blocked = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True); page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("pageerror", lambda e: errors.append(str(e))); page.on("request", lambda req: requests.append(req.url))
        def reject_network(route): blocked.append(route.request.url); route.abort()
        page.route(re.compile(r"^https?://"), reject_network); page.goto(html_path.resolve().as_uri(), wait_until="load")
        page.wait_for_function("window.__PATHSNAP_DEVELOPMENT_MAP_READY__ === true && typeof L !== 'undefined'"); page.wait_for_function("document.querySelector('.leaflet-control-layers-overlays') !== null")
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
        if not all(v > 5 for v in proof["layout_and_canvas"]["visible_layer_color_pixels"].values()):
            raise RuntimeError("One mandatory actual map geometry colour is not visibly rendered")
        page.screenshot(path=str(screenshot_path), full_page=True)
        proof.update({"browser_version": browser.version, "source_html": str(html_path.relative_to(ROOT)), "screenshot_path": str(screenshot_path.relative_to(ROOT)), "viewport": {"width": 1440, "height": 1000}, "page_errors": errors, "requests": requests, "blocked_network_requests": blocked, "capture_method": "actual installed Playwright Chromium, local file:// HTML, HTTP(S) blocked", "created_at_utc": datetime.now(timezone.utc).isoformat()}); browser.close()
    if errors or blocked or proof["layer_count"] < 20: raise RuntimeError(f"Map failed browser verification: {proof}")
    if shutil.which("tesseract"):
        result = subprocess.run(["tesseract", str(screenshot_path), "stdout", "--psm", "11"], text=True, capture_output=True, check=True); safe_output(screenshot_path.with_suffix(".ocr.txt")).write_text(result.stdout, encoding="utf-8"); proof["ocr_text_path"] = str(screenshot_path.with_suffix(".ocr.txt").relative_to(ROOT))
    write_json(screenshot_path.with_suffix(".browser.json"), proof)
    print(f"Actual Chromium screenshot: {screenshot_path.relative_to(ROOT)}; {proof['layer_count']:,} live Leaflet layers; {len(errors)} JS errors")
