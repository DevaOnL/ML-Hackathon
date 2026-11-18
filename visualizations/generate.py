#!/usr/bin/env python3
"""Reproduce factual PathSnap EDA and an INTERNAL DEVELOPMENT-ONLY Folium map.

Default: all 80 training + 10 student-development EDA, all90 GPS galleries, and
an explicitly interim geometry-baseline map. No evaluator data is ever read.
Pass --experiments only AFTER the learning agent publishes complete/stable JSON.
All writes stay inside visualizations/; no web server or network access needed.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
from common import ROOT, MAPS, update_figure_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-gallery", action="store_true")
    parser.add_argument("--skip-map", action="store_true")
    parser.add_argument("--screenshot", action="store_true", help="Use actual installed Chromium; no HTTP server")
    parser.add_argument("--graph-prefix", type=Path, default=Path("Student folder/student_test/00000009/00000009"))
    parser.add_argument("--prediction", type=Path, default=Path("outputs/baseline_student_test_v2/routes/00000009.route"))
    parser.add_argument("--reference", type=Path, default=Path("Student folder/student_test/00000009/00000009.route"))
    parser.add_argument("--prediction-label", default="Geometry baseline v2 prediction — interim, NOT final learned model")
    parser.add_argument("--map-name", default="development_trip_00000009_baseline")
    parser.add_argument("--experiments", type=Path, help="Read COMPLETE stable learning results JSON only; never pass a running output")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.map_name): parser.error("--map-name must be a simple filename stem")
    if not args.skip_eda:
        from eda import generate_eda
        generate_eda(gallery=not args.skip_gallery)
    if not args.skip_map:
        from development_map import generate_development_map, capture_map_screenshot
        def rooted(p): return p if p.is_absolute() else ROOT/p
        path = generate_development_map(rooted(args.graph_prefix), rooted(args.prediction), rooted(args.reference), MAPS/(args.map_name+".html"), prediction_label=args.prediction_label)
        if args.screenshot: capture_map_screenshot(path)
    if args.experiments:
        from model_charts import generate_experiment_charts
        generate_experiment_charts(args.experiments if args.experiments.is_absolute() else ROOT/args.experiments)
    update_figure_manifest()


if __name__ == "__main__":
    main()
