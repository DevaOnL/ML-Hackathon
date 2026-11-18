# Reproducible factual visualizations

These assets describe **80 training + 10 student-development trips only**. No reserved evaluator geometry or labels are read. Plots use the completed, full-dataset [audit](<../analysis/data_audit/summary.json>) and original raw tracks, not the smaller classifier-training sample.

## Reproduce

From the repository root, with the supplied Python 3.14 environment:

~~~bash
.venv/bin/python visualizations/generate.py --screenshot
~~~

This builds the inventory, missing/invalid-value checks, trajectory distributions, timestamp-gap distributions, network-size/connectivity plots, all90 raw-GPS galleries, and an **explicitly interim geometry-baseline** development map. Every original track is read fully and SHA256-checked against the completed file audit. No observations are subsampled for EDA. The plotting style uses system matplotlib; the actual interactive map is built with Folium 1.0 and local Leaflet 1.9.4 assets.

After the learning teammate has explicitly confirmed a **complete, atomically published** results file:

~~~bash
.venv/bin/python visualizations/generate.py --skip-eda --skip-map --experiments experiments/results.json
~~~

The experiment plotter rejects missing models/evaluations or incomplete trial counts. It plots measured CV folds, student-development Top-1 and binary-candidate Macro F1, log loss, training/inference cost, per-model confusion matrices, Bayesian trial trace and untuned/tuned comparisons. It never invents epoch curves for non-epoch algorithms. Model accuracy/Macro F1 here are proxy-label agreement, not independently supplied per-GPS ground truth or reserved-evaluator accuracy. Captions report the actual classifier-observation sampling denominator separately from the complete EDA denominator.

For the final stable learned prediction, pass the graph prefix, prediction and original **development** reference separately:

~~~bash
.venv/bin/python visualizations/generate.py --skip-eda --screenshot \
  --graph-prefix 'Student folder/student_test/00000009/00000009' \
  --prediction outputs/student_test/routes/00000009.route \
  --reference 'Student folder/student_test/00000009/00000009.route' \
  --prediction-label 'Final learned classifier + HMM prediction (see final decoder configuration)' \
  --map-name development_trip_00000009_final
~~~

The command is an example for a final output **only after that file exists and its producer confirms it is stable**. An interim baseline is never silently relabelled final. A map regeneration refreshes the static representative road/route plot to the same explicitly labelled prediction.

## Output evidence

- [Figure manifest](<figure_manifest.json>): PNG paths, captions, source, partition, status, size and SHA256 for the consolidated PDF producer.
- [EDA evidence](<eda_summary.json>): exact all90 counts, timestamp categories/tolerance and all track hashes.
- [Interactive baseline map](<maps/development_trip_00000009_baseline.html>) and [actual browser screenshot](<maps/development_trip_00000009_baseline.png>): student-development trip 00000009, labelled geometry baseline v2 / interim / NOT final.
- [Browser evidence](<maps/development_trip_00000009_baseline.browser.json>) and [screenshot OCR](<maps/development_trip_00000009_baseline.ocr.txt>): real installed Chromium rendering, live Leaflet-layer counts, legend text, zero JavaScript errors and no external requests.
- [Map input provenance](<maps/development_trip_00000009_baseline.metadata.json>): exact graph, prediction, reference, hashes and leakage warning.

The Folium map uses no web tiles and no CDN JavaScript/CSS. All base road geometry comes from the supplied nodes/arcs. It opens directly as a local HTML file with the adjacent vendored assets; the screenshot needs no HTTP server. Leaflet retains its original BSD license in the [vendor directory](<vendor/leaflet/LICENSE>).

## Safety and interpretation

**Never host these development-reference maps in the evaluator frontend or API.** They intentionally contain the permitted development reference route for the technical report. The map generator rejects graph prefixes outside the original train/student_test partitions, rejects a reference that is not the chosen original graph reference, and refuses to use that same file as the prediction. No evaluator service imports this plotting module.

Map layers are raw GPS **blue**, road network **gray**, supplied development reference **green**, prediction **orange dashed**. Layer controls and an explicit legend distinguish coincident geometries. Arc geometry is the supplied pair of endpoint coordinates: road shapes between endpoints are not invented. Broken route joins are left disconnected rather than drawn as fabricated straight links.

All90 EDA charts include every observation. Timestamps near 1 second use absolute tolerance 1e-6 s only to avoid binary floating-point artefacts; exact audit inequality counts are also retained. A timestamp gap is not assumed to be a missing-value row. Apparent speed is GPS displacement divided by supplied seconds, not independently measured speed. Geographical overview axes are longitude/latitude, not a distance projection. GPS galleries use independent local equirectangular-km axes centered on each trip; they must not be compared as equal spatial scales.

Supplied reference routes are chronological arc traversals and have a small number of broken joins. They are not per-GPS labels, and agreement with a derived nearest-reference/motion proxy does not establish physical truth. Network totals sum trip-specific graphs that may overlap spatially; they are not counts of a unique global road network.
