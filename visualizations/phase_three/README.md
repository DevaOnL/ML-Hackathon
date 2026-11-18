# Phase-three visuals (task-26)

Generator: `generate_phase_three.py` — writes **only** under
`visualizations/phase_three/`. Historical `visualizations/plots`,
`visualizations/maps` and the root `figure_manifest.json` are never
touched; the 12 EDA assets are linked read-only with hash verification.

Phase families (user-selected 18 Nov 2026): **logistic_regression,
knn_distance_weighted, bagging** + the tuned variant of the selected
family. GNB/HistGB/RF/ET are historical prior work, never relabeled.

## Exact commands (run from repo root, `.venv/bin/python`)

```bash
# readiness probe (no writes)
.venv/bin/python visualizations/phase_three/generate_phase_three.py status

# after PHASE_RESULTS_READY (experiments/phase_three/results.json complete):
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python visualizations/phase_three/generate_phase_three.py charts

# after the phase decoder publishes reports/phase_three/decoder/student_ablation.json:
.venv/bin/python visualizations/phase_three/generate_phase_three.py decoder

# after PHASE_FINAL_ROUTES_READY (outputs/phase_three_student_test/routes/00000009.route):
.venv/bin/python visualizations/phase_three/generate_phase_three.py map

# rebuild manifest + full verification (hashes, scope, leakage scan, OCR):
.venv/bin/python visualizations/phase_three/generate_phase_three.py manifest
.venv/bin/python visualizations/phase_three/generate_phase_three.py verify

# or everything at once (skips stages whose inputs are not ready, exit 2 when forced):
.venv/bin/python visualizations/phase_three/generate_phase_three.py all
```

Every stage is idempotent (persistent `.figure_registry.json`, upsert by
filename) and refuses incomplete evidence with exit code 2 and a clear
message. `status` reports readiness of results/ablation/route, vendor
Leaflet, and tesseract.

## Output contracts

- `plots/phase_three_*.png` — CV macro-F1 with fold error bars, binary
  log loss, fit/inference timing (2×2 one-quantity-per-panel; log axis
  only on the µs/observation panel, disclosed in the caption), train/CV/
  student gap, **actual** Bayesian trial trace (all 8 configured trials,
  no interpolated curves), tuned-vs-default deltas (tiny gains reported
  exactly, no scale manipulation), confusion matrices for all 4 runs,
  observation-level edge macro-F1 + categorical NLL (graph-namespaced,
  distinct from binary candidate metrics, denominators disclosed),
  raw-classifier continuity, ambiguity strata from the recorded
  `ambiguity_strata` diagnostics (candidate_miss stratum disclosed in
  the caption, not charted), and feature importance computed read-only
  from the frozen `bagging_tuned` artifact (SHA-256 verified against
  results.json before load; mean impurity/Gini over the 48 fitted trees,
  sum sanity-checked ≈ 1.0; bias caveat in the caption). Route ablation
  bars come from the new phase decoder (all 10 trips).
- `maps/phase_three_development_trip_00000009_final.{html,png}` — GPS
  blue, roads gray, reference green, learned prediction orange dashed;
  offline Leaflet; repeats/loops preserved; reference discontinuities
  shown as-is (no invented connectors). Captured with genuine Chromium
  (`*.browser.json`: layer count, JS errors, blocked requests, color
  pixel counts, legend DOM text; `*.ocr.txt` when tesseract exists).
  Internal report map — never evaluator-facing.
- `figure_manifest.json` — relative paths, SHA-256, bytes, exact
  captions with split/proxy caveats, input hashes (results.json, decoder
  ablation, frozen artifact, observation-metrics supplement), 12
  read-only EDA links (hash-verified against the completed root
  manifest), retained historical map links, and the map companions
  (rendered PNG, `*.browser.json` render proof, `*.metadata.json`
  provenance, `*.ocr.txt`) recorded with verified hashes.
- `asset_verification.json` — verification result (exit 0 = passed):
  figure hashes, scope containment, EDA link integrity, evaluator
  leakage scan, OCR captures.

## Honest-evidence rules encoded in the tool

- Every plotted number is read from stable measured files; nothing is
  invented and no epoch curves are fabricated.
- Binary candidate-row metrics, graph-namespaced directed-edge
  observation metrics, categorical NLL, and route multiset/edit metrics
  are always labeled separately with their denominators.
- Proxy-derived labels are disclosed in every caption (PROXY_NOTE).
- Results validation requires exactly the three phase families + tuned
  variant, Bayesian trial count == config, and observation-level
  supplements for every run; incomplete evidence is refused, not
  partially charted.
