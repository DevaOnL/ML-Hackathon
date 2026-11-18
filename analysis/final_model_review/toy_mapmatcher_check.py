"""Toy reference-free MapMatcher API check with the ACTUAL fitted final models.

Builds a synthetic metric graph and a 3-observation GPS track (no route files,
no reference labels, no fitting, no HTTP) and decodes it with each saved
final-five estimator as the live emission model. Records the real learned
emission vectors, checks normalization/separation, and confirms the frozen
selected model behaves identically to the tuned baseline artifact.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mapmatch.learning import load_emission_model, predict_positive_proba
from mapmatch.matcher import MapMatcher, MatchConfig
from tests.test_core import metric_graph, metric_track

OUT = {"models": {}, "failures": []}


def check(name, ok, detail=""):
    entry = {"check": name, "passed": bool(ok), "detail": str(detail)[:300]}
    OUT.setdefault("checks", []).append(entry)
    if not ok:
        OUT["failures"].append(entry)


graph = metric_graph([(0, 0), (100, 0), (200, 0)], [(0, 1), (1, 2), (1, 0), (2, 1)])
track = metric_track([(10, 1), (50, -1), (150, 2)])  # 3 toy observations
config = MatchConfig()

ARTIFACTS = {
    "logistic_regression": "artifacts/baselines/logistic_regression.joblib",
    "knn_distance_weighted": "artifacts/phase_three/baselines/knn_distance_weighted.joblib",
    "bagging": "artifacts/phase_three/baselines/bagging.joblib",
    "gaussian_nb": "artifacts/final_five/baselines/gaussian_nb.joblib",
    "hist_gradient_boosting": "artifacts/final_five/baselines/hist_gradient_boosting.joblib",
    "hist_gradient_boosting_tuned": "artifacts/final_five/baselines/hist_gradient_boosting_tuned.joblib",
    "frozen_selected_model": "artifacts/final_five/model.joblib",
}


class Recorder:
    """Delegates to the real EmissionModel while capturing learned emissions."""

    def __init__(self, model):
        self.model = model
        self.vectors = []

    def score_candidates(self, graph, track, xy, index, candidates):
        s = self.model.score_candidates(graph, track, xy, index, candidates)
        self.vectors.append(np.asarray(s, dtype=float))
        return s

    def score_candidates_batch(self, graph, track, indices, candidates):
        out = self.model.score_candidates_batch(graph, track, indices, candidates)
        for s in out:
            self.vectors.append(np.asarray(s, dtype=float))
        return out

    def __getattr__(self, item):
        return getattr(self.model, item)


vectors = {}
for name, rel in ARTIFACTS.items():
    em = load_emission_model(ROOT / rel)
    rec = Recorder(em)
    result = MapMatcher(graph, config, emission_model=rec).match(track)
    route = list(result.route)
    edges = list(result.matched_edges)
    ok_route = all(0 <= e < 4 for e in edges) and len(edges) == 3 and len(route) >= 1
    nll = result.diagnostics["negative_log_likelihood_relative"]
    ok_diag = ("learned" in str(result.diagnostics["emission_model"])
               and result.diagnostics["uses_reference_route"] is False
               and math.isfinite(nll))
    vecs = rec.vectors
    norms = [float(np.abs(v.sum() - 1.0)) for v in vecs]
    ranges = [(float(v.min()), float(v.max())) for v in vecs]
    ok_em = all(n < 1e-7 for n in norms) and all(0.0 <= lo <= hi <= 1.0 for lo, hi in ranges)
    vectors[name] = vecs
    check(f"toy_match_{name}", ok_route and ok_diag and ok_em,
          f"route={route} edges={edges} nll={nll:.4f} sum_err_max={max(norms):.2e} "
          f"obs0_scores={[round(x, 6) for x in vecs[0].tolist()]}")

# separation: the six estimators must not all emit identical vectors
sigs = {name: tuple(np.round(v[0], 9).tolist()) for name, v in vectors.items()
        if name != "frozen_selected_model"}
check("toy_emission_separation", len(set(sigs.values())) >= 2,
      f"distinct first-observation signatures={len(set(sigs.values()))} of {len(sigs)}")

# frozen selected model must match the tuned baseline artifact exactly
a = load_emission_model(ROOT / ARTIFACTS["frozen_selected_model"])
b = load_emission_model(ROOT / ARTIFACTS["hist_gradient_boosting_tuned"])
va = np.concatenate([v for v in vectors["frozen_selected_model"]])
vb = np.concatenate([v for v in vectors["hist_gradient_boosting_tuned"]])
check("frozen_matches_tuned_baseline_emissions", va.shape == vb.shape
      and float(np.max(np.abs(va - vb))) <= 1e-12,
      f"max_delta={float(np.max(np.abs(va - vb))) if va.shape == vb.shape else 'shape'}")

# GNB strict guard passes through the real toy pipeline (float64 promotion)
gnb_vecs = vectors["gaussian_nb"]
check("gnb_toy_probability_guard", all(abs(float(v.sum()) - 1.0) < 1e-7 for v in gnb_vecs),
      f"sum_errors={[abs(float(v.sum()) - 1.0) for v in gnb_vecs]}")

OUT["summary"] = {"total": len(OUT.get("checks", [])),
                  "passed": sum(c["passed"] for c in OUT.get("checks", [])),
                  "failed": len(OUT["failures"]),
                  "note": "synthetic metric graph + 3-observation track; actual fitted estimators; no reference route, no fitting, no HTTP"}
print(json.dumps(OUT, indent=1))
