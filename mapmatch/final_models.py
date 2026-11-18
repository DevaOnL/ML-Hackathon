"""Exact final five-family estimator registry; no data access or fitting here."""
from __future__ import annotations

from mapmatch.experiments import ExperimentConfig, default_parameters, make_estimator
from mapmatch.phase_models import PHASE_DEFAULT_PARAMETERS, make_phase_estimator

FINAL_FAMILIES = ("logistic_regression", "knn_distance_weighted", "bagging", "gaussian_nb", "hist_gradient_boosting")
FINAL_NEW_FAMILIES = ("gaussian_nb", "hist_gradient_boosting")
FINAL_DEFAULTS = {**{k: dict(PHASE_DEFAULT_PARAMETERS[k]) for k in FINAL_FAMILIES[:3]}, **{k: dict(default_parameters()[k]) for k in FINAL_NEW_FAMILIES}}


def make_final_estimator(name: str, parameters: dict | None = None, *, seed: int = 20261118, n_jobs: int = 2):
    if name not in FINAL_FAMILIES:
        raise ValueError("unknown final-family estimator: " + name)
    params = dict(FINAL_DEFAULTS[name] if parameters is None else parameters)
    if name in FINAL_FAMILIES[:3]:
        return make_phase_estimator(name, params, seed, n_jobs)
    return make_estimator(name, params, ExperimentConfig(seed=seed, n_jobs=n_jobs))
