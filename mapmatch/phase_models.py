"""Phase-three model families: LogisticRegression, distance-weighted KNN, Bagging.

User-selected scope (docs/PHASE_SCOPE.json, task-16, 18 November 2026). This
module only CONSTRUCTS standard scikit-learn estimators and the deterministic
KNN reference-bank policy; saved artifacts therefore contain no custom pickled
classes and load through the existing mapmatch.learning.load_emission_model.
It deliberately does not touch or monkeypatch mapmatch.experiments; shared
primitives are imported read-only by the phase runner.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PHASE_NAME = "three-model-first"
PHASE_MODEL_COUNT = 3

PHASE_FAMILIES = {
    "logistic_regression": "scaled linear discriminative logistic regression",
    "knn_distance_weighted": "distance-weighted k-nearest-neighbour candidate classifier over a class-stratified TRAIN reference bank",
    "bagging": "genuine bootstrap aggregation of decision-tree base learners",
}

# Predeclared modest defaults, fixed before any measurement. The logistic
# regression entry is IDENTICAL to the valid saved shared baseline so its
# grouped CV / all80 fit / student evaluation can be reused without refitting.
PHASE_DEFAULT_PARAMETERS = {
    "logistic_regression": {"C": 1.0, "max_iter": 500, "solver": "lbfgs"},
    "knn_distance_weighted": {"n_neighbors": 25},
    "bagging": {"n_estimators": 40, "max_samples": 0.7, "max_depth": 14, "min_samples_leaf": 3},
}

# Fixed, non-searched KNN mechanics: genuine inverse-distance vote weighting,
# Euclidean distance on standardized features, exact brute-force neighbours.
KNN_FIXED_PARAMETERS = {"weights": "distance", "p": 2, "metric": "minkowski", "algorithm": "brute"}

# Predeclared Bayesian search spaces per family (TRAIN grouped CV only).
PHASE_SEARCH_SPACES = {
    "logistic_regression": {"C": ("log", 0.001, 100.0)},
    "knn_distance_weighted": {"n_neighbors": ("logint", 5, 75)},
    "bagging": {
        "n_estimators": ("int", 30, 80),
        "max_samples": ("float", 0.5, 0.9),
        "max_depth": ("int", 8, 20),
        "min_samples_leaf": ("logint", 2, 10),
    },
}

# Predeclared KNN reference-bank budget (rows per fit), never tuned by results.
KNN_REFERENCE_BANK_BUDGET = 60_000
KNN_BANK_SEED_STRIDE = 7919
KNN_FINAL_FIT_INDEX = 3


def make_phase_estimator(name: str, parameters: dict, seed: int, n_jobs: int):
    """Build a phase-three estimator. Returns pure sklearn objects only."""
    if name == "logistic_regression":
        classifier = LogisticRegression(**dict(parameters), random_state=seed)
        return Pipeline([("scale", StandardScaler()), ("classifier", classifier)])
    if name == "knn_distance_weighted":
        fixed = dict(KNN_FIXED_PARAMETERS)
        classifier = KNeighborsClassifier(n_neighbors=int(parameters["n_neighbors"]), n_jobs=n_jobs, **fixed)
        return Pipeline([("scale", StandardScaler()), ("classifier", classifier)])
    if name == "bagging":
        tree = DecisionTreeClassifier(
            max_depth=int(parameters["max_depth"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            random_state=seed,
        )
        classifier = BaggingClassifier(
            estimator=tree,
            n_estimators=int(parameters["n_estimators"]),
            max_samples=float(parameters["max_samples"]),
            bootstrap=True,
            random_state=seed,
            n_jobs=n_jobs,
        )
        return classifier
    raise ValueError(f"unknown phase-three family {name!r}")


def phase_supports_sample_weight(name: str) -> bool:
    """KNeighborsClassifier.fit accepts no sample_weight; the bank policy substitutes."""
    return name != "knn_distance_weighted"


def phase_preprocessing_description(name: str) -> str:
    if name in ("logistic_regression", "knn_distance_weighted"):
        return "fold-local StandardScaler inside each fit, applied before the classifier"
    return "No learned preprocessing; deterministic numerical features"


def knn_reference_bank(y: np.ndarray, weights: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Deterministic class-stratified, weight-proportional subsample of TRAIN rows.

    - Budget is capped at the available rows and preserves the class prior via
      proportional allocation (class 0 absorbs the rounding remainder).
    - Within a class, rows are drawn WITHOUT replacement with probability
      proportional to the saved training weights (equal observation mass,
      equal trip mass), approximating the weighting KNN cannot take in fit().
    - Deterministic per fit through the explicit seed.
    """
    y = np.asarray(y)
    weights = np.asarray(weights, dtype=float)
    if len(y) != len(weights) or len(y) == 0:
        raise ValueError("bank inputs must be aligned and non-empty")
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("training weights must be positive and finite")
    budget = int(min(budget, len(y)))
    classes, counts = np.unique(y, return_counts=True)
    allocation = np.round(budget * counts / counts.sum()).astype(int)
    allocation[0] += budget - int(allocation.sum())
    rng = np.random.default_rng(seed)
    chosen = []
    for cls, take in zip(classes.tolist(), allocation.tolist()):
        idx = np.flatnonzero(y == cls)
        take = int(min(take, len(idx)))
        if take >= len(idx):
            chosen.append(idx)
            continue
        probability = weights[idx] / weights[idx].sum()
        chosen.append(rng.choice(idx, size=take, replace=False, p=probability))
    bank = np.sort(np.concatenate(chosen)).astype(np.int64)
    if len(np.unique(bank)) != len(bank):
        raise AssertionError("bank sampling produced duplicate rows")
    return bank


def bank_seed(seed: int, fit_index: int) -> int:
    """Deterministic per-fit bank seed; fold fits use 0..2, final all80 uses 3."""
    return int(seed + KNN_BANK_SEED_STRIDE * int(fit_index))
