"""Bounded TRAIN-only logistic-regression/GaussianNB accuracy experiments.

The shared accuracy_common module owns dataset loading, grouped folds, sample
weights, probability validation, and CV metrics.  This worker owns only the
preregistered menus and fold-local estimator construction; importing it neither
loads datasets nor fits a classifier.  Row probabilities belong under the private
linear_bayes directory, never in the aggregate output directory.
"""
from __future__ import annotations

import os

# Clamp, rather than setdefault: inherited BLAS settings must not hog shared CPUs.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_OUTPUT = ROOT / "experiments/accuracy_improvement/linear_bayes"
PRIVATE_OUTPUT = ROOT / "experiments/accuracy_improvement/private/linear_bayes"
SEED = 20261118
MAX_CONFIGURATIONS_PER_FAMILY = 6


@dataclass(frozen=True)
class Candidate:
    family: str
    run_name: str
    profile: str
    parameters: dict[str, Any]


def candidate_menu(family: str | None = None) -> tuple[Candidate, ...]:
    """Return the fixed menu; no score/data-dependent candidate generation."""
    lr = {"max_iter": 500, "solver": "lbfgs"}
    reduced = {"preprocessing": "skew_decorrelate",
               "correlation_threshold": 0.90, "skew_threshold": 0.75}
    menu = (
        Candidate("logistic_regression", "lr_baseline38_c1", "baseline38",
                  {**lr, "C": 1.0, "preprocessing": "baseline"}),
        Candidate("logistic_regression", "lr_context_c01", "context",
                  {**lr, "C": 0.1, "preprocessing": "scale"}),
        Candidate("logistic_regression", "lr_context_c1", "context",
                  {**lr, "C": 1.0, "preprocessing": "scale"}),
        Candidate("logistic_regression", "lr_context_c10", "context",
                  {**lr, "C": 10.0, "preprocessing": "scale"}),
        Candidate("logistic_regression", "lr_compact_c1", "compact",
                  {**lr, "C": 1.0, "preprocessing": "scale"}),
        Candidate("logistic_regression", "lr_physics_c1", "physics",
                  {**lr, "C": 1.0, "preprocessing": "scale"}),
        Candidate("gaussian_nb", "gnb_baseline38_s1e7", "baseline38",
                  {"var_smoothing": 1e-7, "preprocessing": "baseline"}),
        Candidate("gaussian_nb", "gnb_compact_s001", "compact",
                  {**reduced, "var_smoothing": 1e-3}),
        Candidate("gaussian_nb", "gnb_compact_s01", "compact",
                  {**reduced, "var_smoothing": 1e-2}),
        Candidate("gaussian_nb", "gnb_compact_s1", "compact",
                  {**reduced, "var_smoothing": 1e-1}),
        Candidate("gaussian_nb", "gnb_context_s01", "context",
                  {**reduced, "var_smoothing": 1e-2}),
        Candidate("gaussian_nb", "gnb_physics_s01", "physics",
                  {**reduced, "var_smoothing": 1e-2}),
    )
    if family is None:
        return menu
    selected = tuple(item for item in menu if item.family == family)
    if not selected:
        raise ValueError(f"unsupported family: {family}")
    return selected


class Float64Features(TransformerMixin, BaseEstimator):
    """Promote before any learned step, avoiding float32 GNB normalization drift."""

    def fit(self, X, y=None):
        values = check_array(X, dtype=np.float64)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        values = check_array(X, dtype=np.float64)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("feature count differs from fitted data")
        return values


class FoldLocalSkewDecorrelator(TransformerMixin, BaseEstimator):
    """Unsupervised fold-local skew transform, constant removal and redundancy cut.

    The absolute training skew selects signed-log columns.  Their median and IQR
    are learned from training rows only, with standard-deviation/one fallbacks for
    degenerate IQR.  A greedy correlation filter keeps the first representative
    of each highly correlated pair, preserving profile order.  It never sees y.
    Final scaling lives in the following Pipeline step, before GNB smoothing.
    """

    def __init__(self, correlation_threshold=0.90, skew_threshold=0.75,
                 min_variance=1e-12):
        self.correlation_threshold = correlation_threshold
        self.skew_threshold = skew_threshold
        self.min_variance = min_variance

    def fit(self, X, y=None):
        if not 0.0 < self.correlation_threshold <= 1.0:
            raise ValueError("correlation_threshold must be in (0,1]")
        if self.skew_threshold < 0.0 or self.min_variance <= 0.0:
            raise ValueError("skew_threshold must be nonnegative; variance positive")
        values = check_array(X, dtype=np.float64)
        self.n_features_in_ = values.shape[1]
        self.center_ = np.median(values, axis=0)
        std = np.std(values, axis=0, dtype=np.float64)
        q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
        self.scale_ = np.where(q75 - q25 > self.min_variance, q75 - q25,
                               np.where(std > self.min_variance, std, 1.0))
        normalized = (values - np.mean(values, axis=0)) / np.where(std > 0.0, std, 1.0)
        self.training_skew_ = np.mean(normalized ** 3, axis=0)
        self.skewed_ = np.abs(self.training_skew_) > self.skew_threshold
        transformed = self._skew_transform(values)
        self.transformed_mean_ = np.mean(transformed, axis=0)
        self.transformed_std_ = np.std(transformed, axis=0, dtype=np.float64)
        active = np.flatnonzero(self.transformed_std_ ** 2 > self.min_variance)
        if not len(active):
            raise ValueError("cannot fit GaussianNB reduction to all-constant features")
        z = ((transformed[:, active] - self.transformed_mean_[active])
             / self.transformed_std_[active])
        correlation = np.clip((z.T @ z) / len(z), -1.0, 1.0)
        chosen: list[int] = []
        for index in range(len(active)):
            if not chosen or np.all(np.abs(correlation[index, chosen])
                                    < self.correlation_threshold):
                chosen.append(index)
        self.selected_indices_ = active[np.asarray(chosen, dtype=np.int64)]
        self.n_features_out_ = len(self.selected_indices_)
        return self

    def _skew_transform(self, values):
        result = values.copy()
        z = ((result[:, self.skewed_] - self.center_[self.skewed_])
             / self.scale_[self.skewed_])
        result[:, self.skewed_] = np.sign(z) * np.log1p(np.abs(z))
        return result

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        values = check_array(X, dtype=np.float64)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("feature count differs from fitted data")
        return self._skew_transform(values)[:, self.selected_indices_]


def make_estimator(family: str, parameters: dict[str, Any], config=None) -> Pipeline:
    """Build an unfitted sklearn Pipeline compatible with classifier__sample_weight.

    The optional config argument permits common-protocol factory registration;
    only its seed is consumed.  Parallelism is always one.  No calibration or
    relabeling is performed.  All preprocessing is newly fitted in every fold.
    """
    options = dict(parameters)
    preprocessing = options.pop("preprocessing", "scale")
    correlation_threshold = options.pop("correlation_threshold", 0.90)
    skew_threshold = options.pop("skew_threshold", 0.75)
    if preprocessing not in {"baseline", "scale", "skew_decorrelate"}:
        raise ValueError(f"unknown preprocessing: {preprocessing}")
    steps: list[tuple[str, Any]] = [("float64", Float64Features())]
    if preprocessing == "skew_decorrelate":
        if family != "gaussian_nb":
            raise ValueError("skew_decorrelate is preregistered only for GaussianNB")
        steps.append(("reduce", FoldLocalSkewDecorrelator(
            correlation_threshold=correlation_threshold, skew_threshold=skew_threshold)))
    steps.append(("scale", StandardScaler()))
    if family == "logistic_regression":
        seed = getattr(config, "seed", SEED)
        # sklearn defaults to L2 for lbfgs; forbid an accidental change of family.
        if options.get("solver", "lbfgs") != "lbfgs" or options.get("penalty", "l2") != "l2":
            raise ValueError("menu permits only L2/lbfgs logistic regression")
        if options.pop("n_jobs", 1) != 1:
            raise ValueError("linear_bayes must use n_jobs=1")
        classifier = LogisticRegression(**options, random_state=seed, n_jobs=1)
    elif family == "gaussian_nb":
        classifier = GaussianNB(**options)
    else:
        raise ValueError(f"unsupported family: {family}")
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def validate_binary_probabilities(probabilities, n_rows: int | None = None) -> np.ndarray:
    """Strict checks before any clipping; do not repair or renormalize bad output."""
    values = np.asarray(probabilities)
    if values.dtype != np.dtype(np.float64):
        raise ValueError("probabilities must be float64")
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("expected an n-by-2 binary probability matrix")
    if n_rows is not None and len(values) != n_rows:
        raise ValueError("probability row count differs from features")
    if (not np.isfinite(values).all() or np.any(values < 0.0)
            or np.any(values > 1.0)
            or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-10)):
        raise ValueError("invalid finite, bounded, sum-to-one binary probabilities")
    return values


def screen_order(record: dict[str, Any]) -> tuple:
    """Shared frozen ordering; records must already contain like-for-like metrics."""
    metrics = record["metrics"]
    return (-float(metrics["baseline_clip_trip_macro_top1_accuracy"]),
            -float(metrics["baseline_clip_top1_accuracy"]),
            -float(metrics["macro_f1"]), float(metrics["log_loss"]), record["run_name"])


def menu_manifest() -> dict[str, Any]:
    return {"families": {family: [asdict(item) for item in candidate_menu(family)]
                         for family in ("logistic_regression", "gaussian_nb")},
            "max_configurations_per_family": MAX_CONFIGURATIONS_PER_FAMILY,
            "training_protocol": "shared accuracy_common only after explicit handoff",
            "promotion": "screen fold0; top2/family; reuse fold0; fit remaining folds1,2",
            "row_probabilities": str(PRIVATE_OUTPUT.relative_to(ROOT)),
            "metrics": "baseline clipping and raw top1/tie audit; no calibration claims"}


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-menu", action="store_true",
                        help="Print the fixed menu without loading data or fitting")
    args = parser.parse_args(argv)
    if args.show_menu:
        print(json.dumps(menu_manifest(), indent=2, allow_nan=False))
        return 0
    parser.error("Training is unavailable until the lead hands off accuracy_common")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
