"""Synthetic-only tests for the scoped linear/Bayes TRAIN accuracy worker."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.accuracy_linear_bayes import (
    FoldLocalSkewDecorrelator, Float64Features, candidate_menu, make_estimator,
    menu_manifest, screen_order, validate_binary_probabilities,
)


class FixedMenuTests(unittest.TestCase):
    def test_six_configs_per_family_and_no_mutable_shared_parameters(self):
        menu = candidate_menu()
        self.assertEqual(len(menu), 12)
        self.assertEqual(len({item.run_name for item in menu}), 12)
        for family in ("logistic_regression", "gaussian_nb"):
            self.assertEqual(len(candidate_menu(family)), 6)
            self.assertEqual(sum(item.profile == "baseline38"
                                 for item in candidate_menu(family)), 1)
        menu[0].parameters["C"] = 1234
        self.assertEqual(candidate_menu()[0].parameters["C"], 1.0)
        with self.assertRaises(ValueError):
            candidate_menu("unknown")

    def test_code_menu_matches_preregistration_without_reading_any_dataset(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "experiments/accuracy_improvement/linear_bayes/preregistered_menu.json"
        preregistered = json.loads(path.read_text(encoding="utf-8"))
        for family in ("logistic_regression", "gaussian_nb"):
            actual = [{"run_name": item.run_name, "profile": item.profile,
                       "parameters": item.parameters} for item in candidate_menu(family)]
            self.assertEqual(actual, preregistered["menus"][family])
        self.assertEqual(menu_manifest()["max_configurations_per_family"], 6)

    def test_every_constructor_is_unfitted_cloneable_fold_local_and_single_job(self):
        for item in candidate_menu():
            original = dict(item.parameters)
            model = make_estimator(item.family, item.parameters)
            self.assertIsInstance(model, Pipeline)
            self.assertIsInstance(model["scale"], StandardScaler)
            self.assertFalse(hasattr(model["scale"], "mean_"))
            self.assertFalse(hasattr(model["classifier"], "classes_"))
            self.assertEqual(item.parameters, original)
            clone(model)
            if item.family == "logistic_regression":
                self.assertIsInstance(model["classifier"], LogisticRegression)
                self.assertEqual(model["classifier"].n_jobs, 1)
                self.assertEqual(model["classifier"].solver, "lbfgs")
                self.assertNotIn("reduce", model.named_steps)
            else:
                self.assertIsInstance(model["classifier"], GaussianNB)
                self.assertEqual("reduce" in model.named_steps, item.profile != "baseline38")

    def test_rejects_out_of_scope_estimators_and_parallelism(self):
        for family, params in (("random_forest", {}),
                               ("logistic_regression", {"n_jobs": 2}),
                               ("logistic_regression", {"solver": "saga"}),
                               ("logistic_regression", {"penalty": "l1"}),
                               ("logistic_regression", {"preprocessing": "skew_decorrelate"}),
                               ("gaussian_nb", {"preprocessing": "global_scale"})):
            with self.assertRaises(ValueError):
                make_estimator(family, params)


class FoldLocalTransformerTests(unittest.TestCase):
    def test_float64_promotion_and_fitted_feature_count(self):
        transform = Float64Features()
        values = np.arange(20, dtype=np.float32).reshape(5, 4)
        with self.assertRaises(NotFittedError):
            transform.transform(values)
        promoted = transform.fit_transform(values)
        self.assertEqual(promoted.dtype, np.float64)
        np.testing.assert_array_equal(promoted, values)
        with self.assertRaises(ValueError):
            transform.transform(np.ones((2, 5)))

    def test_constant_duplicate_and_negative_correlations_drop_deterministically(self):
        rng = np.random.default_rng(812)
        a = rng.normal(size=1000)
        b = rng.normal(size=1000)
        X = np.column_stack((a, a * 3.0 + 10.0, -a, b, np.ones(len(a))))
        transform = FoldLocalSkewDecorrelator(skew_threshold=1e6).fit(X)
        np.testing.assert_array_equal(transform.selected_indices_, [0, 3])
        self.assertEqual(transform.n_features_out_, 2)
        np.testing.assert_array_equal(transform.transform(X), X[:, [0, 3]])

    def test_train_only_skew_centers_selection_and_inference_does_not_refit(self):
        rng = np.random.default_rng(64)
        X = np.column_stack((rng.exponential(size=512), rng.normal(size=512),
                             rng.integers(0, 2, size=512)))
        transform = FoldLocalSkewDecorrelator().fit(X, np.zeros(len(X)))
        flipped_labels = FoldLocalSkewDecorrelator().fit(X, np.ones(len(X)))
        np.testing.assert_array_equal(transform.selected_indices_, flipped_labels.selected_indices_)
        np.testing.assert_array_equal(transform.skewed_, flipped_labels.skewed_)
        self.assertTrue(transform.skewed_[0])
        self.assertFalse(transform.skewed_[1])
        np.testing.assert_allclose(transform.center_, np.median(X, axis=0), atol=0, rtol=0)
        snapshot = {name: value.copy() for name, value in vars(transform).items()
                    if isinstance(value, np.ndarray)}
        heldout = X * 1e6 + 1e8
        transformed = transform.transform(heldout)
        self.assertTrue(np.isfinite(transformed).all())
        for name, before in snapshot.items():
            np.testing.assert_array_equal(getattr(transform, name), before)
        self.assertGreater(np.linalg.norm(transform.center_ - np.median(heldout, axis=0)), 1e6)

    def test_scaling_and_reduction_are_fitted_only_on_training_rows(self):
        rng = np.random.default_rng(93)
        train = rng.exponential(size=(300, 5))
        y = (train[:, 0] + train[:, 1] > 2.0).astype(np.int8)
        heldout = rng.exponential(size=(50, 5)) + 10000.0
        item = candidate_menu("gaussian_nb")[2]
        model = make_estimator(item.family, item.parameters)
        model.fit(train.astype(np.float32), y, classifier__sample_weight=np.ones(len(train)))
        reduced = model["reduce"].transform(train.astype(np.float32).astype(np.float64))
        np.testing.assert_allclose(model["scale"].mean_, reduced.mean(axis=0), rtol=0, atol=1e-12)
        mean = model["scale"].mean_.copy()
        probabilities = model.predict_proba(heldout.astype(np.float32))
        validate_binary_probabilities(probabilities, len(heldout))
        np.testing.assert_array_equal(model["scale"].mean_, mean)
        self.assertGreater(model["classifier"].epsilon_, 0.0)
        self.assertLessEqual(model["classifier"].epsilon_, item.parameters["var_smoothing"] * 1.000001)

    def test_rejects_constants_nonfinite_bad_shape_and_bad_hyperparameters(self):
        for X in (np.ones((10, 4)), np.array([[0.0, np.nan], [1.0, 2.0]]),
                  np.ones(5)):
            with self.assertRaises(ValueError):
                FoldLocalSkewDecorrelator().fit(X)
        for params in ({"correlation_threshold": 0.0}, {"correlation_threshold": 1.1},
                       {"skew_threshold": -1.0}, {"min_variance": 0.0}):
            with self.assertRaises(ValueError):
                FoldLocalSkewDecorrelator(**params).fit(np.arange(40).reshape(10, 4))
        transform = FoldLocalSkewDecorrelator().fit(np.arange(40).reshape(10, 4))
        with self.assertRaises(ValueError):
            transform.transform(np.arange(50).reshape(10, 5))


class ProbabilityAndSelectionTests(unittest.TestCase):
    def test_strict_finite_bounds_sum_float64_no_silent_repairs(self):
        good = np.array([[0.25, 0.75], [0.0, 1.0]], dtype=np.float64)
        self.assertIs(validate_binary_probabilities(good, 2), good)
        bad = [good.astype(np.float32), [[0.1, 0.900000001]], [[-1e-12, 1.0]],
               [[0.0, 1.000000000001]], [[np.nan, 1.0]], [[np.inf, 0.0]],
               np.ones((3, 1)), np.ones(2)]
        for value in bad:
            with self.assertRaises(ValueError):
                validate_binary_probabilities(value)
        with self.assertRaises(ValueError):
            validate_binary_probabilities(good, 3)

    def test_float32_input_gnb_scores_are_strict_float64(self):
        rng = np.random.default_rng(72)
        X = rng.normal(size=(400, 38)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int8)
        for item in (candidate_menu("gaussian_nb")[0], candidate_menu("gaussian_nb")[1],
                     candidate_menu("logistic_regression")[0]):
            model = make_estimator(item.family, item.parameters)
            model.fit(X, y, classifier__sample_weight=np.ones(len(X)))
            probabilities = model.predict_proba(X)
            validate_binary_probabilities(probabilities, len(X))

    def test_frozen_primary_tie_breaks_do_not_select_on_raw_clipping_advantage(self):
        def record(name, primary, top1, f1, loss):
            return {"run_name": name, "metrics": {
                "baseline_clip_trip_macro_top1_accuracy": primary,
                "baseline_clip_top1_accuracy": top1, "macro_f1": f1, "log_loss": loss,
                "raw_top1_accuracy": 0.99 if name == "bad" else 0.10}}
        records = [record("bad", 0.6, 0.7, 0.9, 0.1),
                   record("z", 0.7, 0.7, 0.8, 0.2),
                   record("a", 0.7, 0.7, 0.8, 0.2)]
        self.assertEqual([item["run_name"] for item in sorted(records, key=screen_order)],
                         ["a", "z", "bad"])

    def test_strict_monotone_map_preserves_real_ranking_clipping_can_create_ties(self):
        raw = np.array([1e-20, 1e-10, 0.25, 0.5, 0.8])
        transformed = np.exp(raw)
        np.testing.assert_array_equal(np.argsort(raw), np.argsort(transformed))
        pair = np.array([1e-20, 1e-10])
        self.assertEqual(int(np.argmax(pair)), 1)
        self.assertEqual(int(np.argmax(np.clip(pair, 1e-7, 1.0 - 1e-7))), 0)


if __name__ == "__main__":
    unittest.main()
