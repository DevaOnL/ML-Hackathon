"""Small synthetic experiment design checks, never another full training run."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from sklearn.pipeline import Pipeline

from mapmatch.experiments import (CandidateDataset, ExperimentConfig, SEARCH_SPACES,
                                  _fit, build_group_folds, cross_validate_model,
                                  decode_parameters, default_parameters,
                                  deterministic_observation_indices, encode_parameters,
                                  expected_improvement, make_estimator)
from mapmatch.features import FEATURE_NAMES
from mapmatch.learning import candidate_classification_metrics, predict_positive_proba


def synthetic_dataset():
    rng = np.random.default_rng(908)
    trips = np.repeat(np.array([f"train-{i}" for i in range(6)]), [2, 3, 2, 4, 2, 3])
    lengths = np.array([2 + (i % 3) for i in range(len(trips))])
    offsets = np.r_[0, lengths.cumsum()]
    X = rng.normal(size=(int(offsets[-1]), len(FEATURE_NAMES)))
    y = np.zeros(len(X), dtype=np.int8)
    for observation, start in enumerate(offsets[:-1]):
        y[start + observation % lengths[observation]] = 1
    X[:, 0] = y * 3 + rng.normal(scale=0.1, size=len(y))
    return CandidateDataset(X=X, y=y, offsets=offsets, observation_trip_ids=trips,
                            observation_indices=np.arange(len(trips)),
                            reference_edges=np.zeros(len(trips), dtype=int),
                            candidate_edges=np.tile(np.arange(4), len(X))[:len(X)],
                            candidate_fractions=np.full(len(X), 0.5),
                            candidate_distances=np.ones(len(X)),
                            proxy_distances=np.ones(len(trips)), ambiguous=np.zeros(len(trips), bool),
                            expanded=np.zeros(len(trips), bool), coverage={})


class ExperimentDesignTests(unittest.TestCase):
    def test_trip_group_folds_keep_every_candidate_and_observation_together(self):
        dataset = synthetic_dataset()
        folds = build_group_folds(dataset, ExperimentConfig(folds=3))
        self.assertEqual(len(folds), 3)
        validation_seen = Counter()
        for fold in folds:
            train, validation = set(fold["train_trip_ids"]), set(fold["validation_trip_ids"])
            self.assertFalse(train & validation)
            self.assertEqual(train | validation, set(dataset.observation_trip_ids))
            row_trips = dataset.row_trip_ids
            self.assertEqual(set(row_trips[fold["train_rows"]]), train)
            self.assertEqual(set(row_trips[fold["validation_rows"]]), validation)
            self.assertFalse(set(fold["train_rows"]) & set(fold["validation_rows"]))
            self.assertEqual(len(fold["train_rows"]) + len(fold["validation_rows"]), len(dataset.y))
            for observation in fold["validation_observations"]:
                expected_rows = set(range(dataset.offsets[observation], dataset.offsets[observation + 1]))
                self.assertTrue(expected_rows <= set(fold["validation_rows"]))
            validation_seen.update(validation)
        self.assertEqual(validation_seen, Counter({trip: 1 for trip in dataset.observation_trip_ids}))

    def test_training_mass_is_balanced_by_trip_and_candidate_count(self):
        dataset = synthetic_dataset()
        weights = dataset.training_weights()
        trip_mass = [weights[dataset.row_trip_ids == trip].sum() for trip in np.unique(dataset.observation_trip_ids)]
        np.testing.assert_allclose(trip_mass, trip_mass[0])
        self.assertAlmostEqual(float(weights.mean()), 1)
        for trip in np.unique(dataset.observation_trip_ids):
            observations = np.flatnonzero(dataset.observation_trip_ids == trip)
            masses = [weights[dataset.offsets[i]:dataset.offsets[i + 1]].sum() for i in observations]
            np.testing.assert_allclose(masses, masses[0])

    def test_at_least_five_distinct_actual_model_families_and_no_random_inner_holdout(self):
        parameters = default_parameters()
        self.assertGreaterEqual(len(parameters), 5)
        types = set()
        for name, config in parameters.items():
            estimator = make_estimator(name, config, ExperimentConfig(n_jobs=1))
            classifier = estimator[-1] if isinstance(estimator, Pipeline) else estimator
            types.add(type(classifier))
            if name in ("logistic_regression", "gaussian_nb"):
                self.assertIsInstance(estimator, Pipeline)
                self.assertEqual(estimator.steps[0][0], "scale")
                self.assertFalse(hasattr(estimator[0], "mean_"))
            if name == "hist_gradient_boosting":
                self.assertFalse(estimator.early_stopping)
        self.assertGreaterEqual(len(types), 5)

    def test_preprocessing_fits_on_training_rows_not_validation_distribution(self):
        dataset = synthetic_dataset()
        fold = build_group_folds(dataset, ExperimentConfig())[0]
        train, validation = fold["train_rows"], fold["validation_rows"]
        features = dataset.X.copy()
        features[validation] += 10000
        parameters = default_parameters()["logistic_regression"]
        estimator = make_estimator("logistic_regression", parameters, ExperimentConfig())
        _fit(estimator, features[train], dataset.y[train], dataset.training_weights(train))
        np.testing.assert_allclose(estimator[0].mean_, features[train].mean(axis=0), atol=1e-9)
        self.assertGreater(float(np.linalg.norm(estimator[0].mean_ - features.mean(axis=0))), 100)
        probabilities = predict_positive_proba(estimator, features[validation])
        self.assertEqual(len(probabilities), len(validation))

    def test_cross_validation_uses_train_only_and_covers_each_out_of_fold_row(self):
        dataset = synthetic_dataset()
        folds = build_group_folds(dataset, ExperimentConfig())
        captured = []
        real_fit = _fit
        def traced_fit(estimator, X, y, weights):
            captured.append((X.copy(), y.copy()))
            return real_fit(estimator, X, y, weights)
        with tempfile.TemporaryDirectory() as root, \
                mock.patch("mapmatch.experiments._fit", side_effect=traced_fit), \
                mock.patch("mapmatch.experiments.log"):
            result = cross_validate_model("logistic_regression", default_parameters()["logistic_regression"],
                                          dataset, folds, ExperimentConfig(), Path(root), "synthetic",
                                          save_predictions=False)
        self.assertEqual(len(captured), 3)
        for (X, y), fold in zip(captured, folds):
            np.testing.assert_array_equal(X, dataset.X[fold["train_rows"]])
            np.testing.assert_array_equal(y, dataset.y[fold["train_rows"]])
        self.assertTrue(np.isfinite(result["objective"]))
        self.assertEqual(result["oof_metrics"]["candidate_rows"], len(dataset.y))
        self.assertEqual(result["oof_metrics"]["observations"], len(dataset.observation_trip_ids))

    def test_deterministic_sampling_includes_endpoints_and_no_duplicates(self):
        sampled = deterministic_observation_indices(1001, 350)
        self.assertEqual(len(sampled), 350)
        self.assertEqual(sampled[0], 0)
        self.assertEqual(sampled[-1], 1000)
        self.assertTrue((np.diff(sampled) > 0).all())
        np.testing.assert_array_equal(deterministic_observation_indices(4, 350), [0, 1, 2, 3])
        np.testing.assert_array_equal(deterministic_observation_indices(4, 0), [0, 1, 2, 3])
        np.testing.assert_array_equal(deterministic_observation_indices(1001, 350), sampled)

    def test_bayesian_parameter_transform_and_expected_improvement(self):
        for name, space in SEARCH_SPACES.items():
            base = default_parameters()[name]
            encoded = encode_parameters(base, space)
            self.assertTrue(((encoded >= 0) & (encoded <= 1)).all(), name)
            decoded = decode_parameters(encoded, base, space)
            for key, value in base.items():
                self.assertAlmostEqual(decoded[key], value, delta=1e-8)
        acquisition = expected_improvement(np.array([0.9, 0.1, 0.5]), np.array([0.01, 0.01, 0.0]), 0.5)
        self.assertTrue(np.isfinite(acquisition).all())
        self.assertTrue((acquisition >= 0).all())
        self.assertGreater(acquisition[0], acquisition[1])
        self.assertEqual(acquisition[2], 0)

    def test_optional_edge_macro_namespaces_graph_local_ids_by_trip(self):
        # Equal numeric ID0 in two different graphs is not the same class.
        # Unqualified pooling would give F1=2/3 for class0 and0 for class1 =>1/3.
        # Qualified classes A:0,B:0,B:1 have F1s1,2/3,0 =>5/9.
        metrics = candidate_classification_metrics([1, 1, 0], [0.9, 0.9, 0.9], [0, 1, 2, 3],
                                                   observation_trip_ids=["A", "B", "B"],
                                                   true_edges=[0, 0, 1], candidate_edges=[0, 0, 0])
        self.assertAlmostEqual(metrics["edge_macro_f1"], 5 / 9)
        self.assertIn("trip_id:", metrics["edge_macro_f1_class_namespace"])


if __name__ == "__main__":
    unittest.main()
