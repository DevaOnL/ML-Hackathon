"""Synthetic-only exact KNN checks; never load any real data partition."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.accuracy_knn import (
    inverse_distance_k_votes, make_knn_estimator, make_knn_pipeline, predict_k_grid,
)


class ExactInverseDistanceKNNTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261118)
        self.X = rng.normal(size=(170, 6))
        self.y = (self.X[:, 0] + .4 * self.X[:, 1] > .15).astype(int)
        self.query = rng.normal(size=(29, 6))

    def test_constructs_only_genuine_bounded_sklearn_pipeline(self):
        estimator = make_knn_pipeline(n_neighbors=15, n_jobs=2)
        self.assertIs(type(estimator), Pipeline)
        self.assertIs(type(estimator.named_steps["scale"]), StandardScaler)
        classifier = estimator.named_steps["classifier"]
        self.assertIs(type(classifier), KNeighborsClassifier)
        self.assertEqual(classifier.weights, "distance")
        self.assertEqual(classifier.p, 2)
        self.assertEqual(classifier.n_jobs, 2)
        self.assertEqual(classifier.algorithm, "kd_tree")

    def test_shared_factory_contract_preserves_parameters_and_rejects_overbudget(self):
        parameters = {"n_neighbors": 15, "algorithm": "ball_tree", "p": 1, "leaf_size": 30}
        before = parameters.copy()
        estimator = make_knn_estimator(parameters, 20261118, 2)
        self.assertEqual(parameters, before)
        self.assertEqual(estimator.named_steps["classifier"].n_neighbors, 15)
        self.assertEqual(estimator.named_steps["classifier"].algorithm, "ball_tree")
        self.assertEqual(estimator.named_steps["classifier"].p, 1)
        with self.assertRaises(ValueError):
            make_knn_estimator(parameters, 20261118, 3)

    def test_random_no_tie_predictions_match_independent_sklearn_at_every_k(self):
        for algorithm, p in (("kd_tree", 2), ("ball_tree", 2), ("kd_tree", 1), ("brute", 2)):
            with self.subTest(algorithm=algorithm, p=p):
                estimator = make_knn_pipeline(n_neighbors=35, algorithm=algorithm, p=p).fit(self.X, self.y)
                actual = predict_k_grid(estimator, self.query, (5, 15, 35), batch_size=11)
                self.assertEqual(actual.query_calls, 3)
                self.assertEqual(actual.cutoff_tie_rows, {5: 0, 15: 0})
                for k, probability in actual.probabilities.items():
                    independent = make_knn_pipeline(n_neighbors=k, algorithm=algorithm, p=p).fit(self.X, self.y)
                    expected = independent.predict_proba(self.query)[:, 1]
                    np.testing.assert_allclose(probability, expected, rtol=2e-14, atol=2e-15)
                self.assertEqual(estimator.named_steps["classifier"].n_neighbors, 35)

    def test_query_once_per_batch_not_once_per_k(self):
        estimator = make_knn_pipeline().fit(self.X, self.y)
        classifier = estimator.named_steps["classifier"]
        with mock.patch.object(classifier, "kneighbors", wraps=classifier.kneighbors) as query:
            prediction = predict_k_grid(estimator, self.query, (5, 15, 35), batch_size=10)
        self.assertEqual(query.call_count, 3)
        self.assertEqual(prediction.query_calls, 3)
        self.assertTrue(all(call.kwargs["n_neighbors"] == 35 for call in query.call_args_list))

    def test_distance_zero_is_exact_sklearn_rule_and_prefix_local(self):
        distances = np.array([[0., 0., 1., 2.], [1., 2., 4., 8.]])
        indices = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
        labels = np.array([1, 0, 1, 0])
        actual = inverse_distance_k_votes(distances, indices, labels, (1, 2, 3, 4))
        np.testing.assert_array_equal(actual[1], [1., 1.])
        np.testing.assert_allclose(actual[2], [.5, 2 / 3], rtol=0, atol=1e-16)
        np.testing.assert_allclose(actual[3], [.5, 5 / 7], rtol=0, atol=1e-16)
        np.testing.assert_allclose(actual[4], [.5, 2 / 3], rtol=0, atol=1e-16)

    def test_duplicate_exact_query_matches_sklearn_and_nonzero_votes_do_not_leak(self):
        X = self.X.copy()
        X[0] = X[1] = np.zeros(X.shape[1])
        y = self.y.copy()
        y[0], y[1] = 0, 1
        query = np.array([np.zeros(X.shape[1]), X[4]])
        estimator = make_knn_pipeline(n_neighbors=15).fit(X, y)
        actual = predict_k_grid(estimator, query, (3, 5, 15)).probabilities
        for k, probability in actual.items():
            expected = make_knn_pipeline(n_neighbors=k).fit(X, y).predict_proba(query)[:, 1]
            np.testing.assert_allclose(probability, expected, rtol=0, atol=2e-15)
            self.assertEqual(probability[0], .5)
            self.assertEqual(probability[1], y[4])

    def test_all_duplicate_conflicting_classes_discloses_cutoff_ties(self):
        X = np.zeros((12, 2))
        y = np.array([0, 1] * 6)
        estimator = make_knn_pipeline(n_neighbors=9).fit(X, y)
        prediction = predict_k_grid(estimator, np.zeros((4, 2)), (3, 5, 9))
        self.assertEqual(prediction.cutoff_tie_rows, {3: 4, 5: 4})
        self.assertEqual(prediction.cutoff_tie_correction_calls, 2)
        self.assertEqual(prediction.query_calls, 3)
        # Only tied rows are re-queried; every k matches standalone sklearn.
        for k, probability in prediction.probabilities.items():
            independent = make_knn_pipeline(n_neighbors=k).fit(X, y)
            np.testing.assert_array_equal(probability, independent.predict_proba(np.zeros((4, 2)))[:, 1])
        self.assertFalse(prediction.maximum_k_boundary_ties_observed)
        with mock.patch.object(estimator.named_steps["classifier"], "kneighbors",
                               wraps=estimator.named_steps["classifier"].kneighbors) as query:
            unchecked = predict_k_grid(estimator, np.zeros((4, 2)), (3, 5, 9), exact_cutoff_ties=False)
        self.assertEqual(query.call_count, 1)
        self.assertEqual(unchecked.cutoff_tie_correction_calls, 0)

    def test_nonzero_equal_distance_cutoffs_match_each_exact_backend(self):
        X = np.vstack([np.eye(4), -np.eye(4)])
        y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        queries = np.zeros((3, 4))
        for algorithm, p in (("kd_tree", 2), ("ball_tree", 2), ("brute", 2), ("kd_tree", 1)):
            with self.subTest(algorithm=algorithm, p=p):
                estimator = make_knn_pipeline(n_neighbors=7, algorithm=algorithm, p=p).fit(X, y)
                prediction = predict_k_grid(estimator, queries, (1, 3, 7), batch_size=2)
                self.assertEqual(prediction.cutoff_tie_rows, {1: 3, 3: 3})
                self.assertEqual(prediction.cutoff_tie_correction_calls, 4)
                for k, probability in prediction.probabilities.items():
                    independent = make_knn_pipeline(n_neighbors=k, algorithm=algorithm, p=p).fit(X, y)
                    np.testing.assert_allclose(probability, independent.predict_proba(queries)[:, 1], rtol=0, atol=1e-15)

    def test_one_class_bank_has_correct_positive_probability(self):
        for label in (0, 1):
            with self.subTest(label=label):
                estimator = make_knn_pipeline().fit(self.X, np.full(len(self.X), label))
                actual = predict_k_grid(estimator, self.query, (5, 35))
                for probability in actual.probabilities.values():
                    np.testing.assert_array_equal(probability, np.full(len(self.query), label))

    def test_no_fit_or_query_is_made_for_empty_query(self):
        estimator = make_knn_pipeline().fit(self.X, self.y)
        with mock.patch.object(estimator.named_steps["classifier"], "kneighbors", side_effect=AssertionError("unexpected query")):
            actual = predict_k_grid(estimator, np.empty((0, self.X.shape[1])), (5, 35))
        self.assertEqual(actual.query_calls, 0)
        self.assertEqual(actual.probabilities[5].shape, (0,))

    def test_preprocessing_only_uses_passed_bank_not_queries(self):
        estimator = make_knn_pipeline().fit(self.X[:80], self.y[:80])
        before = estimator.named_steps["scale"].mean_.copy()
        predict_k_grid(estimator, self.query * 1e4, (5, 15))
        np.testing.assert_array_equal(estimator.named_steps["scale"].mean_, before)
        np.testing.assert_allclose(before, self.X[:80].mean(axis=0), rtol=0, atol=0)

    def test_rejects_invalid_resources_and_undeclared_distance_rules(self):
        for kwargs in ({"n_jobs": 3}, {"n_jobs": -1}, {"p": 3}, {"p": True},
                       {"algorithm": "auto"}, {"n_neighbors": 1.5}, {"leaf_size": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_knn_pipeline(**kwargs)

    def test_rejects_invalid_neighbour_arrays_and_k_values(self):
        d, i, y = np.array([[1., 2.]]), np.array([[0, 1]]), np.array([0, 1])
        for args in ((d, i, y, ()), (d, i, y, (1, 1)), (d, i, y, (0,)),
                     (d, i, y, (3,)), (d, i, y, (1.5,)),
                     (d[:, ::-1], i, y, (1,)),
                     (np.array([[0., np.nan]]), i, y, (1,)),
                     (np.array([[-1., 2.]]), i, y, (1,)),
                     (d, i + 2, y, (1,)), (d, i.astype(float), y, (1,)),
                     (d, i, np.array([0, 2]), (1,))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                inverse_distance_k_votes(*args)

    def test_rejects_wrong_query_features_and_non_distance_classifier(self):
        estimator = make_knn_pipeline().fit(self.X, self.y)
        for X in (np.ones((2, 3)), np.array([[np.nan] * 6])):
            with self.subTest(X=X), self.assertRaises(ValueError):
                predict_k_grid(estimator, X, (5,))
        estimator.set_params(classifier__weights="uniform")
        with self.assertRaises(ValueError):
            predict_k_grid(estimator, self.query, (5,))


if __name__ == "__main__":
    unittest.main()
