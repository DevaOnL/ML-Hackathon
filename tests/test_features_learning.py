"""Feature ordering, proxy-label honesty and trained emission wrapper tests."""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from mapmatch.candidates import CandidateIndex, Candidates
from mapmatch.features import (FEATURE_NAMES, FEATURE_VERSION, PROXY_LABEL_POLICY,
                               CandidateFeatureExtractor, features_for_candidates,
                               reference_observation_edges, reference_observation_labels)
from mapmatch.learning import (EmissionModel, candidate_classification_metrics,
                               candidate_probabilities, load_emission_model, load_model,
                               predict_positive_proba)
from tests.test_core import metric_graph, metric_track


class FeatureExtractionTests(unittest.TestCase):
    def setUp(self):
        self.graph = metric_graph([(0, 0), (100, 0), (0, 30), (100, 30), (50, 0)],
                                  [(0, 1), (1, 0), (2, 3), (3, 2), (4, 4), (0, 1)])
        self.track = metric_track([(20, 2), (40, -1), (60, 1), (80, 0)], [1, 2, 3, 10])
        self.xy = self.graph.project(self.track[:, :2])
        self.index = CandidateIndex(self.graph)
        self.candidates = [self.index.query(point, 50, 10) for point in self.xy]

    def test_schema_shape_values_and_candidate_permutation(self):
        extractor = CandidateFeatureExtractor(self.graph)
        candidates = self.candidates[1]
        features = extractor.features_for_candidates(self.track, self.xy, 1, candidates)
        self.assertEqual(features.shape, (len(candidates), len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features[:, FEATURE_NAMES.index("distance_m")], candidates.distances, rtol=1e-6)
        np.testing.assert_allclose(features[:, FEATURE_NAMES.index("fraction")], candidates.fractions, rtol=1e-6)
        order = np.arange(len(candidates) - 1, -1, -1)
        shuffled = Candidates(candidates.edges[order], candidates.fractions[order], candidates.distances[order])
        permuted = extractor.features_for_candidates(self.track, self.xy, 1, shuffled)
        np.testing.assert_allclose(permuted, features[order], rtol=1e-6, atol=1e-6)
        self.assertEqual(len(set(FEATURE_NAMES)), len(FEATURE_NAMES))

    def test_batched_offsets_preserve_original_observation_order(self):
        extractor = CandidateFeatureExtractor(self.graph)
        chosen = [3, 0, 2]
        candidates = [self.candidates[i] for i in chosen]
        features, offsets = extractor.transform(self.track, chosen, candidates)
        expected_offsets = [0]
        for candidate in candidates:
            expected_offsets.append(expected_offsets[-1] + len(candidate))
        np.testing.assert_array_equal(offsets, expected_offsets)
        for row, index in enumerate(chosen):
            direct = features_for_candidates(self.graph, self.track, self.xy, index, candidates[row])
            np.testing.assert_array_equal(features[offsets[row]:offsets[row + 1]], direct)

    def test_row_id_renumbering_does_not_leak_into_feature_values(self):
        permutation = np.array([4, 3, 5, 1, 2, 0])
        inverse = np.argsort(permutation)
        changed = type(self.graph)(self.graph.nodes, self.graph.arcs[permutation])
        original = self.candidates[2]
        relabeled = Candidates(inverse[original.edges], original.fractions, original.distances)
        first = features_for_candidates(self.graph, self.track, self.xy, 2, original)
        second = features_for_candidates(changed, self.track, self.xy, 2, relabeled)
        np.testing.assert_allclose(first, second, atol=1e-6, rtol=1e-6)
        self.assertFalse(any(name in FEATURE_NAMES for name in ("edge_id", "trip_id", "reference_edge")))

    def test_inference_features_do_not_call_reference_label_functions(self):
        with mock.patch("mapmatch.features.reference_observation_labels", side_effect=AssertionError("reference leakage")), \
                mock.patch("mapmatch.features.reference_observation_edges", side_effect=AssertionError("reference leakage")), \
                mock.patch("mapmatch.data.load_route", side_effect=AssertionError("reference leakage")):
            features, offsets = CandidateFeatureExtractor(self.graph).transform(
                self.track, np.arange(len(self.track)), self.candidates)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(offsets[-1], len(features))

    def test_stationary_zero_length_and_empty_candidates_remain_finite(self):
        track = metric_track([(50, 0)] * 3)
        xy = self.graph.project(track[:, :2])
        candidates = [self.index.query(point, 50, 10) for point in xy]
        features, _ = CandidateFeatureExtractor(self.graph).transform(track, [0, 1, 2], candidates)
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue((features[:, FEATURE_NAMES.index("low_motion")] == 1).all())
        self.assertTrue((features[:, FEATURE_NAMES.index("edge_is_zero_length")] == 1).any())
        empty = Candidates(np.array([], dtype=int), np.array([]), np.array([]))
        features, offsets = CandidateFeatureExtractor(self.graph).transform(track, [0, 2], [empty, empty])
        self.assertEqual(features.shape, (0, len(FEATURE_NAMES)))
        np.testing.assert_array_equal(offsets, [0, 0, 0])

    def test_gap_does_not_create_heading_across_dropout(self):
        track = metric_track([(0, 0), (50, 0), (100, 0)], [1, 100, 200])
        xy = self.graph.project(track[:, :2])
        candidate = self.index.query(xy[1], 10, 10)
        features = features_for_candidates(self.graph, track, xy, 1, candidate)
        for name in ("heading_reliability_short", "heading_reliability_medium", "heading_reliability_long"):
            np.testing.assert_array_equal(features[:, FEATURE_NAMES.index(name)], 0)

    def test_transform_rejects_invalid_observation_indices(self):
        extractor = CandidateFeatureExtractor(self.graph)
        for indices, candidates in (([0], []), ([-1], [self.candidates[0]]),
                                    ([4], [self.candidates[0]]), ([0.5], [self.candidates[0]])):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                extractor.transform(self.track, indices, candidates)

    def test_proxy_labels_are_reference_constrained_but_candidate_independent(self):
        # The reference is 30 m away and intentionally excluded by candidate radius.
        graph = metric_graph([(0, 0), (100, 0), (0, 30), (100, 30)], [(0, 1), (2, 3)])
        track = metric_track([(20, 0), (80, 0)])
        xy = graph.project(track[:, :2])
        self.assertEqual(CandidateIndex(graph).query(xy[0], 5, 10).edges.tolist(), [0])
        labels = reference_observation_labels(graph, track, [1])
        np.testing.assert_array_equal(labels.edges, [1, 1])
        np.testing.assert_allclose(labels.distances, 30, atol=1e-7)
        self.assertFalse(PROXY_LABEL_POLICY["is_ground_truth"])
        self.assertIn("No per-observation labels", PROXY_LABEL_POLICY["limitations"])

    def test_proxy_direction_duplicate_canonicalization_and_ambiguity(self):
        forward = reference_observation_edges(self.graph, self.track, [5, 1])
        np.testing.assert_array_equal(forward, [0] * len(self.track))
        stopped = metric_track([(50, 0)])
        labels = reference_observation_labels(self.graph, stopped, [0, 1])
        self.assertEqual(labels.edges.tolist(), [0])
        self.assertEqual(labels.ambiguous.tolist(), [True])
        np.testing.assert_array_equal(reference_observation_edges(self.graph, self.track, [0], [3, 1]), [0, 0])
        for route in ([], [-1], [6], [0.1]):
            with self.subTest(route=route), self.assertRaises(ValueError):
                reference_observation_edges(self.graph, self.track, route)


class CandidateClassificationMetricTests(unittest.TestCase):
    def test_candidate_misses_count_in_unconditional_top1_denominator(self):
        # First observation correct, second has no reference candidate, third wrong, fourth no candidates.
        metrics = candidate_classification_metrics(
            [1, 0, 0, 0, 0, 1], [0.9, 0.1, 0.8, 0.2, 0.9, 0.1],
            [0, 2, 4, 6, 6], observation_trip_ids=["A", "A", "A", "B"])
        self.assertEqual(metrics["observations"], 4)
        self.assertEqual(metrics["candidate_present_observations"], 2)
        self.assertEqual(metrics["candidate_miss_count"], 2)
        self.assertAlmostEqual(metrics["candidate_miss_rate"], 0.5)
        self.assertAlmostEqual(metrics["top1_accuracy"], 0.25)
        self.assertAlmostEqual(metrics["top1_accuracy_given_candidate"], 0.5)
        self.assertAlmostEqual(metrics["trip_macro_top1_accuracy"], 1 / 6)
        self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 6)
        self.assertIn("proxy", metrics["metric_target"])

    def test_binary_logloss_uses_raw_not_within_observation_normalized_probabilities(self):
        metrics = candidate_classification_metrics([1, 0], [0.8, 0.4], [0, 2])
        expected = -(math.log(0.8) + math.log(0.6)) / 2
        self.assertAlmostEqual(metrics["log_loss"], expected)
        self.assertAlmostEqual(metrics["top1_accuracy"], 1)

    def test_binary_macro_f1_keeps_positive_class_when_subgroup_has_no_positives(self):
        metrics = candidate_classification_metrics([0, 0], [0.1, 0.2], [0, 2])
        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["f1"], 0)
        self.assertEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["candidate_miss_count"], 1)
        self.assertEqual(metrics["top1_accuracy"], 0)

    def test_fractional_offsets_and_nonbinary_targets_cannot_be_silently_truncated(self):
        for target in ([0.5], [2], [256], [math.nan], ["1"]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                candidate_classification_metrics(target, [0.5], [0, 1])
        with self.assertRaises(ValueError):
            candidate_classification_metrics([1], [0.5], [0, 0.5, 1])

    def test_empty_inputs_and_invalid_probabilities(self):
        empty = candidate_classification_metrics([], [], [0])
        self.assertEqual(empty["observations"], 0)
        self.assertIsNone(empty["top1_accuracy"])
        self.assertIsNone(empty["log_loss"])
        for probabilities in ([math.nan], [-0.1], [1.1], [math.inf]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                candidate_classification_metrics([1], probabilities, [0, 1])
        for offsets in ([1, 1], [0, 2], [0, 1, 0]):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                candidate_classification_metrics([1], [0.5], offsets)


class LearnedArtifactTests(unittest.TestCase):
    @staticmethod
    def fitted_estimator():
        rng = np.random.default_rng(76)
        X = rng.normal(size=(30, len(FEATURE_NAMES)))
        y = np.array([0, 1] * 15)
        return LogisticRegression(max_iter=300, random_state=19).fit(X, y)

    def test_positive_probability_uses_class_identity_not_column_position(self):
        class ReversedColumns:
            classes_ = np.array([1, 0])
            def predict_proba(self, X):
                return np.tile([0.8, 0.2], (len(X), 1))
        np.testing.assert_allclose(predict_positive_proba(ReversedColumns(), np.zeros((3, 2))), 0.8)
        class Nonfinite(ReversedColumns):
            def predict_proba(self, X):
                return np.tile([math.nan, 0.2], (len(X), 1))
        with self.assertRaises(ValueError):
            predict_positive_proba(Nonfinite(), np.zeros((2, 2)))

    def test_probability_domain_and_binary_classes_are_strict(self):
        for classes, probabilities in (([0, 1], [0.2, 0.2]), ([0, 1], [-0.1, 1.1]),
                                        ([0, 1], [math.inf, 0]), ([1], [1]),
                                        ([0, 1, 2], [0.1, 0.8, 0.1]), ([0, 2], [0.5, 0.5])):
            class InvalidEstimator:
                classes_ = np.asarray(classes)
                def predict_proba(self, X):
                    return np.tile(probabilities, (len(X), 1))
            with self.subTest(classes=classes, probabilities=probabilities), self.assertRaises(ValueError):
                predict_positive_proba(InvalidEstimator(), np.zeros((2, len(FEATURE_NAMES))))
        estimator = self.fitted_estimator()
        with self.assertRaises(ValueError):
            EmissionModel(estimator, {}, feature_names=FEATURE_NAMES[::-1])
        with self.assertRaises(ValueError):
            EmissionModel(object(), {})

    def test_functional_dict_api_cannot_bypass_schema_validation(self):
        estimator = self.fitted_estimator()
        graph = metric_graph([(0, 0), (100, 0)], [(0, 1)])
        track = metric_track([(20, 0)])
        candidate = CandidateIndex(graph).query(graph.project(track[:, :2])[0])
        for bundle in ({"estimator": estimator}, {"estimator": estimator, "feature_names": FEATURE_NAMES[::-1],
                                                 "feature_version": FEATURE_VERSION}):
            with self.subTest(bundle_keys=list(bundle)), self.assertRaises(ValueError):
                candidate_probabilities(bundle, graph, track, [0], [candidate])

    def test_scalar_batch_and_functional_wrappers_match(self):
        graph = metric_graph([(0, 0), (100, 0)], [(0, 1), (1, 0)])
        track = metric_track([(20, 1), (50, 0), (80, 2)])
        xy = graph.project(track[:, :2])
        candidates = [CandidateIndex(graph).query(point, 10, 10) for point in xy]
        estimator = self.fitted_estimator()
        model = EmissionModel(estimator, {"name": "synthetic"})
        scalar = [model.score_candidates(graph, track, xy, i, candidate) for i, candidate in enumerate(candidates)]
        batch = model.score_candidates_batch(graph, track, [0, 1, 2], candidates)
        functional = candidate_probabilities(estimator, graph, track, [0, 1, 2], candidates)
        for first, second, third in zip(scalar, batch, functional):
            np.testing.assert_allclose(first, second)
            np.testing.assert_allclose(second, third)
            self.assertAlmostEqual(float(first.sum()), 1)
            self.assertTrue(np.isfinite(first).all())
            self.assertTrue((first > 0).all())

    def test_artifact_roundtrip_and_schema_rejection(self):
        estimator = self.fitted_estimator()
        bundle = {"estimator": estimator, "feature_names": FEATURE_NAMES,
                  "feature_version": FEATURE_VERSION, "metadata": {"name": "independent synthetic fixture"}}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trusted model.joblib"
            joblib.dump(bundle, path)
            loaded = load_emission_model(path)
            self.assertEqual(loaded.metadata, bundle["metadata"])
            self.assertEqual(tuple(loaded.feature_names), FEATURE_NAMES)
            features = np.zeros((3, len(FEATURE_NAMES)))
            np.testing.assert_allclose(loaded.predict_proba(features), estimator.predict_proba(features))
            for changed in ({}, {**bundle, "feature_names": FEATURE_NAMES[::-1]},
                            {**bundle, "feature_version": "not-compatible"}):
                joblib.dump(changed, path)
                with self.subTest(keys=list(changed)), self.assertRaises(ValueError):
                    load_model(path)


if __name__ == "__main__":
    unittest.main()
