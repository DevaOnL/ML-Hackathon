"""Learned emissions must control decoding, never just decorate diagnostics."""
import math
import unittest

import numpy as np

from mapmatch.matcher import MapMatcher, MatchConfig
from tests.test_core import metric_graph, metric_track


class PreferenceModel:
    def __init__(self, preferred):
        self.preferred = preferred
        self.calls = []

    def score_candidates(self, graph, track, xy, index, candidates):
        self.calls.append((index, candidates.edges.copy()))
        return np.where(candidates.edges == self.preferred, 0.999, 0.001)


class BatchPreferenceModel(PreferenceModel):
    def __init__(self, preferred):
        super().__init__(preferred)
        self.batch_calls = []

    def score_candidates_batch(self, graph, track, indices, candidates):
        self.batch_calls.append(np.asarray(indices).copy())
        return [np.where(c.edges == self.preferred, 0.999, 0.001) for c in candidates]

    def score_candidates(self, *args):
        raise AssertionError("batch model should not use per-observation dispatch")


class EmissionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.graph = metric_graph([(0, 0), (100, 0)], [(0, 1), (1, 0)])
        self.track = metric_track([(50, 0), (50, 0), (50, 0)])

    def test_changing_classifier_changes_real_decoded_direction(self):
        forward = MapMatcher(self.graph, emission_model=PreferenceModel(0)).match(self.track)
        backward_model = PreferenceModel(1)
        backward = MapMatcher(self.graph, emission_model=backward_model).match(self.track)
        np.testing.assert_array_equal(forward.matched_edges, [0, 0, 0])
        np.testing.assert_array_equal(backward.matched_edges, [1, 1, 1])
        self.assertEqual(forward.route.tolist(), [0])
        self.assertEqual(backward.route.tolist(), [1])
        self.assertEqual([call[0] for call in backward_model.calls], [0, 1, 2])
        self.assertEqual(backward.diagnostics["classifier_only_edges"], [1, 1, 1])
        self.assertIn("learned", backward.diagnostics["emission_model"])
        self.assertFalse(backward.diagnostics["uses_reference_route"])

    def test_batch_and_scalar_paths_preserve_candidate_correspondence(self):
        scalar = MapMatcher(self.graph, emission_model=PreferenceModel(1)).match(self.track)
        model = BatchPreferenceModel(1)
        batch = MapMatcher(self.graph, emission_model=model).match(self.track)
        np.testing.assert_array_equal(batch.route, scalar.route)
        np.testing.assert_array_equal(batch.matched_edges, scalar.matched_edges)
        self.assertEqual(len(model.batch_calls), 1)
        np.testing.assert_array_equal(model.batch_calls[0], [0, 1, 2])

    def test_batch_receives_original_retained_indices_after_thinning(self):
        track = metric_track([(50, 0)] * 12)
        model = BatchPreferenceModel(1)
        config = MatchConfig(observation_distance_m=10, observation_interval=5)
        result = MapMatcher(self.graph, config, emission_model=model).match(track)
        np.testing.assert_array_equal(result.observation_indices, [0, 5, 10, 11])
        np.testing.assert_array_equal(model.batch_calls[0], result.observation_indices)
        self.assertEqual(result.matched_edges.tolist(), [1, 1, 1, 1])

    def test_unnormalized_positive_scores_and_zero_probability_are_supported(self):
        class RawModel:
            def score_candidates(self, graph, track, xy, index, candidates):
                return np.where(candidates.edges == 1, 8.0, 0.0)
        result = MapMatcher(self.graph, emission_model=RawModel()).match(self.track)
        self.assertEqual(result.route.tolist(), [1])
        self.assertTrue(math.isfinite(result.diagnostics["negative_log_likelihood_relative"]))

    def test_invalid_model_probability_vectors_fail_loudly(self):
        for vector in ([0, 0], [-1, 2], [math.nan, 1], [math.inf, 1], [1], [[0.5, 0.5]]):
            class BadModel:
                def score_candidates(self, *args):
                    return vector
            with self.subTest(vector=vector), self.assertRaisesRegex(ValueError, "probabilit"):
                MapMatcher(self.graph, emission_model=BadModel()).match(self.track)

    def test_wrong_batch_observation_count_is_rejected(self):
        class BadBatch:
            def score_candidates_batch(self, *args):
                return [np.array([0.5, 0.5])]
        with self.assertRaisesRegex(ValueError, "observation count"):
            MapMatcher(self.graph, emission_model=BadBatch()).match(self.track)


if __name__ == "__main__":
    unittest.main()
