"""Exhaustive small-state HMM verification independent of the Viterbi recurrence."""
from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from mapmatch.candidates import CandidateIndex
from mapmatch.data import validate_route
from mapmatch.matcher import MapMatcher, MatchConfig, MatchingError
from tests.test_core import independent_distances, metric_graph, metric_track


def path_cost(graph, config, xy, candidates, choices, distances, probabilities=None):
    cost = 0.0
    for index, state in enumerate(choices):
        current = candidates[index]
        cost += config.geometry_weight * 0.5 * (float(current.distances[state]) / config.sigma_m) ** 2
        if probabilities is not None:
            values = np.asarray(probabilities[index], dtype=float)
            probability = float(values[state] / values.sum())
            cost -= config.classifier_weight * math.log(max(probability, 1e-12))
        if index == 0:
            continue
        previous = candidates[index - 1]
        a, b = int(previous.edges[choices[index - 1]]), int(current.edges[state])
        fraction_a, fraction_b = float(previous.fractions[choices[index - 1]]), float(current.fractions[state])
        length_a, length_b = float(graph.edge_lengths[a]), float(graph.edge_lengths[b])
        road = ((1 - fraction_a) * length_a +
                distances[int(graph.heads[a])][int(graph.tails[b])] + fraction_b * length_b)
        if a == b and (fraction_b - fraction_a) * length_a >= -config.backward_tolerance_m:
            road = max((fraction_b - fraction_a) * length_a, 0.0)
        chord = float(np.linalg.norm(xy[index] - xy[index - 1]))
        if road > chord + config.routing_slack_m:
            return math.inf
        cost += abs(road - chord) / config.beta_m
    return cost


class ViterbiOracleTests(unittest.TestCase):
    def test_viterbi_matches_exhaustive_enumeration_for_geometry_and_classifier(self):
        rng = np.random.default_rng(2026)
        graph = metric_graph([(0, 0), (100, 0), (100, 100), (0, 100)],
                             [(0, 1), (1, 2), (2, 3), (3, 0),
                              (1, 0), (2, 1), (3, 2), (0, 3)])
        all_distances = [independent_distances(graph, node) for node in range(len(graph.nodes))]
        for with_model in (False, True):
            for case in range(8):
                samples = rng.uniform(10, 90, (4, 2))
                track = metric_track(samples, [1, 2, 9, 10])
                config = MatchConfig(candidate_radius_m=120, max_candidates=4,
                                     routing_slack_m=2000, backward_tolerance_m=5,
                                     sigma_m=12, beta_m=17)
                xy = graph.project(track[:, :2])
                candidates = [CandidateIndex(graph).query(point, 120, 4) for point in xy]
                probabilities = [rng.uniform(0.01, 1, len(candidate)) for candidate in candidates] if with_model else None

                class FixedBatchModel:
                    def score_candidates_batch(self, received_graph, received_track, indices, received):
                        for expected, actual in zip(candidates, received):
                            np.testing.assert_array_equal(actual.edges, expected.edges)
                        return probabilities

                result = MapMatcher(graph, config, FixedBatchModel() if with_model else None).match(track)
                selected = [int(np.flatnonzero(candidate.edges == edge)[0])
                            for candidate, edge in zip(candidates, result.matched_edges)]
                actual = path_cost(graph, config, xy, candidates, selected, all_distances, probabilities)
                expected = min(path_cost(graph, config, xy, candidates, choices, all_distances, probabilities)
                               for choices in itertools.product(*(range(len(candidate)) for candidate in candidates)))
                with self.subTest(with_model=with_model, case=case):
                    self.assertAlmostEqual(actual, expected, delta=1e-8)
                    self.assertAlmostEqual(result.diagnostics["negative_log_likelihood_relative"], expected, delta=1e-8)
                    self.assertTrue(validate_route(graph, result.route)["valid"])

    def test_global_path_avoids_nearest_point_switch_to_disconnected_parallel_road(self):
        graph = metric_graph([(0, 0), (100, 0), (200, 0),
                              (0, 20), (100, 20), (200, 20)],
                             [(0, 1), (1, 2), (3, 4), (4, 5)])
        track = metric_track([(20, 0), (80, 15), (140, 0), (180, 0)])
        config = MatchConfig(candidate_radius_m=30, sigma_m=5)
        nearest_second = CandidateIndex(graph).query(graph.project(track[1:2, :2])[0], 30, 12)
        self.assertEqual(int(nearest_second.edges[0]), 2)
        result = MapMatcher(graph, config).match(track)
        self.assertEqual(result.route.tolist(), [0, 1])
        self.assertEqual(result.matched_edges.tolist(), [0, 0, 1, 1])

    def test_partial_arc_offsets_not_full_first_and_last_edge_lengths(self):
        graph = metric_graph([(0, 0), (100, 0), (200, 0)], [(0, 1), (1, 2)])
        track = metric_track([(90, 0), (110, 0)])
        result = MapMatcher(graph, MatchConfig(candidate_radius_m=1)).match(track)
        self.assertEqual(result.route.tolist(), [0, 1])
        np.testing.assert_allclose(result.matched_fractions, [0.9, 0.1], atol=1e-8)
        self.assertAlmostEqual(result.diagnostics["negative_log_likelihood_relative"], 0, delta=1e-8)

    def test_real_backwards_motion_requires_and_preserves_a_legal_directed_cycle(self):
        track = metric_track([(80, 0), (20, 0)])
        config = MatchConfig(candidate_radius_m=1, backward_tolerance_m=5, routing_slack_m=1000)
        dead_end = metric_graph([(0, 0), (100, 0)], [(0, 1)])
        with self.assertRaises(MatchingError):
            MapMatcher(dead_end, config).match(track)
        cycle = metric_graph([(0, 0), (100, 0), (50, 100)], [(0, 1), (1, 2), (2, 0)])
        result = MapMatcher(cycle, config).match(track)
        self.assertEqual(result.matched_edges.tolist(), [0, 0])
        self.assertEqual(result.route.tolist(), [0, 1, 2, 0])
        self.assertTrue(validate_route(cycle, result.route)["valid"])
        self.assertGreater(result.diagnostics["negative_log_likelihood_relative"], 0)


if __name__ == "__main__":
    unittest.main()
