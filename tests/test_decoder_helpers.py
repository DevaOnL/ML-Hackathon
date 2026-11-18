"""Read-only decoder calibration helper checks; no dataset or artifact writes."""
from collections import defaultdict
import math
import unittest

import numpy as np

from scripts.decoder_experiments import (CALIBRATION_IDS, adjacent_collapse,
                                        plan, project_selected,
                                        reference_span_distance, sample_indices)
from backend.scoring import classification_metrics
from tests.test_core import metric_graph, metric_track


def reference_context(graph, route):
    route = np.asarray(route, dtype=np.int64)
    occurrence = defaultdict(list)
    for index, edge in enumerate(route):
        occurrence[int(edge)].append(index)
    length_prefix = np.r_[0.0, np.cumsum(graph.edge_lengths[route])]
    broken = graph.heads[route[:-1]] != graph.tails[route[1:]]
    break_prefix = np.r_[0, np.cumsum(broken)]
    return route, occurrence, length_prefix, break_prefix


class DecoderCalibrationHelperTests(unittest.TestCase):
    def test_adjacent_collapse_preserves_nonconsecutive_loop_traversals(self):
        np.testing.assert_array_equal(adjacent_collapse([0, 0, 1, 1, 2, 0, 0]), [0, 1, 2, 0])
        np.testing.assert_array_equal(adjacent_collapse([]), [])
        np.testing.assert_array_equal(adjacent_collapse([0]), [0])

    def test_proxy_projection_distinguishes_supporting_line_from_clamped_segment(self):
        graph = metric_graph([(0, 0), (100, 0), (200, 0), (200, 0)], [(0, 1), (2, 3)])
        xy = graph.project(metric_track([(110, 10), (203, 4)])[:, :2])
        fractions, exact, perpendicular, zero = project_selected(graph, xy, np.array([0, 1]))
        np.testing.assert_allclose(fractions, [1, 0], atol=1e-8)
        np.testing.assert_allclose(exact, [math.sqrt(200), 5], atol=1e-8)
        np.testing.assert_allclose(perpendicular, [10, 5], atol=1e-8)
        np.testing.assert_array_equal(zero, [False, True])

    def test_forward_reference_span_uses_partial_arcs_and_repeated_occurrences(self):
        graph = metric_graph([(0, 0), (100, 0), (200, 0)], [(0, 1), (1, 2), (2, 0)])
        context = reference_context(graph, [0, 1, 2, 0])
        distance, reason = reference_span_distance(graph, *context, 0, 0.1, 1, 0.5)
        self.assertEqual(reason, "identified")
        self.assertAlmostEqual(distance, 140, delta=1e-8)
        # Large backwards projection on the same directed edge needs the actual loop.
        distance, reason = reference_span_distance(graph, *context, 0, 0.8, 0, 0.2)
        self.assertEqual(reason, "identified")
        self.assertAlmostEqual(distance, 340, delta=1e-8)
        # Small backward movement is explicitly treated as jitter, not a full loop.
        distance, reason = reference_span_distance(graph, *context, 0, 0.5, 0, 0.4)
        self.assertEqual((distance, reason), (0, "identified"))

    def test_unidentified_reference_spans_have_explicit_reasons(self):
        graph = metric_graph([(0, 0), (100, 0), (200, 0)], [(0, 1), (1, 2), (2, 0)])
        context = reference_context(graph, [0, 1])
        self.assertEqual(reference_span_distance(graph, *context, 1, 0.8, 0, 0.2),
                         (None, "no_forward_reference_occurrence"))
        self.assertEqual(reference_span_distance(graph, *context, 0, 0.8, 0, 0.2),
                         (None, "backward_proxy_without_reference_loop"))
        broken = metric_graph([(0, 0), (100, 0), (200, 0), (300, 0)], [(0, 1), (2, 3)])
        context = reference_context(broken, [0, 1])
        self.assertEqual(reference_span_distance(broken, *context, 0, 0.2, 1, 0.8),
                         (None, "reference_span_crosses_discontinuity"))

    def test_pointwise_macro_f1_has_fixed_reference_class_denominator(self):
        scored = classification_metrics(np.array([0, 1]), np.array([0, 0]))
        self.assertEqual(scored["accuracy"], 0.5)
        self.assertAlmostEqual(scored["macro_f1"], 1 / 3)
        with self.assertRaises(ValueError):
            classification_metrics(np.array([], dtype=int), np.array([], dtype=int))

    def test_calibration_plan_discloses_in_sample_status_and_freeze(self):
        declaration = plan()
        self.assertEqual(declaration["train_trip_count"], 80)
        self.assertEqual(declaration["calibration_trip_ids"], list(CALIBRATION_IDS))
        self.assertEqual(len(declaration["configurations"]), 3)
        self.assertEqual(len(set(CALIBRATION_IDS)), 8)
        self.assertTrue(any("in-sample" in note for note in declaration["limitations"]))
        self.assertTrue(any("not Baum-Welch" in note for note in declaration["limitations"]))
        self.assertIn("before student-development", declaration["freeze"])
        self.assertIn("failed trip contributes zero", declaration["objective"])

    def test_scale_sampling_retains_original_adjacent_pair_indices(self):
        np.testing.assert_array_equal(sample_indices(5, 256), [0, 1, 2, 3, 4])
        indices = sample_indices(1000, 128, start=1)
        self.assertEqual(len(indices), 128)
        self.assertEqual(indices[0], 1)
        self.assertEqual(indices[-1], 999)
        self.assertTrue((np.diff(indices) > 0).all())


if __name__ == "__main__":
    unittest.main()
