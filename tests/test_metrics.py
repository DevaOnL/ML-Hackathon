"""Independent repeat-aware score and sequence-edit verification."""
from collections import Counter
import math
import unittest

import numpy as np

from mapmatch.metrics import edit_distance, score_routes
from tests.test_core import metric_graph


def full_table_edit(first, second):
    table = [[0] * (len(second) + 1) for _ in range(len(first) + 1)]
    for i in range(len(first) + 1):
        table[i][0] = i
    for j in range(len(second) + 1):
        table[0][j] = j
    for i in range(1, len(first) + 1):
        for j in range(1, len(second) + 1):
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1,
                              table[i - 1][j - 1] + int(first[i - 1] != second[j - 1]))
    return table[-1][-1]


class RouteMetricTests(unittest.TestCase):
    def setUp(self):
        self.graph = metric_graph([(0, 0), (10, 0), (30, 0)],
                                  [(0, 1), (1, 2), (1, 0), (0, 1), (0, 0)])

    def test_missing_repeat_costs_recall_instead_of_set_perfect_score(self):
        scored = score_routes(self.graph, [0, 1], [0, 1, 0])
        self.assertAlmostEqual(scored["length_precision"], 1)
        self.assertAlmostEqual(scored["length_recall"], 0.75)
        self.assertAlmostEqual(scored["length_f1"], 6 / 7)
        self.assertAlmostEqual(scored["edge_f1"], 0.8)
        self.assertEqual(scored["common_arc_count"], 2)
        self.assertFalse(scored["exact_match"])
        self.assertEqual(scored["edit_distance"], 1)

    def test_extra_repeat_costs_precision(self):
        scored = score_routes(self.graph, [0, 1, 0, 0], [0, 1, 0])
        self.assertAlmostEqual(scored["length_precision"], 0.8)
        self.assertAlmostEqual(scored["length_recall"], 1)
        self.assertAlmostEqual(scored["length_f1"], 8 / 9)
        self.assertAlmostEqual(scored["predicted_length_m"], 50, delta=1e-7)
        self.assertAlmostEqual(scored["reference_length_m"], 40, delta=1e-7)
        self.assertAlmostEqual(scored["common_length_m"], 40, delta=1e-7)

    def test_order_is_separate_from_multiset_overlap(self):
        scored = score_routes(self.graph, [1, 0, 0], [0, 1, 0])
        self.assertAlmostEqual(scored["edge_f1"], 1)
        self.assertAlmostEqual(scored["length_f1"], 1)
        self.assertFalse(scored["exact_match"])
        self.assertGreater(scored["edit_distance"], 0)
        self.assertLess(scored["edit_similarity"], 1)

    def test_direction_and_original_duplicate_ids_scored_explicitly(self):
        reversed_route = score_routes(self.graph, [2], [0])
        self.assertEqual(reversed_route["edge_f1"], 0)
        self.assertEqual(reversed_route["length_f1"], 0)
        self.assertEqual(reversed_route["duplicate_canonical_length_f1"], 0)
        duplicates = score_routes(self.graph, [3], [0])
        self.assertEqual(duplicates["length_f1"], 0)
        self.assertEqual(duplicates["duplicate_canonical_length_f1"], 1)
        self.assertFalse(duplicates["exact_match"])

    def test_empty_routes_and_zero_length_geometry_remain_finite(self):
        for prediction, reference, expected in (([], [], 1), ([], [0], 0),
                                                  ([0], [], 0), ([4], [], 0),
                                                  ([4, 4], [4], 2 / 3)):
            with self.subTest(prediction=prediction, reference=reference):
                scored = score_routes(self.graph, prediction, reference)
                for key in ("edge_precision", "edge_recall", "edge_f1", "length_precision",
                            "length_recall", "length_f1", "edit_similarity"):
                    self.assertTrue(math.isfinite(scored[key]), key)
                self.assertAlmostEqual(scored["length_f1"], expected)

    def test_zero_length_count_fallback_must_not_relabel_counts_as_meters(self):
        scored = score_routes(self.graph, [4, 4], [4])
        self.assertEqual(scored["predicted_length_m"], 0)
        self.assertEqual(scored["reference_length_m"], 0)
        self.assertEqual(scored["common_length_m"], 0)
        self.assertAlmostEqual(scored["length_f1"], 2 / 3)

    def test_invalid_ids_and_fractional_rows_do_not_wrap_indexing(self):
        for bad in ([-1], [5], [0.2], [math.nan], [math.inf], [[0]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                score_routes(self.graph, bad, [0])
            with self.subTest(reference=bad), self.assertRaises(ValueError):
                score_routes(self.graph, [0], bad)

    def test_continuity_defects_reported_without_hiding_reference_issue(self):
        scored = score_routes(self.graph, [0, 1], [1, 0, 1])
        self.assertTrue(scored["prediction_connected"])
        self.assertEqual(scored["route_continuity"], 1)
        self.assertFalse(scored["reference_connected"])
        self.assertEqual(scored["reference_discontinuity_count"], 1)
        self.assertIn("Not an official", scored["metric_note"])

    def test_random_repeated_routes_match_counter_oracle(self):
        rng = np.random.default_rng(73)
        for _ in range(60):
            prediction = rng.integers(0, len(self.graph.arcs), int(rng.integers(1, 20))).tolist()
            reference = rng.integers(0, len(self.graph.arcs), int(rng.integers(1, 20))).tolist()
            common = Counter(prediction) & Counter(reference)
            matched_count = sum(common.values())
            count_f1 = 2 * matched_count / (len(prediction) + len(reference))
            matched_length = sum(float(self.graph.edge_lengths[e]) * n for e, n in common.items())
            denominator = float(self.graph.edge_lengths[prediction].sum() +
                                self.graph.edge_lengths[reference].sum())
            length_f1 = 2 * matched_length / denominator if denominator else count_f1
            scored = score_routes(self.graph, prediction, reference)
            self.assertAlmostEqual(scored["edge_f1"], count_f1)
            self.assertAlmostEqual(scored["length_f1"], length_f1)
            expected_edit = full_table_edit(prediction, reference)
            self.assertEqual(scored["edit_distance"], expected_edit)
            self.assertAlmostEqual(scored["edit_similarity"],
                                   1 - expected_edit / max(len(prediction), len(reference)))

    def test_edit_distance_edge_cases_and_independent_oracle(self):
        for first, second in (([], []), ([], [1, 2]), ([1, 2], []), ([0], [1]),
                              ([0, 1, 0], [0, 1]), ([0, 1, 2], [2, 1, 0])):
            self.assertEqual(edit_distance(first, second), full_table_edit(first, second))
            self.assertEqual(edit_distance(first, second), edit_distance(second, first))


if __name__ == "__main__":
    unittest.main()
