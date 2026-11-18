"""Stateless accuracy feature schemas, physics, leakage and invariance tests."""
from __future__ import annotations

from copy import deepcopy
import unittest
from unittest import mock

import numpy as np

from mapmatch.accuracy_features import (
    ACCURACY_FEATURE_VERSION, feature_names, transform_accuracy_features,
)
from mapmatch.features import FEATURE_NAMES, FEATURE_VERSION, PROXY_LABEL_POLICY


_PROFILES = ("context", "compact", "physics")


def baseline_rows(distances, mismatch=None):
    """Synthetic BASE38 rows; tests never load graph/track/route data."""
    distance = np.asarray(distances, dtype=np.float64)
    X = np.zeros((len(distance), 38), dtype=np.float64)
    X[:, FEATURE_NAMES.index("distance_m")] = distance
    X[:, FEATURE_NAMES.index("log1p_distance_m")] = np.log1p(distance)
    X[:, FEATURE_NAMES.index("distance_squared_10m")] = np.minimum((distance / 10.0) ** 2, 10000.0)
    X[:, FEATURE_NAMES.index("distance_to_endpoint_m")] = 20.0
    X[:, FEATURE_NAMES.index("gps_turn_cos")] = 1.0
    X[:, FEATURE_NAMES.index("low_motion")] = 1.0
    X[:, FEATURE_NAMES.index("neighbour_mean_lateral_distance_m")] = distance
    X[:, FEATURE_NAMES.index("neighbour_max_lateral_distance_m")] = distance
    if mismatch is not None:
        X[:, FEATURE_NAMES.index("reliable_direction_mismatch")] = mismatch
        for window in ("short", "medium", "long"):
            X[:, FEATURE_NAMES.index("heading_reliability_" + window)] = 1.0
            X[:, FEATURE_NAMES.index("heading_cos_" + window)] = 1.0 - np.asarray(mismatch)
    return X


def named_column(matrix, name, profile="context"):
    return matrix[:, feature_names(profile).index(name)]


class AccuracyFeatureTests(unittest.TestCase):
    def test_version_schemas_dimensions_and_unchanged_context_prefix(self):
        self.assertEqual(ACCURACY_FEATURE_VERSION, "pathsnap-candidate-v2")
        self.assertEqual(FEATURE_VERSION, "pathsnap-candidate-v1")
        self.assertEqual(len(FEATURE_NAMES), 38)
        self.assertEqual(feature_names(), feature_names("context"))
        self.assertEqual(feature_names("context")[:38], FEATURE_NAMES)
        X = baseline_rows([0, 2, 2, 9], [0, 0, 1, 2]).astype(np.float32)
        for profile, dimensions in (("context", 70), ("compact", 12), ("physics", 3)):
            with self.subTest(profile=profile):
                names = feature_names(profile)
                self.assertIsInstance(names, tuple)
                self.assertEqual(names, feature_names(profile))
                self.assertEqual(len(names), dimensions)
                self.assertEqual(len(names), len(set(names)))
                result = transform_accuracy_features(X, [0, 4], profile)
                self.assertEqual(result.shape, (len(X), dimensions))
                self.assertEqual(result.dtype, np.dtype(np.float64))
                self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(transform_accuracy_features(X, [0, 4])[:, :38], X)
        self.assertLessEqual(len(feature_names("context")), 80)
        self.assertTrue(8 <= len(feature_names("compact")) <= 14)
        self.assertTrue(2 <= len(feature_names("physics")) <= 5)
        forbidden = ("reference", "label", "target", "student", "edge_id", "trip_id",
                     "latitude", "longitude", "timestamp", "severity")
        self.assertFalse(any(part in name for name in feature_names() for part in forbidden))

    def test_known_geometry_energy_gap_rank_and_signed_margins(self):
        X = baseline_rows([0, 3, 4, 0, 10], [1, 0, 0, 0, 0])
        result = transform_accuracy_features(X, [0, 3, 5])
        energy = np.array([32, 9, 16, 0, 100], dtype=float)
        np.testing.assert_array_equal(named_column(result, "geometry_energy_m2"), energy)
        np.testing.assert_allclose(named_column(result, "log1p_geometry_energy_32m2"), np.log1p(energy / 32))
        np.testing.assert_array_equal(named_column(result, "geometry_energy_gap_m2"), [23, 0, 7, 0, 100])
        np.testing.assert_array_equal(named_column(result, "geometry_energy_strict_rank"), [1, 0, 0.5, 0, 1])
        margin = np.array([-23, 7, -7, 100, -100], dtype=float)
        np.testing.assert_allclose(named_column(result, "signed_log1p_geometry_energy_margin_32m2"),
                                   np.sign(margin) * np.log1p(np.abs(margin) / 32))
        np.testing.assert_array_equal(named_column(result, "geometry_energy_distinct_gap_m2"), [7, 7, 7, 100, 100])
        np.testing.assert_allclose(named_column(result, "geometry_energy_soft_support"),
                                   np.exp(-np.array([23, 0, 7, 0, 100]) / 32))

    def test_zero_energy_stationary_ties_and_singletons_are_safe(self):
        X = baseline_rows([0, 0, 2, 0])
        result = transform_accuracy_features(X, [0, 3, 3, 4])
        np.testing.assert_array_equal(named_column(result, "geometry_energy_strict_rank"), [0, 0, 1, 0])
        np.testing.assert_allclose(named_column(result, "signed_log1p_geometry_energy_margin_32m2"),
                                   [0, 0, -np.log1p(4 / 32), 0])
        np.testing.assert_array_equal(named_column(result, "geometry_energy_distinct_gap_m2"), [4, 4, 4, 0])
        np.testing.assert_array_equal(named_column(result, "multiwindow_direction_mismatch"), 0)
        np.testing.assert_array_equal(named_column(result, "direction_energy_fraction"), 0)
        for profile in _PROFILES:
            with self.subTest(profile=profile):
                self.assertTrue(np.isfinite(transform_accuracy_features(X, [0, 3, 3, 4], profile)).all())
                all_zero = transform_accuracy_features(np.zeros((3, 38)), [0, 3], profile)
                self.assertTrue(np.isfinite(all_zero).all())

    def test_multiwindow_reliability_gates_angles_and_disagreement(self):
        X = baseline_rows([4, 4])
        for window, cosine, reliability in (("short", 1.0, 1.0),
                                             ("medium", -1.0, 0.5),
                                             ("long", -1.0, 0.0)):
            X[:, FEATURE_NAMES.index("heading_cos_" + window)] = cosine
            X[:, FEATURE_NAMES.index("heading_reliability_" + window)] = [reliability, 0]
        X[:, FEATURE_NAMES.index("reliable_direction_mismatch")] = [1, 0]
        result = transform_accuracy_features(X, [0, 2])
        np.testing.assert_allclose(named_column(result, "multiwindow_direction_mismatch"), [1 / 3, 0])
        np.testing.assert_allclose(named_column(result, "multiwindow_direction_disagreement"), [0.5, 0])
        np.testing.assert_allclose(named_column(result, "reliable_heading_angle_cost"), [0.5, 0])
        np.testing.assert_allclose(named_column(result, "multiwindow_geometry_energy_m2"), [16 + 32 / 3, 16])
        np.testing.assert_allclose(named_column(result, "distance_heading_interaction"), [2 / 3, 0])
        np.testing.assert_allclose(named_column(result, "direction_energy_fraction"), [0.4, 0])

    def test_candidate_nearest_angular_cost_uses_all_nearest_ties(self):
        X = baseline_rows([1, 1, 2, 0, 0], [2, 0, 1, 2, 0])
        result = transform_accuracy_features(X, [0, 3, 5])
        np.testing.assert_array_equal(named_column(result, "candidate_nearest_angular_cost_m2"), [64, 0, 32, 64, 0])
        # A more distant, better-aligned candidate can have negative relative cost.
        X = baseline_rows([1, 2], [2, 0])
        np.testing.assert_array_equal(named_column(transform_accuracy_features(X, [0, 2]),
                                                   "candidate_nearest_angular_cost_m2"), [0, -64])

    def test_compact_and_physics_match_context_names_without_raw_duplicate_columns(self):
        X = baseline_rows([0, 5, 20, 3], [0, 2, 1, 0])
        context = transform_accuracy_features(X, [0, 3, 4])
        for profile in ("compact", "physics"):
            with self.subTest(profile=profile):
                result = transform_accuracy_features(X, [0, 3, 4], profile)
                expected = context[:, [feature_names().index(name) for name in feature_names(profile)]]
                np.testing.assert_array_equal(result, expected)
                self.assertNotIn("distance_m", feature_names(profile))
                self.assertNotIn("heading_cos_medium", feature_names(profile))

    def test_signed_sqrt_velocity_retains_direction_and_clips_extremes(self):
        X = baseline_rows([0, 0, 0, 0])
        X[:, FEATURE_NAMES.index("motion_along_arc_mps")] = [-40, 0, 40, -1e300]
        X[:, FEATURE_NAMES.index("neighbour_projected_speed_mps")] = [40, 0, -40, 1e300]
        X[:, FEATURE_NAMES.index("heading_reliability_medium")] = 1
        result = transform_accuracy_features(X, [0, 4])
        np.testing.assert_array_equal(named_column(result, "signed_sqrt_motion_along_arc_10mps"), [-2, 0, 2, -10])
        np.testing.assert_array_equal(named_column(result, "signed_sqrt_projected_speed_10mps"), [2, 0, -2, 10])
        np.testing.assert_array_equal(named_column(result, "reliable_reverse_motion"), [2, 0, 0, 10])

    def test_candidate_permutation_equivariance_with_duplicate_values_and_empty_groups(self):
        rng = np.random.default_rng(51)
        X = rng.normal(size=(12, 38))
        X[:, FEATURE_NAMES.index("distance_m")] = [1, 1, 2, 5, 0, 0, 3, 3, 1, 4, 4, 4]
        X[:, FEATURE_NAMES.index("reliable_direction_mismatch")] = [1, 0, 0, 1, 0, 0, 1, 1, 2, 0, 0, 1]
        X[7] = X[6]
        offsets = np.array([0, 4, 4, 9, 12])
        for profile in _PROFILES:
            expected = transform_accuracy_features(X, offsets, profile)
            for _ in range(5):
                order = np.concatenate([rng.permutation(np.arange(lo, hi))
                                        for lo, hi in zip(offsets[:-1], offsets[1:])])
                permuted = transform_accuracy_features(X[order], offsets, profile)
                with self.subTest(profile=profile):
                    np.testing.assert_array_equal(permuted, expected[order])
                    np.testing.assert_array_equal(expected[6], expected[7])

    def test_observations_are_independent_and_empty_groups_do_not_change_values(self):
        X = baseline_rows([1, 3, 9, 1, 4], [1, 0, 0, 2, 0])
        for profile in _PROFILES:
            with self.subTest(profile=profile):
                batched = transform_accuracy_features(X, [0, 3, 3, 5], profile)
                separate = np.concatenate([transform_accuracy_features(X[:3], [0, 3], profile),
                                           transform_accuracy_features(X[3:], [0, 2], profile)])
                np.testing.assert_array_equal(batched, separate)
                np.testing.assert_array_equal(batched, transform_accuracy_features(X, [0, 3, 5], profile))

    def test_empty_matrix_and_integer_valued_float_offsets(self):
        for profile in _PROFILES:
            for offsets in ([0], [0, 0, 0], np.array([0.0, 0.0])):
                with self.subTest(profile=profile, offsets=offsets):
                    result = transform_accuracy_features(np.empty((0, 38), dtype=np.float32), offsets, profile)
                    self.assertEqual(result.shape, (0, len(feature_names(profile))))
                    self.assertEqual(result.dtype, np.dtype(np.float64))
        X = baseline_rows([1, 2])
        np.testing.assert_array_equal(transform_accuracy_features(X, [0.0, 2.0]),
                                      transform_accuracy_features(X, [0, 2]))

    def test_inputs_are_not_mutated_or_shared_including_readonly_and_noncontiguous(self):
        X = baseline_rows([0, 1, 2, 3, 4, 5])[::2]
        offsets = np.array([0, 2, 2, 3], dtype=np.int64)
        original_X, original_offsets = X.copy(), offsets.copy()
        X.setflags(write=False)
        offsets.setflags(write=False)
        for profile in _PROFILES:
            with self.subTest(profile=profile):
                result = transform_accuracy_features(X, offsets, profile)
                self.assertFalse(np.shares_memory(result, X))
                self.assertFalse(np.shares_memory(result, offsets))
                result[:] = 123
                np.testing.assert_array_equal(X, original_X)
                np.testing.assert_array_equal(offsets, original_offsets)

    def test_no_label_calls_file_io_or_baseline_policy_changes(self):
        original_names = FEATURE_NAMES
        original_policy = deepcopy(PROXY_LABEL_POLICY)
        X = baseline_rows([0, 3], [0, 1])
        with mock.patch("mapmatch.features.reference_observation_labels", side_effect=AssertionError("label access")), \
                mock.patch("mapmatch.features.reference_observation_edges", side_effect=AssertionError("label access")), \
                mock.patch("mapmatch.data.load_route", side_effect=AssertionError("file access")), \
                mock.patch("builtins.open", side_effect=AssertionError("file access")), \
                mock.patch("numpy.load", side_effect=AssertionError("file access")):
            for profile in _PROFILES:
                self.assertTrue(np.isfinite(transform_accuracy_features(X, [0, 2], profile)).all())
        self.assertIs(FEATURE_NAMES, original_names)
        self.assertEqual(PROXY_LABEL_POLICY, original_policy)

    def test_extreme_finite_inputs_remain_finite_and_context_prefix_is_exact(self):
        X = np.array([[np.finfo(float).max] * 38, [-np.finfo(float).max] * 38])
        for profile in _PROFILES:
            with self.subTest(profile=profile), np.errstate(all="raise"):
                result = transform_accuracy_features(X, [0, 2], profile)
                self.assertTrue(np.isfinite(result).all())
                if profile == "context":
                    np.testing.assert_array_equal(result[:, :38], X)

    def test_invalid_matrix_shape_dtype_or_nonfinite_values_are_rejected(self):
        invalid = [np.zeros(38), np.zeros((1, 37)), np.zeros((1, 39)), np.zeros((1, 1, 38)),
                   np.zeros((1, 38), dtype=complex), np.zeros((1, 38), dtype=bool),
                   np.full((1, 38), "0"), np.zeros((1, 38), dtype=object), [[1], [1, 2]], None]
        for bad_value in (np.nan, np.inf, -np.inf):
            X = baseline_rows([1])
            X[0, 10] = bad_value
            invalid.append(X)
        for X in invalid:
            for profile in _PROFILES:
                with self.subTest(profile=profile, shape=np.shape(X) if not isinstance(X, list) else "ragged"), self.assertRaises(ValueError):
                    transform_accuracy_features(X, [0, 1], profile)

    def test_invalid_offsets_cannot_truncate_reorder_or_skip_rows(self):
        X = baseline_rows([1, 2, 3])
        invalid = [[], [0], [1, 3], [0, 2], [0, 4], [0, -1, 3], [0, 2, 1, 3],
                   [0, 1.5, 3], [0, np.nan, 3], [0, np.inf, 3], [[0, 3]],
                   ["0", "3"], np.array([0, 3], dtype=object), np.array([0, 3], dtype=complex),
                   [0, 2**64 - 1], None, [False, True]]
        for offsets in invalid:
            for profile in _PROFILES:
                with self.subTest(profile=profile, offsets=offsets), self.assertRaises(ValueError):
                    transform_accuracy_features(X, offsets, profile)

    def test_unknown_profiles_are_rejected_for_schema_transform_and_empty_inputs(self):
        for profile in ("", "baseline", "CONTEXT", None, 0, [], {}):
            with self.subTest(profile=profile):
                with self.assertRaises(ValueError):
                    feature_names(profile)
                with self.assertRaises(ValueError):
                    transform_accuracy_features(baseline_rows([1]), [0, 1], profile)
                with self.assertRaises(ValueError):
                    transform_accuracy_features(np.empty((0, 38)), [0], profile)


if __name__ == "__main__":
    unittest.main()
