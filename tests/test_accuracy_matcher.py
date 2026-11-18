"""TASK52 synthetic-only tests: no provided trips, models or target files are read."""
from __future__ import annotations

import ast
import copy
from dataclasses import replace
import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from mapmatch import accuracy_matcher
from mapmatch.accuracy_features import ACCURACY_FEATURE_VERSION, transform_accuracy_features
from mapmatch.accuracy_matcher import (AccuracyEmissionModel, AccuracyMapMatcher,
    AccuracyTransitionPolicy, TRAINING_SCOPE, accuracy_feature_names,
    load_accuracy_emission_model, strict_binary_probabilities)
from mapmatch.candidates import CandidateIndex, Candidates
from mapmatch.features import CandidateFeatureExtractor, FEATURE_NAMES, FEATURE_VERSION
from mapmatch.matcher import MapMatcher, MatchConfig
from scripts.accuracy_decoder import (DecoderConfiguration, aggregate_route_records,
    assert_paired_models, decode_arrays, decoder_configurations, menu_manifest,
    require_decoder_authorization, score_decoded_route, selection_plan,
    verify_menu_registration)
from tests.test_core import metric_graph, metric_track

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "analysis/accuracy_decoder"


class ProbabilitySpy(ClassifierMixin, BaseEstimator):
    def __init__(self, n_features=38, classes=(0, 1), raw=None):
        self.n_features = n_features
        self.classes = classes
        self.raw = raw
        self.n_features_in_ = n_features
        self.classes_ = np.asarray(classes)
        self.calls = []

    def fit(self, X, y):
        raise AssertionError("spy must never be fitted")

    def predict_proba(self, X):
        self.calls.append(np.asarray(X).copy())
        if self.raw is not None:
            raw = np.asarray(self.raw)
            return np.tile(raw, (len(X), 1)) if raw.ndim == 1 else raw.copy()
        p = 0.2 + 0.6 / (1.0 + np.abs(X[:, 0]))
        return np.column_stack([p if c == 1 else 1.0-p for c in self.classes_])


class Downcast(TransformerMixin, BaseEstimator):
    def __init__(self, n_features=38):
        self.n_features = n_features
        self.n_features_in_ = n_features

    def fit(self, X, y=None):
        raise AssertionError("synthetic pipeline is already configured")

    def transform(self, X):
        return np.asarray(X, dtype=np.float32)


class GNBSpy(GaussianNB):
    def predict_proba(self, X):
        self.scoring_dtype_ = np.asarray(X).dtype
        return super().predict_proba(X)


def metadata(profile="v1", **overrides):
    value = {"profile": profile, "feature_names": accuracy_feature_names(profile),
             "base_feature_names": FEATURE_NAMES, "base_feature_version": FEATURE_VERSION,
             "feature_version": FEATURE_VERSION if profile == "v1" else ACCURACY_FEATURE_VERSION,
             "family": "synthetic", "run_name": "synthetic_only", "parameters": {},
             "fold_id": 0, "fold": 0, "train_trip_ids": ["fit_a", "fit_b"],
             "validation_trip_ids": ["valid_a", "valid_b"], "training_scope": TRAINING_SCOPE,
             "protocol_sha256": "a" * 64,
             "source_sha256": {"common": "b" * 64, "factory_module": "c" * 64, "features": "d" * 64}}
    value.update(overrides)
    return value


def emission(profile="v1", **kwargs):
    return AccuracyEmissionModel(ProbabilitySpy(len(accuracy_feature_names(profile)), **kwargs), metadata(profile))


def synthetic_folds():
    ids = [f"synthetic_{i:03d}" for i in range(64)]
    return [{"fold": fold, "train_trip_ids": [t for i, t in enumerate(ids) if i % 3 != fold],
             "validation_trip_ids": [t for i, t in enumerate(ids) if i % 3 == fold]} for fold in range(3)]


class AccuracyEmissionTests(unittest.TestCase):
    def setUp(self):
        self.graph = metric_graph([(0, 0), (100, 0)], [(0, 1), (1, 0)])
        self.track = metric_track([(20, 1), (25, 1), (21, 1), (30, 1)])
        self.indices = np.array([0, 2, 3])
        index = CandidateIndex(self.graph)
        self.groups = [index.query(p, 80, 12) for p in self.graph.project(self.track[self.indices, :2])]
        c = self.groups[0]
        self.groups[0] = Candidates(c.edges[::-1], c.fractions[::-1], c.distances[::-1])
        c = self.groups[1]
        self.groups[1] = Candidates(c.edges[:1], c.fractions[:1], c.distances[:1])

    def test_all_profiles_transform_whole_group_once_float64_and_preserve_order(self):
        base, offsets = CandidateFeatureExtractor(self.graph).transform(self.track, self.indices, self.groups)
        for profile in ("v1", "context", "compact", "physics"):
            with self.subTest(profile=profile):
                model = emission(profile, classes=(1, 0))
                actual = model.score_candidates_batch(self.graph, self.track, self.indices, self.groups)
                expected_X = np.asarray(base, np.float64) if profile == "v1" else transform_accuracy_features(base, offsets, profile)
                self.assertEqual(len(model.estimator.calls), 1)
                self.assertEqual(model.estimator.calls[0].dtype, np.dtype("float64"))
                np.testing.assert_array_equal(model.estimator.calls[0], expected_X)
                p = 0.2 + 0.6 / (1 + np.abs(expected_X[:, 0]))
                for i, lo, hi, values in zip(self.indices, offsets[:-1], offsets[1:], actual):
                    np.testing.assert_allclose(values, p[lo:hi] / p[lo:hi].sum(), rtol=0, atol=1e-15)
                    np.testing.assert_array_equal(model.records[int(i)].edges, self.groups[list(self.indices).index(i)].edges)
                self.assertEqual(set(model.records), {0, 2, 3})

    def test_full_track_context_not_subsampled_track_is_used(self):
        model = emission("context")
        model.score_candidates_batch(self.graph, self.track, self.indices, self.groups)
        X, offsets = CandidateFeatureExtractor(self.graph).transform(self.track, self.indices, self.groups)
        expected = transform_accuracy_features(X, offsets, "context")
        np.testing.assert_array_equal(model.estimator.calls[0], expected)

    def test_v1_probability_clip_then_positive_group_normalization(self):
        model = emission(raw=np.array([1.0, 0.0], np.float64))
        rows = model.score_candidates_batch(self.graph, self.track, self.indices, self.groups)
        np.testing.assert_allclose(rows[0], [0.5, 0.5])
        np.testing.assert_allclose(rows[1], [1.0])
        self.assertEqual(model.probability_checks["clipped_low_rows"], 5)
        np.testing.assert_array_equal(model.records[0].positive_probabilities, [0, 0])

    def test_both_raw_probability_columns_strictly_checked_before_clip(self):
        invalid = [np.array([0.8, 0.3]), np.array([0.8, 0.200000001]),
                   np.array([np.nan, 0.5]), np.array([np.inf, 0.0]),
                   np.array([-0.1, 1.1]), np.array([0.0, 0.0]),
                   np.array([0.4, 0.6], np.float32), np.array([[0.4, 0.6]])]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "probabilities"):
                emission(raw=raw).score_candidates_batch(self.graph, self.track, self.indices, self.groups)

    def test_final_pipeline_boundary_repromotes_custom_downcast(self):
        spy = ProbabilitySpy(38)
        pipe = Pipeline([("downcast", Downcast()), ("classifier", spy)])
        model = AccuracyEmissionModel(pipe, metadata())
        model.score_candidates_batch(self.graph, self.track, self.indices, self.groups)
        self.assertEqual(spy.calls[0].dtype, np.dtype("float64"))

    def test_real_synthetic_gnb_last_step_receives_float64(self):
        X = np.arange(228, dtype=np.float64).reshape(6, 38) / 100
        classifier = GNBSpy().fit(X, [0, 0, 0, 1, 1, 1])
        pipe = Pipeline([("downcast", Downcast()), ("classifier", classifier)])
        raw = strict_binary_probabilities(pipe, X.astype(np.float32))
        self.assertEqual(classifier.scoring_dtype_, np.dtype("float64"))
        np.testing.assert_allclose(raw.sum(axis=1), 1, rtol=0, atol=1e-10)

    def test_invalid_metadata_and_unfitted_estimator_are_rejected(self):
        changes = [{"profile": "baseline38"}, {"profile": "unknown"},
                   {"feature_names": FEATURE_NAMES[::-1]}, {"base_feature_names": FEATURE_NAMES[:-1]},
                   {"feature_names": None}, {"base_feature_names": None},
                   {"feature_version": ACCURACY_FEATURE_VERSION}, {"base_feature_version": "v2"},
                   {"fold_id": 1}, {"fold_id": True, "fold": True}, {"fold_id": 0.0},
                   {"train_trip_ids": ["valid_a"]}, {"train_trip_ids": ["a", "a"]},
                   {"training_scope": "all80"}, {"source_sha256": {}}, {"protocol_sha256": "bad"}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                AccuracyEmissionModel(ProbabilitySpy(), metadata(**change))
        with self.assertRaisesRegex(ValueError, "fitted"):
            AccuracyEmissionModel(GaussianNB(), metadata())
        for classes in ((0, 0), (1,), ("0", "1"), (False, True)):
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                AccuracyEmissionModel(ProbabilitySpy(classes=classes), metadata())
        with self.assertRaisesRegex(ValueError, "feature count"):
            AccuracyEmissionModel(ProbabilitySpy(37), metadata())

    def test_bad_candidate_groups_or_indices_fail_without_predict(self):
        for indices in ([0, 0, 3], [0, 3, 2], [0, 2], [0, 2, 4], [0, 2, 2.5]):
            model = emission()
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                model.score_candidates_batch(self.graph, self.track, indices, self.groups)
            self.assertEqual(model.estimator.calls, [])
        for group in (Candidates(np.array([], dtype=int), np.array([]), np.array([])),
                      Candidates(np.array([0, 0]), np.array([0.5, 0.5]), np.array([1., 1.])),
                      Candidates(np.array([2]), np.array([0.5]), np.array([1.])),
                      Candidates(np.array([0]), np.array([1.1]), np.array([1.])),
                      Candidates(np.array([0]), np.array([0.5]), np.array([np.nan]))):
            with self.subTest(group=group), self.assertRaises(ValueError):
                emission().score_candidates_batch(self.graph, self.track, [0], [group])

    def test_inference_opens_no_files_and_never_calls_proxy_label_helpers(self):
        model = emission("physics")
        with mock.patch("builtins.open", side_effect=AssertionError("no inference file IO")), \
             mock.patch("mapmatch.features.reference_observation_labels", side_effect=AssertionError("no target")), \
             mock.patch("mapmatch.features.reference_observation_edges", side_effect=AssertionError("no target")):
            model.score_candidates_batch(self.graph, self.track, self.indices, self.groups)
            result = AccuracyMapMatcher(self.graph, emission_model=model).match(self.track)
        self.assertFalse(result.diagnostics["uses_reference_route"])
        tree = ast.parse(inspect.getsource(accuracy_matcher))
        imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names]
        self.assertFalse(any("reference" in name or name in {"load_route", "load_track", "load_graph"} for name in imports))

    def test_trusted_loader_pins_bytes_and_fold_partitions(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            path = Path(root) / "fold_0.joblib"
            bundle = {"estimator": ProbabilitySpy(), **metadata()}
            joblib.dump(bundle, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = {"expected_sha256": digest, "expected_fold_id": 0,
                        "expected_train_trip_ids": ["fit_b", "fit_a"],
                        "expected_validation_trip_ids": ["valid_b", "valid_a"],
                        "expected_protocol_sha256": "a" * 64}
            model = load_accuracy_emission_model(path, **expected)
            self.assertEqual(model.profile, "v1")
            with mock.patch("mapmatch.accuracy_matcher.joblib.load", side_effect=AssertionError("unpickle before hash")):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    load_accuracy_emission_model(path, **{**expected, "expected_sha256": "0" * 64})
            for change in ({"expected_fold_id": 1}, {"expected_fold_id": 0.0},
                           {"expected_train_trip_ids": ["fit_a"]},
                           {"expected_validation_trip_ids": ["other"]}, {"expected_protocol_sha256": "e"*64}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_accuracy_emission_model(path, **{**expected, **change})


class AccuracyTransitionTests(unittest.TestCase):
    def setUp(self):
        self.graph = metric_graph([(0, 0), (100, 0), (0, 30), (100, 30)],
                                  [(0, 1), (1, 0), (2, 3)])
        self.config = MatchConfig(sigma_m=7.5, beta_m=5)

    def test_disabled_policy_exactly_matches_v1_routes_states_and_cost(self):
        track = metric_track([(20, 1), (21, 1), (20, 1), (30, 1)])
        original = MapMatcher(self.graph, self.config, emission()).match(track)
        accuracy = AccuracyMapMatcher(self.graph, self.config, emission()).match(track)
        for name in ("route", "observation_indices", "matched_edges", "matched_fractions", "matched_distances"):
            np.testing.assert_array_equal(getattr(accuracy, name), getattr(original, name))
        self.assertEqual(accuracy.diagnostics["negative_log_likelihood_relative"],
                         original.diagnostics["negative_log_likelihood_relative"])
        self.assertEqual(FEATURE_VERSION, "pathsnap-candidate-v1")

    def test_cheap_slack_is_exact_deadband_without_extra_routing_or_reachability(self):
        previous = Candidates(np.array([0]), np.array([0.2]), np.array([1.]))
        current = Candidates(np.array([0, 2]), np.array([0.4, 0.4]), np.array([1., 1.]))
        original = MapMatcher(self.graph, self.config)
        matcher = AccuracyMapMatcher(self.graph, self.config, transition_policy=AccuracyTransitionPolicy(path_cost_slack_m=10))
        base = original._transitions(previous, current, 15, 300)
        actual = matcher._transitions(previous, current, 15, 300)
        np.testing.assert_array_equal(np.isfinite(base), np.isfinite(actual))
        np.testing.assert_allclose(actual, np.maximum(base-2, 0))
        self.assertEqual(matcher.router.searches, original.router.searches)

    def test_stationary_fraction_backtrack_and_reverse_penalties_are_local(self):
        previous = Candidates(np.array([0]), np.array([0.5]), np.array([0.]))
        current = Candidates(np.array([0, 1]), np.array([0.45, 0.5]), np.array([0., 0.]))
        base = MapMatcher(self.graph, self.config)._transitions(previous, current, 1, 300)
        policy = AccuracyTransitionPolicy(stationary_backtrack_weight=1, immediate_reverse_penalty=2)
        matcher = AccuracyMapMatcher(self.graph, self.config, transition_policy=policy)
        actual = matcher._transitions(previous, current, 1, 300)
        np.testing.assert_allclose(actual-base, [[0.8, 2]], atol=1e-8)
        moving = matcher._transitions(previous, current, 10, 300)
        np.testing.assert_array_equal(moving, MapMatcher(self.graph, self.config)._transitions(previous, current, 10, 300))

    def test_immediate_reverse_cycle_can_be_penalized_but_is_not_deleted(self):
        graph = metric_graph([(0, 0), (1, 0)], [(0, 1), (1, 0)])
        track = metric_track([(0.5, 0), (0.5, 0), (0.5, 0)])
        class Alternating:
            def score_candidates_batch(self, graph, track, indices, candidates):
                return [np.where(c.edges == (1 if j == 1 else 0), 0.9, 0.1) for j, c in enumerate(candidates)]
        config = MatchConfig(beta_m=5, geometry_weight=0)
        original = AccuracyMapMatcher(graph, config, Alternating()).match(track)
        guarded = AccuracyMapMatcher(graph, config, Alternating(), AccuracyTransitionPolicy(immediate_reverse_penalty=2)).match(track)
        self.assertEqual(original.route.tolist(), [0, 1, 0])
        self.assertEqual(original.diagnostics["accuracy_motion"]["route_immediate_reverse_pairs"], 2)
        self.assertEqual(guarded.route.tolist(), [0])
        # Forcing a real moving reversal still retains both arc traversals.
        moving = metric_track([(0, 0), (1, 0), (0, 0)])
        forced = AccuracyMapMatcher(graph, replace(config, classifier_weight=100), Alternating(),
                                    AccuracyTransitionPolicy(immediate_reverse_penalty=2)).match(moving)
        self.assertEqual(forced.route.tolist(), [0, 1, 0])

    def test_same_edge_large_backward_requires_real_loop_no_shortcut(self):
        graph = metric_graph([(0, 0), (100, 0)], [(0, 1), (1, 0)])
        track = metric_track([(80, 0), (10, 0)])
        class Forward:
            def score_candidates_batch(self, graph, track, indices, candidates):
                return [np.where(c.edges == 0, 1.0, 1e-30) for c in candidates]
        config = MatchConfig(classifier_weight=100, beta_m=5)
        result = AccuracyMapMatcher(graph, config, Forward(),
                                    AccuracyTransitionPolicy(stationary_backtrack_weight=1, immediate_reverse_penalty=2)).match(track)
        self.assertEqual(result.route.tolist(), [0, 1, 0])
        self.assertEqual(result.diagnostics["accuracy_motion"]["route_repeated_traversal_count"], 1)

    def test_proactive_k24_is_synthetic_only_and_v1_never_widens_feasible_bad_path(self):
        points, arcs = [], []
        for i in range(24):
            points.extend([(0, i), (100, i)])
            arcs.append((2*i, 2*i+1))
        graph = metric_graph(points, arcs)
        track = metric_track([(20, 0), (21, 0)])
        class PreferLast:
            def score_candidates_batch(self, graph, track, indices, candidates):
                return [np.where(c.edges == 23, 1.0, 1e-12) for c in candidates]
        narrow = MatchConfig(max_candidates=12, candidate_retry_limit=48, geometry_weight=0)
        baseline = MapMatcher(graph, narrow, PreferLast()).match(track)
        proactive = AccuracyMapMatcher(graph, replace(narrow, max_candidates=24), PreferLast()).match(track)
        self.assertEqual(baseline.diagnostics["effective_max_candidates"], 12)
        self.assertEqual(baseline.diagnostics["candidate_retry_attempts"], [])
        self.assertEqual(proactive.matched_edges.tolist(), [23, 23])
        self.assertNotEqual(baseline.route.tolist(), proactive.route.tolist())
        self.assertTrue(all(c.match_config.max_candidates == 12 for c in decoder_configurations()))

    def test_feasible_transition_never_triggers_v1_horizon_expansion(self):
        graph = metric_graph([(0, 0), (100, 0)], [(0, 1)])
        track = metric_track([(20, 0), (21, 0), (22, 0)])
        matcher = AccuracyMapMatcher(graph)
        with mock.patch.object(matcher, "_transitions", wraps=matcher._transitions) as spy:
            result = matcher.match(track)
        self.assertEqual(spy.call_count, 2)
        self.assertEqual(result.diagnostics["expanded_routing_transitions"], 0)
        self.assertTrue(all(call.args[3] < 5000 for call in spy.call_args_list))

    def test_observation_interval_without_distance_retains_every_row(self):
        graph = self.graph
        track = metric_track([(20, 0)] * 12)
        for interval in (1, 3, 5):
            dense = AccuracyMapMatcher(graph, MatchConfig(observation_interval=interval)).match(track)
            np.testing.assert_array_equal(dense.observation_indices, np.arange(12))
            for distance in (5, 10):
                sparse = AccuracyMapMatcher(graph, MatchConfig(observation_distance_m=distance, observation_interval=interval)).match(track)
                expected = sorted(set([0, *range(interval, 11, interval), 11]))
                np.testing.assert_array_equal(sparse.observation_indices, expected)

    def test_empty_candidate_radius_fallback_preserved_under_fixed_k_menu(self):
        graph = metric_graph([(0, 0), (100, 0)], [(0, 1)])
        result = AccuracyMapMatcher(graph, decoder_configurations()[0].config).match(metric_track([(20, 150), (21, 150)]))
        self.assertEqual(result.diagnostics["expanded_candidate_observations"], 2)
        self.assertEqual(result.diagnostics["effective_max_candidates"], 12)

    def test_invalid_transition_policies_fail(self):
        for kwargs in ({"path_cost_slack_m": -1}, {"immediate_reverse_penalty": np.inf},
                       {"stationary_chord_m": np.nan}, {"stationary_backtrack_weight": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AccuracyTransitionPolicy(**kwargs)


class AccuracySelectionTests(unittest.TestCase):
    def test_menu_is_frozen_preregistered_and_bounded_without_data_access(self):
        pin = verify_menu_registration()
        self.assertEqual(len(pin), 64)
        menu = decoder_configurations()
        self.assertEqual(len(menu), 8)
        self.assertEqual(len(set(c.name for c in menu)), 8)
        with mock.patch("builtins.open", side_effect=AssertionError("menu must not read data")):
            self.assertEqual(len(menu_manifest()["configurations"]), 8)
        for c in menu:
            self.assertEqual(c.config.sigma_m, 7.5)
            self.assertEqual(c.config.candidate_retry_limit, 12)
            self.assertEqual(c.config.observation_distance_m, 0)
            self.assertIn(c.config.beta_m, (2, 5, 10))

    def test_six_selection_routes_use_hash_smallest_valid_ids_and_other_folds(self):
        folds = synthetic_folds()
        plan = selection_plan(folds)
        self.assertEqual(len(plan), 6)
        self.assertEqual(len({p["trip_id"] for p in plan}), 6)
        for index in range(3):
            expected = sorted(folds[index]["validation_trip_ids"], key=lambda t: (hashlib.sha256(f"20261118:{t}".encode()).hexdigest(), t))[:2]
            self.assertEqual([p["trip_id"] for p in plan if p["fold_id"] == index], expected)
        shuffled = [dict(f, train_trip_ids=list(reversed(f["train_trip_ids"])),
                         validation_trip_ids=list(reversed(f["validation_trip_ids"]))) for f in reversed(folds)]
        self.assertEqual(plan, selection_plan(shuffled))
        for p in plan:
            self.assertNotIn(p["trip_id"], p["train_trip_ids"])

    def test_malformed_or_non64_fold_plans_fail(self):
        folds = synthetic_folds()
        cases = [folds[:2]]
        overlap = copy.deepcopy(folds)
        overlap[0]["train_trip_ids"].append(overlap[0]["validation_trip_ids"][0])
        cases.append(overlap)
        duplicate = copy.deepcopy(folds)
        duplicate[1]["fold"] = 0
        cases.append(duplicate)
        wrong = copy.deepcopy(folds)
        wrong[0]["train_trip_ids"].append("outer_should_not_appear")
        cases.append(wrong)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                selection_plan(value)
        with self.assertRaisesRegex(ValueError, "population"):
            selection_plan(folds, tuning_trip_ids=["outer"])

    def test_paired_baseline_requires_new_v1_identical_partition_and_protocol(self):
        baseline, candidate = emission(), emission("physics")
        assert_paired_models(baseline, candidate)
        with self.assertRaisesRegex(ValueError, "baseline"):
            assert_paired_models(candidate, baseline)
        for change in ({"protocol_sha256": "e" * 64}, {"fold_id": 1, "fold": 1},
                       {"train_trip_ids": ["different"]}):
            model = AccuracyEmissionModel(ProbabilitySpy(3), metadata("physics", **change))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "differ"):
                assert_paired_models(baseline, model)

    def test_raw_route_authorization_is_separate_from_training(self):
        pins = {"candidate_manifest_sha256": "a"*64, "decoder_menu_sha256": "b"*64}
        for protocol in ({}, {"training_authorized": True},
                         {"training_authorized": True, "decoder_selection": {"authorized": True, **pins}}):
            with self.subTest(protocol=protocol), self.assertRaises(ValueError):
                require_decoder_authorization(protocol, **pins)
        protocol = {"training_authorized": True, "decoder_selection": {"authorized": True, "model_menu_frozen": True, **pins}}
        self.assertTrue(require_decoder_authorization(protocol, **pins)["authorized"])
        with self.assertRaisesRegex(ValueError, "pin"):
            require_decoder_authorization(protocol, **{**pins, "candidate_manifest_sha256": "c"*64})

    def test_decode_arrays_dense_full_track_heldout_and_scoring_only_after_prediction(self):
        graph = metric_graph([(0, 0), (100, 0)], [(0, 1), (1, 0)])
        track = metric_track([(10, 1), (20, 1), (30, 1), (40, 1)])
        model = emission("physics")
        result = decode_arrays(graph, track, model, decoder_configurations()[0], trip_id="valid_a")
        np.testing.assert_array_equal(result.observation_indices, np.arange(4))
        before = result.route.copy()
        metrics = score_decoded_route(graph, track, result, [0])
        np.testing.assert_array_equal(before, result.route)
        for key in ("length_f1", "edge_f1", "proxy_edge_accuracy", "proxy_edge_macro_f1",
                    "route_continuity", "elapsed_seconds", "extra_arc_count", "exact_match"):
            self.assertIn(key, metrics)
        with self.assertRaisesRegex(ValueError, "held out"):
            decode_arrays(graph, track, model, decoder_configurations()[0], trip_id="fit_a")
        changed = DecoderConfiguration("unregistered", replace(decoder_configurations()[0].config, max_candidates=24))
        with self.assertRaisesRegex(ValueError, "menu"):
            decode_arrays(graph, track, model, changed, trip_id="valid_a")

    def test_failed_routes_stay_in_selection_denominator(self):
        records = [{"trip_id": "a", "status": "complete", "metrics": {"length_f1": 1., "edge_f1": 0.5}},
                   {"trip_id": "b", "status": "failed", "error": "synthetic disconnected"}]
        summary = aggregate_route_records(records, ["a", "b"])
        self.assertEqual(summary["mean_length_f1"], 0.5)
        self.assertEqual(summary["mean_edge_f1"], 0.25)
        self.assertEqual(summary["failed_routes"], 1)
        with self.assertRaises(ValueError):
            aggregate_route_records(records[:1], ["a", "b"])
        with self.assertRaises(ValueError):
            aggregate_route_records([records[0], records[0]], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
