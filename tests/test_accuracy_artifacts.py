"""Tiny generated-only role tests: no supplied data, caches, or models are read."""
from __future__ import annotations

import copy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from mapmatch import accuracy_artifacts as artifacts
from mapmatch.accuracy_features import ACCURACY_FEATURE_VERSION, transform_accuracy_features
from mapmatch.accuracy_matcher import (
    AccuracyEmissionModel, TRAINING_SCOPE, accuracy_feature_names, load_accuracy_emission_model,
)
from mapmatch.candidates import CandidateIndex, Candidates
from mapmatch.features import CandidateFeatureExtractor, FEATURE_NAMES, FEATURE_VERSION
from tests.test_core import metric_graph, metric_track


TRAIN = [f"synthetic_{i:03d}" for i in range(64)]
VALID = [f"synthetic_{i:03d}" for i in range(64, 80)]
PROTOCOL = "a" * 64


class ProbabilitySpy(ClassifierMixin, BaseEstimator):
    def __init__(self, columns=38, classes=(0, 1), raw=None):
        self.columns = columns
        self.classes = classes
        self.raw = raw
        self.n_features_in_ = columns
        self.classes_ = np.asarray(classes)
        self.calls = []

    def fit(self, X, y):
        raise AssertionError("no fitting of probability spies")

    def predict_proba(self, X):
        self.calls.append(np.asarray(X).copy())
        if self.raw is not None:
            raw = np.asarray(self.raw)
            return np.tile(raw, (len(X), 1)) if raw.ndim == 1 else raw.copy()
        positive = 0.2 + 0.6 / (1.0 + np.abs(X[:, 0]))
        return np.column_stack([positive if c == 1 else 1.0 - positive for c in self.classes_])


class Downcast(TransformerMixin, BaseEstimator):
    def __init__(self, columns=38):
        self.columns = columns
        self.n_features_in_ = columns

    def fit(self, X, y=None):
        raise AssertionError("no fitting of synthetic transformer")

    def transform(self, X):
        return np.asarray(X, dtype=np.float32)


class GaussianSpy(GaussianNB):
    def predict_proba(self, X):
        self.scoring_dtype_ = np.asarray(X).dtype
        return super().predict_proba(X)


def metadata(role="confirmation", profile="v1", **overrides):
    value = {
        "profile": profile, "feature_names": accuracy_feature_names(profile),
        "base_feature_names": FEATURE_NAMES, "base_feature_version": FEATURE_VERSION,
        "feature_version": FEATURE_VERSION if profile == "v1" else ACCURACY_FEATURE_VERSION,
        "family": "synthetic", "run_name": "synthetic_only", "parameters": {},
        "train_trip_ids": TRAIN.copy(), "validation_trip_ids": VALID.copy(),
        "fold": 3, "fold_id": 3, "training_scope": artifacts.CONFIRMATION_TRAINING_SCOPE,
        "source_sha256": {path: "b" * 64 for path in artifacts.REQUIRED_SOURCE_PATHS},
        "protocol_sha256": PROTOCOL,
    }
    if role == "deployment":
        value.update(fit_role=artifacts.DEPLOYMENT_ROLE, train_trip_ids=TRAIN + VALID,
                     validation_trip_ids=[], fold=None, fold_id=None,
                     training_scope=artifacts.DEPLOYMENT_TRAINING_SCOPE)
    value.update(overrides)
    return value


def model(role="confirmation", profile="v1", **spy_options):
    wrapper = (artifacts.AccuracyConfirmationModel if role == "confirmation"
               else artifacts.AccuracyDeploymentModel)
    return wrapper(ProbabilitySpy(len(accuracy_feature_names(profile)), **spy_options),
                   metadata(role, profile))


def dump_bundle(root, role="confirmation", profile="v1", **overrides):
    bundle = metadata(role, profile, **overrides)
    bundle["estimator"] = ProbabilitySpy(len(accuracy_feature_names(profile)))
    path = Path(root) / "tiny_synthetic.joblib"
    joblib.dump(bundle, path, compress=3)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def load_confirmation(path, digest, **overrides):
    kwargs = dict(expected_sha256=digest, expected_train_trip_ids=TRAIN,
                  expected_validation_trip_ids=VALID, expected_protocol_sha256=PROTOCOL)
    kwargs.update(overrides)
    return artifacts.load_accuracy_confirmation_model(path, **kwargs)


class ArtifactRoleTests(unittest.TestCase):
    def test_existing_confirmation_envelope_needs_no_new_role_or_fake_fold(self):
        with tempfile.TemporaryDirectory() as root:
            path, digest = dump_bundle(root)
            result = load_confirmation(path, digest, expected_train_trip_ids=TRAIN[::-1],
                                       expected_validation_trip_ids=VALID[::-1])
        self.assertIsInstance(result, AccuracyEmissionModel)
        self.assertIsInstance(result, artifacts.AccuracyConfirmationModel)
        self.assertEqual(result.metadata["fold_id"], 3)
        self.assertEqual(result.metadata["fold"], 3)
        self.assertNotIn("fit_role", result.metadata)
        self.assertEqual(result.metadata["training_scope"], artifacts.CONFIRMATION_TRAINING_SCOPE)
        self.assertEqual(result.metadata["train_trip_ids"], tuple(TRAIN))
        self.assertEqual(result.metadata["validation_trip_ids"], tuple(VALID))
        self.assertEqual(result.estimator.calls, [])

    def test_explicit_confirmation_role_and_optional_fold_alias_remain_truthful(self):
        value = metadata(fit_role=artifacts.CONFIRMATION_ROLE)
        value.pop("fold")
        result = artifacts.AccuracyConfirmationModel(ProbabilitySpy(), value)
        self.assertNotIn("fold", result.metadata)
        self.assertEqual(result.metadata["fit_role"], artifacts.CONFIRMATION_ROLE)
        self.assertEqual(value["train_trip_ids"], TRAIN)

    def test_deployment_load_accepts_missing_or_none_folds_and_optional_public_metadata(self):
        for absent in (False, True):
            with self.subTest(absent=absent), tempfile.TemporaryDirectory() as root:
                value = metadata("deployment", public_name="synthetic_public", seed=37)
                if absent:
                    value.pop("fold_id")
                    value.pop("fold")
                value["estimator"] = ProbabilitySpy()
                path = Path(root) / "tiny_synthetic.joblib"
                joblib.dump(value, path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                result = artifacts.load_accuracy_deployment_model(path, expected_sha256=digest)
                pinned = artifacts.load_accuracy_deployment_model(
                    path, expected_sha256=digest, expected_protocol_sha256=PROTOCOL)
            self.assertIsInstance(result, artifacts.AccuracyDeploymentModel)
            self.assertEqual(result.metadata["train_trip_ids"], tuple(TRAIN + VALID))
            self.assertEqual(result.metadata["validation_trip_ids"], ())
            self.assertEqual(result.metadata["public_name"], "synthetic_public")
            self.assertEqual(result.metadata["seed"], 37)
            self.assertEqual(pinned.metadata, result.metadata)
            self.assertEqual("fold_id" in result.metadata, not absent)
            self.assertEqual("fold" in result.metadata, not absent)
            if not absent:
                self.assertIsNone(result.metadata["fold_id"])
                self.assertIsNone(result.metadata["fold"])

    def test_confirmation_rejects_nonconfirmation_roles_folds_and_partitions(self):
        invalid = [
            {"fit_role": artifacts.DEPLOYMENT_ROLE}, {"fit_role": None},
            {"training_scope": TRAINING_SCOPE}, {"training_scope": "historically unseen blind test"},
            {"fold_id": 0, "fold": 0}, {"fold_id": 3.0}, {"fold": True},
            {"fold_id": None}, {"fold": 2}, {"fold_id": np.bool_(True)},
            {"train_trip_ids": TRAIN[:-1]}, {"validation_trip_ids": VALID[:-1]},
            {"train_trip_ids": TRAIN + ["extra"]}, {"validation_trip_ids": VALID + ["extra"]},
            {"train_trip_ids": TRAIN[:-1] + [TRAIN[0]]},
            {"validation_trip_ids": VALID[:-1] + [VALID[0]]},
            {"validation_trip_ids": VALID[:-1] + [TRAIN[0]]},
            {"train_trip_ids": TRAIN[:-1] + ["../invalid"]},
            {"validation_trip_ids": None}, {"train_trip_ids": set(TRAIN)},
        ]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                artifacts.AccuracyConfirmationModel(ProbabilitySpy(), metadata(**change))
        value = metadata()
        value.pop("fold_id")
        with self.assertRaises(ValueError):
            artifacts.AccuracyConfirmationModel(ProbabilitySpy(), value)

    def test_deployment_rejects_other_roles_any_fold_and_nonempty_validation(self):
        invalid = [
            {"fit_role": artifacts.CONFIRMATION_ROLE}, {"fit_role": None},
            {"training_scope": artifacts.CONFIRMATION_TRAINING_SCOPE},
            {"fold_id": 0}, {"fold": 3}, {"fold": False}, {"fold_id": "None"},
            {"validation_trip_ids": ["reserved"]}, {"validation_trip_ids": None},
            {"train_trip_ids": TRAIN}, {"train_trip_ids": TRAIN + VALID + ["extra"]},
            {"train_trip_ids": TRAIN + VALID[:-1] + [TRAIN[0]]},
        ]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                artifacts.AccuracyDeploymentModel(ProbabilitySpy(), metadata("deployment", **change))
        value = metadata("deployment")
        value.pop("fit_role")
        with self.assertRaises(ValueError):
            artifacts.AccuracyDeploymentModel(ProbabilitySpy(), value)

    def test_schema_profile_parameters_protocol_and_source_pins_are_strict(self):
        sources = metadata()["source_sha256"]
        missing_source = dict(sources)
        missing_source.pop("scripts/accuracy_confirmation.py")
        invalid = [
            {"profile": "baseline38"}, {"profile": None},
            {"feature_names": FEATURE_NAMES[::-1]}, {"feature_names": None},
            {"base_feature_names": FEATURE_NAMES[::-1]}, {"base_feature_names": None},
            {"feature_version": ACCURACY_FEATURE_VERSION}, {"base_feature_version": "v2"},
            {"parameters": []}, {"family": "../bad"}, {"run_name": ""},
            {"protocol_sha256": None}, {"protocol_sha256": "A" * 64},
            {"source_sha256": {}}, {"source_sha256": missing_source},
            {"source_sha256": {"common": "b" * 64, "factory_module": "b" * 64}},
            {"source_sha256": dict(sources, **{"mapmatch/accuracy_features.py": "bad"})},
        ]
        for bad_path in ("/absolute/file.py", "../outside.py", "scripts/../file.py",
                         "scripts//file.py", "scripts/./file.py", "common", "C:/file.py",
                         "scripts\\file.py", "./scripts/extra.py", "scripts/extra.py/"):
            invalid.append({"source_sha256": dict(sources, **{bad_path: "b" * 64})})
        for role, wrapper in (("confirmation", artifacts.AccuracyConfirmationModel),
                              ("deployment", artifacts.AccuracyDeploymentModel)):
            for change in invalid:
                with self.subTest(role=role, change=change), self.assertRaises(ValueError):
                    wrapper(ProbabilitySpy(), metadata(role, **change))
            with self.assertRaises(ValueError):
                wrapper(ProbabilitySpy(), None)
        value = metadata()
        value["source_sha256"]["mapmatch/accuracy_artifacts.py"] = "c" * 64
        artifacts.AccuracyConfirmationModel(ProbabilitySpy(), value)

    def test_fitted_feature_count_and_binary_classes_are_required(self):
        bad_estimators = [GaussianNB(), object(), ProbabilitySpy(37), ProbabilitySpy(38.0),
                          ProbabilitySpy(classes=(0,)), ProbabilitySpy(classes=(0, 2)),
                          ProbabilitySpy(classes=("0", "1")), ProbabilitySpy(classes=(0, 0))]
        for wrapper, role in ((artifacts.AccuracyConfirmationModel, "confirmation"),
                              (artifacts.AccuracyDeploymentModel, "deployment")):
            for estimator in bad_estimators:
                with self.subTest(role=role, estimator=estimator), self.assertRaises(ValueError):
                    wrapper(estimator, metadata(role))

    def test_hash_mismatch_prevents_any_unpickle_for_both_loaders(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "not_a_pickle.joblib"
            path.write_bytes(b"tiny generated invalid pickle")
            with mock.patch.object(artifacts.joblib, "load") as unpickle:
                with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                    load_confirmation(path, "0" * 64)
                with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                    artifacts.load_accuracy_deployment_model(path, expected_sha256="0" * 64)
                unpickle.assert_not_called()

    def test_invalid_expected_identity_rejected_before_open_or_unpickle(self):
        invalid = [
            {"expected_sha256": None}, {"expected_sha256": "A" * 64},
            {"expected_protocol_sha256": None}, {"expected_protocol_sha256": "bad"},
            {"expected_train_trip_ids": TRAIN[:-1]}, {"expected_validation_trip_ids": []},
            {"expected_train_trip_ids": TRAIN[:-1] + [TRAIN[0]]},
            {"expected_validation_trip_ids": VALID[:-1] + [TRAIN[0]]},
        ]
        with mock.patch.object(Path, "open", side_effect=AssertionError("must not open")):
            for change in invalid:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_confirmation("does_not_exist", "0" * 64, **change)
            for digest, protocol in (("bad", None), (None, None), ("0" * 64, "bad")):
                with self.subTest(digest=digest, protocol=protocol), self.assertRaises(ValueError):
                    artifacts.load_accuracy_deployment_model(
                        "does_not_exist", expected_sha256=digest, expected_protocol_sha256=protocol)

    def test_unpickle_receives_exact_hashed_snapshot_despite_path_replacement(self):
        for role in ("confirmation", "deployment"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as root:
                path, digest = dump_bundle(root, role)
                original = path.read_bytes()
                unpickle = joblib.load

                def replace_path_and_load(stream):
                    self.assertIsInstance(stream, io.BytesIO)
                    self.assertEqual(stream.getvalue(), original)
                    path.write_bytes(b"different bytes after hash")
                    return unpickle(stream)

                with mock.patch.object(Path, "open", autospec=True, wraps=Path.open) as opened:
                    # Use the real opener so the spy does not supply MagicMock bytes.
                    opened.side_effect = lambda p, *args, **kwargs: io.open(p, *args, **kwargs)
                    with mock.patch.object(artifacts.joblib, "load", side_effect=replace_path_and_load):
                        if role == "confirmation":
                            result = load_confirmation(path, digest)
                        else:
                            result = artifacts.load_accuracy_deployment_model(path, expected_sha256=digest)
                    reads = [call for call in opened.call_args_list if call.args[1:] == ("rb",)]
                    self.assertEqual(len(reads), 1)
                self.assertEqual(result.metadata["protocol_sha256"], PROTOCOL)
                self.assertNotEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_partition_and_protocol_must_match_expected_not_just_well_formed(self):
        with tempfile.TemporaryDirectory() as root:
            path, digest = dump_bundle(root)
            with self.assertRaisesRegex(ValueError, "partition"):
                load_confirmation(path, digest, expected_train_trip_ids=TRAIN[:-1] + ["replacement"])
            with self.assertRaisesRegex(ValueError, "partition"):
                load_confirmation(path, digest, expected_validation_trip_ids=VALID[:-1] + ["replacement"])
            with self.assertRaisesRegex(ValueError, "protocol pin differs"):
                load_confirmation(path, digest, expected_protocol_sha256="c" * 64)
            path, digest = dump_bundle(root, "deployment")
            with self.assertRaisesRegex(ValueError, "protocol pin differs"):
                artifacts.load_accuracy_deployment_model(
                    path, expected_sha256=digest, expected_protocol_sha256="c" * 64)

    def test_wrong_envelope_and_cross_role_loads_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for value in ({}, {"model": ProbabilitySpy()}, [ProbabilitySpy()]):
                path = Path(root) / "tiny_synthetic.joblib"
                joblib.dump(value, path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_confirmation(path, digest)
                with self.assertRaises(ValueError):
                    artifacts.load_accuracy_deployment_model(path, expected_sha256=digest)
            path, digest = dump_bundle(root, "deployment")
            with self.assertRaises(ValueError):
                load_confirmation(path, digest)
            path, digest = dump_bundle(root)
            with self.assertRaises(ValueError):
                artifacts.load_accuracy_deployment_model(path, expected_sha256=digest)

    def test_oof_loader_is_not_weakened_for_either_new_role(self):
        for role in ("confirmation", "deployment"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as root:
                path, digest = dump_bundle(root, role)
                with self.assertRaises(ValueError):
                    load_accuracy_emission_model(
                        path, expected_sha256=digest, expected_fold_id=0,
                        expected_train_trip_ids=TRAIN, expected_validation_trip_ids=VALID)
                with self.assertRaisesRegex(ValueError, "0, 1, or 2"):
                    load_accuracy_emission_model(
                        path, expected_sha256=digest, expected_fold_id=3,
                        expected_train_trip_ids=TRAIN, expected_validation_trip_ids=VALID)
                with self.assertRaises(ValueError):
                    AccuracyEmissionModel(ProbabilitySpy(), metadata(role))


class ArtifactInferenceTests(unittest.TestCase):
    def setUp(self):
        self.graph = metric_graph([(0, 0), (100, 0)], [(0, 1), (1, 0)])
        self.track = metric_track([(20, 1), (25, 1), (21, 1), (30, 1)])
        self.indices = np.array([0, 2, 3])
        index = CandidateIndex(self.graph)
        self.groups = [index.query(point, 80, 12)
                       for point in self.graph.project(self.track[self.indices, :2])]
        first = self.groups[0]
        self.groups[0] = Candidates(first.edges[::-1], first.fractions[::-1], first.distances[::-1])
        second = self.groups[1]
        self.groups[1] = Candidates(second.edges[:1], second.fractions[:1], second.distances[:1])

    def test_api_single_group_adapter_preserves_fulltrack_batch_features(self):
        for role in ("confirmation","deployment"):
            for profile in ("v1","context","compact","physics"):
                with self.subTest(role=role,profile=profile):
                    result=model(role,profile);group=self.groups[0];index=int(self.indices[0])
                    single=result.score_candidates(self.graph,self.track,None,index,group)
                    batch=result.score_candidates_batch(self.graph,self.track,np.array([index]),[group])[0]
                    np.testing.assert_array_equal(single,batch)

    def test_wrappers_inherit_batch_and_probability_guards_without_override(self):
        for wrapper in (artifacts.AccuracyConfirmationModel, artifacts.AccuracyDeploymentModel):
            self.assertIs(wrapper.score_candidates_batch, AccuracyEmissionModel.score_candidates_batch)
            self.assertIs(wrapper.predict_proba, AccuracyEmissionModel.predict_proba)

    def test_both_roles_all_profiles_extract_38_then_transform_groups_once(self):
        base, offsets = CandidateFeatureExtractor(self.graph).transform(self.track, self.indices, self.groups)
        self.assertEqual(base.shape[1], 38)
        for role in ("confirmation", "deployment"):
            for profile in ("v1", "context", "compact", "physics"):
                with self.subTest(role=role, profile=profile):
                    result = model(role, profile, classes=(1, 0))
                    expected = (np.asarray(base, np.float64) if profile == "v1"
                                else transform_accuracy_features(base, offsets, profile))
                    if profile == "v1":
                        scores = result.score_candidates_batch(self.graph, self.track, self.indices, self.groups)
                    else:
                        with mock.patch("mapmatch.accuracy_features.transform_accuracy_features",
                                        wraps=transform_accuracy_features) as transform:
                            scores = result.score_candidates_batch(self.graph, self.track, self.indices, self.groups)
                        transform.assert_called_once()
                        np.testing.assert_array_equal(transform.call_args.args[0], base)
                        np.testing.assert_array_equal(transform.call_args.args[1], offsets)
                    self.assertEqual(len(result.estimator.calls), 1)
                    self.assertEqual(result.estimator.calls[0].dtype, np.dtype(np.float64))
                    np.testing.assert_array_equal(result.estimator.calls[0], expected)
                    positive = 0.2 + 0.6 / (1 + np.abs(expected[:, 0]))
                    for obs, group, lo, hi, score in zip(
                            self.indices, self.groups, offsets[:-1], offsets[1:], scores):
                        np.testing.assert_allclose(score, positive[lo:hi] / positive[lo:hi].sum(), atol=1e-15)
                        np.testing.assert_array_equal(result.records[int(obs)].edges, group.edges)
                        np.testing.assert_array_equal(result.records[int(obs)].positive_probabilities, positive[lo:hi])
                    self.assertEqual(set(result.records), {0, 2, 3})
                    self.assertFalse(result.probability_checks["uses_reference_route"])

    def test_predict_proba_takes_already_transformed_features_without_retransforming(self):
        for role in ("confirmation", "deployment"):
            for profile in ("v1", "context", "compact", "physics"):
                with self.subTest(role=role, profile=profile):
                    result = model(role, profile)
                    X = np.ones((2, len(result.feature_names)), dtype=np.float32)
                    with mock.patch("mapmatch.accuracy_features.transform_accuracy_features",
                                    side_effect=AssertionError("already transformed")):
                        probability = result.predict_proba(X)
                    self.assertEqual(probability.dtype, np.dtype(np.float64))
                    np.testing.assert_array_equal(result.estimator.calls[0], X.astype(np.float64))
                    if profile != "v1":
                        with self.assertRaises(ValueError):
                            result.predict_proba(np.ones((2, 38)))
                    for bad in (np.full((2, X.shape[1]), np.nan), np.ones(X.shape[1])):
                        with self.assertRaises(ValueError):
                            result.predict_proba(bad)

    def test_both_probability_columns_checked_before_any_clip(self):
        invalid = [np.array([0.8, 0.3]), np.array([0.8, 0.200000001]),
                   np.array([np.nan, 0.5]), np.array([np.inf, 0.0]),
                   np.array([-0.1, 1.1]), np.array([0., 0.]),
                   np.array([0.4, 0.6], np.float32), np.array([[0.4, 0.6]])]
        for role in ("confirmation", "deployment"):
            for raw in invalid:
                with self.subTest(role=role, raw=raw), self.assertRaisesRegex(ValueError, "probabilities"):
                    model(role, raw=raw).score_candidates_batch(
                        self.graph, self.track, self.indices, self.groups)
            result = model(role, raw=np.array([1., 0.], np.float64))
            scores = result.score_candidates_batch(self.graph, self.track, self.indices, self.groups)
            np.testing.assert_array_equal(result.records[0].positive_probabilities, [0., 0.])
            np.testing.assert_allclose(scores[0], [.5, .5])
            np.testing.assert_allclose(scores[1], [1.])
            self.assertEqual(result.probability_checks["clipped_low_rows"], 5)

    def test_pipeline_final_boundary_uses_float64_with_tiny_generated_fitted_estimator(self):
        # The only real fit in this file is six completely generated rows.
        X = np.arange(228, dtype=np.float64).reshape(6, 38) / 100
        fitted = GaussianSpy().fit(X, [0, 0, 0, 1, 1, 1])
        for role, wrapper in (("confirmation", artifacts.AccuracyConfirmationModel),
                              ("deployment", artifacts.AccuracyDeploymentModel)):
            with self.subTest(role=role):
                estimator = copy.deepcopy(fitted)
                pipeline = Pipeline([("downcast", Downcast()), ("classifier", estimator)])
                result = wrapper(pipeline, metadata(role))
                probability = result.predict_proba(X.astype(np.float32))
                self.assertEqual(estimator.scoring_dtype_, np.dtype(np.float64))
                np.testing.assert_allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
