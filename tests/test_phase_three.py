"""Synthetic requested-trio regression checks; never real training or live evidence."""
from __future__ import annotations

from dataclasses import fields
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import joblib
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from mapmatch.experiments import CandidateDataset, decode_parameters, encode_parameters
from mapmatch.candidates import CandidateIndex
from mapmatch.data import load_graph, load_track, validate_route
from mapmatch.features import FEATURE_NAMES, FEATURE_VERSION, CandidateFeatureExtractor
from mapmatch.learning import load_emission_model, predict_positive_proba
from mapmatch.matcher import MapMatcher, MatchConfig
from tests.test_core import metric_graph, metric_track
from mapmatch.phase_models import (
    PHASE_DEFAULT_PARAMETERS, PHASE_FAMILIES, PHASE_MODEL_COUNT,
    PHASE_SEARCH_SPACES, bank_seed, knn_reference_bank, make_phase_estimator,
)
from scripts.verify_phase_three_delivery import public_row_checks


class RequestedTrioTests(unittest.TestCase):
    def test_live_activation_requires_actual_three_model_ten_route_api_acceptance(self):
        import copy
        from scripts import activate_phase_three as activation
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(activation,"ROOT",Path(temporary)):
            root=Path(temporary)
            model=root/activation.REPLACEMENTS["PATHSNAP_MODEL"]
            model.parent.mkdir(parents=True)
            model.write_bytes(b"synthetic artifact for validation tests, not a trained model")
            config=model.parent/"match_config.json"
            config.write_text("{}")
            h=hashlib.sha256(model.read_bytes()).hexdigest()
            c=hashlib.sha256(config.read_bytes()).hexdigest()
            documents={
                "reports/phase_three/model_acceptance.json":{"passed":True,"frozen_model_sha256":h,
                    "selected_family":"bagging","selected_model_clean_environment":{"passed":True},
                    "models":[{"family":name,"tested_trip_count":10} for name in PHASE_FAMILIES]},
                "reports/phase_three/offline_delivery_verification.json":{"passed":True,"model_sha256":h,
                    "config_file_sha256":c,"api_compact_config_sha256":"compact"},
                "reports/phase_three/api_integration_acceptance.json":{"passed":True,"trip_count":10,
                    "model_sha256":h,"match_config_hash":"compact","mocked_model_or_inference":False},
                "artifacts/phase_three/model_freeze.json":{"sha256":h},
                "reports/phase_three/decoder/freeze.json":{"model_sha256":h,"config_file_sha256":c},
                "artifacts/phase_three/model_metadata.json":{"model_name":"bagging"}}
            def persist(docs):
                for path,value in docs.items():
                    p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
            persist(documents)
            self.assertEqual(activation.acceptance()["model_sha256"],h)
            mutations=[("reports/phase_three/model_acceptance.json","passed",False),
                       ("reports/phase_three/model_acceptance.json","models",[{"family":"bagging","tested_trip_count":10}]),
                       ("reports/phase_three/model_acceptance.json","selected_model_clean_environment",{"passed":False}),
                       ("reports/phase_three/api_integration_acceptance.json","mocked_model_or_inference",True),
                       ("reports/phase_three/api_integration_acceptance.json","trip_count",9),
                       ("reports/phase_three/api_integration_acceptance.json","match_config_hash","wrong"),
                       ("artifacts/phase_three/model_freeze.json","sha256","wrong"),
                       ("reports/phase_three/decoder/freeze.json","config_file_sha256","wrong"),
                       ("artifacts/phase_three/model_metadata.json","model_name","historical")]
            for path,key,value in mutations:
                with self.subTest(path=path,key=key):
                    altered=copy.deepcopy(documents);altered[path][key]=value;persist(altered)
                    with self.assertRaises(ValueError):activation.acceptance()
                    persist(documents)

    def test_model_acceptance_refuses_cross_environment_version_fixture_or_probability_drift(self):
        from scripts.verify_phase_three_models import cross_environment_probe
        baseline = {"versions":{"python":"test","sklearn":"test"},"feature_count":38,
                    "fixture_sha256":"fixed", "probabilities":[.2,.8]}
        def completed(value):
            return mock.Mock(stdout=json.dumps(value))
        with mock.patch("scripts.verify_phase_three_models.subprocess.run",
                        side_effect=[completed(baseline),completed(baseline)]):
            self.assertTrue(cross_environment_probe(Path("synthetic.joblib"))["passed"])
        for field, value in {"versions":{"python":"other"},"fixture_sha256":"changed",
                             "probabilities":[.21,.8]}.items():
            with self.subTest(field=field), mock.patch("scripts.verify_phase_three_models.subprocess.run",
                    side_effect=[completed(baseline),completed({**baseline,field:value})]):
                with self.assertRaises(AssertionError):
                    cross_environment_probe(Path("synthetic.joblib"))

    def test_activation_changes_only_three_artifact_settings_and_checks_exact_health(self):
        from scripts.activate_phase_three import REPLACEMENTS, health_matches, render_environment
        original = '# local configuration\nPATHSNAP_MODEL="artifacts/model.joblib"\nPATHSNAP_MODEL_METADATA="artifacts/model_metadata.json"\nPATHSNAP_MATCH_CONFIG="artifacts/match_config.json"\nPATHSNAP_DATA_DIR="private dataset"\nPATHSNAP_DATASET_ROLE="development"\n'
        changed = render_environment(original)
        self.assertIn('PATHSNAP_DATA_DIR="private dataset"\n', changed)
        self.assertIn('PATHSNAP_DATASET_ROLE="development"\n', changed)
        self.assertTrue(changed.startswith('# local configuration\n'))
        for key, path in REPLACEMENTS.items():
            self.assertIn(key + '=' + json.dumps(path), changed)
        self.assertEqual(render_environment(changed), changed)
        with self.assertRaises(ValueError):
            render_environment('PATHSNAP_MODEL="incomplete"\n')
        with self.assertRaises(ValueError):
            render_environment(original + 'PATHSNAP_MODEL="duplicate"\n')
        expected = {"model_sha256":"new", "model_name":"bagging", "match_config_hash":"frozen"}
        health = {"status":"ready","model_loaded":True,"model_hash":"new","model_name":"bagging",
                  "match_config_hash":"frozen","trip_count":10,"prediction_cache":False}
        self.assertTrue(health_matches(health, expected))
        for key, value in {"model_hash":"historical", "match_config_hash":"other", "model_name":"historical",
                           "status":"not_ready", "trip_count":9, "prediction_cache":True}.items():
            with self.subTest(field=key):
                self.assertFalse(health_matches({**health,key:value}, expected))

    def test_missing_frozen_config_never_silently_changes_learned_decoder(self):
        from mapmatch import cli
        from backend.config import Settings
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(cli, "PROJECT", Path(temporary)):
                with self.assertRaisesRegex(FileNotFoundError, "frozen requested-trio"):
                    cli._config(None, require_frozen=True)
                # The explicitly requested geometry baseline can use its defaults.
                self.assertIsInstance(cli._config(None, require_frozen=False), MatchConfig)
                config = Path(temporary) / "explicit.json"
                config.write_text("{}")
                self.assertIsInstance(cli._config(str(config), require_frozen=True), MatchConfig)
        with mock.patch.dict("os.environ", {}, clear=True):
            settings = Settings.from_env()
            self.assertTrue(settings.match_config_required)
            self.assertEqual(settings.model_path.parts[-3:], ("artifacts", "phase_three", "model.joblib"))
            self.assertEqual(settings.match_config_path.parts[-3:], ("artifacts", "phase_three", "match_config.json"))

    def test_scope_contains_exactly_requested_three_actual_estimators(self):
        self.assertEqual(PHASE_MODEL_COUNT, 3)
        self.assertEqual(set(PHASE_FAMILIES), {"logistic_regression", "knn_distance_weighted", "bagging"})
        expected = {"logistic_regression": LogisticRegression,
                    "knn_distance_weighted": KNeighborsClassifier, "bagging": BaggingClassifier}
        for name in expected:
            with self.subTest(name=name):
                model = make_phase_estimator(name, PHASE_DEFAULT_PARAMETERS[name], 123, 1)
                if name != "bagging":
                    self.assertIsInstance(model, Pipeline)
                    self.assertIsInstance(model[0], StandardScaler)
                    self.assertFalse(hasattr(model[0], "mean_"))
                    classifier = model[-1]
                else:
                    classifier = model
                    self.assertTrue(classifier.bootstrap)
                    self.assertIsInstance(classifier.estimator, DecisionTreeClassifier)
                    self.assertEqual(classifier.max_features, 1.0)
                self.assertIsInstance(classifier, expected[name])
                if name == "knn_distance_weighted":
                    self.assertEqual(classifier.weights, "distance")
                    self.assertEqual(classifier.algorithm, "brute")
        with self.assertRaises(ValueError):
            make_phase_estimator("hist_gradient_boosting", {}, 123, 1)

    def test_all_three_fit_export_reload_and_run_actual_directed_decoder(self):
        # This is a small synthetic integration test, not a measured student score.
        graph = metric_graph([(0, 0), (100, 0), (200, 0)],
                             [(0, 1), (1, 2), (1, 0), (2, 1)])
        track = metric_track([(10, 1), (40, -1), (80, 0), (120, 1), (160, -1), (190, 0)])
        xy = graph.project(track[:, :2])
        index = CandidateIndex(graph)
        candidates = [index.query(point, 80, 12) for point in xy]
        X, offsets = CandidateFeatureExtractor(graph).transform(track, np.arange(len(track)), candidates)
        targets = np.where(track[:, 0] < graph.nodes[1, 0], 0, 1)
        y = np.concatenate([(candidate.edges == target).astype(np.int8)
                            for candidate, target in zip(candidates, targets)])
        for name in PHASE_FAMILIES:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as temporary:
                parameters = dict(PHASE_DEFAULT_PARAMETERS[name])
                if name == "knn_distance_weighted":
                    parameters["n_neighbors"] = 3
                if name == "bagging":
                    parameters.update(n_estimators=5, max_depth=4, min_samples_leaf=1)
                model = make_phase_estimator(name, parameters, 123, 1)
                kwargs = {} if name == "knn_distance_weighted" else (
                    {"classifier__sample_weight": np.ones(len(y))} if isinstance(model, Pipeline)
                    else {"sample_weight": np.ones(len(y))})
                model.fit(X, y, **kwargs)
                original = predict_positive_proba(model, X)
                self.assertTrue(np.isfinite(original).all())
                self.assertTrue(np.all((original >= 1e-7) & (original <= 1-1e-7)))
                metadata = {"model_name": name, "phase": "synthetic-regression-test"}
                path = Path(temporary) / "model.joblib"
                joblib.dump({"estimator": model, "metadata": metadata,
                             "feature_names": FEATURE_NAMES, "feature_version": FEATURE_VERSION}, path)
                loaded = load_emission_model(path)
                np.testing.assert_array_equal(original, predict_positive_proba(loaded.estimator, X))
                batched = loaded.score_candidates_batch(graph, track, np.arange(len(track)), candidates)
                self.assertEqual(len(batched), len(track))
                for observation, values in enumerate(batched):
                    expected = original[offsets[observation]:offsets[observation+1]]
                    np.testing.assert_allclose(values, expected / expected.sum(), rtol=0, atol=1e-14)
                result = MapMatcher(graph, MatchConfig(max_candidates=12), loaded).match(track)
                self.assertTrue(validate_route(graph, result.route)["valid"])
                self.assertEqual(result.breaks, [])
                self.assertEqual(len(result.matched_edges), len(track))
                self.assertTrue(np.all((result.matched_edges >= 0) & (result.matched_edges < len(graph.arcs))))

    def test_each_real_estimator_serves_fresh_api_predictions_without_reference_files(self):
        # Actual sklearn -> Joblib -> service loader -> audited matcher -> API.
        # TestClient is an in-process transport; this is NOT public-live evidence.
        from fastapi.testclient import TestClient
        from backend.app import create_app
        from backend.config import Settings
        from tests.test_core import write_fixture
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = write_fixture(root / "data", arcs=[(0, 1), (1, 2), (1, 0), (2, 1)])
            graph = load_graph(prefix)
            track = load_track(prefix.with_suffix(".track"))
            index = CandidateIndex(graph)
            candidates = [index.query(point, 80, 12) for point in graph.project(track[:, :2])]
            X, _ = CandidateFeatureExtractor(graph).transform(track, np.arange(len(track)), candidates)
            targets = np.where(track[:, 0] < graph.nodes[1, 0], 0, 1)
            y = np.concatenate([(c.edges == edge).astype(np.int8) for c, edge in zip(candidates, targets)])
            config_path = root / "frozen_config.json"
            config_path.write_text("{}")
            self.assertFalse(prefix.with_suffix(".route").exists())
            for name in PHASE_FAMILIES:
                with self.subTest(model=name):
                    parameters = dict(PHASE_DEFAULT_PARAMETERS[name])
                    if name == "knn_distance_weighted":
                        parameters["n_neighbors"] = 3
                    elif name == "bagging":
                        parameters.update(n_estimators=5, max_depth=4, min_samples_leaf=1)
                    model = make_phase_estimator(name, parameters, 123, 1)
                    model.fit(X, y)
                    model_path = root / (name + ".joblib")
                    joblib.dump({"estimator": model, "feature_names": FEATURE_NAMES,
                                 "feature_version": FEATURE_VERSION,
                                 "metadata": {"model_name": name, "seed": 123}}, model_path)
                    settings = Settings(data_dir=prefix.parent.parent, model_path=model_path,
                                        match_config_path=config_path, match_config_required=True,
                                        expected_trip_count=1, dataset_role="evaluator", frontend_dir=None)
                    with mock.patch("backend.service.log_event"), mock.patch("backend.boundary.log_event"):
                        with TestClient(create_app(settings)) as client:
                            health = client.get("/health").json()
                            self.assertEqual(health["status"], "ready")
                            self.assertEqual(health["model_hash"], hashlib.sha256(model_path.read_bytes()).hexdigest())
                            first = client.post("/match", json={"trip_id": prefix.name})
                            second = client.post("/match", json={"trip_id": prefix.name})
                            self.assertEqual(first.status_code, 200, first.text)
                            self.assertEqual(second.status_code, 200, second.text)
                            a, b = first.json(), second.json()
                            self.assertEqual(a["model"]["name"], name)
                            self.assertEqual(a["diagnostics"]["learned_emission_observations"], len(track))
                            self.assertFalse(a["diagnostics"]["uses_reference_route"])
                            self.assertIsNone(a["metrics"]["accuracy"])
                            self.assertNotEqual(a["request_id"], b["request_id"])
                            self.assertEqual(a["predicted_route"], b["predicted_route"])
                            self.assertNotIn("reference_route", a)
                            self.assertFalse(prefix.with_suffix(".route").exists())

    def test_bank_is_deterministic_unique_bounded_and_preserves_class_proportions(self):
        y = np.r_[np.zeros(80), np.ones(20)]
        weights = np.linspace(.5, 2, 100)
        bank = knn_reference_bank(y, weights, 40, 31)
        np.testing.assert_array_equal(bank, knn_reference_bank(y, weights, 40, 31))
        self.assertEqual(len(bank), len(np.unique(bank)))
        self.assertEqual(len(bank), 40)
        self.assertEqual(int(y[bank].sum()), 8)
        self.assertTrue(np.all((bank >= 0) & (bank < len(y))))
        np.testing.assert_array_equal(knn_reference_bank(y, weights, 200, 31), np.arange(100))
        self.assertEqual(len({bank_seed(20261118, i) for i in range(4)}), 4)

    def test_completed_or_partial_phase_stages_refuse_before_loading_or_fitting(self):
        phase = importlib.import_module("experiments.phase_three.run_phase_three")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase_dir, artifacts = root/"experiments"/"phase_three", root/"artifacts"/"phase_three"
            examples = {"cv": phase_dir/"cv"/"logistic_regression.json",
                        "select": phase_dir/"selection.json", "hpo": phase_dir/"bayesian_trials.jsonl",
                        "freeze": artifacts/"model.joblib", "student": phase_dir/"evaluation"/"bagging.json"}
            with mock.patch.object(phase,"ROOT",root), mock.patch.object(phase,"PHASE_DIR",phase_dir), mock.patch.object(phase,"ARTIFACTS",artifacts), mock.patch.object(phase,"load_train_and_folds",side_effect=AssertionError("must not load data")), mock.patch.object(phase,"load_cached_dataset",side_effect=AssertionError("must not load data")):
                for stage,path in examples.items():
                    with self.subTest(stage=stage):
                        path.parent.mkdir(parents=True,exist_ok=True)
                        path.write_bytes(b"sentinel: never overwrite")
                        with self.assertRaises(FileExistsError):
                            getattr(phase,"stage_"+stage)(phase.phase_config())
                        self.assertEqual(path.read_bytes(),b"sentinel: never overwrite")
                        path.unlink()
                artifact=artifacts/"model_freeze.json"
                artifact.write_text("frozen sentinel")
                for stage in ("cv","select","hpo","freeze"):
                    with self.subTest(frozen_stage=stage), self.assertRaises(FileExistsError):
                        phase.main(["--stage",stage])
                self.assertFalse((phase_dir/"run_config.json").exists())
                self.assertEqual(artifact.read_text(),"frozen sentinel")

    def test_prediction_identity_checks_row_membership_not_just_metric_equality(self):
        from tests.test_experiments import synthetic_dataset
        phase = importlib.import_module("experiments.phase_three.run_phase_three")
        dataset = synthetic_dataset()
        prediction = phase.prediction_arrays(dataset,np.full(len(dataset.y),0.3))
        phase.assert_prediction_identity(dataset,prediction)
        for key in ("y","offsets","observation_trip_ids","observation_indices","candidate_edges","reference_edges"):
            with self.subTest(key=key):
                tampered = dict(prediction)
                tampered[key] = prediction[key].copy()
                if tampered[key].dtype.kind in "US":
                    tampered[key][0] = "wrong-trip"
                else:
                    tampered[key][0] += 1
                with self.assertRaisesRegex(AssertionError,key):
                    phase.assert_prediction_identity(dataset,tampered)

    def test_actual_runner_fits_all_three_with_no_validation_feature_or_label_leakage(self):
        from mapmatch.experiments import ExperimentConfig
        from tests.test_experiments import synthetic_dataset
        phase = importlib.import_module("experiments.phase_three.run_phase_three")
        dataset = synthetic_dataset()
        held_out = np.unique(dataset.observation_trip_ids)[-1]
        training = np.flatnonzero(dataset.row_trip_ids != held_out)
        validation = np.flatnonzero(dataset.row_trip_ids == held_out)
        self.assertFalse(set(dataset.row_trip_ids[training]) & set(dataset.row_trip_ids[validation]))
        self.assertGreater(len(training), 25)
        self.assertGreater(len(validation), 0)
        altered = CandidateDataset(**{field.name:getattr(dataset,field.name) for field in fields(dataset)})
        altered.X = dataset.X.copy()
        altered.y = dataset.y.copy()
        altered.X[validation] += 10_000
        altered.y[validation] = 1-altered.y[validation]
        probe = dataset.X[training[:10]]
        for name in PHASE_FAMILIES:
            with self.subTest(model=name):
                parameters = dict(PHASE_DEFAULT_PARAMETERS[name])
                if name == "bagging":
                    parameters.update(n_estimators=4,max_depth=4,min_samples_leaf=1)
                a,_,_,bank = phase.phase_fit(name,parameters,ExperimentConfig(n_jobs=1),dataset,training,0)
                b,_,_,other_bank = phase.phase_fit(name,parameters,ExperimentConfig(n_jobs=1),altered,training,0)
                np.testing.assert_array_equal(predict_positive_proba(a,probe),predict_positive_proba(b,probe))
                if name != "bagging":
                    np.testing.assert_allclose(a[0].mean_,dataset.X[training].mean(axis=0,dtype=np.float64),atol=1e-9)
                    self.assertEqual(int(a[0].n_samples_seen_),len(training))
                if name == "knn_distance_weighted":
                    self.assertEqual(bank["source_rows"],len(training))
                    self.assertEqual(bank["selected_rows"],len(training))
                    self.assertEqual(bank,other_bank)

    def test_bank_uses_only_explicit_training_slice(self):
        y = np.tile([0, 0, 0, 1], 100)
        weights = np.ones(400)
        training_rows = np.arange(0, 200)
        before = knn_reference_bank(y[training_rows], weights[training_rows], 80, 19)
        y[200:] = 1 - y[200:]
        weights[200:] = 999
        after = knn_reference_bank(y[training_rows], weights[training_rows], 80, 19)
        np.testing.assert_array_equal(before, after)
        self.assertFalse(set(training_rows[after]) & set(range(200, 400)))

    def test_bank_rejects_nonfinite_nonpositive_and_misaligned_weights(self):
        for y, weights in (([], []), ([0, 1], [1]), ([0, 1], [1, 0]),
                           ([0, 1], [1, -1]), ([0, 1], [1, np.nan])):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                knn_reference_bank(np.asarray(y), np.asarray(weights), 2, 1)

    def test_declared_search_spaces_roundtrip_and_stay_in_bounds(self):
        rng = np.random.default_rng(45)
        for name, space in PHASE_SEARCH_SPACES.items():
            base = PHASE_DEFAULT_PARAMETERS[name]
            self.assertEqual(decode_parameters(encode_parameters(base, space), base, space), base)
            for _ in range(20):
                decoded = decode_parameters(rng.uniform(size=len(space)), base, space)
                for key, (_, lower, upper) in space.items():
                    self.assertLessEqual(lower, decoded[key])
                    self.assertLessEqual(decoded[key], upper)

    def test_cache_loader_counts_target_misses_even_when_candidate_sets_nonempty(self):
        phase = importlib.import_module("experiments.phase_three.run_phase_three")
        dataset = CandidateDataset(
            X=np.zeros((4, len(FEATURE_NAMES)), dtype=np.float32),
            y=np.asarray([1, 0, 0, 1], dtype=np.int8), offsets=np.asarray([0, 2, 3, 4]),
            observation_trip_ids=np.asarray(["train-a", "train-a", "train-b"]),
            observation_indices=np.asarray([0, 1, 0]), reference_edges=np.asarray([0, 8, 3]),
            candidate_edges=np.asarray([0, 1, 2, 3]), candidate_fractions=np.zeros(4),
            candidate_distances=np.zeros(4), proxy_distances=np.zeros(3),
            ambiguous=np.zeros(3, dtype=bool), expanded=np.zeros(3, dtype=bool), coverage={})
        self.assertEqual(int((np.diff(dataset.offsets) == 0).sum()), 0)
        with tempfile.TemporaryDirectory() as temp:
            shared = Path(temp)
            (shared / "cache").mkdir()
            fingerprint = "synthetic-fingerprint"
            cache_stem = "train_" + fingerprint[:16]
            cache = shared / "cache" / (cache_stem + ".npz")
            np.savez_compressed(cache, **{f.name: getattr(dataset, f.name) for f in fields(dataset) if f.name != "coverage"})
            cache.with_suffix(".json").write_text(json.dumps({"split": "train", "fingerprint": fingerprint}))
            spec = {"cache": cache_stem, "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                    "observations": 3, "rows": 4, "trips": 2, "misses": 1}
            with mock.patch.object(phase, "SHARED", shared), mock.patch.object(phase, "EXPECTED", {"train": spec}):
                loaded = phase.load_cached_dataset("train")
                np.testing.assert_array_equal(loaded.y, dataset.y)
                spec["misses"] = 0
                with self.assertRaisesRegex(AssertionError, "candidate-miss"):
                    phase.load_cached_dataset("train")

    def test_delivery_parity_rejects_any_recorded_model_geometry_or_state_drift(self):
        expected = {"matched_edge_array_sha256": "states", "predicted_geometry_sha256": "geometry",
                    "observations": 3, "metrics": {"length_f1": .9, "edge_f1": .8}}
        row = {"http_status": 200, "metrics_status": "ok", "model": {"sha256": "model"},
               "match_config_hash": "config", "request_id": "synthetic-request",
               "matched_edge_array_sha256": "states", "predicted_geometry_sha256": "geometry",
               "observation_assignment_count": 3,
               "diagnostics": {"learned_emission_observations": 3, "uses_reference_route": False},
               "metrics": {"route_length_f1": .9, "route_edge_f1": .8}}
        self.assertTrue(all(public_row_checks(row, expected, "model", "config").values()))
        for key in ("matched_edge_array_sha256", "predicted_geometry_sha256", "match_config_hash"):
            with self.subTest(key=key):
                altered = {**row, key: "wrong"}
                self.assertFalse(all(public_row_checks(altered, expected, "model", "config").values()))
        self.assertFalse(all(public_row_checks(row, expected, "different-model", "config").values()))


if __name__ == "__main__":
    unittest.main()
