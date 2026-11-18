"""UNIT TESTS ONLY: mock estimators and temporary release/source fixtures.

No dataset discovery, real model loading, prediction, training, HTTP or service
startup occurs. These checks are not a measured target96 acceptance claim.
"""
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backend import target96_runtime as r


class _FixtureContextModel:
    """A mock score transport, not a learned model or inference result."""

    def __init__(self, metadata):
        self.metadata = metadata
        self.estimator = SimpleNamespace(classes_=(0, 1))
        self.feature_names = ("unit_fixture_feature",)
        self.records = {}
        self.probability_checks = {}

    def score_candidates(self, *args):
        raise AssertionError("unit adapter must use the fulltrack batch interface")

    def score_candidates_batch(self, graph, track, indices, groups):
        self.records[indices[0]] = (graph, track)
        self.probability_checks["fixture"] = True
        return [[1.0] for _ in groups]


class Target96RuntimeUnitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="target96-unit-")
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.root = self.workspace / "artifacts" / "target96"
        self.root.mkdir(parents=True)
        self.path = self.root / "release_manifest.json"
        self.config = asdict(r.MatchConfig())
        self.policy = asdict(r.ArcPolicy(appended_arc_penalty=.7))
        self.metadata = {
            "fit_role": "all_train_deployment",
            "training_scope": "all80 TRAIN final refit after frozen selection; no student/evaluator fitting",
            "profile": "context", "protocol_sha256": "a" * 64,
            "train_trip_ids": [f"train_{i:02d}" for i in range(80)],
            "validation_trip_ids": [],
        }
        (self.root / "model.joblib").write_bytes(b"UNIT FIXTURE: NEVER UNPICKLE")
        self.dump("training_metadata.json", self.metadata)
        self.dump("match_config.json", self.config)
        self.dump("arc_policy.json", self.policy)
        sources = {}
        for name in r._SOURCES:
            source = self.workspace / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("# temporary UNIT TEST source pin fixture\n", encoding="utf-8")
            sources[name] = r.sha256(source)
        self.manifest = {
            "schema": "pathsnap-route96-frozen-release-v1", "profile": "context",
            "files": {name: r.sha256(self.root / name) for name in r._FILES},
            "source_sha256": sources, "model_sha256": r.sha256(self.root / "model.joblib"),
            "training_protocol_sha256": "a" * 64, "route_protocol_sha256": "b" * 64,
            "frozen_selection_sha256": "c" * 64, "baseline_model_sha256": "d" * 64,
            "baseline_config_sha256": "e" * 64, "training_fit_reused": True,
            "student_role": "historicallyexposeddevelopment,notblind",
            "evaluator_role": "never used for training/selection",
            "retraining_or_selection_after_student_allowed": False,
            "configuration": {"name": "unit_fixture", "profile": "context",
                              "match_config": self.config, "arc_policy": self.policy},
        }
        self.receipt = {
            "accepted": True, "target_met": True, "observed_edge_f1": .96,
            "failed_trips": 0, "route_continuity": 1., "mean_length_f1": .97,
            "comparison_sha256": "f" * 64, "not_official_or_blind": True,
            "production_switched": False,
        }
        self.persist()
        self.addCleanup(patch.stopall)
        patch.object(r, "ROOT", self.workspace).start()
        self.context_loader = patch("mapmatch.accuracy_artifacts.load_accuracy_deployment_model",
                                    side_effect=AssertionError("real model load forbidden in unit tests")).start()
        self.v1_loader = patch("mapmatch.learning.load_emission_model",
                               side_effect=AssertionError("real model load forbidden in unit tests")).start()

    def dump(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def persist(self):
        self.dump("release_manifest.json", self.manifest)
        self.receipt["release_manifest_sha256"] = r.sha256(self.path)
        self.dump("acceptance.json", self.receipt)

    def repin(self, name, value):
        self.dump(name, value)
        self.manifest["files"][name] = r.sha256(self.root / name)
        self.persist()

    def test_valid_receipt_returns_lazy_components_without_loading_or_startup(self):
        with patch.object(r, "InferenceRuntime") as runtime:
            manifest, loader, factory = r.release_components(self.path)
        self.assertEqual(manifest, self.manifest)
        self.assertTrue(callable(loader) and callable(factory))
        self.context_loader.assert_not_called()
        self.v1_loader.assert_not_called()
        runtime.assert_not_called()

    def test_gate_fields_fail_closed_even_with_accepted_true(self):
        cases = [("accepted", False), ("accepted", 1), ("target_met", False),
                 ("observed_edge_f1", .959999), ("observed_edge_f1", True),
                 ("observed_edge_f1", "0.99"), ("observed_edge_f1", 1.1),
                 ("failed_trips", 1), ("failed_trips", False), ("failed_trips", 0.),
                 ("route_continuity", .999), ("route_continuity", True),
                 ("mean_length_f1", None), ("not_official_or_blind", False),
                 ("comparison_sha256", "not-a-hash")]
        original = dict(self.receipt)
        for key, value in cases:
            with self.subTest(key=key, value=value):
                self.receipt = dict(original, **{key: value})
                self.persist()
                with self.assertRaises(ValueError):
                    r.release_components(self.path)
        self.context_loader.assert_not_called()

    def test_missing_or_wrong_manifest_bound_receipt_rejected(self):
        (self.root / "acceptance.json").rename(self.root / "saved_receipt.json")
        with self.assertRaises(ValueError):
            r.release_components(self.path)
        self.dump("acceptance.json", dict(self.receipt, release_manifest_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "not accepted"):
            r.release_components(self.path)

    def test_nonfinite_and_duplicate_receipt_json_rejected(self):
        for text in ('{"accepted": true, "accepted": false}', '{"observed_edge_f1": NaN}'):
            with self.subTest(text=text):
                (self.root / "acceptance.json").write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    r.release_components(self.path)

    def test_artifact_allowlist_rejects_missing_and_traversal(self):
        original = dict(self.manifest["files"])
        for files in ({k: v for k, v in original.items() if k != "arc_policy.json"},
                      dict(original, **{"../private": "a" * 64}),
                      dict(original, **{"other.json": "a" * 64})):
            with self.subTest(files=list(files)):
                self.manifest["files"] = files
                self.persist()
                with self.assertRaisesRegex(ValueError, "allowlist"):
                    r.release_components(self.path)

    def test_optional_report_metrics_are_hash_pinned_but_not_inference_input(self):
        self.repin("training_metrics.json", {"unit_fixture_report_only": True})
        r.release_components(self.path)
        self.dump("training_metrics.json", {"drift": True})
        with self.assertRaisesRegex(ValueError, "hash differs"):
            r.release_components(self.path)

    def test_artifact_source_and_model_alias_pins_fail_before_pickle(self):
        _, loader, _ = r.release_components(self.path)
        source = self.workspace / "mapmatch" / "target96_arc.py"
        source.write_text("# changed unit fixture\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            loader(self.root / "model.joblib")
        self.context_loader.assert_not_called()
        self.manifest["source_sha256"]["mapmatch/target96_arc.py"] = r.sha256(source)
        self.manifest["model_sha256"] = "0" * 64
        self.persist()
        with self.assertRaisesRegex(ValueError, "aliases"):
            r.release_components(self.path)

    def test_file_mutation_after_component_creation_rejected_before_pickle(self):
        _, loader, _ = r.release_components(self.path)
        (self.root / "model.joblib").write_bytes(b"changed unit fixture")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            loader(self.root / "model.joblib")
        self.context_loader.assert_not_called()

    def test_source_paths_must_be_scoped_and_required(self):
        self.manifest["source_sha256"]["../../private.py"] = "a" * 64
        self.persist()
        with self.assertRaisesRegex(ValueError, "source path"):
            r.release_components(self.path)
        self.manifest["source_sha256"] = {}
        self.persist()
        with self.assertRaisesRegex(ValueError, "source pins"):
            r.release_components(self.path)

    def test_symlink_artifact_and_manifest_rejected(self):
        model = self.root / "model.joblib"
        original = self.root / "unit_model_fixture"
        model.rename(original)
        model.symlink_to(original)
        with self.assertRaisesRegex(ValueError, "regular local"):
            r.release_components(self.path)
        link = self.root / "linked_manifest.json"
        link.symlink_to(self.path)
        with self.assertRaisesRegex(ValueError, "symlink"):
            r.release_components(link)

    def test_roles_reject_fold_confirmation_and_test_selection_claims(self):
        self.metadata["fit_role"] = "internal_confirmation16"
        self.repin("training_metadata.json", self.metadata)
        with self.assertRaisesRegex(ValueError, "deployment role"):
            r.release_components(self.path)
        self.metadata["fit_role"] = "all_train_deployment"
        self.metadata["fold_id"] = 3
        self.repin("training_metadata.json", self.metadata)
        with self.assertRaisesRegex(ValueError, "no fold"):
            r.release_components(self.path)
        self.manifest["retraining_or_selection_after_student_allowed"] = True
        self.persist()
        with self.assertRaisesRegex(ValueError, "roles"):
            r.release_components(self.path)

    def test_loader_pins_context_deployment_and_stays_graph_track_stateless(self):
        model = _FixtureContextModel(self.metadata)
        self.context_loader.side_effect = None
        self.context_loader.return_value = model
        _, loader, _ = r.release_components(self.path)
        emission = loader(self.root / "model.joblib")
        self.context_loader.assert_called_once_with(
            self.root / "model.joblib", expected_sha256=self.manifest["model_sha256"],
            expected_protocol_sha256=self.manifest["training_protocol_sha256"])
        for _ in range(2):
            graph, track = object(), object()
            self.assertEqual(emission.score_candidates_batch(graph, track, [0], [object()]), [[1.]])
            self.assertEqual(emission.score_candidates(graph, track, None, 0, object()), [1.])
        self.assertEqual(model.records, {})
        self.assertEqual(model.probability_checks, {})
        self.assertEqual(emission._template.records, {})
        self.assertEqual(emission._template.probability_checks, {})
        self.assertFalse(hasattr(emission, "records"))
        self.assertFalse(hasattr(emission, "graph"))
        self.assertFalse(hasattr(emission, "track"))
        self.v1_loader.assert_not_called()

    def test_loaded_context_role_cannot_substitute_confirmation_model(self):
        metadata = dict(self.metadata, fit_role="internal_confirmation16")
        self.context_loader.side_effect = None
        self.context_loader.return_value = _FixtureContextModel(metadata)
        _, loader, _ = r.release_components(self.path)
        with self.assertRaisesRegex(ValueError, "deployment role"):
            loader(self.root / "model.joblib")

    def test_v1_loader_is_explicit_and_requires_all80_baseline_identity(self):
        self.manifest["profile"] = self.manifest["configuration"]["profile"] = "v1"
        self.manifest["baseline_model_sha256"] = self.manifest["model_sha256"]
        metadata = {"training_trip_count": 80, "training_trip_ids": self.metadata["train_trip_ids"]}
        self.repin("training_metadata.json", metadata)
        model = _FixtureContextModel(metadata)
        self.v1_loader.side_effect = None
        self.v1_loader.return_value = model
        _, loader, _ = r.release_components(self.path)
        self.assertIs(loader(self.root / "model.joblib"), model)
        self.v1_loader.assert_called_once_with(self.root / "model.joblib")
        self.context_loader.assert_not_called()
        self.manifest["baseline_model_sha256"] = "0" * 64
        self.persist()
        with self.assertRaisesRegex(ValueError, "baseline deployment"):
            r.release_components(self.path)

    def test_factory_accepts_service_signature_preserves_audit_and_owns_caches(self):
        _, _, factory = r.release_components(self.path)
        graph, audit = object(), SimpleNamespace(score_candidates=lambda *args: None, records={})
        indexes, routers, calls = [], [], []

        def index_factory(*args):
            index = object()
            indexes.append(index)
            return index

        def router_factory(*args):
            router = object()
            routers.append(router)
            return router

        def matcher_factory(*args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace()

        with patch.object(r, "CandidateIndex", side_effect=index_factory), \
                patch.object(r, "HopRouter", side_effect=router_factory), \
                patch.object(r, "ArcPriorMatcher", side_effect=matcher_factory):
            factory(graph, r.MatchConfig(**self.config), emission_model=audit)
            factory(graph, r.MatchConfig(**self.config), emission_model=audit)
        self.assertIsNot(calls[0][1]["shared_index"], calls[1][1]["shared_index"])
        self.assertIsNot(routers[0], routers[1])
        for args, kwargs in calls:
            self.assertIs(args[2], audit)
            self.assertEqual(asdict(args[3]), self.policy)
            self.assertIsInstance(kwargs["shared_index"], r.MemoIndex)
            self.assertNotIn("transition_cache", kwargs)
        with self.assertRaisesRegex(ValueError, "decoder differs"):
            factory(graph, replace(r.MatchConfig(), beta_m=99), emission_model=audit)
        with self.assertRaises(ValueError):
            factory(graph, r.MatchConfig(), emission_model=None)

    def test_memo_index_retains_candidate_identity_across_retry_queries(self):
        class UnitFixtureIndex:
            calls = 0

            def query(self, point, radius, count):
                self.calls += 1
                return object()

        index = UnitFixtureIndex()
        memo = r.MemoIndex(index)
        first = memo.query([1., 2.], 80., 24)
        self.assertIs(memo.query([1., 2.], 80., 24), first)
        self.assertIsNot(memo.query([1., 2.], 80., 48), first)
        self.assertEqual(index.calls, 2)
        self.assertIs(memo.query([1., 2.], 80., 24), first)

    def test_disabled_classifier_or_sparse_config_rejected(self):
        for key, value in (("classifier_weight", 0), ("observation_distance_m", 10)):
            with self.subTest(key=key):
                config = dict(self.config, **{key: value})
                self.manifest["configuration"]["match_config"] = config
                self.repin("match_config.json", config)
                with self.assertRaisesRegex(ValueError, "every observation"):
                    r.release_components(self.path)

    def test_app_construction_is_opt_in_and_does_not_start_runtime(self):
        default_model = r.Settings().model_path
        base = r.Settings(data_dir=self.workspace / "unopened_unit_dataset", frontend_dir=None)
        with patch.object(r.Settings, "from_env", return_value=base), \
                patch.object(r, "InferenceRuntime") as runtime, \
                patch.object(r, "create_app", return_value="unit_app") as app:
            self.assertEqual(r.create_target96_app(manifest_path=self.path), "unit_app")
            settings = runtime.call_args.args[0]
            self.assertEqual(settings.model_path, self.root / "model.joblib")
            self.assertEqual(settings.metadata_path, self.root / "training_metadata.json")
            self.assertTrue(settings.match_config_required)
            runtime.return_value.start.assert_not_called()
            app.assert_called_once_with(settings, runtime=runtime.return_value)
        self.assertEqual(r.Settings().model_path, default_model)
        self.context_loader.assert_not_called()
        with self.assertRaisesRegex(ValueError, "artifact paths"):
            r.create_target96_app(settings=base, manifest_path=self.path)
        with self.assertRaises(TypeError):
            r.create_target96_app(manifest_path=self.path, require_acceptance=False)

    def test_import_has_no_artifact_reads_or_implicit_app(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("import opened artifact")), \
                patch.object(Path, "open", side_effect=AssertionError("import opened artifact")):
            importlib.reload(r)
        self.assertFalse(hasattr(r, "app"))
        self.assertIn("target96", str(r.DEFAULT_MANIFEST))


if __name__ == "__main__":
    unittest.main()
