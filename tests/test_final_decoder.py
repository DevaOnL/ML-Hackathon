"""Synthetic unit tests for scripts/final_five_decoder.py (final five-family
phase). No real training data, no real model artifact, no inference: every
fixture is generated in-memory or in a temporary directory."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier

from mapmatch.features import FEATURE_NAMES
from mapmatch.learning import EmissionModel
from scripts import final_five_decoder as decoder


def synthetic_scale_dir(root: Path) -> tuple[Path, Path, Path]:
    scale_fit = root / "scale_fit.json"
    scale_fit.write_text(json.dumps({
        "fitted": {"sigma_m": 10.5, "beta_m": 15.5}, "sigma_unclamped_m": 10.2,
        "beta_unclamped_m": 15.1, "sigma_clamped": False, "beta_clamped": False,
        "perpendicular_residual_m": {"count": 100}, "sigma_observations_excluded": 3,
        "beta_pair_status_counts": {"identified": 90}, "training_trip_count": 80,
        "reference_discontinuity_count": 5, "plan_sha256": "0" * 64}), encoding="utf-8")
    residuals = root / "training_residual_samples.csv"
    residuals.write_text("residual_m\n1.0\n2.0\n", encoding="utf-8")
    transitions = root / "training_transition_samples.csv"
    transitions.write_text("span_m\n10.0\n20.0\n", encoding="utf-8")
    return scale_fit, residuals, transitions


def synthetic_final_model():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(150, len(FEATURE_NAMES))).astype(np.float32)
    y = (X[:, 0] + 0.2 * X[:, 1] > 0).astype(int)
    estimator = HistGradientBoostingClassifier(
        **decoder.FIXED_HISTGB_PARAMETERS, random_state=0)
    estimator.fit(X, y)
    metadata = {"model_name": "hist_gradient_boosting",
                "run_name": "hist_gradient_boosting_tuned",
                "parameters": dict(decoder.FIXED_HISTGB_PARAMETERS),
                "phase": "five-model-completion", "training_trip_count": 80}
    return EmissionModel(estimator, metadata)


def synthetic_args(root: Path, model: Path):
    scale_fit, residuals, transitions = synthetic_scale_dir(root)
    return types.SimpleNamespace(train_root=root / "no-train",
                                 student_root=root / "no-student", model=model,
                                 baseline_output=None, scale_fit=scale_fit,
                                 scale_residuals=residuals, scale_transitions=transitions)


class FinalDecoderConstantsTests(unittest.TestCase):
    def test_fixed_histgb_parameters_are_the_frozen_tuned_values(self):
        self.assertEqual(decoder.FIXED_HISTGB_PARAMETERS, {
            "max_iter": 104, "learning_rate": 0.15902391896503631,
            "max_leaf_nodes": 22, "min_samples_leaf": 38,
            "l2_regularization": 0.8604220152998319})
        self.assertEqual(decoder.REQUESTED_FAMILIES,
                         ("logistic_regression", "knn_distance_weighted", "bagging",
                          "gaussian_nb", "hist_gradient_boosting"))
        self.assertEqual(decoder.EXPECTED_FINAL_FAMILY, "hist_gradient_boosting")
        self.assertEqual(decoder.CALIBRATION_IDS,
                         tuple(f"{i:08d}" for i in range(1, 72, 10)))
        self.assertEqual(decoder.EXPECTED_STUDENT_OBSERVATIONS, 19160)

    def test_requested_families_reads_final_scope_and_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope = Path(tmp) / "FINAL_SCOPE.json"
            scope.write_text(json.dumps({"model_families": list(decoder.REQUESTED_FAMILIES)}), encoding="utf-8")
            with patch.object(decoder, "FINAL_SCOPE_PATH", scope):
                self.assertEqual(decoder.requested_families(), decoder.REQUESTED_FAMILIES)
            scope.write_text("{ not json", encoding="utf-8")
            with patch.object(decoder, "FINAL_SCOPE_PATH", scope):
                self.assertEqual(decoder.requested_families(), decoder.REQUESTED_FAMILIES)

    def test_evaluator_path_guard(self):
        decoder.assert_no_evaluator_paths(Path("/tmp/train"), Path("/tmp/model.joblib"))
        for bad in ("evaluator_test/trip.nodes", "reports/EVALUATOR/x.json", "/data/evaluator.csv"):
            with self.assertRaises(ValueError):
                decoder.assert_no_evaluator_paths(Path(bad))


class AcceptanceGateTests(unittest.TestCase):
    def _write_gate(self, root: Path, *, model_bytes=b"model", **overrides):
        model = root / "model.joblib"
        model.write_bytes(model_bytes)
        import hashlib
        digest = hashlib.sha256(model_bytes).hexdigest()
        acceptance = {"passed": True, "exactly_five_families": True,
                      "families": list(decoder.REQUESTED_FAMILIES),
                      "all_models": {f: {"passed": True} for f in decoder.REQUESTED_FAMILIES},
                      "all80_train_trips": True, "all10_student_trips": True,
                      "student_used_for_selection": False, "evaluator_accessed": False,
                      "selected_model_sha256": digest}
        clean = {"passed": True, "model_sha256": digest}
        freeze = {"model_sha256": digest}
        acceptance.update(overrides)
        (root / "acceptance.json").write_text(json.dumps(acceptance), encoding="utf-8")
        (root / "clean.json").write_text(json.dumps(clean), encoding="utf-8")
        (root / "freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
        return model

    def _patched_paths(self, root: Path):
        return (patch.object(decoder, "ACCEPTANCE_PATH", root / "acceptance.json"),
                patch.object(decoder, "CLEAN_LOAD_PATH", root / "clean.json"),
                patch.object(decoder, "MODEL_FREEZE_PATH", root / "freeze.json"))

    def test_acceptance_passes_and_pins_model_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = self._write_gate(root)
            patches = self._patched_paths(root)
            with patches[0], patches[1], patches[2]:
                record = decoder.require_final_acceptance(model)
            self.assertTrue(record["acceptance_passed"])
            self.assertTrue(record["clean_load_passed"])
            self.assertEqual(record["model_sha256"], decoder.sha256(model))

    def test_acceptance_fails_on_missing_or_unpassed_or_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = self._write_gate(root)
            patches = self._patched_paths(root)
            with patches[0], patches[1], patches[2]:
                (root / "clean.json").unlink()
                with self.assertRaises(ValueError):
                    decoder.require_final_acceptance(model)
            model2 = self._write_gate(root, model_bytes=b"different", passed=False)
            patches = self._patched_paths(root)
            with patches[0], patches[1], patches[2]:
                with self.assertRaises(ValueError):
                    decoder.require_final_acceptance(model2)
            model3 = self._write_gate(root, model_bytes=b"m3", evaluator_accessed=True)
            patches = self._patched_paths(root)
            with patches[0], patches[1], patches[2]:
                with self.assertRaises(ValueError):
                    decoder.require_final_acceptance(model3)
            model4 = self._write_gate(root, model_bytes=b"m4", families=["a", "b"])
            patches = self._patched_paths(root)
            with patches[0], patches[1], patches[2]:
                with self.assertRaises(ValueError):
                    decoder.require_final_acceptance(model4)
            model5 = self._write_gate(root, model_bytes=b"m5")
            (root / "clean.json").write_text(json.dumps({"passed": True, "model_sha256": "0" * 64}), encoding="utf-8")
            patches = self._patched_paths(root)
            with patches[0], patches[1], patches[2]:
                with self.assertRaises(ValueError):
                    decoder.require_final_acceptance(model5)

    def test_search_refuses_without_plan_then_without_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, report = Path(tmp), Path(tmp) / "report"
            model = root / "model.joblib"
            model.write_bytes(b"unfrozen")
            args = synthetic_args(root, model)
            with patch.object(decoder, "REPORT", report):
                with self.assertRaises(ValueError):
                    decoder.search(args)
                report.mkdir()
                decoder.declare(args)
                patches = self._patched_paths(root)
                with patches[0], patches[1], patches[2]:
                    with self.assertRaises(ValueError) as ctx:
                        decoder.search(args)
                self.assertIn("acceptance is not complete", str(ctx.exception))
                self.assertFalse((root / "no-train").exists())
                self.assertFalse((report / "search_configs.json").exists())


class FinalModelVerificationTests(unittest.TestCase):
    def test_verify_final_model_accepts_only_the_frozen_configuration(self):
        model = synthetic_final_model()
        record = decoder.verify_final_model(model)
        self.assertEqual(record["family"], "hist_gradient_boosting")
        self.assertTrue(record["parameters_match_fixed"])
        check = decoder.family_check(model)
        self.assertIn("hist_gradient_boosting", check["matched_requested_families"])
        self.assertTrue(check["selected_family_matches_metadata"])
        self.assertTrue(check["consistent_with_requested_five"])

    def test_verify_final_model_rejects_other_families_params_and_phases(self):
        model = synthetic_final_model()
        wrong_family = EmissionModel(model.estimator, dict(model.metadata, model_name="bagging"))
        with self.assertRaises(ValueError):
            decoder.verify_final_model(wrong_family)
        wrong_run = EmissionModel(model.estimator, dict(model.metadata, run_name="hist_default"))
        with self.assertRaises(ValueError):
            decoder.verify_final_model(wrong_run)
        wrong_params = dict(model.metadata)
        wrong_params["parameters"] = dict(decoder.FIXED_HISTGB_PARAMETERS, max_iter=103)
        with self.assertRaises(ValueError):
            decoder.verify_final_model(EmissionModel(model.estimator, wrong_params))
        wrong_phase = EmissionModel(model.estimator, dict(model.metadata, phase="phase-three"))
        with self.assertRaises(ValueError):
            decoder.verify_final_model(wrong_phase)
        wrong_trips = EmissionModel(model.estimator, dict(model.metadata, training_trip_count=72))
        with self.assertRaises(ValueError):
            decoder.verify_final_model(wrong_trips)
        rng = np.random.default_rng(3)
        X = rng.normal(size=(60, len(FEATURE_NAMES))).astype(np.float32)
        bag = BaggingClassifier(n_estimators=3, random_state=0).fit(X, (X[:, 0] > 0).astype(int))
        with self.assertRaises(ValueError):
            decoder.verify_final_model(EmissionModel(bag, dict(model.metadata)))


class PlanAndWriteGuardTests(unittest.TestCase):
    def test_declare_is_idempotent_and_detects_protocol_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, report = Path(tmp), Path(tmp) / "report"
            args = synthetic_args(root, root / "model.joblib")
            with patch.object(decoder, "REPORT", report):
                plan = decoder.declare(args)
                self.assertTrue((report / "calibration_plan.json").exists())
                again = decoder.declare(args)
                self.assertEqual(again, plan)
                self.assertEqual([c["name"] for c in plan["configurations"]],
                                 ["A_fitted", "B_twice_beta", "C_classifier_up"])
                self.assertEqual(plan["phase"], "five-model-completion")
                self.assertEqual(len(plan["requested_model_families"]), 5)
                tampered = json.loads((report / "calibration_plan.json").read_text(encoding="utf-8"))
                tampered["configurations"][0]["beta_multiplier"] = 3.0
                (report / "calibration_plan.json").write_text(json.dumps(tampered), encoding="utf-8")
                with self.assertRaises(ValueError):
                    decoder.declare(args)

    def test_write_json_confines_writes_to_final_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, report = Path(tmp), Path(tmp) / "report"
            config = root / "match_config.json"
            with patch.object(decoder, "REPORT", report), patch.object(decoder, "CONFIG_PATH", config):
                decoder.write_json(report / "nested" / "out.json", {"a": 1})
                decoder.write_json(config, {"b": 2})
                self.assertTrue((report / "nested" / "out.json").exists())
                with self.assertRaises(ValueError):
                    decoder.write_json(root / "elsewhere.json", {"c": 3})
                with self.assertRaises(ValueError):
                    decoder.write_csv(root / "elsewhere.csv", [{"d": 4}])


class PureFunctionTests(unittest.TestCase):
    def test_adjacent_collapse_and_overlap_only(self):
        self.assertEqual(decoder.adjacent_collapse([1, 1, 2, 1, 1, 3]).tolist(), [1, 2, 1, 3])
        graph = types.SimpleNamespace(edge_lengths=np.array([10.0, 20.0, 30.0]))
        overlap = decoder.overlap_only(graph, np.array([0, 1, 1, 2]), np.array([0, 1, 2, 0]))
        self.assertAlmostEqual(overlap["edge_f1"], 2 * 3 / (4 + 4))
        self.assertAlmostEqual(overlap["length_f1"], 2 * 60.0 / (80.0 + 70.0))
        self.assertAlmostEqual(overlap["predicted_length_m"], 80.0)
        self.assertAlmostEqual(overlap["reference_length_m"], 70.0)

    def test_aggregate_headline_eligibility_requires_all_ten(self):
        def row(trip, ok=True):
            base = {"trip_id": trip, "method": "classifier_plus_HMM", "status": "ok" if ok else "failed",
                    "observations": 100, "inference_seconds": 1.0, "route_discontinuities": 0,
                    "prediction_connected": True,
                    "candidate_miss_count": 0, "proxy_ambiguous_count": 0,
                    "predicted_arc_count": 50, "reference_arc_count": 50, "common_arc_count": 45,
                    "predicted_length_m": 500.0, "reference_length_m": 500.0, "common_length_m": 450.0,
                    "route_edit_distance": 5, "proxy_accuracy": .9, "proxy_precision": .9,
                    "proxy_recall": .9, "proxy_macro_f1": .9, "route_edge_f1": .9,
                    "route_length_f1": .9, "route_edit_similarity": .9, "route_continuity": 1.0}
            return base

        rows = [row(f"t{i}") for i in range(10)]
        aggregate = decoder.aggregate(rows, {}, 10)["classifier_plus_HMM"]
        self.assertTrue(aggregate["all_ten_headline_eligible"])
        self.assertEqual(aggregate["successful_trip_count"], 10)
        rows[3] = row("t3", ok=False)
        aggregate = decoder.aggregate(rows, {}, 10)["classifier_plus_HMM"]
        self.assertFalse(aggregate["all_ten_headline_eligible"])
        self.assertEqual(aggregate["failed_or_unavailable_trip_count"], 1)
        self.assertLess(aggregate["trip_macro_failures_as_zero"]["proxy_accuracy"], .9)


if __name__ == "__main__":
    unittest.main()
