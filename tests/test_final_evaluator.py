"""Synthetic evaluator guards. No actual evaluator inputs or fitted models are read."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mapmatch import cli
from mapmatch.data import Graph
from mapmatch.matcher import MatchResult
from mapmatch.metrics import score_routes
from scripts import final_evaluator as evaluator


class FinalEvaluatorTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / "outputs/final_evaluator_test"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic_guard_", dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.graph = Graph(np.array([[0, 0], [0.001, 0]]), np.array([[0, 1], [1, 0], [0, 0]]))

    def fixture(self, tid="00000015"):
        prefix = self.root / "raw" / tid / tid
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".nodes").write_text("0 0\n0.001 0\n")
        prefix.with_suffix(".arcs").write_text("0 1\n1 0\n0 0\n")
        prefix.with_suffix(".track").write_text("0 0 1\n0.001 0 2\n")
        prefix.with_suffix(".route").write_text("PRIVATE REFERENCE MUST NOT PARSE\n")
        return prefix

    def test_audit_never_opens_reference(self):
        prefix = self.fixture()
        opened = []
        original = Path.open
        def spy(path, *args, **kwargs):
            opened.append(path)
            if path.suffix == ".route":
                raise AssertionError("reference opened during audit")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", spy):
            row = evaluator.structural_audit(prefix)
        self.assertEqual(row["observations"], 2)
        self.assertIsNone(row["sha256_by_extension"][".route"])
        self.assertTrue(opened)
        self.assertFalse(any(path.suffix == ".route" for path in opened))

    def test_invalid_timestamp_audit_rejected(self):
        prefix = self.fixture()
        prefix.with_suffix(".track").write_text("0 0 2\n0.001 0 1\n")
        with self.assertRaises(ValueError):
            evaluator.structural_audit(prefix)

    def test_exact_layout_and_wrong_trip_set(self):
        prefix = self.fixture()
        with patch.object(evaluator, "ROOT", self.root):
            self.assertEqual(set(evaluator.exact_trips(self.root / "raw", [prefix.name])), {prefix.name})
            with self.assertRaises(ValueError):
                evaluator.exact_trips(self.root / "raw", ["00000017"])

    def test_symlink_and_outside_rejected(self):
        target = self.root / "plain"
        target.write_text("test")
        link = self.root / "link"
        link.symlink_to(target)
        with self.assertRaises(ValueError):
            evaluator.safe_path(link, self.root)
        with self.assertRaises(ValueError):
            evaluator.safe_path(self.root.parent, self.root)

    def test_refuses_overwrite_and_nonfinite_public_json(self):
        report = self.root / "reports"
        output = self.root / "output"
        with patch.multiple(evaluator, ROOT=self.root, REPORT=report, OUTPUT=output):
            evaluator.write_json(report / "one.json", {"passed": True})
            with self.assertRaises(ValueError):
                evaluator.write_json(report / "one.json", {"passed": False})
            with self.assertRaises(ValueError):
                evaluator.write_json(report / "nan.json", {"metric": float("nan")})
            with self.assertRaises(ValueError):
                evaluator.write_json(report / "raw.json", {"raw": str(self.root / "raw")})
            with self.assertRaises(ValueError):
                evaluator.write_json(self.root / "outside.json", {})

    def test_freeze_failure_precedes_input_discovery(self):
        with patch.multiple(evaluator, ROOT=self.root, REPORT=self.root / "reports", OUTPUT=self.root / "empty"), patch.object(evaluator, "validate_freeze", side_effect=ValueError("bad freeze")), patch.object(evaluator, "exact_trips", side_effect=AssertionError("input discovery must not happen")):
            with self.assertRaisesRegex(ValueError, "bad freeze"):
                evaluator.stage_audit()

    def test_scoring_requires_all_ten_prediction_success(self):
        report = self.root / "reports"
        report.mkdir()
        (report / "input_audit_pre_prediction.json").write_text("{}")
        (report / "cli_result.json").write_text(json.dumps({"exit_code": 1}))
        (report / "prediction_verification.json").write_text(json.dumps({"passed": False}))
        with patch.object(evaluator, "REPORT", report), patch.object(evaluator, "validate_freeze", return_value={}), patch.object(evaluator, "load_route", side_effect=AssertionError("reference must not open")):
            with self.assertRaisesRegex(ValueError, "prediction success"):
                evaluator.stage_score()

    def test_cli_predict_pathspy_does_not_open_reference(self):
        prefix = self.fixture()
        result = MatchResult(np.array([0]), np.array([0, 1]), np.array([0, 0]), np.array([0., 1.]), np.zeros(2), [], {"route_length_m": 111.0, "classifier_only_edges": [0, 0]})
        model = self.root / "trusted_model.joblib"
        model.write_bytes(b"synthetic trusted model")
        args = argparse.Namespace(input=str(prefix.parent.parent), output=str(self.root / "cli"), config=None, model=str(model), baseline=False, trip=None)
        original = Path.open
        def spy(path, *args, **kwargs):
            if path == prefix.with_suffix(".route"):
                raise AssertionError("private reference opened in predict")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", spy), patch("mapmatch.learning.load_emission_model", return_value=object()), patch.object(cli, "_config", return_value=cli.MatchConfig()), patch.object(cli.MapMatcher, "match", return_value=result), patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(cli.predict(args), 0)
        self.assertTrue((self.root / "cli/submission.zip").is_file())

    def test_route_multiset_keeps_repetitions_and_zero_length(self):
        value = score_routes(self.graph, [0, 1, 0, 2], [0, 1, 2])
        self.assertEqual(value["common_arc_count"], 3)
        self.assertEqual(value["predicted_arc_count"], 4)
        self.assertFalse(value["exact_match"])
        self.assertLess(value["edge_f1"], 1)
        zero = score_routes(self.graph, [2, 2], [2])
        self.assertAlmostEqual(zero["length_f1"], 2 / 3)
        self.assertIsNone(evaluator.mean_available([{"x": None}], "x"))


if __name__ == "__main__":
    unittest.main()
