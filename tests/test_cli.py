"""CLI integration on synthetic files only, including archive and immutability."""
from __future__ import annotations

import builtins
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from zipfile import ZipFile

from mapmatch.cli import main
from tests.test_core import write_fixture


def run_cli(arguments):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main([str(value) for value in arguments])
    return code, out.getvalue(), err.getvalue()


def config_file(root):
    path = Path(root) / "explicit synthetic config.json"
    path.write_text("{}\n", encoding="utf-8")
    return path


class CommandLineAcceptanceTests(unittest.TestCase):
    def test_prediction_needs_no_reference_and_writes_only_original_arc_ids(self):
        with tempfile.TemporaryDirectory(prefix="cli spaces ") as root:
            source, output = Path(root) / "raw input", Path(root) / "prediction output"
            prefix = write_fixture(source)
            config = config_file(root)
            code, _, err = run_cli(["predict", "--input", source, "--output", output,
                                    "--baseline", "--config", config])
            self.assertEqual(code, 0, err)
            route = output / "routes" / "00000001.route"
            self.assertEqual(route.read_bytes(), b"1\n0\n")
            settings = json.loads((output / "run_config.json").read_text())
            self.assertFalse(settings["uses_reference_routes"])
            self.assertTrue(settings["baseline"])
            self.assertEqual(settings["trip_ids"], ["00000001"])
            self.assertEqual(json.loads((output / "failures.json").read_text()), [])
            with (output / "index.csv").open(newline="") as handle:
                index = list(csv.DictReader(handle))
            self.assertEqual(len(index), 1)
            self.assertEqual(index[0]["sha256"], hashlib.sha256(route.read_bytes()).hexdigest())
            with (output / "observations" / "00000001.csv").open(newline="") as handle:
                samples = list(csv.DictReader(handle))
            self.assertEqual([int(row["observation_index"]) for row in samples], [0, 1, 2, 3])
            self.assertEqual([float(row["timestamp_s"]) for row in samples], [1, 2, 3, 4])
            self.assertEqual([int(row["predicted_arc_id"]) for row in samples], [1, 1, 0, 0])
            self.assertFalse(prefix.with_suffix(".route").exists())

    def test_guarded_reference_is_not_opened_even_when_present(self):
        with tempfile.TemporaryDirectory() as root:
            source, output = Path(root) / "source", Path(root) / "predicted"
            prefix = write_fixture(source)
            forbidden = prefix.with_suffix(".route")
            forbidden.write_text("NEVER OPEN SYNTHETIC REFERENCE\n", encoding="utf-8")
            config = config_file(root)
            original_builtin, original_io = builtins.open, io.open
            def guard(original):
                def guarded(file, *args, **kwargs):
                    if isinstance(file, (str, bytes, Path)) and Path(file).resolve() == forbidden.resolve():
                        raise AssertionError("prediction opened a reference")
                    return original(file, *args, **kwargs)
                return guarded
            with mock.patch("mapmatch.cli.load_route", side_effect=AssertionError("reference loader used")), \
                    mock.patch("builtins.open", side_effect=guard(original_builtin)), \
                    mock.patch("io.open", side_effect=guard(original_io)):
                code, _, err = run_cli(["predict", "--input", source, "--output", output,
                                        "--baseline", "--config", config])
            self.assertEqual(code, 0, err)
            self.assertEqual((output / "routes" / "00000001.route").read_bytes(), b"1\n0\n")

    def test_archive_is_flat_complete_and_byte_deterministic_across_hash_seeds(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "input"
            write_fixture(source, "00000009")
            write_fixture(source, "00000002")
            config = config_file(root)
            archives = []
            for run, seed in enumerate(("19", "876")):
                output = Path(root) / f"run {run}"
                command = [sys.executable, "-m", "mapmatch", "predict", "--input", str(source),
                           "--output", str(output), "--baseline", "--config", str(config)]
                process = subprocess.run(command, capture_output=True, text=True, timeout=30,
                                         env={**os.environ, "PYTHONHASHSEED": seed})
                self.assertEqual(process.returncode, 0, process.stderr)
                archive = output / "submission.zip"
                archives.append(archive.read_bytes())
                with ZipFile(archive) as zipped:
                    self.assertIsNone(zipped.testzip())
                    self.assertEqual(zipped.namelist(), ["00000002.route", "00000009.route"])
                    for member in zipped.infolist():
                        self.assertEqual(member.date_time, (2026, 1, 1, 0, 0, 0))
                        self.assertFalse(Path(member.filename).is_absolute())
                        self.assertNotIn("..", Path(member.filename).parts)
                        self.assertEqual(zipped.read(member.filename), (output / "routes" / member.filename).read_bytes())
                        self.assertEqual(zipped.read(member.filename), b"1\n0\n")
            self.assertEqual(archives[0], archives[1])

    def test_missing_trained_model_does_not_silently_fallback_to_geometry(self):
        with tempfile.TemporaryDirectory() as root:
            source, output = Path(root) / "input", Path(root) / "predictions"
            write_fixture(source)
            code, _, err = run_cli(["predict", "--input", source, "--output", output,
                                    "--model", Path(root) / "missing.joblib", "--config", config_file(root)])
            self.assertNotEqual(code, 0)
            self.assertIn("missing.joblib", err)
            self.assertFalse((output / "submission.zip").exists())
            self.assertFalse((output / "routes").exists())

    def test_selected_trip_ids_reject_unknown_and_traversal_values(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "input"
            write_fixture(source, "00000001")
            write_fixture(source, "00000002")
            config = config_file(root)
            for trip in ("../00000001", "..", "/etc/passwd", "missing"):
                output = Path(root) / "unused"
                code, _, _ = run_cli(["predict", "--input", source, "--output", output,
                                       "--trip", trip, "--baseline", "--config", config])
                self.assertNotEqual(code, 0)
                self.assertFalse(output.exists())
            output = Path(root) / "chosen"
            code, _, err = run_cli(["predict", "--input", source, "--output", output,
                                    "--trip", "00000002", "--trip", "00000002",
                                    "--baseline", "--config", config])
            self.assertEqual(code, 0, err)
            self.assertEqual(sorted(path.name for path in (output / "routes").iterdir()), ["00000002.route"])

    def test_output_under_input_and_stale_output_are_rejected_without_changes(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "input"
            prefix = write_fixture(source)
            before = {path: path.read_bytes() for path in prefix.parent.iterdir()}
            config = config_file(root)
            for output in (source, source / "new predictions"):
                code, _, _ = run_cli(["predict", "--input", source, "--output", output,
                                       "--baseline", "--config", config])
                self.assertNotEqual(code, 0)
            output = Path(root) / "stale"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_bytes(b"keep me")
            code, _, _ = run_cli(["predict", "--input", source, "--output", output,
                                   "--baseline", "--config", config])
            self.assertNotEqual(code, 0)
            self.assertEqual(sentinel.read_bytes(), b"keep me")
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["keep.txt"])
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_disconnected_trip_creates_failure_record_not_valid_looking_archive(self):
        with tempfile.TemporaryDirectory() as root:
            source, output = Path(root) / "input", Path(root) / "output"
            write_fixture(source, points=[(0, 0), (100, 0), (1000, 0), (1100, 0)],
                          arcs=[(0, 1), (2, 3)], samples=[(20, 0), (1080, 0)])
            code, _, _ = run_cli(["predict", "--input", source, "--output", output,
                                   "--baseline", "--config", config_file(root)])
            self.assertEqual(code, 1)
            self.assertFalse((output / "submission.zip").exists())
            self.assertFalse((output / "routes" / "00000001.route").exists())
            failures = json.loads((output / "failures.json").read_text())
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["status"], "failed")

    def test_explicit_evaluation_reports_actual_macro_and_reference_defects(self):
        with tempfile.TemporaryDirectory() as root:
            source, predictions = Path(root) / "student_test", Path(root) / "predictions"
            prefix = write_fixture(source)
            prefix.with_suffix(".route").write_text("1\n0\n", encoding="utf-8")
            code, _, err = run_cli(["predict", "--input", source, "--output", predictions,
                                    "--baseline", "--config", config_file(root)])
            self.assertEqual(code, 0, err)
            scores = Path(root) / "scores.json"
            code, _, err = run_cli(["evaluate", "--input", source, "--predictions", predictions, "--output", scores])
            self.assertEqual(code, 0, err)
            report = json.loads(scores.read_text())
            self.assertEqual(report["summary"]["trip_count"], 1)
            self.assertEqual(report["summary"]["macro"]["length_f1"], 1)
            self.assertEqual(report["summary"]["exact_route_count"], 1)
            self.assertFalse(report["summary"]["official_score"])

    def test_validate_without_reference_flag_never_opens_label(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "input"
            prefix = write_fixture(source)
            prefix.with_suffix(".route").write_text("invalid reference sentinel\n")
            output = Path(root) / "validation.json"
            with mock.patch("mapmatch.cli.load_route", side_effect=AssertionError("unexpected label access")):
                code, _, err = run_cli(["validate", "--input", source, "--output", output])
            self.assertEqual(code, 0, err)
            self.assertFalse(json.loads(output.read_text())["reference_labels_read"])

    def test_validate_references_must_fail_if_reference_route_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "input"
            prefix = write_fixture(source)
            prefix.with_suffix(".route").write_text("999\n")
            code, _, _ = run_cli(["validate", "--input", source, "--output", Path(root) / "validation.json", "--references"])
            self.assertNotEqual(code, 0, "validate reported success for an invalid route ID")

    def test_validate_cannot_overwrite_original_raw_file(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "input"
            prefix = write_fixture(source)
            target = prefix.with_suffix(".track")
            before = target.read_bytes()
            code, _, _ = run_cli(["validate", "--input", source, "--output", target])
            self.assertNotEqual(code, 0, "validate must protect original input files")
            self.assertEqual(target.read_bytes(), before)

    def test_evaluate_cannot_overwrite_original_reference_file(self):
        with tempfile.TemporaryDirectory() as root:
            source, predicted = Path(root) / "input", Path(root) / "predictions"
            prefix = write_fixture(source)
            target = prefix.with_suffix(".route")
            target.write_bytes(b"1\n0\n")
            predicted.mkdir()
            (predicted / "00000001.route").write_bytes(b"1\n0\n")
            before = target.read_bytes()
            code, _, _ = run_cli(["evaluate", "--input", source, "--predictions", predicted, "--output", target])
            self.assertNotEqual(code, 0, "evaluate must protect original references")
            self.assertEqual(target.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
