"""Synthetic final-map integrity tests. No private GPS, model fitting or browser needed."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("final_map", ROOT / "visualizations/final_five/generate_final_map.py")
mapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapper)
from mapmatch.data import Graph


class TestFinalMap(unittest.TestCase):
    def identity(self):
        metadata = {"model_name": mapper.FAMILY, "run_name": mapper.RUN, "training_trip_count": 80}
        freeze = {"model_sha256": mapper.MODEL_SHA256, "metadata_sha256": "m", **metadata}
        decoder = {"model_sha256": mapper.MODEL_SHA256, "config_file_sha256": mapper.CONFIG_SHA256, "config": {"x": 1}}
        run = {"model_sha256": mapper.MODEL_SHA256, "config": {"x": 1}, "baseline": False,
               "uses_reference_routes": False, "trip_ids": [mapper.TRIP_ID]}
        return [metadata, freeze, decoder, run, mapper.MODEL_SHA256, mapper.CONFIG_SHA256, "m", {"x": 1}]

    def graph(self):
        return Graph(np.array([[13., 42.], [13.001, 42.], [13.001, 42.001]]),
                     np.array([[0, 1], [1, 0], [1, 2]]))

    def test_wrong_model_hash_family_or_metadata_rejected(self):
        args = self.identity()
        self.assertEqual(mapper.verify_identity(*args), "HistGradientBoosting (tuned)")
        args[4] = "0" * 64
        with self.assertRaisesRegex(ValueError, "wrong final model"):
            mapper.verify_identity(*args)
        args = self.identity()
        args[0]["model_name"] = "bagging"
        with self.assertRaisesRegex(ValueError, "wrong model family"):
            mapper.verify_identity(*args)
        args = self.identity()
        args[6] = "changed"
        with self.assertRaisesRegex(ValueError, "metadata hash"):
            mapper.verify_identity(*args)

    def test_wrong_decoder_and_reference_based_prediction_rejected(self):
        args = self.identity()
        args[5] = "0" * 64
        with self.assertRaisesRegex(ValueError, "decoder hash"):
            mapper.verify_identity(*args)
        args = self.identity()
        args[3]["uses_reference_routes"] = True
        with self.assertRaisesRegex(ValueError, "reference-free"):
            mapper.verify_identity(*args)

    def test_per_arc_geometry_preserves_order_and_repeat_occurrences(self):
        records, integrity = mapper.traversal_records(self.graph(), np.array([0, 1, 0, 2]))
        self.assertEqual([row["arc_id"] for row in records], [0, 1, 0, 2])
        self.assertEqual([row["occurrence"] for row in records], [1, 1, 2, 1])
        self.assertEqual(records[0]["coordinates_latlon"], [[42., 13.], [42., 13.001]])
        self.assertEqual(integrity["arc_rows"], 4)
        self.assertEqual(integrity["duplicate_occurrences_beyond_first"], 1)
        self.assertEqual(integrity["broken_joins"], 0)
        self.assertEqual(integrity["connectors_inserted"], 0)

    def test_broken_joins_recorded_without_inserted_connectors(self):
        records, integrity = mapper.traversal_records(self.graph(), np.array([0, 0]))
        self.assertEqual(len(records), 2)
        self.assertEqual(integrity["broken_join_positions"], [0])
        self.assertEqual(integrity["connectors_inserted"], 0)

    def test_empty_invalid_or_zero_geometry_rejected(self):
        for route in (np.array([], dtype=int), np.array([-1]), np.array([3])):
            with self.assertRaises(ValueError):
                mapper.traversal_records(self.graph(), route)
        graph = Graph(np.array([[13., 42.], [13., 42.]]), np.array([[0, 1]]))
        with self.assertRaisesRegex(ValueError, "zero-length"):
            mapper.traversal_records(graph, np.array([0]))

    def test_only_original_development_trip_and_frozen_prediction_are_allowed(self):
        with self.assertRaisesRegex(ValueError, "original Student development"):
            mapper.allowed_inputs(ROOT / "synthetic" / "00000009", ROOT / "synthetic.route")
        with self.assertRaisesRegex(ValueError, "frozen final Student CLI"):
            mapper.allowed_inputs(ROOT / "Student folder/student_test/00000009/00000009",
                                  ROOT / "outputs/phase_three_student_test/routes/00000009.route")

    def test_scope_guard_rejects_all_symlinks_and_outside_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "final_five"
            base.mkdir()
            target = base / "real"
            target.mkdir()
            link = base / "linked"
            link.symlink_to(target, target_is_directory=True)
            with patch.object(mapper, "FIVE_DIR", base):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    mapper.output(link / "map.html")
                with self.assertRaisesRegex(ValueError, "outside"):
                    mapper.output(root / "map.html")
                self.assertEqual(mapper.output(base / "maps" / "map.html"), base / "maps" / "map.html")


if __name__ == "__main__":
    unittest.main()
