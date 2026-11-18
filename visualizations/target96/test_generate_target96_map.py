"""Synthetic helpers only: no real student data, classifiers, training, or HTTP."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("target96_map_helpers", ROOT / "scripts/generate_target96_map.py")
mapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapper)
from mapmatch.data import Graph


class Target96MapHelpers(unittest.TestCase):
    def acceptance(self):
        return {"accepted": True, "target_met": True, "observed_edge_f1": .98, "baseline_edge_f1": .95,
                "failed_trips": 0, "route_continuity": 1., "mean_length_f1": .99,
                "not_official_or_blind": True, "production_switched": False,
                "release_manifest_sha256": "a" * 64, "comparison_sha256": "b" * 64}

    def summary(self):
        return {"release_manifest_sha256": "a" * 64, "trips": 10, "observations": 19160,
                "target_met": True, "target_threshold": .96, "selection_frozen_before_student": True,
                "evaluator_used": False, "reference_after_all_predictions": True,
                "candidate": {"edge_f1": .98, "failures": 0, "route_continuity": 1., "length_f1": .99},
                "baseline": {"edge_f1": .95, "length_f1": .98},
                "trip_bindings": [{"trip_id": f"{i:08d}",
                                   "path": f"experiments/target96/student_comparison/{i:08d}.json",
                                   "sha256": "c" * 64} for i in range(10)]}

    def graph(self):
        return Graph(np.array([[13., 42.], [13.001, 42.], [13.001, 42.001]]),
                     np.array([[0, 1], [1, 0], [1, 2]]))

    def test_accepted_threshold_flags_and_hashes_fail_closed(self):
        mapper.validate_acceptance(self.acceptance())
        for key, bad in (("accepted", False), ("accepted", 1), ("target_met", False),
                         ("observed_edge_f1", .959999), ("observed_edge_f1", float("nan")),
                         ("observed_edge_f1", True), ("observed_edge_f1", 1.01),
                         ("failed_trips", 1), ("failed_trips", False), ("route_continuity", .99),
                         ("not_official_or_blind", False), ("production_switched", True),
                         ("release_manifest_sha256", "not-a-hash"), ("comparison_sha256", None)):
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                value = self.acceptance(); value[key] = bad
                mapper.validate_acceptance(value)

    def test_explicit_lead_notice_precedes_all_real_reads(self):
        with patch.object(mapper, "pinned_json", side_effect=AssertionError("no data read")) as read:
            with self.assertRaisesRegex(ValueError, "Lead accepted notice"):
                mapper.load_evidence(None, "a" * 64)
            with self.assertRaisesRegex(ValueError, "accepted-file SHA256"):
                mapper.load_evidence("Actual Lead accepted notice", None)
            read.assert_not_called()

    def test_rejected_acceptance_never_reads_release_or_student(self):
        with patch.object(mapper, "pinned_json", return_value={"accepted": False}) as read:
            with self.assertRaisesRegex(ValueError, "not been accepted"):
                mapper.load_evidence("Actual Lead accepted notice", "a" * 64)
            self.assertEqual(read.call_count, 1)
            self.assertEqual(read.call_args.args[0], mapper.ACCEPTANCE)

    def test_summary_requires_dense_allten_and_bound_trip09(self):
        row = mapper.validate_summary(self.acceptance(), self.summary(), "a" * 64)
        self.assertEqual(row["trip_id"], "00000009")
        for key, bad in (("observations", 19159), ("trips", 9), ("target_met", False),
                         ("selection_frozen_before_student", False), ("evaluator_used", True),
                         ("release_manifest_sha256", "d" * 64)):
            value = self.summary(); value[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                mapper.validate_summary(self.acceptance(), value, "a" * 64)
        value = self.summary(); value["trip_bindings"][-1] = copy.deepcopy(value["trip_bindings"][0])
        with self.assertRaises(ValueError):
            mapper.validate_summary(self.acceptance(), value, "a" * 64)
        value = self.summary(); value["trip_bindings"][-1]["path"] = "../elsewhere.json"
        with self.assertRaises(ValueError):
            mapper.validate_summary(self.acceptance(), value, "a" * 64)

    def test_length_f1_gate_not_just_edge_threshold(self):
        value = self.summary(); value["baseline"]["length_f1"] = 1.
        with self.assertRaisesRegex(ValueError, "length-F1 gate"):
            mapper.validate_summary(self.acceptance(), value, "a" * 64)

    def test_sequence_hash_cannot_replace_real_file_byte_hash(self):
        route = np.array([0, 1, 0, 2])
        serialized = b"0\n1\n0\n2\n"
        byte_hash = hashlib.sha256(serialized).hexdigest()
        record = {"route_sha256": mapper.integer_hash(route), "route_file_sha256": byte_hash}
        self.assertNotEqual(byte_hash, record["route_sha256"])
        mapper.check_route_identity(route, byte_hash, record)
        with self.assertRaisesRegex(ValueError, "file-byte"):
            mapper.check_route_identity(route, record["route_sha256"], record)
        with self.assertRaisesRegex(ValueError, "sequence"):
            mapper.check_route_identity(route[::-1], byte_hash, record)
        with self.assertRaisesRegex(ValueError, "file-byte"):
            mapper.check_route_identity(route, byte_hash, {"route_sha256": mapper.integer_hash(route)})

    def test_original_arc_order_repeats_endpoints_and_no_join_invention(self):
        records, integrity = mapper.traversal_records(self.graph(), np.array([0, 1, 0, 2]))
        self.assertEqual([r["arc_id"] for r in records], [0, 1, 0, 2])
        self.assertEqual([r["occurrence"] for r in records], [1, 1, 2, 1])
        self.assertEqual(records[0]["coordinates_latlon"], [[42., 13.], [42., 13.001]])
        self.assertEqual(integrity["arc_rows"], 4)
        self.assertEqual(integrity["duplicate_occurrences_beyond_first"], 1)
        self.assertEqual(integrity["broken_joins"], 0)
        self.assertEqual(integrity["connectors_inserted"], 0)
        records, integrity = mapper.traversal_records(self.graph(), np.array([0, 0]))
        self.assertEqual(integrity["broken_join_positions"], [0])
        self.assertEqual(len(records), 2)
        self.assertEqual(integrity["connectors_inserted"], 0)

    def test_invalid_empty_and_zero_length_route_refused(self):
        for route in (np.array([], dtype=int), np.array([-1]), np.array([3])):
            with self.assertRaises(ValueError):
                mapper.traversal_records(self.graph(), route)
        graph = Graph(np.array([[13., 42.], [13., 42.]]), np.array([[0, 1]]))
        with self.assertRaisesRegex(ValueError, "zero-length"):
            mapper.traversal_records(graph, np.array([0]))

    def test_canonical_scopes_no_history_overwrites(self):
        with self.assertRaises(ValueError):
            mapper.safe_output(ROOT / "visualizations/final_five/maps/do-not-touch.html")
        with self.assertRaises(ValueError):
            mapper.safe_output(ROOT / "reports/target96/elsewhere.json")
        with self.assertRaises(ValueError):
            mapper.safe_output(Path(__file__))
        for value in ("../outside", "/tmp/elsewhere", "mapmatch/../unsafe", "mapmatch//data.py", "mapmatch\\data.py"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mapper.repo_path(value)

    def test_metrics_reject_nonfinite_and_wrong_values(self):
        mapper.compare_metrics({"edge_f1": .97, "prediction_connected": True}, {"edge_f1": .97, "prediction_connected": True}, "test")
        for actual in ({"edge_f1": .96}, {"edge_f1": float("nan")}, {"edge_f1": True}):
            with self.assertRaises(ValueError):
                mapper.compare_metrics(actual, {"edge_f1": .97}, "test")

    def test_synthetic_folium_has_only_local_assets_alllayers_and_repeat_records(self):
        graph = self.graph(); route = np.array([0, 1, 0, 2])
        records, integrity = mapper.traversal_records(graph, route)
        fixture = {"graph": graph, "gps": np.array([[13.0001, 42., 0.], [13.0008, 42., 1.], [13.001, 42.0008, 2.]]),
                   "predicted": route, "ref": route, "prediction_records": records, "reference_records": records,
                   "prediction_integrity": integrity, "reference_integrity": integrity,
                   "model_metadata": {"model_name": "synthetic-only"},
                   "release": {"model_sha256": "0" * 64, "configuration": {"name": "synthetic-only"}},
                   "acceptance": {"observed_edge_f1": .98}, "metrics": {"edge_f1": 1., "length_f1": 1.}, "synthetic": True}
        text = mapper.build_html(fixture)
        self.assertIn("SYNTHETIC helper fixture - not a result", text)
        self.assertIn("../../frontend/vendor/leaflet/leaflet.js", text)
        self.assertNotIn("L.tileLayer(", text)
        self.assertIn("connect-src &apos;none&apos;", text)
        self.assertIn("window.__TARGET96_SOURCE_IDS__", text)
        self.assertIn('"prediction_arc_ids": [0, 1, 0, 2]', text)
        for color in mapper.COLORS.values():
            self.assertIn(color, text)


if __name__ == "__main__":
    unittest.main()
