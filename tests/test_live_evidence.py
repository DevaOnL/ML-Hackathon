"""Synthetic checks for real-network evidence bookkeeping, never live inference."""
from __future__ import annotations
import hashlib
import json
import struct
import unittest
from unittest import mock

from scripts import live_evidence as live
from scripts.capture_api_terminal import terminal_frame_flags


class LiveEvidenceContractTests(unittest.TestCase):
    def test_terminal_persistence_guard_requires_marker_both_events_and_exact_request(self):
        request_id = "abcdef0123456789"
        text = "ACTUAL API RECEIVING TERMINAL\nmatch_started\nmatch_completed\nRequest " + request_id
        self.assertTrue(all(terminal_frame_flags(text, request_id).values()))
        self.assertTrue(all(terminal_frame_flags(text.replace("0", "@"), request_id).values()))
        for token in ("ACTUAL API RECEIVING TERMINAL", "match_started", "match_completed", request_id):
            with self.subTest(token=token):
                self.assertFalse(all(terminal_frame_flags(text.replace(token, ""), request_id).values()))
        self.assertFalse(all(terminal_frame_flags(text, "different-request").values()))

    def test_api_geometry_parts_are_not_misread_as_arc_features(self):
        payload = {"request_id": "synthetic-request", "trip_id": "00000009",
                   "model": {"sha256": "f" * 64}, "match_config_hash": "c" * 64,
                   "metrics_status": "ok", "metrics": {"accuracy": 1},
                   "predicted_route": {"type": "FeatureCollection", "features": [
                       {"type": "Feature", "properties": {"part_index": 0, "arc_count": 2},
                        "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 0], [1, 1]]}}]},
                   "raw_gps": {"features": [{}]},
                   "observation_assignments": [{"edge_id": 5}, {"edge_id": 8}]}
        row = live.summarize_response(payload, "00000009", 200, 12)
        self.assertEqual(row["prediction_feature_count"], 1)
        self.assertEqual(row["observation_assignment_count"], 2)
        expected = hashlib.sha256(struct.pack("<qq", 5, 8)).hexdigest()
        self.assertEqual(row["matched_edge_array_sha256"], expected)
        self.assertNotIn("route_file_sha256", row)

    def test_private_reference_field_is_rejected_recursively(self):
        for key in live.PRIVATE_KEYS:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                live.inspect_public({"nested": [{key: []}]})
        live.inspect_public({"proxy_reference": True, "metrics": {"accuracy": .9}})

    def test_selected_response_and_nonempty_request_id_required(self):
        payload = {"request_id": "x", "trip_id": "other", "predicted_route": {"features": []}}
        with self.assertRaises(RuntimeError):
            live.summarize_response(payload, "00000009", 200, 0)
        payload.update(trip_id="00000009", request_id="")
        with self.assertRaises(RuntimeError):
            live.summarize_response(payload, "00000009", 200, 0)

    def test_health_must_match_freeze_ready_count_and_no_prediction_cache(self):
        health = {"status": "ready", "model_loaded": True, "model_hash": "m",
                  "match_config_hash": "c", "trip_count": 10, "prediction_cache": False}
        with mock.patch.object(live, "expected_freeze", return_value=({"model_sha256": "m"}, "c")):
            live.check_health(health)
            for key, bad in (("status", "not_ready"), ("model_loaded", False),
                             ("model_hash", "wrong"), ("match_config_hash", "wrong"),
                             ("trip_count", 9), ("prediction_cache", True)):
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    live.check_health(dict(health, **{key: bad}))


if __name__ == "__main__":
    unittest.main()
