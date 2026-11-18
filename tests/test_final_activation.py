"""Synthetic guards only: no private datasets, runtime env, real HTTP or services."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import activate_final_five as activate
from scripts import verify_final_api as verify


def complete_fixture():
    expected = {"model_sha256": verify.MODEL_SHA256, "config_file_sha256": verify.CONFIG_SHA256,
                "match_config_hash": verify.COMPACT_CONFIG_SHA256, "frozen_config": {"synthetic": True}}
    offline = {}
    groups = {}
    for dataset in verify.DATASET_IDS:
        offline[dataset] = {}
        groups[dataset] = {"trip_count": 10, "checks": {"exact_ten": True}, "trips": {}}
        for index in range(10):
            tid = f"toy-{index}"
            reference = {"observations": 3, "predicted_geometry_sha256": "b" * 64,
                         "matched_edge_array_sha256": "c" * 64, "route_sha256": "d" * 64,
                         "metrics_status": "ok", "metrics": {"length_f1": .8, "edge_f1": .9, "edit_similarity": .7, "route_continuity": 1.}}
            offline[dataset][tid] = reference
            checks = dict.fromkeys(verify.REQUIRED_ROW_CHECKS, True)
            row = {"dataset_id": dataset, "trip_id": tid, "http_status": 200,
                   "request_id": dataset + "-" + tid, "model": {"name": "hist_gradient_boosting", "sha256": verify.MODEL_SHA256},
                   "match_config_hash": verify.COMPACT_CONFIG_SHA256,
                   "predicted_geometry_sha256": reference["predicted_geometry_sha256"],
                   "matched_edge_array_sha256": reference["matched_edge_array_sha256"],
                   "observation_assignment_count": 3, "metrics_status": "ok",
                   "metrics": {"route_length_f1": .8, "route_edge_f1": .9, "sequence_similarity": .7, "route_continuity": 1.},
                   "diagnostics": {"observation_count": 3, "matched_observation_count": 3,
                                   "learned_emission_observations": 3, "uses_reference_route": False},
                   "response_checks": deepcopy(checks), "checks": checks,
                   "cli_fixture_checks": {"toy_fixture": True},
                   "candidate_audit": {"checks": dict.fromkeys(verify.REQUIRED_AUDIT_CHECKS, True),
                                       "learned_emission_observations": 3, "learned_candidate_probability_count": 9,
                                       "probability_sum_error_max": 1e-16}}
            groups[dataset]["trips"][tid] = row
    report = {"passed": True, "trip_count": 20, "mocked_model_or_inference": False,
              **expected, "frozen_config_before": expected["frozen_config"], "frozen_config_after": expected["frozen_config"],
              "checks": dict.fromkeys(verify.REQUIRED_INTEGRATION_CHECKS, True), "datasets": groups,
              "repeated_request": {"excluded_from_twenty_headline": True, "request_id": "extra-fresh", "checks": {"fresh": True}}}
    return report, offline, expected


class EnvironmentGuards(unittest.TestCase):
    def original(self, ending=b"\n"):
        return ending.join([b"# Original unrelated text", b"PATHSNAP_MODEL=old/model.joblib",
                            b"PATHSNAP_MODEL_METADATA=old/model_metadata.json", b"PATHSNAP_MATCH_CONFIG=old/match_config.json",
                            b"PATHSNAP_DATASET_ROLE=development", b"UNRELATED_PRIVATE_VALUE=not-a-real-secret", b""])

    def test_six_managed_values_only_and_unrelated_exact_bytes(self):
        original = self.original()
        rendered = activate.render_environment(original)
        self.assertIn(b"UNRELATED_PRIVATE_VALUE=not-a-real-secret\n", rendered)
        self.assertIn(b"# Original unrelated text\n", rendered)
        self.assertIn(b"PATHSNAP_DATASET_ROLE=development\n", rendered)
        values = {line.split(b"=", 1)[0].decode(): line.split(b"=", 1)[1] for line in rendered.splitlines() if line and not line.startswith(b"#")}
        self.assertEqual({key: json.loads(values[key]) for key in activate.REPLACEMENTS}, activate.REPLACEMENTS)
        self.assertEqual(set(values) - set(activate.REPLACEMENTS), {"PATHSNAP_DATASET_ROLE", "UNRELATED_PRIVATE_VALUE"})

    def test_existing_evaluator_keys_updated_not_duplicated(self):
        original = self.original() + b"PATHSNAP_EVALUATOR_DATA_DIR=old\nPATHSNAP_EVALUATOR_REFERENCE_DIR=old\nPATHSNAP_EVALUATOR_EXPECTED_TRIP_COUNT=3\n"
        rendered = activate.render_environment(original)
        for key in activate.REPLACEMENTS:
            self.assertEqual(rendered.count(key.encode() + b"="), 1)

    def test_duplicate_managed_setting_rejected(self):
        for key in activate.REPLACEMENTS:
            original = self.original()
            if key not in activate.REQUIRED_ORIGINAL:
                original += key.encode() + b"=first\n"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "repeated"):
                activate.render_environment(original + key.encode() + b"=second\n")

    def test_missing_explicit_model_setting_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            activate.render_environment(b"PATHSNAP_MODEL=a\n")

    def test_unrelated_crlf_line_kept_exact(self):
        original = self.original(b"\r\n")
        rendered = activate.render_environment(original)
        self.assertIn(b"UNRELATED_PRIVATE_VALUE=not-a-real-secret\r\n", rendered)
        self.assertTrue(rendered.startswith(b"# Original unrelated text\r\n"))

    def test_non_utf8_nul_and_wrong_type_rejected(self):
        for original in (b"\xff", self.original() + b"\0", "not bytes"):
            with self.subTest(original_type=type(original).__name__), self.assertRaises((ValueError, UnicodeError)):
                activate.render_environment(original)

    def test_mutation_guard_rejects_different_current_bytes(self):
        with self.assertRaisesRegex(ValueError, "changed after validation"):
            activate.require_unchanged_bytes(b"new", b"old")

    def test_apply_requires_lead_go(self):
        activate.require_go(False, None)
        for value in (None, "yes", "GO"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Lead GO"):
                activate.require_go(True, value)
        activate.require_go(True, activate.GO_TOKEN)

    def test_default_validation_does_not_open_private_environment_or_network(self):
        expected = complete_fixture()[2]
        with patch.object(activate, "acceptance", return_value=expected), patch.object(activate, "source_manifest", return_value={"toy": "sha"}), patch.object(activate, "activate", side_effect=AssertionError("private activation called")), patch("builtins.print"):
            activate.main([])


class ActivationRollback(unittest.TestCase):
    def test_readiness_failure_restores_exact_bytes_and_restarts(self):
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / "toy.env"
            original = b"# exact\r\nTOY=old\r\n"
            env.write_bytes(original)
            calls = []
            def writer(path, contents):
                calls.append(("write", contents))
                Path(path).write_bytes(contents)
            def restart():
                calls.append(("restart",))
            def verify_failed():
                raise ValueError("synthetic readiness identity mismatch")
            with self.assertRaisesRegex(RuntimeError, "exact previous environment restored"):
                activate.mutate_with_rollback(env, original, b"TOY=new\n", writer=writer, restart=restart, verify=verify_failed)
            self.assertEqual(env.read_bytes(), original)
            self.assertEqual(calls, [("write", b"TOY=new\n"), ("restart",), ("write", original), ("restart",)])

    def test_current_mutation_prevents_any_write_or_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / "toy.env"
            env.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed after validation"):
                activate.mutate_with_rollback(env, b"old", b"new", writer=lambda *_: self.fail("write"), restart=lambda: self.fail("restart"), verify=lambda: True)

    def test_success_calls_real_dependency_seams_once(self):
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / "toy.env"
            env.write_bytes(b"old")
            calls = []
            def writer(path, value):
                calls.append("write")
                Path(path).write_bytes(value)
            result = activate.mutate_with_rollback(env, b"old", b"new", writer=writer, restart=lambda: calls.append("restart"), verify=lambda: {"ready": True})
            self.assertEqual(result, {"ready": True})
            self.assertEqual(env.read_bytes(), b"new")
            self.assertEqual(calls, ["write", "restart"])


class APIAcceptanceCompleteness(unittest.TestCase):
    def test_complete_synthetic_twenty_is_accepted(self):
        self.assertTrue(verify.validate_api_acceptance(*complete_fixture()))

    def reject(self, mutate):
        report, offline, expected = complete_fixture()
        mutate(report)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            verify.validate_api_acceptance(report, offline, expected)

    def test_top_level_pass_cannot_hide_missing_trip(self):
        self.reject(lambda r: r["datasets"]["evaluator"]["trips"].pop("toy-9"))

    def test_all_required_global_checks_must_exist(self):
        for key in verify.REQUIRED_INTEGRATION_CHECKS:
            with self.subTest(key=key):
                self.reject(lambda r: r["checks"].pop(key))

    def test_every_per_trip_check_is_revalidated(self):
        for key in verify.REQUIRED_ROW_CHECKS:
            with self.subTest(key=key):
                self.reject(lambda r: r["datasets"]["evaluator"]["trips"]["toy-9"]["checks"].pop(key))
                self.reject(lambda r: r["datasets"]["evaluator"]["trips"]["toy-9"]["checks"].__setitem__(key, False))

    def test_wrong_model_sha_cannot_pass_true_check_map(self):
        self.reject(lambda r: r.__setitem__("model_sha256", "0" * 64))
        self.reject(lambda r: r["datasets"]["evaluator"]["trips"]["toy-2"]["model"].__setitem__("sha256", "0" * 64))

    def test_wrong_config_sha_cannot_pass_true_check_map(self):
        self.reject(lambda r: r.__setitem__("config_file_sha256", "0" * 64))
        self.reject(lambda r: r["datasets"]["development"]["trips"]["toy-2"].__setitem__("match_config_hash", "0" * 64))

    def test_geometry_and_assignment_sha_recomputed_from_records(self):
        for key in ("predicted_geometry_sha256", "matched_edge_array_sha256"):
            with self.subTest(key=key):
                self.reject(lambda r: r["datasets"]["development"]["trips"]["toy-0"].__setitem__(key, "0" * 64))

    def test_probability_audit_completeness_and_finiteness(self):
        self.reject(lambda r: r["datasets"]["development"]["trips"]["toy-0"]["candidate_audit"]["checks"].clear())
        self.reject(lambda r: r["datasets"]["development"]["trips"]["toy-0"]["candidate_audit"].__setitem__("probability_sum_error_max", float("nan")))
        self.reject(lambda r: r["datasets"]["development"]["trips"]["toy-0"]["candidate_audit"].__setitem__("learned_emission_observations", 2))

    def test_duplicate_request_ids_rejected(self):
        self.reject(lambda r: r["datasets"]["evaluator"]["trips"]["toy-0"].__setitem__("request_id", "development-toy-0"))

    def test_full_config_mutation_rejected(self):
        self.reject(lambda r: r.__setitem__("frozen_config_after", {"synthetic": False}))

    def test_public_reference_and_proxy_array_guard(self):
        for payload in ({"reference_route": [1]}, {"diagnostics": {"proxy_labels": [1, 2]}}, {"metric_notes": ["/home/private/route"]}):
            with self.subTest(payload=payload), self.assertRaises((ValueError, RuntimeError)):
                verify.assert_public_payload(payload)

    def test_nonfinite_metrics_rejected(self):
        self.reject(lambda r: r["datasets"]["evaluator"]["trips"]["toy-2"]["metrics"].__setitem__("log_loss", float("inf")))


class UnknownDatasetEarlyBoundary(unittest.TestCase):
    def test_unknown_id_never_reads_body_or_calls_fastapi(self):
        from backend.boundary import PublicBoundary
        async def check():
            messages = []
            async def must_not_receive():
                raise AssertionError("unknown dataset request body was read")
            async def must_not_call_app(*_):
                raise AssertionError("unknown dataset reached FastAPI")
            async def send(message):
                messages.append(message)
            scope = {"type": "http", "method": "POST", "path": "/datasets/untrustedcanary/match", "scheme": "https", "headers": [(b"host", b"example.invalid"), (b"content-length", b"99999999"), (b"content-type", b"application/octet-stream"), (b"origin", b"https://untrusted.invalid")], "query_string": b""}
            await PublicBoundary(must_not_call_app)(scope, must_not_receive, send)
            self.assertEqual(messages[0]["status"], 404)
            headers = dict(messages[0]["headers"])
            self.assertIn(b"no-store", headers[b"cache-control"])
            self.assertIn(b"x-request-id", headers)
            payload = json.loads(b"".join(m.get("body", b"") for m in messages[1:]))
            self.assertNotIn("untrustedcanary", json.dumps(payload))
        asyncio.run(check())


class EvidenceGuards(unittest.TestCase):
    def test_png_dimensions_are_actual_header_not_assumed_viewport(self):
        import struct
        from scripts.final_live_evidence import png_dimensions
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "toy.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 8 + struct.pack(">II", 320, 2110))
            self.assertEqual(png_dimensions(path), (320, 2110))
            path.write_bytes(b"not a PNG screenshot")
            with self.assertRaisesRegex(ValueError, "PNG"):
                png_dimensions(path)

    def test_all_metrics_parity_includes_nll_without_route_only_shortcut(self):
        from scripts.final_live_evidence import metric_equality
        self.assertTrue(metric_equality({"route_length_f1": .9, "log_loss": .2}, {"route_length_f1": .9, "log_loss": .2}))
        self.assertFalse(metric_equality({"route_length_f1": .9, "log_loss": .3}, {"route_length_f1": .9, "log_loss": .2}))
        self.assertFalse(metric_equality({"route_length_f1": .9}, {"route_length_f1": .9, "log_loss": .2}))
        self.assertFalse(metric_equality({"log_loss": float("nan")}, {"log_loss": .2}))

    def test_terminal_requires_same_browser_trip_model_and_request(self):
        from scripts import final_live_evidence as live
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "toy.png"
            image.write_bytes(b"synthetic bytes used only to test hash guard")
            row = {"request_id": "toy-browser", "trip_id": "toy-trip", "dataset_id": "evaluator"}
            events = [{"event": event, "request_id": row["request_id"], "trip_id": row["trip_id"],
                       "dataset_role": row["dataset_id"], "model_hash": verify.MODEL_SHA256,
                       "match_config_hash": verify.COMPACT_CONFIG_SHA256} for event in ("match_started", "match_completed")]
            terminal = {"mocked": False, "valid": True, "ocr_privacy_guard_passed": True,
                        "request_id": row["request_id"], "events": events, "image": "toy.png",
                        "image_sha256": verify.sha256(image), "journal_unit": "path-snap-api.service"}
            browser = {"responses": [row]}
            with patch.object(live, "ROOT", root):
                self.assertTrue(all(live.terminal_checks(terminal, browser).values()))
                for key, invalid in (("model_hash", "0" * 64), ("match_config_hash", "0" * 64), ("trip_id", "wrong"), ("dataset_role", "development")):
                    changed = deepcopy(terminal)
                    changed["events"][1][key] = invalid
                    self.assertFalse(live.terminal_checks(changed, browser)["original_frozen_events"])
                changed = deepcopy(terminal)
                changed["request_id"] = "not-a-browser-request"
                self.assertFalse(live.terminal_checks(changed, browser)["browser_request_identity"])
                changed = deepcopy(terminal)
                changed["events"].pop()
                self.assertFalse(live.terminal_checks(changed, browser)["one_started_and_completed"])


if __name__ == "__main__":
    unittest.main()
