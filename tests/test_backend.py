"""In-process API boundary and server-private scoring verification; no sockets."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient
import numpy as np

from backend.app import create_app
from backend.config import Settings
from backend.emissions import AuditedEmissionModel
from backend.scoring import ScoreSummary, score_prediction
from backend.service import InferenceRuntime
from mapmatch.candidates import CandidateIndex
from mapmatch.matcher import MapMatcher
from tests.test_core import metric_graph, metric_track, write_fixture

PRIVATE_SENTINEL = "PRIVATE_REFERENCE_SENTINEL_83925177"


class UniformLearnedModel:
    metadata = {"model_name": "Synthetic classifier", "seed": 73,
                "reference_route": PRIVATE_SENTINEL, "training_path": "/private/server/labels"}
    feature_names = ("synthetic_feature",)

    def __init__(self):
        self.batch_calls = 0
        self.scalar_calls = 0

    def score_candidates(self, graph, track, xy, index, candidates):
        self.scalar_calls += 1
        return np.ones(len(candidates), dtype=float) / len(candidates)

    def score_candidates_batch(self, graph, track, indices, candidates):
        self.batch_calls += 1
        return [np.ones(len(candidate), dtype=float) / len(candidate) for candidate in candidates]


@contextmanager
def api_fixture(*, scorer=None, factory=None, model_present=True, with_reference=True,
                expected_count=1, frontend=True, model_loader=None):
    with tempfile.TemporaryDirectory(prefix="private api ") as temporary:
        root = Path(temporary)
        data = root / "dataset"
        prefix = write_fixture(data, arcs=[(0, 1), (1, 2)])
        reference_root = root / "private reference storage"
        reference = reference_root / prefix.parent.name / (prefix.name + ".route")
        if with_reference:
            reference.parent.mkdir(parents=True)
            reference.write_text("0\n1\n", encoding="utf-8")
        model_path = root / "trusted_model.joblib"
        if model_present:
            model_path.write_bytes(b"synthetic trusted artifact; loader is injected")
        web = root / "frontend"
        if frontend:
            web.mkdir()
            (web / "index.html").write_text("<!doctype html><title>Independent UI fixture</title>", encoding="utf-8")
            for name in ("styles.css", "app.mjs", "api.mjs", "favicon.svg"):
                (web / name).write_text("/* public synthetic asset */", encoding="utf-8")
            vendor = web / "vendor" / "leaflet"
            vendor.mkdir(parents=True)
            (vendor / "leaflet.js").write_text("/* public vendor */")
            (web / "private.route").write_text(PRIVATE_SENTINEL)
            (vendor / "private.route").write_text(PRIVATE_SENTINEL)
            if with_reference:
                (web / "escape.mjs").symlink_to(reference)
                (vendor / "escape.js").symlink_to(reference)
        settings = Settings(data_dir=data, reference_dir=reference_root, model_path=model_path,
                            match_config_path=None, metadata_path=None, expected_trip_count=expected_count,
                            frontend_dir=web if frontend else None,
                            cors_origins=("https://evaluator.example",), dataset_role="development")
        model = UniformLearnedModel()
        runtime = InferenceRuntime(settings, model_loader=model_loader or (lambda path: model),
                                   matcher_factory=factory, scorer=scorer)
        app = create_app(settings, runtime=runtime)
        with mock.patch("backend.service.log_event"), mock.patch("backend.boundary.log_event"):
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client, runtime, model, root, reference


class ApiBoundaryTests(unittest.TestCase):
    def assert_safe_headers(self, response):
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertRegex(response.headers["x-request-id"], r"^[0-9a-f]{32}$")
        self.assertIn("frame-ancestors", response.headers["content-security-policy"])

    def test_health_and_trip_metadata_never_read_reference(self):
        with mock.patch("backend.scoring.load_route", side_effect=AssertionError("startup/list opened reference")):
            with api_fixture() as (client, runtime, model, root, reference):
                health = client.get("/health")
                trips = client.get("/trips")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json()["status"], "ready")
                self.assertEqual(health.json()["trip_count"], 1)
                self.assertFalse(health.json()["prediction_cache"])
                self.assertEqual(trips.json()["count"], 1)
                self.assertEqual(trips.json()["trips"][0]["trip_id"], "00000001")
                self.assertEqual(set(trips.json()["trips"][0]), {"trip_id", "observation_count", "duration_s"})
                for response in (health, trips):
                    self.assert_safe_headers(response)
                    self.assertNotIn(str(root), response.text)
                    self.assertNotIn(PRIVATE_SENTINEL, response.text)
                self.assertEqual(model.batch_calls + model.scalar_calls, 0)

    def test_fresh_learned_inference_precedes_reference_scoring_and_filters_internals(self):
        events = []
        def factory(graph, config, *, emission_model):
            class TracedMatcher:
                def match(self, track):
                    events.append("inference_started")
                    result = MapMatcher(graph, config, emission_model=emission_model).match(track)
                    result.diagnostics["reference_route"] = PRIVATE_SENTINEL
                    result.diagnostics["server_path"] = "/private/server/path"
                    events.append("inference_completed")
                    return result
            return TracedMatcher()
        def scorer(graph, track, result, model, config, reference):
            self.assertEqual(events[-1], "inference_completed")
            self.assertEqual(reference.read_text(), "0\n1\n")
            self.assertEqual(len(model.records), len(result.observation_indices))
            events.append("reference_read")
            return ScoreSummary({"accuracy": 0.75, "macro_f1": 0.5, "log_loss": math.inf,
                                 "reference_route": PRIVATE_SENTINEL, "reference_geometry": [[77, 88]]},
                                "ok", ["Synthetic aggregate-only metrics.", PRIVATE_SENTINEL, "/private/server/reference.route"])
        with api_fixture(scorer=scorer, factory=factory) as (client, runtime, model, root, reference):
            responses = [client.post("/match", json={"trip_id": "00000001"}) for _ in range(2)]
            self.assertEqual(events, ["inference_started", "inference_completed", "reference_read"] * 2)
            self.assertEqual(model.batch_calls, 2, "each request must perform fresh batched inference")
            self.assertEqual(model.scalar_calls, 0)
            for response in responses:
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["metrics"]["accuracy"], 0.75)
                self.assertIsNone(body["metrics"]["log_loss"])
                self.assertEqual(body["diagnostics"]["learned_emission_observations"], 4)
                self.assertFalse(body["diagnostics"]["uses_reference_route"])
                self.assertEqual(body["diagnostics"]["break_count"], 0)
                self.assertNotIn(PRIVATE_SENTINEL, response.text)
                self.assertNotIn("reference_geometry", response.text)
                self.assertNotIn(str(root), response.text)
                self.assertNotIn("/private/server", response.text)
                self.assert_safe_headers(response)
                self.assertEqual(body["request_id"], response.headers["x-request-id"])
                self.assertEqual(body["model"]["sha256"], hashlib.sha256(runtime.settings.model_path.read_bytes()).hexdigest())
                self.assertEqual(body["predicted_route"]["type"], "FeatureCollection")
                self.assertEqual(len(body["observation_assignments"]), 4)
                self.assertTrue(all(value >= 0 for value in body["timings"].values()))
            self.assertNotEqual(responses[0].json()["request_id"], responses[1].json()["request_id"])

    def test_default_server_scoring_returns_only_aggregate_reference_metrics(self):
        with api_fixture() as (client, runtime, model, root, reference):
            response = client.post("/match", json={"trip_id": "00000001"})
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(body["metrics_status"], "ok")
            self.assertTrue(body["proxy_reference"])
            self.assertEqual(body["metrics"]["accuracy"], 1)
            self.assertEqual(body["metrics"]["route_length_f1"], 1)
            self.assertEqual(body["metrics"]["candidate_miss_rate"], 0)
            self.assertNotIn("reference_route", body)
            self.assertNotIn("reference_edges", body)
            self.assertTrue(any("proxy" in note for note in body["metric_notes"]))

    def test_missing_reference_leaves_live_prediction_but_accuracy_unavailable(self):
        with api_fixture(with_reference=False) as (client, runtime, model, root, reference):
            response = client.post("/match", json={"trip_id": "00000001"})
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(body["metrics_status"], "unavailable")
            self.assertIsNone(body["metrics"]["accuracy"])
            self.assertEqual(body["metrics"]["route_continuity"], 1)
            self.assertEqual(model.batch_calls, 1)

    def test_reference_scoring_error_never_replaces_live_prediction_or_leaks_details(self):
        def broken_scorer(*args):
            raise RuntimeError(PRIVATE_SENTINEL + "/private/reference/path")
        with api_fixture(scorer=broken_scorer) as (client, runtime, model, root, reference):
            response = client.post("/match", json={"trip_id": "00000001"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["metrics_status"], "error")
            self.assertIsNone(response.json()["metrics"]["accuracy"])
            self.assertNotIn(PRIVATE_SENTINEL, response.text)
            self.assertNotIn("/private/reference", response.text)
            self.assertEqual(len(response.json()["predicted_route"]["features"]), 1)

    def test_missing_model_is_503_with_no_geometry_fallback(self):
        with api_fixture(model_present=False) as (client, runtime, model, root, reference):
            self.assertEqual(client.get("/health").json()["status"], "not_ready")
            response = client.post("/match", json={"trip_id": "00000001"})
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("predicted_route", response.json())
            self.assertNotIn(str(root), response.text)
            self.assertEqual(model.batch_calls, 0)
            self.assert_safe_headers(response)

    def test_matcher_cannot_bypass_learned_model_audit(self):
        def baseline_factory(graph, config, *, emission_model):
            return MapMatcher(graph, config)
        with api_fixture(factory=baseline_factory) as (client, runtime, model, root, reference):
            response = client.post("/match", json={"trip_id": "00000001"})
            self.assertEqual(response.status_code, 500, response.text)
            self.assertEqual(response.json()["detail"]["code"], "model_not_used")
            self.assertNotIn("predicted_route", response.json())

    def test_partial_emission_audit_is_rejected_before_scoring(self):
        def incomplete_factory(graph, config, *, emission_model):
            class Incomplete:
                def match(self, track):
                    result = MapMatcher(graph, config).match(track)
                    xy = graph.project(track[:, :2])
                    candidates = CandidateIndex(graph).query(xy[0], 80, 12)
                    emission_model.score_candidates(graph, track, xy, 0, candidates)
                    return result
            return Incomplete()
        def forbidden_scorer(*args):
            raise AssertionError("incomplete learned inference must never score a reference")
        with api_fixture(factory=incomplete_factory, scorer=forbidden_scorer) as (client, runtime, model, root, reference):
            response = client.post("/match", json={"trip_id": "00000001"})
            self.assertEqual(response.status_code, 500, response.text)
            self.assertEqual(response.json()["detail"]["code"], "model_audit_incomplete")
            self.assertNotIn("predicted_route", response.json())

    def test_invalid_extra_and_path_trip_inputs_are_sanitized_before_model_calls(self):
        with api_fixture() as (client, runtime, model, root, reference):
            for payload in ({}, {"trip_id": "../private"}, {"trip_id": "/etc/passwd"},
                            {"trip_id": ".."}, {"trip_id": 1}, {"trip_id": True},
                            {"trip_id": "00000001", "reference_route": PRIVATE_SENTINEL},
                            {"trip_id": "évil"}, {"trip_id": "x" * 65},
                            {"trip_id": "00000001\n"}, []):
                response = client.post("/match", json=payload)
                with self.subTest(payload=payload):
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertNotIn(PRIVATE_SENTINEL, response.text)
                    self.assertNotIn("/etc/passwd", response.text)
                    self.assertNotIn(str(root), response.text)
                    self.assert_safe_headers(response)
            unknown = client.post("/match", json={"trip_id": "missing"})
            self.assertEqual(unknown.status_code, 422)
            self.assertEqual(unknown.json()["detail"]["code"], "invalid_trip_id")
            self.assertEqual(model.batch_calls + model.scalar_calls, 0)

    def test_content_type_size_and_content_encoding_limits(self):
        with api_fixture() as (client, runtime, model, root, reference):
            for headers, content, expected in (
                ({"Content-Type": "text/plain"}, "{}", 415),
                ({"Content-Type": "application/json", "Content-Encoding": "gzip"}, "{}", 415),
                ({"Content-Type": "application/json"}, "{" + "x" * 2000, 413),
                ({"Content-Type": "application/json"}, "not json", 422),
            ):
                with self.subTest(headers=headers, expected=expected):
                    response = client.post("/match", headers=headers, content=content)
                    self.assertEqual(response.status_code, expected, response.text)
                    self.assert_safe_headers(response)
            self.assertEqual(model.batch_calls + model.scalar_calls, 0)

    def test_excess_concurrency_returns_retryable_busy_instead_of_duplicate_work(self):
        with api_fixture() as (client, runtime, model, root, reference):
            self.assertTrue(runtime._slots.acquire(blocking=False))
            try:
                response = client.post("/match", json={"trip_id": "00000001"})
            finally:
                runtime._slots.release()
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.headers["retry-after"], "3")
            self.assertEqual(model.batch_calls, 0)
            healthy = client.post("/match", json={"trip_id": "00000001"})
            self.assertEqual(healthy.status_code, 200, healthy.text)

    def test_post_origin_must_be_same_origin_or_exact_explicit_allowlist(self):
        with api_fixture() as (client, runtime, model, root, reference):
            for origin in ("https://attacker.invalid", "https://evaluator.example.attacker.invalid", "null",
                           "http://testserver.attacker.invalid", "http://testserver:80.attacker.invalid",
                           "http://testserver:0", "http://testserver/", "http://testserver other",
                           "http://testserver, https://attacker.invalid"):
                response = client.post("/match", json={"trip_id": "00000001"}, headers={"Origin": origin})
                with self.subTest(origin=origin):
                    self.assertEqual(response.status_code, 403, response.text)
                    self.assert_safe_headers(response)
                    self.assertNotIn("access-control-allow-origin", response.headers)
                    self.assertEqual(model.batch_calls, 0)
            for origin in ("http://testserver", "https://evaluator.example"):
                response = client.post("/match", json={"trip_id": "00000001"}, headers={"Origin": origin})
                self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(model.batch_calls, 2)

    def test_cors_is_exact_allowlist_not_wildcard(self):
        with api_fixture() as (client, runtime, model, root, reference):
            allowed = client.options("/match", headers={"Origin": "https://evaluator.example",
                                     "Access-Control-Request-Method": "POST",
                                     "Access-Control-Request-Headers": "content-type"})
            self.assertEqual(allowed.status_code, 200)
            self.assertEqual(allowed.headers["access-control-allow-origin"], "https://evaluator.example")
            self.assertNotIn("access-control-allow-credentials", allowed.headers)
            rejected = client.options("/match", headers={"Origin": "https://evaluator.example.attacker.invalid",
                                      "Access-Control-Request-Method": "POST"})
            self.assertEqual(rejected.status_code, 400)
            self.assertNotIn("access-control-allow-origin", rejected.headers)
            self.assert_safe_headers(rejected)

    def test_only_allowlisted_frontend_assets_serve_no_data_docs_listing_or_traversal(self):
        blocked = ("/docs", "/redoc", "/openapi.json", "/private.route", "/dataset", "/dataset/",
                   "/Student%20folder/student_test/00000001/00000001.route", "/.git/config",
                   "/.env", "/artifacts/model.joblib", "/analysis/data_audit/summary.json",
                   "/vendor/", "/vendor/leaflet/", "/vendor/leaflet/private.route",
                   "/vendor/leaflet/escape.js", "/escape.mjs", "/%2e%2e/private.route",
                   "/%2e%2e%2fprivate.route", "/vendor/leaflet/../../../private.route",
                   "/vendor/leaflet/%2e%2e/%2e%2e/private.route", r"/vendor\leaflet\leaflet.js")
        with api_fixture() as (client, runtime, model, root, reference):
            for path in ("/", "/index.html", "/styles.css", "/app.mjs", "/api.mjs", "/favicon.svg", "/vendor/leaflet/leaflet.js"):
                response = client.get(path)
                self.assertEqual(response.status_code, 200, (path, response.text))
                self.assert_safe_headers(response)
            for path in blocked:
                response = client.get(path)
                with self.subTest(path=path):
                    self.assertEqual(response.status_code, 404, response.text)
                    self.assertNotIn(PRIVATE_SENTINEL, response.text)
                    self.assertNotIn(str(root), response.text)
                    self.assert_safe_headers(response)

    def test_dataset_count_mismatch_is_not_ready_and_never_silently_omits_trips(self):
        with api_fixture(expected_count=10) as (client, runtime, model, root, reference):
            health = client.get("/health").json()
            self.assertEqual(health["status"], "error")
            self.assertEqual(health["trip_count"], 0)
            self.assertEqual(health["errors"][0]["code"], "trip_count_mismatch")
            self.assertEqual(client.get("/trips").status_code, 503)


class AuditedEmissionTests(unittest.TestCase):
    def setUp(self):
        self.graph = metric_graph([(0, 0), (100, 0)], [(0, 1), (1, 0)])
        self.track = metric_track([(20, 0), (50, 0), (80, 0)])
        self.xy = self.graph.project(self.track[:, :2])
        self.candidates = [CandidateIndex(self.graph).query(point, 20, 10) for point in self.xy]

    def test_batch_audit_preserves_original_indices_and_immutable_candidate_copies(self):
        underlying = UniformLearnedModel()
        audit = AuditedEmissionModel(underlying)
        result = audit.score_candidates_batch(self.graph, self.track, [0, 2], [self.candidates[0], self.candidates[2]])
        self.assertEqual(underlying.batch_calls, 1)
        self.assertEqual(underlying.scalar_calls, 0)
        self.assertEqual(set(audit.records), {0, 2})
        self.assertEqual(audit.call_count, 2)
        for index, probabilities in zip((0, 2), result):
            saved = audit.records[index]
            np.testing.assert_array_equal(saved.edges, self.candidates[index].edges)
            np.testing.assert_array_equal(saved.probabilities, probabilities)
            self.assertFalse(saved.edges.flags.writeable)
            self.assertFalse(saved.probabilities.flags.writeable)
        result[0][0] = 0.25
        self.assertEqual(audit.records[0].probabilities[0], 0.5)
        second = AuditedEmissionModel(underlying)
        self.assertEqual(second.records, {})

    def test_audit_rejects_invalid_model_batch_and_observation_probabilities(self):
        for vector in ([0, 1], [-0.1, 1.1], [math.nan, 1], [0.2, 0.2], [0.5]):
            class BadModel:
                def score_candidates(self, *args):
                    return vector
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                AuditedEmissionModel(BadModel()).score_candidates(
                    self.graph, self.track, self.xy, 0, self.candidates[0])
        audit = AuditedEmissionModel(UniformLearnedModel())
        with self.assertRaises(ValueError):
            audit.score_candidates_batch(self.graph, self.track, [0], self.candidates)
        self.assertEqual(audit.call_count, 0)


class CategoricalScoreDefinitionTests(unittest.TestCase):
    def test_normalized_target_mass_uses_final_one_e_minus_seven_floor(self):
        graph = metric_graph([(0, 0), (100, 0)], [(0, 1), (1, 0), (0, 1)])
        track = metric_track([(50, 0)])
        result = SimpleNamespace(route=np.array([0]), observation_indices=np.array([0]),
                                 matched_edges=np.array([0]))
        with tempfile.TemporaryDirectory() as root:
            reference = Path(root) / "synthetic.route"
            reference.write_text("0\n")
            for edges, probabilities, truth, expected_mass, expected_miss in (
                ([0, 1], [1e-10, 1 - 1e-10], 0, 1e-7, 0),
                ([0, 1], [0.9, 0.1], 2, 0.9, 0),
                ([0], [1.0], 1, 1e-7, 1),
                ([0, 2, 1], [0.2, 0.3, 0.5], 0, 0.5, 0),
            ):
                model = SimpleNamespace(records={0: SimpleNamespace(
                    edges=np.asarray(edges), probabilities=np.asarray(probabilities))})
                with self.subTest(truth=truth, probabilities=probabilities), \
                        mock.patch("mapmatch.features.reference_observation_edges", return_value=np.array([truth])):
                    scored = score_prediction(graph, track, result, model, None, reference)
                self.assertAlmostEqual(scored.metrics["log_loss"], -math.log(expected_mass), places=12)
                self.assertEqual(scored.metrics["candidate_miss_rate"], expected_miss)
                self.assertTrue(any("1e-7" in note for note in scored.metric_notes))


if __name__ == "__main__":
    unittest.main()
