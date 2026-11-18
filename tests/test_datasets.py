"""Dataset catalogue API tests: synthetic fixtures only, no sockets, no real model.

The two dataset roots deliberately share one trip ID with different geometry so
per-dataset registries and graph caches stay provably separate, while the frozen
model, decoder configuration and inference budget remain shared.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient
import numpy as np

from backend.app import create_app
from backend.config import Settings
from backend.scoring import ScoreSummary, empty_metrics
from backend.service import InferenceRuntime
from mapmatch.matcher import MapMatcher
from tests.test_core import write_fixture

PRIVATE_SENTINEL = "PRIVATE_REFERENCE_SENTINEL_83925177"
# Distinct geometry for the same trip ID in the evaluator partition.
EVALUATOR_POINTS = [(0, 0), (150, 0), (300, 0)]
EVALUATOR_ARCS = [(1, 2), (0, 1), (1, 0), (2, 1)]
EVALUATOR_SAMPLES = [(30, 1), (70, -1), (220, 1), (260, 0)]


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
def dataset_api_fixture(*, evaluator=False, evaluator_reference=True, sibling=False,
                        scorer=None, factory=None, model_loader=None):
    with tempfile.TemporaryDirectory(prefix="private datasets ") as temporary:
        root = Path(temporary)
        data = root / "dataset"
        write_fixture(data, trip_id="00000001")
        reference_root = root / "private reference storage"
        development_reference = reference_root / "00000001" / "00000001.route"
        development_reference.parent.mkdir(parents=True)
        development_reference.write_text("0\n1\n", encoding="utf-8")
        model_path = root / "trusted_model.joblib"
        model_path.write_bytes(b"synthetic trusted artifact; loader is injected")
        web = root / "frontend"
        web.mkdir()
        (web / "index.html").write_text("<!doctype html><title>Independent UI fixture</title>", encoding="utf-8")
        for name in ("styles.css", "app.mjs", "api.mjs", "favicon.svg"):
            (web / name).write_text("/* public synthetic asset */", encoding="utf-8")
        evaluator_data = None
        evaluator_reference_root = None
        if evaluator:
            evaluator_data = root / "evaluator dataset"
            prefix = write_fixture(evaluator_data, trip_id="00000001",
                                   points=EVALUATOR_POINTS, arcs=EVALUATOR_ARCS, samples=EVALUATOR_SAMPLES)
            evaluator_reference_root = root / "evaluator reference storage"
            if evaluator_reference:
                evaluator_reference = evaluator_reference_root / prefix.parent.name / (prefix.name + ".route")
                evaluator_reference.parent.mkdir(parents=True)
                evaluator_reference.write_text("0\n1\n", encoding="utf-8")
        elif sibling:
            # A valid dataset directory exists on disk but is never configured:
            # nothing may be discovered or served implicitly from it.
            write_fixture(root / "unconfigured sibling", trip_id="00000001",
                          points=EVALUATOR_POINTS, arcs=EVALUATOR_ARCS, samples=EVALUATOR_SAMPLES)
        settings = Settings(data_dir=data, reference_dir=reference_root, model_path=model_path,
                            match_config_path=None, metadata_path=None, expected_trip_count=1,
                            frontend_dir=web, cors_origins=("https://evaluator.example",),
                            dataset_role="development",
                            evaluator_data_dir=evaluator_data,
                            evaluator_reference_dir=evaluator_reference_root,
                            evaluator_expected_trip_count=1 if evaluator else 10)
        model = UniformLearnedModel()
        runtime = InferenceRuntime(settings, model_loader=model_loader or (lambda path: model),
                                   matcher_factory=factory, scorer=scorer)
        app = create_app(settings, runtime=runtime)
        with mock.patch("backend.service.log_event") as service_log, \
                mock.patch("backend.boundary.log_event"), \
                mock.patch("backend.datasets.log_event") as datasets_log:
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client, runtime, model, root, app, datasets_log, service_log


class DatasetCatalogueTests(unittest.TestCase):
    def assert_safe_headers(self, response):
        self.assertIn("no-store", response.headers["cache-control"])
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertRegex(response.headers["x-request-id"], r"^[0-9a-f]{32}$")

    def test_catalogue_unset_reports_unconfigured_evaluator(self):
        with dataset_api_fixture() as (client, runtime, model, root, app, datasets_log, service_log):
            response = client.get("/datasets")
            self.assertEqual(response.status_code, 200, response.text)
            entries = response.json()["datasets"]
            self.assertEqual([entry["id"] for entry in entries], ["development", "evaluator"])
            self.assertEqual(entries[0], {"id": "development", "label": "Student development",
                                          "role": "development", "available": True, "trip_count": 1,
                                          "expected_trip_count": 1, "status": "ready"})
            self.assertEqual(entries[1], {"id": "evaluator", "label": "Evaluator",
                                          "role": "evaluator", "available": False, "trip_count": 0,
                                          "expected_trip_count": 10, "status": "not_configured"})
            self.assert_safe_headers(response)
            self.assertNotIn(str(root), response.text)

    def test_legacy_health_is_unchanged_when_evaluator_unset(self):
        with dataset_api_fixture() as (client, runtime, model, root, app, datasets_log, service_log):
            health = client.get("/health")
            self.assertEqual(health.status_code, 200, health.text)
            body = health.json()
            self.assertNotIn("datasets", body)
            self.assertEqual(body["status"], "ready")
            self.assertEqual(body["dataset_role"], "development")
            self.assertEqual(body["trip_count"], 1)

    def test_unconfigured_evaluator_requests_answer_safe_503(self):
        with dataset_api_fixture() as (client, runtime, model, root, app, datasets_log, service_log):
            trips = client.get("/datasets/evaluator/trips")
            match = client.post("/datasets/evaluator/match", json={"trip_id": "00000001"})
            for response in (trips, match):
                self.assertEqual(response.status_code, 503, response.text)
                self.assertEqual(response.json()["detail"]["code"], "dataset_not_configured")
                self.assertNotIn(str(root), response.text)
            self.assertEqual(model.batch_calls + model.scalar_calls, 0)

    def test_no_implicit_discovery_of_unconfigured_directories(self):
        with dataset_api_fixture(sibling=True) as (client, runtime, model, root, app, datasets_log, service_log):
            catalogue = client.get("/datasets").json()["datasets"]
            self.assertEqual(catalogue[1]["trip_count"], 0)
            self.assertEqual(catalogue[1]["status"], "not_configured")
            self.assertEqual(client.get("/datasets/evaluator/trips").status_code, 503)


class ConfiguredDatasetTests(unittest.TestCase):
    def assert_safe_headers(self, response):
        self.assertIn("no-store", response.headers["cache-control"])
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertRegex(response.headers["x-request-id"], r"^[0-9a-f]{32}$")

    def test_catalogue_lists_both_datasets(self):
        with dataset_api_fixture(evaluator=True) as (client, runtime, model, root, app, datasets_log, service_log):
            response = client.get("/datasets")
            self.assertEqual(response.status_code, 200, response.text)
            entries = response.json()["datasets"]
            self.assertEqual([entry["id"] for entry in entries], ["development", "evaluator"])
            self.assertEqual(entries[1], {"id": "evaluator", "label": "Evaluator",
                                          "role": "evaluator", "available": True, "trip_count": 1,
                                          "expected_trip_count": 1, "status": "ready"})
            self.assertEqual(entries[0]["status"], "ready")
            self.assert_safe_headers(response)
            self.assertNotIn(str(root), response.text)
            self.assertNotIn(PRIVATE_SENTINEL, response.text)

    def test_health_includes_dataset_summaries_once_configured(self):
        with dataset_api_fixture(evaluator=True) as (client, runtime, model, root, app, datasets_log, service_log):
            body = client.get("/health").json()
            self.assertEqual(body["status"], "ready")
            self.assertEqual([entry["id"] for entry in body["datasets"]], ["development", "evaluator"])
            self.assertEqual(body["datasets"][1]["status"], "ready")

    def test_trips_reports_dataset_identity_with_shared_trip_id(self):
        with dataset_api_fixture(evaluator=True) as (client, runtime, model, root, app, datasets_log, service_log):
            for dataset_id, role in (("development", "development"), ("evaluator", "evaluator")):
                response = client.get(f"/datasets/{dataset_id}/trips")
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["dataset_id"], dataset_id)
                self.assertEqual(body["dataset_role"], role)
                self.assertEqual(body["count"], 1)
                self.assertEqual(body["trips"][0]["trip_id"], "00000001")
                self.assert_safe_headers(response)
                self.assertNotIn(str(root), response.text)
            datasets_log.assert_any_call("dataset_trips_served", dataset_id="evaluator",
                                         dataset_role="evaluator", trip_count=1)

    def test_match_uses_per_dataset_graphs_for_the_same_trip_id(self):
        with dataset_api_fixture(evaluator=True) as (client, runtime, model, root, app, datasets_log, service_log):
            development = client.post("/datasets/development/match", json={"trip_id": "00000001"})
            evaluator = client.post("/datasets/evaluator/match", json={"trip_id": "00000001"})
            self.assertEqual(development.status_code, 200, development.text)
            self.assertEqual(evaluator.status_code, 200, evaluator.text)
            development_body, evaluator_body = development.json(), evaluator.json()
            self.assertEqual(development_body["dataset_id"], "development")
            self.assertEqual(development_body["dataset_role"], "development")
            self.assertEqual(evaluator_body["dataset_id"], "evaluator")
            self.assertEqual(evaluator_body["dataset_role"], "evaluator")
            # Same trip ID, different partitions: the raw GPS geometry proves
            # each dataset matched its own fixture rather than a shared cache.
            development_coordinates = development_body["raw_gps"]["features"][0]["geometry"]["coordinates"]
            evaluator_coordinates = evaluator_body["raw_gps"]["features"][0]["geometry"]["coordinates"]
            self.assertNotEqual(development_coordinates, evaluator_coordinates)
            for body in (development_body, evaluator_body):
                self.assertIn("metrics_status", body)
                self.assertNotIn(str(root), str(body))
            # One shared frozen model instance served both datasets.
            self.assertEqual(model.batch_calls, 2)
            self.assertEqual(model.scalar_calls, 0)

    def test_match_scoring_uses_real_geometry_with_reference(self):
        with dataset_api_fixture(evaluator=True) as (client, runtime, model, root, app, datasets_log, service_log):
            response = client.post("/datasets/evaluator/match", json={"trip_id": "00000001"})
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertIn(body["metrics_status"], {"ok", "partial"})
            self.assertIsNotNone(body["metrics"]["accuracy"])
            self.assertTrue(body["proxy_reference"])

    def test_match_without_reference_reports_unavailable_metrics(self):
        with dataset_api_fixture(evaluator=True, evaluator_reference=False) as (client, runtime, model, root, app, datasets_log, service_log):
            response = client.post("/datasets/evaluator/match", json={"trip_id": "00000001"})
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(body["metrics_status"], "unavailable")
            self.assertIsNone(body["metrics"]["accuracy"])
            self.assertEqual(model.batch_calls, 1, "prediction must run even without a reference")

    def test_reference_is_consulted_only_after_prediction(self):
        events = []

        def factory(graph, config, *, emission_model):
            class TracedMatcher:
                def match(self, track):
                    events.append("inference_started")
                    result = MapMatcher(graph, config, emission_model=emission_model).match(track)
                    events.append("inference_completed")
                    return result
            return TracedMatcher()

        def scorer(graph, track, result, audited_model, config, reference):
            self.assertEqual(events[-1], "inference_completed")
            self.assertEqual(model.batch_calls, 1, "inference must complete before the private store is read")
            events.append("reference_read")
            self.assertIsNotNone(reference)
            self.assertEqual(reference.read_text(), "0\n1\n")
            return ScoreSummary(empty_metrics(graph, result), "ok", [])

        with dataset_api_fixture(evaluator=True, factory=factory, scorer=scorer) as (client, runtime, model, root, app, datasets_log, service_log):
            response = client.post("/datasets/evaluator/match", json={"trip_id": "00000001"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(events, ["inference_started", "inference_completed", "reference_read"])

    def test_unknown_dataset_ids_and_traversal_are_rejected(self):
        with dataset_api_fixture(evaluator=True) as (client, runtime, model, root, app, datasets_log, service_log):
            for method, path in (
                ("GET", "/datasets/unknown/trips"),
                ("POST", "/datasets/unknown/match"),
                ("GET", "/datasets/%2e%2e/trips"),
                ("POST", "/datasets/%2e%2e/match"),
                ("GET", "/datasets/development/extra/trips"),
                ("GET", "/datasets/development/%2e%2e/trips"),
            ):
                with self.subTest(method=method, path=path):
                    response = client.request(method, path, json={"trip_id": "00000001"} if method == "POST" else None)
                    self.assertIn(response.status_code, (404, 422), response.text)
                    if response.status_code == 404:
                        self.assertIn(response.json()["detail"]["code"], {"unknown_dataset", "http_404"})
            self.assertEqual(model.batch_calls + model.scalar_calls, 0)

    def test_wrong_methods_are_rejected(self):
        with dataset_api_fixture(evaluator=True) as (client, runtime, model, root, app, datasets_log, service_log):
            for method, path in (
                ("GET", "/datasets/development/match"),
                ("POST", "/datasets/development/trips"),
                ("POST", "/datasets"),
                ("POST", "/datasets/development/match/extra"),
                ("DELETE", "/datasets/evaluator/trips"),
                ("PUT", "/datasets/evaluator/match"),
            ):
                with self.subTest(method=method, path=path):
                    response = client.request(method, path, json={"trip_id": "00000001"})
                    # 405 when only the method is wrong; GETs on POST-only paths
                    # fall through to the static allowlist and answer 404.
                    self.assertIn(response.status_code, (404, 405), response.text)
                    self.assertIn(response.json()["detail"]["code"], ("http_404", "http_405"))
            self.assertEqual(model.batch_calls + model.scalar_calls, 0)

    def test_match_bodies_are_strict_on_dataset_routes(self):
        with dataset_api_fixture(evaluator=True) as (client, runtime, model, root, app, datasets_log, service_log):
            for payload in (
                {"trip_id": "00000001", "reference_route": PRIVATE_SENTINEL},
                {},
                {"trip_id": 5},
                {"trip_id": "../private"},
                {"trip_id": "99999999"},
            ):
                with self.subTest(payload=payload):
                    response = client.post("/datasets/development/match", json=payload)
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertIn(response.json()["detail"]["code"], {"invalid_request", "invalid_trip_id"})
                    self.assertNotIn(PRIVATE_SENTINEL, response.text)
            self.assertEqual(model.batch_calls + model.scalar_calls, 0)

    def test_boundary_limits_apply_to_dataset_match(self):
        with dataset_api_fixture(evaluator=True) as (client, runtime, model, root, app, datasets_log, service_log):
            for headers, content, expected in (
                ({"Content-Type": "text/plain"}, "{}", 415),
                ({"Content-Type": "application/json", "Content-Encoding": "gzip"}, "{}", 415),
                ({"Content-Type": "application/json"}, "{" + "x" * 2000, 413),
                ({"Content-Type": "application/json"}, "not json", 422),
            ):
                with self.subTest(headers=headers, expected=expected):
                    response = client.post("/datasets/evaluator/match", headers=headers, content=content)
                    self.assertEqual(response.status_code, expected, response.text)
            self.assertEqual(model.batch_calls + model.scalar_calls, 0)

    def test_cross_origin_is_denied_before_inference(self):
        with dataset_api_fixture(evaluator=True) as (client, runtime, model, root, app, datasets_log, service_log):
            denied = client.post("/datasets/evaluator/match", json={"trip_id": "00000001"},
                                 headers={"Origin": "https://evil.example"})
            self.assertEqual(denied.status_code, 403, denied.text)
            self.assertEqual(denied.json()["detail"]["code"], "origin_not_allowed")
            self.assertEqual(model.batch_calls + model.scalar_calls, 0)
            allowed = client.post("/datasets/evaluator/match", json={"trip_id": "00000001"},
                                  headers={"Origin": "https://evaluator.example"})
            self.assertEqual(allowed.status_code, 200, allowed.text)

    def test_one_global_inference_budget_across_datasets(self):
        with dataset_api_fixture(evaluator=True) as (client, runtime, model, root, app, datasets_log, service_log):
            evaluator_runtime = app.state.datasets.evaluator
            self.assertIsNotNone(evaluator_runtime)
            self.assertIs(evaluator_runtime._slots, runtime._slots)
            self.assertTrue(runtime._slots.acquire(blocking=False))
            try:
                for path in ("/match", "/datasets/development/match", "/datasets/evaluator/match"):
                    with self.subTest(path=path):
                        response = client.post(path, json={"trip_id": "00000001"})
                        self.assertEqual(response.status_code, 429, response.text)
                        self.assertEqual(response.json()["detail"]["code"], "inference_busy")
                        self.assertEqual(response.headers.get("retry-after"), "3")
            finally:
                runtime._slots.release()
            recovered = client.post("/datasets/evaluator/match", json={"trip_id": "00000001"})
            self.assertEqual(recovered.status_code, 200, recovered.text)
            self.assertEqual(model.batch_calls, 1)

    def test_model_and_configuration_are_shared_not_duplicated(self):
        with dataset_api_fixture(evaluator=True) as (client, runtime, model, root, app, datasets_log, service_log):
            evaluator_runtime = app.state.datasets.evaluator
            client.get("/health")
            self.assertIs(evaluator_runtime.model, runtime.model)
            self.assertIs(evaluator_runtime.match_config, runtime.match_config)
            self.assertEqual(evaluator_runtime.match_config_hash, runtime.match_config_hash)
            self.assertIsNot(evaluator_runtime.registry, runtime.registry)
            self.assertIsNot(evaluator_runtime.graphs, runtime.graphs)

    def test_dataset_logs_carry_roles_without_paths(self):
        captured = []

        def record(event, **values):
            captured.append((event, values))

        with dataset_api_fixture(evaluator=True) as (client, runtime, model, root, app, datasets_log, service_log):
            service_log.side_effect = record
            client.post("/datasets/evaluator/match", json={"trip_id": "00000001"})
        started = [values for event, values in captured if event == "match_started"]
        self.assertTrue(any(values.get("dataset_role") == "evaluator" and values.get("trip_id") == "00000001"
                            for values in started))
        for _, values in captured:
            for value in values.values():
                self.assertNotIn(str(root), str(value))


if __name__ == "__main__":
    unittest.main()
