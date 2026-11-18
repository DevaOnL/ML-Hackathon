"""Direct-ASGI and tiny local-fixture review probes. No sockets or real assets."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from threading import Event
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from backend.app import create_app
from backend.boundary import PublicBoundary
from backend.config import Settings
from backend.datasets import DatasetService
from backend.errors import ServiceError
from backend.service import InferenceRuntime
from mapmatch.matcher import MapMatcher
from tests.test_core import write_fixture

HERE = Path(__file__).resolve().parent


class ToyModel:
    metadata = {"model_name": "Synthetic review classifier"}
    feature_names = ("toy",)

    def score_candidates(self, graph, track, xy, index, candidates):
        return np.full(len(candidates), 1.0 / len(candidates))

    def score_candidates_batch(self, graph, track, indices, candidates):
        return [np.full(len(c), 1.0 / len(c)) for c in candidates]


def scope(path, method="POST", headers=()):
    return {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
            "query_string": b"", "headers": [(b"host", b"testserver"), *headers],
            "client": ("127.0.0.1", 1), "server": ("testserver", 80)}


async def call(app, path, *, method="POST", headers=(), chunks=(), must_not_receive=False):
    received, sent = [], []
    pending = list(chunks)

    async def receive():
        received.append(True)
        if must_not_receive:
            raise AssertionError("The rejected request consumed its body.")
        if pending:
            body = pending.pop(0)
            return {"type": "http.request", "body": body, "more_body": bool(pending)}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await app(scope(path, method, headers), receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert dict(start["headers"])[b"cache-control"] == b"no-store, no-transform"
    return start["status"], len(received), sent


async def boundary_checks():
    inner_calls = []

    async def inner(s, receive, send):
        inner_calls.append(s["path"])
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    boundary = PublicBoundary(inner, allowed_origins=("https://allowed.invalid",))
    unknown = []
    for path in ("/datasets/unknown/match", "/datasets/../match", "/datasets/development-extra/match",
                 "/datasets//match", "/datasets/development\\bad/match"):
        for method in ("POST", "GET", "PUT", "DELETE", "OPTIONS"):
            status, received, _ = await call(boundary, path, method=method, must_not_receive=True)
            assert status == 404 and received == 0
            unknown.append([method, path, status, received])
    assert inner_calls == []

    guarded = []
    for path in ("/match", "/datasets/development/match", "/datasets/evaluator/match"):
        json_header = [(b"content-type", b"application/json")]
        cases = [([(b"content-type", b"text/plain")], 415),
                 (json_header + [(b"content-encoding", b"gzip")], 415),
                 (json_header + [(b"content-encoding", b"br")], 415),
                 (json_header + [(b"content-length", b"1025")], 413),
                 (json_header + [(b"content-length", b"-1")], 400),
                 (json_header + [(b"content-length", b"invalid")], 400),
                 (json_header + [(b"origin", b"https://attacker.invalid")], 403),
                 (json_header + [(b"origin", b"http://testserver"), (b"origin", b"http://testserver")], 403)]
        for headers, expected in cases:
            status, received, _ = await call(boundary, path, headers=headers, must_not_receive=True)
            assert status == expected and received == 0
            guarded.append([path, expected, received])
        status, received, _ = await call(boundary, path, headers=json_header, chunks=[b"a" * 600, b"b" * 600])
        assert status == 413 and received == 2
        guarded.append([path, status, received])
    assert inner_calls == []

    # Real routing, still direct ASGI with no startup or request transport.
    settings = Settings(data_dir=HERE / "absent-synthetic-input", model_path=HERE / "absent-toy-model",
                        match_config_path=None, frontend_dir=None, expected_trip_count=1, dataset_role="development")
    app = create_app(settings)
    methods = []
    for method, path in (("PUT", "/datasets/evaluator/match"), ("DELETE", "/match"),
                         ("POST", "/datasets/development/trips"), ("POST", "/datasets"),
                         ("POST", "/datasets/development/match/extra"), ("GET", "/datasets/development/match")):
        status, received, _ = await call(app, path, method=method, must_not_receive=True)
        assert status in (404, 405) and received == 0
        methods.append([method, path, status, received])
    return {"unknown_ids_before_receive": unknown, "guarded_rejections": guarded,
            "unsupported_methods_before_receive": methods}


def settings_fixture(root):
    write_fixture(root / "development", trip_id="same-id")
    write_fixture(root / "secondary", trip_id="same-id",
                  points=[(0, 0), (150, 0), (300, 0)],
                  arcs=[(0, 1), (1, 2)], samples=[(30, 1), (70, -1), (220, 1), (260, 0)])
    artifact = root / "toy.joblib"
    artifact.write_bytes(b"synthetic artifact version one; no serialized weights")
    return Settings(data_dir=root / "development", model_path=artifact, match_config_path=None,
                    frontend_dir=None, expected_trip_count=1, dataset_role="development",
                    evaluator_data_dir=root / "secondary", evaluator_expected_trip_count=1)


async def cancellation_check():
    with tempfile.TemporaryDirectory(prefix="cancel_", dir=HERE) as temporary:
        settings = settings_fixture(Path(temporary))
        entered, release, finished = Event(), Event(), Event()

        def factory(graph, config, *, emission_model):
            class BlockingMatcher:
                def match(self, track):
                    entered.set()
                    if not release.wait(10):
                        raise AssertionError("Synthetic worker was never released.")
                    return MapMatcher(graph, config, emission_model=emission_model).match(track)
            return BlockingMatcher()

        runtime = InferenceRuntime(settings, model_loader=lambda _: ToyModel(), matcher_factory=factory)
        service = DatasetService(runtime)
        service.start()
        original_match = runtime.match

        def observed_match(*args):
            try:
                return original_match(*args)
            finally:
                finished.set()

        runtime.match = observed_match
        app = create_app(settings, runtime=runtime, dataset_service=service)
        body = json.dumps({"trip_id": "same-id"}).encode()
        task = asyncio.create_task(call(app, "/datasets/development/match",
                                      headers=[(b"content-type", b"application/json")], chunks=[body]))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            blocked = False
            try:
                service.match("evaluator", "same-id", "synthetic-second-request")
            except ServiceError as exc:
                blocked = exc.status_code == 429 and exc.code == "inference_busy"
            assert blocked, "Client cancellation freed the server semaphore before the worker ended."
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 5)
            if not task.done():
                await task
        free_after_finish = runtime._slots.acquire(blocking=False)
        assert free_after_finish
        runtime._slots.release()
        service.close()
        return {"cancelled_asgi_task": True, "other_dataset_busy_until_worker_finishes": blocked,
                "semaphore_available_after_worker_finishes": free_after_finish}


def shared_model_hash_observation():
    # Deliberate trusted-file mutation after primary startup. This is separate
    # from the supported immutable-asset deployment and is not a remote attack.
    with tempfile.TemporaryDirectory(prefix="identity_", dir=HERE) as temporary:
        settings = settings_fixture(Path(temporary))
        count = []
        model = ToyModel()

        def loader(path):
            count.append(str(path))
            return model

        primary = InferenceRuntime(settings, model_loader=loader)
        service = DatasetService(primary)
        primary.start()
        primary_hash = primary.model_hash
        settings.model_path.write_bytes(b"synthetic artifact version two; changed after primary loaded")
        service.evaluator.start()
        child = service.evaluator
        observation = {"synthetic_loader_calls": len(count), "same_model_object": child.model is primary.model,
                       "same_config_object": child.match_config is primary.match_config,
                       "primary_loaded_hash": primary_hash, "secondary_reported_hash": child.model_hash,
                       "current_file_hash": hashlib.sha256(settings.model_path.read_bytes()).hexdigest(),
                       "hashes_equal": child.model_hash == primary_hash,
                       "precondition": "Trusted artifact bytes were replaced between primary and secondary startup."}
        assert observation["same_model_object"] and observation["same_config_object"] and len(count) == 1
        assert not observation["hashes_equal"], "The observed hash mismatch no longer reproduces."
        service.close()
        return observation


async def main():
    with patch("backend.service.log_event"), patch("backend.boundary.log_event"), patch("backend.datasets.log_event"):
        result = {"boundary": await boundary_checks(), "cancellation": await cancellation_check(),
                  "shared_model_identity_mutation_observation": shared_model_hash_observation(),
                  "transport": "Direct in-memory ASGI only; no socket, new HTTP request, raw input, fitted production model, or service operation."}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    asyncio.run(main())
