"""Stateful inputs/model, request-local live inference, and isolated evaluation.

Only GPS tracks, graph inputs and the trusted fitted model survive a request.
Predicted routes, candidate audits, reference arrays and API results are never
cached. The scorer is invoked after the decoder has returned its prediction.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import re
from threading import BoundedSemaphore, RLock
import time

import numpy as np

from .config import Settings
from .emissions import AuditedEmissionModel
from .errors import ServiceError
from .geometry import observation_assignments, predicted_geojson, prediction_arrays, raw_gps_geojson
from .registry import GraphCache, TripRegistry
from .scoring import PROXY_NOTE, ScoreSummary, empty_metrics, public_score, score_prediction

LOGGER = logging.getLogger("pathsnap.api")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def log_event(event: str, **values) -> None:
    # Callers supply fixed scalar fields, never request bodies, paths, labels or geometry.
    LOGGER.info(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **values}, sort_keys=True, allow_nan=False))


def load_emission_model(path: Path):
    """Lazy import: GET /health still works while training assets are absent."""
    from mapmatch.learning import load_emission_model as loader
    return loader(path)


def create_matcher(graph, config, *, emission_model):
    from mapmatch.matcher import MapMatcher
    return MapMatcher(graph, config, emission_model=emission_model)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value, default: str) -> str:
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _+().:-]{0,95}", value):
        return value
    return default


class InferenceRuntime:
    def __init__(self, settings: Settings, *, model_loader=None, matcher_factory=None, scorer=None):
        self.settings = settings
        self.model_loader = model_loader or load_emission_model
        self.matcher_factory = matcher_factory or create_matcher
        self.scorer = scorer or score_prediction
        self.registry: TripRegistry | None = None
        self.graphs: GraphCache | None = None
        self.model = None
        self.model_hash: str | None = None
        self.model_info: dict = {"name": None, "sha256": None, "algorithm": "learned_classifier_hmm_viterbi"}
        self.match_config = None
        self.match_config_hash: str | None = None
        self.dataset_error: ServiceError | None = None
        self.config_error: ServiceError | None = None
        self.model_error: ServiceError | None = None
        self.metadata_loaded = False
        self.started = False
        self._model_lock = RLock()
        self._slots = BoundedSemaphore(settings.inference_concurrency)

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        try:
            self.registry = TripRegistry(self.settings.data_dir, self.settings.expected_trip_count)
            self.graphs = GraphCache(self.registry, self.settings.graph_cache_size)
        except ServiceError as exc:
            self.dataset_error = exc
        except Exception:
            self.dataset_error = ServiceError("dataset_invalid", "The configured dataset could not be registered.")
        self._load_config()
        self.ensure_model()
        log_event("startup", **self._trace(), status=self.health(retry_model=False)["status"], model_loaded=self.model is not None,
                  trip_count=len(self.registry) if self.registry is not None else 0,
                  dataset_error=self.dataset_error.code if self.dataset_error else None,
                  model_error=self.model_error.code if self.model_error else None,
                  config_error=self.config_error.code if self.config_error else None)

    def close(self) -> None:
        if self.graphs is not None:
            self.graphs.clear()

    def _trace(self) -> dict:
        return {"dataset_role": self.settings.dataset_role, "model_hash": self.model_hash, "match_config_hash": self.match_config_hash}

    def _load_config(self) -> None:
        try:
            from mapmatch.matcher import MatchConfig
            path = self.settings.match_config_path
            values = {}
            if path is not None and path.is_file():
                with path.open(encoding="utf-8") as source:
                    values = json.load(source)
                if not isinstance(values, dict):
                    raise ValueError("Expected an object of MatchConfig fields.")
            elif self.settings.match_config_required:
                raise FileNotFoundError("Explicit match configuration is missing.")
            config = MatchConfig(**values)
            if float(getattr(config, "classifier_weight", 1.0)) <= 0:
                raise ValueError("The public service requires a nonzero learned classifier weight.")
            serialized = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
            self.match_config = config
            self.match_config_hash = hashlib.sha256(serialized.encode()).hexdigest()
            self.config_error = None
        except Exception:
            self.match_config = None
            self.match_config_hash = None
            self.config_error = ServiceError("match_config_invalid", "The server decoder configuration is missing, invalid, or disables learned emissions.")

    def ensure_model(self) -> None:
        if self.model is not None:
            return
        with self._model_lock:
            if self.model is not None:
                return
            if not self.settings.model_path.is_file():
                self.model_error = ServiceError("model_missing", "The trained classifier artifact is unavailable. Train or install it on the server; no baseline fallback is used.")
                return
            try:
                before = self.settings.model_path.stat()
                model = self.model_loader(self.settings.model_path)
                digest = _sha256(self.settings.model_path)
                after = self.settings.model_path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("The artifact changed while being loaded; retry once the export is complete.")
                if not callable(getattr(model, "score_candidates", None)):
                    raise ValueError("The loaded classifier lacks the required emission interface.")
                metadata = dict(getattr(model, "metadata", {}) or {})
                metadata_path = self.settings.metadata_path
                self.metadata_loaded = bool(metadata)
                if metadata_path is not None and metadata_path.is_file():
                    with metadata_path.open(encoding="utf-8") as source:
                        sidecar = json.load(source)
                    if not isinstance(sidecar, dict):
                        raise ValueError("Model metadata must be a JSON object.")
                    metadata.update(sidecar)
                    self.metadata_loaded = True
                name = _safe_name(metadata.get("model_name"), "trained_candidate_classifier")
                public_metadata = {"name": name, "sha256": digest, "algorithm": "learned_classifier_hmm_viterbi"}
                feature_names = getattr(model, "feature_names", None) or metadata.get("feature_names")
                if isinstance(feature_names, (tuple, list)):
                    public_metadata["feature_count"] = len(feature_names)
                if isinstance(metadata.get("seed"), int) and not isinstance(metadata.get("seed"), bool):
                    public_metadata["seed"] = metadata["seed"]
                self._load_config()  # A model exported after startup may bring its decoder configuration.
                self.model_hash = digest
                self.model_info = public_metadata
                self.model = model
                self.model_error = None
                log_event("model_loaded", **self._trace(), model_name=name, metadata_loaded=self.metadata_loaded)
            except Exception as exc:
                self.model_error = ServiceError("model_load_failed", "The trained classifier or metadata failed validation. Check trusted server artifacts and dependency versions; no baseline fallback is used.")
                log_event("model_load_failed", **self._trace(), error_type=type(exc).__name__)

    def health(self, *, retry_model: bool = True) -> dict:
        if retry_model:
            self.ensure_model()
        errors = [error.public() for error in (self.dataset_error, self.config_error, self.model_error) if error]
        ready = self.registry is not None and self.match_config is not None and self.model is not None and not errors
        status = "ready" if ready else ("error" if self.dataset_error or self.config_error else "not_ready")
        return {
            "status": status, "model_loaded": self.model is not None,
            "model_name": self.model_info["name"], "algorithm": self.model_info["algorithm"],
            "model_hash": self.model_hash, "model_metadata_loaded": self.metadata_loaded,
            "match_config_hash": self.match_config_hash, "dataset_role": self.settings.dataset_role,
            "trip_count": len(self.registry) if self.registry is not None else 0,
            "expected_trip_count": self.settings.expected_trip_count,
            "inference_concurrency": self.settings.inference_concurrency,
            "prediction_cache": False, "errors": errors,
        }

    def trips(self) -> dict:
        if self.registry is None:
            raise self.dataset_error or ServiceError("dataset_unavailable", "The server trip registry is unavailable.")
        return {"trips": self.registry.metadata(), "count": len(self.registry), "expected_count": self.settings.expected_trip_count, "dataset_role": self.settings.dataset_role}

    def match(self, trip_id: str, request_id: str) -> dict:
        started = time.perf_counter()
        if self.registry is None:
            raise self.dataset_error or ServiceError("dataset_unavailable", "The server trip registry is unavailable.")
        trip = self.registry.get(trip_id)  # Membership precedes model or filesystem work.
        if not self._slots.acquire(blocking=False):
            raise ServiceError("inference_busy", "The server is processing another trip. Retry when that inference completes.", 429)
        try:
            self.ensure_model()
            if self.model is None:
                raise self.model_error or ServiceError("model_unavailable", "The trained classifier is unavailable.")
            if self.match_config is None or self.config_error:
                raise self.config_error or ServiceError("match_config_invalid", "The server decoder configuration is invalid.")
            if self.graphs is None:
                raise ServiceError("dataset_unavailable", "The server graph registry is unavailable.")
            log_event("match_started", request_id=request_id, trip_id=trip.trip_id, **self._trace())
            inference_started = time.perf_counter()
            graph = self.graphs.get(trip)
            audited_model = AuditedEmissionModel(self.model)
            try:
                # No reference path, array or scorer result is reachable by the matcher.
                matcher = self.matcher_factory(graph, self.match_config, emission_model=audited_model)
                result = matcher.match(trip.track)
                if audited_model.call_count < 1:
                    raise ServiceError("model_not_used", "The decoder did not call the trained classifier. No baseline route is returned.", 500)
                route, indices, edges, fractions, distances = prediction_arrays(graph, trip.track, result)
                if any(int(index) not in audited_model.records for index in indices):
                    raise ServiceError("model_audit_incomplete", "The decoder did not obtain learned emissions for every matched observation.", 500)
            except ServiceError:
                raise
            except Exception as exc:
                log_event("matching_failed", request_id=request_id, trip_id=trip.trip_id, **self._trace(), error_type=type(exc).__name__)
                raise ServiceError("matching_failed", "Live learned-model route reconstruction failed. Consult the server log using this request ID; no route was fabricated.", 500) from exc
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            scoring_started = time.perf_counter()
            try:
                # The reference store is first consulted AFTER prediction is complete.
                reference_path = self.registry.reference_path(trip, self.settings.reference_dir)
                summary = self.scorer(graph, trip.track, result, audited_model, self.match_config, reference_path)
                score = public_score(summary)
            except Exception as exc:
                log_event("scoring_failed", request_id=request_id, trip_id=trip.trip_id, **self._trace(), error_type=type(exc).__name__)
                score = public_score(ScoreSummary(empty_metrics(graph, result), "error", ["Server-side reference scoring failed; the returned prediction is live and unmodified. Accuracy metrics are unavailable.", PROXY_NOTE]))
            scoring_ms = (time.perf_counter() - scoring_started) * 1000.0
            # Explicit serialization allowlists prevent accidental labels in diagnostics.
            diagnostics = {
                "observation_count": int(len(trip.track)), "matched_observation_count": int(len(indices)),
                "route_arc_count": int(len(route)), "route_length_m": float(np.sum(graph.edge_lengths[route])),
                "break_count": int(np.sum(graph.heads[route[:-1]] != graph.tails[route[1:]])),
                "learned_emission_observations": len(audited_model.records),
                "lateral_distance_median_m": float(np.median(distances)),
                "lateral_distance_p95_m": float(np.quantile(distances, 0.95)),
                "uses_reference_route": False,
            }
            result_diagnostics = result.diagnostics if isinstance(result.diagnostics, dict) else {}
            effective_candidates = result_diagnostics.get("effective_max_candidates", self.match_config.max_candidates)
            if isinstance(effective_candidates, (int, np.integer)) and not isinstance(effective_candidates, bool) and effective_candidates > 0:
                diagnostics["effective_max_candidates"] = int(effective_candidates)
            attempts = result_diagnostics.get("candidate_retry_attempts", [])
            if isinstance(attempts, list):
                diagnostics["candidate_retry_attempt_count"] = len(attempts)
            diagnostics["requested_max_candidates"] = int(self.match_config.max_candidates)
            response = {
                "request_id": request_id, "trip_id": trip.trip_id, "dataset_role": self.settings.dataset_role,
                "predicted_route": predicted_geojson(graph, route), "raw_gps": raw_gps_geojson(trip.track),
                "observation_assignments": observation_assignments(graph, indices, edges, fractions, distances),
                **score, "diagnostics": diagnostics, "model": dict(self.model_info),
                "match_config_hash": self.match_config_hash,
                "timings": {"inference_ms": round(inference_ms, 3), "scoring_ms": round(scoring_ms, 3), "total_ms": round((time.perf_counter() - started) * 1000.0, 3)},
            }
            # No arbitrary result.diagnostics, reference metadata, paths or model internals.
            log_event("match_completed", request_id=request_id, trip_id=trip.trip_id, **self._trace(),
                      inference_ms=response["timings"]["inference_ms"], scoring_ms=response["timings"]["scoring_ms"],
                      total_ms=response["timings"]["total_ms"], metrics_status=score["metrics_status"],
                      accuracy=score["metrics"]["accuracy"], macro_f1=score["metrics"]["macro_f1"],
                      top1_accuracy=score["metrics"]["top1_accuracy"], route_length_f1=score["metrics"]["route_length_f1"],
                      proxy_reference=True)
            return response
        except ServiceError as exc:
            log_event("match_rejected", request_id=request_id, trip_id=trip.trip_id, **self._trace(), error_code=exc.code, status_code=exc.status_code)
            raise
        finally:
            self._slots.release()
