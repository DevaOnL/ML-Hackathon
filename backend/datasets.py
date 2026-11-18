"""Public multi-dataset view: one frozen model and budget, per-dataset inputs.

Dataset IDs are a closed allowlist decided before any registry, model or
filesystem work, so a client can never select paths, configurations or model
files. Each dataset keeps its own trip registry and graph cache; the loaded
emission model, the decoder configuration and the inference concurrency budget
are shared with the primary runtime. No prediction or result is cached here.
"""
from __future__ import annotations

from dataclasses import replace

from .config import Settings
from .errors import ServiceError
from .service import InferenceRuntime, log_event

DATASET_IDS = ("development", "evaluator")
DATASET_LABELS = {"development": "Student development", "evaluator": "Evaluator"}


def evaluator_settings(settings: Settings) -> Settings | None:
    """Build the evaluator runtime settings, or None when not configured.

    Only an explicitly trusted evaluator_data_dir enables the partition; no
    directory is ever derived or discovered implicitly.
    """
    if settings.evaluator_data_dir is None:
        return None
    return replace(
        settings,
        data_dir=settings.evaluator_data_dir,
        reference_dir=settings.evaluator_reference_dir,
        dataset_role="evaluator",
        expected_trip_count=settings.evaluator_expected_trip_count,
    )


class _SharedStateRuntime(InferenceRuntime):
    """Evaluator runtime reusing the primary's frozen model and configuration."""

    def __init__(self, settings: Settings, primary: InferenceRuntime):
        super().__init__(settings, model_loader=self._load_primary_model,
                         matcher_factory=primary.matcher_factory, scorer=primary.scorer)
        self._primary_runtime = primary

    def _load_primary_model(self, path):
        # Exactly one frozen classifier instance is loaded for every dataset;
        # this runtime never loads or replaces the trusted artifact itself.
        primary = self._primary_runtime
        primary.ensure_model()
        if primary.model is None:
            raise RuntimeError("The shared emission model is unavailable.")
        return primary.model

    def _load_config(self) -> None:
        # One decoder configuration object serves both datasets; this runtime
        # never parses its own copy of the trusted configuration file.
        primary = self._primary_runtime
        primary.ensure_model()
        with primary._model_lock:
            self.match_config = primary.match_config
            self.match_config_hash = primary.match_config_hash
            self.config_error = primary.config_error


class DatasetService:
    """Wraps runtimes behind the dataset catalogue and dataset-scoped routes."""

    def __init__(self, primary: InferenceRuntime, *, evaluator: InferenceRuntime | None = None):
        self.primary = primary
        if evaluator is None:
            settings = evaluator_settings(primary.settings)
            if settings is not None:
                evaluator = _SharedStateRuntime(settings, primary)
        if evaluator is not None:
            # One global inference budget across datasets: a busy request on
            # development also answers 429 for the evaluator partition.
            evaluator._slots = primary._slots
        self.evaluator = evaluator

    @property
    def configured(self) -> bool:
        return self.evaluator is not None

    def start(self) -> None:
        self.primary.start()
        if self.evaluator is not None:
            self.evaluator.start()

    def close(self) -> None:
        if self.evaluator is not None:
            self.evaluator.close()
        self.primary.close()

    def runtime_for(self, dataset_id: str) -> InferenceRuntime:
        # Closed allowlist, decided before any registry or model work.
        if dataset_id == "development":
            return self.primary
        if dataset_id == "evaluator":
            if self.evaluator is None:
                raise ServiceError("dataset_not_configured", "The evaluator dataset is not configured on this server.", 503)
            return self.evaluator
        raise ServiceError("unknown_dataset", "This dataset is not available.", 404)

    def _entry(self, dataset_id: str, runtime: InferenceRuntime) -> dict:
        health = runtime.health(retry_model=False)
        return {
            "id": dataset_id,
            "label": DATASET_LABELS[dataset_id],
            "role": runtime.settings.dataset_role,
            "available": runtime.registry is not None,
            "trip_count": health["trip_count"],
            "expected_trip_count": runtime.settings.expected_trip_count,
            "status": health["status"],
        }

    def _entries(self) -> list[dict]:
        entries = [self._entry("development", self.primary)]
        if self.evaluator is None:
            # No implicit discovery: an unconfigured partition reports zero.
            entries.append({
                "id": "evaluator", "label": DATASET_LABELS["evaluator"], "role": "evaluator",
                "available": False, "trip_count": 0,
                "expected_trip_count": self.primary.settings.evaluator_expected_trip_count,
                "status": "not_configured",
            })
        else:
            entries.append(self._entry("evaluator", self.evaluator))
        return entries

    def catalogue(self) -> dict:
        entries = self._entries()
        log_event("datasets_catalogue_served", dataset_count=len(entries), evaluator_configured=self.configured)
        return {"datasets": entries}

    def global_health(self) -> dict:
        payload = self.primary.health()
        if self.evaluator is not None:
            payload["datasets"] = self._entries()
        return payload

    def trips(self, dataset_id: str) -> dict:
        runtime = self.runtime_for(dataset_id)
        payload = runtime.trips()
        payload["dataset_id"] = dataset_id
        log_event("dataset_trips_served", dataset_id=dataset_id, dataset_role=payload["dataset_role"], trip_count=payload["count"])
        return payload

    def match(self, dataset_id: str, trip_id: str, request_id: str) -> dict:
        runtime = self.runtime_for(dataset_id)
        response = runtime.match(trip_id, request_id)
        response["dataset_id"] = dataset_id
        return response
