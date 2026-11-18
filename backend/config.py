"""Trusted local configuration; HTTP clients cannot select paths or model files."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "Student folder" / "student_test"


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _positive_integer(value: str, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _origin(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("CORS origins must be exact http(s) origins, not patterns") from exc
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.path or parsed.query
            or parsed.fragment or "*" in value or port == 0):
        raise ValueError("CORS origins must be exact http(s) origins without a path or wildcard")
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path = DEFAULT_DATA_DIR
    reference_dir: Path | None = None
    model_path: Path = ROOT / "artifacts" / "model.joblib"
    metadata_path: Path | None = None
    match_config_path: Path | None = ROOT / "artifacts" / "match_config.json"
    match_config_required: bool = False
    cors_origins: tuple[str, ...] = ()
    expected_trip_count: int = 10
    dataset_role: str | None = None
    graph_cache_size: int = 2
    inference_concurrency: int = 1
    frontend_dir: Path | None = ROOT / "frontend"

    def __post_init__(self) -> None:
        for name in ("data_dir", "model_path"):
            object.__setattr__(self, name, _path(getattr(self, name)))
        for name in ("reference_dir", "metadata_path", "match_config_path", "frontend_dir"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _path(value))
        if self.metadata_path is None:
            object.__setattr__(self, "metadata_path", self.model_path.parent / "model_metadata.json")
        role = self.dataset_role
        if role is None:
            role = "development" if self.data_dir == DEFAULT_DATA_DIR.resolve() else "evaluator"
        if role not in {"development", "evaluator"}:
            raise ValueError("PATHSNAP_DATASET_ROLE must be development or evaluator")
        object.__setattr__(self, "dataset_role", role)
        for name in ("expected_trip_count", "graph_cache_size", "inference_concurrency"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(self, "cors_origins", tuple(dict.fromkeys(_origin(v) for v in self.cors_origins)))

    @classmethod
    def from_env(cls) -> "Settings":
        model_path = _path(os.environ.get("PATHSNAP_MODEL", "artifacts/model.joblib"))
        metadata = os.environ.get("PATHSNAP_MODEL_METADATA")
        reference = os.environ.get("PATHSNAP_REFERENCE_DIR")
        match_config = os.environ.get("PATHSNAP_MATCH_CONFIG", "artifacts/match_config.json")
        return cls(
            data_dir=_path(os.environ.get("PATHSNAP_DATA_DIR", str(DEFAULT_DATA_DIR))),
            reference_dir=_path(reference) if reference else None,
            model_path=model_path,
            metadata_path=_path(metadata) if metadata else None,
            match_config_path=_path(match_config) if match_config else None,
            match_config_required="PATHSNAP_MATCH_CONFIG" in os.environ and bool(match_config),
            cors_origins=tuple(v.strip() for v in os.environ.get("PATHSNAP_CORS_ORIGINS", "").split(",") if v.strip()),
            dataset_role=os.environ.get("PATHSNAP_DATASET_ROLE") or None,
            graph_cache_size=_positive_integer(os.environ.get("PATHSNAP_GRAPH_CACHE_SIZE", "2"), "PATHSNAP_GRAPH_CACHE_SIZE"),
            inference_concurrency=_positive_integer(os.environ.get("PATHSNAP_INFERENCE_CONCURRENCY", "1"), "PATHSNAP_INFERENCE_CONCURRENCY"),
        )
