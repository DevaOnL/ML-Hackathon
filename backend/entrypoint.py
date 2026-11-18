"""Select an explicitly configured, accepted accuracy release; preserve v1 defaults."""
from pathlib import Path
from .config import Settings

ROOT=Path(__file__).resolve().parents[1]


def create_configured_app():
    settings=Settings.from_env()
    if settings.model_path.resolve()==ROOT/"artifacts/accuracy_improvement/model.joblib":
        from .accuracy_runtime import create_accuracy_app
        return create_accuracy_app()
    from .app import create_app
    return create_app()
