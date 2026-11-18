"""Uvicorn entry point: python -m uvicorn backend.main:app."""
from .entrypoint import create_configured_app

app = create_configured_app()
