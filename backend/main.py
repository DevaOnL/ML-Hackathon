"""Uvicorn entry point: python -m uvicorn backend.main:app."""
from .app import create_app

app = create_app()
