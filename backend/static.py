"""Shared exact public-asset policy for same-origin and optional Laptop-2 hosting."""
from __future__ import annotations

from pathlib import Path

STATIC_TOP_LEVEL = {"index.html", "styles.css", "app.mjs", "api.mjs", "favicon.svg"}
STATIC_VENDOR_SUFFIXES = {".js", ".css", ".png"}
STATIC_MEDIA_TYPES = {".html": "text/html", ".css": "text/css", ".mjs": "text/javascript", ".js": "text/javascript", ".svg": "image/svg+xml", ".png": "image/png"}


def public_asset(root: Path | None, asset_path: str) -> tuple[Path, str] | None:
    if root is None:
        return None
    root = root.resolve()
    relative = Path(asset_path or "index.html")
    if relative.is_absolute() or any(part in {".", ".."} or part.startswith(".") for part in relative.parts) or "\\" in asset_path:
        return None
    top_level = relative.as_posix() in STATIC_TOP_LEVEL
    vendor = relative.parts[:2] == ("vendor", "leaflet") and relative.suffix.lower() in STATIC_VENDOR_SUFFIXES
    if not (top_level or vendor):
        return None
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file() or target.suffix.lower() not in STATIC_MEDIA_TYPES:
        return None
    return target, STATIC_MEDIA_TYPES[target.suffix.lower()]
