"""Shared, read-only integrity checks for the final 22 model + one decoder charts."""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

MODEL_FILENAMES = (
    "final_five_cv_macro_f1.png", "final_five_primary_objective.png",
    "final_five_student_top1_macro_f1.png", "final_five_binary_log_loss.png",
    "final_five_log_loss_partitions.png", "final_five_selection_comparison.png",
    *(f"final_five_confusion_{run}.png" for run in (
        "logistic_regression", "knn_distance_weighted", "bagging", "gaussian_nb",
        "hist_gradient_boosting", "hist_gradient_boosting_tuned")),
    "final_five_confusion_matrices_all.png", "final_five_fold_stability.png",
    "final_five_generalization_gap.png", "final_five_timing.png",
    "final_five_tuned_vs_untuned.png", "final_five_hpo_progress.png",
    "final_five_hpo_parameters.png", "final_five_observation_edge_metrics.png",
    "final_five_continuity.png", "final_five_denominators.png",
)
DECODER_FILENAMES = ("final_five_route_ablation.png",)
EXPECTED_BY_STAGE = {"models": set(MODEL_FILENAMES), "decoder": set(DECODER_FILENAMES)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_output(path: Path, base: Path) -> Path:
    """Reject symlink outputs, including links whose targets stay within base."""
    path = path.absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"symlink output is forbidden: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"writes outside {base} are forbidden: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def png_dpi_and_size(path: Path) -> tuple[tuple[float | None, float | None], tuple[int, int]]:
    raw = path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path.name}: not a PNG file")
    width = height = None
    dpi = (None, None)
    pos = 8
    while pos + 12 <= len(raw):
        length, kind = struct.unpack(">I4s", raw[pos:pos + 8])
        body = raw[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            width, height = struct.unpack(">II", body[:8])
        elif kind == b"pHYs" and body[8:9] == b"\x01":
            dpi = tuple(value * 0.0254 for value in struct.unpack(">II", body[:8]))
        pos += length + 12
        if kind == b"IEND":
            break
    if width is None or height is None:
        raise ValueError(f"{path.name}: PNG has no IHDR")
    return dpi, (width, height)


def verify_record_integrity(record: dict, root: Path) -> None:
    """Check files, both DPI axes, dimensions, and hashes without rerendering."""
    path = root / record["path"]
    pdf = root / record["pdf_path"]
    for file, digest in ((path, record["sha256"]), (pdf, record["pdf_sha256"])):
        if not file.is_file() or file.is_symlink() or sha256_file(file) != digest:
            raise ValueError(f"figure file missing, linked, or hash drift: {file}")
    dpi, pixels = png_dpi_and_size(path)
    if record.get("dpi") != 300 or any(d is None or abs(d - 300) > 2 for d in dpi):
        raise ValueError(f"figure DPI is not 300 on both axes: {path}")
    if pixels != (record["pixels"]["width"], record["pixels"]["height"]):
        raise ValueError(f"figure pixel dimensions drift: {path}")


def verify_input_pins(node: object, root: Path) -> None:
    if isinstance(node, dict):
        if "path" in node and "sha256" in node:
            path = root / node["path"]
            if not path.is_file() or sha256_file(path) != node["sha256"]:
                raise ValueError(f"input pin missing or hash drift: {path}")
        else:
            for value in node.values():
                verify_input_pins(value, root)
    elif isinstance(node, list):
        for value in node:
            verify_input_pins(value, root)
