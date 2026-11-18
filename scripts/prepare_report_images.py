#!/usr/bin/env python3
"""Prepare lossless, original-pixel report details from genuine frozen screenshots.

Default: read-only verification. --create refuses existing crops or manifests.
Nothing is resized, annotated, reflowed, recolored, recaptured, or redrawn.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/final_five/report_images"
SHOTS = ROOT / "reports/final_five/live/screenshots"
BROWSER = ROOT / "reports/final_five/live/browser_evidence.json"
TERMINAL = ROOT / "reports/final_five/live/api_terminal_evidence.json"
MANIFEST = OUT / "manifest.json"
PEER_OCR = ROOT / "analysis/final_report_layout/crop_coordinate_ocr"

# (name, immutable screenshot stem, half-open original-pixel box, purpose)
SPECS = (
    ("development_dataset_controls", "ui_desktop", (32, 300, 720, 415), "Selected Student development dataset tab and original catalogue context."),
    ("development_selected_run", "ui_desktop", (48, 900, 354, 1075), "Selected development trip and genuine run control."),
    ("development_scores", "ui_desktop", (385, 1090, 890, 1345), "Development live result cards with HMM accuracy, continuity and macro F1."),
    ("development_execution", "ui_desktop", (890, 1485, 1385, 1735), "Development execution details: model identity, full hash, request ID and dataset."),
    ("evaluator_dataset_controls", "ui_evaluator", (715, 300, 1408, 415), "Selected evaluator dataset tab and separate catalogue context."),
    ("evaluator_scores", "ui_evaluator", (385, 1090, 890, 1345), "Evaluator live result cards, shown separately from Student development."),
    ("light_scores", "ui_light", (385, 1090, 890, 1345), "The same genuine development score region in the captured light theme."),
    ("mobile_390_selected_run", "ui_mobile", (16, 1125, 374, 1315), "390-pixel mobile selected trip and run control."),
    ("mobile_390_scores", "ui_mobile", (16, 1900, 374, 2410), "390-pixel mobile evaluator result header and first four result cards."),
    ("mobile_320_scores", "ui_mobile_320", (16, 2095, 304, 2610), "320-pixel mobile evaluator result header and first four result cards."),
    ("terminal_identity", "api_terminal", (0, 120, 1120, 380), "Authentic request ID, trip, dataset, full model and config hashes. Wider crop preserves the complete original hash lines."),
    ("terminal_completion", "api_terminal", (0, 395, 970, 630), "Authentic started/completed journal events, metrics status and inference/scoring/total timings."),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path):
    return Path(path).absolute().relative_to(ROOT).as_posix()


def safe_output(path, base=None):
    base = OUT if base is None else Path(base)
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink output forbidden")
    require(path.resolve().is_relative_to(base.resolve()), "output outside report_images is forbidden")
    return path


def validate_box(box, size):
    require(len(box) == 4 and all(type(n) is int for n in box), "crop box must contain four integers")
    left, top, right, bottom = box
    require(0 <= left < right <= size[0] and 0 <= top < bottom <= size[1], "crop box outside source bounds")


def pixel_digest(image):
    header = json.dumps({"mode": image.mode, "size": image.size}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + image.tobytes()).hexdigest()


def verify_source(path, expected_hash):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "source is missing or linked")
    require(sha(path) == expected_hash, "source hash mismatch")


def crop_image(source, expected_hash, box, target, base=None):
    verify_source(source, expected_hash)
    target = safe_output(target, base)
    require(not target.exists(), "refusing to overwrite existing crop")
    with Image.open(source) as image:
        image.load()
        validate_box(box, image.size)
        crop = image.crop(box)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            crop.save(stream, format="PNG", optimize=False, compress_level=9)
        with Image.open(target) as actual:
            actual.load()
            require(actual.mode == crop.mode and actual.size == crop.size and actual.tobytes() == crop.tobytes(),
                    "output pixels differ from exact source crop")
            digest = pixel_digest(actual)
        return {"source_width": image.width, "source_height": image.height,
                "output_width": crop.width, "output_height": crop.height,
                "pixel_sha256": digest, "pixel_equality_checks": True}


def sources_from_evidence():
    browser, terminal = read_json(BROWSER), read_json(TERMINAL)
    require(browser.get("mocked") is False and browser.get("passed") is True, "browser evidence must be genuine and passed")
    require(terminal.get("mocked") is False and terminal.get("valid") is True
            and terminal.get("ocr_privacy_guard_passed") is True, "terminal evidence must be genuine and privacy-checked")
    records = {Path(row["path"]).stem: dict(row) for row in browser["screenshots"]}
    records["api_terminal"] = {"path": terminal["image"], "sha256": terminal["image_sha256"]}
    require(set(records) == {"ui_desktop", "ui_route_map", "ui_light", "ui_evaluator", "ui_mobile", "ui_mobile_320", "api_terminal"},
            "expected exactly seven immutable screenshots")
    for name, record in records.items():
        path = ROOT / record["path"]
        require(path.absolute().parent == SHOTS.absolute() and path.stem == name and path.suffix == ".png",
                "screenshot path is not allowlisted")
        verify_source(path, record["sha256"])
        with Image.open(path) as image:
            for key, value in (("width", image.width), ("height", image.height)):
                require(key not in record or record[key] == value, "evidence screenshot size mismatch")
                record[key] = value
    return records


def ocr_crop(path, box):
    result = subprocess.run(["tesseract", str(path), "stdout", "--psm", "11", "tsv"],
                            text=True, capture_output=True, check=True, timeout=120)
    lines, confidences = {}, []
    for raw in result.stdout.splitlines()[1:]:
        fields = raw.split("\t", 11)
        if len(fields) != 12 or not fields[11].strip() or float(fields[10]) < 0:
            continue
        key = tuple(fields[2:5])
        row = {"text": fields[11], "bbox": [int(fields[6]), int(fields[7]),
               int(fields[6]) + int(fields[8]), int(fields[7]) + int(fields[9])]}
        lines.setdefault(key, []).append(row)
        confidences.append(float(fields[10]))
    words = [word for group in lines.values() for word in group]
    require(len(words) >= 2, "crop has insufficient actual OCR text")
    return {"method": "tesseract --psm 11; actual saved crop, original pixels",
            "text": "\n".join(" ".join(w["text"] for w in group) for group in lines.values()),
            "mean_word_confidence": round(sum(confidences) / len(confidences), 2),
            "words": words, "crop_box": list(box)}


def create():
    sources = sources_from_evidence()
    targets = [MANIFEST] + [OUT / (name + suffix) for name, *_ in SPECS for suffix in (".png", ".ocr.json")]
    for path in targets:
        require(not safe_output(path).exists(), "refusing to overwrite existing report image output")
    inputs = {relative(BROWSER): sha(BROWSER), relative(TERMINAL): sha(TERMINAL),
              relative(Path(__file__)): sha(Path(__file__))}
    crops = []
    for name, source_name, box, why in SPECS:
        source = sources[source_name]
        path, target = ROOT / source["path"], OUT / f"{name}.png"
        with Image.open(path) as image:
            validate_box(box, image.size)
        if not source_name.startswith("api_"):
            require(box[2] - box[0] <= 700 and box[3] - box[1] <= (650 if "mobile" in source_name else 430),
                    "UI crop is too large to remain readable in an A4 frame")
        details = crop_image(path, source["sha256"], box, target)
        ocr = ocr_crop(target, box)
        ocr_path = OUT / f"{name}.ocr.json"
        with safe_output(ocr_path).open("x", encoding="utf-8") as stream:
            json.dump(ocr, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        peer_path = PEER_OCR / f"{source_name}.lines.json"
        peer_lines = []
        if peer_path.is_file():
            inputs[relative(peer_path)] = sha(peer_path)
            for row in read_json(peer_path):
                left, top, right, bottom = row["bbox"]
                if box[0] <= left and box[1] <= top and right <= box[2] and bottom <= box[3]:
                    peer_lines.append(row)
        crops.append({"name": name, "source_path": source["path"], "source_sha256": source["sha256"],
                      "crop_box": list(box), "path": relative(target), "sha256": sha(target),
                      "output_path": relative(target), "output_sha256": sha(target), "why": why,
                      "transform": "lossless original-pixel rectangle only; no resize or annotation",
                      "ocr_path": relative(ocr_path), "ocr_sha256": sha(ocr_path),
                      "ocr_mean_word_confidence": ocr["mean_word_confidence"],
                      "source_ocr_lines_within_crop": peer_lines,
                      "recommended_print_width_pt": min(504, 437 * details["output_width"] / details["output_height"]),
                      **details})
    for source in sources.values():
        verify_source(ROOT / source["path"], source["sha256"])
    for path, digest in inputs.items():
        require(sha(ROOT / path) == digest, "provenance input changed during preparation")
    manifest = {"schema_version": 1, "passed": True, "crop_count": len(crops), "crops": crops,
                "sources": list(sources.values()), "inputs": [{"path": path, "sha256": digest} for path, digest in sorted(inputs.items())],
                "source_hashes_unchanged_before_after": True, "all_pixel_equality_checks": True,
                "coordinate_convention": "half-open [left, top, right, bottom] original screenshot pixels",
                "determinism": "fixed crop boxes, lossless PNG, no timestamps or image resizing",
                "inspection": "Actual OCR positions and exact pixel comparison; no manual visual inspection claimed"}
    with safe_output(MANIFEST).open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    return check()


def check():
    manifest = read_json(MANIFEST)
    require(manifest.get("passed") is True and manifest.get("crop_count") == len(SPECS), "crop manifest incomplete")
    sources = sources_from_evidence()
    require(manifest["sources"] == list(sources.values()), "source evidence changed")
    for record in manifest["inputs"]:
        require(sha(ROOT / record["path"]) == record["sha256"], "crop provenance hash mismatch")
    require(len(manifest["crops"]) == len(SPECS), "crop manifest count mismatch")
    for record, (name, source_name, box, _) in zip(manifest["crops"], SPECS):
        require(record["name"] == name and record["crop_box"] == list(box), "unexpected crop specification")
        source = sources[source_name]
        require(record["source_sha256"] == source["sha256"] and record["source_path"] == source["path"], "source binding mismatch")
        target = safe_output(ROOT / record["path"])
        require(sha(target) == record["sha256"], "crop output hash mismatch")
        require(sha(ROOT / record["ocr_path"]) == record["ocr_sha256"], "crop OCR evidence hash mismatch")
        with Image.open(ROOT / source["path"]) as image, Image.open(target) as actual:
            expected = image.crop(box)
            require(actual.mode == expected.mode and actual.size == expected.size and actual.tobytes() == expected.tobytes(),
                    "crop does not equal immutable source pixels")
            require(pixel_digest(actual) == record["pixel_sha256"], "crop pixel hash mismatch")
            require(actual.size == (record["output_width"], record["output_height"]), "crop dimensions mismatch")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--create", action="store_true", help="Create new report crops; refuse any existing target")
    group.add_argument("--check", action="store_true", help="Read-only verification (default)")
    args = parser.parse_args(argv)
    manifest = create() if args.create else check()
    print(f"verified {manifest['crop_count']} original-pixel report crops; manifest SHA-256 {sha(MANIFEST)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
