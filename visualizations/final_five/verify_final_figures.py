#!/usr/bin/env python3
"""Verification program for the final-five figures.

Checks, without any visual inspection by a human:
  1. every manifest PNG exists and its sha256 matches the pinned hash;
  2. PNG resolution metadata says 300 dpi and pixel dimensions match the record;
  3. the paired PDF exists and matches its pinned hash;
  4. tesseract OCR of every PNG yields substantial text, no missing-glyph
     placeholders, no "nan"/"inf" artifacts, and at least one expected title
     token per figure (readability check at the rendered font sizes);
  5. mean OCR word confidence is at least the threshold (default 55);
  6. every measured input file listed in the manifest still hashes to its
     recorded value (drift detection), and the re-pinned read-only EDA
     assets still match their original phase-three hashes.

Writes visualizations/final_five/asset_verification.json and exits non-zero
when any check fails. No figure is regenerated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FIVE_DIR = Path(__file__).resolve().parent
ROOT = FIVE_DIR.parents[1]
OCR_DIR = FIVE_DIR / "ocr"
sys.path.insert(0, str(ROOT))
from visualizations.final_five.figure_integrity import (
    EXPECTED_BY_STAGE, png_dpi_and_size, safe_output, verify_record_integrity,
)
MIN_TEXT_CHARS = 80
MIN_MEAN_CONFIDENCE = 55.0
TOFU_MARKS = ("□", "�", "â–¯")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()





def ocr_png(path: Path) -> tuple[str, float, int]:
    """Run tesseract; returns (text, mean word confidence, word count)."""
    proc = subprocess.run(["tesseract", str(path), "stdout", "--psm", "11"],
                          capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract failed on {path.name}: {proc.stderr.strip()}")
    text = proc.stdout
    tsv = subprocess.run(["tesseract", str(path), "stdout", "--psm", "11", "tsv"],
                         capture_output=True, text=True, timeout=180)
    confs = []
    if tsv.returncode != 0:
        raise RuntimeError(f"tesseract TSV failed on {path.name}: {tsv.stderr.strip()}")
    for line in tsv.stdout.splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) >= 12 and fields[10].strip() not in ("", "-1") and fields[11].strip():
            try:
                conf = float(fields[10])
            except ValueError:
                continue
            if conf >= 0:
                confs.append(conf)
    mean_conf = statistics.mean(confs) if confs else 0.0
    return text, mean_conf, len(confs)


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower())


def verify_figure(record: dict, problems: list[str], min_conf: float) -> dict:
    png = ROOT / record["path"]
    pdf = ROOT / record["pdf_path"]
    result: dict = {"path": record["path"], "checks": {}}
    if not png.exists():
        problems.append(f"{record['path']}: PNG missing")
        return result
    digest = sha256_file(png)
    result["checks"]["png_sha256"] = digest == record["sha256"]
    if not result["checks"]["png_sha256"]:
        problems.append(f"{record['path']}: PNG hash drift (manifest {record['sha256'][:12]}..., "
                        f"file {digest[:12]}...)")
    dpi, (width, height) = png_dpi_and_size(png)
    result["checks"]["dpi_300"] = record.get("dpi") == 300 and all(
        d is not None and abs(d - 300) <= 2.0 for d in dpi)
    if not result["checks"]["dpi_300"]:
        problems.append(f"{record['path']}: PNG dpi metadata is {dpi}, expected ~300")
    result["checks"]["pixels"] = (width, height) == (record["pixels"]["width"], record["pixels"]["height"])
    if not result["checks"]["pixels"]:
        problems.append(f"{record['path']}: pixel size {(width, height)} != manifest "
                        f"{(record['pixels']['width'], record['pixels']['height'])}")
    if pdf.exists():
        result["checks"]["pdf_sha256"] = sha256_file(pdf) == record["pdf_sha256"]
        if not result["checks"]["pdf_sha256"]:
            problems.append(f"{record['path']}: PDF hash drift")
    else:
        result["checks"]["pdf_sha256"] = False
        problems.append(f"{record['path']}: PDF missing")

    text, mean_conf, words = ocr_png(png)
    safe_output(OCR_DIR / (Path(record["path"]).stem + ".txt"), FIVE_DIR).write_text(
        text, encoding="utf-8")
    result["ocr"] = {"chars": len(text.strip()), "words": words,
                     "mean_confidence": round(mean_conf, 2)}
    result["checks"]["ocr_text_present"] = len(text.strip()) >= MIN_TEXT_CHARS
    if not result["checks"]["ocr_text_present"]:
        problems.append(f"{record['path']}: OCR extracted only {len(text.strip())} chars; "
                        f"figure may be unreadable")
    low = [m for m in TOFU_MARKS if m in text]
    result["checks"]["no_missing_glyphs"] = not low
    if low:
        problems.append(f"{record['path']}: OCR text contains missing-glyph placeholder(s) {low!r}")
    if re.search(r"\b(nan|inf(inity)?)\b", text.lower()):
        result["checks"]["no_nan_inf"] = False
        problems.append(f"{record['path']}: OCR text contains nan/inf artifact")
    else:
        result["checks"]["no_nan_inf"] = True
    haystack = norm(text)
    tokens = [norm(t) for t in record.get("ocr_expect", [])]
    result["checks"]["expected_token_found"] = any(t and t in haystack for t in tokens)
    if not result["checks"]["expected_token_found"]:
        problems.append(f"{record['path']}: none of the expected tokens {record.get('ocr_expect')} "
                        f"survived OCR; text may be unreadable or mislabeled")
    result["checks"]["mean_confidence"] = mean_conf >= min_conf
    if not result["checks"]["mean_confidence"]:
        problems.append(f"{record['path']}: mean OCR confidence {mean_conf:.1f} < {min_conf}")
    return result


def verify_inputs(manifest: dict, problems: list[str]) -> dict:
    results = {}
    def walk(node):
        if isinstance(node, dict):
            if "path" in node and "sha256" in node and isinstance(node.get("sha256"), str) \
                    and len(node["sha256"]) == 64:
                path = ROOT / node["path"]
                if not path.exists():
                    problems.append(f"input {node['path']} missing")
                    results[node["path"]] = "missing"
                    return
                current = sha256_file(path)
                results[node["path"]] = "match" if current == node["sha256"] else "DRIFT"
                if current != node["sha256"]:
                    problems.append(f"input {node['path']} drifted from its manifest hash")
                return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(manifest.get("inputs", {}))
    eda = []
    for entry in manifest.get("eda_reused_read_only", []):
        path = ROOT / entry["path"]
        ok = path.exists() and sha256_file(path) == entry.get("sha256")
        eda.append({"path": entry["path"], "still_matches_phase_three_pin": ok})
        if not ok:
            problems.append(f"EDA re-pin broken for {entry['path']}")
    return {"inputs": results, "eda_reuse_pins": eda}


def validate_manifest_structure(manifest: dict, problems: list[str]) -> list[dict]:
    records = manifest.get("figures", [])
    paths = [r.get("path") for r in records]
    if len(paths) != len(set(paths)):
        problems.append("manifest has duplicate figure paths")
    counts = {stage: sum(r.get("stage") == stage for r in records) for stage in EXPECTED_BY_STAGE}
    for stage, names in EXPECTED_BY_STAGE.items():
        actual = {Path(r["path"]).name for r in records if r.get("stage") == stage}
        if actual != names:
            problems.append(f"{stage} figures incomplete/unexpected: missing {sorted(names - actual)}, "
                            f"extra {sorted(actual - names)}")
    if manifest.get("partial") is not False:
        problems.append("partial manifest cannot pass final 22-model + 1-decoder verification")
    if manifest.get("figure_count") != len(records) or len(records) != 23:
        problems.append("figure_count must equal all 23 model + decoder records")
    for key, stage in (("model_figure_count", "models"), ("decoder_figure_count", "decoder")):
        if manifest.get(key) != counts[stage]:
            problems.append(f"{key} disagrees with figures")
    for key in ("decoder_report", "decoder_source"):
        if key not in manifest.get("inputs", {}):
            problems.append(f"missing input pin: {key}")
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify final-five figures via OCR and hashes")
    parser.add_argument("--manifest", type=Path, default=FIVE_DIR / "figure_manifest.json")
    parser.add_argument("--min-conf", type=float, default=MIN_MEAN_CONFIDENCE)
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    problems: list[str] = []
    records = validate_manifest_structure(manifest, problems)
    figures = [verify_figure(r, problems, args.min_conf) for r in records]
    input_report = verify_inputs(manifest, problems)
    report = {
        "passed": not problems,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "verifier": "visualizations/final_five/verify_final_figures.py",
        "manifest": str(args.manifest),
        "thresholds": {"min_text_chars": MIN_TEXT_CHARS, "min_mean_confidence": args.min_conf},
        "tooling": "tesseract " + subprocess.run(["tesseract", "--version"], capture_output=True,
                                                 text=True).stdout.splitlines()[0].split()[-1],
        "figure_count": len(figures),
        "figures": figures,
        "input_hashes": input_report["inputs"],
        "eda_reuse_pins": input_report["eda_reuse_pins"],
        "problems": problems,
        "note": "verification is OCR- and hash-based; no human visual inspection was performed "
                "or claimed",
    }
    out = safe_output(FIVE_DIR / "asset_verification.json", FIVE_DIR)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"verified {len(figures)} figures; {len(problems)} problem(s)")
    for p in problems:
        print(f"  PROBLEM: {p}")
    print(f"report: {out}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
