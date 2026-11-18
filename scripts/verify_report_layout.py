#!/usr/bin/env python3
"""Automated PDF text/layout/render checks, explicitly NOT a human visual review.

Uses the installed PyMuPDF analysis tool and Tesseract only for an optional OCR
sample. Does not modify the PDF or replace unavailable evidence with mock images.
All pages are rendered at 100 dpi into the separately scoped output directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_pdf(pdf, output, *, require_final=False, ocr=False):
    import fitz
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "pdf": str(pdf),
              "pdf_sha256": sha256(pdf), "method": "actual PDF text/span geometry and page rasterization; no human/model visual inspection claimed",
              "require_final": require_final, "pages": [], "defects": [], "warnings": []}
    all_text = []
    with fitz.open(pdf) as doc:
        report["page_count"] = len(doc)
        for index, page in enumerate(doc):
            page_number = index + 1
            text = page.get_text("text")
            all_text.append(text)
            spans, line_rects, outside = [], [], []
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    rect = fitz.Rect(line["bbox"])
                    line_text = " ".join(s["text"] for s in line["spans"]).strip()
                    if line_text:
                        line_rects.append((rect, line_text))
                    for span in line["spans"]:
                        if not span["text"].strip():
                            continue
                        spans.append(span)
                        rect = fitz.Rect(span["bbox"])
                        if (rect.x0 < -.5 or rect.y0 < -.5 or rect.x1 > page.rect.width + .5
                                or rect.y1 > page.rect.height + .5):
                            outside.append({"text": span["text"][:120], "bbox": list(rect)})
            overlaps = []
            for a, (rect_a, text_a) in enumerate(line_rects):
                for rect_b, text_b in line_rects[a+1:]:
                    overlap = rect_a & rect_b
                    if overlap.is_empty or min(rect_a.get_area(), rect_b.get_area()) <= 0:
                        continue
                    fraction = overlap.get_area() / min(rect_a.get_area(), rect_b.get_area())
                    if fraction > .24 and overlap.width > 3 and overlap.height > 2:
                        overlaps.append({"line_a": text_a[:110], "line_b": text_b[:110], "overlap_fraction": round(fraction, 4)})
            images = []
            for info in page.get_image_info():
                rect = fitz.Rect(info["bbox"])
                ratio = (rect.width / rect.height) / (info["width"] / info["height"])
                images.append({"bbox": list(rect), "pixel_size": [info["width"], info["height"]],
                               "aspect_ratio_distortion": abs(1-ratio),
                               "effective_horizontal_dpi": info["width"] / max(rect.width, 1e-9) * 72})
                if abs(1-ratio) > .025:
                    report["defects"].append(f"page {page_number}: image aspect ratio distorted")
                if (rect.x0 < -.5 or rect.y0 < -.5 or rect.x1 > page.rect.width+.5 or rect.y1 > page.rect.height+.5):
                    report["defects"].append(f"page {page_number}: image extends outside page")
            raster = page.get_pixmap(matrix=fitz.Matrix(100/72, 100/72), alpha=False)
            path = output / f"page-{page_number:02d}.png"
            raster.save(path)
            page_info = {"page": page_number, "size_points": list(page.rect)[2:], "text_characters": len(text),
                         "text_spans": len(spans), "images": images, "outside_text": outside,
                         "overlap_candidates": overlaps, "render": str(path), "render_sha256": sha256(path)}
            if outside:
                report["defects"].append(f"page {page_number}: {len(outside)} text spans outside page")
            if overlaps:
                # Tables, links and superscripts can create conservative false positives.
                report["warnings"].append(f"page {page_number}: {len(overlaps)} overlapping line candidates require inspection")
            if len(text.strip()) < 30 and not images:
                report["defects"].append(f"page {page_number}: blank or almost blank content page")
            if ocr and page_number in {1, len(doc), max(1, len(doc)//2)}:
                recognized = subprocess.run(["tesseract", str(path), "stdout"], capture_output=True, text=True, check=True).stdout
                ocr_path = output / f"page-{page_number:02d}-ocr.txt"
                ocr_path.write_text(recognized)
                page_info["ocr_characters"] = len(recognized.strip())
                page_info["ocr_source"] = str(ocr_path)
                if len(recognized.strip()) < 30:
                    report["defects"].append(f"page {page_number}: rendered page OCR has too little readable content")
            report["pages"].append(page_info)
    full_text = "\n\n".join(all_text)
    (output / "extracted_text.txt").write_text(full_text)
    forbidden = re.findall(r"\b(?:TODO|TBD|FIXME|PENDING|PLACEHOLDER|NaN)\b|\?\?\?", full_text)
    report["forbidden_tokens"] = sorted(set(forbidden))
    report["preview_watermark"] = "PREVIEW - NOT FOR SUBMISSION" in full_text or "PREVIEW" in full_text
    if require_final:
        if forbidden:
            report["defects"].append("final PDF contains unresolved placeholder tokens")
        if report["preview_watermark"]:
            report["defects"].append("final PDF still has a preview watermark")
        required = {"Logistic Regression": r"logistic\s+regression", "KNN/distance": r"\bknn\b|nearest.neighbou?r",
                    "Bagging": r"bagging", "directed Viterbi": r"viterbi", "Bayesian search": r"bayesian",
                    "proxy-label limitation": r"proxy", "live deployment": r"ml\.d3va\.dev",
                    "deferred five-model requirement": r"deferred", "absent evaluator limitation": r"evaluator"}
        report["required_sections"] = {name: bool(re.search(pattern, full_text, re.IGNORECASE)) for name, pattern in required.items()}
        report["defects"] += ["missing required final report topic: " + name for name, found in report["required_sections"].items() if not found]
    report["passed"] = not report["defects"]
    (output / "layout_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-final", action="store_true")
    parser.add_argument("--ocr", action="store_true")
    args = parser.parse_args()
    if args.output.resolve().exists():
        parser.error("choose a new layout-review directory; do not overwrite reviewed pages")
    report = inspect_pdf(args.pdf.resolve(), args.output.resolve(), require_final=args.require_final, ocr=args.ocr)
    print(json.dumps({k: report[k] for k in ("passed", "page_count", "pdf_sha256", "preview_watermark", "defects", "warnings")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
