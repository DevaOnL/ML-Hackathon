#!/usr/bin/env python3
"""Validate actual plot provenance, offline HTML and private development-map guards."""
from __future__ import annotations
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent/".matplotlib"))
import struct
from common import ROOT, OUT, MAPS, read_json, sha256, write_json
from development_map import load_development_geometry


def must_fail(prefix, prediction, reference):
    try:
        load_development_geometry(prefix, prediction, reference)
    except ValueError:
        return True
    raise AssertionError("Map-development isolation guard accepted unsafe input")


def main():
    manifest = read_json(OUT/"figure_manifest.json"); checks = []
    for record in manifest:
        p = ROOT/record["path"]
        assert p.resolve().is_relative_to(OUT)
        assert sha256(p) == record["sha256"]
        assert p.stat().st_size == record["bytes"]
        raw = p.read_bytes(); assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", raw[16:24])
        assert width >= 800 and height >= 500
        assert p.stat().st_size < 3_000_000
        for key in ("caption", "source", "partition", "status"): assert record[key]
        checks.append({"path": record["path"], "size_pixels": [width, height], "verified_sha256": record["sha256"]})
    eda = read_json(OUT/"eda_summary.json")
    assert eda["trip_counts"] == {"train": 80, "student_test": 10}
    assert sum(eda["gps_rows"].values()) == sum(row["rows"] for row in eda["track_input_hashes"])
    for split in ("train", "student_test"):
        stats = eda["timestamps"][split]
        assert stats["steps"] == eda["gps_rows"][split]-eda["trip_counts"][split]
        assert sum(stats["category_counts"].values()) == stats["steps"]
        assert stats["nonpositive_steps"] == 0
    prefix = ROOT/"Student folder/student_test/00000009/00000009"
    pred = ROOT/"outputs/baseline_student_test_v2/routes/00000009.route"
    guards = {"reject_evaluator_graph": must_fail(ROOT/"evaluator/00000009", pred, prefix.with_suffix(".route")), "reject_other_reference": must_fail(prefix, pred, ROOT/"Student folder/student_test/00000007/00000007.route"), "reject_reference_as_prediction": must_fail(prefix, prefix.with_suffix(".route"), prefix.with_suffix(".route"))}
    maps = []
    for metadata_path in sorted(MAPS.glob("*.metadata.json")):
        data = read_json(metadata_path); p = ROOT/data["map_path"]
        assert data["contains_reference_geometry"] and data["development_only"] and data["do_not_serve_to_evaluator"]
        assert data["partition"] in ("train", "student_test") and data["tiles"] is None
        for path, digest in data["input_sha256"].items(): assert sha256(ROOT/path) == digest
        text = p.read_text()
        assert "INTERNAL DEVELOPMENT REPORT MAP" in text and "never serve in evaluator" in text
        assert "tileLayer(" not in text
        browser_path = p.with_suffix(".browser.json")
        if browser_path.exists():
            browser = read_json(browser_path)
            assert not browser["page_errors"] and not browser["blocked_network_requests"]
            assert browser["checked_layers"] == 5 and browser["canvas_count"] >= 1
            assert len(browser["requests"]) == 3
            assert all(path.startswith("file://") for path in browser["requests"])
        maps.append({"path": str(p.relative_to(ROOT)), "metadata_path": str(metadata_path.relative_to(ROOT)), "browser_evidence_exists": browser_path.exists(), "bytes": p.stat().st_size})
    report = {"status": "passed", "all_trip_eda_scope": eda["trip_counts"], "gps_rows": eda["gps_rows"], "plots": checks, "maps": maps, "safety_guards": guards, "no_evaluator_data_accessed": True}
    write_json(OUT/"asset_verification.json", report)
    print(f"Verified {len(checks)} actual PNG plots and {len(maps)} development-only Folium maps; all scope/leakage/provenance guards passed")


if __name__ == "__main__": main()
