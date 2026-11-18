#!/usr/bin/env python3
"""Read-only privacy/secret/size audit of the exact staged publication set.

Never prints matching values. Raw organizer files and row-level proxy caches
are disallowed. Report-only derived maps are allowed; evaluator assets remain
an explicit allowlist enforced by backend source/tests.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[1]
SECRET_RULES = {
    "github_token": rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})",
    "private_key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "aws_access_key": rb"AKIA[0-9A-Z]{16}",
    "cloudflare_secret_value": rb'"(?:apiToken|TunnelSecret)"\s*:\s*"[A-Za-z0-9+/=_-]{24,}"',
}
RAW_EXTENSIONS = {".nodes", ".arcs", ".track", ".docx", ".mp4", ".pt", ".wav", ".pem", ".key"}
RAW_PARTS = {"Student folder", "PathSnap_student_materials", ".venv", ".venv-clean", ".git"}
# Audited aggregate/hash-only evidence, never a directory-wide exemption.
PUBLIC_EVALUATOR_REPORTS = frozenset(
    "reports/final_five/evaluator/" + filename for filename in (
        "freeze_validation.json", "stage_plan.json", "input_audit_pre_prediction.json",
        "prediction_attempt.json", "cli_result.json", "prediction_verification.json",
        "reference_scoring_started.json", "route_metrics.json", "input_audit.json",
        "expected_trips.json", "acceptance.json", "test_and_privacy_verification.json",
    )
)
PUBLIC_EVALUATOR_DOCUMENTS = frozenset({"docs/EVALUATOR_SECTION_PLAN.md"})
# Four reviewed original-pixel screenshot derivatives; no raw evaluator exemption.
PUBLIC_EVALUATOR_SCREENSHOT_DETAILS = frozenset({
    "reports/final_five/report_images/evaluator_dataset_controls.png",
    "reports/final_five/report_images/evaluator_dataset_controls.ocr.json",
    "reports/final_five/report_images/evaluator_scores.png",
    "reports/final_five/report_images/evaluator_scores.ocr.json",
})


def audit(paths):
    findings, files = [], []
    for name in sorted(paths):
        relative = Path(name)
        path = ROOT / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT):
            findings.append({"path": name, "rule": "not_regular_scoped_file"})
            continue
        if (set(relative.parts) & RAW_PARTS or relative.suffix.lower() in RAW_EXTENSIONS
                or relative.name == ".env" or relative.name.startswith(".env.")
                or "cache" in relative.parts or "predictions" in relative.parts
                or relative.suffix.lower() == ".npz"
                or (any(part.lower().startswith(("evaluator", "private")) for part in relative.parts)
                    and relative.as_posix() not in PUBLIC_EVALUATOR_REPORTS | PUBLIC_EVALUATOR_DOCUMENTS | PUBLIC_EVALUATOR_SCREENSHOT_DETAILS)
                or (relative.suffix == ".route" and relative.parts[0] != "outputs")):
            findings.append({"path": name, "rule": "raw_private_or_row_level_input"})
        # Audit the exact index blob, not a concurrently edited working copy.
        data = subprocess.check_output(["git", "show", ":" + name], cwd=ROOT)
        if relative.suffix.lower() == ".zip":
            if relative.parts[0] != "outputs" or relative.name != "submission.zip":
                findings.append({"path": name, "rule": "non_submission_archive"})
            else:
                try:
                    with ZipFile(io.BytesIO(data)) as archive:
                        entries = archive.infolist()
                        valid = (len(entries) == 10 and len({entry.filename for entry in entries}) == 10
                                 and all(re.fullmatch(r"[0-9]{8}\.route", entry.filename)
                                         and not entry.is_dir() and entry.file_size < 5_000_000 for entry in entries)
                                 and archive.testzip() is None)
                        if not valid:
                            findings.append({"path": name, "rule": "invalid_flat_ten_route_submission"})
                        if valid:
                            for entry in entries:
                                member = archive.read(entry)
                                if any(re.search(pattern, member) for pattern in SECRET_RULES.values()):
                                    findings.append({"path": name, "rule": "secret_in_submission_member"})
                except (BadZipFile, RuntimeError, ValueError):
                    findings.append({"path": name, "rule": "invalid_submission_archive"})
        if len(data) >= 90 * 1024 * 1024:
            findings.append({"path": name, "rule": "github_large_blob"})
        for rule, pattern in SECRET_RULES.items():
            if re.search(pattern, data):
                findings.append({"path": name, "rule": rule})
        files.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "exact staged index blobs", "files": files, "file_count": len(files),
            "total_bytes": sum(x["bytes"] for x in files), "findings": findings,
            "passed": not findings, "secret_values_echoed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = subprocess.check_output(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"], cwd=ROOT)
    result = audit([x.decode() for x in raw.split(b"\0") if x])
    if args.output:
        path = args.output.resolve()
        if not path.is_relative_to(ROOT / "reports"):
            raise ValueError("scan output must stay under reports")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
