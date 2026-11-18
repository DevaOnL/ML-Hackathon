"""Independent public-only final source/evidence/output integrity checks.

No model deserialization, inference, fits, raw data or network requests.
"""
from __future__ import annotations
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("final_report_integrity", ROOT / "scripts/build_final_report.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

manifest_path = ROOT / "reports/final_five/report_manifest.json"
manifest = m.strict_json(manifest_path.read_text())
state = m.build_state("check")
values = m.resolve_fact_values(state)
template = m.TEMPLATE_PATH.read_text()
resolved_md = m.expand_tables(m.resolve_tokens(template, values), state)
guards = m.run_final_guards(state, values, resolved_md, template)
checks = {}
issues = []
checks["manifest_final_schema2"] = manifest.get("mode") == "final" and manifest.get("schema_version") == 2
checks["all17_guards_pass_now"] = len(guards) == 17 and all(g.get("ok") is True for g in guards)
checks["all17_persisted_guards_pass"] = len(manifest.get("guards", [])) == 17 and all(g.get("ok") is True for g in manifest["guards"])
checks["facts360_inputs50_images55"] = (len(m.FACTS), len(state["inputs"]), len(manifest.get("assets", {}))) == (360, 50, 55)
checks["persisted_integrity_check"] = not m.verify_persisted_manifest(state, values)
checks["registry_digest_matches_current"] = manifest.get("registry_sha256") == m._registry_sha256()

bindings = []
for kind in ("builder", "template", "markdown", "pdf"):
    r = manifest[kind]
    target = m._hash_binding(r["path"], r["sha256"])
    bindings.append({"kind": kind, "path": r["path"], "sha256": m.sha256_file(target), "bytes": target.stat().st_size})
for kind in ("inputs", "assets"):
    for key, r in manifest[kind].items():
        path, sha = (key, r) if kind == "assets" else (r["path"], r["sha256"])
        m._hash_binding(path, sha)
checks["all_declared_input_asset_output_hashes_current"] = True

md_path = ROOT / manifest["markdown"]["path"]
pdf_path = ROOT / manifest["pdf"]["path"]
md = md_path.read_text()
pages, text = m.pdf_page_count_and_text(pdf_path)
checks["canonical_md_exactly_resolves_current_template"] = md == resolved_md
checks["pdf_is_exact57_pages"] = pages == manifest["pdf"]["pages"] == 57
checks["pdf_has_no_draft_or_unresolved_placeholder"] = not any(t in text for t in ("DRAFT - NOT FINAL", "{{fact:", "[[PENDING", "FIGURE PENDING"))
freeze_claim = next(line for line in md.splitlines() if line.startswith("The selected model froze at "))
checks["FRR01_closed_deployment_export_hash"] = m.PINNED_MODEL_SHA256 in freeze_claim and "deployment-export" in freeze_claim
checks["comparison_bundle_hash_retained_separately"] = "47ea723d350567b6eeba827744e2c1a26c9baf915d322e29a590f45922ed8eeb" in md and "separately recorded comparison artifact" in freeze_claim
checks["FRR02_closed_sorted_terminal_status_registry"] = next(f for f in m.FACTS if f["name"] == "req_resolved")["count"]["in"] == sorted(m.REQ_TERMINAL_STATUSES)

matrix = m._data(state, "reqs")
rows = matrix["requirements"]
status_counts = dict(Counter(r["status"] for r in rows))
checks["all47_unique_terminal_requirement_rows"] = len(rows) == len({r["id"] for r in rows}) == 47 and all(r["status"] in m.REQ_TERMINAL_STATUSES for r in rows)
checks["status_distribution_41done_5human_1exception"] = status_counts == {"done_evidence_exists": 41, "human_external_blocker": 5, "user_exception": 1}
evidence = []
evidence_errors = []
for row in rows:
    records = row.get("final_evidence", [])
    if records and not isinstance(records, list):
        evidence_errors.append({"row":row["id"], "error":"final_evidence is not list"})
        continue
    for r in records:
        try:
            target = m._hash_binding(r["path"], r["sha256"])
            is_json = target.suffix == ".json"
            if is_json:
                m.strict_json(target.read_text())
            evidence.append({"row": row["id"], "path": r["path"], "sha256": r["sha256"], "strict_json_validated": is_json})
        except Exception as exc:
            evidence_errors.append({"row":row["id"], "record":r, "error":str(exc)})
checks["every_provided_requirement_evidence_hash_and_json_valid"] = not evidence_errors
prepub = m.strict_json((ROOT / "reports/final_five/report_prepublication_acceptance.json").read_text())
checkpoint = ROOT / "analysis/final_release/candidate_corrected/checkpoint.json"
checkpoint_data = m.strict_json(checkpoint.read_text())
checks["RDL03_binds_accepted_corrected_candidate_not_final_self_hash"] = (prepub["candidate_checkpoint_sha256"] == m.sha256_file(checkpoint) and prepub["candidate_pdf_sha256"] == checkpoint_data["pdf"]["sha256"] == "c5ea75013c51d835a6538c680d7439b7f18bc71173f9d35340900b1e675fed5f" and prepub.get("canonical_final_build_required") is True)
test_record = m._data(state, "testsjson")
checks["positive_complete_test_record_passes_bound_logs"] = m.verify_test_record(state)[0]

output = {"passed":all(checks.values()), "checks":checks, "source_output_pins":bindings,
          "manifest_sha256":m.sha256_file(manifest_path), "registry_sha256":m._registry_sha256(),
          "pdf_pages":pages, "current_guards":guards, "persisted_manifest_issues":m.verify_persisted_manifest(state, values),
          "requirement_status_counts":status_counts,"requirements_with_explicit_final_evidence":sum(bool(r.get("final_evidence")) for r in rows),
          "requirement_evidence_bindings":len(evidence),"unique_requirement_evidence_paths":len({r["path"] for r in evidence}),
          "strict_json_evidence_binding_count":sum(r["strict_json_validated"] for r in evidence),
          "requirement_evidence":evidence,"evidence_errors":evidence_errors,"test_record":test_record,
          "release_scope":{"no_fits_or_model_inference":True,"no_raw_private_inputs":True,"no_http":True,"no_release_source_writes":True}}
print(json.dumps(output, indent=2))
sys.exit(0 if output["passed"] else 1)
