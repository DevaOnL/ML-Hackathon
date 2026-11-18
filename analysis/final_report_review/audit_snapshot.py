#!/usr/bin/env python3
"""Read report/public JSON contracts only; never render, fit, infer or fetch."""
from __future__ import annotations
import copy
import hashlib
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts/build_final_report.py"
OUT = ROOT / "analysis/final_report_review"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def bool_checks(value, path=()):
    rows = []
    if isinstance(value, dict):
        for key, child in value.items():
            rows.extend(bool_checks(child, path + (str(key),)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(bool_checks(child, path + (str(index),)))
    elif isinstance(value, bool) and any("check" in part for part in path):
        rows.append({"path": ".".join(path), "value": value})
    return rows


def run():
    source = BUILDER.read_bytes()
    module = types.ModuleType("readonly_report_contract_audit")
    module.__file__ = str(BUILDER)
    sys.modules[module.__name__] = module
    exec(compile(source, str(BUILDER), "exec"), module.__dict__)
    state = module.build_state("check")
    values = module.resolve_fact_values(state)
    unresolved = [{**fact, "error": values.get(fact["name"] + "__err")}
                  for fact in module.FACTS if values.get(fact["name"]) is None]
    registry_md = "\n".join("![registered](" + path + ")" for path in module.FIGURE_REGISTRY.values())
    scaffold = "\n".join(["{{fact:" + fact["name"] + "}}" for fact in module.FACTS] +
                          [fact["row"] for fact in module.FACTS if fact.get("row")] + [registry_md])
    md = module.resolve_tokens(scaffold, values)
    guards = module.run_final_guards(state, values, md)
    extra_paths = ["reports/final_five/deployment_activation.json", "visualizations/final_five/map_manifest.json",
                   "visualizations/final_five/maps/final_development_trip_00000009.metadata.json",
                   "visualizations/final_five/maps/final_development_trip_00000009.browser.json"]
    public = {path: read_json(path) for path in extra_paths}
    delivery = state["inputs"]["deliveryv"]["data"]
    live = state["inputs"]["liveapi"]["data"]
    browser = state["inputs"]["browser"]["data"]
    term = state["inputs"]["termv"]["data"]
    checks = bool_checks(delivery)
    figures = state["inputs"]["figman5"]["data"]["figures"]
    figure_errors, figure_notes = module.verify_figures(state, registry_md)
    mutations = []
    for key in ["apiv", "deliveryv"]:
        modified = copy.deepcopy(state)
        modified["inputs"][key]["data"]["checks"] = []
        ok, detail = module.verify_delivery(modified, key, key)
        mutations.append({"name": key + "_empty_checks", "guard_accepts_invalid": bool(ok), "detail": detail})
    modified = copy.deepcopy(state)
    nested = next(row["path"] for row in checks if row["path"].startswith("datasets.") and ".trips." in row["path"])
    parts = nested.split(".")
    node = modified["inputs"]["deliveryv"]["data"]
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = False
    ok, detail = module.verify_delivery(modified, "deliveryv", "deliveryv")
    mutations.append({"name": "nested_delivery_false", "path": nested, "guard_accepts_invalid": bool(ok), "detail": detail})
    modified = copy.deepcopy(state)
    modified["inputs"]["testsjson"] = {"data": {"total": 376, "passed": 376, "failures": [], "complete_release_suite": False}}
    release_guard = next(row for row in module.run_final_guards(modified, values, md) if row["name"] == "tests_final_passed")
    mutations.append({"name": "incomplete_release_suite", "guard_accepts_invalid": release_guard["ok"], "detail": release_guard["detail"]})
    modified = copy.deepcopy(state)
    modified["inputs"]["testsjson"] = {"data": {"failures": []}}
    test_guard = next(row for row in module.run_final_guards(modified, values, md) if row["name"] == "tests_final_passed")
    mutations.append({"name": "missing_test_counts", "guard_accepts_invalid": test_guard["ok"], "detail": test_guard["detail"]})
    modified = copy.deepcopy(state)
    modified["inputs"]["browser"]["data"]["screenshots"] = []
    errors, notes = module.verify_figures(modified, registry_md)
    mutations.append({"name": "missing_live_image_hash_binding", "guard_accepts_invalid": not any("hash" in error.lower() or "live" in error.lower() for error in errors), "errors": errors, "notes": notes})
    template = module.TEMPLATE_PATH
    template_review = {"exists": template.exists(), "status": "not assessed: active child has not written template"}
    if template.exists():
        text = template.read_text()
        resolved = module.resolve_tokens(text, values)
        columns = ["Model", "Algorithm Family", "Key Configuration", "CV Macro F1-score", "Top-1 Accuracy", "Route continuity", "Student-Test Macro F1-score", "Log Loss", "Inference Time"]
        pipe_rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in text.splitlines() if line.strip().startswith("|")]
        template_review = {"exists": True, "sha256": digest(template), "scan_problems": module.scan_template(text),
                           "exact_nine_column_table_header": columns in pipe_rows,
                           "historical_names_mentioned": {name: name.lower() in text.lower() for name in ["RandomForest", "ExtraTrees", "bagging_tuned"]},
                           "epoch_and_na_text_present": "epoch" in text.lower() and "N/A" in text,
                           "status": "mechanical text checks only; no PDF or visual review"}
    return {
        "audit_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "read-only import/build_state/fact resolution/public JSON/hash checks; writes restricted to analysis/final_report_review",
        "report_accepted": False,
        "source_sha256": hashlib.sha256(source).hexdigest(), "source_sha256_at_end": digest(BUILDER),
        "source_changed_during_run": hashlib.sha256(source).hexdigest() != digest(BUILDER),
        "declared_inputs": len(module.INPUTS), "missing_inputs": state["missing"], "load_errors": state["load_errors"],
        "facts_total": len(module.FACTS), "facts_resolved": len(module.FACTS)-len(unresolved), "unresolved_facts": unresolved,
        "guard_context": "synthetic all-facts/all-registry Markdown scaffold; not the authored report/template",
        "scaffold_guard_results": guards,
        "evidence_schemas": {"delivery_check_leaf_count": len(checks), "delivery_check_leaf_false_paths": [row["path"] for row in checks if not row["value"]],
            "delivery_top_check_count": len(delivery.get("checks", {})), "delivery_top_check_types": sorted({type(v).__name__ for v in delivery.get("checks", {}).values()}),
            "delivery_defects": delivery.get("defects"), "live_keys": list(live), "live_matches": len(live.get("matches", [])),
            "live_attempted_count": live.get("attempted_count"), "live_successful_count": live.get("successful_count"),
            "terminal_events": [{"event": row.get("event"), "request_id": row.get("request_id"), "trip_id": row.get("trip_id"), "has_model_hash": "model_hash" in row} for row in term.get("events", [])],
            "browser_shots": [{"path": row.get("path"), "exists": (ROOT / row["path"]).is_file(), "hash_matches": digest(ROOT / row["path"]) == row.get("sha256")} for row in browser.get("screenshots", [])]},
        "registry": {"final_figure_count": len(figures), "missing_final_figures": [row["path"] for row in figures if row["path"] not in module.REGISTRY_BY_PATH],
            "errors": figure_errors, "notes": figure_notes,
            "actual_map_registered": "visualizations/final_five/maps/final_development_trip_00000009.png" in module.REGISTRY_BY_PATH,
            "public_map_inputs_declared": [spec["path"] for spec in module.INPUTS if "map_manifest" in spec["path"] or "final_development_trip" in spec["path"]],
            "canonical_map_files": public["visualizations/final_five/map_manifest.json"]["files"]},
        "hardening_mutation_probes": mutations,
        "test_evidence": {key: state["inputs"].get("testsjson", {}).get("data", {}).get(key) for key in ["total", "passed", "failures", "complete_release_suite"]},
        "static_checks": {"datetime_import": "from datetime import datetime, timezone" in source.decode(), "datetime_namespace_calls": [i for i,line in enumerate(source.decode().splitlines(),1) if "datetime.datetime" in line or "datetime.timezone" in line],
            "scan_template_on_resolved_md": "scan_template(resolved_md)" in source.decode()},
        "extra_public_input_sha256": {path: digest(ROOT / path) for path in extra_paths},
        "template_review": template_review,
        "limitations": ["No fit, probability inference, raw evaluator data, HTTP, private config, preview/final render or PDF inspection.", "Builder/template remain active in another child; findings apply only to the pinned snapshot.", "Delivery check booleans are not the same as all booleans: exact_match=false is a measured result, not a failed acceptance check."]
    }


if __name__ == "__main__":
    result = run()
    output = OUT / "audit_snapshot.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "source_sha256": result["source_sha256"], "source_changed_during_run": result["source_changed_during_run"], "facts_resolved": result["facts_resolved"], "facts_total": result["facts_total"], "failed_scaffold_guards": [row["name"] for row in result["scaffold_guard_results"] if not row["ok"]], "template": result["template_review"]["status"]}, indent=2))
