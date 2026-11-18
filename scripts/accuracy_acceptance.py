#!/usr/bin/env python3
"""Independently recompute frozen evidence gates; never train or change decisions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scripts import accuracy_common as c,accuracy_confirmation as f


def load(path):
    def reject(value):raise ValueError("nonfinite JSON token "+value)
    return json.loads(Path(path).read_text(),parse_constant=reject)


def finite_fraction(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<=value<=1:raise ValueError("metric outside finite[0,1]")
    return float(value)


def internal():
    frozen=f.verify_frozen();protocol=load(c.PROTOCOL_PATH);ids=protocol["confirmation_trip_ids"];freeze=c.sha256_file(f.FREEZE);checks=[]
    for family in c.FAMILIES:
        record=load(f.CONFIRMATION/(family+".json"))
        if record["family"]!=family or record["status"]!="complete" or record["frozen_selection_sha256"]!=freeze:raise ValueError("wrong classifier confirmation identity")
        for label in ("baseline","candidate"):
            p=record["records"][label];m=p["metrics"]
            if sorted(t["trip_id"] for t in m["per_trip"])!=sorted(ids) or len(m["per_trip"])!=16 or m["observations"]!=5600 or any(t["observations"]!=350 for t in m["per_trip"]):raise ValueError("classifier observation/trip denominator differs")
            values=[finite_fraction(t["top1_accuracy"]) for t in m["per_trip"]]
            if not np.isclose(np.mean(values),finite_fraction(m["trip_macro_top1_accuracy"]),rtol=0,atol=1e-12):raise ValueError("classifier trip macro does not recompute")
            for k in ("accuracy","macro_f1","edge_macro_f1"):finite_fraction(m[k])
            if not np.isfinite(m["log_loss"]) or m["log_loss"]<0 or c.sha256_file(c.ROOT/p["model_path"])!=p["model_sha256"]:raise ValueError("classifier likelihood/model drift")
        delta=record["records"]["candidate"]["metrics"]["trip_macro_top1_accuracy"]-record["records"]["baseline"]["metrics"]["trip_macro_top1_accuracy"]
        if not np.isclose(delta,record["metric_deltas"]["trip_macro_top1_accuracy"],rtol=0,atol=1e-12):raise ValueError("classifier delta inconsistent")
        checks.append({"family":family,"top1_delta":delta,"top1_improved":delta>0,"record_sha256":c.sha256_file(f.CONFIRMATION/(family+".json"))})
    routes=load(f.CONFIRMATION/"route_summary.json");family=frozen["decoder_selection"]["emission_family"]
    if routes["routes"]!=16 or routes["family"]!=family or routes["frozen_selection_sha256"]!=freeze or [b["trip_id"] for b in routes["trip_bindings"]]!=ids:raise ValueError("route population identity differs")
    records=[]
    for trip,b in zip(ids,routes["trip_bindings"]):
        path=c.ROOT/b["path"];r=load(path)
        if c.sha256_file(path)!=b["sha256"] or r["trip_id"]!=trip or r["paired_fit_record_sha256"]!=c.sha256_file(f.CONFIRMATION/(family+".json")) or r["frozen_selection_sha256"]!=freeze:raise ValueError("route evidence identity drift")
        records.append(r)
    for label in ("baseline","candidate"):
        for k in ("length_f1","edge_f1","route_continuity","proxy_edge_accuracy","proxy_edge_macro_f1"):
            values=[finite_fraction(r["records"][label]["metrics"][k] if k in r["records"][label]["metrics"] else r["records"][label][k]) for r in records]
            if not np.isclose(np.mean(values),finite_fraction(routes[label][k]),rtol=0,atol=1e-12):raise ValueError("route metric does not match actual distinct per-trip records")
        if sum(not r["records"][label]["success"] for r in records)!=routes[label]["failures"]:raise ValueError("failure count differs")
    winner=next(x for x in checks if x["family"]==family)
    passed=winner["top1_improved"] and routes["candidate"]["length_f1"]>=routes["baseline"]["length_f1"] and routes["candidate"]["edge_f1"]>=routes["baseline"]["edge_f1"] and routes["candidate"]["failures"]==0
    return {"internal_passed":bool(passed),"frozen_selection_sha256":freeze,"family_results":checks,"route_summary_sha256":c.sha256_file(f.CONFIRMATION/"route_summary.json"),"all_five_top1_improved":all(x["top1_improved"] for x in checks),"role":"internalhistoricallyexposedTRAINnotblind","source_sha256":c.sha256_file(__file__)}


def accept():
    report=internal();path=c.ROOT/"artifacts/accuracy_improvement/release_manifest.json";manifest=load(path);student_path=c.EXPERIMENT_ROOT/"student_comparison/summary.json";student=load(student_path)
    if student["trips"]!=10 or len(student["trip_bindings"])!=10 or len({b["trip_id"] for b in student["trip_bindings"]})!=10 or student["frozen_selection_sha256"]!=report["frozen_selection_sha256"] or student["export_manifest_sha256"]!=c.sha256_file(path) or student["no_tuning_feedback"] is not True or student["evaluator_used"] is not False:raise ValueError("student final comparison identity differs")
    records=[]
    for b in student["trip_bindings"]:
        p=c.ROOT/b["path"];r=load(p)
        if r["trip_id"]!=b["trip_id"] or c.sha256_file(p)!=b["sha256"] or r["export_manifest_sha256"]!=c.sha256_file(path) or r["frozen_selection_sha256"]!=report["frozen_selection_sha256"]:raise ValueError("student evidence hash/identity drift")
        records.append(r)
    for label in ("baseline","candidate"):
        for k in ("length_f1","edge_f1","route_continuity","proxy_edge_accuracy","proxy_edge_macro_f1","classifier_top1_accuracy"):
            values=[finite_fraction(r["records"][label]["metrics"][k] if k in r["records"][label]["metrics"] else r["records"][label][k]) for r in records]
            if not np.isclose(np.mean(values),finite_fraction(student[label][k]),rtol=0,atol=1e-12):raise ValueError("student final means do not recompute")
    accepted=report["internal_passed"] and student["candidate"]["failures"]==0 and all(student["candidate"][k]>=student["baseline"][k] for k in ("length_f1","edge_f1","route_continuity"))
    result={**report,"created_at_utc":c.utc_now(),"accepted":bool(accepted),"release_manifest_sha256":c.sha256_file(path),"student_comparison_sha256":c.sha256_file(student_path),"student_development_used_for_tuning":False,"evaluator_used_for_tuning":False,"deployed":False,"reason":"Frozen candidate improves internal primary and preserves both route F1 metrics and continuity in final historical-student comparison" if accepted else "Preserve original deployment: final frozen candidate did not pass every nonregression gate","inference_entrypoint_sha256":c.sha256_file(c.ROOT/"backend/entrypoint.py"),"main_entrypoint_sha256":c.sha256_file(c.ROOT/"backend/main.py")}
    c.write_new_json(path.parent/"acceptance.json",result);c.write_new_json(c.ROOT/"reports/accuracy_improvement/release_decision.json",result);print(json.dumps(result,indent=2));return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--accept",action="store_true");a=p.parse_args()
    if a.accept:accept()
    else:
        result=internal();print(json.dumps(result,indent=2))
        if not result["internal_passed"]:raise SystemExit(1)
