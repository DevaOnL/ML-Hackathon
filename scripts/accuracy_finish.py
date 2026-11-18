#!/usr/bin/env python3
"""Summarize completed evidence concisely; never launches tuning or mutates freeze."""
from __future__ import annotations
import json
from pathlib import Path
from scripts import accuracy_common as c,accuracy_confirmation as confirmation


def summary():
    frozen=confirmation.verify_frozen();rows=[]
    for family in c.FAMILIES:
        path=confirmation.CONFIRMATION/(family+".json")
        if not path.exists():continue
        r=json.loads(path.read_text());a=r["records"]["baseline"]["metrics"];b=r["records"]["candidate"]["metrics"]
        rows.append({"family":family,"baseline_top1":a["trip_macro_top1_accuracy"],"candidate_top1":b["trip_macro_top1_accuracy"],"top1_delta_pp":100*(b["trip_macro_top1_accuracy"]-a["trip_macro_top1_accuracy"]),"baseline_binary_macro_f1":a["macro_f1"],"candidate_binary_macro_f1":b["macro_f1"],"baseline_binary_log_loss":a["log_loss"],"candidate_binary_log_loss":b["log_loss"],"paired_trip_delta":r["paired_trip_top1"],"source":path.relative_to(c.ROOT).as_posix(),"sha256":c.sha256_file(path)})
    out={"frozen_selection_sha256":c.sha256_file(confirmation.FREEZE),"confirmation_role":"same64fitting/reserved16internalconfirmation; historicallyexposednotblind","families":rows}
    for name,path in (("route_confirmation",confirmation.CONFIRMATION/"route_summary.json"),("student_development",c.EXPERIMENT_ROOT/"student_comparison/summary.json")):
        if path.exists():out[name]=json.loads(path.read_text())
    print(json.dumps(out,indent=2,allow_nan=False));return out
if __name__=="__main__":summary()
