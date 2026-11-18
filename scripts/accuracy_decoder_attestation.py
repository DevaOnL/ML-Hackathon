#!/usr/bin/env python3
"""Verify already measured OOF decoder execution against original source/model pins.

A separate audit, not retroactively substituted measurement metadata. All original
route result files stay immutable; this fails unless the exact original manifest
inputs and strict fold artifacts remain byte-identical. No predictions or targets.
"""
from __future__ import annotations
import json
from pathlib import Path
from scripts import accuracy_common as c,accuracy_confirmation as confirmation
from scripts.accuracy_route_validation import OUTPUT,AUTHORIZATION,MENU,selected_routes
from scripts.accuracy_decoder import require_decoder_authorization,verify_menu_registration

ATTESTATION=c.EXPERIMENT_ROOT/"decoder"/"execution_attestation.json"


def attest():
    auth=json.loads(AUTHORIZATION.read_text());mp=c.ROOT/auth["candidate_manifest_path"];m=json.loads(mp.read_text());md=c.sha256_file(mp);menu=verify_menu_registration(MENU)
    require_decoder_authorization(auth,candidate_manifest_sha256=md,decoder_menu_sha256=menu)
    if auth["training_protocol_sha256"]!=c.sha256_file(c.PROTOCOL_PATH) or m["protocol_sha256"]!=c.sha256_file(c.PROTOCOL_PATH):raise ValueError("original decoder protocol changed")
    for relative,digest in m["source_sha256"].items():
        if c.sha256_file(c.ROOT/relative)!=digest:raise ValueError("original decoder dependency changed: "+relative)
    pair={}
    for label,key in (("candidate","candidate_trial"),("baseline","baseline_trial")):
        p=c.ROOT/m[key+"_path"]
        if c.sha256_file(p)!=m[key+"_sha256"]:raise ValueError("authorized trial record changed")
        record=json.loads(p.read_text());identity=[]
        if record["protocol_sha256"]!=m["protocol_sha256"]:raise ValueError("model training protocol differs")
        for f in record["folds"]:
            model=c.ROOT/record["private_output"]/f"fold_{f['fold']}.joblib"
            if c.sha256_file(model)!=f["private_model_sha256"]:raise ValueError("original fold model changed")
            identity.append({"fold":f["fold"],"sha256":f["private_model_sha256"],"train_trip_ids":f["train_trip_ids"],"validation_trip_ids":f["validation_trip_ids"]})
        pair[label]={"trial_path":m[key+"_path"],"trial_sha256":m[key+"_sha256"],"fold_models":identity}
    routes=[]
    for expected in selected_routes(json.loads(c.PROTOCOL_PATH.read_text())):
        path=OUTPUT/(expected["trip_id"]+".json");r=json.loads(path.read_text())
        if r["candidate_trial_sha256"]!=pair["candidate"]["trial_sha256"] or r["emission_family"]!=m["family"] or r["emission_trial"]!=m["trial"]:raise ValueError("measured route candidate identity differs")
        if r["protocol_sha256"]!=m["protocol_sha256"] or r["decoder_menu_sha256"]!=menu or r["model_train_trip_ids"]!=expected["train_trip_ids"] or r["model_validation_trip_ids"]!=expected["validation_trip_ids"]:raise ValueError("measured route fold/menu/partition differs")
        for relative,digest in r["source_sha256"].items():
            if m["source_sha256"].get(relative)!=digest:raise ValueError("measured route source differs from preregistered manifest")
        names={x["name"] for x in json.loads(MENU.read_text())["configurations"]};observed=[(p["kind"],p["configuration"]) for p in r["predictions"]]
        if len(observed)!=16 or set(observed)!={(kind,name) for kind in ("candidate","v1_control") for name in names}:raise ValueError("incomplete paired decoder menu")
        if any(p["observations"]!=r["fulltrack_observations"] for p in r["predictions"]):raise ValueError("dense denominator mismatch")
        routes.append({"trip_id":r["trip_id"],"path":path.relative_to(c.ROOT).as_posix(),"sha256":c.sha256_file(path)})
    result={"schema":"accuracy-decoder-execution-attestation-v1","created_at_utc":c.utc_now(),"status":"pass","measurement_files_modified":False,"scope":"postexecution integrity verification of original frozen inputs; not an independent recreation of measurements","authorization_path":AUTHORIZATION.relative_to(c.ROOT).as_posix(),"authorization_sha256":c.sha256_file(AUTHORIZATION),"candidate_manifest_path":mp.relative_to(c.ROOT).as_posix(),"candidate_manifest_sha256":md,"decoder_menu_sha256":menu,"protocol_sha256":m["protocol_sha256"],"complete_authorized_source_sha256":m["source_sha256"],"paired_trial_model_bindings":pair,"route_result_bindings":routes,"audit_source_sha256":c.sha256_file(__file__),"confirmation_used":False,"student_used":False,"evaluator_used":False}
    c.write_new_json(ATTESTATION,result);return result

if __name__=="__main__":attest()
