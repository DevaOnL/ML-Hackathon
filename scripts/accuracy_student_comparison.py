#!/usr/bin/env python3
"""Final named student-development comparison after freeze; no selection feedback.

The student set is historically development data, not a blind test. No evaluator
input is accepted. Released artifacts stay unchanged and both full predictions
finish before a trip reference is opened.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from mapmatch.data import discover_trips,load_graph,load_track,load_route
from mapmatch.features import reference_observation_labels
from mapmatch.learning import load_emission_model
from mapmatch.matcher import MapMatcher,MatchConfig
from mapmatch.accuracy_matcher import AccuracyMapMatcher,AccuracyTransitionPolicy
from scripts import accuracy_common as c,accuracy_confirmation as confirmation
from scripts.accuracy_route_validation import scalar_prediction,score_prediction
from backend.emissions import AuditedEmissionModel

OUTPUT=c.EXPERIMENT_ROOT/"student_comparison"


def run_trip(trip_id):
    frozen=confirmation.verify_frozen();manifest_path=c.ROOT/"artifacts/accuracy_improvement/release_manifest.json";manifest=json.loads(manifest_path.read_text());root=manifest_path.parent
    if manifest["frozen_selection_sha256"]!=c.sha256_file(confirmation.FREEZE):raise ValueError("all80exportnotboundtofrozeninternalconfirmation")
    for name,digest in manifest["files"].items():
        if c.sha256_file(root/name)!=digest:raise ValueError("exportartifactdrift")
    prefixes={p.name:p for p in discover_trips(c.ROOT/"Student folder"/"student_test")}
    if trip_id not in prefixes:raise ValueError("notaregisteredstudentdevelopmenttrip")
    output=OUTPUT/(trip_id+".json");private=c.PRIVATE_ROOT/"student_comparison"/trip_id
    if output.exists() or private.exists():raise FileExistsError("studentcomparisonalreadyrun")
    from mapmatch.accuracy_artifacts import load_accuracy_deployment_model
    candidate=load_accuracy_deployment_model(root/"model.joblib",expected_sha256=manifest["model_sha256"],expected_protocol_sha256=manifest["protocol_sha256"]);old_path=c.ROOT/"artifacts/final_five/model.joblib";old_config_path=c.ROOT/"artifacts/final_five/match_config.json"
    if c.sha256_file(old_path)!="c9cc8e9f36838f656a94d07ed69632979df9981ca94a9daf3b52fbce052e4b34" or c.sha256_file(old_config_path)!="10405aa6b6f370e41d4ddcb01207fd1e8f58900fd0395e541ee6bc78146168c5":raise ValueError("releasedbaselineidentitychanged")
    old_model=load_emission_model(old_path);old_config=MatchConfig(**json.loads(old_config_path.read_text()));config=MatchConfig(**json.loads((root/"match_config.json").read_text()));transition=AccuracyTransitionPolicy(**json.loads((root/"transition_policy.json").read_text()));prefix=prefixes[trip_id];graph=load_graph(prefix);track=load_track(prefix.with_suffix(".track"));private.mkdir(parents=True,exist_ok=False)
    old_audit=AuditedEmissionModel(old_model);new_audit=AuditedEmissionModel(candidate)
    new_result,new_summary=scalar_prediction(graph,track,new_audit,{"name":frozen["decoder_selection"]["configuration"]["name"],"config":config,"transition":transition})
    old_result,old_summary=scalar_prediction(graph,track,old_audit,{"name":"released_v1","config":old_config,"transition":AccuracyTransitionPolicy()})
    reference=load_route(prefix.with_suffix(".route"));labels=reference_observation_labels(graph,track,reference);scores={}
    for label,result,summary,audit in (("baseline",old_result,old_summary,old_audit),("candidate",new_result,new_summary,new_audit)):
        scores[label]=score_prediction(graph,track,result,reference,summary,labels)
        if result is None:
            scores[label]["metrics"]["edit_similarity"]=0.
            scores[label].update(classifier_top1_accuracy=0.,candidate_miss_rate=1.,candidate_normalized_nll=float(-np.log(1e-7)))
        if result is not None:
            np.savez_compressed(private/(label+".npz"),route=result.route,matched_edges=result.matched_edges,observation_indices=result.observation_indices)
            top1=[];misses=[];ll=[]
            for i,t in enumerate(labels.edges):
                rec=audit.records[i];matches=np.flatnonzero(rec.edges==t);top1.append(bool(rec.edges[int(np.argmax(rec.probabilities))]==t));misses.append(len(matches)==0);ll.append(-np.log(max(float(rec.probabilities[matches[0]]) if len(matches) else 1e-7,1e-7)))
            scores[label]["classifier_top1_accuracy"]=float(np.mean(top1));scores[label]["candidate_miss_rate"]=float(np.mean(misses));scores[label]["candidate_normalized_nll"]=float(np.mean(ll))
    out={"created_at_utc":c.utc_now(),"trip_id":trip_id,"records":scores,"frozen_selection_sha256":c.sha256_file(confirmation.FREEZE),"export_manifest_sha256":c.sha256_file(manifest_path),"comparison_role":"historicallyusedstudentdevelopmentfinalcomparisonnotblind","reference_loaded_after_both_predictions":True,"no_tuning_feedback":True,"evaluator_used":False,"source_sha256":c.sha256_file(__file__)}
    confirmation.verify_frozen();c.write_new_json(output,out);print(json.dumps({"event":"student_development_comparison","trip_id":trip_id,"baseline_lengthF1":scores["baseline"]["metrics"]["length_f1"],"candidate_lengthF1":scores["candidate"]["metrics"]["length_f1"]}),flush=True);return out


def aggregate():
    frozen=confirmation.verify_frozen();ids=sorted(p.name for p in discover_trips(c.ROOT/"Student folder"/"student_test"));records=[json.loads((OUTPUT/(t+".json")).read_text()) for t in ids]
    manifest=c.ROOT/"artifacts/accuracy_improvement/release_manifest.json"
    if len(records)!=10 or len({r["trip_id"] for r in records})!=10:raise ValueError("studentcomparisonpopulationidentityfailed")
    for trip,record in zip(ids,records):
        if record["trip_id"]!=trip or record["frozen_selection_sha256"]!=c.sha256_file(confirmation.FREEZE) or record["export_manifest_sha256"]!=c.sha256_file(manifest) or record["source_sha256"]!=c.sha256_file(__file__) or record.get("no_tuning_feedback") is not True or record.get("reference_loaded_after_both_predictions") is not True or record.get("evaluator_used") is not False:raise ValueError("student trip/export/source provenance differs")
    result={"created_at_utc":c.utc_now(),"trips":len(records),"role":"historicalstudentdevelopmentnotblindfinalcomparison","frozen_selection_sha256":c.sha256_file(confirmation.FREEZE),"no_tuning_feedback":True,"evaluator_used":False}
    for label in ("baseline","candidate"):
        result[label]={k:float(np.mean([r["records"][label]["metrics"][k] for r in records])) for k in ("length_f1","edge_f1","route_continuity","edit_similarity")}
        result[label].update({k:float(np.mean([r["records"][label][k] for r in records])) for k in ("proxy_edge_accuracy","proxy_edge_macro_f1","elapsed_seconds","classifier_top1_accuracy","candidate_miss_rate","candidate_normalized_nll")});result[label]["failures"]=sum(not r["records"][label]["success"] for r in records)
    result["deltas"]={k:result["candidate"][k]-result["baseline"][k] for k in result["baseline"]};result["trip_bindings"]=[{"trip_id":r["trip_id"],"path":(OUTPUT/(r["trip_id"]+".json")).relative_to(c.ROOT).as_posix(),"sha256":c.sha256_file(OUTPUT/(r["trip_id"]+".json"))} for r in records];result["export_manifest_sha256"]=c.sha256_file(manifest);c.write_new_json(OUTPUT/"summary.json",result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--trip");p.add_argument("--aggregate",action="store_true");a=p.parse_args(argv)
    if a.aggregate:aggregate()
    elif a.trip:run_trip(a.trip)
    else:p.error("choose --trip or --aggregate")
if __name__=="__main__":main()
