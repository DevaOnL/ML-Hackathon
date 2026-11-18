#!/usr/bin/env python3
"""One-shot dense internal TRAIN16 route confirmation, after final science freeze.

No test-led tuning: use the exact candidate decoder frozen on tuning64 OOF;
baseline selects its own strongest decoder from the same eight tuning configs.
Both emissions were fitted on identical TRAIN64, never on these16 this iteration.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from mapmatch.data import discover_trips,load_graph,load_track,load_route
from mapmatch.features import reference_observation_labels
from mapmatch.matcher import MatchConfig
from mapmatch.accuracy_matcher import AccuracyTransitionPolicy
from scripts import accuracy_common as common,accuracy_confirmation as confirmation
from scripts.accuracy_route_validation import scalar_prediction,score_prediction

OUTPUT=confirmation.CONFIRMATION/"routes"


def frozen_configurations(frozen):
    choice=frozen["decoder_selection"]["configuration"]
    candidate={"name":choice["name"],"config":MatchConfig(**choice["match_config"]),"transition":AccuracyTransitionPolicy(**choice["transition_policy"])}
    record=json.loads((common.ROOT/frozen["decoder_selection"]["path"]).read_text())
    best=max(record["all_configurations"],key=lambda c:(c["v1_control"]["length_f1"],c["v1_control"]["edge_f1"],-c["order"]))
    baseline={"name":best["name"],"config":MatchConfig(**best["match_config"]),"transition":AccuracyTransitionPolicy(**best["transition_policy"])}
    return {"baseline":baseline,"candidate":candidate}


def run_trip(trip_id):
    frozen=confirmation.verify_frozen();protocol=json.loads(common.PROTOCOL_PATH.read_text())
    if trip_id not in protocol["confirmation_trip_ids"]:raise ValueError("only16prospectivelyreservedTRAINtrips allowed")
    family=frozen["decoder_selection"]["emission_family"];record_path=confirmation.CONFIRMATION/(family+".json");record=json.loads(record_path.read_text())
    if record["frozen_selection_sha256"]!=common.sha256_file(confirmation.FREEZE) or record["status"]!="complete":raise ValueError("paired64fitclassifiers notqualified")
    output=OUTPUT/(trip_id+".json");private=common.PRIVATE_ROOT/"route_confirmation"/trip_id
    if output.exists() or private.exists():raise FileExistsError("reservedroute alreadyclaimed/scored")
    from mapmatch.accuracy_artifacts import load_accuracy_confirmation_model
    models={}
    for label in ("baseline","candidate"):
        part=record["records"][label]
        models[label]=load_accuracy_confirmation_model(common.ROOT/part["model_path"],expected_sha256=part["model_sha256"],expected_train_trip_ids=protocol["tuning_trip_ids"],expected_validation_trip_ids=protocol["confirmation_trip_ids"],expected_protocol_sha256=frozen["protocol_sha256"])
    prefix=next(p for p in discover_trips(common.ROOT/"Student folder"/"train") if p.name==trip_id);graph=load_graph(prefix);track=load_track(prefix.with_suffix(".track"));configs=frozen_configurations(frozen);predictions={}
    private.mkdir(parents=True,exist_ok=False)
    # Complete BOTH predictions before loading this trip's reference.
    for label in ("baseline","candidate"):
        result,summary=scalar_prediction(graph,track,models[label],configs[label]);predictions[label]=(result,summary)
        if result is not None:np.savez_compressed(private/(label+".npz"),route=result.route,matched_edges=result.matched_edges,observation_indices=result.observation_indices)
    reference=load_route(prefix.with_suffix(".route"));labels=reference_observation_labels(graph,track,reference);scored={}
    for label,(result,summary) in predictions.items():scored[label]=score_prediction(graph,track,result,reference,summary,labels)
    output_record={"created_at_utc":common.utc_now(),"trip_id":trip_id,"family":family,"records":scored,"fulltrack_observations":len(track),"frozen_selection_sha256":common.sha256_file(confirmation.FREEZE),"paired_fit_record_sha256":common.sha256_file(record_path),"comparison":"one-shot dense route comparison on internal TRAIN16; historical80exposure disclosed","candidate_decoder_source":"frozen6tuningroutesOOF highestlengthF1","baseline_decoder_source":"ownhighestlengthF1ontheSAME6tuningroutesOOF8configmenu","reference_loaded_after_both_predictions":True,"student_used":False,"evaluator_used":False}
    confirmation.verify_frozen();common.write_new_json(output,output_record);print(json.dumps({"event":"reserved_route_confirmation","trip_id":trip_id,"baseline_lengthF1":scored["baseline"]["metrics"]["length_f1"],"candidate_lengthF1":scored["candidate"]["metrics"]["length_f1"]}),flush=True);return output_record


def aggregate():
    frozen=confirmation.verify_frozen();protocol=json.loads(common.PROTOCOL_PATH.read_text());records=[json.loads((OUTPUT/(t+".json")).read_text()) for t in protocol["confirmation_trip_ids"]]
    family=frozen["decoder_selection"]["emission_family"];paired=confirmation.CONFIRMATION/(family+".json");configs=frozen_configurations(frozen)
    for trip,record in zip(protocol["confirmation_trip_ids"],records):
        if record["trip_id"]!=trip or record["family"]!=family or record["paired_fit_record_sha256"]!=common.sha256_file(paired) or record["frozen_selection_sha256"]!=common.sha256_file(confirmation.FREEZE) or record.get("reference_loaded_after_both_predictions") is not True or record.get("student_used") is not False or record.get("evaluator_used") is not False:raise ValueError("reserved route population/model/fit/freeze mismatch")
        for label in ("baseline","candidate"):
            if record["records"][label]["configuration"]!=configs[label]["name"] or record["records"][label]["observations"]!=record["fulltrack_observations"]:raise ValueError("reserved route configuration/denominator mismatch")
    if len({r["trip_id"] for r in records})!=16:raise ValueError("reserved routes must be16unique trips")
    result={"created_at_utc":common.utc_now(),"routes":len(records),"comparison_role":"internalTRAIN16historicallyexposednotblind","family":frozen["decoder_selection"]["emission_family"],"frozen_selection_sha256":common.sha256_file(confirmation.FREEZE),"student_used":False,"evaluator_used":False}
    for label in ("baseline","candidate"):
        result[label]={k:float(np.mean([r["records"][label]["metrics"][k] for r in records])) for k in ("length_f1","edge_f1","route_continuity")}
        result[label].update({k:float(np.mean([r["records"][label][k] for r in records])) for k in ("proxy_edge_accuracy","proxy_edge_macro_f1","elapsed_seconds")});result[label]["failures"]=sum(not r["records"][label]["success"] for r in records)
    result["deltas"]={k:result["candidate"][k]-result["baseline"][k] for k in ("length_f1","edge_f1","route_continuity","proxy_edge_accuracy","proxy_edge_macro_f1")}
    paired=[]
    for key in ("length_f1","edge_f1","proxy_edge_accuracy","proxy_edge_macro_f1"):
        def panel(label):return {"per_trip":[{"trip_id":r["trip_id"],"top1_accuracy":r["records"][label]["metrics"][key] if key in r["records"][label]["metrics"] else r["records"][label][key]} for r in records]}
        p=confirmation.paired_trip_delta(panel("baseline"),panel("candidate"));p["metric"]=key;paired.append(p)
    result["paired_trip_deltas"]=paired;result["trip_bindings"]=[{"trip_id":r["trip_id"],"path":(OUTPUT/(r["trip_id"]+".json")).relative_to(common.ROOT).as_posix(),"sha256":common.sha256_file(OUTPUT/(r["trip_id"]+".json"))} for r in records];common.write_new_json(confirmation.CONFIRMATION/"route_summary.json",result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--trip");p.add_argument("--aggregate",action="store_true");a=p.parse_args(argv)
    if a.aggregate:aggregate()
    elif a.trip:run_trip(a.trip)
    else:p.error("choose --trip or --aggregate")
if __name__=="__main__":main()
