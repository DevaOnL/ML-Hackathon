#!/usr/bin/env python3
"""Dense TRAIN-only fold-qualified route validation, never student/evaluator.

Each fold estimator scores every original GPS row with fulltrack motion windows.
Only scalar metrics are public. Original graph/track/reference/routes stay private.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from mapmatch.accuracy_matcher import AccuracyMapMatcher,AccuracyTransitionPolicy,load_accuracy_emission_model
from mapmatch.matcher import MatchConfig,MatchingError
from mapmatch.data import discover_trips,load_graph,load_track,load_route,validate_route
from mapmatch.features import reference_observation_labels
from mapmatch.metrics import score_routes
from scripts import accuracy_common as common,accuracy_confirmation as confirmation

OUTPUT=common.EXPERIMENT_ROOT/"decoder"/"results"
MENU=common.EXPERIMENT_ROOT/"decoder"/"preregistered_menu.json"
AUTHORIZATION=common.EXPERIMENT_ROOT/"decoder"/"execution_authorization.json"


def selected_routes(protocol):
    result=[]
    for fold in protocol["folds"]:
        ids=sorted(fold["validation_trip_ids"],key=lambda t:(hashlib.sha256(f"{common.SEED}:{t}".encode()).hexdigest(),t))[:2]
        for trip in ids:result.append({"trip_id":trip,"fold":fold["fold"],"train_trip_ids":fold["train_trip_ids"],"validation_trip_ids":fold["validation_trip_ids"]})
    if len(result)!=6 or len({r["trip_id"] for r in result})!=6 or any(r["trip_id"] in r["train_trip_ids"] or r["trip_id"] not in r["validation_trip_ids"] for r in result):raise ValueError("six OOF route plan invalid")
    return result


def configurations(menu):
    """Interpret only predeclared fields; descriptive policy notes are not kwargs."""
    policy={k:v for k,v in menu.get("candidate_policy",{}).items() if k in MatchConfig.__dataclass_fields__}
    base=MatchConfig(**{**dict(sigma_m=7.5,beta_m=5,max_candidates=12,candidate_retry_limit=12,candidate_radius_m=80,max_radius_m=1000,observation_distance_m=0,observation_interval=5),**menu.get("base_config_overrides",{}),**policy})
    records=[]
    for i,item in enumerate(menu["configurations"]):
        config=replace(base,**item["match_overrides"]);transition=AccuracyTransitionPolicy(**item["transition_policy"])
        if config.max_candidates!=12 or config.candidate_retry_limit!=12 or config.observation_distance_m!=0:raise ValueError("first decoder menu must keep fixedK12 and dense observations")
        records.append({"order":i,"name":item["name"],"config":config,"transition":transition})
    if len(records)>8 or len({r["name"] for r in records})!=len(records):raise ValueError("bounded decoder menu invalid")
    return records


class CachedEmission:
    """Same graph/track/candidate-policy reuse only; check all candidate geometry."""
    def __init__(self,model):self.model=model;self.key=None;self.scores=None;self.graph=None
    def score_candidates_batch(self,graph,track,indices,candidates):
        h=hashlib.sha256();h.update(np.asarray(track).tobytes());h.update(np.asarray(indices).tobytes())
        for c in candidates:
            for a in (c.edges,c.fractions,c.distances):h.update(np.asarray(a).tobytes())
        key=h.hexdigest()
        if self.key is None:self.scores=self.model.score_candidates_batch(graph,track,indices,candidates);self.key=key;self.graph=graph
        elif self.graph is not graph or self.key!=key:raise ValueError("candidate/track policy changed under emission cache")
        return [p.copy() for p in self.scores]


def trip_prefix(trip_id,allowed):
    if trip_id not in allowed:raise ValueError("route trip not in tuning-only selection plan")
    prefixes={p.name:p for p in discover_trips(common.ROOT/"Student folder"/"train")}
    if trip_id not in prefixes:raise ValueError("TRAIN trip not found")
    return prefixes[trip_id]


def load_fold(record,fold):
    f=next(f for f in record["folds"] if f["fold"]==fold["fold"])
    return load_accuracy_emission_model(common.ROOT/record["private_output"]/f"fold_{fold['fold']}.joblib",expected_sha256=f["private_model_sha256"],expected_fold_id=fold["fold"],expected_train_trip_ids=fold["train_trip_ids"],expected_validation_trip_ids=fold["validation_trip_ids"],expected_protocol_sha256=record["protocol_sha256"])


def scalar_prediction(graph,track,model,configuration):
    start=time.perf_counter()
    try:
        with threadpool_limits(limits=common.THREADS):result=AccuracyMapMatcher(graph,configuration["config"],model,configuration["transition"]).match(track)
        if not np.array_equal(result.observation_indices,np.arange(len(track))):raise ValueError("dense denominator silently changed")
        valid=validate_route(graph,result.route)
        if not valid["valid"]:raise ValueError("successful decoder returned disconnected route")
        return result,{"success":True,"elapsed_seconds":time.perf_counter()-start,"observations":len(track),"configuration":configuration["name"],"match_config":asdict(configuration["config"]),"transition_policy":asdict(configuration["transition"]),"diagnostics":{k:v for k,v in result.diagnostics.items() if k in ("accuracy_motion","candidate_expanded_observations","candidate_count_min","candidate_count_max","candidate_retry_attempts")}}
    except MatchingError as exc:
        return None,{"success":False,"elapsed_seconds":time.perf_counter()-start,"observations":len(track),"configuration":configuration["name"],"failure":str(exc),"metrics":{"length_f1":0.,"edge_f1":0.,"route_continuity":0.,"prediction_connected":False,"exact_match":False},"proxy_edge_accuracy":0.,"proxy_edge_macro_f1":0.}


def score_prediction(graph,track,result,reference,summary,labels):
    if result is None:return summary
    summary=dict(summary);summary["metrics"]=score_routes(graph,result.route,reference)
    predicted=graph.canonical_edges[result.matched_edges]
    summary["proxy_edge_accuracy"]=float(np.mean(predicted==labels.edges));summary["proxy_edge_macro_f1"]=float(f1_score(labels.edges,predicted,average="macro",zero_division=0));summary["extra_arc_count"]=summary["metrics"]["predicted_arc_count"]-summary["metrics"]["common_arc_count"];summary["route_sha256"]=hashlib.sha256(np.asarray(result.route,dtype="<i8").tobytes()).hexdigest()
    return summary


def run_trip(trip_id,family,trial_name):
    protocol=json.loads(common.PROTOCOL_PATH.read_text());plan=selected_routes(protocol);fold=next((r for r in plan if r["trip_id"]==trip_id),None)
    if fold is None:raise ValueError("trip not among six predetermined OOF routes")
    from scripts.accuracy_decoder import verify_menu_registration,require_decoder_authorization
    menu_digest=verify_menu_registration(MENU)
    auth=json.loads(AUTHORIZATION.read_text())
    candidate_manifest=common.ROOT/auth["candidate_manifest_path"]
    require_decoder_authorization(auth,candidate_manifest_sha256=common.sha256_file(candidate_manifest),decoder_menu_sha256=menu_digest)
    manifest=json.loads(candidate_manifest.read_text())
    if manifest["family"]!=family or manifest["trial"]!=trial_name or manifest["protocol_sha256"]!=common.sha256_file(common.PROTOCOL_PATH):raise ValueError("decoder execution candidate binding mismatch")
    if any(common.sha256_file(common.ROOT/p)!=digest for p,digest in manifest["source_sha256"].items()):raise ValueError("decoder source freeze differs from authorized manifest")
    menu=json.loads(MENU.read_text());configs=configurations(menu);path=OUTPUT/(trip_id+".json")
    start_pins={p:common.sha256_file(common.ROOT/p) for p in ("scripts/accuracy_route_validation.py","mapmatch/accuracy_matcher.py","scripts/accuracy_decoder.py")}
    start_menu=common.sha256_file(MENU)
    start_protocol=common.sha256_file(common.PROTOCOL_PATH)
    if path.exists():raise FileExistsError("route validation cannot overwrite scored trip")
    records=confirmation.verified_full_trials(family);candidate=next((r for r,p in records if r["name"]==trial_name),None);baseline=next(r for r,p in records if r["profile"]=="v1")
    if candidate is None:raise ValueError("unverified emission trial")
    from scripts.accuracy_decoder import assert_paired_models
    new_model=load_fold(candidate,fold);old_model=load_fold(baseline,fold);assert_paired_models(old_model,new_model)
    prefix=trip_prefix(trip_id,[r["trip_id"] for r in plan]);graph=load_graph(prefix);track=load_track(prefix.with_suffix(".track"));emission=CachedEmission(new_model);base_emission=CachedEmission(old_model);prediction_records=[];reference=None;labels=None
    private=common.PRIVATE_ROOT/"decoder_validation"/trip_id
    if private.exists():raise FileExistsError("route private output already reserved")
    private.mkdir(parents=True,exist_ok=False)
    for cfg in configs:
        result,summary=scalar_prediction(graph,track,emission,cfg)
        if reference is None:reference=load_route(prefix.with_suffix(".route"));labels=reference_observation_labels(graph,track,reference)
        summary=score_prediction(graph,track,result,reference,summary,labels);summary["kind"]="candidate"
        if result is not None:np.savez_compressed(private/(cfg["name"]+".npz"),route=result.route,matched_edges=result.matched_edges,observation_indices=result.observation_indices)
        prediction_records.append(summary);print(json.dumps({"event":"dense_oof_route","trip_id":trip_id,"config":cfg["name"],"length_f1":summary["metrics"]["length_f1"],"edge_f1":summary["metrics"]["edge_f1"],"success":summary["success"]}),flush=True)
    for cfg in configs:
        result,summary=scalar_prediction(graph,track,base_emission,cfg);summary=score_prediction(graph,track,result,reference,summary,labels);summary["kind"]="v1_control";prediction_records.append(summary)
        if result is not None:np.savez_compressed(private/("v1_"+cfg["name"]+".npz"),route=result.route,matched_edges=result.matched_edges,observation_indices=result.observation_indices)
    out={"created_at_utc":common.utc_now(),"trip_id":trip_id,"fold":fold["fold"],"emission_family":family,"emission_trial":trial_name,"candidate_trial_sha256":common.sha256_file(next(p for r,p in records if r["name"]==trial_name)),"protocol_sha256":common.sha256_file(common.PROTOCOL_PATH),"decoder_menu_sha256":common.sha256_file(MENU),"source_sha256":{p:common.sha256_file(common.ROOT/p) for p in ("scripts/accuracy_route_validation.py","mapmatch/accuracy_matcher.py","scripts/accuracy_decoder.py")},"model_train_trip_ids":fold["train_trip_ids"],"model_validation_trip_ids":fold["validation_trip_ids"],"fulltrack_observations":len(track),"predictions":prediction_records,"confirmation_used":False,"student_used":False,"evaluator_used":False,"reference_used_for_prediction":False}
    if out["source_sha256"]!=start_pins or out["decoder_menu_sha256"]!=start_menu or out["protocol_sha256"]!=start_protocol:raise ValueError("decoder source/menu/protocol changed during measured route")
    common.write_new_json(path,out);return out


def select_configuration():
    protocol=json.loads(common.PROTOCOL_PATH.read_text());plan=selected_routes(protocol);menu=json.loads(MENU.read_text());cfgs=configurations(menu);records=[json.loads((OUTPUT/(r["trip_id"]+".json")).read_text()) for r in plan];models={(r["emission_family"],r["emission_trial"]) for r in records}
    if len(models)!=1 or any(r["protocol_sha256"]!=common.sha256_file(common.PROTOCOL_PATH) or r["decoder_menu_sha256"]!=common.sha256_file(MENU) for r in records):raise ValueError("mixed decoder/model protocol records")
    for record,fold in zip(records,plan):
        if record["trip_id"]!=fold["trip_id"] or record["fold"]!=fold["fold"] or record["model_train_trip_ids"]!=fold["train_trip_ids"] or record["model_validation_trip_ids"]!=fold["validation_trip_ids"]:raise ValueError("decoder route fold identity changed")
        if any(common.sha256_file(common.ROOT/p)!=digest for p,digest in record["source_sha256"].items()):raise ValueError("decoder implementation changed before selection")
        if any(p.get("observations")!=record["fulltrack_observations"] for p in record["predictions"]):raise ValueError("decoder route observation denominator changed")
    table=[]
    for cfg in cfgs:
        row={"name":cfg["name"],"order":cfg["order"],"match_config":asdict(cfg["config"]),"transition_policy":asdict(cfg["transition"])}
        for kind in ("candidate","v1_control"):
            a=[next(p for p in r["predictions"] if p["configuration"]==cfg["name"] and p["kind"]==kind) for r in records]
            row[kind]={key:float(np.mean([p["metrics"][key] for p in a])) for key in ("length_f1","edge_f1","route_continuity")}
            row[kind].update({key:float(np.mean([p[key] for p in a])) for key in ("proxy_edge_accuracy","proxy_edge_macro_f1","elapsed_seconds")});row[kind]["failures"]=sum(not p["success"] for p in a)
        table.append(row)
    winner=max(table,key=lambda t:(t["candidate"]["length_f1"],t["candidate"]["edge_f1"],-t["order"]));family,trial=next(iter(models))
    decision={"created_at_utc":common.utc_now(),"selection_complete":True,"selection_role":"TRAIN64 six-route dense OOF development selection, not confirmation","selection_rule":"equal-trip mean lengthF1 thenedgeF1 thenmenuorder; failureszero","emission_family":family,"emission_trial":trial,"selected_configuration":winner,"all_configurations":table,"selection_trips":[r["trip_id"] for r in plan],"result_bindings":[{"path":str((OUTPUT/(r["trip_id"]+".json")).relative_to(common.ROOT)),"sha256":common.sha256_file(OUTPUT/(r["trip_id"]+".json"))} for r in plan],"protocol_sha256":common.sha256_file(common.PROTOCOL_PATH),"decoder_menu_sha256":common.sha256_file(MENU),"confirmation_used":False,"student_used":False,"evaluator_used":False}
    common.write_new_json(common.EXPERIMENT_ROOT/"decoder"/"selection.json",decision);return decision


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--trip");p.add_argument("--family",choices=common.FAMILIES);p.add_argument("--trial");p.add_argument("--select",action="store_true");a=p.parse_args(argv)
    if a.select:select_configuration()
    elif a.trip and a.family and a.trial:run_trip(a.trip,a.family,a.trial)
    else:p.error("choose --select or --trip/--family/--trial")

if __name__=="__main__":main()
