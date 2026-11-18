#!/usr/bin/env python3
"""Export a frozen genuine all80 TRAIN accuracy candidate; never fit on test data.

Export does not authorize deployment. Internal confirmation and a final named
student-development comparison remain independent gates; evaluator is not read.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import joblib
import numpy as np
from scripts import accuracy_common as c,accuracy_confirmation as confirmation,accuracy_trees as tuning
from mapmatch.features import FEATURE_NAMES,FEATURE_VERSION

ARTIFACTS=c.ROOT/"artifacts"/"accuracy_improvement"


def export():
    frozen=confirmation.verify_frozen();family=frozen["decoder_selection"]["emission_family"];selection=frozen["families"][family]["candidate"]
    status=json.loads((confirmation.CONFIRMATION/(family+".json")).read_text());routes=json.loads((confirmation.CONFIRMATION/"route_summary.json").read_text())
    freeze_sha=c.sha256_file(confirmation.FREEZE)
    if status["frozen_selection_sha256"]!=freeze_sha or routes["frozen_selection_sha256"]!=freeze_sha:raise ValueError("confirmation evidence not bound to current freeze")
    protocol=json.loads(c.PROTOCOL_PATH.read_text())
    if routes.get("routes")!=16 or [b["trip_id"] for b in routes.get("trip_bindings",[])]!=protocol["confirmation_trip_ids"]:raise ValueError("unique complete16route confirmation evidence required")
    for binding in routes["trip_bindings"]:
        if c.sha256_file(c.ROOT/binding["path"])!=binding["sha256"]:raise ValueError("confirmation route evidence drift")
    if any(p["trip_count"]!=16 for p in routes.get("paired_trip_deltas",[])) or len(routes.get("paired_trip_deltas",[]))!=4:raise ValueError("paired route metrics did not use16distincttrips")
    for family_key in c.FAMILIES:
        r=json.loads((confirmation.CONFIRMATION/(family_key+".json")).read_text())
        if r["status"]!="complete" or r["frozen_selection_sha256"]!=freeze_sha:raise ValueError("all five family confirmations must finish")
    classifier_delta=status["records"]["candidate"]["metrics"]["trip_macro_top1_accuracy"]-status["records"]["baseline"]["metrics"]["trip_macro_top1_accuracy"]
    length_delta=routes["candidate"]["length_f1"]-routes["baseline"]["length_f1"];edge_delta=routes["candidate"]["edge_f1"]-routes["baseline"]["edge_f1"]
    if routes.get("family")!=family or status.get("family")!=family or not all(np.isfinite(x) for x in (classifier_delta,length_delta,edge_delta)) or not np.allclose([classifier_delta,length_delta,edge_delta],[status["metric_deltas"]["trip_macro_top1_accuracy"],routes["deltas"]["length_f1"],routes["deltas"]["edge_f1"]],rtol=0,atol=1e-12):raise ValueError("confirmation gate numbers nonfinite/inconsistent or wrong family")
    if classifier_delta<=0 or length_delta<0 or edge_delta<0 or routes["candidate"]["failures"]:raise ValueError("candidate failed internal top1/fullroute export gate; no test-led retry allowed")
    if ARTIFACTS.exists():raise FileExistsError("refuse to overwrite any accuracy artifact release")
    ARTIFACTS.mkdir(parents=True,exist_ok=False)
    data=c.load_full_train_dataset();profile=selection["profile"];X=c.profile_features(data.X,data.offsets,profile);estimator=tuning.FACTORIES[family](selection["parameters"],c.SEED,c.THREADS)
    seconds,fit_warnings,bank=c.fit_estimator(estimator,X,data.y,data.training_weights(),family,3,"knn_bank" if family=="knn_distance_weighted" else "weighted",60000)
    bundle={"estimator":estimator,"fit_role":"all_train_deployment","training_scope":"all80 TRAIN final refit after frozen selection; no student/evaluator fitting","profile":profile,"feature_names":c.profile_names(profile),"base_feature_names":FEATURE_NAMES,"base_feature_version":FEATURE_VERSION,"feature_version":FEATURE_VERSION if profile=="v1" else "pathsnap-candidate-v2","family":family,"run_name":selection["run_name"]+"_all80","parameters":selection["parameters"],"train_trip_ids":sorted(np.unique(data.observation_trip_ids).tolist()),"validation_trip_ids":[],"protocol_sha256":frozen["protocol_sha256"],"source_sha256":frozen["source_sha256"],"model_name":"accuracy_improved_"+family,"seed":c.SEED,"frozen_selection_sha256":freeze_sha}
    path=ARTIFACTS/"model.joblib";joblib.dump(bundle,path,compress=3);model_sha=c.sha256_file(path)
    from mapmatch.accuracy_artifacts import load_accuracy_deployment_model
    wrapper=load_accuracy_deployment_model(path,expected_sha256=model_sha,expected_protocol_sha256=frozen["protocol_sha256"]);p,checks=c.strict_positive_probabilities(wrapper.estimator,X[:500]);assert np.isfinite(p).all()
    choice=frozen["decoder_selection"]["configuration"];c.write_new_json(ARTIFACTS/"match_config.json",choice["match_config"]);c.write_new_json(ARTIFACTS/"transition_policy.json",choice["transition_policy"])
    metadata={k:v for k,v in bundle.items() if k!="estimator"};metadata.update({"model_sha256":model_sha,"created_at_utc":c.utc_now(),"training_seconds":seconds,"training_observations":len(data.observation_trip_ids),"training_candidate_rows":len(data.y),"fit_warnings":fit_warnings,"bank":bank,"probability_checks":checks,"confirmation_record_sha256":c.sha256_file(confirmation.CONFIRMATION/(family+".json")),"route_confirmation_sha256":c.sha256_file(confirmation.CONFIRMATION/"route_summary.json"),"deployment_authorized":False,"student_used_for_fit":False,"evaluator_used_for_fit":False})
    c.write_new_json(ARTIFACTS/"model_metadata.json",metadata)
    manifest={"schema":"pathsnap-accuracy-export-v1","created_at_utc":c.utc_now(),"model_sha256":model_sha,"protocol_sha256":frozen["protocol_sha256"],"frozen_selection_sha256":freeze_sha,"decoder_configuration_name":choice["name"],"source_sha256":{**frozen["source_sha256"],"scripts/accuracy_export.py":c.sha256_file(__file__)},"files":{f.name:c.sha256_file(f) for f in ARTIFACTS.iterdir() if f.is_file()},"student_development_comparison_required":True,"evaluator_not_used_for_selection":True,"deployment_authorized":False}
    c.write_new_json(ARTIFACTS/"release_manifest.json",manifest);confirmation.verify_frozen();print(json.dumps({"event":"accuracy_export","family":family,"model_sha256":model_sha,"profile":profile,"feature_count":len(wrapper.feature_names),"training_seconds":seconds}),flush=True);return manifest

if __name__=="__main__":export()
