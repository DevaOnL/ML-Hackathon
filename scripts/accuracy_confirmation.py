#!/usr/bin/env python3
"""Freeze TRAIN tuning decisions and perform one paired internal confirmation.

Not a blind official test: all80 trips were seen by historical released fits.
Nothing here accepts an evaluator/student input path or performs test-led tuning.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from scripts import accuracy_common as common, accuracy_trees as tuning
from mapmatch.features import FEATURE_NAMES, FEATURE_VERSION

CONFIRMATION = common.EXPERIMENT_ROOT / "confirmation"
FREEZE = CONFIRMATION / "frozen_selection.json"


def verified_full_trials(family):
    protocol=json.loads(common.PROTOCOL_PATH.read_text())
    for relative,digest in protocol["source_freeze"].items():
        if common.sha256_file(common.ROOT/relative)!=digest:
            raise ValueError("registered source drift before selection: "+relative)
    directory=common.EXPERIMENT_ROOT/tuning.OUTPUT_NAMES[family]
    candidates={c["run_name"]:c for c in tuning.menu(family)}
    records=[]
    for path in sorted((directory/"cv").glob("*_full.json")):
        record=json.loads(path.read_text());stem=record["name"][:-5]
        if stem not in candidates:raise ValueError("unregistered completed trial")
        tuning.validate_saved(record,candidates[stem],family,[0,1,2])
        if len(record["folds"])!=3:raise ValueError("incomplete fold records")
        declared=protocol["folds"]
        for fold in record["folds"]:
            f=declared[fold["fold"]]
            if fold["train_trip_ids"]!=f["train_trip_ids"] or fold["validation_trip_ids"]!=f["validation_trip_ids"]:raise ValueError("trial fold trip manifest drift")
            model=common.ROOT/record["private_output"]/f"fold_{fold['fold']}.joblib"
            if common.sha256_file(model)!=fold["private_model_sha256"]:raise ValueError("trial fold artifact drift")
        records.append((record,path))
    promotion=json.loads((directory/"promotion.json").read_text())
    expected={c["run_name"]+"_full" for c in promotion["promoted"]}
    amendment=common.EXPERIMENT_ROOT/"BUDGET_AMENDMENT.json"
    if family=="bagging" and amendment.exists():
        budget=json.loads(amendment.read_text())
        if budget.get("applied_before_any_confirmation_score") is not True or budget.get("confirmation_scored") is not False:raise ValueError("invalid late budget amendment")
        expected=set(budget["completed_required_runs"])
        records=[(r,p) for r,p in records if r["name"] in expected]
    if {r["name"] for r,p in records}!=expected or len(records)<2 or not any(r["profile"]=="v1" for r,p in records):raise ValueError("missing full3 controls/finalists; all promoted trials must complete before selection")
    return records


def freeze_selection(decoder_record=None):
    selected={}
    for family in common.FAMILIES:
        records=verified_full_trials(family)
        winner,path=max(records,key=lambda rp:(*common.rank_trial(rp[0]),rp[0]["name"]))
        baseline,bpath=next((r,p) for r,p in records if r["profile"]=="v1")
        def binding(r,p):return {"run_name":r["name"],"family":r["family"],"profile":r["profile"],"parameters":r["parameters"],"trial_path":p.relative_to(common.ROOT).as_posix(),"trial_sha256":common.sha256_file(p),"tuning_metrics":r["oof_metrics"],"private_output":r["private_output"]}
        selected[family]={"candidate":binding(winner,path),"baseline":binding(baseline,bpath)}
    if decoder_record is None:raise ValueError("freeze requires completed OOF decoder decision, not later outer tuning")
    path=Path(decoder_record).resolve()
    if not path.is_relative_to(common.EXPERIMENT_ROOT/"decoder") or not path.is_file():raise ValueError("decoder selection record must be an accuracy experiment output")
    d=json.loads(path.read_text())
    if d.get("confirmation_used") is not False or d.get("student_used") is not False or d.get("evaluator_used") is not False or d.get("selection_complete") is not True:raise ValueError("decoder selection qualification failed")
    menu=common.EXPERIMENT_ROOT/"decoder"/"preregistered_menu.json"
    if d.get("protocol_sha256")!=common.sha256_file(common.PROTOCOL_PATH) or d.get("decoder_menu_sha256")!=common.sha256_file(menu):raise ValueError("decoder decision protocol/menu provenance drift")
    from scripts.accuracy_route_validation import selected_routes
    expected_trips=[r["trip_id"] for r in selected_routes(json.loads(common.PROTOCOL_PATH.read_text()))]
    if d.get("selection_trips")!=expected_trips or len(d.get("result_bindings",[]))!=6:raise ValueError("decoder selection route population differs")
    for binding in d["result_bindings"]:
        p=(common.ROOT/binding["path"]).resolve()
        if not p.is_relative_to(common.EXPERIMENT_ROOT/"decoder"/"results") or common.sha256_file(p)!=binding["sha256"]:raise ValueError("decoder route evidence drift")
    attestation=common.EXPERIMENT_ROOT/"decoder"/"execution_attestation.json"
    a=json.loads(attestation.read_text())
    if a.get("status")!="pass" or a.get("protocol_sha256")!=d["protocol_sha256"] or a.get("decoder_menu_sha256")!=d["decoder_menu_sha256"] or a.get("route_result_bindings")!=[{"trip_id":t,"path":b["path"],"sha256":b["sha256"]} for t,b in zip(expected_trips,d["result_bindings"])]:raise ValueError("complete decoder execution attestation required")
    family=d.get("emission_family");trial=d.get("emission_trial")
    if family not in selected or selected[family]["candidate"]["run_name"]!=trial:raise ValueError("decoder emission does not match frozen family candidate")
    best=max(selected,key=lambda f:(*common.rank_trial({"oof_metrics":selected[f]["candidate"]["tuning_metrics"]}),f))
    if family!=best:raise ValueError("decoder family differs from registered highest tuning-top1 family")
    from scripts.accuracy_decoder import decoder_configurations
    registered=decoder_configurations();recomputed=[]
    evidence=[json.loads((common.ROOT/b["path"]).read_text()) for b in d["result_bindings"]]
    from dataclasses import asdict
    for order,cfg in enumerate(registered):
        row={"name":cfg.name,"order":order,"match_config":asdict(cfg.match_config),"transition_policy":asdict(cfg.transition_policy)}
        for kind in ("candidate","v1_control"):
            values=[next(p for p in r["predictions"] if p["kind"]==kind and p["configuration"]==cfg.name) for r in evidence]
            row[kind]={key:float(np.mean([p["metrics"][key] for p in values])) for key in ("length_f1","edge_f1","route_continuity")}
            row[kind].update({key:float(np.mean([p[key] for p in values])) for key in ("proxy_edge_accuracy","proxy_edge_macro_f1","elapsed_seconds")});row[kind]["failures"]=sum(not p["success"] for p in values)
        recomputed.append(row)
    winner=max(recomputed,key=lambda row:(row["candidate"]["length_f1"],row["candidate"]["edge_f1"],-row["order"]))
    if d["all_configurations"]!=recomputed or d["selected_configuration"]!=winner:raise ValueError("decoder selected configuration/table differs from measured preregistered evidence")
    baseline_decoder=max(recomputed,key=lambda row:(row["v1_control"]["length_f1"],row["v1_control"]["edge_f1"],-row["order"]))
    manifest={"schema":"pathsnap-accuracy-frozen-selection-v1","created_at_utc":common.utc_now(),"protocol_sha256":common.sha256_file(common.PROTOCOL_PATH),"decision_rule":["highest TRAIN64 three-fold equal-trip observation top1","higher binary macroF1","lower binary logloss","reverse lexicographic run name exact tie"],"families":selected,"decoder_selection":{"path":path.relative_to(common.ROOT).as_posix(),"sha256":common.sha256_file(path),"configuration":d["selected_configuration"],"baseline_configuration":baseline_decoder,"emission_family":d["emission_family"],"emission_trial":d["emission_trial"],"execution_attestation_path":attestation.relative_to(common.ROOT).as_posix(),"execution_attestation_sha256":common.sha256_file(attestation)},"source_sha256":{relative:common.sha256_file(common.ROOT/relative) for relative in ["scripts/accuracy_common.py","scripts/accuracy_trees.py","scripts/accuracy_linear_bayes.py","scripts/accuracy_knn.py","mapmatch/accuracy_features.py","scripts/accuracy_confirmation.py","mapmatch/accuracy_matcher.py","scripts/accuracy_decoder.py","scripts/accuracy_route_validation.py","scripts/accuracy_route_confirmation.py","scripts/accuracy_decoder_attestation.py","scripts/accuracy_export.py","scripts/accuracy_student_comparison.py","backend/accuracy_runtime.py","mapmatch/accuracy_artifacts.py","mapmatch/features.py","mapmatch/learning.py","mapmatch/experiments.py","mapmatch/phase_models.py","mapmatch/data.py","mapmatch/metrics.py","mapmatch/matcher.py","mapmatch/routing.py","mapmatch/candidates.py"]},"confirmation_scored":False,"student_scored":False,"evaluator_scored":False,"historical_exposure_caveat":"Reserved16 were excluded prospectively in THIS iteration, but prior released models/hyperparameters used all80 TRAIN; this is internal confirmation, never a new blind official test.","no_tuning_after_confirmation":True}
    amendment=common.EXPERIMENT_ROOT/"BUDGET_AMENDMENT.json"
    if amendment.exists():manifest["budget_amendment"]={"path":amendment.relative_to(common.ROOT).as_posix(),"sha256":common.sha256_file(amendment),"reason":"userrequestedexpeditedfinishbeforeconfirmation"}
    common.write_new_json(FREEZE,manifest)
    return manifest


def paired_trip_delta(baseline,candidate):
    a={x["trip_id"]:x["top1_accuracy"] for x in baseline["per_trip"]};b={x["trip_id"]:x["top1_accuracy"] for x in candidate["per_trip"]}
    if not a or len(a)!=len(baseline["per_trip"]) or len(b)!=len(candidate["per_trip"]):raise ValueError("paired metrics require nonempty unique trip records")
    if set(a)!=set(b):raise ValueError("paired trip populations differ")
    ids=sorted(a);delta=np.array([b[t]-a[t] for t in ids]);rng=np.random.default_rng(common.SEED)
    boot=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
    return {"metric":"unconditional observation top1, equal-trip paired delta","trip_count":len(delta),"mean_delta":float(delta.mean()),"trip_bootstrap_95_percentile_interval":np.quantile(boot,[.025,.975]).tolist(),"bootstrap_repeats":10000,"bootstrap_seed":common.SEED,"improved_trips":int(np.sum(delta>1e-12)),"decreased_trips":int(np.sum(delta< -1e-12)),"unchanged_trips":int(np.sum(np.abs(delta)<=1e-12)),"minimum_trip_delta":float(delta.min()),"maximum_trip_delta":float(delta.max()),"per_trip":[{"trip_id":t,"baseline":a[t],"candidate":b[t],"delta":b[t]-a[t]} for t in ids],"interpretation":"Trip-resampling uncertainty on internal retrospectively exposed TRAIN; not an independent official test confidence guarantee."}


def verify_frozen():
    d=json.loads(FREEZE.read_text())
    if d["protocol_sha256"]!=common.sha256_file(common.PROTOCOL_PATH):raise ValueError("protocol changed after freeze")
    if "budget_amendment" in d:
        amendment=d["budget_amendment"]
        if common.sha256_file(common.ROOT/amendment["path"])!=amendment["sha256"]:raise ValueError("frozen budget amendment changed")
    decoder=d.get("decoder_selection")
    if decoder is not None:
        decision=common.ROOT/decoder["path"]
        if common.sha256_file(decision)!=decoder["sha256"]:raise ValueError("frozen decoder decision changed")
        selection=json.loads(decision.read_text())
        if common.sha256_file(common.ROOT/decoder["execution_attestation_path"])!=decoder["execution_attestation_sha256"]:raise ValueError("frozen decoder execution attestation changed")
        attestation=json.loads((common.ROOT/decoder["execution_attestation_path"]).read_text())
        for relative,digest in attestation["complete_authorized_source_sha256"].items():
            if common.sha256_file(common.ROOT/relative)!=digest:raise ValueError("authorized decoder source changed: "+relative)
        for binding in selection["result_bindings"]:
            if common.sha256_file(common.ROOT/binding["path"])!=binding["sha256"]:raise ValueError("frozen decoder evidence changed")
        if common.sha256_file(common.EXPERIMENT_ROOT/"decoder"/"preregistered_menu.json")!=selection["decoder_menu_sha256"]:raise ValueError("frozen decoder menu changed")
    for relative,digest in d["source_sha256"].items():
        if common.sha256_file(common.ROOT/relative)!=digest:raise ValueError("source changed after freeze: "+relative)
    for family,parts in d["families"].items():
        for label,part in parts.items():
            if common.sha256_file(common.ROOT/part["trial_path"])!=part["trial_sha256"]:raise ValueError("frozen trial record changed")
            record=json.loads((common.ROOT/part["trial_path"]).read_text())
            for key,digest in record["source_sha256"].items():
                relative="scripts/accuracy_common.py" if key=="common" else "scripts/accuracy_trees.py" if key=="factory_module" else "mapmatch/accuracy_features.py" if key=="features" else key
                if common.sha256_file(common.ROOT/relative)!=digest:raise ValueError("transitive frozen trial source drift: "+relative)
    return d


def confirm_family(family):
    frozen=verify_frozen();output=CONFIRMATION/(family+".json");private=common.PRIVATE_ROOT/"confirmation"/family
    if output.exists() or private.exists():raise FileExistsError("one-shot confirmation already reserved or complete")
    private.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(common.PROTOCOL_PATH.read_text());data=common.load_full_train_dataset()
    train=common.subset_dataset(data,np.flatnonzero(np.isin(data.observation_trip_ids,protocol["tuning_trip_ids"])));valid=common.subset_dataset(data,np.flatnonzero(np.isin(data.observation_trip_ids,protocol["confirmation_trip_ids"])))
    if set(train.observation_trip_ids)&set(valid.observation_trip_ids) or len(np.unique(train.observation_trip_ids))!=64 or len(np.unique(valid.observation_trip_ids))!=16:raise ValueError("prospective confirmation partition failed")
    records={}
    for label in ("baseline","candidate"):
        item=frozen["families"][family][label];profile=item["profile"];X=common.profile_features(train.X,train.offsets,profile);V=common.profile_features(valid.X,valid.offsets,profile)
        estimator=tuning.FACTORIES[family](item["parameters"],common.SEED,common.THREADS)
        seconds,fit_warnings,bank=common.fit_estimator(estimator,X,train.y,train.training_weights(),family,3,"knn_bank" if family=="knn_distance_weighted" else "weighted",60000)
        start=time.perf_counter()
        with threadpool_limits(limits=common.THREADS):p,checks=common.strict_positive_probabilities(estimator,V)
        inference=time.perf_counter()-start
        np.savez_compressed(private/(label+"_probabilities.npz"),probabilities=p,offsets=valid.offsets,observation_trip_ids=valid.observation_trip_ids,y=valid.y)
        bundle={"estimator":estimator,"profile":profile,"feature_names":common.profile_names(profile),"base_feature_names":FEATURE_NAMES,"base_feature_version":FEATURE_VERSION,"feature_version":FEATURE_VERSION if profile=="v1" else "pathsnap-candidate-v2","family":family,"run_name":item["run_name"]+"_confirmation","parameters":item["parameters"],"fold":3,"fold_id":3,"train_trip_ids":protocol["tuning_trip_ids"],"validation_trip_ids":protocol["confirmation_trip_ids"],"training_scope":"TRAIN64 fitting, reserved16 one-shot internal confirmation; never historicallyunseen","source_sha256":frozen["source_sha256"],"protocol_sha256":frozen["protocol_sha256"]}
        model=private/(label+".joblib");joblib.dump(bundle,model,compress=3)
        records[label]={"metrics":common.metrics_for(valid,p),"training_seconds":seconds,"inference_seconds":inference,"fit_warnings":fit_warnings,"bank":bank,"probability_checks":checks,"model_path":model.relative_to(common.ROOT).as_posix(),"model_sha256":common.sha256_file(model),"profile":profile,"run_name":item["run_name"]}
        print(json.dumps({"event":"confirmation","family":family,"label":label,"top1":records[label]["metrics"]["trip_macro_top1_accuracy"]}),flush=True)
    result={"created_at_utc":common.utc_now(),"family":family,"status":"complete","frozen_selection_sha256":common.sha256_file(FREEZE),"protocol_sha256":frozen["protocol_sha256"],"comparison_role":"one-shot internally reserved TRAIN16; historicallyexposed","records":records,"paired_trip_top1":paired_trip_delta(records["baseline"]["metrics"],records["candidate"]["metrics"]),"no_selection_feedback":True,"student_used":False,"evaluator_used":False}
    result["metric_deltas"]={key:records["candidate"]["metrics"][key]-records["baseline"]["metrics"][key] for key in ("trip_macro_top1_accuracy","accuracy","macro_f1","log_loss","edge_macro_f1")}
    verify_frozen();common.write_new_json(output,result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--freeze",action="store_true");p.add_argument("--decoder-record");p.add_argument("--family",choices=common.FAMILIES);a=p.parse_args(argv)
    if a.freeze:freeze_selection(a.decoder_record)
    elif a.family:confirm_family(a.family)
    else:p.error("choose --freeze or --family")

if __name__=="__main__":main()
