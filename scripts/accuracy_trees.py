#!/usr/bin/env python3
"""Central bounded five-family accuracy experiment launch; TRAIN tuning64 only."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from scripts import accuracy_common as common


def tree_menu(family):
    if family == "bagging":
        baseline={"n_estimators":40,"max_samples":.7,"max_depth":14,"min_samples_leaf":3}
        return [
            {"family":family,"run_name":"bagging_v1_control","profile":"v1","parameters":baseline},
            {"family":family,"run_name":"bagging_context_control","profile":"context","parameters":baseline},
            {"family":family,"run_name":"bagging_context_deeper","profile":"context","parameters":{**baseline,"n_estimators":64,"max_depth":20,"min_samples_leaf":3}},
            {"family":family,"run_name":"bagging_context_smooth","profile":"context","parameters":{**baseline,"n_estimators":64,"max_depth":18,"min_samples_leaf":6}},
            {"family":family,"run_name":"bagging_compact","profile":"compact","parameters":{**baseline,"n_estimators":64,"max_depth":18,"min_samples_leaf":4}},
            {"family":family,"run_name":"bagging_physics","profile":"physics","parameters":{**baseline,"n_estimators":64,"max_depth":12,"min_samples_leaf":8}}]
    if family == "hist_gradient_boosting":
        baseline={"max_iter":104,"learning_rate":.15902391896503631,"max_leaf_nodes":22,"min_samples_leaf":38,"l2_regularization":.8604220152998319}
        return [
            {"family":family,"run_name":"histgb_v1_control","profile":"v1","parameters":baseline},
            {"family":family,"run_name":"histgb_context_control","profile":"context","parameters":baseline},
            {"family":family,"run_name":"histgb_context_long","profile":"context","parameters":{**baseline,"max_iter":240,"learning_rate":.075,"max_leaf_nodes":23,"min_samples_leaf":40,"l2_regularization":1.}},
            {"family":family,"run_name":"histgb_context_smooth","profile":"context","parameters":{**baseline,"max_iter":240,"learning_rate":.055,"max_leaf_nodes":15,"min_samples_leaf":60,"l2_regularization":2.}},
            {"family":family,"run_name":"histgb_compact","profile":"compact","parameters":{**baseline,"max_iter":240,"learning_rate":.075,"max_leaf_nodes":23}},
            {"family":family,"run_name":"histgb_context_large","profile":"context","parameters":{**baseline,"max_iter":180,"learning_rate":.10,"max_leaf_nodes":31,"min_samples_leaf":30,"l2_regularization":2.}}]
    raise ValueError("unsupported tree family")


def make_bagging_estimator(parameters, seed, n_jobs):
    p=dict(parameters)
    tree=DecisionTreeClassifier(max_depth=p.pop("max_depth"),min_samples_leaf=p.pop("min_samples_leaf"),random_state=seed)
    return BaggingClassifier(estimator=tree,**p,random_state=seed,n_jobs=n_jobs)


def make_histgb_estimator(parameters, seed, n_jobs):
    return HistGradientBoostingClassifier(**dict(parameters),random_state=seed,early_stopping=False)


def make_lr_estimator(parameters, seed, n_jobs):
    from scripts.accuracy_linear_bayes import make_estimator
    return make_estimator("logistic_regression",parameters,SimpleNamespace(seed=seed))


def make_gnb_estimator(parameters, seed, n_jobs):
    from scripts.accuracy_linear_bayes import make_estimator
    return make_estimator("gaussian_nb",parameters,SimpleNamespace(seed=seed))


def make_knn_estimator(parameters, seed, n_jobs):
    from scripts.accuracy_knn import make_knn_pipeline
    return make_knn_pipeline(**dict(parameters),n_jobs=min(2,n_jobs))


def menu(family):
    if family in ("bagging","hist_gradient_boosting"):
        return tree_menu(family)
    if family in ("logistic_regression","gaussian_nb"):
        from scripts.accuracy_linear_bayes import candidate_menu
        return [asdict(c) for c in candidate_menu(family)]
    if family == "knn_distance_weighted":
        base={"algorithm":"kd_tree","p":2,"leaf_size":40}
        configs=[("knn_v1_control","v1",{**base,"algorithm":"brute","n_neighbors":25}), ("knn_compact5","compact",{**base,"n_neighbors":5}), ("knn_compact15","compact",{**base,"n_neighbors":15}), ("knn_compact35","compact",{**base,"n_neighbors":35}), ("knn_physics15","physics",{**base,"n_neighbors":15}), ("knn_compact_l1","compact",{**base,"p":1,"n_neighbors":15})]
        return [{"family":family,"run_name":n,"profile":p,"parameters":cfg} for n,p,cfg in configs]
    raise ValueError("unknown family")


FACTORIES={"bagging":make_bagging_estimator,"hist_gradient_boosting":make_histgb_estimator,"logistic_regression":make_lr_estimator,"gaussian_nb":make_gnb_estimator,"knn_distance_weighted":make_knn_estimator}
OUTPUT_NAMES={"logistic_regression":"linear_bayes_lr","gaussian_nb":"linear_bayes_gnb","knn_distance_weighted":"knn","bagging":"bagging","hist_gradient_boosting":"histgb"}


def validate_saved(record, candidate, family, fold_ids):
    profile="v1" if candidate["profile"]=="baseline38" else candidate["profile"]
    expected_name=candidate["run_name"]+("_screen" if list(fold_ids)==[0] else "_full")
    if record.get("name")!=expected_name or record.get("status")!="complete" or record.get("family")!=family or record.get("parameters")!=candidate["parameters"] or record.get("profile")!=profile or record.get("fold_ids")!=list(fold_ids) or record.get("protocol_sha256")!=common.sha256_file(common.PROTOCOL_PATH):
        raise ValueError("saved trial provenance differs; refusing unsafe resume/promotion")
    sources=record.get("source_sha256",{})
    if sources.get("common")!=common.sha256_file(common.__file__) or sources.get("factory_module")!=common.sha256_file(__file__):
        raise ValueError("saved trial implementation drift")
    for key,digest in sources.items():
        if key in ("common","factory_module"):continue
        path=common.ROOT/"mapmatch/accuracy_features.py" if key=="features" else common.ROOT/key
        if common.sha256_file(path)!=digest:raise ValueError("saved implementation drift: "+key)
    return record


def run(family, stage):
    output=common.EXPERIMENT_ROOT/OUTPUT_NAMES[family]
    candidates=menu(family)
    protocol=json.loads(common.PROTOCOL_PATH.read_text())
    if not protocol.get("training_authorized"):raise ValueError("actual training is not authorized")
    if protocol["family_menus"].get(family)!=candidates:raise ValueError("current menu differs from preregistration")
    for item in candidates:common.profile_names(item["profile"]);FACTORIES[family](item["parameters"],common.SEED,1)
    if stage == "screen":
        path=output/"central_menu.json"
        if path.exists():
            if json.loads(path.read_text())["candidates"]!=candidates:raise ValueError("existing menu differs")
        else:common.write_new_json(path,{"created_utc":common.utc_now(),"candidates":candidates,"prospective_confirmation_used":False,"training_scope":"tuning64 only","stage":"fold0 screen, top2 plus v1 baseline promoted to full3 fresh fits"})
        for c in candidates:
            record_path=output/"cv"/(c["run_name"]+"_screen.json")
            if record_path.exists():validate_saved(json.loads(record_path.read_text()),c,family,[0]);continue
            common.run_cv_trial(family,c["run_name"]+"_screen",c["parameters"],c["profile"],output,FACTORIES[family],fold_ids=[0],fit_policy="knn_bank" if family=="knn_distance_weighted" else "weighted")
        return
    screens=[(c,validate_saved(json.loads((output/"cv"/(c["run_name"]+"_screen.json")).read_text()),c,family,[0])) for c in candidates]
    promoted=sorted(screens,key=lambda cr:(*common.rank_trial(cr[1]),cr[0]["run_name"]),reverse=True)[:2]
    baseline=next(cr for cr in screens if cr[0]["profile"] in ("v1","baseline38"))
    if baseline[0]["run_name"] not in {c["run_name"] for c,r in promoted}:promoted.append(baseline)
    if stage == "full":
        promotion=output/"promotion.json"
        if promotion.exists():
            if json.loads(promotion.read_text())["promoted"]!=[c for c,r in promoted]:raise ValueError("promotion provenance differs")
        else:common.write_new_json(promotion,{"created_utc":common.utc_now(),"rule":"fold0 equal-trip top1; binary macroF1; lower binary logloss; runname tie; top2 plus pairedbaseline full3 fresh fits","promoted":[c for c,r in promoted],"screen_selection_is_tuning":True,"confirmation_not_used":True})
        for c,_ in promoted:
            record_path=output/"cv"/(c["run_name"]+"_full.json")
            if record_path.exists():validate_saved(json.loads(record_path.read_text()),c,family,[0,1,2]);continue
            common.run_cv_trial(family,c["run_name"]+"_full",c["parameters"],c["profile"],output,FACTORIES[family],fit_policy="knn_bank" if family=="knn_distance_weighted" else "weighted")
        return
    raise ValueError("stage must be screen or full")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--family",choices=common.FAMILIES,required=True);p.add_argument("--stage",choices=("screen","full"),required=True);a=p.parse_args(argv);run(a.family,a.stage)


if __name__=="__main__":main()
