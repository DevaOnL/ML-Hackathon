#!/usr/bin/env python3
"""New TRAIN-only route-edge-F1-primary study, separate from prior frozen studies."""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import hashlib,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from mapmatch.candidates import CandidateIndex
from mapmatch.data import discover_trips,load_graph,load_track,load_route,validate_route,write_route
from mapmatch.matcher import MatchConfig,MatchingError
from mapmatch.routing import Router
from mapmatch.target96 import RoutePolicy,RouteTargetMatcher,MemoIndex,MemoEmissions
from mapmatch.metrics import score_routes
from mapmatch.accuracy_matcher import load_accuracy_emission_model
from scripts import accuracy_common as c,accuracy_confirmation as old
ROOT=c.ROOT;EXP=ROOT/'experiments/target96';PROTOCOL=EXP/'PROTOCOL.json'
SOURCE_PATHS=['scripts/target96.py','mapmatch/target96.py','mapmatch/matcher.py','mapmatch/routing.py','mapmatch/candidates.py','mapmatch/data.py','mapmatch/features.py','mapmatch/accuracy_features.py','mapmatch/accuracy_matcher.py','mapmatch/accuracy_artifacts.py','mapmatch/metrics.py']

def sourcepins():return {p:c.sha256_file(ROOT/p) for p in SOURCE_PATHS}

def menu():
    base=MatchConfig(sigma_m=7.160645916267228,beta_m=2,max_candidates=12,candidate_retry_limit=48,classifier_weight=1,geometry_weight=1)
    items=[]
    def add(name,profile='context',beta=5,geometry=.25,reverse=0,switch=0,width=12):
        items.append({'name':name,'profile':profile,'match_config':asdict(replace(base,beta_m=beta,geometry_weight=geometry,max_candidates=width)),'route_policy':asdict(RoutePolicy(reverse_penalty=reverse,edge_switch_penalty=switch))})
    add('v1_release_control','v1',2,1)
    add('context_release_decoder',beta=2,geometry=1)
    add('context_b5_g025')
    add('context_b5_g025_r1',reverse=1)
    add('context_b5_g025_r3',reverse=3)
    add('context_b5_g025_r8',reverse=8)
    add('context_b2_g025_r3',beta=2,reverse=3)
    add('context_b10_g025_r3',beta=10,reverse=3)
    add('v1_b5_g025_r3','v1',reverse=3)
    add('context_b5_g01_r3',geometry=.1,reverse=3)
    add('context_b5_g025_r3_s02',reverse=3,switch=.2)
    add('context_b5_g025_r3_k24',reverse=3,width=24)
    return items

def preregister():
    if PROTOCOL.exists():raise FileExistsError('new protocol exists; no rewriting')
    prior=json.loads(c.PROTOCOL_PATH.read_text());routes=[]
    for fold in prior['folds']:
        ids=sorted(fold['validation_trip_ids'],key=lambda t:(hashlib.sha256(f'route96:{c.SEED}:{t}'.encode()).hexdigest(),t))[:6]
        for trip in ids:routes.append({'trip_id':trip,'fold':fold['fold'],'train_trip_ids':fold['train_trip_ids'],'validation_trip_ids':fold['validation_trip_ids']})
    trials={}
    for record,path in old.verified_full_trials('hist_gradient_boosting'):
        if record['name'] in ('histgb_v1_control_full','histgb_context_large_full'):
            trials[record['profile']]={'path':path.relative_to(ROOT).as_posix(),'sha256':c.sha256_file(path),'private_output':record['private_output'],'folds':[{'fold':f['fold'],'sha256':f['private_model_sha256']} for f in record['folds']]}
    assert set(trials)=={'v1','context'}
    manifest={'schema':'route96-preregistered-v1','created_at_utc':c.utc_now(),'target_metric':'equal-trip exact-original-directed-route edgeF1; full reference, repeated traversals, failureszero','target_threshold':.96,'baseline_student_metric':.9513611360981356,'source_sha256':sourcepins(),'prior_model_protocol_sha256':c.sha256_file(c.PROTOCOL_PATH),'trials':trials,'selection_routes':routes,'menu':menu(),'primary_selection':['fewest_failed_routes','highest_mean_route_edge_f1','highest_mean_route_length_f1','lower_mean_seconds','earlier_menu_order'],'guardrails':'zero candidate failures; candidate lengthF1 no morethan0.002 below pairedcontrol; continuity1','evaluator_used':False,'student_used_for_tuning':False,'route_postprocessing':False,'status':'authorized_before_measurement','deviations':[],'historical_exposure':'new exploratory routeedgeprimary study; all80TRAINhistoricallyexposed; prior64/16confirmationnotblindnotreset','candidate_policy_note':'Retry12/24/48 recomputes emissions perpool. K24menu may change candidate-relativefeatures. Never crosspool NLLselection. No thinning.'}
    c.write_new_json(PROTOCOL,manifest);return manifest

def verified():
    p=json.loads(PROTOCOL.read_text())
    if p['source_sha256']!=sourcepins() or p['status']!='authorized_before_measurement':raise ValueError('frozen study implementation drift')
    if p['prior_model_protocol_sha256']!=c.sha256_file(c.PROTOCOL_PATH):raise ValueError('prior fitting protocol drift')
    for t in p['trials'].values():
        if c.sha256_file(ROOT/t['path'])!=t['sha256']:raise ValueError('trial drift')
        for fold in t['folds']:
            if c.sha256_file(ROOT/t['private_output']/f"fold_{fold['fold']}.joblib")!=fold['sha256']:raise ValueError('fold model drift')
    return p

def loadfold(protocol,profile,fold):
    trial=protocol['trials'][profile];f=next(x for x in trial['folds'] if x['fold']==fold['fold'])
    return load_accuracy_emission_model(ROOT/trial['private_output']/f"fold_{fold['fold']}.joblib",expected_sha256=f['sha256'],expected_fold_id=fold['fold'],expected_train_trip_ids=fold['train_trip_ids'],expected_validation_trip_ids=fold['validation_trip_ids'],expected_protocol_sha256=protocol['prior_model_protocol_sha256'])

def runtrip(trip_id):
    p=verified();fold=next((f for f in p['selection_routes'] if f['trip_id']==trip_id),None)
    if fold is None:raise ValueError('only preregistered tuning TRAIN route allowed')
    dest=EXP/'routes'/(trip_id+'.json')
    if dest.exists():raise FileExistsError('immutable route result exists')
    private=EXP/'private'/trip_id
    if private.exists() and any(private.iterdir()):raise FileExistsError('partial measured outputs exist; no implicit replay')
    private.mkdir(parents=True,exist_ok=True)
    paths={p.name:p for p in discover_trips(ROOT/'Student folder/train')};prefix=paths[trip_id];graph=load_graph(prefix);track=load_track(prefix.with_suffix('.track'))
    index=MemoIndex(CandidateIndex(graph,40));router=Router(graph,2048);transitions={};models={profile:MemoEmissions(loadfold(p,profile,fold)) for profile in p['trials']};records=[];results=[]
    for cfg in p['menu']:
        start=time.perf_counter();result=None
        try:
            with threadpool_limits(limits=2):result=RouteTargetMatcher(graph,MatchConfig(**cfg['match_config']),models[cfg['profile']],RoutePolicy(**cfg['route_policy']),shared_index=index,shared_router=router,transition_cache=transitions).match(track)
            if not np.array_equal(result.observation_indices,np.arange(len(track))) or not validate_route(graph,result.route)['valid']:raise ValueError('dense directedroute invariant failed')
            np.savez_compressed(private/(cfg['name']+'.npz'),route=result.route,matched_edges=result.matched_edges,matched_fractions=result.matched_fractions)
            record={'name':cfg['name'],'success':True,'observations':len(track),'elapsed_seconds':time.perf_counter()-start,'effective_max_candidates':result.diagnostics['effective_max_candidates'],'candidate_retry_attempts':result.diagnostics['candidate_retry_attempts'],'route_sha256':hashlib.sha256(np.asarray(result.route,dtype='<i8').tobytes()).hexdigest()}
        except MatchingError as exc:
            record={'name':cfg['name'],'success':False,'observations':len(track),'elapsed_seconds':time.perf_counter()-start,'failure':str(exc)}
        records.append(record);results.append(result);print(json.dumps({'event':'route96_train_prediction','trip_id':trip_id,**record}),flush=True)
    # Reference opened only after every decoder prediction is fixed.
    reference=load_route(prefix.with_suffix('.route'))
    for record,result in zip(records,results):
        record['metrics']=score_routes(graph,result.route,reference) if result is not None else {'edge_f1':0.,'length_f1':0.,'route_continuity':0.,'edit_similarity':0.}
    verified();out={'trip_id':trip_id,'fold':fold['fold'],'observations':len(track),'predictions':records,'protocol_sha256':c.sha256_file(PROTOCOL),'reference_loaded_after_all_predictions':True,'student_used':False,'evaluator_used':False,'input_sha256':{s:c.sha256_file(prefix.with_suffix(s)) for s in ('.nodes','.arcs','.track','.route')}};c.write_new_json(dest,out);return out

def select():
    p=verified();records=[]
    for fold in p['selection_routes']:
        path=EXP/'routes'/(fold['trip_id']+'.json');r=json.loads(path.read_text())
        if r['trip_id']!=fold['trip_id'] or r['fold']!=fold['fold'] or r['protocol_sha256']!=c.sha256_file(PROTOCOL) or r['reference_loaded_after_all_predictions'] is not True or [x['name'] for x in r['predictions']]!=[x['name'] for x in p['menu']]:raise ValueError('route evidence identity differs')
        records.append(r)
    table=[]
    for order,cfg in enumerate(p['menu']):
        values=[next(x for x in r['predictions'] if x['name']==cfg['name']) for r in records]
        table.append({'configuration':cfg,'order':order,'route_count':len(values),'edge_f1':float(np.mean([x['metrics']['edge_f1'] for x in values])),'length_f1':float(np.mean([x['metrics']['length_f1'] for x in values])),'route_continuity':float(np.mean([x['metrics']['route_continuity'] for x in values])),'mean_seconds':float(np.mean([x['elapsed_seconds'] for x in values])),'failures':sum(not x['success'] for x in values)})
    baseline=table[0];eligible=[x for x in table if not x['failures'] and x['route_continuity']==1 and x['length_f1']>=baseline['length_f1']-.002]
    if not eligible:raise ValueError('no candidate satisfies immutable route gates')
    winner=max(eligible,key=lambda x:(x['edge_f1'],x['length_f1'],-x['mean_seconds'],-x['order']))
    frozen={'created_at_utc':c.utc_now(),'schema':'route96-frozen-selection-v1','protocol_sha256':c.sha256_file(PROTOCOL),'selected':winner,'baseline':baseline,'all_configurations':table,'route_evidence':[{'trip_id':r['trip_id'],'path':(EXP/'routes'/(r['trip_id']+'.json')).relative_to(ROOT).as_posix(),'sha256':c.sha256_file(EXP/'routes'/(r['trip_id']+'.json'))} for r in records],'source_sha256':sourcepins(),'student_used_for_selection':False,'evaluator_used':False};c.write_new_json(EXP/'frozen_selection.json',frozen);return frozen

def main():
    a=argparse.ArgumentParser();a.add_argument('--preregister',action='store_true');a.add_argument('--trip');a.add_argument('--select',action='store_true');args=a.parse_args()
    if args.preregister:r=preregister()
    elif args.trip:r=runtrip(args.trip)
    elif args.select:r=select()
    else:a.error('choose mode')
    if not args.trip:print(json.dumps(r,indent=2))
if __name__=='__main__':main()
