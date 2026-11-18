#!/usr/bin/env python3
"""TRAIN-only edge-F1 arc-prior refinement; stage2+remaining whole-trip validation."""
from __future__ import annotations
import argparse,hashlib,json,time
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from mapmatch.candidates import CandidateIndex
from mapmatch.data import discover_trips,load_graph,load_track,load_route,validate_route
from mapmatch.matcher import MatchConfig,MatchingError
from mapmatch.target96 import MemoIndex,MemoEmissions
from mapmatch.target96_arc import ArcPolicy,ArcPriorMatcher,HopRouter
from mapmatch.metrics import score_routes
from scripts import accuracy_common as c,target96 as old
EXP=old.EXP/'precision';PROTOCOL=EXP/'PROTOCOL.json';ADVANCE=EXP/'finalists.json';FREEZE=EXP/'frozen_selection.json'
PATHS=['scripts/target96_precision.py','mapmatch/target96_arc.py','tests/test_target96_arc.py']
def pins():return {p:c.sha256_file(c.ROOT/p) for p in PATHS}
def register():
    oldp=old.verified();previous=json.loads((old.EXP/'frozen_selection.json').read_text());base=previous['selected']['configuration'];items=[]
    def add(name,arc,profile='context',width=24):
        cfg=dict(base['match_config']);cfg['max_candidates']=width
        items.append({'name':name,'profile':profile,'match_config':cfg,'arc_policy':asdict(ArcPolicy(appended_arc_penalty=arc,reverse_penalty=3,reverse_chord_limit_m=20))})
    add('stage1_best_control',0)
    for value in (.25,.5,1.,2.,5.):add('context_k24_arc'+str(value),value)
    add('v1_k24_arc1',1.,'v1');add('context_k12_arc1',1.,width=12)
    items.append({'name':'v1_release_control','profile':'v1','match_config':oldp['menu'][0]['match_config'],'arc_policy':asdict(ArcPolicy())})
    prior=json.loads(c.PROTOCOL_PATH.read_text());plan=[]
    for fold in prior['folds']:
        for trip in fold['validation_trip_ids']:plan.append({'trip_id':trip,**fold})
    assert len(plan)==64 and len({x['trip_id'] for x in plan})==64
    initial=[x['trip_id'] for x in oldp['selection_routes']]
    protocol={'schema':'route96-precision-stage2-v1','created_at_utc':c.utc_now(),'old_protocol_sha256':c.sha256_file(old.PROTOCOL),'previous_selection_sha256':c.sha256_file(old.EXP/'frozen_selection.json'),'source_sha256':pins(),'initial_trip_ids':initial,'all_routes':plan,'menu':items,'staged_selection':'Measure all9 on prespecified18 TRAIN OOF trips. Advance top2 excluding v1release plus v1release. Complete those3 only on remaining46. Final edgeF1 ranking includes all64; no duplicated initialcounts. Not blind. No student/evaluator feedback.','objective':['fewest_failed_routes','highest_equal_trip_edge_f1','highest_equal_trip_edge_precision','highest_length_f1','earlier_menu_order'],'gates':'No increased failures compared with pairedv1; continuity1 for every successful output; meanlengthF1≥pairedv1-0.002','student_used':False,'evaluator_used':False,'reason_from_TRAIN':'Stage1 precision dominated by many short extra traversals. Prior counts exact shortest-metre connector arcs in Viterbi only; no repair/deletion/reference edits.'};c.write_new_json(PROTOCOL,protocol);return protocol

def verified():
    old.verified();p=json.loads(PROTOCOL.read_text())
    if p['source_sha256']!=pins() or p['old_protocol_sha256']!=c.sha256_file(old.PROTOCOL) or p['previous_selection_sha256']!=c.sha256_file(old.EXP/'frozen_selection.json'):raise ValueError('precision protocol drift')
    return p

def configs(p,trip):
    if trip in p['initial_trip_ids']:return p['menu']
    a=json.loads(ADVANCE.read_text())
    if a['protocol_sha256']!=c.sha256_file(PROTOCOL):raise ValueError('advance freeze drift')
    return [x for x in p['menu'] if x['name'] in a['names']]

def runtrip(trip):
    p=verified();fold=next((x for x in p['all_routes'] if x['trip_id']==trip),None)
    if fold is None:raise ValueError('not frozen TRAINtrip')
    choices=configs(p,trip);dest=EXP/'routes'/(trip+'.json');private=old.EXP/'private'/'precision'/trip
    if dest.exists() or private.exists():raise FileExistsError('immutable route outputs exist')
    paths={x.name:x for x in discover_trips(c.ROOT/'Student folder/train')};prefix=paths[trip];graph=load_graph(prefix);track=load_track(prefix.with_suffix('.track'));private.mkdir(parents=True,exist_ok=False)
    index=MemoIndex(CandidateIndex(graph,40));router=HopRouter(graph,2048);transitions={};models={profile:MemoEmissions(old.loadfold(old.verified(),profile,fold)) for profile in {x['profile'] for x in choices}};predictions=[];results=[]
    for cfg in choices:
        start=time.perf_counter();result=None
        try:
            with threadpool_limits(limits=2):result=ArcPriorMatcher(graph,MatchConfig(**cfg['match_config']),models[cfg['profile']],ArcPolicy(**cfg['arc_policy']),shared_index=index,shared_router=router,transition_cache=transitions).match(track)
            if not np.array_equal(result.observation_indices,np.arange(len(track))) or not validate_route(graph,result.route)['valid']:raise ValueError('dense/connected invariant failed')
            np.savez_compressed(private/(cfg['name']+'.npz'),route=result.route,matched_edges=result.matched_edges,matched_fractions=result.matched_fractions)
            record={'name':cfg['name'],'success':True,'observations':len(track),'elapsed_seconds':time.perf_counter()-start,'effective_max_candidates':result.diagnostics['effective_max_candidates'],'candidate_retry_attempts':result.diagnostics['candidate_retry_attempts'],'route_sha256':hashlib.sha256(np.asarray(result.route,dtype='<i8').tobytes()).hexdigest()}
        except MatchingError as exc:record={'name':cfg['name'],'success':False,'observations':len(track),'elapsed_seconds':time.perf_counter()-start,'failure':str(exc)}
        predictions.append(record);results.append(result);print(json.dumps({'event':'route96_precision_prediction','trip_id':trip,**record}),flush=True)
    reference=load_route(prefix.with_suffix('.route'))
    for record,result in zip(predictions,results):record['metrics']=score_routes(graph,result.route,reference) if result is not None else {k:0. for k in ('edge_f1','edge_precision','edge_recall','length_f1','route_continuity','edit_similarity')}
    verified();out={'trip_id':trip,'fold':fold['fold'],'observations':len(track),'predictions':predictions,'reference_loaded_after_all_predictions':True,'protocol_sha256':c.sha256_file(PROTOCOL),'student_used':False,'evaluator_used':False,'input_sha256':{s:c.sha256_file(prefix.with_suffix(s)) for s in ('.nodes','.arcs','.track','.route')}};c.write_new_json(dest,out);return out

def aggregate_rows(p,trips,names):
    records=[]
    for trip in trips:
        r=json.loads((EXP/'routes'/(trip+'.json')).read_text())
        if r['trip_id']!=trip or r['protocol_sha256']!=c.sha256_file(PROTOCOL) or not r['reference_loaded_after_all_predictions']:raise ValueError('measured identity differs')
        records.append(r)
    table=[]
    for order,cfg in enumerate(p['menu']):
        if cfg['name'] not in names:continue
        values=[next(x for x in r['predictions'] if x['name']==cfg['name']) for r in records]
        row={'configuration':cfg,'order':order,'trips':len(values),'failures':sum(not x['success'] for x in values),'mean_seconds':float(np.mean([x['elapsed_seconds'] for x in values]))}
        for k in ('edge_f1','edge_precision','edge_recall','length_f1','route_continuity','edit_similarity'):row[k]=float(np.mean([x['metrics'][k] for x in values]))
        row['successful_connected']=all(x['metrics']['prediction_connected'] for x in values if x['success']);table.append(row)
    return records,table

def rank(x):return (-x['failures'],x['edge_f1'],x['edge_precision'],x['length_f1'],-x['order'])
def advance():
    p=verified();records,table=aggregate_rows(p,p['initial_trip_ids'],[x['name'] for x in p['menu']]);baseline=next(x for x in table if x['configuration']['name']=='v1_release_control')
    eligible=[x for x in table if x['configuration']['name']!='v1_release_control' and x['successful_connected'] and x['failures']<=baseline['failures'] and x['length_f1']>=baseline['length_f1']-.002]
    names=[x['configuration']['name'] for x in sorted(eligible,key=rank,reverse=True)[:2]]+['v1_release_control']
    if len(names)!=3:raise ValueError('two initial candidates did not pass gates')
    r={'created_at_utc':c.utc_now(),'protocol_sha256':c.sha256_file(PROTOCOL),'names':names,'initial_scores':table,'student_used':False,'evaluator_used':False};c.write_new_json(ADVANCE,r);return r

def freeze():
    p=verified();a=json.loads(ADVANCE.read_text());records,table=aggregate_rows(p,[x['trip_id'] for x in p['all_routes']],a['names']);baseline=next(x for x in table if x['configuration']['name']=='v1_release_control');eligible=[x for x in table if x['successful_connected'] and x['failures']<=baseline['failures'] and x['length_f1']>=baseline['length_f1']-.002]
    if not eligible:raise ValueError('no final choice passes frozen gates')
    winner=max(eligible,key=rank);r={'created_at_utc':c.utc_now(),'protocol_sha256':c.sha256_file(PROTOCOL),'advance_sha256':c.sha256_file(ADVANCE),'selected':winner,'baseline':baseline,'all64_scores':table,'source_sha256':pins(),'old_source_sha256':old.sourcepins(),'route_evidence':[{'trip_id':x['trip_id'],'path':(EXP/'routes'/(x['trip_id']+'.json')).relative_to(c.ROOT).as_posix(),'sha256':c.sha256_file(EXP/'routes'/(x['trip_id']+'.json'))} for x in records],'student_used_for_selection':False,'evaluator_used':False,'all64_oof_observations':sum(x['observations'] for x in records)};c.write_new_json(FREEZE,r);return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--trip');p.add_argument('--advance',action='store_true');p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.register:r=register()
    elif a.trip:r=runtrip(a.trip)
    elif a.advance:r=advance()
    elif a.freeze:r=freeze()
    else:p.error('choose mode')
    if not a.trip:print(json.dumps(r,indent=2))
if __name__=='__main__':main()
