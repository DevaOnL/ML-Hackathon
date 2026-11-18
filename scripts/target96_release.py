#!/usr/bin/env python3
"""Freeze exact artifacts, then one full student-development comparison. No retuning."""
from __future__ import annotations
import argparse,hashlib,json,shutil,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from mapmatch.data import discover_trips,load_graph,load_track,load_route,validate_route,write_route
from mapmatch.matcher import MapMatcher,MatchConfig,MatchingError
from mapmatch.target96_arc import ArcPriorMatcher,ArcPolicy,HopRouter
from mapmatch.target96 import MemoIndex
from mapmatch.candidates import CandidateIndex
from mapmatch.features import reference_observation_labels,CandidateFeatureExtractor
from mapmatch.learning import load_emission_model,candidate_classification_metrics,predict_positive_proba
from mapmatch.accuracy_features import transform_accuracy_features
from mapmatch.accuracy_artifacts import load_accuracy_deployment_model
from mapmatch.metrics import score_routes
from backend.emissions import AuditedEmissionModel
from scripts import accuracy_common as c,target96 as first,target96_precision as study
ART=c.ROOT/'artifacts/target96';OUT=first.EXP/'student_comparison';MANIFEST=ART/'release_manifest.json'
ORIGINAL_MODEL='c9cc8e9f36838f656a94d07ed69632979df9981ca94a9daf3b52fbce052e4b34'
ORIGINAL_CONFIG='10405aa6b6f370e41d4ddcb01207fd1e8f58900fd0395e541ee6bc78146168c5'
SOURCE_PATHS=['scripts/target96_release.py','mapmatch/target96_arc.py','mapmatch/target96.py','mapmatch/matcher.py','mapmatch/metrics.py','mapmatch/features.py','mapmatch/accuracy_features.py','mapmatch/accuracy_artifacts.py','mapmatch/accuracy_matcher.py','mapmatch/learning.py','backend/emissions.py']

def pins():return {p:c.sha256_file(c.ROOT/p) for p in SOURCE_PATHS}

def export():
    study.verified();f=json.loads(study.FREEZE.read_text());advance=json.loads(study.ADVANCE.read_text())
    if advance['protocol_sha256']!=c.sha256_file(study.PROTOCOL) or len(advance['names'])!=3 or len(set(advance['names']))!=3 or 'v1_release_control' not in advance['names'] or f['advance_sha256']!=c.sha256_file(study.ADVANCE) or {x['configuration']['name'] for x in f['all64_scores']}!=set(advance['names']):raise ValueError('exact frozen finalist protocol/denominator differs')
    choice=f['selected']['configuration'];ART.mkdir(parents=True,exist_ok=False)
    profile=choice['profile'];source=c.ROOT/('artifacts/accuracy_improvement/model.joblib' if profile=='context' else 'artifacts/final_five/model.joblib');metadata=c.ROOT/('artifacts/accuracy_improvement/model_metadata.json' if profile=='context' else 'artifacts/final_five/model_metadata.json')
    for name,path in [('model.joblib',source),('training_metadata.json',metadata)]:shutil.copyfile(path,ART/name)
    c.write_new_json(ART/'match_config.json',choice['match_config']);c.write_new_json(ART/'arc_policy.json',choice['arc_policy'])
    base=c.ROOT/'artifacts/final_five'
    if c.sha256_file(base/'model.joblib')!=ORIGINAL_MODEL or c.sha256_file(base/'match_config.json')!=ORIGINAL_CONFIG:raise ValueError('released baseline changed')
    manifest={'schema':'pathsnap-route96-frozen-release-v1','created_at_utc':c.utc_now(),'model_sha256':c.sha256_file(ART/'model.joblib'),'training_protocol_sha256':c.sha256_file(c.PROTOCOL_PATH),'route_protocol_sha256':c.sha256_file(study.PROTOCOL),'frozen_selection_sha256':c.sha256_file(study.FREEZE),'profile':profile,'configuration':choice,'files':{p.name:c.sha256_file(p) for p in ART.iterdir() if p.is_file()},'source_sha256':pins(),'baseline_model_sha256':ORIGINAL_MODEL,'baseline_config_sha256':ORIGINAL_CONFIG,'training_fit_reused':True,'model_changed':c.sha256_file(source)!=ORIGINAL_MODEL,'student_role':'historicallyexposeddevelopment,notblind','evaluator_role':'never used for training/selection','final_comparison_gate':'all10trips/all19160GPS;zero failures;continuity1;edgeF1>=0.96;lengthF1>=baseline-0.002','deployment_authorized':False,'retraining_or_selection_after_student_allowed':False,'baseline_route_reference':'reports/final_five/decoder/student_ablation.json','timing_note':'final fullroute times independent uncached requests; TRAIN shared-cache searchtime is not a production benchmark'}
    if profile=='context':
        dataset=c.load_full_train_dataset();X=transform_accuracy_features(dataset.X,dataset.offsets,profile=profile)
        model=load_accuracy_deployment_model(ART/'model.joblib',expected_sha256=manifest['model_sha256'],expected_protocol_sha256=manifest['training_protocol_sha256'])
        with threadpool_limits(limits=2):raw=predict_positive_proba(model.estimator,X)
        metrics=candidate_classification_metrics(dataset.y,raw,dataset.offsets,dataset.observation_trip_ids,dataset.reference_edges,dataset.candidate_edges)
        c.write_new_json(ART/'training_metrics.json',{'role':'all80 TRAIN resubstitution, K12 candidatecache, notOOF','metrics':metrics,'model_sha256':manifest['model_sha256'],'source_cache_sha256':c.TRAIN_CACHE_SHA256})
        manifest['files']['training_metrics.json']=c.sha256_file(ART/'training_metrics.json')
    c.write_new_json(MANIFEST,manifest);print(json.dumps({'event':'route96_frozen_artifacts','profile':profile,'model_sha256':manifest['model_sha256'],'configuration':choice['name']}));return manifest

def verify():
    study.verified();m=json.loads(MANIFEST.read_text());f=json.loads(study.FREEZE.read_text())
    if m['frozen_selection_sha256']!=c.sha256_file(study.FREEZE) or m['route_protocol_sha256']!=c.sha256_file(study.PROTOCOL) or m['source_sha256']!=pins() or m['configuration']!=f['selected']['configuration']:raise ValueError('frozen final choice/source drift')
    for name,sha in m['files'].items():
        if c.sha256_file(ART/name)!=sha:raise ValueError('frozen final artifact drift')
    if c.sha256_file(c.ROOT/'artifacts/final_five/model.joblib')!=ORIGINAL_MODEL or c.sha256_file(c.ROOT/'artifacts/final_five/match_config.json')!=ORIGINAL_CONFIG:raise ValueError('baseline changed')
    return m

def load_candidate(m):
    if m['profile']=='context':return load_accuracy_deployment_model(ART/'model.joblib',expected_sha256=m['model_sha256'],expected_protocol_sha256=m['training_protocol_sha256'])
    if m['profile']=='v1':return load_emission_model(ART/'model.joblib')
    raise ValueError('unfrozen modelprofile')

class FullAudit(AuditedEmissionModel):
    def __init__(self,model):super().__init__(model);self.candidates={}
    def _record(self,graph,track,index,candidates,probabilities):
        p=super()._record(graph,track,index,candidates,probabilities);self.candidates[int(index)]=candidates;return p

def observation_metrics(graph,track,labels,result,audit,model,profile,private):
    indices=np.arange(len(track));groups=[audit.candidates[int(i)] for i in indices]
    X,offsets=CandidateFeatureExtractor(graph).transform(track,indices,groups)
    values=transform_accuracy_features(X,offsets,profile=profile) if profile!='v1' else X
    start=time.perf_counter()
    with threadpool_limits(limits=2):raw=predict_positive_proba(model.estimator,values)
    probability_seconds=time.perf_counter()-start
    edges=np.concatenate([g.edges for g in groups]);truth=np.repeat(labels.edges,np.diff(offsets));y=(edges==truth).astype(np.int8)
    metrics=candidate_classification_metrics(y,raw,offsets,np.full(len(track),'trip'),labels.edges,edges)
    q=np.concatenate([audit.records[int(i)].probabilities for i in indices]);raw_top=np.array([g.edges[int(np.argmax(raw[lo:hi]))] for g,lo,hi in zip(groups,offsets[:-1],offsets[1:])]);q_top=np.array([audit.records[int(i)].edges[int(np.argmax(audit.records[int(i)].probabilities))] for i in indices])
    if not np.array_equal(raw_top,q_top):raise ValueError('rawmodel/auditednormalizedargmax mismatch')
    nll=[]
    for i,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
        where=np.flatnonzero(edges[lo:hi]==labels.edges[i]);nll.append(-np.log(max(float(q[lo+where[0]]) if len(where) else 1e-7,1e-7)))
    raw_route=raw_top[np.r_[True,raw_top[1:]!=raw_top[:-1]]]
    proxy_pred=graph.canonical_edges[result.matched_edges];proxy_truth=labels.edges
    metrics.update({'observation_edge_macro_f1':float(f1_score(proxy_truth,raw_top,average='macro',zero_division=0)),'hmm_nonimproving_observation':bool(np.mean(proxy_pred==proxy_truth)<np.mean(raw_top==proxy_truth)),'observation_nll':float(np.mean(nll)),'probability_call_seconds':probability_seconds,'decoded_proxy_accuracy':float(np.mean(proxy_pred==proxy_truth)),'decoded_proxy_macro_f1':float(f1_score(proxy_truth,proxy_pred,average='macro',zero_division=0)),'raw_adjacency':float(np.mean((raw_top[1:]==raw_top[:-1])|(graph.heads[raw_top[:-1]]==graph.tails[raw_top[1:]]))) if len(raw_top)>1 else 1.})
    np.savez_compressed(private,raw=raw,y=y,offsets=offsets,edges=edges,q=q,truth_edges=labels.edges,raw_top=raw_top,decoded_edges=proxy_pred,nll=np.asarray(nll))
    return metrics,raw_route

def runtrip(trip):
    m=verify();prefixes={p.name:p for p in discover_trips(c.ROOT/'Student folder/student_test')}
    if trip not in prefixes:raise ValueError('studentdevelopmenttrip only')
    destination=OUT/(trip+'.json');private=first.EXP/'private'/'final_student'/trip
    if destination.exists() or private.exists():raise FileExistsError('final comparison not repeatable')
    prefix=prefixes[trip];graph=load_graph(prefix);track=load_track(prefix.with_suffix('.track'));private.mkdir(parents=True,exist_ok=False)
    original=load_emission_model(c.ROOT/'artifacts/final_five/model.joblib');candidate=load_candidate(m);cfg=MatchConfig(**m['configuration']['match_config']);policy=ArcPolicy(**m['configuration']['arc_policy']);results={};records={};audits={};models={'baseline':original,'candidate':candidate}
    for label in ('candidate','baseline'):
        model=models[label];audit=FullAudit(model);audits[label]=audit;start=time.perf_counter();result=None
        try:
            with threadpool_limits(limits=2):result=(ArcPriorMatcher(graph,cfg,audit,policy,shared_index=MemoIndex(CandidateIndex(graph,cfg.sample_spacing_m)),shared_router=HopRouter(graph,cfg.cache_size)) if label=='candidate' else MapMatcher(graph,MatchConfig(**json.loads((c.ROOT/'artifacts/final_five/match_config.json').read_text())),audit)).match(track)
            if not np.array_equal(result.observation_indices,np.arange(len(track))) or not validate_route(graph,result.route)['valid'] or len(audit.records)!=len(track):raise ValueError('dense/connected/learned invariant')
            np.savez_compressed(private/(label+'.npz'),route=result.route,matched_edges=result.matched_edges,matched_fractions=result.matched_fractions,observation_indices=result.observation_indices)
            records[label]={'success':True,'elapsed_seconds':time.perf_counter()-start,'observations':len(track),'effective_max_candidates':result.diagnostics['effective_max_candidates'],'candidate_retry_attempts':result.diagnostics['candidate_retry_attempts'],'route_sha256':hashlib.sha256(np.asarray(result.route,dtype='<i8').tobytes()).hexdigest()};route_path=c.ROOT/'outputs/target96'/label/trip/(trip+'.route');write_route(route_path,result.route);records[label]['route_file_sha256']=c.sha256_file(route_path)
        except MatchingError as exc:records[label]={'success':False,'elapsed_seconds':time.perf_counter()-start,'observations':len(track),'failure':str(exc)}
        results[label]=result
    # Every route prediction is fixed before any student reference is available.
    reference=load_route(prefix.with_suffix('.route'));labels=reference_observation_labels(graph,track,reference)
    for label in ('baseline','candidate'):
        result=results[label]
        if result is None:
            records[label]['metrics']={k:0. for k in ('edge_precision','edge_recall','edge_f1','length_f1','edit_similarity','route_continuity')};records[label]['observation_metrics']=None;continue
        records[label]['metrics']=score_routes(graph,result.route,reference)
        obs,raw_route=observation_metrics(graph,track,labels,result,audits[label],models[label],m['profile'] if label=='candidate' else 'v1',private/(label+'_observations.npz'));records[label]['observation_metrics']=obs;records[label]['classifier_only_route']=score_routes(graph,raw_route,reference)
    verify();out={'created_at_utc':c.utc_now(),'trip_id':trip,'observations':len(track),'records':records,'release_manifest_sha256':c.sha256_file(MANIFEST),'reference_loaded_after_both_predictions':True,'source_sha256':c.sha256_file(__file__),'student_role':'historically exposed development; no blindtestclaim','evaluator_used':False,'input_sha256':{s:c.sha256_file(prefix.with_suffix(s)) for s in ('.nodes','.arcs','.track','.route')}};c.write_new_json(destination,out);print(json.dumps({'event':'route96_final_development','trip_id':trip,'baseline_edgeF1':records['baseline']['metrics']['edge_f1'],'candidate_edgeF1':records['candidate']['metrics']['edge_f1']}),flush=True);return out

def aggregate():
    m=verify();ids=sorted(p.name for p in discover_trips(c.ROOT/'Student folder/student_test'));records=[json.loads((OUT/(t+'.json')).read_text()) for t in ids]
    if len(ids)!=10 or len(set(ids))!=10 or sum(r['observations'] for r in records)!=19160:raise ValueError('same-ten dense comparison denominator differs')
    for trip,r in zip(ids,records):
        if r['trip_id']!=trip or r['release_manifest_sha256']!=c.sha256_file(MANIFEST) or not r['reference_loaded_after_both_predictions'] or r['source_sha256']!=c.sha256_file(__file__):raise ValueError('final evidence binding differs')
    result={'created_at_utc':c.utc_now(),'trips':10,'observations':19160,'role':'historicallyexposedstudentdevelopmentnotblind','release_manifest_sha256':c.sha256_file(MANIFEST),'selection_frozen_before_student':True,'evaluator_used':False,'reference_after_all_predictions':True}
    for label in ('baseline','candidate'):
        values=[r['records'][label] for r in records];row={k:float(np.mean([v['metrics'][k] for v in values])) for k in ('edge_precision','edge_recall','edge_f1','length_f1','edit_similarity','route_continuity')};row.update({'failures':sum(not v['success'] for v in values),'mean_seconds':float(np.mean([v['elapsed_seconds'] for v in values]))})
        if not row['failures']:
            row['observation_trip_macro']={k:float(np.mean([v['observation_metrics'][k] for v in values])) for k in ('top1_accuracy','observation_edge_macro_f1','observation_nll','decoded_proxy_accuracy','decoded_proxy_macro_f1','raw_adjacency','macro_f1','log_loss','probability_call_seconds')}
            arrays=[np.load(first.EXP/'private'/'final_student'/trip/(label+'_observations.npz')) for trip in ids];raw=np.concatenate([a['raw'] for a in arrays]);y=np.concatenate([a['y'] for a in arrays]);offsets=np.r_[0,np.cumsum(np.concatenate([np.diff(a['offsets']) for a in arrays]))];edges=np.concatenate([a['edges'] for a in arrays]);truth=np.concatenate([a['truth_edges'] for a in arrays]);tripids=np.concatenate([np.full(len(a['truth_edges']),trip) for trip,a in zip(ids,arrays)])
            pooled=candidate_classification_metrics(y,raw,offsets,tripids,truth,edges);actual=[(trip,int(e)) for trip,a in zip(ids,arrays) for e in a['truth_edges']];pred=[(trip,int(e)) for trip,a in zip(ids,arrays) for e in a['raw_top']];decoded=[(trip,int(e)) for trip,a in zip(ids,arrays) for e in a['decoded_edges']];classes={x:i for i,x in enumerate(sorted(set(actual+pred+decoded)))};ay=[classes[x] for x in actual];py=[classes[x] for x in pred];dy=[classes[x] for x in decoded]
            pooled['observation_edge_macro_f1']=float(f1_score(ay,py,average='macro',zero_division=0));pooled['decoded_edge_macro_f1']=float(f1_score(ay,dy,average='macro',zero_division=0));pooled['observation_nll']=float(np.mean(np.concatenate([a['nll'] for a in arrays])));pooled['decoded_proxy_accuracy']=float(np.mean(np.asarray(ay)==dy));pooled['probability_call_total_seconds']=sum(v['observation_metrics']['probability_call_seconds'] for v in values);row['observation_pooled']=pooled
            selected=np.concatenate([a['raw_top'] for a in arrays]);row['observation_pooled']['candidate_count_min']=int(min(np.min(np.diff(a['offsets'])) for a in arrays));row['observation_pooled']['candidate_count_max']=int(max(np.max(np.diff(a['offsets'])) for a in arrays));row['observation_pooled']['raw_adjacency_trip_macro']=float(np.mean([v['observation_metrics']['raw_adjacency'] for v in values]))
            row['classifier_only_route']={k:float(np.mean([v['classifier_only_route'][k] for v in values])) for k in ('edge_f1','length_f1','edit_similarity','route_continuity')}
        result[label]=row
    result['deltas']={k:result['candidate'][k]-result['baseline'][k] for k in ('edge_precision','edge_recall','edge_f1','length_f1','edit_similarity','route_continuity','failures')};b=result['baseline'];a=result['candidate'];passed=a['edge_f1']>=.96 and not a['failures'] and a['route_continuity']==1 and a['length_f1']>=b['length_f1']-.002
    result['target_met']=bool(passed);result['target_metric']='equal-trip original-directed-route multiset edgeF1';result['target_threshold']=.96;result['trip_bindings']=[{'trip_id':trip,'path':(OUT/(trip+'.json')).relative_to(c.ROOT).as_posix(),'sha256':c.sha256_file(OUT/(trip+'.json'))} for trip in ids];c.write_new_json(OUT/'summary.json',result)
    c.write_new_json(ART/'acceptance.json',{'created_at_utc':c.utc_now(),'accepted':bool(passed),'target_met':bool(passed),'observed_edge_f1':a['edge_f1'],'baseline_edge_f1':b['edge_f1'],'failed_trips':a['failures'],'route_continuity':a['route_continuity'],'mean_length_f1':a['length_f1'],'release_manifest_sha256':c.sha256_file(MANIFEST),'comparison_sha256':c.sha256_file(OUT/'summary.json'),'not_official_or_blind':True,'production_switched':False});return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true');p.add_argument('--trip');p.add_argument('--aggregate',action='store_true');a=p.parse_args()
    if a.export:r=export()
    elif a.trip:r=runtrip(a.trip)
    elif a.aggregate:r=aggregate()
    else:p.error('choose mode')
    if not a.trip:print(json.dumps(r,indent=2))
if __name__=='__main__':main()
