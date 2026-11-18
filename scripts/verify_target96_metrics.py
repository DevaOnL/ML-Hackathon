#!/usr/bin/env python3
"""Independent scalar recomputation; never changes model, reference or prediction."""
from pathlib import Path
import hashlib,json
import numpy as np
from mapmatch.data import discover_trips,load_graph,load_route,load_track,validate_route
from mapmatch.metrics import score_routes
ROOT=Path(__file__).resolve().parents[1]
def main():
    s=json.loads((ROOT/'experiments/target96/student_comparison/summary.json').read_text());a=json.loads((ROOT/'artifacts/target96/acceptance.json').read_text());m=ROOT/'artifacts/target96/release_manifest.json';assert hashlib.sha256(m.read_bytes()).hexdigest()==a['release_manifest_sha256'];rows=[]
    for prefix in discover_trips(ROOT/'Student folder/student_test'):
        rec=json.loads((ROOT/'experiments/target96/student_comparison'/(prefix.name+'.json')).read_text());g=load_graph(prefix);prediction=load_route(ROOT/'outputs/target96/candidate'/prefix.name/(prefix.name+'.route'));track=load_track(prefix.with_suffix('.track'));assert len(track)==rec['observations'];assert validate_route(g,prediction)['valid'];assert hashlib.sha256(np.asarray(prediction,dtype='<i8').tobytes()).hexdigest()==rec['records']['candidate']['route_sha256'];r=score_routes(g,prediction,load_route(prefix.with_suffix('.route')))
        for k in ('edge_f1','length_f1','edit_similarity','route_continuity'):assert abs(r[k]-rec['records']['candidate']['metrics'][k])<1e-14
        rows.append(r)
    for k in ('edge_precision','edge_recall','edge_f1','length_f1','edit_similarity','route_continuity'):assert abs(float(np.mean([r[k] for r in rows]))-s['candidate'][k])<1e-14
    assert len(rows)==10 and s['observations']==19160 and s['candidate']['edge_f1']>=.96 and s['candidate']['failures']==0 and s['candidate']['route_continuity']==1
    result={'independent_recomputation':True,'trip_count':10,'observations':19160,'edge_f1':s['candidate']['edge_f1'],'all_metrics_exact_match':True,'model_retrained':False,'routes_modified':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};path=ROOT/'reports/target96/independent_metric_check.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+chr(10));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
