#!/usr/bin/env python3
"""Genuine existing-public-URL browser screenshots bound to accepted route hashes."""
from pathlib import Path
import hashlib,json,time
from playwright.sync_api import sync_playwright
from backend.geometry import predicted_geojson
from mapmatch.data import load_graph,load_route,load_track
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/target96/live';BASE='https://ml.d3va.dev'

def main():
    acceptance=json.loads((ROOT/'artifacts/target96/acceptance.json').read_text());manifest=json.loads((ROOT/'artifacts/target96/release_manifest.json').read_text());assert acceptance['accepted'] is True
    OUT.mkdir(parents=True,exist_ok=True);evidence={'url':BASE,'mocked':False,'request_body':{'trip_id':'00000009'},'model_hash':manifest['model_sha256'],'screenshots':{},'errors':[],'role':'actual public application after accepted release; student development not blind'}
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True);context=browser.new_context(viewport={'width':1440,'height':1080},device_scale_factor=1,service_workers='block');page=context.new_page();page.on('pageerror',lambda error:evidence['errors'].append(str(error)))
        response=page.goto(BASE,wait_until='networkidle');assert response.status==200
        page.wait_for_function("() => document.querySelectorAll('#trip-list input[name=trip-choice]').length === 10",timeout=30000)
        page.locator('#trip-list input[value="00000009"]').locator('..').click();page.wait_for_function("() => !document.getElementById('match-button').disabled")
        with page.expect_response(lambda r:r.url==BASE+'/datasets/development/match' and r.request.method=='POST',timeout=120000) as info:page.locator('#match-button').click()
        res=info.value;assert res.status==200;body=res.json();assert body['model']['sha256']==manifest['model_sha256'];assert res.request.post_data_json=={'trip_id':'00000009'}
        page.wait_for_function("() => !document.getElementById('match-button').disabled && document.getElementById('result-hash').textContent.length === 64",timeout=30000)
        prefix=ROOT/'Student folder/student_test/00000009/00000009';graph=load_graph(prefix);track=load_track(prefix.with_suffix('.track'));route=load_route(ROOT/'outputs/target96/candidate/00000009/00000009.route');expected=predicted_geojson(graph,route)
        def locate(obj):
            if isinstance(obj,dict):
                for key,val in obj.items():
                    if val==expected:return True
                    if isinstance(val,(list,dict)) and locate(val):return True
            elif isinstance(obj,list):return any(locate(x) for x in obj if isinstance(x,(list,dict)))
            return False
        assert locate(body),'public predicted geometry differs from frozen final route'
        evidence.update(http_status=res.status,request_id=body.get('request_id'),response_metrics=body['metrics'],timings=body['timings'],frozen_prediction_geometry_matches=True,trip_count=page.locator('#trip-list input[name="trip-choice"]').count())
        page.locator('#result-details').evaluate('el => el.open=true');assert page.locator('#result-hash').inner_text()==manifest['model_sha256']
        for name,selector in [('ui_route_map','#route-map'),('ui_controls','#trip-section'),('ui_scores','#score-section')]:
            loc=page.locator(selector)
            if loc.count():
                loc.scroll_into_view_if_needed();page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');path=OUT/(name+'.png');assert not path.exists();loc.screenshot(path=str(path));evidence['screenshots'][name]={'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        path=OUT/'ui_desktop.png';assert not path.exists();page.screenshot(path=str(path),full_page=True);evidence['screenshots']['ui_desktop']={'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()};browser.close()
    assert not evidence['errors'],evidence['errors'];(OUT/'browser_evidence.json').write_text(json.dumps(evidence,indent=2)+chr(10));print('Actual public UI screenshot/model/route match PASS')
if __name__=='__main__':main()
