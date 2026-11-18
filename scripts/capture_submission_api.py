#!/usr/bin/env python3
"""Capture an actual HTTPS API response in a real terminal on the receiving host."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/target96/terminal'
CLASS='pathsnap-submission-response'
TITLE='PathSnap | actual HTTP response on receiving host'

def run(*args):return subprocess.check_output(args,text=True)

def display():
    OUT.mkdir(parents=True,exist_ok=True)
    base='https://ml.d3va.dev';headers={'User-Agent':'Mozilla/5.0','Content-Type':'application/json'}
    with urlopen(Request(base+'/health',headers=headers),timeout=20) as r:health_status=r.status;health=json.load(r)
    url=base+'/datasets/development/match'
    with urlopen(Request(url,data=b'{"trip_id":"00000009"}',headers=headers,method='POST'),timeout=50) as r:status=r.status;raw=r.read();body=json.loads(raw)
    summary={k:body[k] for k in ('trip_id','dataset_id','dataset_role','metrics_status','metrics','timings') if k in body}
    record={'captured_at_utc':datetime.now(timezone.utc).isoformat(),'url':url,'request':{'trip_id':'00000009'},'http_status':status,'health_http_status':health_status,'model_hash':health.get('model_hash'),'response_sha256':hashlib.sha256(raw).hexdigest(),'response_projection':summary,'projection_note':'Unmodified selected scalar fields from actual HTTPS JSON; geometry omitted from terminal only.','mocked':False,'host_role':'same host receiving existing API; separate physical laptop not claimed'}
    (OUT/'api_response.json').write_text(json.dumps(record,indent=2)+chr(10))
    print('PATHSNAP — ACTUAL API RESPONSE / receiving host terminal',flush=True)
    print('GET '+base+'/health   HTTP '+str(health_status)+'  '+str(health.get('status')),flush=True)
    print('Model: '+str(health.get('model_hash')),flush=True)
    print('POST '+url,flush=True);print('Request: {"trip_id":"00000009"}',flush=True)
    print('HTTP '+str(status)+' OK',flush=True)
    for k in ('trip_id','dataset_id','dataset_role','metrics_status'):
        if k in summary:print(k+': '+str(summary[k]),flush=True)
    metrics=summary.get('metrics',{})
    for k in ('route_edge_f1','route_length_f1','route_continuity'):print(k+': '+str(metrics.get(k)),flush=True)
    print('Timing: '+json.dumps(summary.get('timings',{})),flush=True)
    print('Actual JSON fields shown; route coordinates omitted for readability.',flush=True)
    (OUT/'ready.json').write_text(json.dumps({'pid':os.getpid()}))

def capture():
    OUT.mkdir(parents=True,exist_ok=True)
    monitor=next(m for m in json.loads(run('hyprctl','-j','monitors')) if m['focused'])
    previous=json.loads(run('hyprctl','-j','activewindow')).get('address')
    if monitor.get('specialWorkspace',{}).get('name'):raise RuntimeError('Refuse to disturb active special workspace')
    proc=subprocess.Popen(['alacritty','--class',CLASS,'--title',TITLE,'--hold','-o','window.opacity=1.0','-o','font.size=11.5','-o','colors.primary.background="#07111f"','-o','colors.primary.foreground="#e4edf6"','-e',sys.executable,'-B',str(Path(__file__).resolve()),'--display'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    address=None;shown=False
    try:
        deadline=time.monotonic()+65
        while time.monotonic()<deadline:
            matches=[c for c in json.loads(run('hyprctl','-j','clients')) if c.get('class')==CLASS and c.get('pid')==proc.pid]
            if matches and (OUT/'ready.json').exists():address=matches[0]['address'];break
            if proc.poll() is not None:raise RuntimeError('Terminal request failed')
            time.sleep(.15)
        if not address:raise RuntimeError('Actual response not ready')
        width=1480;height=580;x=60;y=100;window=json.dumps('address:'+address)
        run('hyprctl','dispatch','hl.dsp.window.move({workspace="special:pathsnap-submission", window='+window+', silent=true})')
        run('hyprctl','dispatch','hl.dsp.window.float({action="set", window='+window+'})')
        run('hyprctl','dispatch','hl.dsp.window.resize({window='+window+', x='+str(width)+', y='+str(height)+', exact=true})')
        run('hyprctl','dispatch','hl.dsp.window.move({window='+window+', x='+str(x)+', y='+str(y)+', exact=true})')
        run('hyprctl','dispatch','hl.dsp.workspace.toggle_special("pathsnap-submission")');shown=True
        run('hyprctl','dispatch','hl.dsp.focus({window='+window+'})')
        deadline=time.monotonic()+10;frame=None;ocr=''
        while time.monotonic()<deadline:
            c=json.loads(run('hyprctl','-j','activewindow'))
            if c.get('address')!=address or c.get('class')!=CLASS:raise RuntimeError('No screenshot of another window allowed')
            geometry=f"{c['at'][0]},{c['at'][1]} {c['size'][0]}x{c['size'][1]}";candidate=subprocess.check_output(['grim','-g',geometry,'-'])
            c=json.loads(run('hyprctl','-j','activewindow'))
            if c.get('address')!=address:raise RuntimeError('Focus changed; no screenshot saved')
            recognized=subprocess.run(['tesseract','stdin','stdout'],input=candidate,capture_output=True,check=True).stdout.decode()
            if '200' in recognized and '00000009' in recognized and 'ACTUAL API RESPONSE' in recognized and 'route_continuity' in recognized:frame=candidate;ocr=recognized;break
            time.sleep(.12)
        if frame is None:raise RuntimeError('No correct actual response frame found')
        (OUT/'api_response.png').write_bytes(frame);(OUT/'api_response_ocr.txt').write_text(ocr)
        (OUT/'capture.json').write_text(json.dumps({'mocked':False,'source':'actual Alacritty window running HTTPS request on API receiving host','image_sha256':hashlib.sha256(frame).hexdigest(),'response_record_sha256':hashlib.sha256((OUT/'api_response.json').read_bytes()).hexdigest(),'window_class':CLASS,'physical_two_laptop_demo_claimed':False},indent=2)+chr(10));print('Actual HTTP200 terminal response captured')
    finally:
        if shown:run('hyprctl','dispatch','hl.dsp.workspace.toggle_special("pathsnap-submission")')
        if address:subprocess.run(['hyprctl','dispatch','hl.dsp.window.close({window='+json.dumps('address:'+address)+'})'],capture_output=True)
        if proc.poll() is None:proc.terminate()
        proc.wait(timeout=5)
        if previous:subprocess.run(['hyprctl','dispatch','hl.dsp.focus({window='+json.dumps('address:'+previous)+'})'],capture_output=True)
if __name__=='__main__':display() if '--display' in sys.argv else capture()
