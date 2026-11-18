"""Independent cross-process registry stability probe; no evidence/model loading."""
from pathlib import Path
import json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[2]
child='from pathlib import Path; import importlib.util,json; p=Path("scripts/build_final_report.py"); s=importlib.util.spec_from_file_location("seed_registry",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(json.dumps({"registry_sha256":m._registry_sha256(),"req_resolved":next(f for f in m.FACTS if f["name"]=="req_resolved")}))'
results=[]
for seed in ("1","2","42"):
 env={**os.environ,"PYTHONHASHSEED":seed,"PYTHONDONTWRITEBYTECODE":"1"}
 completed=subprocess.run([sys.executable,"-B","-c",child],cwd=ROOT,env=env,text=True,capture_output=True,check=True)
 results.append({"seed":seed,**json.loads(completed.stdout)})
print(json.dumps({"passed":len({r["registry_sha256"] for r in results})==1,"process_count":3,"results":results},indent=2))
