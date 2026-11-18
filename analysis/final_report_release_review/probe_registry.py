from pathlib import Path
import importlib.util,json
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location("registry_probe",ROOT/"scripts/build_final_report.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def snap(label):
 print(label,m._registry_sha256());(Path(__file__).parent/(label+"_registry.json")).write_text(json.dumps({"inputs":m.INPUTS,"facts":m.FACTS,"figures":m.FIGURE_REGISTRY},sort_keys=True,indent=2))
snap("imported")
state=m.build_state("check");snap("built_state")
values=m.resolve_fact_values(state);snap("resolved")
text=m.TEMPLATE_PATH.read_text();md=m.expand_tables(m.resolve_tokens(text,values),state);guards=m.run_final_guards(state,values,md,text);snap("guarded")
print("persisted",m.strict_json(m.MANIFEST_PATH.read_text())["registry_sha256"])
