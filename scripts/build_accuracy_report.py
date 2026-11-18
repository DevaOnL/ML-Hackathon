#!/usr/bin/env python3
"""Build a measured accuracy addendum without modifying the accepted final report."""
from __future__ import annotations
import csv
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether
from scripts import accuracy_common as c,accuracy_confirmation as confirmation

ROOT=c.ROOT
OUTPUT=ROOT/"reports/accuracy_improvement"


def load(path):return json.loads(Path(path).read_text())


def pct(v):return f"{100*v:.3f}%"


def pp(v):return f"{100*v:+.3f} pp"


def assemble():
    frozen=confirmation.verify_frozen();family_results={f:load(confirmation.CONFIRMATION/(f+".json")) for f in c.FAMILIES};routes=load(confirmation.CONFIRMATION/"route_summary.json");student=load(c.EXPERIMENT_ROOT/"student_comparison/summary.json");decision=load(c.EXPERIMENT_ROOT/"decoder/selection.json");tests=load(OUTPUT/"test_results.json");release=load(OUTPUT/"release_decision.json")
    freeze_sha=c.sha256_file(confirmation.FREEZE)
    if any(r["frozen_selection_sha256"]!=freeze_sha for r in family_results.values()) or routes["frozen_selection_sha256"]!=freeze_sha or student["frozen_selection_sha256"]!=freeze_sha:raise ValueError("report mixes experiment freezes")
    rows=[]
    for family in c.FAMILIES:
        f=family_results[family];a=f["records"]["baseline"]["metrics"];b=f["records"]["candidate"]["metrics"];t=frozen["families"][family];tb=t["baseline"]["tuning_metrics"];tn=t["candidate"]["tuning_metrics"]
        row={"family":family,"selected_run":t["candidate"]["run_name"],"feature_profile":t["candidate"]["profile"],"tuning_baseline_top1":tb["trip_macro_top1_accuracy"],"tuning_candidate_top1":tn["trip_macro_top1_accuracy"],"confirmation_baseline_top1":a["trip_macro_top1_accuracy"],"confirmation_candidate_top1":b["trip_macro_top1_accuracy"],"confirmation_delta_top1":b["trip_macro_top1_accuracy"]-a["trip_macro_top1_accuracy"],"confirmation_baseline_binary_macro_f1":a["macro_f1"],"confirmation_candidate_binary_macro_f1":b["macro_f1"],"confirmation_baseline_binary_log_loss":a["log_loss"],"confirmation_candidate_binary_log_loss":b["log_loss"],"confirmation_baseline_edge_macro_f1":a["edge_macro_f1"],"confirmation_candidate_edge_macro_f1":b["edge_macro_f1"],"paired_trip_bootstrap_interval":f["paired_trip_top1"]["trip_bootstrap_95_percentile_interval"],"improved_trips":f["paired_trip_top1"]["improved_trips"],"decreased_trips":f["paired_trip_top1"]["decreased_trips"]};rows.append(row)
    summary={"created_at_utc":datetime.now(timezone.utc).isoformat(),"frozen_selection_sha256":freeze_sha,"families":rows,"route_confirmation":routes,"student_development_comparison":student,"decoder_tuning":decision,"test_results":tests,"release_decision":release,"limitations":["Proxy GPS edge labels are geometry-and-motion route projections, not independent timestamp ground truth; geometry-derived target favors geometric predictors.","All80 TRAIN trips were historically used by earlier releases; prospective16 internal confirmation is not historicallyunseen or an official blind test.","Student is historically used development data. Results are final comparisons after freeze and did not feed new tuning.","Evaluator was already scored historically and was not accessed for this new tuning or selection.","Tuning estimates reuse validation for selection; internal confirmation and paired trip-resampling intervals do not establish independent official-test generalization.","Individual metric regressions, failures and negative configurations remain visible; no blanket all-accuracies claim."]}
    OUTPUT.mkdir(parents=True,exist_ok=True);c.write_new_json(OUTPUT/"accuracy_results.json",summary)
    with (OUTPUT/"model_comparison.csv").open("x",newline="") as s:
        w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return summary


def build_markdown(d):
    text=["# PathSnap accuracy-improvement addendum","",f"Measured experiment frozen at {d['frozen_selection_sha256']}. The earlier accepted technical report is preserved, not overwritten.","","## Paired internal TRAIN16 classifier confirmation","","Both methods refitted on identical TRAIN64. Exactly 350 deterministic observations per reserved trip. Earlier releases historically used these trips; not a blind official test.","","| Model | Control top-1 | Candidate top-1 | Change | Binary macro F1 control / new | Binary LL control / new |","|---|---:|---:|---:|---:|---:|"]
    for r in d["families"]:text.append(f"| {r['family']} | {pct(r['confirmation_baseline_top1'])} | {pct(r['confirmation_candidate_top1'])} | {pp(r['confirmation_delta_top1'])} | {pct(r['confirmation_baseline_binary_macro_f1'])} / {pct(r['confirmation_candidate_binary_macro_f1'])} | {r['confirmation_baseline_binary_log_loss']:.6f} / {r['confirmation_candidate_binary_log_loss']:.6f} |")
    for heading,key in (("Dense internal TRAIN16 route confirmation","route_confirmation"),("Final student-development comparison","student_development_comparison")):
        r=d[key];text.extend(["",f"## {heading}","","| Metric | Baseline | Candidate | Change |","|---|---:|---:|---:|"])
        for metric in ("length_f1","edge_f1","proxy_edge_accuracy","proxy_edge_macro_f1","route_continuity"):
            a=r["baseline"][metric];b=r["candidate"][metric];text.append(f"| {metric} | {pct(a)} | {pct(b)} | {pp(b-a)} |")
    text.extend(["","## Validation and release","",f"Tests: {json.dumps(d['test_results'],sort_keys=True)}","",f"Release decision: {json.dumps(d['release_decision'],sort_keys=True)}","","## Limits","",*["- "+l for l in d["limitations"]],"","## Reproduction","","1. Use the pinned private authorized TRAIN cache and `experiments/accuracy_improvement/PROTOCOL.json`.","2. Run `python -B -m scripts.accuracy_trees --family FAMILY --stage screen`, then `--stage full`. Existing results are never overwritten; verified identical completed trials may resume.","3. Execute the six authorized OOF routes with `scripts.accuracy_route_validation`; attest original source/model evidence before selecting the decoder.","4. Freeze all families and decoder with `scripts.accuracy_confirmation --freeze`, then run eachfamily once and all16 dense routes via `scripts.accuracy_route_confirmation`.","5. Export the accepted candidate via `scripts.accuracy_export`, perform the final named student comparison and run the complete regression/release integrity gates.","","[Protocol](<../../experiments/accuracy_improvement/PROTOCOL.json>) · [Frozen selection](<../../experiments/accuracy_improvement/confirmation/frozen_selection.json>) · [Full measured results](<accuracy_results.json>) · [Comparison CSV](<model_comparison.csv>)",""])
    (OUTPUT/"ACCURACY_REPORT.md").write_text("\n".join(text))


def build_pdf(d):
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name="SmallBody",parent=styles["BodyText"],fontName="Helvetica",fontSize=9.5,leading=13,spaceAfter=7));styles.add(ParagraphStyle(name="Cell",fontName="Helvetica",fontSize=8.2,leading=10.7));P=lambda s:Paragraph(html.escape(str(s)),styles["SmallBody"]);C=lambda s:Paragraph(html.escape(str(s)),styles["Cell"]);story=[]
    def title(s):story.append(Paragraph(s,styles["Heading1"]));story.append(Spacer(1,3*mm))
    def table(rows,widths):
        t=Table([[C(x) for x in row] for row in rows],colWidths=widths,repeatRows=1,hAlign="LEFT");t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e6e9e7")),("VALIGN",(0,0),(-1,-1),"TOP"),("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor("#74857a")),("LINEBELOW",(0,1),(-1,-1),.25,colors.HexColor("#d2d8d4")),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]));story.append(t);story.append(Spacer(1,4*mm))
    title("PathSnap / Accuracy-improvement addendum");story.append(P("Genuine learned-model and dense-route experiments. Original accepted report and released baseline preserved. Exact outcomes and declines are reported separately; these are not organizer-certified scores."));story.append(P("Five prescribed families: Logistic Regression, distance-weighted KNN, genuine bootstrap Bagging, Gaussian Naive Bayes and HistGradientBoosting. Six fixed configurations per family were screened. Four families completed top2 plus control full-CV; Bagging retained its leading finalist plus control after the user requested an expedited finish, before confirmation scoring."));title("1. Comparison design and leakage controls");story.append(P("A seed-and-trip SHA256 partition prospectively reserved16 of80 TRAIN trips for a single comparison in this iteration. All80 were historically used by earlier releases, so this is internal confirmation, not unseen or blind test data. Both candidate and control fit on the same remaining64 with identical observation samples and fitting weights."));story.append(P("Stateless geometry/context transforms are opt-in. Scaling, skew correction and correlation pruning learn only from fitting rows. Candidate pools and proxy labels remain unchanged. Float64 model probabilities must be finite, bounded and sum toone within1e-10 before the unchanged1e-7 emission clip. Candidate misses stay in all-observation denominators."));story.append(P("The primary classification objective is unconditional equal-trip observation top1. Binary candidate macroF1/logloss and graph-qualified observation edgeF1 are separate, noninterchangeable measures. The labels are route-projection proxies rather than independent per-GPS truth."));story.append(P("Frozen selection SHA256: "+d["frozen_selection_sha256"]));story.append(PageBreak())
    title("2. Paired internal classifier confirmation");rows=[["Family","v1 control top1","New top1","Delta","New profile"]]
    for r in d["families"]:rows.append([r["family"].replace("_"," "),pct(r["confirmation_baseline_top1"]),pct(r["confirmation_candidate_top1"]),pp(r["confirmation_delta_top1"]),r["feature_profile"]])
    table(rows,[50*mm,29*mm,28*mm,28*mm,30*mm]);story.append(P("All16 trips contribute350 unchanged deterministic observations each. The same newly refitted control is the paired denominator; historical80-trip CV is not substituted. Higher top1 does not imply every secondary metric improves."));rows=[["Family","Binary macroF1 old / new","Binary log loss old / new"]]
    for r in d["families"]:rows.append([r["family"].replace("_"," "),pct(r["confirmation_baseline_binary_macro_f1"])+" / "+pct(r["confirmation_candidate_binary_macro_f1"]),f"{r['confirmation_baseline_binary_log_loss']:.5f} / {r['confirmation_candidate_binary_log_loss']:.5f}"])
    table(rows,[50*mm,58*mm,57*mm]);story.append(PageBreak())
    title("3. Per-trip uncertainty and tuning context");rows=[["Family","Tuning64 old / new top1","Reserved16 delta 95% trip-bootstrap interval","Trips + / -"]]
    for r in d["families"]:
        lo,hi=r["paired_trip_bootstrap_interval"];rows.append([r["family"].replace("_"," "),pct(r["tuning_baseline_top1"])+" / "+pct(r["tuning_candidate_top1"]),pp(lo)+" to "+pp(hi),f"{r['improved_trips']} / {r['decreased_trips']}"])
    table(rows,[42*mm,43*mm,59*mm,21*mm]);story.append(P("Intervals resample whole trips10000 times with seed20261118. They describe variation within the retrospectively exposed internal population, not a guarantee of independent official-test generalization. Tuning64 values are exploratory selection estimates. Failed/worse screen configurations remain in the experiment records."));story.append(P("Selected candidates: "+"; ".join(r["selected_run"] for r in d["families"])));story.append(PageBreak())
    title("4. Dense directed-route confirmation");story.append(P("The decoder menu was chosen on six predetermined tuning64 routes using their own fold-qualified estimators. Full raw tracks were scored for every GPS row; the350-observation classifier archives were not misrepresented as dense OOF emissions. Candidate policy K12/radius80 with empty-only fallback stayed fixed. Optional transition penalties never edited routes or deleted cycles after prediction."));rows=[["Metric","Internal16 baseline","Internal16 new","Delta"]];r=d["route_confirmation"]
    for k in ("length_f1","edge_f1","proxy_edge_accuracy","proxy_edge_macro_f1","route_continuity"):rows.append([k.replace("_"," "),pct(r["baseline"][k]),pct(r["candidate"][k]),pp(r["candidate"][k]-r["baseline"][k])])
    table(rows,[55*mm,37*mm,37*mm,36*mm]);story.append(P("The candidate uses its best OOF decoder, while the original-feature control receives its own strongest configuration from the same eight-choice/six-route budget. Both emissions are fitted on the same64. Failures contributezero and every reserved route remains in the denominator. Internal confirmation never chooses a replacement configuration."));story.append(PageBreak())
    title("5. Final student-development comparison");story.append(P("Only after model/feature/decoder freeze and internal confirmation, the candidate was refitted on all80 TRAIN trips and compared with the immutable released all80 model. The ten student trips are historically used development data, not a new blind test. Both predictions finish before a trip reference is opened. No result feeds further tuning. The already-scored evaluator was not used for selection."));rows=[["Metric","Released baseline","Frozen candidate","Delta"]];r=d["student_development_comparison"]
    for k in ("length_f1","edge_f1","proxy_edge_accuracy","proxy_edge_macro_f1","route_continuity"):rows.append([k.replace("_"," "),pct(r["baseline"][k]),pct(r["candidate"][k]),pp(r["candidate"][k]-r["baseline"][k])])
    table(rows,[55*mm,37*mm,37*mm,36*mm]);story.append(P("Failed trips, baseline / new: "+f"{r['baseline'].get('failures',0)} / {r['candidate'].get('failures',0)}. All failures remain in the ten-trip denominator as zero scores. The new fixed-K12 decoder was rejected; the live original retains its 12-to-48 retry policy."));story.append(P("Mean seconds/trip, baseline / new: "+f"{r['baseline']['elapsed_seconds']:.3f} / {r['candidate']['elapsed_seconds']:.3f}"));story.append(PageBreak())
    title("6. Verification, release and remaining limits");t=d["test_results"];story.append(P(f"Regression evidence: {t['python_tests']} Python + {t['hosting_tests']} hosting + {t['node_tests']} Node checks = {t['total_tests']} passing; static smoke {t['static_smoke']}."));release=d["release_decision"];story.append(P("Release accepted: "+str(release["accepted"])+". "+release["reason"]));story.append(P("Exact test logs, all family delta records, model/source hashes and route population bindings are retained in the accompanying machine-readable evidence."))
    for l in d["limitations"]:story.append(P("- "+l))
    story.append(P("Reproduction: see the separately delivered Markdown/JSON/CSV results, pinned protocol, complete fixed menu, source hashes, immutable trial records and route execution attestation. Original private graphs/GPS/references and row-level probability archives are not public deliverables. Automated Sol6.1MAX review is not human or organizer approval."))
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor("#b8c1ba"));canvas.line(22*mm,18*mm,187*mm,18*mm);canvas.setFont("Helvetica",8);canvas.drawString(22*mm,13*mm,"PathSnap / Measured accuracy addendum / internal and development evidence");canvas.drawRightString(187*mm,13*mm,str(doc.page));canvas.restoreState()
    path=OUTPUT/"PathSnap_Accuracy_Improvement_Addendum.pdf"
    if path.exists():raise FileExistsError("reportalreadybuilt")
    SimpleDocTemplate(str(path),pagesize=(210*mm,297*mm),rightMargin=23*mm,leftMargin=22*mm,topMargin=22*mm,bottomMargin=24*mm,title="PathSnap Accuracy-improvement Addendum",author="PathSnap project",pageCompression=1).build(story,onFirstPage=footer,onLaterPages=footer);return path


def main():
    d=assemble();build_markdown(d);pdf=build_pdf(d);manifest={"created_at_utc":c.utc_now(),"frozen_selection_sha256":d["frozen_selection_sha256"],"builder_sha256":c.sha256_file(__file__),"files":{p.name:c.sha256_file(p) for p in [pdf,OUTPUT/"ACCURACY_REPORT.md",OUTPUT/"accuracy_results.json",OUTPUT/"model_comparison.csv"]}};c.write_new_json(OUTPUT/"report_manifest.json",manifest);print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
