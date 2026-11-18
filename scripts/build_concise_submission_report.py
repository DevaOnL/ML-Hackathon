#!/usr/bin/env python3
"""Build only the report sections/questions requested in the offline assignment."""
from __future__ import annotations
import hashlib
import html
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/submission"
FIG=OUT/"figures"
REPO="https://github.com/DevaOnL/ML-Hackathon/blob/feat/train-only-accuracy-improvement/"
INPUTS={}


def load(path):
    p=ROOT/path;INPUTS[path]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())


def image(path,width):
    from PIL import Image as PILImage
    p=ROOT/path if not Path(path).is_absolute() else Path(path)
    INPUTS[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    with PILImage.open(p) as im:w,h=im.size
    return Image(str(p),width=width,height=width*h/w)


def fmt(v,digits=3):return f"{100*v:.{digits}f}%"


def num(v):return f"{v:.4f}"


def prose(text):
    return "".join(part if part.startswith("<") else prose_plain(part) for part in re.split(r"(<[^>]+>)",str(text)))


def prose_plain(text):
    text=re.sub(r"([A-Za-z]{2,})(?=[0-9])",r"\1 ",str(text))
    text=re.sub(r"(F1|L2)(?=[0-9])",r"\1 ",text)
    text=re.sub(r"(?<=[0-9])(?=[sm]\b)"," ",text)
    text=re.sub(r"(Section)([0-9])",r"\1 \2",text)
    text=re.sub(r";(?=\S)","; ",text)
    text=re.sub(r",(?=[A-Za-z])",", ",text)
    text=re.sub(r"(?<=[A-Za-z]),(?=[0-9])",", ",text)
    return text


def evidence():
    current=load("experiments/final_five/current/results.json")
    seq=load("experiments/final_five/current/sequence_classification.json")
    old=load("experiments/results.json")
    phase=load("experiments/phase_three/results.json")
    phases=load("experiments/phase_three/sequence_classification.json")["models"]
    rows=[]
    names=[("logistic_regression","Logistic Regression","linear"),("knn_distance_weighted","Distance-weighted KNN","instance"),("bagging","Bagging","ensemble"),("bagging_tuned","Bagging tuned","ensemble"),("gaussian_nb","GaussianNB","probabilistic"),("random_forest","Random Forest","ensemble"),("extra_trees","Extra Trees","ensemble"),("hist_gradient_boosting","HistGB","boosting"),("hist_gradient_boosting_tuned","HistGB tuned","boosting")]
    for name,label,family in names:
        source=phase if name=="bagging_tuned" else old if name in ("random_forest","extra_trees") else current
        cv=source["cv"][name];ev=source["evaluations"][name]
        obs=phases[name]["student_test"] if name=="bagging_tuned" else ev["student_observation_metrics"] if name in ("random_forest","extra_trees") else seq[name]
        rows.append(dict(name=name,label=label,family=family,params=cv["parameters"],cv=cv["oof_metrics"],train=ev["train_resubstitution_metrics"],student=ev["student_metrics"],obs=obs,fit=ev["training_seconds"],inference=ev["student_inference_seconds"]))
    extra=[]
    for family,label in [("logistic_regression","Logistic Regression"),("knn_distance_weighted","KNN"),("bagging","Bagging"),("gaussian_nb","GaussianNB"),("hist_gradient_boosting","HistGB")]:
        p=f"experiments/accuracy_improvement/confirmation/{family}.json";r=load(p)
        extra.append((label,r))
    return rows,current,extra


def plots(rows,audit):
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.18,"figure.facecolor":"white","axes.axisbelow":True})
    labels=[r["label"].replace("Logistic Regression","LR").replace("Distance-weighted KNN","KNN").replace("Random Forest","RF").replace("Extra Trees","ExtraTrees") for r in rows]
    fig,axs=plt.subplots(2,2,figsize=(11.6,7.5),constrained_layout=True)
    fields=[("CV binary Macro F1",[r["cv"]["macro_f1"] for r in rows]),("Student observation Top-1",[r["student"]["top1_accuracy"] for r in rows]),("Student binary Macro F1",[r["student"]["macro_f1"] for r in rows]),("Student binary log loss (lower is better)",[r["student"]["log_loss"] for r in rows])]
    for ax,(title,values) in zip(axs.flat,fields):
        bars=ax.barh(labels,values,color=["#245a82" if r["name"]==("histgb_route_final" if any(v["name"]=="histgb_route_final" for v in rows) else "hist_gradient_boosting_tuned") else "#97b3bd" for r in rows]);ax.invert_yaxis();ax.set_title(title,loc="left",fontsize=10,fontweight="bold");ax.set_xlim(0,1 if "loss" not in title else .42);ax.set_xlabel("fraction" if "loss" not in title else "log loss")
        for b,v in zip(bars,values):ax.text(min(v+.012,ax.get_xlim()[1]-.067),b.get_y()+b.get_height()/2,f"{v:.3f}",va="center",fontsize=8)
    fig.savefig(FIG/"model_comparison.png",dpi=180);plt.close(fig)
    trips=load("analysis/data_audit/per_trip.json")
    fig,axs=plt.subplots(1,2,figsize=(10.6,2.5),constrained_layout=True)
    for split,label,col in [("train","TRAIN80","#245a82"),("student_test","Student10","#b86d32")]:
        vals=[r["track"]["count"] if isinstance(r.get("track"),dict) else r.get("track_rows",0) for r in trips if r.get("split")==split]
        if not vals:
            vals=[r["rows"]["track"] for r in trips if r.get("partition")==split and "rows" in r]
        if vals:axs[0].plot(sorted(vals),label=label,color=col)
    if not axs[0].lines:
        for split,label,col in [("train","TRAIN80","#245a82"),("student_test","Student10","#b86d32")]:
            d=audit["splits"][split]["per_trip_track_points"];axs[0].plot([0,25,50,75,100],[d["p0"],d["p25"],d["p50"],d["p75"],d["p100"]],marker="o",label=label,color=col)
        axs[0].set_xlabel("Trip-size percentile")
    else:axs[0].set_xlabel("Trips ordered by size")
    axs[0].set_title("GPS observations per trip",loc="left");axs[0].set_ylabel("Observations");axs[0].legend(frameon=False,fontsize=8)
    for split,label,col in [("train","TRAIN80","#245a82"),("student_test","Student10","#b86d32")]:
        d=audit["splits"][split]["pooled_row_weighted_distributions"]["track_time_deltas"];axs[1].plot([50,95,99,100],[d["p50"],d["p95"],d["p99"],d["p100"]],marker="o",label=label,color=col)
    axs[1].set_yscale("log");axs[1].set_xlabel("Interval percentile");axs[1].set_ylabel("Seconds (log scale)");axs[1].set_title("Observed timestamp gaps",loc="left");axs[1].legend(frameon=False,fontsize=8)
    fig.savefig(FIG/"eda.png",dpi=180);plt.close(fig)


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    rows,current,extra=evidence();by={r["name"]:r for r in rows};audit=load("analysis/data_audit/summary.json");bayes=load("experiments/final_five/current/bayesian_optimization.json");hmm=load("reports/final_five/decoder/student_ablation.json");evalr=load("reports/final_five/evaluator/route_metrics.json");mapmeta=load("visualizations/final_five/maps/final_development_trip_00000009.metadata.json");selection=load("experiments/accuracy_improvement/confirmation/frozen_selection.json");latest=load("experiments/accuracy_improvement/student_comparison/summary.json")
    target=None
    if (ROOT/"artifacts/target96/acceptance.json").exists():
        acceptance=load("artifacts/target96/acceptance.json")
        if acceptance.get("accepted") is True:
            target=load("experiments/target96/student_comparison/summary.json");target_manifest=load("artifacts/target96/release_manifest.json");target_train=load("artifacts/target96/training_metrics.json");target_cv=load("experiments/accuracy_improvement/histgb/cv/histgb_context_large_full.json");target_routes=load("experiments/target96/precision/frozen_selection.json")
            if target["target_met"] is not True or target["candidate"]["edge_f1"]<.96 or acceptance["comparison_sha256"]!=hashlib.sha256((ROOT/"experiments/target96/student_comparison/summary.json").read_bytes()).hexdigest():raise ValueError("unaccepted target96 report input")
            s=target["candidate"]["observation_pooled"];r={"name":"histgb_route_final","label":"HistGB final (70F)","family":"boosting","params":target_manifest["configuration"],"cv":target_cv["oof_metrics"],"train":target_train["metrics"],"student":s,"obs":{"observation_edge_macro_f1":s["observation_edge_macro_f1"],"observation_categorical_log_loss_clipped":s["observation_nll"],"raw_classifier_continuity":s["raw_adjacency_trip_macro"]},"fit":load("artifacts/target96/training_metadata.json")["training_seconds"],"inference":s["probability_call_total_seconds"]};rows.append(r);by[r["name"]]=r
    plots(rows,audit)
    fontdir=Path(matplotlib.get_data_path())/"fonts/ttf"
    for name,file in [("DV","DejaVuSans.ttf"),("DV-Bold","DejaVuSans-Bold.ttf"),("DV-Italic","DejaVuSans-Oblique.ttf")]:pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
    pdfmetrics.registerFontFamily("DV",normal="DV",bold="DV-Bold",italic="DV-Italic",boldItalic="DV-Bold")
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Body",fontName="DV",fontSize=9.1,leading=12.1,spaceAfter=5))
    styles.add(ParagraphStyle(name="Small",fontName="DV",fontSize=8,leading=10.4,spaceAfter=4))
    styles.add(ParagraphStyle(name="TitleC",fontName="DV-Bold",fontSize=17,leading=21,spaceAfter=9,textColor=colors.HexColor("#193e54")))
    styles.add(ParagraphStyle(name="H1C",fontName="DV-Bold",fontSize=14,leading=18,spaceAfter=7,textColor=colors.HexColor("#193e54")))
    styles.add(ParagraphStyle(name="H2C",fontName="DV-Bold",fontSize=10.5,leading=14,spaceBefore=7,spaceAfter=5))
    styles.add(ParagraphStyle(name="CellC",fontName="DV",fontSize=7.5,leading=9.8))
    styles.add(ParagraphStyle(name="CellTiny",fontName="DV",fontSize=6.8,leading=8.7))
    story=[];md=[];W=174*mm
    def para(text,kind="Body"):
        text=prose(text)
        story.append(Paragraph(text,styles[kind]));md.append(text.replace("<b>","**").replace("</b>","**"))
    def head(text,level=1):
        story.append(Paragraph(text,styles["H1C" if level==1 else "H2C"]));md.extend(["",("##" if level==1 else "###")+" "+text,""])
    def table(data,widths,tiny=False):
        sty=styles["CellTiny" if tiny else "CellC"]
        def cell(x):return Paragraph(prose(x),sty)
        t=Table([[cell(x) for x in row] for row in data],colWidths=[x*mm for x in widths],repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e8eff2")),("VALIGN",(0,0),(-1,-1),"TOP"),("LINEBELOW",(0,0),(-1,0),.6,colors.HexColor("#507487")),("LINEBELOW",(0,1),(-1,-1),.25,colors.HexColor("#d4dfe3")),("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),("TOPPADDING",(0,0),(-1,-1),3 if tiny else 4),("BOTTOMPADDING",(0,0),(-1,-1),3 if tiny else 4)]))
        story.append(t);story.append(Spacer(1,5));md.extend(["| "+" | ".join(str(x).replace("<br/>","; ") for x in row)+" |" for row in data])
    def source(label,path):para(f'<font color="#486879"><link href="{REPO+path}">Source: {html.escape(label)}</link></font>',"Small")
    def page():story.append(PageBreak())
    def fig(path,caption,width=W):story.append(image(path,width));story.append(Spacer(1,4));para(caption,"Small");md.extend(["",f"![{caption}](<../../{path}>)",""])
    def stat(name):
        r=by[name];s=r["student"];o=r["obs"]
        return f"Student Top-1 {fmt(s['top1_accuracy'])}; observation edge Macro F1 {num(o['observation_edge_macro_f1'])}; observation NLL {num(o['observation_categorical_log_loss_clipped'])}. Training Top-1 {fmt(r['train']['top1_accuracy'])} versus grouped-CV {fmt(r['cv']['trip_macro_top1_accuracy'])}."
    story.append(Paragraph("PathSnap — Technical Report",styles["TitleC"]));md.append("# PathSnap — Technical Report")
    head("2. Dataset Understanding and Exploratory Data Analysis")
    table([["Partition","Trips","GPS observations","Nodes / directed arcs"],["Training",80,"208,417","1,933,485 / 3,876,904"],["Student testing",10,"19,160","90,689 / 185,041"],["Evaluator testing",10,"19,674","Final scoring only; not used for training"]],[35,15,41,83])
    para("<b>Preprocessing.</b> Each trip has a .track (longitude, latitude, timestamp), .nodes (longitude, latitude), .arcs (tail, head) and .route (ordered directed arc IDs). Arc IDs are zero-based row indices local to a trip. Every file was parsed and checked for finite values, valid coordinates/endpoints and ordered timestamps. No missing, NaN or nonfinite fields were found, so no imputation was needed. Local equirectangular projection converts coordinates to metres; repeated GPS rows, parallel arcs, cycles and reference order are preserved.")
    table([["Quality check","TRAIN80","Student10"],["Missing files / blank rows / invalid coordinates","0 / 0 / 0","0 / 0 / 0"],["Nonpositive time intervals / isolated nodes","0 / 0","0 / 0"],["Median / longest GPS gap","1 s / 153,064 s","1 s / 210 s"],["Repeated-coordinate GPS rows / broken reference joins","572 / 13","6 / 1"]],[91,42,41])
    para("<b>Patterns.</b> Most intervals are one second, but long gaps and uneven trip sizes rule out assuming uniform motion or randomly splitting GPS rows. Median trip sizes are 1,529 TRAIN and 1,420 student observations. Trip networks may have several directed components (median strong-component counts127 and53.5 respectively); disconnected references and nearby parallel roads limit label certainty.")
    para("<b>Features and targets.</b> The base38 features describe distance/rank, projection, arc length, heading/reliability, neighbouring offsets, motion/gap context and endpoint degree. The final70-feature variant adds stateless geometry-energy, rank and best-rival-margin transforms. They distinguish close parallel/opposite roads using motion and connectivity. Baseline candidate pools start at12; the final route-tuned pipeline uses24, retry48 on reachability failure, within80 m. References supply route sequences, not per-GPS labels: training labels use the closest reference-route segment plus a weak heading term. All observation scores therefore measure this proxy, not independent GPS truth. LR/KNN/GaussianNB scaling is learned inside each TRAIN fold.")
    fig("reports/submission/figures/eda.png","Measured TRAIN/student trip-size and timestamp patterns. A raw GPS trajectory and its actual road network appear in the layered Folium map in Section6.")
    source("Data and feature implementation","mapmatch/features.py")
    page()
    head("3. Model Experimentation and Comparison")
    para("Baselines use three-fold whole-trip CV on TRAIN80 (350 observations/trip,28,000 total). The final70-feature model uses the separately fixed TRAIN64 CV; its complete route decoder is chosen by equal-trip route-edge F1 on all64 dense held-out trajectories. All student results use19,160 observations. Historical student development is not blind; no evaluator data choose settings. All tested families and tuned variants are retained.")
    head("3.1 Overall model comparison",2)
    cfg={"logistic_regression":"C=1;500 iterations","knn_distance_weighted":"k=25;distance weights;60k bank","bagging":"40 trees;depth14;leaf3;sample0.7","bagging_tuned":"48 trees;depth13;leaf2;sample0.7161","gaussian_nb":"smoothing10⁻⁷","random_forest":"100 trees;depth16;leaf4;features0.7","extra_trees":"100 trees;depth22;leaf3;features0.8","hist_gradient_boosting":"120 rounds;rate0.09;23 leaves","hist_gradient_boosting_tuned":"104 rounds;rate0.1590;22 leaves"}
    if target:cfg["histgb_route_final"]="70F;180 rounds;31 leaves"
    data=[["Model","Algorithm Family","Key Configuration","CV Macro F1-score","Top-1 Accuracy","Route continuity","Student-Test Macro F1-score","Log Loss","Inference Time"]]
    for r in rows:data.append([r["label"],r["family"],cfg[r["name"]],num(r["cv"]["macro_f1"]),fmt(r["student"]["top1_accuracy"],2),"N/A: no route",num(r["student"]["macro_f1"]),num(r["student"]["log_loss"]),f"{r['inference']:.3f}s"])
    table(data,[23,15,29,17,16,17,20,13,24],True)
    para("CV/Student Macro F1 is a <b>binary candidate-row</b> metric at threshold0.5; binary log loss uses unthresholded clipped probabilities. Top-1 is pooled <b>observation</b> accuracy; all candidate misses count wrong. Standalone classifiers do not reconstruct routes, so route continuity is N/A. Timing measures the student probability call only, excluding features/routing; retained timings are not repeated hardware-normalised benchmarks.","Small")
    data=[["Model","Binary accuracy","Positive precision / recall","Observation edge Macro F1","Observation NLL","Raw adjacency","Fit time"]]
    for r in rows:
        s=r["student"];o=r["obs"];data.append([r["label"],num(s["accuracy"]),f"{s['precision']:.4f} / {s['recall']:.4f}",num(o["observation_edge_macro_f1"]),num(o["observation_categorical_log_loss_clipped"]),fmt(o["raw_classifier_continuity"],2),f"{r['fit']:.2f}s"])
    table(data,[27,20,29,28,24,25,21],True)
    para("Observation Macro F1 uses trip-qualified directed-road classes. Observation NLL normalises positive candidate scores per GPS row and floors absent targets at10⁻⁷; it is not binary log loss. Raw adjacency checks same-edge/head-to-tail neighbours without inserting connectors. Candidate misses: baseline TRAIN578/28,000; student19/19,160. Final K24 student12/19,160=0.0626%. Full-route scores appear in Section5.","Small")
    if target:para("Final70F CV: TRAIN64/22,400 observations, K12; final student metrics: all19,160 observations, K24 (retry48). This wider candidate policy is explicit; older rows retain their original80-trip CV and K12 classification metrics. Route continuity for the actual final pipeline is1.0000 (Section5), not classifier-only adjacency.","Small")
    source("Final comparison data","experiments/target96/student_comparison/summary.json" if target else "experiments/final_five/current/results.json")
    page()
    head("3.2 Results and interpretation for each model")
    para("Student results below use observation-level road classes; binary scores are reported separately in Section3.1.","Small")
    head("Logistic Regression",2)
    para("A fast scaled linear baseline tests whether distance, heading and local context already separate correct candidates. Its limited nonlinear interactions are the main restriction. Initial settings: C=1, L2 regularisation, L-BFGS,500 iterations; weighted binary log loss is minimised. "+stat("logistic_regression")+" The almost equal training/CV scores show little memorisation; its lower CV score than trees suggests limited model capacity.")
    head("Distance-weighted KNN",2)
    para("Neighbour voting can model local nonlinear candidate similarity without assuming a functional boundary. It requires scaling and is costly for a large candidate matrix. Initial settings: k=25, Euclidean distance and inverse-distance votes; a weighted class-stratified60,000-row TRAIN bank bounds memory. There is no iterative training loss; k and the bank determine the estimate. "+stat("knn_distance_weighted")+" The3.09-point train/CV gap indicates overfitting;187.08 s student inference is unsuitable compared with historical38-feature HistGB's0.50 s.")
    head("Bagging",2)
    para("Bootstrapped decision trees capture nonlinear interactions and reduce single-tree variance, at greater fit cost. Initial settings:40 trees, max_depth14, min_samples_leaf3, max_samples0.7; trees minimise weighted Gini impurity and their probabilities are averaged. "+stat("bagging")+" The4.92-point gap suggests overfitting. A Bayesian variant used48 trees, depth13, leaf2 and sample0.7161: CV Top-1 improved89.146%→89.250%, but student Top-1 fell90.480%→90.339% (edge Macro F10.8248; NLL0.2817), so it was not the final model.")
    head("Gaussian Naive Bayes",2)
    para("A cheap probabilistic baseline tests whether per-feature class distributions suffice. Correlated geometric features violate conditional independence and can make probabilities overconfident. Initial var_smoothing=10⁻⁷; weighted Gaussian class means/variances are fitted by likelihood, without gradient training. "+stat("gaussian_nb")+" The small train/CV gap is consistent with limited capacity; binary student log loss0.3591 is much worse than boosted trees0.0552.")
    source("KNN/Bagging training and bank policy","mapmatch/phase_models.py")
    page()
    head("3.2 Results and interpretation for each model (continued)")
    head("Random Forest",2)
    para("Bagged trees with feature subsampling capture nonlinear distance/heading interactions while reducing correlated-tree variance. Initial settings:100 trees, depth16, min_samples_leaf4, max_features0.7; weighted Gini splits with averaged probabilities. "+stat("random_forest")+" The6.15-point training/CV gap is larger than HistGB's; higher training accuracy did not justify selecting it. Student binary log loss0.0548 is good, but its CV Top-1 is below HistGB.")
    head("Extra Trees",2)
    para("Randomised split thresholds provide a fast nonlinear ensemble and an alternative variance/bias tradeoff. Initial settings:100 trees, depth22, min_samples_leaf3, max_features0.8, bootstrap=False; random candidate splits are compared by weighted Gini impurity. "+stat("extra_trees")+" The7.78-point training/CV gap is the largest among these baselines; stronger memorisation did not improve student Top-1 over Bagging/HistGB.")
    head("HistGradientBoosting",2)
    para("Sequential trees optimise probability estimates for interacting distance, heading and motion features; histogram splits keep fitting/inference practical. Initial settings:120 iterations, learning_rate0.09,23 leaves, min_samples_leaf40, L2=0.5, early_stopping=False. The loss is weighted binary log loss. "+stat("hist_gradient_boosting")+" Bayesian tuning gives104 iterations, rate0.159024,22 leaves, leaf minimum38, L2=0.860422. Tuned student results are Top-1 90.678%, edge Macro F10.8346 and NLL0.2767. Its training/CV gap rises from1.58 to2.06 points: tuning does not prove reduced overfitting.")
    head("Final route-tuned HistGB",2)
    if target:
        s=target["candidate"]["observation_pooled"];tr=target_train["metrics"];cv=target_cv["oof_metrics"]
        para(f"The final70-feature model uses180 rounds, rate0.1, 31 leaves, leaf minimum30 and L2=2; objective remains weighted binary log loss. All80 TRAIN trips provide the final fit. Its Top-1 is{fmt(tr['top1_accuracy'])} in training versus{fmt(cv['trip_macro_top1_accuracy'])} in TRAIN64 CV, so overfitting remains visible. Student pooled Top-1 is{fmt(s['top1_accuracy'])}, observation edge Macro F1{num(s['observation_edge_macro_f1'])} and NLL{num(s['observation_nll'])}. This classifier with the route-tuned HMM reaches{fmt(target['candidate']['edge_f1'])} route-edge F1; classifier accuracy is a separate metric.")
    else:
        para("HistGB was selected by the highest baseline grouped-CV Top-1 (89.268%, tuned89.339%) with practical inference and finite probabilities for HMM emissions.")
    para("None of these estimators recorded epoch-wise train/validation histories; no artificial training curves are supplied. Actual cross-model plots follow. All model configurations/trials remain available in the repository.","Small")
    source("Initial estimators and loss implementation","mapmatch/experiments.py")
    page()
    head("3.3 Measured model visualisations")
    fig("reports/submission/figures/model_comparison.png","All trained algorithm families and tuned configurations. Baseline CV uses TRAIN80; final70F CV uses TRAIN64 (K12). Final student results use K24, older rows use their recorded candidate pools; all use the same10 trips/19,160 GPS rows. Macro F1/log loss are binary candidate-row metrics.")
    head("Other tested feature configurations",2)
    para("A later TRAIN-only search tested compact12-, physics3- and context70-feature variants of the five main families. The following candidates were fitted on the same64 trips as their controls and scored once on16 internally reserved TRAIN trips. Earlier releases had used all80 trips; these are internal comparisons, not blind/student-test scores.","Small")
    data=[["Family","Selected profile / configuration","Control Top-1","Candidate Top-1"]]
    desc={"Logistic Regression":"compact;C=1","KNN":"compact;k=35,p=2","Bagging":"context;64 trees,depth18,leaf6","GaussianNB":"compact;smoothing0.1","HistGB":"context;180 rounds,rate0.1,31 leaves"}
    for label,r in extra:data.append([label,desc[label],fmt(r["records"]["baseline"]["metrics"]["trip_macro_top1_accuracy"]),fmt(r["records"]["candidate"]["metrics"]["trip_macro_top1_accuracy"])])
    table(data,[36,78,30,30])
    para("The objectives are unchanged. LR/GaussianNB binary Macro F1 decreased despite higher Top-1; separate full-student scoring for those four candidates is N/A (not run). The earlier context HistGB with fixedK12 failed2/10 student routes (edge F10.7611), so that pipeline was rejected. The final model is the separately frozen route-tuned pipeline in Sections4–5; prior failed results remain unchanged.","Small")
    if target:
        data=[["Full-route student metric","Previous final","New final"],["Equal-trip route-edge F1",fmt(target["baseline"]["edge_f1"]),fmt(target["candidate"]["edge_f1"])],["Length-weighted F1",num(target["baseline"]["length_f1"]),num(target["candidate"]["length_f1"])],["Continuity / failed trips","1.000 /0","1.000 /0"]];table(data,[100,37,37],True)
    source("Additional tested configurations and confirmation results","reports/accuracy_improvement/accuracy_results.json")
    page()
    head("4. Bayesian Hyperparameter Optimisation")
    para("The baseline HistGB Bayesian search used a Gaussian-process Matérn-5/2 surrogate and expected improvement. Its objectives used three-fold whole-trip CV on TRAIN80, maximising equal-trip observation Top-1. This favours the correct road for every trip rather than the many negative candidate rows. Student/evaluator scores were not search objectives.")
    table([["Hyperparameter","Range / distribution","Reason"],["learning_rate","0.025–0.2;log-uniform","Explore small and large boosting steps"],["max_iter","70–200;integer","Bound capacity and fitting cost"],["max_leaf_nodes","7–47;integer","Control tree complexity"],["min_samples_leaf","15–160;log-integer","Regularise rare/noisy candidates"],["l2_regularization","0.01–20;log-uniform","Regularise leaf values"]],[49,55,70])
    para("<b>Budget and best values.</b>12 recorded trials comprise one measured baseline, three seeded initial designs and eight model-based proposals. Search time was162.61 s. The best Bayesian baseline configuration is104 iterations, rate0.1590239,22 leaves, minimum leaf38 and L2=0.8604220; it was refitted on all80 TRAIN trips.")
    base=by["hist_gradient_boosting"];tuned=by["hist_gradient_boosting_tuned"]
    data=[["Same-data metric","Untuned","Tuned"]]
    for label,a,b in [("TRAIN CV Top-1",base['cv']['trip_macro_top1_accuracy'],tuned['cv']['trip_macro_top1_accuracy']),("TRAIN CV binary Macro F1",base['cv']['macro_f1'],tuned['cv']['macro_f1']),("Student observation Top-1",base['student']['top1_accuracy'],tuned['student']['top1_accuracy']),("Student observation edge Macro F1",base['obs']['observation_edge_macro_f1'],tuned['obs']['observation_edge_macro_f1']),("Student binary Macro F1",base['student']['macro_f1'],tuned['student']['macro_f1']),("Student binary log loss",base['student']['log_loss'],tuned['student']['log_loss']),("Student observation NLL",base['obs']['observation_categorical_log_loss_clipped'],tuned['obs']['observation_categorical_log_loss_clipped'])]:data.append([label,num(a),num(b)])
    data.extend([["Student probability-call time",f"{base['inference']:.3f}s",f"{tuned['inference']:.3f}s"],["All-TRAIN fit time",f"{base['fit']:.3f}s",f"{tuned['fit']:.3f}s"]]);table(data,[100,37,37])
    para("<b>Did tuning help?</b> CV Top-1 rises by0.071 percentage points; student Top-1 rises by0.021 points and inference is faster. Student edge/binary Macro F1 and observation NLL improve slightly. CV binary Macro F1 and student binary log loss worsen. The gain is modest and does not justify an across-the-board improvement claim.")
    if target:
        para(f"<b>Final route tuning.</b> A separate TRAIN-only search optimised route-edge F1:12 joint settings on18 OOF trips; then9 precision settings on18, advancing two plus a paired control to all64 dense OOF trips. Candidate width12/24, beta2/5/10, geometry0.1/0.25/1, reverse cost0/1/3/8 and appended-arc cost0/0.25/0.5/1/2/5 were tested in fixed menus. Selected:K24, beta5, geometry0.25, reverse3 at GPS chord≤20 m and appended-arc cost5. All64 edge F1 is{target_routes['baseline']['edge_f1']:.4f}→{target_routes['selected']['edge_f1']:.4f}, with the same3 failed TRAIN routes scored zero in both means. The final10-trip comparison was run after freezing this choice; it is development evidence, not a new blind test.","Small")
    source("Search ranges, trial objectives and best values","experiments/final_five/current/bayesian_optimization.json")
    page()
    head("5. HMM and Viterbi Route Reconstruction")
    para("<b>Hidden states.</b> Each state is a candidate directed arc and the GPS point's projected fraction along it. Initial candidates are the closest24 within80 m; an empty set expands the radius, up to1000 m. Every original GPS row is decoded.")
    para("<b>Emissions.</b> The selected classifier's clipped positive scores p are normalised within each observation, q=p/Σp. A candidate's final emission cost is E=0.25×½(d/σ)²−log(q), combining lateral distance d with learned compatibility; σ=7.16065 m. These are HMM emission potentials, not an HMM posterior probability.")
    para("<b>Transitions and Viterbi.</b> Let D be directed shortest-metre travel distance, g the GPS chord, n the exact number of arcs reconstruction appends, and R an endpoint-opposite edge switch with g≤20 m. Final cost T=|D−g|/5+5n+3R; unreachable transitions remain infinite. Vₜ(j)=Eₜ(j)+minᵢ[Vₜ₋₁(i)+Tᵢⱼ] stores backpointers, then reconstructs those same shortest-metre graph arcs. Finite arc/reversal costs discourage short excursions before decoding; no output arcs are removed.")
    para("<b>Gaps and disconnected segments.</b> At a gap, transitions still use observed displacement and directed reachability; heading windows stop across gaps above20 s. Routing starts at the GPS chord plus300 m slack; if all transitions fail, the search expands to at least5000 m or ten times that horizon. Continued reachability failure widens candidate width24→48 and recomputes candidate-relative features/emissions. If no connected sequence exists, inference reports failure rather than teleporting or fabricating edges. Real cycles and repeated traversals remain in the output.")
    para("<b>Measured effect.</b> Both methods below use the same final candidate sets and classifier probabilities on all10 student trips/19,160 observations. Classifier-only picks each local argmax, collapses adjacent equal arcs and adds no connecting paths. Route F1 compares the full supplied reference sequence; repeated arcs count. Values are equal-trip means.")
    raw=hmm['aggregate']['classifier_only_raw']['trip_macro'];dec=hmm['aggregate']['classifier_plus_HMM']['trip_macro']
    if target:
        c=target['candidate'];o=c['observation_trip_macro'];a=c['classifier_only_route']
        raw={'proxy_accuracy':o['top1_accuracy'],'proxy_macro_f1':o['observation_edge_macro_f1'],'route_edge_f1':a['edge_f1'],'route_length_f1':a['length_f1'],'route_edit_similarity':a['edit_similarity'],'route_continuity':a['route_continuity'],'inference_seconds':o['probability_call_seconds']}
        dec={'proxy_accuracy':o['decoded_proxy_accuracy'],'proxy_macro_f1':o['decoded_proxy_macro_f1'],'route_edge_f1':c['edge_f1'],'route_length_f1':c['length_f1'],'route_edit_similarity':c['edit_similarity'],'route_continuity':c['route_continuity'],'inference_seconds':c['mean_seconds']}
    data=[["Student metric","Classifier only","Classifier + HMM"]]
    for label,k in [("Observation proxy accuracy","proxy_accuracy"),("Observation edge Macro F1","proxy_macro_f1"),("Directed-arc route F1","route_edge_f1"),("Length-weighted route F1","route_length_f1"),("Sequence edit similarity","route_edit_similarity"),("Route continuity","route_continuity")]:data.append([label,num(raw[k]),num(dec[k])])
    data.append(["Mean prediction time/trip",f"{raw['inference_seconds']:.3f}s",f"{dec['inference_seconds']:.3f}s"]);table(data,[100,37,37])
    if target:
        para(f"The final pipeline improves route-edge F1 from{fmt(target['baseline']['edge_f1'])} to <b>{fmt(target['candidate']['edge_f1'])}</b> on the same10 trips/19,160 GPS rows, with zero failures and100% continuity. Length F1 rises{target['baseline']['length_f1']:.4f}→{target['candidate']['length_f1']:.4f}; edit similarity{target['baseline']['edit_similarity']:.4f}→{target['candidate']['edit_similarity']:.4f}. This is route F1, not classification accuracy. Every trip improves pointwise versus its own raw classifier, but improvement is not universal across models: original trip00000007 proxy accuracy fell97.267%→96.307% with HMM while its route F1 improved. The reference break on trip00000076 is retained.")
    else:para("The original HMM route-edge F1 is0.9514 with10/10 connected student routes.")
    if target:
        ev=load("experiments/target96/evaluator/summary.json");em=ev.get("macro",ev.get("summary",{}).get("macro",ev))
        if "aggregate" in ev:em=ev["aggregate"]
        para(f"<b>Frozen evaluator results.</b> On10 trips/19,674 observations, edge F1{em['edge_f1']:.4f}, length F1{em['length_f1']:.4f}, edit similarity{em['edit_similarity']:.4f} and continuity{em['route_continuity']:.4f}, with zero failures. Every prediction was sealed before references opened; no evaluator score changed the model. These are local results, not organiser-certified scores.","Small")
    else:
        em=evalr["summary"]["macro"];para(f"Original evaluator edge F1{em['edge_f1']:.4f}.","Small")
    para("Classifier-only timing is the probability call; HMM timing includes full feature generation, candidates and routing. Their difference is not isolated DP overhead.","Small")
    source("Paired classifier/HMM experiment","experiments/target96/student_comparison/summary.json" if target else "reports/final_five/decoder/student_ablation.json")
    page()
    if target:final_map=load("reports/target96/map_evidence.json")
    head("6. Geographical Visualisation Using Folium")
    para("Student trip00000009 shows original GPS observations, its supplied directed road graph, the reference route and the submitted model's predicted route. The screenshot is exported from the actual interactive Folium map; no road geometry or prediction was drawn manually.")
    fig(final_map["screenshot_path"] if target else "visualizations/final_five/maps/final_development_trip_00000009.png","Folium map of student trip00000009. Road/GPS/reference/prediction layers use distinct colours, labelled controls and an automatically fitted extent.")
    if target:
        map_metrics=final_map["route_metrics"];pi=final_map["prediction_integrity"]
        para(f"The map contains1,049 GPS points,6,745 nodes and13,106 directed arcs. The accepted prediction has{map_metrics['predicted_arc_count']} original arc traversals versus{map_metrics['reference_arc_count']} in the reference; repeated occurrences remain. It is connected. Directed-arc F1 is{map_metrics['edge_f1']:.4f}, length-weighted F1{map_metrics['length_f1']:.4f} and sequence edit similarity{map_metrics['edit_similarity']:.4f}. Its exact route/model/policy hashes match the final comparison.")
    else:para("The original map depicts444 predicted arc traversals and429 reference arcs.")
    para("The layer control lets the reader isolate GPS, network, reference and prediction; the legend identifies each. Clickable features show their original IDs. This is a development/reference map; evaluator reference geometry is not delivered to the browser.")
    htmlmap=final_map["html_path"] if target else "visualizations/final_five/maps/final_development_trip_00000009.html"
    generator="scripts/generate_target96_map.py" if target else "visualizations/final_five/generate_final_map.py"
    para(f'<b>Interactive map:</b> <link href="{REPO+htmlmap}">Folium HTML in the repository</link>. <b>Generator:</b> <link href="{REPO+generator}">map generator</link>.')
    source("Map source and measured route statistics","reports/target96/map_evidence.json" if target else "visualizations/final_five/maps/final_development_trip_00000009.metadata.json")
    page()
    head("7. API, UI and Deployment")
    para('<b>Repository:</b> <link href="https://github.com/DevaOnL/ML-Hackathon">github.com/DevaOnL/ML-Hackathon</link>. <b>Application:</b> <link href="https://ml.d3va.dev">ml.d3va.dev</link>. Laptop1 runs FastAPI, the saved HistGB classifier, road graphs, Viterbi and private reference scoring. Laptop2 opens the browser UI, selects one of ten trips and displays the returned GPS/road/prediction layers and scores. No model training or truth file is sent by the UI.')
    para("<b>Request flow.</b> GET /health checks readiness; GET /datasets lists available partitions; GET /datasets/{id}/trips lists exact trip IDs. POST /datasets/{id}/match receives only {trip_id}. The server loads graph/track, generates candidates, predicts and decodes, then separately reads the reference to score. The response contains route geometry, metrics and timing. The browser clears stale results when changing trips/datasets; references remain server-side.","Small")
    shotroot='reports/target96/live/report_images' if target else 'reports/final_five/report_images'
    mapshot='reports/target96/live/ui_route_map.png' if target else 'reports/final_five/live/screenshots/ui_route_map.png'
    termshot='reports/target96/terminal/api_response.png' if target else 'reports/submission/evidence/api_response.png'
    story.append(image(shotroot+'/development_dataset_controls.png',W));story.append(Spacer(1,5))
    panel=Table([[image(mapshot,111*mm),image(shotroot+'/development_scores.png',59*mm)]],colWidths=[113*mm,61*mm]);panel.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0)]));story.append(panel);story.append(Spacer(1,4));para("Working UI: dataset/trip selection, reconstructed route and measured scores. These are screenshots of the submitted application.","Small")
    story.append(image(termshot,W));story.append(Spacer(1,3))
    md.extend([f'![Working UI controls](<../../{shotroot}/development_dataset_controls.png>)',f'![Working route map](<../../{mapshot}>)',f'![Measured UI scores](<../../{shotroot}/development_scores.png>)',f'![Actual HTTP200 response on the receiving host](<../../{termshot}>)'])
    para("Receiving-host terminal: an actual HTTPS POST for student trip00000009 returns HTTP200, route scores and timings. Selected JSON fields are shown without changing their values; geometry is omitted for readability. This verifies the real response on the API host, not a physical two-laptop demonstration.","Small")
    para("<b>Deployment.</b> The existing HTTPS tunnel forwards to the API on127.0.0.1:8000; the same service serves the UI. Dependencies and run commands are in README. The selected model/config are loaded once; inference is limited to one request at a time and predicted routes are not response-cached. All ten student and ten evaluator trips are supported. The existing service now loads the accepted70-feature HistGB with the frozen route-tuned policy; health and public-response geometry match its exact model and route hashes. The earlier fixedK12 failed pipeline remains rejected.","Small")
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#bdcdd5'));canvas.line(18*mm,14*mm,192*mm,14*mm);canvas.setFont('DV',7.5);canvas.setFillColor(colors.HexColor('#536b76'));canvas.drawString(18*mm,9.6*mm,'PathSnap | Technical Report');canvas.drawRightString(192*mm,9.6*mm,str(doc.page));canvas.restoreState()
    path=OUT/'PathSnap_Final_Report.pdf'
    SimpleDocTemplate(str(path),pagesize=(210*mm,297*mm),rightMargin=18*mm,leftMargin=18*mm,topMargin=17*mm,bottomMargin=20*mm,title='PathSnap — Technical Report',author='PathSnap',pageCompression=1).build(story,onFirstPage=footer,onLaterPages=footer)
    (OUT/'PathSnap_Final_Report.md').write_text(chr(10).join(md)+chr(10))
    for p in (ROOT/'PathSnap_student_materials').glob('*.docx'):
        with ZipFile(p) as z:ET.fromstring(z.read('word/document.xml'))
        INPUTS[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'report':'reports/submission/PathSnap_Final_Report.pdf','required_sections':['2 Dataset / EDA','3 All actual models / comparison','4 Bayesian HPO','5 HMM / Viterbi','6 Folium','7 API / UI / deployment'],'inputs':INPUTS,'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'outputs':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,OUT/'PathSnap_Final_Report.md',FIG/'eda.png',FIG/'model_comparison.png']},'measurement_rerun':False,'old_report_untouched':True}
    (OUT/'report_sources.json').write_text(json.dumps(manifest,indent=2)+chr(10))
    print(json.dumps({'pdf':str(path),'sources':len(INPUTS),'bytes':path.stat().st_size},indent=2))

if __name__=='__main__':build()
