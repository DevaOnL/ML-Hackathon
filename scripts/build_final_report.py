#!/usr/bin/env python3
"""Build the consolidated final technical report for the PathSnap five-family phase.

This builder is the last line of defense between the project's evidence files
and the submitted PDF. It deliberately separates two documents:

* reports/final_five/REPORT_TEMPLATE.md - the authored template. It keeps its
  {{fact:name}} tokens forever. Every number, hash and timestamp in the final
  prose arrives through one of those tokens from a JSON evidence file.
* reports/final_five/TECHNICAL_REPORT.md - the resolved document, produced only
  by this script, never edited by hand.

Modes:

* --check   read-only readiness summary. No file is written. Reports which
             declared inputs are missing, which facts fail to resolve and
             which guards do not pass yet.
* --preview renders a watermarked DRAFT PDF. Unresolved optional facts appear
             as [[PENDING:name]] markers and placeholder boxes replace missing
             figures. A preview never claims to be final.
* --final   fails closed. Every declared input must exist, every fact must
             resolve from its JSON pointer, every guard must pass, no PENDING
             marker may remain, and every figure must be hash-bound to a
             figure manifest before the unwatermarked PDF is written.

Design rules carried over from the phase-three builder, plus final-phase
additions:

* facts read actual JSON pointers (dig), never free text in prose;
* the template token set is cross-checked against the fact registry so tokens
  cannot silently disappear between template versions;
* every future evidence file (decoder, live, evaluator, verification, tests,
  independent audit, final figures) is declared explicitly and fails closed in
  --final until it exists and passes its guard;
* strict guards are never relaxed forward - if a fact is wrong, the evidence
  or the template is patched, not the guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import os
import struct
import tempfile
from dataclasses import asdict
from urllib.parse import quote, unquote, urlsplit
from html import unescape
from datetime import datetime, timezone
from pathlib import Path

# Canonical installation is scripts/build_final_report.py. The ancestor fallback
# lets the isolated proposal use the actual checkout, without a wrong parents[1].
_SOURCE = Path(__file__).resolve()
ROOT = (_SOURCE.parents[1] if _SOURCE.parent.name == "scripts" else
        next((p for p in _SOURCE.parents if (p / "pyproject.toml").is_file()
              and (p / "mapmatch").is_dir()), _SOURCE.parents[1]))

TEMPLATE_PATH = ROOT / "reports" / "final_five" / "REPORT_TEMPLATE.md"
MD_PATH = ROOT / "reports" / "final_five" / "TECHNICAL_REPORT.md"
PDF_PATH = ROOT / "reports" / "final_five" / "PathSnap_Final_Technical_Report.pdf"
MANIFEST_PATH = ROOT / "reports" / "final_five" / "report_manifest.json"

REPORT_TITLE = "PathSnap"
REPORT_SUBTITLE = ("Final Technical Report - Map Matching as "
                   "Candidate Classification")
REPORT_TARGET_PAGES = None  # Completeness/readability, not an arbitrary page count.

# ---------------------------------------------------------------------------
# pins: the final contract. Guards compare live evidence against these.
# ---------------------------------------------------------------------------

FIVE_FAMILIES = [
    "logistic_regression",
    "knn_distance_weighted",
    "bagging",
    "gaussian_nb",
    "hist_gradient_boosting",
]
SELECTED_FAMILY = "hist_gradient_boosting"
TUNED_RUN = "hist_gradient_boosting_tuned"
SIX_CONFIGS = FIVE_FAMILIES + [TUNED_RUN]
PINNED_MODEL_SHA256 = (
    "c9cc8e9f36838f656a94d07ed69632979df9981ca94a9daf3b52fbce052e4b34"
)

# Statuses that count as resolved in the 47-row requirement matrix. Anything
# else is pending and blocks --final.
REQ_TERMINAL_STATUSES = {
    "done_evidence_exists",
    "user_exception",
    "human_external_blocker",
}

# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

INPUTS = []


def inp(key, rel, role, final, kind="json", owner="final-phase"):
    INPUTS.append({
        "key": key, "path": rel, "role": role, "final": bool(final),
        "kind": kind, "owner": owner,
    })


# --- current, already-measured evidence -------------------------------------
inp("results5", "experiments/final_five/current/results.json",
    "corrected five-family CV / student / Bayesian HPO results (single source of metrics)", True)
inp("accept", "reports/final_five/all_five_model_acceptance.json",
    "estimator acceptance record for the five families and the tuned sixth config", True)
inp("clean", "artifacts/final_five/clean_load_report.json",
    "frozen-model clean load replay on a 2000-row TRAIN fixture", True)
inp("mfreeze", "artifacts/final_five/model_freeze.json",
    "model freeze record (hashes, timestamps, selection flags)", True)
inp("mmeta", "artifacts/final_five/model_metadata.json",
    "frozen model metadata (sampling, weighting, seed, timing, proxy policy)", True)
inp("fschema", "artifacts/final_five/feature_schema.json",
    "frozen 38-feature schema and version", True)
inp("model", "artifacts/final_five/model.joblib",
    "frozen tuned HistGradientBoosting artifact (hash-checked, never refit here)", True, kind="file")
inp("seq", "experiments/final_five/current/sequence_classification.json",
    "observation-level edge F1 and categorical NLL per configuration", True)
inp("reqs", "analysis/final_requirements/requirements_matrix.json",
    "47-row requirement traceability matrix over the five organizer documents", True)
inp("audit", "analysis/data_audit/summary.json",
    "full-dataset structural audit (splits, arcs, nodes, tracks)", True)
inp("pertrip", "analysis/data_audit/per_trip.json",
    "per-trip audit details incl. the trip 00000076 broken reference join", True)
inp("gnbdiag", "reports/final_five/gnb_probability_diagnostic.json",
    "GaussianNB float32 probability diagnostic that stopped the first attempt", True)
inp("figman_eda", "visualizations/figure_manifest.json",
    "EDA figure manifest (hash binding for the 12 reused exploratory figures)", True)
inp("protocol", "experiments/final_five/PROTOCOL.md",
    "frozen five-family protocol (partitions, HPO reuse rules, KNN bank)", True, kind="file")
inp("reqdoc", "docs/FINAL_REQUIREMENTS.md",
    "requirement audit narrative accompanying the matrix", True, kind="file")
inp("provsumm", "analysis/final_requirements/provenance_summary.md",
    "provenance summary of the five organizer documents", True, kind="file")
inp("modelcard", "artifacts/final_five/MODEL_CARD.md",
    "model card (prose; report numbers come from JSON, never the card)", True, kind="file")
inp("trainlog", "reports/final_five/training_corrected.log",
    "corrected training log (fresh GNB / HistGB / tuned HistGB fits)", True, kind="file")
inp("histaudit", "analysis/five_model_audit/gnb_histgb_audit.json",
    "historical artifact audit used to keep historical and current numbers apart", False)

# --- future evidence: exact expected paths, fail closed until present --------
inp("dfreeze", "reports/final_five/decoder/freeze.json",
    "final decoder configuration freeze (scale, weights, search objectives)", True, owner="decoder child")
inp("dplan", "reports/final_five/decoder/calibration_plan.json",
    "predeclared decoder calibration plan over TRAIN trips", True, owner="decoder child")
inp("dsearch", "reports/final_five/decoder/search_results.jsonl",
    "per-trip decoder configuration search results", True, kind="file", owner="decoder child")
inp("dablt", "reports/final_five/decoder/student_ablation.json",
    "raw classifier vs classifier+HMM ablation on the 10 student trips", True, owner="decoder child")
inp("dsums", "reports/final_five/decoder/student_route_checksums.json",
    "checksums binding delivered student routes to the frozen decoder", True, owner="decoder child")
inp("mconfig5", "artifacts/final_five/match_config.json",
    "final deployed match configuration consumed by the API", True, owner="decoder child")
inp("offlinev", "reports/final_five/offline_delivery_verification.json",
    "offline CLI delivery verification for the final freeze", True, owner="verification")
inp("apiv", "reports/final_five/api_integration_acceptance.json",
    "API integration acceptance against the offline delivery", True, owner="verification")
inp("deliveryv", "reports/final_five/delivery_verification.json",
    "consolidated delivery verification (offline vs API parity)", True, owner="verification")
inp("liveapi", "reports/final_five/live/live_api.json",
    "live HTTPS evidence: 10 student + 10 evaluator parity matches", True, owner="live")
inp("browser", "reports/final_five/live/browser_evidence.json",
    "real Chromium UI evidence (DOM, console, network, screenshots)", True, owner="live")
inp("termv", "reports/final_five/live/api_terminal_evidence.json",
    "terminal journal evidence with OCR and request binding", True, owner="live")
inp("evalacc", "reports/final_five/evaluator/acceptance.json",
    "evaluator partition acceptance (post-freeze, never used for selection)", True, owner="evaluator")
inp("evalmetrics", "reports/final_five/evaluator/route_metrics.json",
    "evaluator route metrics from the frozen configuration", True, owner="evaluator")
inp("evalaudit", "reports/final_five/evaluator/input_audit.json",
    "evaluator input audit (schema/range checks vs development data)", True, owner="evaluator")
inp("indaudit", "analysis/final_model_review/final_review_report.json",
    "independent audit of the current corrected results (strict, no refits)", True, owner="independent audit")
inp("testsjson", "reports/final_five/test_results.json",
    "final unit test run record (actual count, never hardcoded)", True, owner="lead")
inp("figman5", "visualizations/final_five/figure_manifest.json",
    "final figure manifest (hash binding for the six-configuration figures)", True, owner="figures child")


# Supplemental authored-manuscript facts; all are required measured evidence.
inp("historical_results", "experiments/results.json", "Historical RF and ExtraTrees comparison only, not final family selection", True, kind="json")
inp("historical_phase3", "experiments/phase_three/results.json", "Historical tuned Bagging appendix only, not final-five selection", True, kind="json")
inp("map_metadata", "visualizations/final_five/maps/final_development_trip_00000009.metadata.json", "Actual final tuned-model development Folium map provenance", True, kind="json")
inp("evaltestqa", "reports/final_five/evaluator/test_and_privacy_verification.json", "Actual offline evaluator synthetic guards and privacy QA", True, kind="json")
inp("mapman", "visualizations/final_five/map_manifest.json", "Final development-only Folium HTML and rendered-asset hash manifest", True, kind="json")
inp("map_browser", "visualizations/final_five/maps/final_development_trip_00000009.browser.json", "Actual offline final Folium browser verification", True, kind="json")
inp("map_html", "visualizations/final_five/maps/final_development_trip_00000009.html", "Actual final development-only interactive Folium map", True, kind="file")
inp("api20", "reports/final_five/api_integration_acceptance.json", "Actual ASGI twenty-trip evaluator metrics for manuscript", True)
inp("selection5", "experiments/final_five/current/selection.json", "TRAIN-only five-family selection record", True, kind="json")
inp("bayes5", "experiments/final_five/current/bayesian_optimization.json", "Current revalidated historical Bayesian study", True, kind="json")
inp("bayeslog", "experiments/bayesian_trials.jsonl", "Twelve actual historical Bayesian trial records", True, kind="file")

inp("report_crops", "reports/final_five/report_images/manifest.json", "Twelve lossless readable screenshot details with original-evidence provenance", True)
inp("crop_script", "scripts/prepare_report_images.py", "Deterministic report-image preparation and pixel verification", True, kind="file")

INPUTS_BY_KEY = {e["key"]: e for e in INPUTS}

# figure paths referenced by the template. EDA figures are hash-bound through
# figman_eda; final figures through figman5; live screenshots through the live
# evidence JSONs themselves. Names for final_five figures follow the existing
# visualizations/plots conventions and are confirmed against the figures
# child's manifest before --final passes.
FIGURE_REGISTRY = {
    "fig_eda_inventory": "visualizations/plots/dataset_inventory.png",
    "fig_eda_quality": "visualizations/plots/missing_invalid_and_anomalies.png",
    "fig_eda_trajdist": "visualizations/plots/trajectory_distributions.png",
    "fig_eda_timestamps": "visualizations/plots/timestamp_gaps.png",
    "fig_eda_network": "visualizations/plots/road_network_sizes_connectivity.png",
    "fig_eda_coverage": "visualizations/plots/gps_geographic_coverage.png",
    "fig_eda_gallery1": "visualizations/plots/gps_trip_gallery_train_01.png",
    "fig_eda_gallery2": "visualizations/plots/gps_trip_gallery_train_02.png",
    "fig_eda_gallery3": "visualizations/plots/gps_trip_gallery_train_03.png",
    "fig_eda_gallery4": "visualizations/plots/gps_trip_gallery_train_04.png",
    "fig_eda_gallery5": "visualizations/plots/gps_trip_gallery_student_test_01.png",
    "fig_eda_route": "visualizations/plots/representative_road_route.png",
    "fig_cv_top1": "visualizations/final_five/plots/final_five_primary_objective.png",
    "fig_cv_f1": "visualizations/final_five/plots/final_five_cv_macro_f1.png",
    "fig_logloss": "visualizations/final_five/plots/final_five_binary_log_loss.png",
    "fig_logloss_parts": "visualizations/final_five/plots/final_five_log_loss_partitions.png",
    "fig_student": "visualizations/final_five/plots/final_five_student_top1_macro_f1.png",
    "fig_selection": "visualizations/final_five/plots/final_five_selection_comparison.png",
    "fig_confusion_lr": "visualizations/final_five/plots/final_five_confusion_logistic_regression.png",
    "fig_confusion_knn": "visualizations/final_five/plots/final_five_confusion_knn_distance_weighted.png",
    "fig_confusion_bag": "visualizations/final_five/plots/final_five_confusion_bagging.png",
    "fig_confusion_gnb": "visualizations/final_five/plots/final_five_confusion_gaussian_nb.png",
    "fig_confusion_hgb": "visualizations/final_five/plots/final_five_confusion_hist_gradient_boosting.png",
    "fig_confusion_tuned": "visualizations/final_five/plots/final_five_confusion_hist_gradient_boosting_tuned.png",
    "fig_confusion_all": "visualizations/final_five/plots/final_five_confusion_matrices_all.png",
    "fig_fold_stability": "visualizations/final_five/plots/final_five_fold_stability.png",
    "fig_gen_gap": "visualizations/final_five/plots/final_five_generalization_gap.png",
    "fig_times": "visualizations/final_five/plots/final_five_timing.png",
    "fig_hpo": "visualizations/final_five/plots/final_five_hpo_progress.png",
    "fig_hpo_params": "visualizations/final_five/plots/final_five_hpo_parameters.png",
    "fig_tuned_default": "visualizations/final_five/plots/final_five_tuned_vs_untuned.png",
    "fig_obs_metrics": "visualizations/final_five/plots/final_five_observation_edge_metrics.png",
    "fig_continuity": "visualizations/final_five/plots/final_five_continuity.png",
    "fig_denominators": "visualizations/final_five/plots/final_five_denominators.png",
    "fig_route_abl": "visualizations/final_five/plots/final_five_route_ablation.png",
    "shot_desktop": "reports/final_five/live/screenshots/ui_desktop.png",
    "shot_route_map": "reports/final_five/live/screenshots/ui_route_map.png",
    "shot_mobile": "reports/final_five/live/screenshots/ui_mobile.png",
    "shot_terminal": "reports/final_five/live/screenshots/api_terminal.png",
}
FIGURE_REGISTRY.update({
    "fig_map_final": "visualizations/final_five/maps/final_development_trip_00000009.png",
    "shot_light": "reports/final_five/live/screenshots/ui_light.png",
    "shot_evaluator": "reports/final_five/live/screenshots/ui_evaluator.png",
    "shot_mobile_320": "reports/final_five/live/screenshots/ui_mobile_320.png",
})
REPORT_CROP_NAMES = (
    "development_dataset_controls", "development_selected_run", "development_scores",
    "development_execution", "evaluator_dataset_controls", "evaluator_scores",
    "light_scores", "mobile_390_selected_run", "mobile_390_scores",
    "mobile_320_scores", "terminal_identity", "terminal_completion",
)
FIGURE_REGISTRY.update({"detail_" + name: "reports/final_five/report_images/" + name + ".png"
                        for name in REPORT_CROP_NAMES})
REGISTRY_BY_PATH = {v: k for k, v in FIGURE_REGISTRY.items()}


# ---------------------------------------------------------------------------
# JSON digging
# ---------------------------------------------------------------------------

def dig(data, pointer):
    """Resolve a dotted pointer against parsed JSON.

    Handles three lookup shapes found in this project's evidence files:

    * plain nested dictionaries: a.b.c;
    * flat dictionaries whose keys already contain dots (the data audit
      summary: "arcs.unique_directed_pairs"). Longest key match wins.
    * list-row selectors: key=value.rest.of.pointer - the first list item
      whose key equals value is descended into.

    "len:<tail>" returns the length of the list/dict/str at the tail instead
    of the value itself.
    """
    if pointer.startswith("len:"):
        cur = dig(data, pointer[4:])
        if isinstance(cur, (list, dict, str)):
            return len(cur)
        raise KeyError("len: target is not a list/dict/str")
    parts = pointer.split(".")
    i = 0
    cur = data
    while i < len(parts):
        part = parts[i]
        if isinstance(cur, dict):
            if part in cur:
                cur = cur[part]
                i += 1
                continue
            # greedy longest dotted-key match for flat dictionaries
            matched = False
            for j in range(len(parts), i, -1):
                cand = ".".join(parts[i:j])
                if cand in cur:
                    cur = cur[cand]
                    i = j
                    matched = True
                    break
            if matched:
                continue
            raise KeyError("key %r not found (have %d keys)" % (part, len(cur)))
        if isinstance(cur, list):
            if "=" in part:
                key, _, val = part.partition("=")
                for item in cur:
                    if isinstance(item, dict) and str(item.get(key)) == val:
                        cur = item
                        i += 1
                        break
                else:
                    raise KeyError("no list item with %s=%s" % (key, val))
                continue
            raise KeyError("list encountered at %r without key=value selector" % part)
        raise KeyError("cannot descend %r into %s" % (part, type(cur).__name__))
    return cur


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# fact registry
# ---------------------------------------------------------------------------

FACTS = []


def fact(name, file, pick, fmt="{:}", row=None, optional=False):
    FACTS.append({
        "name": name, "file": file, "pick": pick, "fmt": fmt,
        "row": row, "optional": bool(optional),
    })


def count(name, file, part, value, in_=None, not_in=None, row=None, fmt="{:,d}"):
    FACTS.append({
        "name": name, "file": file, "count": {"part": part, "value": value,
                                              "in": in_, "not_in": not_in},
        "fmt": fmt, "row": row, "optional": False,
    })


def computed(name, left, op, right, fmt="{:}", row=None):
    FACTS.append({
        "name": name, "computed": {"left": left, "op": op, "right": right},
        "fmt": fmt, "row": row, "optional": False,
    })


# --- identities -------------------------------------------------------------
fact("phase_completed", "results5", "completed_at_utc")
fact("final_run_name", "results5", "final_run_name")
fact("sel_model_run", "results5", "selected_model")
fact("families_json", "results5", "families", fmt="json")
fact("student_role", "results5", "student_role")
fact("evaluator_status", "results5", "evaluator_status")
fact("cv_protocol", "results5", "bayesian.objective")

# --- per-configuration facts (five families + tuned) ------------------------
for fam, short in [("logistic_regression", "lr"), ("knn_distance_weighted", "knn"),
                   ("bagging", "bag"), ("gaussian_nb", "gnb"),
                   ("hist_gradient_boosting", "hgb")]:
    base = fam
    fact("cv_obj_" + short, "results5", "cv.%s.oof_metrics.trip_macro_top1_accuracy" % base,
         fmt="{:.4f}", row="| %s |" % fam)
    fact("cv_f1_" + short, "results5", "cv.%s.oof_metrics.macro_f1" % base,
         fmt="{:.4f}", row="| %s |" % fam)
    fact("cv_ll_" + short, "results5", "cv.%s.oof_metrics.log_loss" % base,
         fmt="{:.4f}", row="| %s |" % fam)
    fact("params_" + short, "results5", "cv.%s.parameters" % base, fmt="json", row="| %s |" % fam)
    fact("st_top1_" + short, "results5", "evaluations.%s.student_metrics.trip_macro_top1_accuracy" % base,
         fmt="{:.4f}", row="| %s |" % fam)
    fact("st_f1_" + short, "results5", "evaluations.%s.student_metrics.macro_f1" % base,
         fmt="{:.4f}", row="| %s |" % fam)
    fact("st_ll_" + short, "results5", "evaluations.%s.student_metrics.log_loss" % base,
         fmt="{:.4f}", row="| %s |" % fam)
    fact("seq_edge_" + short, "seq", "%s.trip_macro_observation_edge_f1" % base,
         fmt="{:.4f}", row="| %s |" % fam)
    fact("seq_nll_" + short, "seq", "%s.observation_categorical_log_loss_given_candidate" % base,
         fmt="{:.4f}", row="| %s |" % fam)
    fact("sha_" + short, "accept", "all_models.%s.sha256" % base, row="| %s |" % fam)

# tuned sixth configuration lives under its own run name
fact("cv_obj_tuned", "results5", "cv.%s.oof_metrics.trip_macro_top1_accuracy" % TUNED_RUN,
     fmt="{:.4f}", row="| %s |" % TUNED_RUN)
fact("cv_f1_tuned", "results5", "cv.%s.oof_metrics.macro_f1" % TUNED_RUN,
     fmt="{:.4f}", row="| %s |" % TUNED_RUN)
fact("cv_ll_tuned", "results5", "cv.%s.oof_metrics.log_loss" % TUNED_RUN,
     fmt="{:.4f}", row="| %s |" % TUNED_RUN)
fact("params_tuned", "results5", "cv.%s.parameters" % TUNED_RUN, fmt="json", row="| %s |" % TUNED_RUN)
fact("st_top1_tuned", "results5", "evaluations.%s.student_metrics.trip_macro_top1_accuracy" % TUNED_RUN,
     fmt="{:.4f}", row="| %s |" % TUNED_RUN)
fact("st_f1_tuned", "results5", "evaluations.%s.student_metrics.macro_f1" % TUNED_RUN,
     fmt="{:.4f}", row="| %s |" % TUNED_RUN)
fact("st_ll_tuned", "results5", "evaluations.%s.student_metrics.log_loss" % TUNED_RUN,
     fmt="{:.4f}", row="| %s |" % TUNED_RUN)
fact("seq_edge_tuned", "seq", "%s.trip_macro_observation_edge_f1" % TUNED_RUN,
     fmt="{:.4f}", row="| %s |" % TUNED_RUN)
fact("seq_nll_tuned", "seq", "%s.observation_categorical_log_loss_given_candidate" % TUNED_RUN,
     fmt="{:.4f}", row="| %s |" % TUNED_RUN)
fact("sha_tuned", "accept", "all_models.%s.sha256" % TUNED_RUN, row="| %s |" % TUNED_RUN)

# --- cross-config fold statistics and timing --------------------------------
fact("cv_t1m_hgb", "results5", "cv.hist_gradient_boosting.top1_fold_mean", fmt="{:.4f}")
fact("cv_t1s_hgb", "results5", "cv.hist_gradient_boosting.top1_fold_std", fmt="{:.4f}")
fact("cv_t1m_tuned", "results5", "cv.%s.top1_fold_mean" % TUNED_RUN, fmt="{:.4f}")
fact("cv_t1s_tuned", "results5", "cv.%s.top1_fold_std" % TUNED_RUN, fmt="{:.4f}")
fact("cv_sec_tuned", "results5", "cv.%s.training_seconds_total" % TUNED_RUN, fmt="{:.1f}")
fact("cv_sec_bag", "results5", "cv.bagging.training_seconds_total", fmt="{:.1f}")
fact("cv_sec_knn_inf", "results5", "cv.knn_distance_weighted.inference_seconds_total", fmt="{:.1f}")
fact("st_prec_tuned", "results5", "evaluations.%s.student_metrics.precision" % TUNED_RUN, fmt="{:.4f}")
fact("st_rec_tuned", "results5", "evaluations.%s.student_metrics.recall" % TUNED_RUN, fmt="{:.4f}")
fact("st_inf_tuned", "results5", "evaluations.%s.student_inference_seconds" % TUNED_RUN, fmt="{:.3f}")

# honest deltas: tuned vs default HistGB on both partitions
computed("cv_gain", "cv_obj_tuned", "-", "cv_obj_hgb", fmt="{:+.4f}")
computed("st_gain", "st_top1_tuned", "-", "st_top1_hgb", fmt="{:+.4f}")

# --- candidate accounting (label taxonomy) ----------------------------------
# TRAIN partition: frozen model metadata is the canonical counter.
fact("train_obs", "mmeta", "training_observations", fmt="{:,d}")
fact("train_rows", "mmeta", "training_candidate_rows", fmt="{:,d}")
fact("train_miss", "mmeta", "metrics.grouped_cv.candidate_miss_count", fmt="{:,d}")
computed("train_present", "train_obs", "-", "train_miss", fmt="{:,d}")
computed("train_miss_rate", "train_miss", "/", "train_obs", fmt="{:.4%}")
# student partition: the sequence-classification recount is the canonical counter.
fact("student_obs", "seq", "logistic_regression.observations", fmt="{:,d}")
fact("student_miss", "seq", "logistic_regression.candidate_miss_count", fmt="{:,d}")
computed("student_present", "student_obs", "-", "student_miss", fmt="{:,d}")
computed("student_miss_rate", "student_miss", "/", "student_obs", fmt="{:.4%}")


# --- student row counts from the acceptance record ---------------------------
fact("student_rows", "accept", "all_models.%s.student_rows" % TUNED_RUN, fmt="{:,d}")

# --- HPO study ---------------------------------------------------------------
fact("hpo_method", "results5", "bayesian.method")
fact("hpo_space", "results5", "bayesian.search_space", fmt="json")
fact("hpo_best_params", "results5", "bayesian.best_parameters", fmt="json")
fact("hpo_trials", "results5", "bayesian.trials_including_measured_prior", fmt="{:,d}")
fact("hpo_seeded", "results5", "bayesian.seeded_initial_design_trials", fmt="{:,d}")
fact("hpo_modelbased", "results5", "bayesian.model_based_trials", fmt="{:,d}")
fact("hpo_fresh_trials", "results5", "bayesian.fresh_hpo_trials", fmt="{:,d}")
fact("hpo_baseline", "results5", "bayesian.baseline_objective", fmt="{:.4f}")
fact("hpo_best", "results5", "bayesian.best_objective", fmt="{:.4f}")
fact("hpo_seconds", "results5", "bayesian.seconds", fmt="{:.1f}")
fact("hpo_reuse_note", "results5", "bayesian.reuse_provenance.note")
computed("hpo_gain", "hpo_best", "-", "hpo_baseline", fmt="{:+.4f}")

# --- model freeze and environment -------------------------------------------
fact("frozen_utc", "results5", "model_freeze.frozen_at_utc")
fact("frozen_bytes", "results5", "model_freeze.bytes", fmt="{:,d}")
fact("fschema_sha", "results5", "model_freeze.feature_schema_sha256")
fact("mmeta_sha", "results5", "model_freeze.metadata_sha256")
fact("ver_python", "results5", "versions.python")
fact("ver_sklearn", "results5", "versions.sklearn")
fact("ver_numpy", "results5", "versions.numpy")
fact("ver_scipy", "results5", "versions.scipy")
fact("ver_joblib", "results5", "versions.joblib")

# --- acceptance and clean-load ------------------------------------------------
fact("accept_fresh", "accept", "fresh_fits", fmt="json")
fact("clean_delta", "clean", "max_probability_delta", fmt="{:g}")
fact("clean_atol", "clean", "atol", fmt="{:g}")

# --- frozen model metadata ----------------------------------------------------
fact("mm_fit_data", "mmeta", "fit_data")
fact("mm_fit_seconds", "mmeta", "timing.fit_all80_seconds", fmt="{:.2f}")
fact("mm_inf_seconds", "mmeta", "timing.train_inference_seconds", fmt="{:.3f}")
fact("mm_train_trips", "mmeta", "training_trip_count", fmt="{:,d}")
fact("mm_seed", "mmeta", "seed", fmt="{:}")
fact("mm_feature_version", "mmeta", "feature_version")

# --- feature schema -----------------------------------------------------------
fact("feat_count", "fschema", "count", fmt="{:,d}")
fact("feat_version", "fschema", "feature_version")

# --- data audit (TRAIN + student_test) ----------------------------------------
fact("aud_trips", "audit", "trip_count", fmt="{:,d}")
fact("aud_files", "audit", "file_count", fmt="{:,d}")
fact("aud_arcs_pairs", "audit", "splits.train.arcs.unique_directed_pairs", fmt="{:,d}")
fact("aud_broken_train", "audit", "splits.train.route_arc_offset_0.broken_joins", fmt="{:,d}")
fact("aud_broken_student", "audit", "splits.student_test.route_arc_offset_0.broken_joins", fmt="{:,d}")
fact("aud_steps_over", "audit", "splits.train.track.steps_over_100mps_if_seconds", fmt="{:,d}")
fact("aud_time_delta", "audit", "splits.train.track.time_delta_not_one", fmt="{:,d}")
fact("aud_dup_nodes", "audit", "splits.train.nodes.duplicate_coordinate_extra_nodes", fmt="{:,d}")
fact("aud_dup_rows", "audit", "splits.train.track.duplicate_coordinate_extra_rows", fmt="{:,d}")
fact("aud_nonpos", "audit", "splits.train.track.nonpositive_time_deltas", fmt="{:,d}")
fact("aud_repeated", "audit", "splits.train.route_validity.repeated_arc_extra_occurrences", fmt="{:,d}")
fact("trip76_broken", "pertrip", "trip_id=00000076.route_validity.arc_index_offset_0.broken_joins", fmt="{:,d}")

# --- requirements matrix --------------------------------------------------------
fact("req_total", "reqs", "len:requirements", fmt="{:,d}")
count("req_done", "reqs", "requirements.status", "done_evidence_exists")
count("req_partial", "reqs", "requirements.status", "partially_done")
count("req_human", "reqs", "requirements.status", "human_external_blocker")
count("req_notdone", "reqs", "requirements.status", "not_yet_done")
count("req_deferred", "reqs", "requirements.status", "deferred_by_user")
count("req_exception", "reqs", "requirements.status", "user_exception")
count("req_inprogress", "reqs", "requirements.status", "in_progress_delegated")
count("req_resolved", "reqs", "requirements.status", None, in_=sorted(REQ_TERMINAL_STATUSES))

# --- GNB probability diagnostic --------------------------------------------------
fact("gnb_affected", "gnbdiag", "affected_rows", fmt="{:,d}")
fact("gnb_outside", "gnbdiag", "outside_bound_entries", fmt="{:,d}")
fact("gnb_rowsum", "gnbdiag", "max_rowsum_error", fmt="{:.2e}")

# --- decoder calibration and search (final freeze) --------------------------------
fact("dec_sigma", "dfreeze", "config.sigma_m", fmt="{:.3f}")
fact("dec_beta", "dfreeze", "config.beta_m", fmt="{:.3f}")
fact("dec_maxcand", "dfreeze", "config.max_candidates", fmt="{:d}")
fact("dec_retry", "dfreeze", "config.candidate_retry_limit", fmt="{:d}")
fact("dec_radius", "dfreeze", "config.candidate_radius_m", fmt="{:.0f}")
fact("dec_slack", "dfreeze", "config.routing_slack_m", fmt="{:.0f}")
fact("dec_cls_w", "dfreeze", "config.classifier_weight", fmt="{:.2f}")
fact("dec_geo_w", "dfreeze", "config.geometry_weight", fmt="{:.2f}")
fact("dec_btol", "dfreeze", "config.backward_tolerance_m", fmt="{:.0f}")
fact("dec_spacing", "dfreeze", "config.sample_spacing_m", fmt="{:.0f}")
fact("dec_search_trips", "dfreeze", "len:calibration_trip_ids", fmt="{:d}")
fact("dec_frozen_utc", "dfreeze", "frozen_utc")
fact("dec_selected", "dfreeze", "selected_configuration")
fact("dec_a_lenf1", "dfreeze", "objectives.configuration=A_fitted.mean_length_f1", fmt="{:.4f}")
fact("dec_b_lenf1", "dfreeze", "objectives.configuration=B_twice_beta.mean_length_f1", fmt="{:.4f}")
fact("dec_c_lenf1", "dfreeze", "objectives.configuration=C_classifier_up.mean_length_f1", fmt="{:.4f}")
fact("dec_a_edgef1", "dfreeze", "objectives.configuration=A_fitted.mean_edge_f1", fmt="{:.4f}")
fact("dec_b_edgef1", "dfreeze", "objectives.configuration=B_twice_beta.mean_edge_f1", fmt="{:.4f}")
fact("dec_c_edgef1", "dfreeze", "objectives.configuration=C_classifier_up.mean_edge_f1", fmt="{:.4f}")
fact("dplan_trips", "dplan", "train_trip_count", fmt="{:,d}")

# --- decoder ablation on the student partition -------------------------------------
fact("abl_trips", "dablt", "trip_count", fmt="{:d}")
fact("abl_failed", "dablt", "len:failed_trip_ids", fmt="{:d}")
fact("abl_obs", "dablt", "total_observations", fmt="{:,d}")
fact("abl_raw_top1", "dablt", "aggregate.classifier_only_raw.trip_macro.proxy_accuracy", fmt="{:.4f}")
fact("abl_raw_edge", "dablt", "aggregate.classifier_only_raw.trip_macro.route_edge_f1", fmt="{:.4f}")
fact("abl_raw_len", "dablt", "aggregate.classifier_only_raw.trip_macro.route_length_f1", fmt="{:.4f}")
fact("abl_hmm_top1", "dablt", "aggregate.classifier_plus_HMM.trip_macro.proxy_accuracy", fmt="{:.4f}")
fact("abl_hmm_edge", "dablt", "aggregate.classifier_plus_HMM.trip_macro.route_edge_f1", fmt="{:.4f}")
fact("abl_hmm_len", "dablt", "aggregate.classifier_plus_HMM.trip_macro.route_length_f1", fmt="{:.4f}")
fact("abl_geo_edge", "dablt", "aggregate.geometry_HMM_default_reused.trip_macro.route_edge_f1", fmt="{:.4f}")
fact("abl_geo_len", "dablt", "aggregate.geometry_HMM_default_reused.trip_macro.route_length_f1", fmt="{:.4f}")
fact("abl_hmm_seconds", "dablt", "aggregate.classifier_plus_HMM.trip_macro.inference_seconds", fmt="{:.2f}")
computed("abl_hmm_edge_gain", "abl_hmm_edge", "-", "abl_raw_edge", fmt="{:+.4f}")

# --- offline CLI delivery verification ----------------------------------------------
fact("off_edge_f1", "offlinev", "trip_macro.edge_f1", fmt="{:.4f}")
fact("off_len_f1", "offlinev", "trip_macro.length_f1", fmt="{:.4f}")
fact("off_edit", "offlinev", "trip_macro.edit_similarity", fmt="{:.4f}")
fact("off_cont", "offlinev", "trip_macro.route_continuity", fmt="{:.4f}")

# --- evaluator (post-freeze reference scoring) ----------------------------------------
fact("eval_trips", "evalacc", "trip_count", fmt="{:d}")
fact("eval_edge_f1", "evalmetrics", "summary.macro.edge_f1", fmt="{:.4f}")
fact("eval_len_f1", "evalmetrics", "summary.macro.length_f1", fmt="{:.4f}")
fact("eval_edit", "evalmetrics", "summary.macro.edit_similarity", fmt="{:.4f}")
fact("eval_cont", "evalmetrics", "summary.macro.route_continuity", fmt="{:.4f}")
fact("eval_top1", "evalmetrics", "summary.macro.top1_accuracy", fmt="{:.4f}")
fact("eval_mf1", "evalmetrics", "summary.macro.macro_f1", fmt="{:.4f}")

# --- independent audit ------------------------------------------------------------------
fact("ind_findings", "indaudit", "len:verified_findings", fmt="{:d}")

# --- live API / browser / terminal evidence (arrives with the live child) ---------------
fact("live_base", "liveapi", "base_url", optional=True)
fact("live_matches", "liveapi", "successful_count", fmt="{:d}", optional=True)
fact("live_requested", "liveapi", "all_twenty_requested", optional=True)
fact("live_health_sha", "liveapi", "health.model_hash", optional=True)
fact("term_valid", "termv", "valid", optional=True)
fact("term_ocr", "termv", "ocr_privacy_guard_passed", optional=True)
fact("browser_errors", "browser", "len:browser_errors", fmt="{:d}", optional=True)
fact("browser_console", "browser", "len:console_errors", fmt="{:d}", optional=True)

# --- final test record (written by the lead's final test run) ---------------------------
fact("tests_total", "testsjson", "total", fmt="{:,d}", optional=True)
fact("tests_passed", "testsjson", "passed", fmt="{:,d}", optional=True)
fact("tests_failed", "testsjson", "len:failures", fmt="{:d}", optional=True)


# Additional family, history, evaluator and live-evidence bindings.
fact("lr_infer_s", "results5", "evaluations.logistic_regression.student_inference_seconds", fmt="{:.3f}", optional=False)
fact("lr_fit_s", "results5", "evaluations.logistic_regression.training_seconds", fmt="{:.3f}", optional=False)
fact("lr_train_top1", "results5", "evaluations.logistic_regression.train_resubstitution_metrics.top1_accuracy", fmt="{:.4f}", optional=False)
fact("lr_train_binary_f1", "results5", "evaluations.logistic_regression.train_resubstitution_metrics.macro_f1", fmt="{:.4f}", optional=False)
fact("lr_train_ll", "results5", "evaluations.logistic_regression.train_resubstitution_metrics.log_loss", fmt="{:.4f}", optional=False)
fact("lr_binary_accuracy", "results5", "evaluations.logistic_regression.student_metrics.accuracy", fmt="{:.4f}", optional=False)
fact("lr_binary_precision", "results5", "evaluations.logistic_regression.student_metrics.precision", fmt="{:.4f}", optional=False)
fact("lr_binary_recall", "results5", "evaluations.logistic_regression.student_metrics.recall", fmt="{:.4f}", optional=False)
fact("lr_pooled_top1", "seq", "logistic_regression.top1_accuracy", fmt="{:.4f}", optional=False)
fact("lr_pooled_edge_f1", "seq", "logistic_regression.observation_edge_macro_f1", fmt="{:.4f}", optional=False)
fact("lr_all_nll", "seq", "logistic_regression.observation_categorical_log_loss_clipped", fmt="{:.4f}", optional=False)
fact("lr_raw_continuity", "seq", "logistic_regression.raw_classifier_continuity", fmt="{:.4f}", optional=False)
fact("lr_change_continuity", "seq", "logistic_regression.edge_change_continuity", fmt="{:.4f}", optional=False)
fact("knn_infer_s", "results5", "evaluations.knn_distance_weighted.student_inference_seconds", fmt="{:.3f}", optional=False)
fact("knn_fit_s", "results5", "evaluations.knn_distance_weighted.training_seconds", fmt="{:.3f}", optional=False)
fact("knn_train_top1", "results5", "evaluations.knn_distance_weighted.train_resubstitution_metrics.top1_accuracy", fmt="{:.4f}", optional=False)
fact("knn_train_binary_f1", "results5", "evaluations.knn_distance_weighted.train_resubstitution_metrics.macro_f1", fmt="{:.4f}", optional=False)
fact("knn_train_ll", "results5", "evaluations.knn_distance_weighted.train_resubstitution_metrics.log_loss", fmt="{:.4f}", optional=False)
fact("knn_binary_accuracy", "results5", "evaluations.knn_distance_weighted.student_metrics.accuracy", fmt="{:.4f}", optional=False)
fact("knn_binary_precision", "results5", "evaluations.knn_distance_weighted.student_metrics.precision", fmt="{:.4f}", optional=False)
fact("knn_binary_recall", "results5", "evaluations.knn_distance_weighted.student_metrics.recall", fmt="{:.4f}", optional=False)
fact("knn_pooled_top1", "seq", "knn_distance_weighted.top1_accuracy", fmt="{:.4f}", optional=False)
fact("knn_pooled_edge_f1", "seq", "knn_distance_weighted.observation_edge_macro_f1", fmt="{:.4f}", optional=False)
fact("knn_all_nll", "seq", "knn_distance_weighted.observation_categorical_log_loss_clipped", fmt="{:.4f}", optional=False)
fact("knn_raw_continuity", "seq", "knn_distance_weighted.raw_classifier_continuity", fmt="{:.4f}", optional=False)
fact("knn_change_continuity", "seq", "knn_distance_weighted.edge_change_continuity", fmt="{:.4f}", optional=False)
fact("bag_infer_s", "results5", "evaluations.bagging.student_inference_seconds", fmt="{:.3f}", optional=False)
fact("bag_fit_s", "results5", "evaluations.bagging.training_seconds", fmt="{:.3f}", optional=False)
fact("bag_train_top1", "results5", "evaluations.bagging.train_resubstitution_metrics.top1_accuracy", fmt="{:.4f}", optional=False)
fact("bag_train_binary_f1", "results5", "evaluations.bagging.train_resubstitution_metrics.macro_f1", fmt="{:.4f}", optional=False)
fact("bag_train_ll", "results5", "evaluations.bagging.train_resubstitution_metrics.log_loss", fmt="{:.4f}", optional=False)
fact("bag_binary_accuracy", "results5", "evaluations.bagging.student_metrics.accuracy", fmt="{:.4f}", optional=False)
fact("bag_binary_precision", "results5", "evaluations.bagging.student_metrics.precision", fmt="{:.4f}", optional=False)
fact("bag_binary_recall", "results5", "evaluations.bagging.student_metrics.recall", fmt="{:.4f}", optional=False)
fact("bag_pooled_top1", "seq", "bagging.top1_accuracy", fmt="{:.4f}", optional=False)
fact("bag_pooled_edge_f1", "seq", "bagging.observation_edge_macro_f1", fmt="{:.4f}", optional=False)
fact("bag_all_nll", "seq", "bagging.observation_categorical_log_loss_clipped", fmt="{:.4f}", optional=False)
fact("bag_raw_continuity", "seq", "bagging.raw_classifier_continuity", fmt="{:.4f}", optional=False)
fact("bag_change_continuity", "seq", "bagging.edge_change_continuity", fmt="{:.4f}", optional=False)
fact("gnb_infer_s", "results5", "evaluations.gaussian_nb.student_inference_seconds", fmt="{:.3f}", optional=False)
fact("gnb_fit_s", "results5", "evaluations.gaussian_nb.training_seconds", fmt="{:.3f}", optional=False)
fact("gnb_train_top1", "results5", "evaluations.gaussian_nb.train_resubstitution_metrics.top1_accuracy", fmt="{:.4f}", optional=False)
fact("gnb_train_binary_f1", "results5", "evaluations.gaussian_nb.train_resubstitution_metrics.macro_f1", fmt="{:.4f}", optional=False)
fact("gnb_train_ll", "results5", "evaluations.gaussian_nb.train_resubstitution_metrics.log_loss", fmt="{:.4f}", optional=False)
fact("gnb_binary_accuracy", "results5", "evaluations.gaussian_nb.student_metrics.accuracy", fmt="{:.4f}", optional=False)
fact("gnb_binary_precision", "results5", "evaluations.gaussian_nb.student_metrics.precision", fmt="{:.4f}", optional=False)
fact("gnb_binary_recall", "results5", "evaluations.gaussian_nb.student_metrics.recall", fmt="{:.4f}", optional=False)
fact("gnb_pooled_top1", "seq", "gaussian_nb.top1_accuracy", fmt="{:.4f}", optional=False)
fact("gnb_pooled_edge_f1", "seq", "gaussian_nb.observation_edge_macro_f1", fmt="{:.4f}", optional=False)
fact("gnb_all_nll", "seq", "gaussian_nb.observation_categorical_log_loss_clipped", fmt="{:.4f}", optional=False)
fact("gnb_raw_continuity", "seq", "gaussian_nb.raw_classifier_continuity", fmt="{:.4f}", optional=False)
fact("gnb_change_continuity", "seq", "gaussian_nb.edge_change_continuity", fmt="{:.4f}", optional=False)
fact("hgb_infer_s", "results5", "evaluations.hist_gradient_boosting.student_inference_seconds", fmt="{:.3f}", optional=False)
fact("hgb_fit_s", "results5", "evaluations.hist_gradient_boosting.training_seconds", fmt="{:.3f}", optional=False)
fact("hgb_train_top1", "results5", "evaluations.hist_gradient_boosting.train_resubstitution_metrics.top1_accuracy", fmt="{:.4f}", optional=False)
fact("hgb_train_binary_f1", "results5", "evaluations.hist_gradient_boosting.train_resubstitution_metrics.macro_f1", fmt="{:.4f}", optional=False)
fact("hgb_train_ll", "results5", "evaluations.hist_gradient_boosting.train_resubstitution_metrics.log_loss", fmt="{:.4f}", optional=False)
fact("hgb_binary_accuracy", "results5", "evaluations.hist_gradient_boosting.student_metrics.accuracy", fmt="{:.4f}", optional=False)
fact("hgb_binary_precision", "results5", "evaluations.hist_gradient_boosting.student_metrics.precision", fmt="{:.4f}", optional=False)
fact("hgb_binary_recall", "results5", "evaluations.hist_gradient_boosting.student_metrics.recall", fmt="{:.4f}", optional=False)
fact("hgb_pooled_top1", "seq", "hist_gradient_boosting.top1_accuracy", fmt="{:.4f}", optional=False)
fact("hgb_pooled_edge_f1", "seq", "hist_gradient_boosting.observation_edge_macro_f1", fmt="{:.4f}", optional=False)
fact("hgb_all_nll", "seq", "hist_gradient_boosting.observation_categorical_log_loss_clipped", fmt="{:.4f}", optional=False)
fact("hgb_raw_continuity", "seq", "hist_gradient_boosting.raw_classifier_continuity", fmt="{:.4f}", optional=False)
fact("hgb_change_continuity", "seq", "hist_gradient_boosting.edge_change_continuity", fmt="{:.4f}", optional=False)
fact("tuned_infer_s", "results5", "evaluations.hist_gradient_boosting_tuned.student_inference_seconds", fmt="{:.3f}", optional=False)
fact("tuned_fit_s", "results5", "evaluations.hist_gradient_boosting_tuned.training_seconds", fmt="{:.3f}", optional=False)
fact("tuned_train_top1", "results5", "evaluations.hist_gradient_boosting_tuned.train_resubstitution_metrics.top1_accuracy", fmt="{:.4f}", optional=False)
fact("tuned_train_binary_f1", "results5", "evaluations.hist_gradient_boosting_tuned.train_resubstitution_metrics.macro_f1", fmt="{:.4f}", optional=False)
fact("tuned_train_ll", "results5", "evaluations.hist_gradient_boosting_tuned.train_resubstitution_metrics.log_loss", fmt="{:.4f}", optional=False)
fact("tuned_binary_accuracy", "results5", "evaluations.hist_gradient_boosting_tuned.student_metrics.accuracy", fmt="{:.4f}", optional=False)
fact("tuned_binary_precision", "results5", "evaluations.hist_gradient_boosting_tuned.student_metrics.precision", fmt="{:.4f}", optional=False)
fact("tuned_binary_recall", "results5", "evaluations.hist_gradient_boosting_tuned.student_metrics.recall", fmt="{:.4f}", optional=False)
fact("tuned_pooled_top1", "seq", "hist_gradient_boosting_tuned.top1_accuracy", fmt="{:.4f}", optional=False)
fact("tuned_pooled_edge_f1", "seq", "hist_gradient_boosting_tuned.observation_edge_macro_f1", fmt="{:.4f}", optional=False)
fact("tuned_all_nll", "seq", "hist_gradient_boosting_tuned.observation_categorical_log_loss_clipped", fmt="{:.4f}", optional=False)
fact("tuned_raw_continuity", "seq", "hist_gradient_boosting_tuned.raw_classifier_continuity", fmt="{:.4f}", optional=False)
fact("tuned_change_continuity", "seq", "hist_gradient_boosting_tuned.edge_change_continuity", fmt="{:.4f}", optional=False)
fact("historical_rf_params", "historical_results", "cv.random_forest.parameters", fmt="json", optional=False)
fact("historical_rf_cv_f1", "historical_results", "cv.random_forest.oof_metrics.macro_f1", fmt="{:.4f}", optional=False)
fact("historical_rf_cv_obj", "historical_results", "cv.random_forest.oof_metrics.trip_macro_top1_accuracy", fmt="{:.4f}", optional=False)
fact("historical_rf_student_f1", "historical_results", "evaluations.random_forest.student_metrics.macro_f1", fmt="{:.4f}", optional=False)
fact("historical_rf_student_top1", "historical_results", "evaluations.random_forest.student_metrics.top1_accuracy", fmt="{:.4f}", optional=False)
fact("historical_rf_student_trip_top1", "historical_results", "evaluations.random_forest.student_metrics.trip_macro_top1_accuracy", fmt="{:.4f}", optional=False)
fact("historical_rf_student_ll", "historical_results", "evaluations.random_forest.student_metrics.log_loss", fmt="{:.4f}", optional=False)
fact("historical_rf_infer_s", "historical_results", "evaluations.random_forest.student_inference_seconds", fmt="{:.3f}", optional=False)
fact("historical_rf_fit_s", "historical_results", "evaluations.random_forest.training_seconds", fmt="{:.3f}", optional=False)
fact("historical_et_params", "historical_results", "cv.extra_trees.parameters", fmt="json", optional=False)
fact("historical_et_cv_f1", "historical_results", "cv.extra_trees.oof_metrics.macro_f1", fmt="{:.4f}", optional=False)
fact("historical_et_cv_obj", "historical_results", "cv.extra_trees.oof_metrics.trip_macro_top1_accuracy", fmt="{:.4f}", optional=False)
fact("historical_et_student_f1", "historical_results", "evaluations.extra_trees.student_metrics.macro_f1", fmt="{:.4f}", optional=False)
fact("historical_et_student_top1", "historical_results", "evaluations.extra_trees.student_metrics.top1_accuracy", fmt="{:.4f}", optional=False)
fact("historical_et_student_trip_top1", "historical_results", "evaluations.extra_trees.student_metrics.trip_macro_top1_accuracy", fmt="{:.4f}", optional=False)
fact("historical_et_student_ll", "historical_results", "evaluations.extra_trees.student_metrics.log_loss", fmt="{:.4f}", optional=False)
fact("historical_et_infer_s", "historical_results", "evaluations.extra_trees.student_inference_seconds", fmt="{:.3f}", optional=False)
fact("historical_et_fit_s", "historical_results", "evaluations.extra_trees.training_seconds", fmt="{:.3f}", optional=False)
fact("historical_bt_params", "historical_phase3", "cv.bagging_tuned.parameters", fmt="json", optional=False)
fact("historical_bt_cv_f1", "historical_phase3", "cv.bagging_tuned.oof_metrics.macro_f1", fmt="{:.4f}", optional=False)
fact("historical_bt_cv_obj", "historical_phase3", "cv.bagging_tuned.oof_metrics.trip_macro_top1_accuracy", fmt="{:.4f}", optional=False)
fact("historical_bt_student_f1", "historical_phase3", "evaluations.bagging_tuned.student_metrics.macro_f1", fmt="{:.4f}", optional=False)
fact("historical_bt_student_top1", "historical_phase3", "evaluations.bagging_tuned.student_metrics.top1_accuracy", fmt="{:.4f}", optional=False)
fact("historical_bt_student_trip_top1", "historical_phase3", "evaluations.bagging_tuned.student_metrics.trip_macro_top1_accuracy", fmt="{:.4f}", optional=False)
fact("historical_bt_student_ll", "historical_phase3", "evaluations.bagging_tuned.student_metrics.log_loss", fmt="{:.4f}", optional=False)
fact("historical_bt_infer_s", "historical_phase3", "evaluations.bagging_tuned.student_inference_seconds", fmt="{:.3f}", optional=False)
fact("historical_bt_fit_s", "historical_phase3", "evaluations.bagging_tuned.training_seconds", fmt="{:.3f}", optional=False)
fact("train_cv_folds", "results5", "len:cv.logistic_regression.folds", fmt="{:}", optional=False)
fact("evaluator_first_open", "evalacc", "chronology.first_input_open_started_utc", fmt="{:}", optional=False)
fact("evaluator_reference_start", "evalacc", "chronology.reference_scoring_started_at_utc", fmt="{:}", optional=False)
fact("evaluator_predictions_completed", "evalacc", "chronology.prediction_completed_at_utc", fmt="{:}", optional=False)
fact("evaluator_obs", "evalmetrics", "summary.full_observation_proxy.observations_scored", fmt="{:,d}", optional=False)
fact("evaluator_exact", "evalmetrics", "summary.exact_route_count", fmt="{:}", optional=False)
fact("evaluator_refbreaks", "evalmetrics", "summary.reference_discontinuities", fmt="{:}", optional=False)
fact("evaluator_decoded_accuracy", "evalmetrics", "summary.full_observation_proxy.decoded.accuracy", fmt="{:.4f}", optional=False)
fact("evaluator_decoded_f1", "evalmetrics", "summary.full_observation_proxy.decoded.macro_f1", fmt="{:.4f}", optional=False)
fact("evaluator_classifier_accuracy", "evalmetrics", "summary.full_observation_proxy.classifier_only.accuracy", fmt="{:.4f}", optional=False)
fact("evaluator_classifier_f1", "evalmetrics", "summary.full_observation_proxy.classifier_only.macro_f1", fmt="{:.4f}", optional=False)
fact("evaluator_weak15_edge", "evalacc", "expected_trips.00000015.metrics.edge_f1", fmt="{:.4f}", optional=False)
fact("evaluator_weak15_length", "evalacc", "expected_trips.00000015.metrics.length_f1", fmt="{:.4f}", optional=False)
fact("evaluator_zip_sha", "evalacc", "submission_zip_sha256", fmt="{:}", optional=False)
fact("overlap_exact_duplicate", "evalaudit", "overlap_report.any_full_trip_duplicate", fmt="{:}", optional=False)
fact("overlap_any_file", "evalaudit", "overlap_report.any_identical_input_file", fmt="{:}", optional=False)
fact("overlap_id_matches", "evalaudit", "overlap_report.id_matches", fmt="json", optional=False)
fact("overlap_train_count", "evalaudit", "overlap_report.training_trip_count_checked", fmt="{:}", optional=False)
fact("overlap_student_count", "evalaudit", "overlap_report.student_trip_count_checked", fmt="{:}", optional=False)
fact("evaluator_guard_tests", "evaltestqa", "tests_run", fmt="{:}", optional=False)
fact("map_points", "map_metadata", "gps_points", fmt="{:}", optional=False)
fact("map_pred_arcs", "map_metadata", "prediction_arc_geometry_count", fmt="{:}", optional=False)
fact("map_ref_arcs", "map_metadata", "reference_route_integrity.arc_rows", fmt="{:}", optional=False)
fact("map_edge_f1", "map_metadata", "route_metrics_recomputed_equal_delivery.edge_f1", fmt="{:.4f}", optional=False)
fact("map_length_f1", "map_metadata", "route_metrics_recomputed_equal_delivery.length_f1", fmt="{:.4f}", optional=False)
fact("map_folium_version", "map_metadata", "render.folium_version", fmt="{:}", optional=False)
fact("terminal_request_id", "termv", "request_id", fmt="{:}", optional=False)
fact("terminal_start_utc", "termv", "events.event=match_started.timestamp", fmt="{:}", optional=False)
fact("terminal_finish_utc", "termv", "events.event=match_completed.timestamp", fmt="{:}", optional=False)
fact("terminal_dataset", "termv", "events.event=match_started.dataset_role", fmt="{:}", optional=False)
fact("terminal_http_request_id", "termv", "events.event=http_request.request_id", fmt="{:}", optional=False)
fact("browser_live_request_id", "browser", "responses.dataset_id=development.request_id", fmt="{:}", optional=False)
fact("browser_evaluator_request_id", "browser", "responses.dataset_id=evaluator.request_id", fmt="{:}", optional=False)
fact("browser_evaluator_accuracy", "browser", "responses.dataset_id=evaluator.metrics.accuracy", fmt="{:.4f}", optional=False)
fact("api_eval_15_nll", "api20", "datasets.evaluator.trips.00000015.metrics.log_loss", fmt="{:.6f}", optional=False)
fact("api_eval_15_miss", "api20", "datasets.evaluator.trips.00000015.metrics.candidate_miss_rate", fmt="{:.6f}", optional=False)
fact("api_eval_17_nll", "api20", "datasets.evaluator.trips.00000017.metrics.log_loss", fmt="{:.6f}", optional=False)
fact("api_eval_17_miss", "api20", "datasets.evaluator.trips.00000017.metrics.candidate_miss_rate", fmt="{:.6f}", optional=False)
fact("api_eval_23_nll", "api20", "datasets.evaluator.trips.00000023.metrics.log_loss", fmt="{:.6f}", optional=False)
fact("api_eval_23_miss", "api20", "datasets.evaluator.trips.00000023.metrics.candidate_miss_rate", fmt="{:.6f}", optional=False)
fact("api_eval_30_nll", "api20", "datasets.evaluator.trips.00000030.metrics.log_loss", fmt="{:.6f}", optional=False)
fact("api_eval_30_miss", "api20", "datasets.evaluator.trips.00000030.metrics.candidate_miss_rate", fmt="{:.6f}", optional=False)
fact("api_eval_49_nll", "api20", "datasets.evaluator.trips.00000049.metrics.log_loss", fmt="{:.6f}", optional=False)
fact("api_eval_49_miss", "api20", "datasets.evaluator.trips.00000049.metrics.candidate_miss_rate", fmt="{:.6f}", optional=False)
fact("api_eval_59_nll", "api20", "datasets.evaluator.trips.00000059.metrics.log_loss", fmt="{:.6f}", optional=False)
fact("api_eval_59_miss", "api20", "datasets.evaluator.trips.00000059.metrics.candidate_miss_rate", fmt="{:.6f}", optional=False)
fact("api_eval_67_nll", "api20", "datasets.evaluator.trips.00000067.metrics.log_loss", fmt="{:.6f}", optional=False)
fact("api_eval_67_miss", "api20", "datasets.evaluator.trips.00000067.metrics.candidate_miss_rate", fmt="{:.6f}", optional=False)
fact("api_eval_73_nll", "api20", "datasets.evaluator.trips.00000073.metrics.log_loss", fmt="{:.6f}", optional=False)
fact("api_eval_73_miss", "api20", "datasets.evaluator.trips.00000073.metrics.candidate_miss_rate", fmt="{:.6f}", optional=False)
fact("api_eval_87_nll", "api20", "datasets.evaluator.trips.00000087.metrics.log_loss", fmt="{:.6f}", optional=False)
fact("api_eval_87_miss", "api20", "datasets.evaluator.trips.00000087.metrics.candidate_miss_rate", fmt="{:.6f}", optional=False)
fact("api_eval_94_nll", "api20", "datasets.evaluator.trips.00000094.metrics.log_loss", fmt="{:.6f}", optional=False)
fact("api_eval_94_miss", "api20", "datasets.evaluator.trips.00000094.metrics.candidate_miss_rate", fmt="{:.6f}", optional=False)

# Final mode never treats pending live/test facts as optional.
for _fact in FACTS:
    _fact["optional"] = False

FACTS_BY_NAME = {f["name"]: f for f in FACTS}


# ---------------------------------------------------------------------------
# input loading and fact resolution
# ---------------------------------------------------------------------------



def resolve_fact_values(state):
    resolved = {}
    for f in FACTS:
        name = f["name"]
        err = None
        val = None
        try:
            if "computed" in f:
                left = resolved[f["computed"]["left"]]
                right = resolved[f["computed"]["right"]]
                op = f["computed"]["op"]
                if left is None or right is None:
                    raise KeyError("operand unresolved")
                if op == "+":
                    val = left + right
                elif op == "-":
                    val = left - right
                elif op == "*":
                    val = left * right
                elif op == "/":
                    val = (left / right) if right else None
                else:
                    raise KeyError("unknown op %r" % op)
            elif "count" in f:
                rec = state["inputs"].get(f["file"])
                if rec is None or rec.get("data") is None:
                    raise KeyError("input %s missing" % f["file"])
                spec = f["count"]
                ptr, _, field = spec["part"].rpartition(".")
                rows = dig(rec["data"], ptr) if ptr else rec["data"]
                if not isinstance(rows, list):
                    raise KeyError("count target is not a list")
                val = 0
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    v = row.get(field)
                    if spec.get("in") is not None:
                        if v in spec["in"]:
                            val += 1
                    elif spec.get("not_in") is not None:
                        if v not in spec["not_in"]:
                            val += 1
                    elif v == spec.get("value"):
                        val += 1
            else:
                rec = state["inputs"].get(f["file"])
                if rec is None or rec.get("data") is None:
                    raise KeyError("input %s missing" % f["file"])
                val = dig(rec["data"], f["pick"])
        except Exception as exc:  # noqa: BLE001 - unresolved is a reportable state
            err = str(exc)
        resolved[name] = val
        resolved[name + "__err"] = err
        if val is None:
            resolved[name + "__str"] = "[[PENDING:%s]]" % name
        elif f.get("fmt") == "json":
            resolved[name + "__str"] = json.dumps(val, sort_keys=True)
        else:
            try:
                resolved[name + "__str"] = f.get("fmt", "{:}").format(val)
            except Exception:  # noqa: BLE001 - format failure is a defect
                resolved[name + "__str"] = str(val)
    return resolved


TOKEN_RE = re.compile(r"\{\{fact:([a-z0-9_]+)\}\}")


def resolve_tokens(md, resolved):
    def repl(m):
        name = m.group(1)
        if name in resolved:
            return resolved[name + "__str"]
        return "[[PENDING:%s]]" % name

    return TOKEN_RE.sub(repl, md)




# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def md_image_paths(md):
    return re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", md)




# ---------------------------------------------------------------------------
# template scanning
# ---------------------------------------------------------------------------

FORBIDDEN_PATH_FRAGMENTS = [
    "reports/phase_three/",
    "artifacts/phase_three/",
    "experiments/phase_three/",
    "reports/decoder/",
    "experiments/final_five/failed",
]




# ---------------------------------------------------------------------------
# final guards: --final fails closed unless every one passes
# ---------------------------------------------------------------------------

def _data(state, key):
    rec = state["inputs"].get(key)
    if rec is None or rec.get("data") is None:
        return None
    return rec["data"]



# ---------------------------------------------------------------------------
# Hardened evidence primitives. Only public declared files are opened here.
# ---------------------------------------------------------------------------
PINNED_CONFIG_FILE_SHA256 = "10405aa6b6f370e41d4ddcb01207fd1e8f58900fd0395e541ee6bc78146168c5"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
CHECK_KEY_RE = re.compile(r"(?:^|_)checks\Z")
MAP_BASE = "visualizations/final_five/maps/final_development_trip_00000009"
REQUIRED_TABLE_HEADERS = ["Model", "Algorithm Family", "Key Configuration", "CV Macro F1-score",
                          "Top-1 Accuracy", "Route continuity", "Student-Test Macro F1-score",
                          "Log Loss", "Inference Time"]


def _positive_int(value):
    return type(value) is int and value > 0


def _finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return True


def strict_json(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON key: " + key)
            value[key] = item
        return value
    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    result = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not _finite(result):
        raise ValueError("nonfinite numeric JSON value")
    return result


def scoped_file(name, *, must_exist=True):
    """Reject lexical aliases, traversal and every symlink component."""
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("empty or padded path")
    if any(ord(c) < 32 for c in name) or "\\" in name or unquote(name) != name:
        raise ValueError("encoded/control/backslash path")
    relative = Path(name)
    if (relative.is_absolute() or relative.as_posix() != name
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("path must be canonical repository-relative")
    root = ROOT.resolve()
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink is not an evidence file")
    if not current.resolve().is_relative_to(root):
        raise ValueError("path escaped repository root")
    if must_exist and (not current.is_file() or current.stat().st_size == 0):
        raise ValueError("missing, empty or nonregular evidence file")
    return current


def _json_file(path):
    return strict_json(path.read_text(encoding="utf-8"))


def _hash_binding(path, expected):
    if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
        raise ValueError("missing or malformed SHA-256 binding for " + str(path))
    target = scoped_file(path)
    if sha256_file(target) != expected:
        raise ValueError("SHA-256 mismatch: " + path)
    return target


def build_state(mode):
    state = {"mode": mode, "inputs": {}, "missing": [], "load_errors": [], "assets": {}}
    for entry in INPUTS:
        rec = dict(entry)
        try:
            path = scoped_file(entry["path"])
            rec["sha256"] = sha256_file(path)
            rec["data"] = _json_file(path) if entry["kind"] == "json" else None
            state["inputs"][entry["key"]] = rec
        except Exception as exc:
            state["missing"].append(rec)
            state["load_errors"].append("%s: %s" % (entry["key"], exc))
    return state


def evaluate_facts(resolved_md, resolved):
    rows = []
    for fact_record in FACTS:
        name = fact_record["name"]
        value = resolved.get(name)
        error = resolved.get(name + "__err")
        formatted = resolved.get(name + "__str", "")
        ok = value is not None and error is None and _finite(value) and formatted in resolved_md
        rows.append({"name": name, "ok": bool(ok),
                     "note": "resolved measured value present" if ok else "unresolved, invalid or absent fact: %s" % error})
    return rows


def _check_tree(value, path):
    if not isinstance(value, dict) or not value:
        raise ValueError(path + " must be a nonempty dictionary of boolean checks")
    for key, item in value.items():
        label = path + "." + str(key)
        if isinstance(item, dict):
            _check_tree(item, label)
        elif item is not True:
            raise ValueError(label + " must be boolean true")


def validate_check_containers(data, *, require_root=True):
    """Check only named check containers; metric exact_match=false is valid."""
    if not isinstance(data, dict) or not data:
        raise ValueError("evidence must be a nonempty object")
    if require_root:
        _check_tree(data.get("checks"), "checks")
    count = 0
    def walk(value, path="root"):
        nonlocal count
        if isinstance(value, dict):
            for key, item in value.items():
                label = path + "." + str(key)
                if CHECK_KEY_RE.search(str(key)):
                    _check_tree(item, label)
                    count += 1
                else:
                    walk(item, label)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, path + "[%d]" % index)
    walk(data)
    if not count:
        raise ValueError("no named check containers")
    return count


def verify_delivery(state, key, label):
    try:
        data = _data(state, key)
        if not isinstance(data, dict) or data.get("passed") is not True:
            raise ValueError("passed must be boolean true")
        count = validate_check_containers(data)
        if data.get("defects", []) != []:
            raise ValueError("defects is not an empty list")
        return True, "%d nonempty check containers passed" % count
    except Exception as exc:
        return False, "%s: %s" % (label, exc)


def _public_dependency(path):
    if not isinstance(path, str):
        return False
    parts = Path(path).parts
    return not (any(p in {"Student folder", "PathSnap_student_materials", "cache", "predictions", "data"}
                    or p.lower().startswith("private") or p.lower().startswith("evaluator") and parts[:3] != ("reports", "final_five", "evaluator")
                    for p in parts)
                or Path(path).suffix.lower() in {".npz", ".nodes", ".arcs", ".track", ".route", ".docx"})


def _binding_records(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield value
        for item in value.values():
            yield from _binding_records(item)
    elif isinstance(value, list):
        for item in value:
            yield from _binding_records(item)


def _png_binding(state, entry):
    path = entry.get("path")
    target = _hash_binding(path, entry.get("sha256"))
    with target.open("rb") as source:
        header = source.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("invalid PNG signature: " + path)
    width, height = struct.unpack(">II", header[16:24])
    if width < 1 or height < 1 or width * height > 100_000_000:
        raise ValueError("invalid or excessive PNG dimensions: " + path)
    from PIL import Image as PILImage
    with PILImage.open(target) as image:
        if image.format != "PNG":
            raise ValueError("PNG format mismatch: " + path)
        image.verify()
    if "pixels" in entry:
        pixels = entry["pixels"]
        expected = {"width": width, "height": height} if isinstance(pixels, dict) else [width, height]
        if pixels != expected:
            raise ValueError("PNG pixel dimensions differ from manifest: " + path)
    if "width" in entry and (entry["width"], entry.get("height")) != (width, height):
        raise ValueError("screenshot dimensions differ from evidence: " + path)
    if "bytes" in entry and entry["bytes"] != target.stat().st_size:
        raise ValueError("PNG byte count differs from evidence: " + path)
    state.setdefault("assets", {})[path] = entry["sha256"]


def verify_report_crops(state):
    """Recheck exact original pixels, rather than trusting a derivative pass flag."""
    manifest = _data(state, "report_crops")
    _require(isinstance(manifest, dict) and manifest.get("passed") is True
             and manifest.get("crop_count") == 12
             and manifest.get("all_pixel_equality_checks") is True, "report crop manifest incomplete")
    rows = manifest.get("crops")
    _require(isinstance(rows, list) and len(rows) == 12
             and {r.get("name") for r in rows if isinstance(r, dict)} == set(REPORT_CROP_NAMES),
             "report must contain exactly twelve unique screenshot details")
    originals = {r["path"]: r["sha256"] for r in _data(state, "browser")["screenshots"]}
    terminal = _data(state, "termv")
    originals[terminal["image"]] = terminal["image_sha256"]
    inputs = manifest.get("inputs")
    _require(isinstance(inputs, list) and inputs, "crop provenance inputs missing")
    for entry in inputs:
        _hash_binding(entry.get("path"), entry.get("sha256"))
    from PIL import Image as PILImage
    for row in rows:
        path = "reports/final_five/report_images/" + row["name"] + ".png"
        _require(row.get("path") == path == row.get("output_path")
                 and row.get("sha256") == row.get("output_sha256"), "crop output alias mismatch")
        original = row.get("source_path")
        _require(original in originals and row.get("source_sha256") == originals[original],
                 "crop source is not a genuine captured screenshot")
        source = _hash_binding(original, originals[original])
        target = _hash_binding(path, row.get("sha256"))
        box = row.get("crop_box")
        _require(isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box),
                 "crop coordinates must be four integers")
        with PILImage.open(source) as full, PILImage.open(target) as detail:
            left, top, right, bottom = box
            _require(0 <= left < right <= full.width and 0 <= top < bottom <= full.height,
                     "crop coordinates exceed screenshot")
            expected = full.crop(box)
            _require(detail.mode == expected.mode and detail.size == expected.size
                     and detail.tobytes() == expected.tobytes(), "report detail is not an exact source pixel crop")
            _require(full.size == (row.get("source_width"), row.get("source_height"))
                     and detail.size == (row.get("output_width"), row.get("output_height")),
                     "report crop dimensions differ")
            header = json.dumps({"mode": detail.mode, "size": detail.size}, sort_keys=True).encode()
            pixel_hash = hashlib.sha256(header + b"\0" + detail.tobytes()).hexdigest()
            _require(pixel_hash == row.get("pixel_sha256") and row.get("pixel_equality_checks") is True,
                     "report crop pixel hash differs")
        _hash_binding(row.get("ocr_path"), row.get("ocr_sha256"))
        _png_binding(state, row)
    return rows


def verify_figures(state, resolved_md):
    problems = []
    referenced = set(md_image_paths(resolved_md))
    try:
        eda = _data(state, "figman_eda")
        final = _data(state, "figman5")
        browser = _data(state, "browser")
        terminal = _data(state, "termv")
        mapman = _data(state, "mapman")
        if not isinstance(eda, list) or not eda:
            raise ValueError("EDA manifest must be a nonempty list")
        if (not isinstance(final, dict) or final.get("partial") is not False
                or final.get("figure_count") != 23 or final.get("model_figure_count") != 22
                or final.get("decoder_figure_count") != 1):
            raise ValueError("final figure manifest must contain complete 22-model + 1-decoder figures")
        entries = final.get("figures")
        expected = {p for p in REGISTRY_BY_PATH if p.startswith("visualizations/final_five/plots/")}
        if (not isinstance(entries, list) or len(entries) != 23
                or {e.get("path") for e in entries if isinstance(e, dict)} != expected):
            raise ValueError("final manifest figure set is not exactly the 23 registered figures")
        public_bindings = [r for r in _binding_records(final.get("inputs")) if _public_dependency(r["path"])]
        if not public_bindings:
            raise ValueError("final manifest has no public source bindings")
        for record in public_bindings:
            _hash_binding(record["path"], record["sha256"])
        shots = browser.get("screenshots") if isinstance(browser, dict) else None
        if not isinstance(shots, list) or not shots:
            raise ValueError("browser screenshot bindings missing")
        if not isinstance(terminal, dict):
            raise ValueError("terminal screenshot binding missing")
        terminal_entry = {"path": terminal.get("image"), "sha256": terminal.get("image_sha256")}
        maps = mapman.get("files") if isinstance(mapman, dict) else None
        if not isinstance(maps, list) or not maps:
            raise ValueError("final map files manifest missing")
        crops = verify_report_crops(state)
        all_entries = eda + entries + shots + [terminal_entry] + maps + crops
        bindings = {}
        for entry in all_entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError("malformed figure binding")
            path = entry["path"]
            if path in bindings and bindings[path].get("sha256") != entry.get("sha256"):
                raise ValueError("conflicting image bindings: " + path)
            bindings[path] = entry
        for path in sorted(REGISTRY_BY_PATH):
            if path not in referenced:
                problems.append("registry figure never referenced in template: " + path)
            if path not in bindings:
                problems.append("figure has no canonical hash binding: " + path)
                continue
            try:
                _png_binding(state, bindings[path])
            except Exception as exc:
                problems.append(str(exc))
        for path in sorted(referenced - set(REGISTRY_BY_PATH)):
            problems.append("template references unregistered image: " + path)
    except Exception as exc:
        problems.append(str(exc))
    return problems, []


def md_image_paths(md):
    return [a or b for a, b in re.findall(r"!\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+\"[^\"]*\")?\s*\)", md)]


def _historical_path_problems(md):
    problems = []
    historical_level = None
    for line in md.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2).lower()
            if historical_level is not None and level <= historical_level:
                historical_level = None
            if "historical" in title or "archive" in title:
                historical_level = level
        for fragment in FORBIDDEN_PATH_FRAGMENTS:
            if fragment in line and historical_level is None and "historical" not in line.lower():
                problems.append("unlabelled stale path fragment: " + fragment)
    return problems


def scan_template(md, check_tokens=True):
    problems = []
    if check_tokens:
        used = set(TOKEN_RE.findall(md))
        problems += ["template token not in fact registry: {{fact:%s}}" % n for n in sorted(used - set(FACTS_BY_NAME))]
        problems += ["non-optional fact never used in template: " + n for n in FACTS_BY_NAME if n not in used]
        directives = re.findall(r"\{\{([^}]+)\}\}", md)
        for directive in directives:
            if not re.fullmatch(r"fact:[a-z0-9_]+", directive) and directive != "table:requirements":
                problems.append("unsupported template directive: " + directive)
    elif "{{" in md or "[[PENDING" in md:
        problems.append("unresolved template directive or PENDING marker")
    for match in re.finditer(r"\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+\"[^\"]*\")?\s*\)", md):
        target = match.group(1) or match.group(2)
        if target.startswith("#"):
            continue
        parsed = urlsplit(target)
        if parsed.scheme:
            if parsed.scheme not in {"https", "http"} or not parsed.hostname:
                problems.append("unsafe external link: " + target)
            continue
        path = target.split("#", 1)[0]
        try:
            scoped_file(path)
        except Exception as exc:
            problems.append("invalid/missing template link %s: %s" % (path, exc))
    problems += _historical_path_problems(md)
    return problems


def _require(value, message):
    if not value:
        raise ValueError(message)


def verify_test_record(state):
    try:
        data = _data(state, "testsjson")
        _require(isinstance(data, dict), "test evidence missing")
        _require(_positive_int(data.get("total")) and type(data.get("passed")) is int
                 and data["passed"] == data["total"] and data.get("failures") == [], "test totals/failures invalid")
        _require(data.get("complete_release_suite") is True, "complete_release_suite must be true")
        _require(data.get("static_smoke_passed") is True and data.get("synthetic_tests_only") is True
                 and data.get("production_fit_or_inference") is False, "suite provenance flags invalid")
        counts = data.get("counts")
        _require(isinstance(counts, dict) and set(counts) == {"python", "hosting", "node"}
                 and all(_positive_int(v) for v in counts.values()) and sum(counts.values()) == data["total"], "subsuite counts invalid")
        suites = data.get("suites")
        _require(isinstance(suites, list) and len(suites) == 4
                 and {s.get("suite") for s in suites if isinstance(s, dict)} == {"python", "hosting", "node", "static"}, "exact four suite records required")
        for suite in suites:
            name = suite["suite"]
            _require(type(suite.get("exit_code")) is int and suite["exit_code"] == 0, "suite exit is not zero: " + name)
            log = _hash_binding(suite.get("log"), suite.get("log_sha256")).read_text(encoding="utf-8")
            if name in {"python", "hosting"}:
                found = re.findall(r"Ran (\d+) tests? in ", log)
                _require(len(found) == 1 and int(found[0]) == counts[name] and re.search(r"(?m)^OK\s*$", log) is not None, "unittest log count/status mismatch: " + name)
            elif name == "node":
                tests = re.findall(r"(?:# |ℹ )?tests\s+(\d+)", log)
                passed = re.findall(r"(?:# |ℹ )?pass\s+(\d+)", log)
                failed = re.findall(r"(?:# |ℹ )?fail\s+(\d+)", log)
                _require(tests and passed and failed and int(tests[-1]) == counts[name]
                         and int(passed[-1]) == counts[name] and int(failed[-1]) == 0, "Node log count/status mismatch")
            else:
                _require("PASS" in log and "FAIL" not in log, "static smoke log is not passing")
        return True, "positive exact counts and four hash-bound passing suite logs"
    except Exception as exc:
        return False, str(exc)


def _compact_config(state):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from mapmatch.matcher import MatchConfig
    config = _data(state, "mconfig5")
    _require(isinstance(config, dict) and config.get("classifier_weight", 0) > 0, "learned config missing")
    encoded = json.dumps(asdict(MatchConfig(**config)), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _check_response(row, dataset, trip_id, compact):
    _require(isinstance(row, dict), "response is not an object")
    _require(row.get("http_status") == 200 and row.get("metrics_status") == "ok", "response status/scoring invalid")
    _require(row.get("dataset_id") == dataset and row.get("trip_id") == trip_id, "response dataset/trip mismatch")
    _require(row.get("model", {}).get("sha256") == PINNED_MODEL_SHA256
             and row["model"].get("name") == SELECTED_FAMILY and row.get("match_config_hash") == compact, "response model/config hash missing or different")
    _require(isinstance(row.get("request_id"), str) and re.fullmatch(r"[0-9a-f]{32}", row["request_id"]), "response request ID invalid")
    diag = row.get("diagnostics", {})
    observations = diag.get("observation_count")
    _require(_positive_int(observations) and observations == diag.get("matched_observation_count")
             == diag.get("learned_emission_observations") == row.get("observation_assignment_count")
             and diag.get("uses_reference_route") is False, "response learned observation coverage invalid")
    _require(isinstance(row.get("metrics"), dict) and bool(row["metrics"]) and _finite(row), "response missing or nonfinite metrics")
    validate_check_containers(row)
    _check_tree(row.get("response_checks"), "response_checks")
    return row["request_id"]


def verify_identity_and_twenty(state):
    compact = _compact_config(state)
    config = _data(state, "mconfig5")
    config_hash = state["inputs"].get("mconfig5", {}).get("sha256")
    _require(config_hash == PINNED_CONFIG_FILE_SHA256, "decoder file hash differs from pinned freeze")
    model_hashes = [("model bytes", state["inputs"].get("model", {}).get("sha256"))]
    for key, pointer in [("results5", "model_freeze.model_sha256"), ("mfreeze", "model_sha256"),
                         ("accept", "selected_model_sha256"), ("clean", "model_sha256"),
                         ("dfreeze", "model_sha256"), ("dsums", "model_sha256"),
                         ("offlinev", "model_sha256"), ("apiv", "model_sha256"),
                         ("deliveryv", "model_sha256"), ("liveapi", "model_sha256"),
                         ("evalacc", "freeze_provenance.model_sha256")]:
        model_hashes.append((key, dig(_data(state, key), pointer)))
    _require(all(value == PINNED_MODEL_SHA256 for _, value in model_hashes), "missing/different model hash across freezes/delivery")
    for key in ("dfreeze", "dsums", "offlinev", "apiv", "deliveryv", "liveapi"):
        _require(_data(state, key).get("config_file_sha256") == config_hash, key + " decoder file hash missing/different")
    _require(_data(state, "evalacc").get("freeze_provenance", {}).get("config_file_sha256") == config_hash, "evaluator decoder hash missing/different")
    _require(_data(state, "dfreeze").get("config") == config, "full decoder freeze differs")
    for key in ("apiv", "deliveryv"):
        _require(_data(state, key).get("frozen_config") == config, key + " full decoder config differs")
    for key in ("apiv", "liveapi"):
        data = _data(state, key)
        _require(data.get("match_config_hash") == compact, key + " compact configuration missing/different")
        health = data.get("health", {})
        _require(health.get("model_hash") == PINNED_MODEL_SHA256 and health.get("match_config_hash") == compact
                 and health.get("model_loaded") is True and health.get("status") == "ready"
                 and health.get("prediction_cache") is False, key + " health identity/readiness invalid")
    _require(_data(state, "offlinev").get("api_compact_config_sha256") == compact
             and _data(state, "deliveryv").get("match_config_hash") == compact
             and _data(state, "evalacc").get("freeze_provenance", {}).get("api_compact_config_sha256") == compact,
             "offline/delivery/evaluator compact hash missing/different")
    primary = _data(state, "apiv")
    _require(primary.get("trip_count") == 20 and primary.get("dataset_count") == 2
             and primary.get("actual_estimator") == "HistGradientBoostingClassifier" and primary.get("mocked_model_or_inference") is False
             and primary.get("prediction_cache") is False, "actual ASGI20 provenance invalid")
    _require(primary.get("frozen_config_before") == config == primary.get("frozen_config_after"), "ASGI changed decoder configuration")
    primary_rows = {}
    request_ids = set()
    for key in ("apiv", "deliveryv"):
        datasets = _data(state, key).get("datasets")
        _require(isinstance(datasets, dict) and set(datasets) == {"development", "evaluator"}, key + " must contain exactly two datasets")
        for dataset, partition in datasets.items():
            trips = partition.get("trips")
            _require(partition.get("trip_count") == 10 and isinstance(trips, dict) and len(trips) == 10,
                     key + " partition must contain ten distinct trips")
            for tid, row in trips.items():
                _require(isinstance(tid, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", tid), "unsafe trip identity")
                if key == "apiv":
                    rid = _check_response(row, dataset, tid, compact)
                    _require(rid not in request_ids, "duplicate ASGI request ID")
                    request_ids.add(rid)
                    primary_rows[dataset, tid] = row
                else:
                    for container in ("offline_checks", "api_checks", "actual_cli_checks", "candidate_audit_checks", "public_checks"):
                        _check_tree(row.get(container), "delivery." + dataset + "." + tid + "." + container)
                    _require((dataset, tid) in primary_rows, "delivery trip not present in ASGI20")
                    ref = primary_rows[dataset, tid]
                    _require(row.get("matched_edge_array_sha256") == ref.get("matched_edge_array_sha256")
                             and row.get("predicted_geometry_sha256") == ref.get("predicted_geometry_sha256"), "delivery prediction fingerprints differ")
    live = _data(state, "liveapi")
    matches = live.get("matches")
    _require(live.get("passed") is True and live.get("mocked") is False and live.get("all_twenty_requested") is True
             and live.get("attempted_count") == live.get("successful_count") == 20
             and isinstance(matches, list) and len(matches) == 20, "public evidence does not contain actual20 successes")
    seen, public_ids = set(), set()
    for row in matches:
        pair = row.get("dataset_id"), row.get("trip_id")
        _require(pair in primary_rows and pair not in seen, "public dataset/trip duplicate or mismatch")
        seen.add(pair)
        rid = _check_response(row, *pair, compact)
        _require(rid not in public_ids, "duplicate public request ID")
        public_ids.add(rid)
        reference = primary_rows[pair]
        _require(row.get("metrics") == reference.get("metrics")
                 and row.get("predicted_geometry_sha256") == reference.get("predicted_geometry_sha256")
                 and row.get("matched_edge_array_sha256") == reference.get("matched_edge_array_sha256"), "public/ASGI metrics or fingerprints differ")
    _require(seen == set(primary_rows), "public coverage differs from ASGI20")
    counts = _data(state, "deliveryv").get("counts")
    _require(counts == {"offline": 20, "in_process_asgi": 20, "public_post": 20,
                        "additional_asgi_repeat": 1, "browser_post": 2, "receiving_terminals": 1}, "consolidated delivery counts invalid")
    repeated = primary.get("repeated_request")
    pair = (repeated.get("dataset_id"), repeated.get("trip_id")) if isinstance(repeated, dict) else None
    _require(pair in primary_rows and _check_response(repeated, *pair, compact) not in request_ids
             and repeated.get("excluded_from_twenty_headline") is True, "ASGI fresh repeat invalid")
    return compact


def verify_browser_terminal(state, compact):
    browser = _data(state, "browser")
    _require(isinstance(browser, dict) and browser.get("passed") is True and browser.get("mocked") is False, "actual browser evidence missing")
    for key in ("browser_errors", "console_errors", "external_requests"):
        _require(browser.get(key) == [], "browser " + key + " not an empty list")
    _check_tree(browser.get("checks"), "browser.checks")
    responses = browser.get("responses")
    _require(isinstance(responses, list) and len(responses) == 2
             and {r.get("dataset_id") for r in responses if isinstance(r, dict)} == {"development", "evaluator"}, "exact two browser dataset responses required")
    response_ids = set()
    for row in responses:
        response_ids.add(_check_response(row, row.get("dataset_id"), row.get("trip_id"), compact))
    _require(len(response_ids) == 2, "browser request IDs must be distinct")
    layouts = browser.get("layout_checks")
    _require(isinstance(layouts, list) and len(layouts) == 16, "sixteen responsive theme/dataset layouts required")
    combos = set()
    for row in layouts:
        _require(isinstance(row, dict) and row.get("horizontal_overflow") is False
                 and row.get("viewport_width") == row.get("document_width"), "responsive horizontal overflow")
        combos.add((row.get("viewport_width"), row.get("dataset_id"), row.get("theme")))
    _require(len(combos) == 16, "duplicate responsive layout combinations")
    terminal = _data(state, "termv")
    _require(isinstance(terminal, dict) and terminal.get("valid") is True
             and terminal.get("ocr_privacy_guard_passed") is True and terminal.get("mocked") is False, "actual privacy-checked terminal missing")
    request_id = terminal.get("request_id")
    _require(request_id in response_ids, "terminal not tied to an actual browser request")
    browser_row = next(r for r in responses if r["request_id"] == request_id)
    events = terminal.get("events")
    _require(isinstance(events, list) and len(events) >= 3 and all(isinstance(e, dict) for e in events), "terminal events malformed")
    _require({e.get("event") for e in events} == {"match_started", "match_completed", "http_request"}, "terminal event sequence incomplete")
    for event in events:
        _require(event.get("request_id") == request_id, "uncorrelated terminal event")
        if event.get("event") != "http_request":
            _require(event.get("model_hash") == PINNED_MODEL_SHA256 and event.get("match_config_hash") == compact
                     and event.get("trip_id") == browser_row.get("trip_id")
                     and event.get("dataset_role") == browser_row.get("dataset_id"), "inference terminal identity/hash missing/different")
    return "two real browser requests and their hash-bound receiving terminal"


def verify_map(state):
    manifest = _data(state, "mapman")
    metadata = _data(state, "map_metadata")
    browser = _data(state, "map_browser")
    _require(isinstance(manifest, dict) and manifest.get("passed") is True, "final map manifest not passed")
    _require(all(manifest.get(key) is True for key in ("development_only", "contains_reference_geometry", "do_not_serve_in_application")), "map privacy scope flags invalid")
    _require(manifest.get("map_path") == MAP_BASE + ".html"
             and manifest.get("metadata_path") == MAP_BASE + ".metadata.json"
             and manifest.get("browser_path") == MAP_BASE + ".browser.json", "map paths are not the actual final map")
    _require(manifest.get("model_artifact_sha256") == PINNED_MODEL_SHA256
             and manifest.get("decoder_config_sha256") == PINNED_CONFIG_FILE_SHA256, "map manifest model/decoder identity differs")
    entries = manifest.get("files")
    _require(isinstance(entries, list) and len(entries) == 6
             and {e.get("path") for e in entries if isinstance(e, dict)} == {MAP_BASE + s for s in (".html", ".png", ".pdf", ".metadata.json", ".browser.json", ".ocr.txt")}, "map six-file manifest incomplete")
    for entry in entries:
        _hash_binding(entry.get("path"), entry.get("sha256"))
    _require(isinstance(metadata, dict) and metadata.get("model_artifact_sha256") == PINNED_MODEL_SHA256
             and metadata.get("decoder_config_sha256") == PINNED_CONFIG_FILE_SHA256
             and metadata.get("model_family") == SELECTED_FAMILY and metadata.get("model_run") == TUNED_RUN
             and metadata.get("development_only") is True and metadata.get("do_not_serve_in_application") is True,
             "map metadata is not the frozen final development prediction")
    _require(isinstance(browser, dict) and browser.get("source_html") == MAP_BASE + ".html"
             and browser.get("screenshot_path") == MAP_BASE + ".png", "map browser paths differ")
    for key in ("page_errors", "console_errors", "failed_requests", "external_requests"):
        _require(browser.get(key) == [], "map browser errors/requests present")
    for key in ("all_overlays_inside_viewport", "no_overlay_overlap", "prediction_geometry_matches_every_cli_arc",
                "all_full_graph_and_gps_coordinates_match", "reference_toggle_verified", "no_external_requests", "no_console_errors"):
        _require(browser.get(key) is True, "map browser check missing/false: " + key)
    _require(browser.get("prediction_geometry_latlon_sha256") == metadata.get("prediction_geometry_latlon_sha256"), "map exact geometry hashes differ")
    return "actual final HTML/PNG/PDF plus metadata/browser/OCR files are hash-bound"


def verify_selection_and_hpo(state):
    data, acceptance = _data(state, "results5"), _data(state, "accept")
    _require(isinstance(data, dict) and data.get("family_count") == 5
             and sorted(data.get("families", [])) == sorted(FIVE_FAMILIES), "final family set is not exactly five")
    _require(data.get("selected_model") == SELECTED_FAMILY and data.get("final_run_name") == TUNED_RUN, "selected family/run differs")
    for key in ("cv", "evaluations"):
        _require(isinstance(data.get(key), dict) and set(data[key]) == set(SIX_CONFIGS), "not exactly six " + key + " configurations")
    comparison = data.get("comparison")
    _require(isinstance(comparison, list) and len(comparison) == 6
             and {r.get("model") for r in comparison if isinstance(r, dict)} == set(SIX_CONFIGS), "six-row comparison incomplete")
    _require(isinstance(acceptance, dict) and acceptance.get("passed") is True
             and acceptance.get("exactly_five_families") is True and acceptance.get("all80_train_trips") is True
             and acceptance.get("all10_student_trips") is True and acceptance.get("student_used_for_selection") is False
             and acceptance.get("evaluator_accessed") is False, "model acceptance flags invalid")
    models = acceptance.get("all_models")
    _require(isinstance(models, dict) and set(models) == set(SIX_CONFIGS), "all-six actual estimator acceptance missing")
    for name, record in models.items():
        _require(record.get("passed") is True and record.get("training_trip_count") == 80
                 and record.get("student_trip_count") == 10 and _positive_int(record.get("strict_fixture_rows"))
                 and isinstance(record.get("max_probability_delta"), (int, float))
                 and 0 <= record["max_probability_delta"] <= 1e-12, "invalid actual estimator acceptance: " + name)
        _hash_binding(record.get("artifact"), record.get("sha256"))
    bayesian = data.get("bayesian")
    _require(isinstance(bayesian, dict) and bayesian == _data(state, "bayes5"), "Bayesian summary differs from consolidated results")
    trials = bayesian.get("trials")
    _require(isinstance(trials, list) and len(trials) == 12 and [t.get("trial") for t in trials] == list(range(12)), "actual twelve-trial sequence missing")
    schedule = ["measured_baseline_prior"] + ["seeded_initial_design"] * 3 + ["gaussian_process_expected_improvement"] * 8
    _require([t.get("proposal") for t in trials] == schedule and bayesian.get("reused") is True
             and bayesian.get("fresh_hpo_trials") == 0 and bayesian.get("trials_including_measured_prior") == 12
             and bayesian.get("seeded_initial_design_trials") == 3 and bayesian.get("model_based_trials") == 8, "HPO reuse schedule/counts differ")
    _require("TRAIN" in str(bayesian.get("selection_data")) and "not loaded" in str(bayesian.get("selection_data")), "HPO selection boundary missing")
    reuse = bayesian.get("reuse_provenance", {})
    _hash_binding(reuse.get("summary_file"), reuse.get("summary_sha256"))
    trial_log = _hash_binding(reuse.get("trial_log_file"), reuse.get("trial_log_sha256"))
    _require([strict_json(line) for line in trial_log.read_text().splitlines() if line.strip()] == trials, "actual historical trial log differs")
    pins = reuse.get("trial_cv_sha256")
    _require(isinstance(pins, dict) and len(pins) == 12, "twelve CV evidence hashes required")
    for path, digest in pins.items():
        _hash_binding(path, digest)
    _require(len({json.dumps(t["parameters"], sort_keys=True) for t in trials}) == 12, "duplicate HPO trial parameters")
    for name in SIX_CONFIGS:
        cv, evaluation = data["cv"][name], data["evaluations"][name]
        folds = cv.get("folds")
        _require(isinstance(folds, list) and len(folds) == 3, "exact three whole-trip folds required: " + name)
        validation_ids, all_training_ids = [], set()
        for fold in folds:
            train, validation = fold.get("train_trip_ids"), fold.get("validation_trip_ids")
            _require(isinstance(train, list) and isinstance(validation, list) and train and validation
                     and len(train) == len(set(train)) and len(validation) == len(set(validation))
                     and not set(train) & set(validation) and len(set(train) | set(validation)) == 80, "CV trip leakage/coverage: " + name)
            validation_ids.extend(validation)
            all_training_ids.update(train)
        _require(len(validation_ids) == len(set(validation_ids)) == 80, "CV validation coverage is not all80 once: " + name)
        _require(cv.get("parameters") == evaluation.get("parameters") and _finite(cv) and _finite(evaluation), "CV/evaluation parameter or numeric drift: " + name)
    _require(bayesian.get("best_parameters") == data["cv"][TUNED_RUN].get("parameters")
             and bayesian.get("best_objective") == data["cv"][TUNED_RUN].get("objective")
             and bayesian.get("baseline_objective") == data["cv"][SELECTED_FAMILY].get("objective"), "HPO tuned/default summaries drifted")
    metadata, freeze, decoder = _data(state, "mmeta"), _data(state, "mfreeze"), _data(state, "dfreeze")
    _require(metadata.get("fresh_fit") is True and metadata.get("training_trip_count") == 80
             and metadata.get("selection", {}).get("student_used") is False
             and metadata.get("selection", {}).get("evaluator_used") is False
             and freeze.get("student_used_for_selection") is False and freeze.get("evaluator_used_for_selection") is False
             and decoder.get("student_results_used_for_selection") is False and decoder.get("evaluator_accessed") is False,
             "student/evaluator selection-contamination flags missing/true")
    for key, field in (("mmeta", "metadata_sha256"), ("fschema", "feature_schema_sha256")):
        _require(freeze.get(field) == state["inputs"].get(key, {}).get("sha256"), "model sidecar hash mismatch: " + key)
    _require(acceptance.get("results_sha256") == state["inputs"].get("results5", {}).get("sha256"), "acceptance results hash mismatch")
    return "five families, six actual fits, disjoint TRAIN folds and genuine reused twelve-trial HPO"


def verify_requirements(state):
    data = _data(state, "reqs")
    rows = data.get("requirements") if isinstance(data, dict) else None
    _require(isinstance(rows, list) and len(rows) == 47 and all(isinstance(r, dict) for r in rows), "exact 47 requirement objects required")
    _require(len({r.get("id") for r in rows}) == 47 and all(isinstance(r.get("id"), str) and r["id"] for r in rows), "duplicate/empty requirement IDs")
    pending = [r["id"] for r in rows if r.get("status") not in REQ_TERMINAL_STATUSES]
    _require(not pending, "unresolved requirement rows: " + ", ".join(pending))
    return "47 uniquely identified terminal requirement rows"


def run_final_guards(state, resolved, resolved_md, template_text=None):
    guards = []
    def check(name, function):
        try:
            detail = function()
            guards.append({"name": name, "ok": True, "detail": str(detail or "passed")})
        except Exception as exc:
            guards.append({"name": name, "ok": False, "detail": str(exc)})
    def declared_inputs():
        _require(not state["load_errors"], "; ".join(state["load_errors"]))
        _require(not [e for e in state["missing"] if e.get("final")], "mandatory final inputs missing")
        for record in state["inputs"].values():
            _hash_binding(record["path"], record["sha256"])
        return "all mandatory paths regular, canonical, nonempty and hash-bound"
    check("mandatory_inputs_and_snapshot", declared_inputs)
    check("five_families_six_configs_train_hpo", lambda: verify_selection_and_hpo(state))
    for key in ("offlinev", "apiv", "deliveryv", "evalacc"):
        def delivery(key=key):
            ok, detail = verify_delivery(state, key, key)
            _require(ok, detail)
            return detail
        check(key + "_recursive_delivery_checks", delivery)
    compact = None
    def identity():
        nonlocal compact
        compact = verify_identity_and_twenty(state)
        return "pinned model/file/compact hashes across actual offline20/ASGI20/public20"
    check("all_twenty_identity_and_full_config", identity)
    check("browser_two_and_terminal_binding", lambda: verify_browser_terminal(state, compact or _compact_config(state)))
    check("actual_final_folium_map", lambda: verify_map(state))
    check("requirements_all47_resolved", lambda: verify_requirements(state))
    def tests():
        ok, detail = verify_test_record(state)
        _require(ok, detail)
        return detail
    check("complete_release_suite_and_bound_logs", tests)
    def facts():
        _require(len(FACTS_BY_NAME) == len(FACTS), "duplicate fact names")
        failed = [r["name"] for r in evaluate_facts(resolved_md, resolved) if not r["ok"]]
        _require(not failed, "unresolved/absent facts: " + ", ".join(failed))
        return "%d actual measured facts resolved" % len(FACTS)
    check("every_registered_fact_resolved", facts)
    def template():
        _require(isinstance(template_text, str), "raw template required for token coverage")
        problems = scan_template(template_text, check_tokens=True) + scan_template(resolved_md, check_tokens=False)
        _require(not problems, "; ".join(problems))
        return "raw tokens covered; resolved text has no directives, missing links or unlabelled stale paths"
    check("raw_template_and_resolved_output", template)
    def figures():
        problems, notes = verify_figures(state, resolved_md)
        _require(not problems and not notes, "; ".join(problems + notes))
        _require(MAP_BASE + ".html" in resolved_md, "actual final Folium HTML link is absent")
        return "23 final + 12 EDA + actual map + seven original live captures + twelve exact-pixel readable details bound"
    check("all_figures_png_dimensions_hashes", figures)
    def clean_load():
        data = _data(state, "clean")
        _require(isinstance(data, dict) and data.get("passed") is True
                 and type(data.get("max_probability_delta")) in (int, float)
                 and type(data.get("atol")) in (int, float)
                 and math.isfinite(data["atol"]) and 0 <= data["max_probability_delta"] <= data["atol"] <= 1e-12,
                 "strict clean-load probability check invalid")
        _require(data.get("parent_versions") == _data(state, "results5").get("versions"), "clean-load software versions differ")
    check("clean_reload_and_versions", clean_load)
    def ablation():
        data = _data(state, "dablt")
        _require(data.get("trip_count") == 10 and data.get("failed_trip_ids") == []
                 and _positive_int(data.get("total_observations")) and data["total_observations"] == data.get("expected_total_observations")
                 and data.get("all_ten_headline_eligible") is True and data.get("evaluator_accessed") is False, "all-ten failure-inclusive ablation invalid")
    check("decoder_all_ten_original_observations", ablation)
    def independent():
        data = _data(state, "indaudit")
        _require(data.get("passed") is True and data.get("critical_defects") == [], "independent model audit not passed")
    check("independent_model_audit", independent)
    return guards


# ---------------------------------------------------------------------------
# PDF rendering (reportlab). The report is authored in markdown and rendered
# to A4 with a TOC, page numbers, and a DRAFT watermark in --preview mode.
# ---------------------------------------------------------------------------

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as _pdfcanvas
    from reportlab.platypus import (CondPageBreak, Image, PageBreak, Paragraph,
                                    SimpleDocTemplate, Spacer, Table,
                                    TableStyle)
    from reportlab.platypus.tableofcontents import TableOfContents

    HAS_PDF = True
except ImportError:  # pragma: no cover - .venv provides reportlab
    HAS_PDF = False

if HAS_PDF:
    PAGE_W, PAGE_H = A4
    MARGIN = 16 * mm
    FRAME_W = PAGE_W - 2 * MARGIN

_DRAFT = [False]


if HAS_PDF:

    class NumberedCanvas(_pdfcanvas.Canvas):
        """Two-pass canvas: page numbers, footer, and the DRAFT watermark."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_states = []

        def showPage(self):  # noqa: N802 - reportlab API
            self._saved_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._saved_states)
            for state in self._saved_states:
                self.__dict__.update(state)
                self._draw_decorations(total)
                super().showPage()
            super().save()

        def _draw_decorations(self, total):
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.grey)
            self.drawString(MARGIN, 9 * mm, "PathSnap Final Technical Report")
            self.drawRightString(PAGE_W - MARGIN, 9 * mm,
                                 "Page %d of %d" % (self._pageNumber, total))
            if _DRAFT[0]:
                self.setFont("Helvetica-Bold", 56)
                self.setFillColor(colors.Color(0.75, 0.75, 0.75, alpha=0.30))
                self.translate(PAGE_W / 2.0, PAGE_H / 2.0)
                self.rotate(45)
                self.drawCentredString(0, 0, "DRAFT - NOT FINAL")
            self.restoreState()

    class TocDoc(SimpleDocTemplate):
        """Notifies TOC entries and writes PDF outline bookmarks."""

        def afterFlowable(self, flowable):
            if isinstance(flowable, Paragraph):
                style_name = flowable.style.name
                if style_name in ("H1", "H2"):
                    text = flowable.getPlainText()
                    if text == "Contents":
                        return
                    level = 0 if style_name == "H1" else 1
                    previous = getattr(self, "_outline_level", -1)
                    level = min(level, previous + 1)
                    self._outline_level = level
                    self.notify("TOCEntry", (level, text, self.page))
                    key = "toc-%s" % sha256_text(text + str(self.page))[:12]
                    self.canv.bookmarkPage(key)
                    self.canv.addOutlineEntry(text, key, level, 0)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("RTitle", parent=ss["Title"], fontSize=26,
                                leading=32, spaceAfter=10),
        "subtitle": ParagraphStyle("RSub", parent=ss["Title"], fontSize=13,
                                   leading=18, spaceAfter=18,
                                   textColor=colors.darkgrey),
        "meta": ParagraphStyle("RMeta", parent=ss["BodyText"], fontSize=9.5,
                               leading=13, spaceAfter=4),
        "H1": ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15,
                             leading=19, spaceBefore=6, spaceAfter=8),
        "H2": ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5,
                             leading=16, spaceBefore=10, spaceAfter=5),
        "H3": ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10.5,
                             leading=14, spaceBefore=8, spaceAfter=4),
        "Body": ParagraphStyle("RBody", parent=ss["BodyText"], fontSize=9.5,
                               leading=13.5, spaceAfter=6),
        "Bullet": ParagraphStyle("RBullet", parent=ss["BodyText"],
                                 fontSize=9.5, leading=13.5, spaceAfter=3,
                                 leftIndent=12, bulletIndent=4),
        "Caption": ParagraphStyle("RCap", parent=ss["BodyText"], fontSize=8,
                                  leading=10.5, spaceAfter=10,
                                  textColor=colors.darkgrey),
        "Cell": ParagraphStyle("RCell", parent=ss["BodyText"], fontSize=8.1,
                               leading=10.2, spaceAfter=0),
        "CellHead": ParagraphStyle("RCellHead", parent=ss["BodyText"],
                                   fontSize=8.1, leading=10.2, spaceAfter=0,
                                   fontName="Helvetica-Bold"),
        "Code": ParagraphStyle("RCode", parent=ss["BodyText"],
                               fontName="Courier", fontSize=7.4,
                               leading=9.4, spaceAfter=6),
    }


def esc(text):
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def inline(text):
    """Escape a markdown fragment, then re-apply bold/italic/code markers."""
    text = text.translate(str.maketrans({"σ": "sigma", "β": "beta", "λ": "lambda", "Σ": "sum", "π": "pi", "∈": "in", "∑": "sum", "−": "-", "→": "->", "≤": "<=", "≥": ">=", "≠": "!=", "×": "x"}))
    out = esc(text)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", out)
    _bt = chr(96)
    out = re.sub(_bt + "([^" + _bt + "]+)" + _bt,
                 r'<font face="Courier">\1</font>', out)
    def pdf_link(match):
        label, angle_target, bare_target = match.groups()
        target = unescape(angle_target or bare_target)
        parsed = urlsplit(target)
        if parsed.scheme:
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("unsafe report link")
            url = target
        else:
            if target.startswith("#"):
                # Markdown section anchors have no stable ReportLab outline key.
                return label
            path = target.split("#", 1)[0]
            if not path or path.startswith("/") or ".." in Path(path).parts:
                raise ValueError("report link must be repository-relative")
            url = "https://github.com/DevaOnL/ML-Hackathon/blob/main/" + quote(target, safe="/#")
        return '<link href="%s" color="#245b99">%s</link>' % (esc(url).replace('"', "&quot;"), label)
    out = re.sub(r"\[([^]\n]+)\]\((?:&lt;(.+?)&gt;|([^\s)]+))\)", pdf_link, out)
    # Built-in PDF fonts cannot render zero-width spaces. Paragraph already
    # splits long words; keep identifiers intact without visible fallback glyphs.
    return out


def _md_table_rows(block_lines):
    rows = []
    for line in block_lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
            continue
        rows.append(cells)
    return rows


def _flow_table(rows, styles):
    data = []
    for i, row in enumerate(rows):
        style = styles["CellHead"] if i == 0 else styles["Cell"]
        data.append([Paragraph(inline(c), style) for c in row])
    ncols = max(len(r) for r in data)
    weights = ([1.2, 1.05, 2.5] + [0.9] * 6) if ncols == 9 else [1] * ncols
    if ncols == 4 and rows[0][:2] == ["id", "area"]:
        # Long original requirement prose gets room; IDs remain unbroken.
        weights = [1.05, 1.0, 5.5, 1.6]
    widths = [FRAME_W * weight / sum(weights) for weight in weights]
    table = Table(data, colWidths=widths, repeatRows=1, splitByRow=1)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.92, 0.92, 0.92)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return table


IMG_LINE_RE = re.compile(
    r"^!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)$")


def _flow_image(path, alt, styles):
    fpath = ROOT / path
    if not fpath.exists():
        placeholder = Table(
            [[Paragraph("FIGURE PENDING: %s" % esc(path), styles["Cell"])]],
            colWidths=[FRAME_W])
        placeholder.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.7, colors.grey),
            ("BACKGROUND", (0, 0), (-1, -1), colors.Color(0.96, 0.96, 0.96)),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ]))
        return [placeholder]
    reader = ImageReader(str(fpath))
    iw, ih = reader.getSize()
    max_w = FRAME_W
    max_h = PAGE_H * 0.52
    scale = min(max_w / iw, max_h / ih)
    if not path.startswith("reports/final_five/report_images/"):
        scale = min(scale, 1.0)
    flows = [Image(str(fpath), width=iw * scale, height=ih * scale)]
    if alt:
        flows.append(Paragraph(esc(alt), styles["Caption"]))
    return flows


def md_to_flowables(md, styles):
    flows = []
    lines = md.splitlines()
    i = 0
    para_buf = []
    first_h1 = [True]

    def flush_para():
        if para_buf:
            text = " ".join(x.strip() for x in para_buf).strip()
            if text:
                flows.append(Paragraph(inline(text), styles["Body"]))
            del para_buf[:]

    _fence = chr(96) * 3
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        m = IMG_LINE_RE.match(stripped)
        if m:
            flush_para()
            flows.extend(_flow_image(m.group(2), m.group(1), styles))
            flows.append(Spacer(1, 4))
            i += 1
            continue
        if stripped.startswith(_fence):
            flush_para()
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith(_fence):
                code.append(lines[i])
                i += 1
            i += 1  # closing fence
            text = "<br/>".join(esc(c) for c in code)
            if text:
                flows.append(Paragraph(text, styles["Code"]))
            continue
        if stripped.startswith("|"):
            flush_para()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            rows = _md_table_rows(block)
            if rows:
                flows.append(Spacer(1, 2))
                flows.append(_flow_table(rows, styles))
                flows.append(Spacer(1, 6))
            continue
        if stripped.startswith("###"):
            flush_para()
            flows.append(Paragraph(inline(stripped.lstrip("#").strip()),
                                   styles["H3"]))
            i += 1
            continue
        if stripped.startswith("##"):
            flush_para()
            flows.append(Paragraph(inline(stripped.lstrip("#").strip()),
                                   styles["H2"]))
            i += 1
            continue
        if stripped.startswith("# "):
            flush_para()
            if first_h1[0]:
                first_h1[0] = False
            else:
                # Keep a heading with following content without emitting an empty
                # page when the previous flowable has already exhausted a frame.
                flows.append(CondPageBreak(100))
            flows.append(Paragraph(inline(stripped[1:].strip()), styles["H1"]))
            i += 1
            continue
        if stripped.startswith(("- ", "* ")):
            flush_para()
            flows.append(Paragraph(inline(stripped[2:].strip()),
                                   styles["Bullet"], bulletText="\u2022"))
            i += 1
            continue
        if not stripped:
            flush_para()
            i += 1
            continue
        para_buf.append(line)
        i += 1
    flush_para()
    return flows


def render_pdf(md, pdf_path, draft):
    if not HAS_PDF:
        raise RuntimeError("reportlab is not installed in this interpreter")
    styles = _styles()
    _DRAFT[0] = bool(draft)
    doc = TocDoc(
        str(pdf_path), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="%s - %s" % (REPORT_TITLE, REPORT_SUBTITLE),
        author="PathSnap final-phase build",
    )
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC1", parent=styles["Body"], fontSize=9.5,
                       leading=13, leftIndent=6, firstLineIndent=-6,
                       spaceBefore=3),
        ParagraphStyle("TOC2", parent=styles["Body"], fontSize=8.6,
                       leading=11.5, leftIndent=18, firstLineIndent=-6),
    ]
    story = [
        Paragraph(REPORT_TITLE, styles["title"]),
        Paragraph(esc(REPORT_SUBTITLE), styles["subtitle"]),
        Paragraph(esc(
            "Generated %s%s" % (
                datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"),
                " - DRAFT PREVIEW, not the final deliverable" if draft
                else "")),
            styles["meta"]),
        Spacer(1, 10),
        Paragraph("Contents", styles["H2"]),
        toc,
        PageBreak(),
    ]
    story.extend(md_to_flowables(md, styles))
    doc.multiBuild(story, canvasmaker=NumberedCanvas)
    return pdf_path


def pdf_page_count_and_text(pdf_path):
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover
        return None, ""
    reader = PdfReader(str(pdf_path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return len(reader.pages), text

# ---------------------------------------------------------------------------
# readiness, manifest, and CLI
# ---------------------------------------------------------------------------

def collect_readiness(state, fact_rows, guards):
    missing_final = [m for m in state["missing"] if m.get("final")]
    blocking = []
    for m in missing_final:
        blocking.append("missing final input %s (%s, owner: %s)"
                        % (m["path"], m["key"], m.get("owner", "?")))
    for err in state["load_errors"]:
        blocking.append("load error: %s" % err)
    for row in fact_rows:
        if not row["ok"]:
            blocking.append("fact not satisfied: %s (%s)"
                            % (row["name"], row["note"]))
    for g in guards:
        if not g["ok"]:
            blocking.append("guard failed: %s - %s" % (g["name"], g["detail"]))
    return blocking


def print_check_report(state, resolved, fact_rows, guards, template_problems):
    print("=== PathSnap final report builder: CHECK (read-only) ===")
    present = len(state["inputs"])
    print("inputs: %d registered, %d present, %d missing"
          % (len(INPUTS), present, len(state["missing"])))
    for m in state["missing"]:
        tag = "FINAL-BLOCKING" if m["final"] else "optional"
        print("  missing [%s] %s = %s (owner: %s)"
              % (tag, m["key"], m["path"], m.get("owner", "?")))
    for err in state["load_errors"]:
        print("  load error: %s" % err)
    if template_problems:
        print("template problems:")
        for p in template_problems:
            print("  - %s" % p)
    else:
        print("template problems: none")
    bad_facts = [r for r in fact_rows if not r["ok"]]
    opt_pending = [r for r in fact_rows if r["ok"]
                   and "optional" in r["note"]]
    print("facts: %d total, %d unresolved-blocking, %d optional-pending"
          % (len(fact_rows), len(bad_facts), len(opt_pending)))
    for r in bad_facts:
        print("  - %s: %s" % (r["name"], r["note"]))
    failed = [g for g in guards if not g["ok"]]
    print("guards: %d total, %d failed" % (len(guards), len(failed)))
    for g in failed:
        print("  FAIL %s: %s" % (g["name"], g["detail"]))
    print("readiness: %s"
          % ("READY for --final" if not failed and not bad_facts
             and not state["missing"] and not state["load_errors"]
             else "NOT READY"))


TABLE_RE = re.compile(r"\{\{table:([a-z0-9_]+)\}\}")


def _requirements_table_md(state):
    """Render the full requirement matrix as a markdown table."""
    data = _data(state, "reqs") or {}
    rows = data.get("requirements") or []
    lines = ["| id | area | requirement | status |",
             "| --- | --- | --- | --- |"]
    for r in rows:
        rid = str(r.get("id", ""))
        area = str(r.get("area", ""))
        text = str(r.get("requirement", "")).replace("|", "\\|")
        text = text.replace("\n", " ")
        status = str(r.get("status", ""))
        lines.append("| %s | %s | %s | %s |" % (rid, area, text, status))
    return "\n".join(lines)


def expand_tables(md, state):
    """Expand {{table:name}} markers from structured inputs."""
    def repl(m):
        key = m.group(1)
        if key == "requirements":
            return _requirements_table_md(state)
        return m.group(0)

    return TABLE_RE.sub(repl, md)


# The hardened entry point is defined after every function below.


# Persisted artifacts are verified independently of a fresh build; --check is
# strictly read-only and refuses stale facts, sources, outputs and asset bytes.
def _registry_sha256():
    return sha256_text(json.dumps({"inputs": INPUTS, "facts": FACTS,
                                  "figures": FIGURE_REGISTRY}, sort_keys=True))


def verify_persisted_manifest(state, resolved=None, *, require_final=True):
    if not MANIFEST_PATH.exists():
        return []
    problems = []
    try:
        relative = MANIFEST_PATH.relative_to(ROOT).as_posix()
        manifest = _json_file(scoped_file(relative))
        _require(isinstance(manifest, dict), "persisted manifest is not an object")
        if require_final:
            _require(manifest.get("mode") == "final", "persisted report manifest is not final")
        _require(manifest.get("schema_version") == 2 and manifest.get("registry_sha256") == _registry_sha256(),
                 "persisted fact/input/figure registry changed")
        builder = manifest.get("builder", {})
        _require(builder.get("path") == Path(__file__).resolve().relative_to(ROOT).as_posix()
                 and builder.get("sha256") == sha256_file(Path(__file__).resolve()), "persisted builder source changed")
        template = manifest.get("template", {})
        _require(template.get("path") == TEMPLATE_PATH.relative_to(ROOT).as_posix(), "persisted template path differs")
        _hash_binding(template.get("path"), template.get("sha256"))
        records = manifest.get("inputs")
        _require(isinstance(records, dict) and set(records) == set(state["inputs"]), "persisted input registry differs")
        for key, record in records.items():
            _require(record.get("path") == state["inputs"][key]["path"]
                     and record.get("sha256") == state["inputs"][key]["sha256"], "persisted evidence drift: " + key)
            _hash_binding(record.get("path"), record.get("sha256"))
        for key in ("markdown", "pdf"):
            record = manifest.get(key)
            _require(isinstance(record, dict), "persisted " + key + " binding absent")
            _hash_binding(record.get("path"), record.get("sha256"))
        _require(manifest["markdown"]["path"] == MD_PATH.relative_to(ROOT).as_posix()
                 and manifest["pdf"]["path"] == PDF_PATH.relative_to(ROOT).as_posix(), "persisted output paths differ")
        assets = manifest.get("assets")
        _require(isinstance(assets, dict) and set(REGISTRY_BY_PATH) <= set(assets), "persisted figure bindings incomplete")
        for path, digest in assets.items():
            _hash_binding(path, digest)
        if resolved is not None:
            expected = {fact["name"]: resolved.get(fact["name"] + "__str") for fact in FACTS}
            _require(manifest.get("facts") == expected, "persisted fact registry values changed")
        guards = manifest.get("guards")
        _require(isinstance(guards, list) and guards and all(isinstance(g, dict) and g.get("ok") is True for g in guards),
                 "persisted guards are not all explicitly true")
        _require(_positive_int(manifest.get("pdf", {}).get("pages")), "persisted page count missing")
    except Exception as exc:
        problems.append(str(exc))
    return problems


def write_manifest(mode, state, resolved, guards, md, pdf_path=None, pdf_pages=None, *, destination=None):
    manifest = {
        "schema_version": 2, "mode": mode,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(Path(__file__).resolve())},
        "template": {"path": TEMPLATE_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(TEMPLATE_PATH)},
        "registry_sha256": _registry_sha256(), "report_title": REPORT_TITLE,
        "report_subtitle": REPORT_SUBTITLE, "pinned_model_sha256": PINNED_MODEL_SHA256,
        "pinned_config_file_sha256": PINNED_CONFIG_FILE_SHA256,
        "page_policy": "No artificial page-count target. Full measured content and readable figures/tables.",
        "inputs": {key: {"path": rec["path"], "sha256": rec["sha256"], "final": rec["final"]}
                   for key, rec in sorted(state["inputs"].items())},
        "facts": {f["name"]: resolved.get(f["name"] + "__str") for f in FACTS},
        "assets": dict(sorted(state.get("assets", {}).items())), "guards": guards,
        "markdown": {"path": MD_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_text(md)},
    }
    if pdf_path is not None:
        manifest["pdf"] = {"path": PDF_PATH.relative_to(ROOT).as_posix(),
                           "sha256": sha256_file(pdf_path), "pages": pdf_pages}
    target = destination if destination is not None else MANIFEST_PATH
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def _safe_output(path):
    relative = path.relative_to(ROOT).as_posix()
    scoped_file(relative, must_exist=False)
    _require(not path.exists() or path.is_file(), "output is not a regular file")
    return path


def _evidence_unchanged(state, template_hash, builder_hash):
    _require(sha256_file(TEMPLATE_PATH) == template_hash, "template changed during build")
    _require(sha256_file(Path(__file__).resolve()) == builder_hash, "builder changed during build")
    for record in state["inputs"].values():
        _hash_binding(record["path"], record["sha256"])
    for path, digest in state.get("assets", {}).items():
        _hash_binding(path, digest)


def _render_and_publish(mode, state, resolved, guards, md):
    # Validate everything and render a temporary PDF before touching final names.
    # A failed PDF/text or drift check leaves the previously published trio intact.
    for path in (MD_PATH, PDF_PATH, MANIFEST_PATH):
        _safe_output(path)
    template_hash = sha256_file(TEMPLATE_PATH)
    builder_hash = sha256_file(Path(__file__).resolve())
    _evidence_unchanged(state, template_hash, builder_hash)
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".report-build-", dir=PDF_PATH.parent) as temporary:
        temporary = Path(temporary)
        candidate_pdf, candidate_md, candidate_manifest = (temporary / n for n in ("candidate.pdf", "candidate.md", "manifest.json"))
        render_pdf(md, candidate_pdf, draft=mode == "preview")
        pages, text = pdf_page_count_and_text(candidate_pdf)
        _require(_positive_int(pages), "PDF extraction/page count failed")
        if mode == "final":
            for label, ok in {"title": REPORT_TITLE in text, "tuned model": TUNED_RUN in text,
                              "no draft watermark": "DRAFT - NOT FINAL" not in text,
                              "no figure placeholders": "FIGURE PENDING" not in text,
                              "no unresolved facts": "[[PENDING" not in text and "{{fact:" not in text}.items():
                _require(ok, "PDF text verification failed: " + label)
        else:
            _require("DRAFT" in text and "NOT FINAL" in text, "preview watermark absent from PDF text")
        _evidence_unchanged(state, template_hash, builder_hash)
        candidate_md.write_text(md, encoding="utf-8")
        write_manifest(mode, state, resolved, guards, md, candidate_pdf, pages, destination=candidate_manifest)
        # Commit marker is the manifest, replaced last. Interrupted publication
        # cannot pass the persisted --check unless every output binding matches.
        os.replace(candidate_pdf, PDF_PATH)
        os.replace(candidate_md, MD_PATH)
        os.replace(candidate_manifest, MANIFEST_PATH)
    return pages


def _configure_paths(root=None, template=None, output_dir=None, mode="check"):
    global ROOT, TEMPLATE_PATH, MD_PATH, PDF_PATH, MANIFEST_PATH
    if root is not None:
        ROOT = Path(root).resolve()
    directory = Path(output_dir).resolve() if output_dir else ROOT / "reports/final_five"
    _require(directory.is_relative_to(ROOT.resolve()), "output directory must stay under the selected root")
    TEMPLATE_PATH = Path(template).resolve() if template else ROOT / "reports/final_five/REPORT_TEMPLATE.md"
    _require(TEMPLATE_PATH.is_relative_to(ROOT.resolve()), "template must stay under the selected root")
    if mode == "preview":
        MD_PATH = directory / "TECHNICAL_REPORT.preview.md"
        PDF_PATH = directory / "PathSnap_Final_Technical_Report.preview.pdf"
        MANIFEST_PATH = directory / "report_manifest.preview.json"
    else:
        MD_PATH = directory / "TECHNICAL_REPORT.md"
        PDF_PATH = directory / "PathSnap_Final_Technical_Report.pdf"
        MANIFEST_PATH = directory / "report_manifest.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the evidence-guarded PathSnap technical report.")
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in ("check", "preview", "final"):
        group.add_argument("--" + mode, action="store_true")
    parser.add_argument("--root", help="explicit repository root for isolated test checkout/proposal use")
    parser.add_argument("--template", help="authored token template under root")
    parser.add_argument("--output-dir", help="report directory under root; preview always has distinct names")
    args = parser.parse_args(argv)
    mode = "check" if args.check else "preview" if args.preview else "final"
    try:
        _configure_paths(args.root, args.template, args.output_dir, mode)
        scoped_file(TEMPLATE_PATH.relative_to(ROOT).as_posix())
        template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
        state = build_state(mode)
        resolved = resolve_fact_values(state)
        md = expand_tables(resolve_tokens(template_text, resolved), state)
        fact_rows = evaluate_facts(md, resolved)
        guards = run_final_guards(state, resolved, md, template_text)
        template_problems = scan_template(template_text, check_tokens=True) + scan_template(md, check_tokens=False)
        blocking = collect_readiness(state, fact_rows, guards) + template_problems
        if mode == "check":
            persisted = verify_persisted_manifest(state, resolved)
            if persisted:
                guards.append({"name": "persisted_manifest_current", "ok": False, "detail": "; ".join(persisted)})
                blocking += persisted
            print_check_report(state, resolved, fact_rows, guards, template_problems)
            print("readiness: " + ("READY for --final" if not blocking else "NOT READY"))
            return 0 if not blocking else 1
        if mode == "final" and blocking:
            print("ERROR: --final refused, %d blocking issue(s):" % len(blocking))
            for issue in blocking:
                print("  - " + issue)
            return 1
        if mode == "preview":
            bad = [p for p in template_problems if "token not in fact registry" in p or "unsupported template directive" in p]
            if bad:
                raise ValueError("; ".join(bad))
            md = "> DRAFT PREVIEW - NOT FINAL. Unresolved evidence is not acceptance.\n\n" + md
        if not HAS_PDF:
            raise ValueError("reportlab unavailable")
        pages = _render_and_publish(mode, state, resolved, guards, md)
        print("%s report: %s (%d pages)" % (mode, PDF_PATH, pages))
        print("markdown: %s\nmanifest: %s" % (MD_PATH, MANIFEST_PATH))
        return 0
    except Exception as exc:
        print("ERROR: %s" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
