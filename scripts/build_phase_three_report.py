#!/usr/bin/env python3
"""Guarded builder for the phase-three PathSnap technical report.

Modes (exactly one required):
  --check    read-only readiness audit; no files are written; exit 0 only
             when every final guard would pass
  --preview  render a watermarked draft PDF; unresolved evidence appears
             as red PENDING markers and missing figures as placeholder boxes
  --final    strict mode; renders only when every guard passes and writes
             the resolved fact values back into the markdown source so the
             markdown and the PDF agree literally

Measured metrics use evidence bindings listed in INPUTS. Definitions and
methodological prose require independent review. Figures are hash-checked against
the figure manifests, and the live screenshots must be hash-bound inside
the live evidence JSON files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import textwrap
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, HRFlowable,
                                KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table,
                                TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "reports" / "phase_three" / "TECHNICAL_REPORT.md"
TEMPLATE_PATH = ROOT / "reports" / "phase_three" / "REPORT_TEMPLATE.md"
PDF_PATH = ROOT / "reports" / "phase_three" / "PathSnap_Technical_Report.pdf"
MANIFEST_PATH = ROOT / "reports" / "phase_three" / "report_manifest.json"

BT = chr(96)
CODE_FENCE = BT * 3
TOKEN_RE = re.compile(r"\{\{fact:([A-Za-z0-9_]+)\}\}")
PENDING_MARKER_RE = re.compile(r"\[\[PENDING:([A-Za-z0-9_]+)\]\]")
IMG_LINE_RE = re.compile(r"^!\[[^\]]*\]\(([^)]+)\)\s*$")
LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
ITAL_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
CODE_SPAN_RE = re.compile(BT + r"([^" + BT + r"]+)" + BT)
FORBIDDEN_RE = re.compile(r"\b(?:TODO|TBD|FIXME|PENDING|PLACEHOLDER|XXX)\b")
NAN_RE = re.compile(r"\bNaN\b")

FORBIDDEN_TOKENS = ["TODO", "TBD", "FIXME", "PENDING", "PLACEHOLDER", "XXX", "???", "NaN"]
OLD_PATH_REFS = [
    "reports/final_cli",
    "outputs/student_test/",
    "reports/live_static",
    "reports/interim_live",
    "artifacts/model.joblib",
    "reports/TECHNICAL_REPORT.md",
]
WATERMARK_TEXT = "DRAFT PREVIEW - NOT FINAL EVIDENCE"
PHASE_FAMILIES = ["logistic_regression", "knn_distance_weighted", "bagging"]
SCREENSHOTS = ["ui_desktop.png", "ui_route_map.png", "ui_mobile.png", "api_terminal.png"]
LIVE_SHOT_DIR = ROOT / "reports" / "phase_three" / "live" / "screenshots"
ROUTES_DIR = ROOT / "outputs" / "phase_three_student_test"
FINAL_CLI_DIR = ROOT / "reports" / "phase_three" / "final_cli"
MODEL_PATH = ROOT / "artifacts" / "phase_three" / "model.joblib"
MATCH_CONFIG_PATH = ROOT / "artifacts" / "phase_three" / "match_config.json"


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def inp(key, rel, role, final=True, glob=None):
    return {"key": key, "rel": rel, "role": role, "final": final, "glob": glob}


INPUTS = [
    inp("results3", "experiments/phase_three/results.json", "phase-three learning results (consolidated)"),
    inp("mfreeze", "artifacts/phase_three/model_freeze.json", "phase-three model freeze record"),
    inp("model", "artifacts/phase_three/model.joblib", "phase-three frozen model artifact"),
    inp("mmeta", "artifacts/phase_three/model_metadata.json", "phase-three model metadata"),
    inp("mconfig", "artifacts/phase_three/match_config.json", "phase-three frozen match configuration"),
    inp("selection", "experiments/phase_three/selection.json", "phase-three family selection record"),
    inp("dfreeze", "reports/phase_three/decoder/freeze.json", "phase-three decoder freeze"),
    inp("ablation", "reports/phase_three/decoder/student_ablation.json", "phase-three decoder ablation (all 10 student trips)"),
    inp("finalcli", "reports/phase_three/final_cli", "phase-three final CLI evidence", glob="*.json"),
    inp("routes", "outputs/phase_three_student_test", "phase-three student-test route outputs", glob="*.route"),
    inp("liveapi", "reports/phase_three/live/live_api.json", "phase-three live API evidence (10 successful matches)"),
    inp("browser", "reports/phase_three/live/browser_evidence.json", "phase-three live browser evidence"),
    inp("terminal", "reports/phase_three/live/api_terminal_evidence.json", "phase-three API terminal evidence"),
    inp("public3", "infrastructure/evidence/public-phase-three.json", "phase-three public deployment verification"),
    inp("figman3", "visualizations/phase_three/figure_manifest.json", "phase-three figure manifest"),
    # shared evidence, present on disk
    inp("figman", "visualizations/figure_manifest.json", "shared figure manifest"),
    inp("eda", "visualizations/eda_summary.json", "shared EDA summary"),
    inp("audit", "analysis/data_audit/summary.json", "shared data audit summary"),
    inp("pertrip", "analysis/data_audit/per_trip.json", "shared per-trip data audit"),
    inp("repro", "reports/reproducibility.json", "shared clean-install reproducibility record"),
    inp("hist", "experiments/results.json", "historical five-family results (provenance only)"),
    inp("baseline", "reports/baseline_student_metrics.json", "historical geometry-only baseline (provenance only)"),
    inp("offverify", "reports/phase_three/offline_delivery_verification.json", "phase-three offline delivery verification (route hashes)"),
    inp("routeeval", "reports/phase_three/final_cli/route_evaluation.json", "phase-three final CLI route evaluation summary"),
    inp("knnacc", "reports/phase_three/knn_acceptance.json", "KNN artifact acceptance replay"),
    inp("knnx", "reports/phase_three/knn_runtime_diagnostic.json", "KNN cross-build runtime diagnostic"),
    inp("bagclean", "reports/phase_three/hardened_clean_environment_acceptance.json", "selected-model clean-environment acceptance"),
    inp("maccept", "reports/phase_three/model_acceptance.json", "phase-three model acceptance record"),
    inp("pmparity", "reports/phase_three/public_model_parity_acceptance.json", "public-model parity acceptance (all ten trips)"),
    inp("histcli", "reports/final_cli/route_metrics.json", "historical pre-phase-three CLI route metrics (provenance only)"),
    inp("scalefit", "reports/decoder/scale_fit.json", "historical TRAIN residual scale fit (provenance)"),
    inp("scope", "docs/PHASE_SCOPE.json", "phase scope contract (user instruction record)"),
    inp("delivery", "reports/phase_three/delivery_verification.json", "independent actual all-ten route/state/HTTPS/terminal reconciliation"),
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utcnow():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def collect_inputs():
    state = []
    for e in INPUTS:
        p = ROOT / e["rel"]
        rec = {
            "key": e["key"],
            "path": e["rel"],
            "role": e["role"],
            "final": e["final"],
            "exists": p.exists(),
            "sha256": None,
            "bytes": None,
            "files": None,
            "parse_error": None,
        }
        if e["glob"]:
            files = []
            if p.exists():
                for f in sorted(p.rglob(e["glob"])):
                    files.append({
                        "path": str(f.relative_to(ROOT).as_posix()),
                        "sha256": sha256_file(f),
                        "bytes": f.stat().st_size,
                    })
            rec["files"] = files
        elif p.exists() and p.is_file():
            rec["sha256"] = sha256_file(p)
            rec["bytes"] = p.stat().st_size
        state.append(rec)
    return state


def load_data(inputs_state):
    glob_keys = {e["key"] for e in INPUTS if e["glob"]}
    binary_keys = {"model"}
    data = {}
    for rec in inputs_state:
        key = rec["key"]
        if key in glob_keys:
            data[key] = {"__files__": rec["files"] or []}
            continue
        if key in binary_keys or not rec["exists"]:
            data[key] = None
            continue
        try:
            data[key] = json.loads((ROOT / rec["path"]).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            rec["parse_error"] = str(exc)
            data[key] = None
    return data


# --------------------------------------------------------------------------
# fact DSL
# --------------------------------------------------------------------------

FACTS = []


def fact(name, file, pick, fmt="{:.4f}", row=None, row_dyn=None, md=None,
         optional=False, fallback=None):
    picks = pick if isinstance(pick, list) else [pick]
    FACTS.append({
        "name": name, "file": file, "picks": picks, "fmt": fmt,
        "row": row, "row_dyn": row_dyn, "md": md,
        "optional": optional, "fallback": fallback,
    })


def comp(sel, *fields):
    return ["comparison[" + sel + "]." + f for f in fields]


def agg(method, *fields):
    return ["aggregate." + method + ".trip_macro." + f for f in fields]


FAMILY_ROWS = [
    ("lr", "logistic_regression", "| Logistic regression |"),
    ("knn", "knn_distance_weighted", "| Distance-weighted KNN |"),
    ("bag", "bagging", "| Bagged decision trees |"),
]

for pfx, model_id, anchor in FAMILY_ROWS:
    sel = "model=" + model_id
    fact(pfx + "_params", "results3", comp(sel, "parameters"), fmt="json", row=anchor)
    fact(pfx + "_family", "results3", comp(sel, "family"), fmt="{}", row=anchor)
    fact(pfx + "_cv_f1", "results3", comp(sel, "cv_macro_f1_fold_mean", "macro_f1"), row=anchor)
    fact(pfx + "_cv_f1_std", "results3", comp(sel, "cv_macro_f1_fold_std", "macro_f1_fold_std"), row=anchor)
    fact(pfx + "_cv_top1", "results3", comp(sel, "cv_top1_accuracy", "top1_fold_mean", "top1_proxy_accuracy", "objective"), row=anchor)
    fact(pfx + "_cv_logloss", "results3", comp(sel, "cv_log_loss", "log_loss"), row=anchor)
    fact(pfx + "_student_f1", "results3", comp(sel, "student_macro_f1", "student_test_macro_f1"), row=anchor)
    fact(pfx + "_student_top1", "results3", comp(sel, "student_top1_accuracy", "student_top1", "student_objective"), row=anchor)
    fact(pfx + "_student_logloss", "results3", comp(sel, "student_log_loss", "student_test_log_loss"), row=anchor)
    fact(pfx + "_continuity", "results3", comp(sel, "student_raw_classifier_continuity", "raw_classifier_continuity"), row=anchor)
    fact(pfx + "_route_continuity", "results3", comp(sel, "route_continuity"), row=anchor)
    fact(pfx + "_cv_seconds", "results3", comp(sel, "cv_inference_seconds", "inference_seconds_total"), fmt="{:.3f}", row=anchor)
    fact(pfx + "_fit_seconds", "results3", comp(sel, "fit_all80_seconds", "training_seconds_total"), fmt="{:.1f}", row=anchor)
    fact(pfx + "_obs_edge_f1", "results3", comp(sel, "student_observation_edge_macro_f1", "student_edge_macro_f1"), row=anchor)
    fact(pfx + "_obs_nll", "results3", comp(sel, "student_observation_categorical_log_loss_clipped", "student_observation_categorical_nll", "student_categorical_nll", "student_observation_nll"), row=anchor)
    fact(pfx + "_miss_rate", "results3", comp(sel, "student_candidate_miss_rate", "student_miss_rate"), row=anchor)

# tuned variant of the selected family
tsel = "model=~tuned"
fact("tuned_params", "results3", comp(tsel, "parameters"), fmt="json", row_dyn="| Tuned ")
fact("tuned_family", "results3", comp(tsel, "family"), fmt="{}", row_dyn="| Tuned ")
fact("tuned_cv_f1", "results3", comp(tsel, "cv_macro_f1_fold_mean", "macro_f1"), row_dyn="| Tuned ")
fact("tuned_cv_f1_std", "results3", comp(tsel, "cv_macro_f1_fold_std", "macro_f1_fold_std"), row_dyn="| Tuned ")
fact("tuned_cv_top1", "results3", comp(tsel, "cv_top1_accuracy", "top1_fold_mean", "top1_proxy_accuracy", "objective"), row_dyn="| Tuned ")
fact("tuned_cv_logloss", "results3", comp(tsel, "cv_log_loss", "log_loss"), row_dyn="| Tuned ")
fact("tuned_student_f1", "results3", comp(tsel, "student_macro_f1", "student_test_macro_f1"), row_dyn="| Tuned ")
fact("tuned_student_top1", "results3", comp(tsel, "student_top1_accuracy", "student_top1", "student_objective"), row_dyn="| Tuned ")
fact("tuned_student_logloss", "results3", comp(tsel, "student_log_loss", "student_test_log_loss"), row_dyn="| Tuned ")
fact("tuned_continuity", "results3", comp(tsel, "student_raw_classifier_continuity", "raw_classifier_continuity"), row_dyn="| Tuned ")
fact("tuned_route_continuity", "results3", comp(tsel, "route_continuity"), row_dyn="| Tuned ")
fact("tuned_cv_seconds", "results3", comp(tsel, "cv_inference_seconds", "inference_seconds_total"), fmt="{:.3f}", row_dyn="| Tuned ")
fact("tuned_fit_seconds", "results3", comp(tsel, "fit_all80_seconds", "training_seconds_total"), fmt="{:.1f}", row_dyn="| Tuned ")
fact("tuned_obs_edge_f1", "results3", comp(tsel, "student_observation_edge_macro_f1", "student_edge_macro_f1"), row_dyn="| Tuned ")
fact("tuned_obs_nll", "results3", comp(tsel, "student_observation_categorical_log_loss_clipped", "student_observation_categorical_nll", "student_categorical_nll", "student_observation_nll"), row_dyn="| Tuned ")
fact("tuned_miss_rate", "results3", comp(tsel, "student_candidate_miss_rate", "student_miss_rate"), row_dyn="| Tuned ")

# bayesian tuning block (results3) + selection record
fact("bayes_method", "results3", "bayesian.method", fmt="{}")
fact("bayes_model", "results3", "bayesian.model_name", fmt="{}")
fact("bayes_space", "results3", "bayesian.search_space", fmt="json")
fact("bayes_objective", "results3", "bayesian.objective", fmt="{}")
fact("bayes_trials", "results3", "bayesian.trials_including_measured_prior", fmt="{:.0f}")
fact("bayes_model_trials", "results3", "bayesian.model_based_trials", fmt="{:.0f}")
fact("bayes_seeded_trials", "results3", "bayesian.seeded_initial_design_trials", fmt="{:.0f}")
fact("bayes_best_params", "results3", "bayesian.best_parameters", fmt="json")
fact("bayes_best_obj", "results3", "bayesian.best_objective")
fact("bayes_baseline_obj", "results3", "bayesian.baseline_objective")
fact("bayes_seconds", "results3", "bayesian.seconds", fmt="{:.1f}")
fact("selection_model", "results3", "selection_model", fmt="{}")
fact("sel_model", "selection", "selected_model", fmt="{}")
fact("sel_objective", "selection", "objective_name", fmt="{}")

# run configuration
fact("cfg_folds", "results3", "config.folds", fmt="{:.0f}")
fact("cfg_seed", "results3", "config.seed", fmt="{:.0f}")
fact("cfg_trials", "results3", "config.bayesian_trials", fmt="{:.0f}")
fact("cfg_obs_per_trip", "results3", "config.train_observations_per_trip", fmt="{:.0f}")
fact("cfg_radius", "results3", "config.candidate_radius_m", fmt="{:.0f}")
fact("cfg_maxcand", "results3", "config.max_candidates", fmt="{:.0f}")
fact("cfg_spacing", "results3", "config.sample_spacing_m", fmt="{:.0f}")

# coverage (coverage_summary is nested per split in results.json)
fact("cov_train_trips", "results3", ["coverage_summary.train.trips", "coverage_summary.train_trips"], fmt="{:.0f}")
fact("cov_student_trips", "results3", ["coverage_summary.student_test.trips", "coverage_summary.student_trips"], fmt="{:.0f}")
fact("cov_train_obs", "results3", ["coverage_summary.train.observations", "coverage_summary.train_observations"], fmt="{:,}")
fact("cov_train_total_obs", "results3", ["coverage_summary.train.candidate_rows", "coverage_summary.train_total_observations"], fmt="{:,}")
fact("cov_student_obs", "results3", ["coverage_summary.student_test.observations", "coverage_summary.student_observations"], fmt="{:,}")
fact("cov_student_rows", "results3", ["coverage_summary.student_test.candidate_rows", "coverage_summary.student_candidate_rows"], fmt="{:,}")
fact("cov_train_miss", "results3", ["coverage_summary.train.candidate_miss_count", "coverage_summary.train_candidate_miss_count"], fmt="{:,}")
fact("cov_student_miss", "results3", ["coverage_summary.student_test.candidate_miss_count", "coverage_summary.student_candidate_miss_count"], fmt="{:,}")

# versions
fact("ver_python", "results3", "versions.python", fmt="{}")
fact("ver_numpy", "results3", "versions.numpy", fmt="{}")
fact("ver_sklearn", "results3", ["versions.scikit_learn", "versions.sklearn"], fmt="{}")
fact("ver_scipy", "results3", "versions.scipy", fmt="{}")
fact("repro_numpy_clean", "repro", "versions.numpy", fmt="{}")
fact("repro_sklearn_clean", "repro", "versions.scikit-learn", fmt="{}")
fact("repro_unittests", "repro", "clean_install.unittest_count", fmt="{:,}")

# frozen model artifact
fact("model_sha", "mfreeze", ["model_sha256", "sha256", "artifact_sha256", "joblib_sha256"], fmt="{}")

# decoder freeze
fact("dec_selected", "dfreeze", ["selected_configuration", "selected_configuration.name"], fmt="{}")
fact("dec_sigma", "dfreeze", "config.sigma_m", fmt="{:.3f}")
fact("dec_beta", "dfreeze", "config.beta_m", fmt="{:.3f}")
fact("dec_maxcand", "dfreeze", "config.max_candidates", fmt="{:.0f}")
fact("dec_retry", "dfreeze", "config.candidate_retry_limit", fmt="{:.0f}")
fact("dec_slack", "dfreeze", "config.routing_slack_m", fmt="{:.0f}")
fact("dec_cls_weight", "dfreeze", "config.classifier_weight", fmt="{:.3f}")
fact("dec_geo_weight", "dfreeze", "config.geometry_weight", fmt="{:.3f}")
fact("dec_frozen_utc", "dfreeze", "frozen_utc", fmt="{}")

# decoder ablation
fact("abl_trips", "ablation", "trip_count", fmt="{:.0f}")
fact("abl_elapsed", "ablation", "elapsed_seconds", fmt="{:.1f}")

ABL_FIELDS = [
    ("proxy_acc", ("top1_proxy_accuracy", "proxy_accuracy", "top1"), "{:.4f}"),
    ("edge_f1", ("route_edge_f1", "edge_macro_f1", "edge_f1", "route_edge_macro_f1"), "{:.4f}"),
    ("len_f1", ("route_length_f1", "length_f1", "len_f1", "route_length_macro_f1"), "{:.4f}"),
    ("edit", ("route_edit_similarity", "edit_similarity"), "{:.4f}"),
    ("cont", ("route_continuity", "continuity"), "{:.4f}"),
    ("nll", ("categorical_emission_log_loss", "categorical_nll", "observation_nll", "nll"), "{:.4f}"),
    ("miss", ("candidate_miss_rate", "miss_rate"), "{:.4f}"),
    ("seconds", ("inference_seconds", "seconds", "elapsed_seconds"), "{:.1f}"),
]
ABL_METHODS = [
    ("raw", "classifier_only_raw", False),
    ("hmm", "classifier_plus_HMM", False),
    ("geo", "geometry_HMM_default_reused", True),
]
for pfx, method, opt in ABL_METHODS:
    for suffix, alts, fmt in ABL_FIELDS:
        fact("abl_" + pfx + "_" + suffix, "ablation", agg(method, *alts), fmt=fmt,
             optional=opt, fallback="not re-run in this phase")
    for suffix, alt in (("pooled_len_f1", "route_length_f1"), ("pooled_edge_f1", "route_edge_f1")):
        fact("abl_" + pfx + "_" + suffix, "ablation",
             ["aggregate." + method + ".pooled." + alt,
              "aggregate." + method + ".pooled." + alt.replace("route_edge", "edge_macro").replace("route_length", "length")],
             optional=True, fallback="not recorded")

# dataset / audit facts
fact("ds_trips", "audit", "trip_count", fmt="{:,}")
fact("ds_files", "audit", "file_count", fmt="{:,}")
fact("ds_nodes", "audit", "row_totals_by_extension.nodes", fmt="{:,}")
fact("ds_arcs", "audit", "row_totals_by_extension.arcs", fmt="{:,}")
fact("ds_route_rows", "audit", "row_totals_by_extension.route", fmt="{:,}")
fact("ds_track_rows", "audit", "row_totals_by_extension.track", fmt="{:,}")
fact("ds_train_trips", "audit", "splits.train.trip_count", fmt="{:.0f}")
fact("ds_student_trips", "audit", "splits.student_test.trip_count", fmt="{:.0f}")
fact("ds_train_broken_joins", "audit", "splits.train.route_arc_offset_0.broken_joins", fmt="{:,}")
fact("ds_student_broken_joins", "audit", "splits.student_test.route_arc_offset_0.broken_joins", fmt="{:,}")
fact("ds_student_repeated_arcs", "audit", "splits.student_test.route_validity.repeated_arc_extra_occurrences", fmt="{:,}")
fact("ds_student_multicomponent_trips", "audit", "len:splits.student_test.trips_with_route_more_than_one_strong_component", fmt="{:.0f}")
fact("ds_student_repeated_trips", "audit", "len:splits.student_test.trips_with_route_repeated_arcs", fmt="{:.0f}")
fact("ds_trip76_broken", "pertrip", "trip=00000076.route_validity.arc_index_offset_0.broken_joins", fmt="{:,}")
fact("ds_gps_train", "eda", "gps_rows.train", fmt="{:,}")
fact("ds_gps_student", "eda", "gps_rows.student_test", fmt="{:,}")
fact("ds_one_second_steps", "eda", "timestamps.train.approximately_one_second_steps", fmt="{:,}")
fact("ds_max_gap", "eda", "timestamps.train.maximum_gap_seconds", fmt="{:,}")
fact("ds_train_strong", "eda", "road_network_totals.train.strong_components", fmt="{:,}")
fact("ds_train_weak", "eda", "road_network_totals.train.weak_components", fmt="{:,}")
fact("ds_student_weak", "eda", "road_network_totals.student_test.weak_components", fmt="{:,}")
fact("ds_baseline_edit", "baseline", "summary.macro.edit_similarity")

# live / deployment (public origin and its checks come from the live and
# public verification records; the stale pre-phase deployment-status file is
# deliberately not an input for any claim in this report)
fact("live_matches", "liveapi", "len:matches", fmt="{:.0f}")
fact("live_url", "liveapi", "base_url", fmt="{}", optional=True, fallback="not recorded")
fact("live_checked_at", "liveapi", ["completed_at_utc", "collected_at_utc", "checked_at", "generated_at_utc", "verified_at_utc"], fmt="{}", optional=True, fallback="not recorded")
fact("public_checked_at", "public3", ["checked_at", "verified_at_utc", "generated_at_utc"], fmt="{}", optional=True, fallback="not recorded")
fact("public_target", "public3", ["target", "url", "public_url"], fmt="{}", optional=True, fallback="not recorded")
fact("public_model_name", "public3", "health.model_name", fmt="{}", optional=True, fallback="not recorded")

# KNN reference bank policy (bound, not narrated)
fact("knn_bank_rows", "results3", "knn_bank_policy.budget", fmt="{:,}")

# runtime determinism records
fact("knnacc_replay_delta", "knnacc", "fixture_max_abs_probability_delta", fmt="{:.1f}")
fact("knnacc_bank_rows", "knnacc", "reference_bank_rows", fmt="{:,}")
fact("knnx_max_delta", "knnx", "max_abs_delta", fmt="{:.2e}")
fact("knnx_fixture", "knnx", "fixture", fmt="{}")
fact("knnx_rows_over", "knnx", "rows_delta_over_1e12", fmt="{:.0f}")
fact("bagclean_rows", "bagclean", "predictions", fmt="{:,}")
fact("bagclean_delta", "bagclean", "max_abs_probability_delta", fmt="{:.1f}")

# final CLI route evaluation (delivered routes)
fact("re_len_f1", "routeeval", "summary.macro.length_f1")
fact("re_edge_f1", "routeeval", "summary.macro.edge_f1")
fact("re_edit_sim", "routeeval", "summary.macro.edit_similarity")
fact("re_connected", "routeeval", "summary.connected_prediction_count", fmt="{:.0f}")
fact("re_exact", "routeeval", "summary.exact_route_count", fmt="{:.0f}")
fact("re_ref_discont", "routeeval", "summary.reference_discontinuities", fmt="{:.0f}")

# decoder configuration search (in-sample, TRAIN trips only)
fact("dec_search_trips", "dfreeze", "len:calibration_trip_ids", fmt="{:.0f}")
fact("dec_search_selected", "dfreeze", "selected_configuration", fmt="{}")
fact("dec_search_a_f1", "dfreeze", "objectives[configuration=A_fitted].mean_length_f1")
fact("dec_search_b_f1", "dfreeze", "objectives[configuration=B_twice_beta].mean_length_f1")
fact("dec_search_c_f1", "dfreeze", "objectives[configuration=C_classifier_up].mean_length_f1")

# historical pre-phase-three delivery (provenance only)
fact("hist_cli_len_f1", "histcli", "summary.macro.length_f1")
fact("hist_cli_edge_f1", "histcli", "summary.macro.edge_f1")

# historical provenance (appendix only)
fact("hist_lr_cv_f1", "hist", comp("model=logistic_regression", "cv_macro_f1_fold_mean"))
fact("hist_gnb_cv_f1", "hist", comp("model=gaussian_nb", "cv_macro_f1_fold_mean"))
fact("hist_rf_cv_f1", "hist", comp("model=random_forest", "cv_macro_f1_fold_mean"))
fact("hist_et_cv_f1", "hist", comp("model=extra_trees", "cv_macro_f1_fold_mean"))
fact("hist_hgb_cv_f1", "hist", comp("model=hist_gradient_boosting", "cv_macro_f1_fold_mean"))
fact("hist_baseline_len_f1", "baseline", "summary.macro.length_f1")
fact("hist_baseline_edge_f1", "baseline", "summary.macro.edge_f1")
fact("hist_numpy", "hist", "versions.numpy", fmt="{}")

# computed deltas: resolved from raw fact values (never from rendered text),
# rendered with an explicit sign so wins and losses read as they are
COMPUTED = []


def computed(name, left, right=None, right_from=None, fmt="{:+.4f}"):
    COMPUTED.append({"name": name, "left": left, "right": right,
                     "right_from": right_from, "fmt": fmt})


for _pfx in ("lr", "knn", "bag", "tuned"):
    computed(_pfx + "_gap_f1", _pfx + "_student_f1", _pfx + "_cv_f1")
    computed(_pfx + "_gap_top1", _pfx + "_student_top1", _pfx + "_cv_top1")
# tuned vs the selected family's untuned CV macro F1; the right operand is
# looked up dynamically from the selection record (for example bagging_cv_f1)
computed("tuned_gain_cv_f1", "tuned_cv_f1", right_from="sel_model")
# historical pre-phase-three delivery minus the current delivered length F1;
# rendered with an explicit sign so the direction is never narrated away
computed("hist_cli_len_delta", "hist_cli_len_f1", "re_len_f1")

FACTS_BY_NAME = {f["name"]: f for f in list(FACTS) + list(COMPUTED)}


# --------------------------------------------------------------------------
# dig: dotted-key-tolerant deep access with list selectors
# --------------------------------------------------------------------------

def dig(obj, pick):
    pick = pick.strip()
    if pick.startswith("len:"):
        return len(dig(obj, pick[4:]))
    segs = []
    for raw in pick.split("."):
        if raw == "":
            continue
        # expand bracketed selectors: comparison[model=bagging].parameters
        # becomes the two segments "comparison", "model=bagging"
        while "[" in raw:
            head, rest = raw.split("[", 1)
            sel, raw = rest.split("]", 1)
            if head:
                segs.append(head)
            if sel:
                segs.append(sel)
        if raw:
            segs.append(raw)
    cur = obj
    i = 0
    while i < len(segs):
        seg = segs[i]
        if isinstance(cur, list):
            hit = None
            if seg == "model=~tuned":
                matches = [r for r in cur
                           if str(r.get("model", "")) not in PHASE_FAMILIES
                           and str(r.get("model", "")).rsplit("_tuned", 1)[0] in PHASE_FAMILIES]
                if len(matches) > 1:
                    raise KeyError("multiple tuned comparison rows")
                hit = matches[0] if matches else None
            elif "=" in seg:
                # generic entry selector: any list of dicts keyed by a field,
                # for example objectives[configuration=A_fitted] or
                # comparison[model=bagging] or trip=00000076
                key, want = seg.split("=", 1)
                field = "trip_id" if key == "trip" else key
                for r in cur:
                    if isinstance(r, dict) and str(r.get(field, "")) == want:
                        hit = r
                        break
            elif seg.isdigit():
                idx = int(seg)
                if -len(cur) <= idx < len(cur):
                    hit = cur[idx]
            if hit is None:
                raise KeyError("list selector %r matched nothing" % seg)
            cur = hit
            i += 1
            continue
        if isinstance(cur, dict):
            # dicts in this project contain literal dotted keys (for example
            # "route_arc_offset_0.broken_joins"), so try the longest join of
            # the remaining segments as a single key first
            for k in range(min(6, len(segs) - i), 0, -1):
                key = ".".join(segs[i:i + k])
                if key in cur:
                    cur = cur[key]
                    i += k
                    break
            else:
                raise KeyError("key %r not found (remaining pick: %s)" % (segs[i], ".".join(segs[i:])))
            continue
        raise KeyError("cannot descend into %r at segment %r" % (type(cur).__name__, seg))
    return cur


def render_value(value, fmt):
    if value is None:
        return None
    if fmt == "json":
        if isinstance(value, str):
            return value.strip()
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    if fmt == "{}":
        return str(value)
    try:
        return fmt.format(value)
    except (TypeError, ValueError):
        return str(value)


def resolve_fact_values(data):
    values = {}
    for f in FACTS:
        rec = {
            "name": f["name"], "value": None, "pick": None, "error": None,
            "optional": f["optional"], "file": f["file"], "raw": None,
        }
        src = data.get(f["file"])
        if src is None:
            rec["error"] = "input %s not loadable" % f["file"]
        else:
            last = None
            for pick in f["picks"]:
                try:
                    v = dig(src, pick)
                except Exception as exc:  # noqa: BLE001
                    last = "%s (%s)" % (pick, exc)
                    continue
                if v is None:
                    last = "%s (null)" % pick
                    continue
                rendered = render_value(v, f["fmt"])
                if rendered is None:
                    last = "%s (null)" % pick
                    continue
                rec["value"] = rendered
                rec["raw"] = v
                rec["pick"] = pick
                break
            if rec["value"] is None and rec["error"] is None:
                rec["error"] = "unresolved: %s" % (last or "no picks")
        values[f["name"]] = rec
    resolve_computed(values)
    return values


def resolve_computed(values):
    for c in COMPUTED:
        rec = {
            "name": c["name"], "value": None, "pick": "computed", "error": None,
            "optional": False, "file": "computed", "raw": None,
        }
        try:
            lv = values.get(c["left"], {}).get("raw")
            if c.get("right_from"):
                sel = values.get(c["right_from"], {}).get("raw")
                # translate the selected model id to its table fact prefix
                # (for example "bagging" -> "bag")
                pfx = next((p for p, mid, _a in FAMILY_ROWS if mid == sel), str(sel))
                rname = pfx + "_cv_f1"
            else:
                rname = c["right"]
            rv = values.get(rname, {}).get("raw")
            if not isinstance(lv, (int, float)) or isinstance(lv, bool):
                raise ValueError("left operand %r unresolved or not numeric" % c["left"])
            if not isinstance(rv, (int, float)) or isinstance(rv, bool):
                raise ValueError("right operand %r unresolved or not numeric" % rname)
            d = lv - rv
            rec["raw"] = d
            rec["value"] = c["fmt"].format(d)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = "computed: %s" % exc
        values[c["name"]] = rec


def resolve_tokens(md_text, values):
    def repl(m):
        name = m.group(1)
        rec = values.get(name)
        if rec is not None and rec.get("value") is not None:
            return rec["value"]
        f = FACTS_BY_NAME.get(name)
        if f is not None and f.get("optional") and f.get("fallback"):
            return f["fallback"]
        return "[[PENDING:%s]]" % name
    return TOKEN_RE.sub(repl, md_text)


def evaluate_facts(resolved_md, values):
    lines = resolved_md.splitlines()
    out = []
    for f in list(FACTS) + list(COMPUTED):
        rec = dict(values[f["name"]])
        rec["ok"] = False
        if rec["value"] is None:
            if f.get("optional"):
                rec["ok"] = True
                rec["note"] = "optional fact unresolved; fallback text used"
                rec["error"] = None
        else:
            rendered = rec["value"]
            anchor = f.get("row") or f.get("row_dyn")
            if anchor:
                rows = [ln for ln in lines if ln.strip().startswith(anchor)]
                if not rows:
                    rec["error"] = "row anchor %r not found" % anchor
                elif not any(rendered in ln for ln in rows):
                    rec["error"] = "value not present in rows starting %r" % anchor
                else:
                    rec["ok"] = True
            else:
                if rendered in resolved_md:
                    rec["ok"] = True
                else:
                    rec["error"] = "value not present in text"
            if f.get("md") and rec["ok"]:
                want = f["md"].format(value=rendered)
                if want not in resolved_md:
                    rec["ok"] = False
                    rec["error"] = "md template %r not found" % f["md"]
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def md_images(md_text):
    out = []
    for line in md_text.splitlines():
        m = IMG_LINE_RE.match(line.strip())
        if m:
            out.append(m.group(1).strip())
    return out


def manifest_entries(data):
    entries = {}
    for key in ("figman", "figman3"):
        d = data.get(key)
        figs = None
        if isinstance(d, list):
            figs = d
        elif isinstance(d, dict):
            figs = d.get("figures") if isinstance(d.get("figures"), list) else None
            if figs is None and d and all(isinstance(v, dict) for v in d.values()):
                figs = list(d.values())
        for e in figs or []:
            if isinstance(e, dict):
                p = e.get("path") or e.get("file") or e.get("rel")
                s = e.get("sha256")
                if p and s:
                    entries[str(p)] = str(s)
    return entries


def verify_figures(md_text, data):
    results = []
    entries = manifest_entries(data)
    for path in md_images(md_text):
        rel = str(Path(path).as_posix())
        p = ROOT / rel
        rec = {"path": rel, "exists": p.exists(), "error": None, "manifest": rel in entries}
        if not p.exists():
            rec["error"] = "file missing"
        elif rel.startswith("reports/phase_three/live/"):
            # live screenshots are hash-bound inside the live evidence JSONs;
            # that binding is enforced by the screenshots_hash_bound guard
            rec["manifest"] = "live-evidence-guard"
        elif rel in entries:
            if sha256_file(p) != entries[rel]:
                rec["error"] = "sha256 mismatch vs figure manifest"
        else:
            rec["error"] = "not listed in a figure manifest"
        results.append(rec)
    return results


# --------------------------------------------------------------------------
# text scans
# --------------------------------------------------------------------------

def scan_tokens(raw_md, resolved_md):
    # forbidden wording and stale paths are authored text: scan the template.
    # [[PENDING:...]] markers are by design in drafts and guarded separately.
    forbidden = sorted(set(FORBIDDEN_RE.findall(raw_md)))
    old_paths = sorted(p for p in OLD_PATH_REFS if p in raw_md)
    if NAN_RE.search(resolved_md) or NAN_RE.search(raw_md):
        forbidden.append("NaN")
    return {
        "forbidden": forbidden,
        "old_paths": old_paths,
        "pending_markers": PENDING_MARKER_RE.findall(resolved_md),
    }


# --------------------------------------------------------------------------
# final guards
# --------------------------------------------------------------------------

def iter_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_strings(v)


def run_phase_guards(data, resolved_md, raw_md=None):
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    res = data.get("results3") if isinstance(data.get("results3"), dict) else {}

    try:
        rows = res.get("comparison") if isinstance(res.get("comparison"), list) else []
        base = sorted(str(r.get("model", "")) for r in rows if str(r.get("model", "")) in PHASE_FAMILIES)
        tuned = [str(r.get("model", "")) for r in rows if str(r.get("model", "")) not in PHASE_FAMILIES]
        add("three_families",
            base == sorted(PHASE_FAMILIES) and len(tuned) <= 1
            and all(t.rsplit("_tuned", 1)[0] in PHASE_FAMILIES for t in tuned),
            "base rows: %r; other rows: %r" % (base, tuned))
    except Exception as exc:  # noqa: BLE001
        add("three_families", False, "error: %s" % exc)

    actual_model_sha = sha256_file(MODEL_PATH) if MODEL_PATH.exists() else None
    declared_mfreeze = None
    try:
        mf = data.get("mfreeze") or {}
        for key in ("model_sha256", "sha256", "artifact_sha256", "joblib_sha256"):
            v = mf.get(key)
            if isinstance(v, str) and len(v) == 64:
                declared_mfreeze = v
                break
        add("model_freeze_hash",
            declared_mfreeze is not None and actual_model_sha is not None and declared_mfreeze == actual_model_sha,
            "model_freeze.json declares %s; actual sha256 of artifacts/phase_three/model.joblib is %s"
            % (declared_mfreeze, actual_model_sha))
    except Exception as exc:  # noqa: BLE001
        add("model_freeze_hash", False, "error: %s" % exc)

    try:
        df = data.get("dfreeze") or {}
        df_sha = df.get("model_sha256")
        cfg_sha = sha256_file(MATCH_CONFIG_PATH) if MATCH_CONFIG_PATH.exists() else None
        ab = data.get("ablation") or {}
        ab_cfg = None
        frz = ab.get("freeze")
        if isinstance(frz, dict):
            ab_cfg = frz.get("config_file_sha256")
        chain_ok = (
            isinstance(df_sha, str) and df_sha == actual_model_sha
            and declared_mfreeze is not None and declared_mfreeze == df_sha
            and (ab_cfg is None or ab_cfg == cfg_sha)
        )
        add("decoder_freeze_chain", chain_ok,
            "decoder freeze model_sha256=%s; model_freeze=%s; ablation config_file_sha256=%s; actual match_config sha256=%s"
            % (df_sha, declared_mfreeze, ab_cfg, cfg_sha))
    except Exception as exc:  # noqa: BLE001
        add("decoder_freeze_chain", False, "error: %s" % exc)

    try:
        ab = data.get("ablation") or {}
        failed = ab.get("failed_trip_ids") or []
        agg = ab.get("aggregate") or {}
        add("ablation_all_ten",
            ab.get("trip_count") == 10 and not failed
            and ab.get("all_ten_headline_eligible") is True
            and "classifier_only_raw" in agg and "classifier_plus_HMM" in agg,
            "trip_count=%r failed=%r eligible=%r methods=%r"
            % (ab.get("trip_count"), failed, ab.get("all_ten_headline_eligible"), sorted(agg)[:6]))
    except Exception as exc:  # noqa: BLE001
        add("ablation_all_ten", False, "error: %s" % exc)

    try:
        cfg = res.get("config") or {}
        bay = res.get("bayesian") or {}
        add("bayesian_trials_declared",
            cfg.get("bayesian_trials") == 8 and bay.get("trials_including_measured_prior") == 8,
            "config.bayesian_trials=%r; bayesian.trials_including_measured_prior=%r (phase pins 8)"
            % (cfg.get("bayesian_trials"), bay.get("trials_including_measured_prior")))
    except Exception as exc:  # noqa: BLE001
        add("bayesian_trials_declared", False, "error: %s" % exc)

    try:
        la = data.get("liveapi") or {}
        matches = la.get("matches")
        # live_api.json records http_status per match (no "status" key)
        statuses = [m.get("http_status") for m in (matches or [])]
        la_frz = la.get("freeze") or {}
        add("live_api_ten_successes",
            isinstance(matches, list) and len(matches) == 10
            and all(s == 200 for s in statuses)
            and la.get("successful_count") == 10
            and la.get("mocked") is False
            and la_frz.get("model_sha256") == actual_model_sha,
            "matches=%r http_statuses=%r successful_count=%r mocked=%r freeze_model_sha=%s"
            % (len(matches) if isinstance(matches, list) else matches,
               sorted(set(statuses)), la.get("successful_count"), la.get("mocked"),
               "matches frozen model" if la_frz.get("model_sha256") == actual_model_sha else "MISMATCH"))
    except Exception as exc:  # noqa: BLE001
        add("live_api_ten_successes", False, "error: %s" % exc)

    try:
        ev_strings = set()
        for ev in (data.get("liveapi"), data.get("browser"), data.get("terminal")):
            if ev is not None:
                ev_strings.update(iter_strings(ev))
        missing = []
        unbound = []
        shot_hashes = {}
        for name in SCREENSHOTS:
            p = LIVE_SHOT_DIR / name
            if not p.exists():
                missing.append(name)
                continue
            h = sha256_file(p)
            shot_hashes[name] = h
            if h not in ev_strings:
                unbound.append(name)
        term = data.get("terminal") or {}
        term_strings = set(iter_strings(term))
        term_ok = "api_terminal.png" in shot_hashes and shot_hashes["api_terminal.png"] in term_strings
        add("screenshots_hash_bound",
            not missing and not unbound and term_ok,
            "missing=%r unbound=%r terminal_binds_api_terminal=%r" % (missing, unbound, term_ok))
    except Exception as exc:  # noqa: BLE001
        add("screenshots_hash_bound", False, "error: %s" % exc)

    try:
        # bound to infrastructure/evidence/public-phase-three.json alone; the
        # pre-phase deployment-status.json is stale and backs nothing here
        p3 = data.get("public3")
        p3ok = isinstance(p3, dict)
        health = p3.get("health") if p3ok else {}
        trip_ids = p3.get("trip_ids") if p3ok else None
        la_frz = (data.get("liveapi") or {}).get("freeze") or {}
        add("public_site_verified",
            p3ok and p3.get("stage") == "ready"
            and isinstance(health, dict) and health.get("status") == "ready"
            and health.get("model_loaded") is True
            and health.get("model_hash") == actual_model_sha
            and health.get("match_config_hash") == la_frz.get("api_compact_config_sha256")
            and isinstance(trip_ids, list) and len(trip_ids) == 10
            and str(p3.get("target") or "").startswith("https://"),
            "stage=%r health.status=%r model_hash_matches=%r config_hash_matches=%r trip_ids=%r target=%r"
            % (p3.get("stage") if p3ok else None,
               health.get("status") if isinstance(health, dict) else None,
               health.get("model_hash") == actual_model_sha if isinstance(health, dict) else None,
               health.get("match_config_hash") == la_frz.get("api_compact_config_sha256") if isinstance(health, dict) else None,
               len(trip_ids) if isinstance(trip_ids, list) else trip_ids,
               p3.get("target") if p3ok else None))
    except Exception as exc:  # noqa: BLE001
        add("public_site_verified", False, "error: %s" % exc)

    try:
        route_files = sorted(ROUTES_DIR.rglob("*.route")) if ROUTES_DIR.exists() else []
        add("ten_cli_routes", len(route_files) == 10,
            "route files under outputs/phase_three_student_test: %d (%s)"
            % (len(route_files), ", ".join(p.name for p in route_files[:12])))
    except Exception as exc:  # noqa: BLE001
        add("ten_cli_routes", False, "error: %s" % exc)

    try:
        # the ten delivered route files are hash-bound per trip inside
        # reports/phase_three/offline_delivery_verification.json
        ov = data.get("offverify") or {}
        ov_trips = ov.get("trips") if isinstance(ov.get("trips"), dict) else {}
        ov_checks = ov.get("checks") if isinstance(ov.get("checks"), dict) else {}
        route_files = sorted(ROUTES_DIR.rglob("*.route")) if ROUTES_DIR.exists() else []
        mismatches = []
        for p in route_files:
            rec = ov_trips.get(p.stem)
            if not isinstance(rec, dict):
                mismatches.append("%s: no verification record" % p.name)
                continue
            if sha256_file(p) != rec.get("route_sha256"):
                mismatches.append("%s: route sha256 mismatch" % p.name)
        jfiles = sorted(FINAL_CLI_DIR.rglob("*.json")) if FINAL_CLI_DIR.exists() else []
        checks_ok = all(v is True for v in ov_checks.values())
        add("final_cli_evidence",
            len(route_files) == 10 and not mismatches and checks_ok
            and ov.get("model_sha256") == actual_model_sha and bool(jfiles),
            "route files=%d bound=%d checks_all_true=%r model_sha_matches=%r final_cli json=%d mismatches=%r"
            % (len(route_files), len(route_files) - len([m for m in mismatches if "no verification" in m]),
               checks_ok, ov.get("model_sha256") == actual_model_sha, len(jfiles), mismatches[:3]))
    except Exception as exc:  # noqa: BLE001
        add("final_cli_evidence", False, "error: %s" % exc)

    try:
        # every trip's decoded geometry must be identical through the public
        # HTTP API and the offline CLI delivery (per-trip sha256 parity)
        la = data.get("liveapi") or {}
        ov = data.get("offverify") or {}
        ov_trips = ov.get("trips") if isinstance(ov.get("trips"), dict) else {}
        lam = {m.get("trip_id"): m for m in (la.get("matches") or []) if isinstance(m, dict)}
        diffs = []
        for tid, rec in ov_trips.items():
            m = lam.get(tid)
            if m is None:
                diffs.append("%s: not in live evidence" % tid)
                continue
            if m.get("matched_edge_array_sha256") != rec.get("matched_edge_array_sha256"):
                diffs.append("%s: edge array" % tid)
            if m.get("predicted_geometry_sha256") != rec.get("predicted_geometry_sha256"):
                diffs.append("%s: geometry" % tid)
        add("live_offline_parity",
            len(ov_trips) == 10 and len(lam) == 10 and not diffs,
            "trips compared=%d live matches=%d differences=%r"
            % (len(ov_trips), len(lam), diffs[:4]))
    except Exception as exc:  # noqa: BLE001
        add("live_offline_parity", False, "error: %s" % exc)

    try:
        ma = data.get("maccept") or {}
        ka = data.get("knnacc") or {}
        pm = data.get("pmparity") or {}
        add("acceptance_records",
            ma.get("passed") is True and ka.get("passed") is True
            and pm.get("passed") is True and pm.get("all_ten_passed") is True
            and pm.get("model_sha256") == actual_model_sha,
            "model_acceptance=%r knn_acceptance=%r public_parity=%r all_ten=%r parity_model_sha_matches=%r"
            % (ma.get("passed"), ka.get("passed"), pm.get("passed"), pm.get("all_ten_passed"),
               pm.get("model_sha256") == actual_model_sha))
    except Exception as exc:  # noqa: BLE001
        add("acceptance_records", False, "error: %s" % exc)

    try:
        kn = data.get("knnx") or {}
        bc = data.get("bagclean") or {}
        add("runtime_determinism",
            kn.get("binary_threshold_decisions_equal") is True
            and bc.get("passed") is True and bc.get("model_sha256") == actual_model_sha
            and bc.get("max_abs_probability_delta") == 0.0
            and (data.get("knnacc") or {}).get("fixture_max_abs_probability_delta") == 0.0,
            "knn_cross_build_binary_equal=%r clean_env_passed=%r clean_env_model_sha_matches=%r "
            "clean_env_delta=%r original_runtime_replay_delta=%r"
            % (kn.get("binary_threshold_decisions_equal"), bc.get("passed"),
               bc.get("model_sha256") == actual_model_sha, bc.get("max_abs_probability_delta"),
               (data.get("knnacc") or {}).get("fixture_max_abs_probability_delta")))
    except Exception as exc:  # noqa: BLE001
        add("runtime_determinism", False, "error: %s" % exc)

    try:
        dfz = data.get("dfreeze") or {}
        # the freeze record transparently declares that earlier-phase student
        # results existed as prior context (prior_student_context_available);
        # the guarded contract is that the predeclared search itself never
        # read or used student results when scoring the configurations
        add("decoder_student_blindness",
            dfz.get("student_results_seen_for_selection") is False
            and dfz.get("student_results_used_for_selection") is False,
            "student_results_seen_for_selection=%r used_for_selection=%r (prior_student_context_available=%r is declared context, not search input)"
            % (dfz.get("student_results_seen_for_selection"),
               dfz.get("student_results_used_for_selection"),
               dfz.get("prior_student_context_available")))
    except Exception as exc:  # noqa: BLE001
        add("decoder_student_blindness", False, "error: %s" % exc)

    try:
        delivery = data.get("delivery") or {}
        trip_records = delivery.get("trips") or {}
        checks_global = delivery.get("checks") or {}
        add("complete_delivery_reconciliation",
            delivery.get("passed") is True and delivery.get("defects") == []
            and delivery.get("model_sha256") == actual_model_sha
            and delivery.get("config_file_sha256") == cfg_sha
            and len(trip_records) == 10 and bool(checks_global)
            and all(v is True for v in checks_global.values())
            and all(isinstance(v.get("checks"), dict) and v["checks"]
                    and all(x is True for x in v["checks"].values())
                    for v in trip_records.values()),
            "Independent all-ten routes, original GPS state order, public HTTP, browser and receiving-journal reconciliation")
    except Exception as exc:
        add("complete_delivery_reconciliation", False, "error: %s" % exc)

    try:
        deferred = (data.get("scope") or {}).get("deferred_evaluator_website_request") or {}
        add("new_evaluator_request_deferred",
            deferred.get("human_reported_trip_count") == 10
            and deferred.get("used_for_training_selection_or_decoder") is False
            and "not inspected or evaluated" in deferred.get("status", ""),
            "Ten new trips are user-reported, not inventoried or assessed; separate website work is deferred")
    except Exception as exc:
        add("new_evaluator_request_deferred", False, "error: %s" % exc)

    try:
        scan = scan_tokens(raw_md if raw_md is not None else resolved_md, resolved_md)
        add("no_stale_path_references", not scan["old_paths"], "references: %r" % scan["old_paths"])
        add("no_pending_markers", not scan["pending_markers"],
            "pending tokens: %r" % scan["pending_markers"][:20])
    except Exception as exc:  # noqa: BLE001
        add("no_stale_path_references", False, "error: %s" % exc)

    return checks


# --------------------------------------------------------------------------
# markdown -> pdf
# --------------------------------------------------------------------------

MARGIN = 18 * mm
PAGE_W, PAGE_H = A4
FRAME_W = PAGE_W - 2 * MARGIN


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(text):
    s = esc(text)
    s = LINK_RE.sub(lambda m: '<link href="%s" color="#1a56a0"><u>%s</u></link>' % (m.group(2), m.group(1)), s)
    s = CODE_SPAN_RE.sub(lambda m: '<font face="Courier" size="8">%s</font>' % m.group(1), s)
    s = BOLD_RE.sub(r"<b>\1</b>", s)
    s = ITAL_RE.sub(r"<i>\1</i>", s)
    return s


def build_styles():
    ss = getSampleStyleSheet()
    st = {}
    st["Title"] = ParagraphStyle("Title", parent=ss["Title"], fontName="Helvetica-Bold",
                                 fontSize=19, leading=24, spaceAfter=10, spaceBefore=0)
    st["H2"] = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=13.5, leading=17,
                              spaceBefore=16, spaceAfter=6, keepWithNext=1, textColor=colors.HexColor("#10233a"))
    st["H3"] = ParagraphStyle("H3", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                              spaceBefore=12, spaceAfter=4, keepWithNext=1)
    st["H4"] = ParagraphStyle("H4", fontName="Helvetica-Bold", fontSize=10.2, leading=13,
                              spaceBefore=9, spaceAfter=3, keepWithNext=1)
    st["Body"] = ParagraphStyle("Body", fontName="Helvetica", fontSize=9.5, leading=13.8,
                                spaceAfter=6, alignment=TA_JUSTIFY)
    st["Bullet"] = ParagraphStyle("Bullet", fontName="Helvetica", fontSize=9.5, leading=13.4,
                                  leftIndent=14, bulletIndent=4, spaceAfter=3)
    st["Quote"] = ParagraphStyle("Quote", fontName="Helvetica-Oblique", fontSize=9.5, leading=13.8,
                                 leftIndent=10, rightIndent=6, spaceBefore=6, spaceAfter=8,
                                 textColor=colors.HexColor("#404040"), borderPadding=4,
                                 borderColor=colors.HexColor("#b8c4d0"), borderWidth=0.7)
    st["Code"] = ParagraphStyle("Code", fontName="Courier", fontSize=7.6, leading=10,
                                backColor=colors.HexColor("#F4F6F8"), borderColor=colors.HexColor("#C8D0DA"),
                                borderWidth=0.5, borderPadding=4, spaceBefore=4, spaceAfter=8)
    st["Caption"] = ParagraphStyle("Caption", fontName="Helvetica", fontSize=8, leading=10.5,
                                   textColor=colors.HexColor("#5a6570"), spaceBefore=3, spaceAfter=10)
    st["Cell"] = ParagraphStyle("Cell", fontName="Helvetica", fontSize=8, leading=10.4)
    st["CellHead"] = ParagraphStyle("CellHead", fontName="Helvetica-Bold", fontSize=8, leading=10.4)
    st["TOCTitle"] = ParagraphStyle("TOCTitle", fontName="Helvetica-Bold", fontSize=13.5, leading=17,
                                    spaceBefore=6, spaceAfter=6, textColor=colors.HexColor("#10233a"))
    st["TOC1"] = ParagraphStyle("TOC1", fontName="Helvetica-Bold", fontSize=9.5, leading=14, leftIndent=0)
    st["TOC2"] = ParagraphStyle("TOC2", fontName="Helvetica", fontSize=9, leading=12.5, leftIndent=14)
    return st


def parse_markdown(md_text):
    blocks = []
    lines = md_text.splitlines()
    i = 0
    n = len(lines)
    para = []

    def flush_para():
        if para:
            blocks.append({"type": "p", "text": " ".join(x.strip() for x in para)})
            para.clear()

    while i < n:
        stripped = lines[i].strip()
        if re.match(r"^" + CODE_FENCE + r"[A-Za-z0-9_-]*$", stripped):
            flush_para()
            i += 1
            code = []
            while i < n and lines[i].strip() != CODE_FENCE:
                code.append(lines[i])
                i += 1
            i += 1
            blocks.append({"type": "code", "text": "\n".join(code)})
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            flush_para()
            blocks.append({"type": "h", "level": len(m.group(1)), "text": m.group(2).strip()})
            i += 1
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            flush_para()
            inner = stripped[4:-3].strip()
            if inner == "pagebreak":
                blocks.append({"type": "pagebreak"})
            elif inner == "toc":
                blocks.append({"type": "toc"})
            i += 1
            continue
        if re.match(r"^-{3,}$", stripped):
            flush_para()
            blocks.append({"type": "hr"})
            i += 1
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped)
        if m:
            flush_para()
            blocks.append({"type": "img", "alt": m.group(1).strip(), "path": m.group(2).strip()})
            i += 1
            continue
        if stripped.startswith("|") and stripped.count("|") >= 2:
            flush_para()
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                row_line = lines[i].strip()
                if not re.match(r"^\|[\s:\-|]+\|$", row_line):
                    tbl.append([c.strip() for c in row_line.strip("|").split("|")])
                i += 1
            caption = None
            if i < n and lines[i].strip().startswith("Table:"):
                caption = lines[i].strip()[len("Table:"):].strip()
                i += 1
            blocks.append({"type": "table",
                           "header": tbl[0] if tbl else [],
                           "rows": tbl[1:],
                           "caption": caption})
            continue
        if stripped.startswith(">"):
            flush_para()
            qlines = []
            while i < n and lines[i].strip().startswith(">"):
                qlines.append(lines[i].strip().lstrip(">").strip())
                i += 1
            blocks.append({"type": "quote", "text": " ".join(q for q in qlines if q)})
            continue
        if stripped.startswith("- ") or stripped.startswith("* "):
            flush_para()
            items = []
            while i < n and (lines[i].strip().startswith("- ") or lines[i].strip().startswith("* ")):
                items.append(lines[i].strip()[2:].strip())
                i += 1
            blocks.append({"type": "ul", "items": items})
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m:
            flush_para()
            items = []
            while i < n:
                mm = re.match(r"^(\d+)\.\s+(.*)$", lines[i].strip())
                if not mm:
                    break
                items.append(mm.group(2).strip())
                i += 1
            blocks.append({"type": "ol", "items": items})
            continue
        if stripped == "":
            flush_para()
            i += 1
            continue
        para.append(lines[i])
        i += 1
    flush_para()
    return blocks


def table_flowable(header, rows, styles):
    ncols = max(len(header), max((len(r) for r in rows), default=0))
    header = list(header) + [""] * (ncols - len(header))
    fs = 6.8 if ncols >= 8 else (7.4 if ncols >= 5 else 8.2)
    cell_style = ParagraphStyle("cell-%d" % ncols, parent=styles["Cell"], fontSize=fs, leading=fs + 2.2)
    head_style = ParagraphStyle("cellh-%d" % ncols, parent=styles["CellHead"], fontSize=fs, leading=fs + 2.2)

    def cell(text, head=False):
        return Paragraph(inline(str(text)), head_style if head else cell_style)

    weights = []
    for c in range(ncols):
        w = min(len(str(header[c])), 60)
        for r in rows:
            if c < len(r):
                w = max(w, min(len(str(r[c])), 60))
        weights.append(max(w, 4))
    total = float(sum(weights))
    widths = [FRAME_W * w / total for w in weights]
    data = [[cell(h, True) for h in header]]
    for r in rows:
        r = list(r) + [""] * (ncols - len(r))
        data.append([cell(v) for v in r])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF3")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9AA5B1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")]),
    ]))
    return t


class FigureImage(Flowable):
    def __init__(self, path):
        Flowable.__init__(self)
        self._path = Path(path)
        self._reader = None
        self._w = 0
        self._h = 0

    def wrap(self, avail_width, avail_height):
        max_w = min(avail_width, FRAME_W)
        max_h = 175 * mm
        if self._path.exists():
            if self._reader is None:
                self._reader = ImageReader(str(self._path))
            iw, ih = self._reader.getSize()
            s = min(max_w / float(iw), max_h / float(ih))
            self._w, self._h = iw * s, ih * s
        else:
            self._w, self._h = max_w * 0.72, 42 * mm
        return self._w, self._h

    def draw(self):
        if self._reader is None:
            self.canv.saveState()
            self.canv.setStrokeColor(colors.HexColor("#B00020"))
            self.canv.setFillColor(colors.HexColor("#FDECEA"))
            self.canv.rect(0, 0, self._w, self._h, stroke=1, fill=1)
            self.canv.setFillColor(colors.HexColor("#B00020"))
            self.canv.setFont("Helvetica-Bold", 10)
            self.canv.drawCentredString(self._w / 2.0, self._h / 2.0 + 4, "MISSING FIGURE")
            self.canv.setFont("Helvetica", 8)
            label = str(self._path)
            if len(label) > 96:
                label = "..." + label[-93:]
            self.canv.drawCentredString(self._w / 2.0, self._h / 2.0 - 8, label)
            self.canv.restoreState()
        else:
            self.canv.drawImage(self._reader, 0, 0, self._w, self._h)

    def __deepcopy__(self, memo):
        c = FigureImage.__new__(FigureImage)
        c.__dict__.update({k: v for k, v in self.__dict__.items() if k != "_reader"})
        c._reader = None
        return c


def build_story(blocks, styles):
    story = []
    fig_no = [0]
    tab_no = [0]
    for b in blocks:
        t = b["type"]
        if t == "h":
            lvl = min(b["level"], 4)
            style = styles["Title"] if lvl == 1 else styles["H%d" % lvl]
            story.append(Paragraph(inline(b["text"]), style))
        elif t == "p":
            story.append(Paragraph(inline(b["text"]), styles["Body"]))
        elif t == "quote":
            story.append(Paragraph(inline(b["text"]), styles["Quote"]))
        elif t == "hr":
            story.append(HRFlowable(width="100%", thickness=0.6,
                                    color=colors.HexColor("#BBBBBB"), spaceBefore=6, spaceAfter=6))
        elif t == "pagebreak":
            story.append(PageBreak())
        elif t == "toc":
            story.append(Paragraph("Contents", styles["TOCTitle"]))
            toc = TableOfContents()
            toc.levelStyles = [styles["TOC1"], styles["TOC2"]]
            toc.dotsMinLevel = 0
            story.append(toc)
            story.append(PageBreak())
        elif t == "code":
            wrapped = []
            for ln in b["text"].splitlines():
                if len(ln) > 104:
                    wrapped.extend(textwrap.wrap(ln, width=104, subsequent_indent="    ",
                                                  break_long_words=True, break_on_hyphens=False))
                else:
                    wrapped.append(ln)
            story.append(Preformatted("\n".join(wrapped), styles["Code"]))
        elif t == "img":
            fig_no[0] += 1
            img = FigureImage(b["path"])
            cap = Paragraph(esc("Figure %d: %s" % (fig_no[0], b["alt"])), styles["Caption"])
            story.append(KeepTogether([img, cap]))
            story.append(Spacer(1, 4))
        elif t == "ul":
            for item in b["items"]:
                story.append(Paragraph(inline(item), styles["Bullet"], bulletText="\u2022"))
            story.append(Spacer(1, 3))
        elif t == "ol":
            for idx, item in enumerate(b["items"], 1):
                story.append(Paragraph(inline(item), styles["Bullet"], bulletText="%d." % idx))
            story.append(Spacer(1, 3))
        elif t == "table":
            tab_no[0] += 1
            tf = table_flowable(b["header"], b["rows"], styles)
            parts = [tf]
            if b.get("caption"):
                parts.append(Paragraph(esc("Table %d: %s" % (tab_no[0], b["caption"])), styles["Caption"]))
            if len(b["rows"]) <= 12:
                story.append(KeepTogether(parts))
            else:
                story.extend(parts)
            story.append(Spacer(1, 6))
    return story


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename, watermark=False):
        BaseDocTemplate.__init__(self, filename, pagesize=A4,
                                 leftMargin=MARGIN, rightMargin=MARGIN,
                                 topMargin=22 * mm, bottomMargin=20 * mm,
                                 title="PathSnap Technical Report - Phase Three",
                                 author="PathSnap team",
                                 subject="Map matching with a learned emission model and HMM route reconstruction")
        self._watermark = watermark
        self._toc_seq = 0
        frame = Frame(MARGIN, 20 * mm, FRAME_W, PAGE_H - 42 * mm, id="body")
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=self._on_page)])

    def _on_page(self, canv, doc):
        canv.saveState()
        if self._watermark:
            canv.saveState()
            canv.setFont("Helvetica-Bold", 40)
            canv.setFillColor(colors.Color(0.78, 0.16, 0.16, alpha=0.28))
            canv.translate(PAGE_W / 2.0, PAGE_H / 2.0)
            canv.rotate(36)
            canv.drawCentredString(0, 0, WATERMARK_TEXT)
            canv.restoreState()
        if doc.page > 1:
            canv.setFont("Helvetica", 7.5)
            canv.setFillColor(colors.HexColor("#666666"))
            canv.drawString(MARGIN, PAGE_H - 13 * mm, "PathSnap - phase-three technical report")
            canv.drawRightString(PAGE_W - MARGIN, PAGE_H - 13 * mm,
                                 "development scores only; evaluator unassessed")
        canv.restoreState()

    def handle_documentBegin(self):
        # multiBuild runs several passes; TOC entry keys must be identical
        # across passes or the index never converges.
        self._toc_seq = 0
        BaseDocTemplate.handle_documentBegin(self)

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        name = flowable.style.name
        if name not in ("H2", "H3"):
            return
        text = flowable.getPlainText()
        level = 0 if name == "H2" else 1
        self._toc_seq += 1
        key = "toc-%04d" % self._toc_seq
        self.canv.bookmarkPage(key)
        self.notify("TOCEntry", (level, text, self.page, key))


class NumberedCanvas(pdfcanvas.Canvas):
    def __init__(self, *args, **kwargs):
        pdfcanvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_states = []

    def showPage(self):
        self._saved_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_states)
        for state in self._saved_states:
            self.__dict__.update(state)
            self.draw_footer(total)
            pdfcanvas.Canvas.showPage(self)
        pdfcanvas.Canvas.save(self)

    def draw_footer(self, total):
        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#666666"))
        self.drawString(MARGIN, 10 * mm, "Generated by scripts/build_phase_three_report.py")
        stamp = getattr(self, "_render_stamp", "")
        if stamp:
            self.drawCentredString(PAGE_W / 2.0, 10 * mm, stamp)
        self.drawRightString(PAGE_W - MARGIN, 10 * mm, "Page %d of %d" % (self._pageNumber, total))
        self.restoreState()


def make_canvasmaker(stamp):
    class StampedCanvas(NumberedCanvas):
        _render_stamp = stamp
    return StampedCanvas


def render_pdf(md_text, out_path, mode):
    styles = build_styles()
    blocks = parse_markdown(md_text)
    story = build_story(blocks, styles)
    doc = ReportDoc(str(out_path), watermark=(mode == "preview"))
    stamp = "%s UTC - %s render" % (utcnow(), mode.upper())
    doc.multiBuild(story, canvasmaker=make_canvasmaker(stamp))
    return doc.page


def extract_pdf_text(path):
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts), len(reader.pages)


def extraction_checks(pdf_text, n_pages, md_text, mode):
    problems = []
    if mode == "final":
        if WATERMARK_TEXT in pdf_text:
            problems.append("watermark text present in final PDF")
        if n_pages < 10:
            problems.append("final PDF has only %d pages" % n_pages)
        for token in FORBIDDEN_TOKENS:
            if token == "???":
                if "???" in pdf_text:
                    problems.append("forbidden token ??? in extracted text")
            elif re.search(r"\b%s\b" % re.escape(token), pdf_text):
                problems.append("forbidden token %r in extracted text" % token)
    else:
        if WATERMARK_TEXT not in pdf_text:
            problems.append("preview watermark missing")
    blocks = parse_markdown(md_text)
    for b in blocks:
        if b["type"] == "h" and b["level"] == 2:
            probe = b["text"][:24]
            if probe and probe not in pdf_text:
                problems.append("heading not found in PDF text: %r" % b["text"][:60])
    n_imgs = len([b for b in blocks if b["type"] == "img"])
    found = len(re.findall(r"Figure \d+:", pdf_text))
    if found < n_imgs:
        problems.append("figure captions in PDF text %d < %d images" % (found, n_imgs))
    if mode == "final" and "MISSING FIGURE" in pdf_text:
        problems.append("missing figure placeholder in final PDF")
    return problems


# --------------------------------------------------------------------------
# readiness
# --------------------------------------------------------------------------

def run_readiness(raw_md, resolved_md, values, data, inputs_state):
    fact_eval = evaluate_facts(resolved_md, values)
    figures = verify_figures(raw_md, data)
    scan = scan_tokens(raw_md, resolved_md)
    guards = run_phase_guards(data, resolved_md, raw_md=raw_md)
    unknown_tokens = sorted(set(TOKEN_RE.findall(raw_md)) - set(FACTS_BY_NAME))
    missing_final = [r["path"] for r in inputs_state
                     if r["final"] and not r["exists"]
                     and not (r["files"] is not None and r["files"])]
    facts_without_tokens = sorted(
        f["name"] for f in list(FACTS) + list(COMPUTED)
        if not f.get("optional") and ("{{fact:%s}}" % f["name"]) not in raw_md)
    ready = (
        not missing_final
        and all(r["ok"] for r in fact_eval)
        and not any(fg["error"] for fg in figures)
        and not unknown_tokens
        and not scan["forbidden"]
        and not scan["old_paths"]
        and not scan["pending_markers"]
        and all(g["ok"] for g in guards)
    )
    return {
        "fact_eval": fact_eval,
        "figures": figures,
        "scan": scan,
        "guards": guards,
        "unknown_tokens": unknown_tokens,
        "missing_final_inputs": missing_final,
        "facts_without_tokens": facts_without_tokens,
        "ready": ready,
    }


def print_readiness(r):
    print("== phase-three report readiness ==")
    print("mode target: FINAL (three families + tuned, all guards)")
    if r["missing_final_inputs"]:
        print("missing final inputs (%d):" % len(r["missing_final_inputs"]))
        for p in r["missing_final_inputs"]:
            print("  - %s" % p)
    bad = [x for x in r["fact_eval"] if not x["ok"]]
    good = [x for x in r["fact_eval"] if x["ok"]]
    print("facts: %d ok, %d unresolved/error" % (len(good), len(bad)))
    for x in bad[:40]:
        print("  - %s: %s" % (x["name"], x["error"]))
    fig_bad = [f for f in r["figures"] if f["error"]]
    print("figures: %d referenced, %d with problems" % (len(r["figures"]), len(fig_bad)))
    for f in fig_bad[:20]:
        print("  - %s: %s" % (f["path"], f["error"]))
    if r["unknown_tokens"]:
        print("unknown tokens in template: %r" % r["unknown_tokens"])
    if r["facts_without_tokens"]:
        print("required facts with no token in template: %r" % r["facts_without_tokens"])
    if r["scan"]["forbidden"]:
        print("forbidden tokens in text: %r" % r["scan"]["forbidden"])
    if r["scan"]["old_paths"]:
        print("stale path references: %r" % r["scan"]["old_paths"])
    if r["scan"]["pending_markers"]:
        print("pending evidence tokens (%d): %r" % (len(r["scan"]["pending_markers"]),
                                                    r["scan"]["pending_markers"][:25]))
    failed = [g for g in r["guards"] if not g["ok"]]
    print("guards: %d/%d pass" % (len(r["guards"]) - len(failed), len(r["guards"])))
    for g in failed:
        print("  - FAIL %s: %s" % (g["name"], g["detail"]))
    print("overall: %s" % ("READY FOR --final" if r["ready"] else "NOT READY"))


def write_manifest(payload):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Guarded phase-three technical report builder")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="read-only readiness audit (no writes)")
    g.add_argument("--preview", action="store_true", help="render watermarked draft PDF")
    g.add_argument("--final", action="store_true", help="render final PDF only if every guard passes")
    args = ap.parse_args(argv)

    if not MD_PATH.exists():
        print("markdown template missing: %s" % MD_PATH)
        return 2

    source_path = TEMPLATE_PATH if TEMPLATE_PATH.exists() else MD_PATH
    raw_md = source_path.read_text(encoding="utf-8")
    inputs_state = collect_inputs()
    data = load_data(inputs_state)
    values = resolve_fact_values(data)
    resolved_md = resolve_tokens(raw_md, values)
    readiness = run_readiness(raw_md, resolved_md, values, data, inputs_state)

    n_tokens = len(TOKEN_RE.findall(raw_md))
    n_resolved = n_tokens - len(readiness["scan"]["pending_markers"])

    def manifest_payload(mode, pdf_pages=None, blocked=None, problems=None):
        return {
            "schema": "pathsnap-phase-three-report-manifest",
            "version": 1,
            "template": {"path": str(source_path.relative_to(ROOT)), "sha256": sha256_file(source_path)},
            "builder_sha256": sha256_file(Path(__file__)),
            "mode": mode,
            "generated_at_utc": utcnow(),
            "ready": readiness["ready"] if mode != "final" else (blocked is None or not blocked),
            "blocked": blocked,
            "markdown": {
                "path": str(MD_PATH.relative_to(ROOT).as_posix()),
                "sha256": sha256_file(MD_PATH),
                "tokens_total": n_tokens,
                "tokens_resolved": n_resolved,
                "pending_tokens": readiness["scan"]["pending_markers"],
            },
            "pdf": (None if pdf_pages is None else {
                "path": str(PDF_PATH.relative_to(ROOT).as_posix()),
                "sha256": sha256_file(PDF_PATH),
                "pages": pdf_pages,
            }),
            "extraction_problems": problems or [],
            "inputs": inputs_state,
            "facts": readiness["fact_eval"],
            "figures": readiness["figures"],
            "guards": readiness["guards"],
        }

    if args.check:
        print_readiness(readiness)
        return 0 if readiness["ready"] else 1

    if args.preview:
        pages = render_pdf(resolved_md, PDF_PATH, "preview")
        pdf_text, n_pages = extract_pdf_text(PDF_PATH)
        problems = extraction_checks(pdf_text, n_pages, raw_md, "preview")
        write_manifest(manifest_payload("preview", pdf_pages=n_pages, problems=problems))
        print("PREVIEW written: %s (%d pages, watermarked)" % (PDF_PATH, pages))
        for p in problems:
            print("  note: %s" % p)
        return 0

    # --final
    blockers = []
    for p in readiness["missing_final_inputs"]:
        blockers.append("missing final input: %s" % p)
    for rec in readiness["fact_eval"]:
        if not rec["ok"]:
            blockers.append("fact %s: %s" % (rec["name"], rec["error"]))
    for fg in readiness["figures"]:
        if fg["error"]:
            blockers.append("figure %s: %s" % (fg["path"], fg["error"]))
    if readiness["unknown_tokens"]:
        blockers.append("unknown tokens in template: %r" % readiness["unknown_tokens"])
    for hit in readiness["scan"]["forbidden"]:
        blockers.append("forbidden token in text: %s" % hit)
    for hit in readiness["scan"]["old_paths"]:
        blockers.append("stale path reference: %s" % hit)
    for name in readiness["scan"]["pending_markers"]:
        blockers.append("unresolved evidence token: %s" % name)
    for gd in readiness["guards"]:
        if not gd["ok"]:
            blockers.append("guard failed: %s (%s)" % (gd["name"], gd["detail"]))

    if blockers:
        print("FINAL BLOCKED - %d blocker(s):" % len(blockers))
        for b in blockers:
            print("  - %s" % b)
        write_manifest(manifest_payload("final-blocked", blocked=blockers))
        return 1

    try:
        pages = render_pdf(resolved_md, PDF_PATH, "final")
    except Exception as exc:  # noqa: BLE001
        print("final render failed: %s" % exc)
        write_manifest(manifest_payload("final-blocked", blocked=["render failed: %s" % exc]))
        return 1

    pdf_text, n_pages = extract_pdf_text(PDF_PATH)
    problems = extraction_checks(pdf_text, n_pages, raw_md, "final")
    if problems:
        PDF_PATH.unlink(missing_ok=True)
        print("FINAL BLOCKED - extraction problems after render:")
        for p in problems:
            print("  - %s" % p)
        write_manifest(manifest_payload("final-blocked", blocked=problems))
        return 1

    if TOKEN_RE.search(raw_md):
        MD_PATH.write_text(resolved_md, encoding="utf-8")
    write_manifest(manifest_payload("final", pdf_pages=n_pages, problems=problems))
    print("FINAL report written: %s (%d pages)" % (PDF_PATH, pages))
    print("markdown source with resolved values: %s" % MD_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
