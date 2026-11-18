#!/usr/bin/env python3
"""Independent post-freeze audit for phase three (three-model-first).

Read-only: reruns NO training and NO inference. Verifies record/artifact
consistency by recomputation from saved arrays. Writes only its own report,
analysis/review/phase_three/audit_report.json, and exits nonzero on any FAIL.

Usage (clean environment, from the repository root):
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        .venv-clean/bin/python analysis/review/phase_three/audit_phase_three.py

Pre-freeze the script reports MISSING for absent stages and exits 1; it is
intended to pass only after select + hpo + freeze + student + decoder + live
evidence are complete. WARN results (source-hash drift) require human
re-review of the changed file, not a code change here.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mapmatch.experiments import _metrics  # noqa: E402
from mapmatch.phase_models import PHASE_SEARCH_SPACES  # noqa: E402


def load_prediction_npz(path: Path) -> dict:
    """Same layout as run_phase_three.prediction_arrays (independent copy)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

PHASE = ROOT / "experiments" / "phase_three"
ARTIFACTS = ROOT / "artifacts" / "phase_three"
DECODER = ROOT / "reports" / "phase_three" / "decoder"
LIVE = ROOT / "reports" / "phase_three" / "live"
CACHE = ROOT / "experiments" / "cache"

FAMILIES = ("logistic_regression", "knn_distance_weighted", "bagging")
EXPECTED = {
    "train": {"obs": 28000, "rows": 269463, "trips": 80, "misses": 578},
    "student_test": {"obs": 19160, "rows": 175778, "trips": 10, "misses": 19},
}
HPO_TOTAL, HPO_SEEDED, HPO_PRIOR = 8, 2, 1
BASELINE_HASHES = {
    "experiments/phase_three/run_phase_three.py":
        "dac7f23afb61c04208d70b6096eddd7ef5352847f1196a82b98e49e925d5df5c",
    "mapmatch/phase_models.py":
        "f9500e31b0a7a57ea50536001572637d1527c3e93d0d39520aa4b707979e063d",
    "scripts/phase_three_decoder.py":
        "f52f27fe2cb6899a54e41720351009a94b9b6b3f83826bb684eec51ee7edd384",
}
METRIC_KEYS = ("macro_f1", "log_loss", "top1_accuracy", "trip_macro_top1_accuracy")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(Path(path).read_text())


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def record(self, name: str, status: str, detail) -> None:
        self.checks.append({"name": name, "status": status, "detail": detail})
        print(f"[{status}] {name}: {detail}")

    def check(self, name: str, condition: bool, detail, warn_only: bool = False) -> bool:
        if condition:
            self.record(name, "PASS", detail)
        else:
            self.record(name, "WARN" if warn_only else "FAIL", detail)
        return bool(condition)

    def missing(self, name: str, detail) -> None:
        self.record(name, "MISSING", detail)


def as_dataset(arrays: dict) -> SimpleNamespace:
    return SimpleNamespace(y=arrays["y"], offsets=arrays["offsets"],
                           observation_trip_ids=arrays["observation_trip_ids"])


def recompute_metrics(npz_path: Path, cache_arrays: dict | None = None) -> dict:
    stored = load_prediction_npz(npz_path)
    if cache_arrays is not None:
        for key in ("y", "offsets", "observation_trip_ids"):
            if not np.array_equal(stored[key], cache_arrays[key]):
                raise AssertionError(f"{npz_path.name}: {key} differs from cache")
    return _metrics(as_dataset(stored), stored["probabilities"])


def domain_ok(npz_path: Path) -> tuple[bool, str]:
    stored = load_prediction_npz(npz_path)
    p = np.asarray(stored["probabilities"], float)
    ok = (np.isfinite(p).all() and p.min() >= 1e-7 - 1e-15 and p.max() <= 1 - 1e-7 + 1e-15)
    return ok, f"min={p.min():.3e} max={p.max():.3e} n={len(p)}"


def load_cache(split: str) -> dict:
    pattern = "train_*.npz" if split == "train" else "student_test_*.npz"
    paths = sorted(CACHE.glob(pattern))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected exactly one {pattern} in {CACHE}")
    with np.load(paths[0], allow_pickle=False) as z:
        return {k: z[k] for k in ("y", "offsets", "observation_trip_ids", "X")}


def audit() -> Audit:
    a = Audit()

    # 1. Source drift vs the audited revision --------------------------------
    for rel, want in BASELINE_HASHES.items():
        got = sha256(ROOT / rel)
        a.check(f"source_hash:{rel}", got == want,
                "unchanged" if got == want else f"CHANGED {got[:16]}... (was {want[:16]}...); re-review required",
                warn_only=True)

    # 2. Cache integrity and miss recomputation ------------------------------
    caches: dict[str, dict] = {}
    for split, spec in EXPECTED.items():
        try:
            arrays = load_cache(split)
        except FileNotFoundError as exc:
            a.missing(f"cache:{split}", str(exc))
            continue
        y, offs, trips, X = arrays["y"], arrays["offsets"], arrays["observation_trip_ids"], arrays["X"]
        positive = np.concatenate([[0], np.cumsum(y == 1)])
        seg = positive[offs[1:]] - positive[offs[:-1]]
        miss = int((seg == 0).sum())
        empty = int((np.diff(offs) == 0).sum())
        ok = (len(offs) - 1 == spec["obs"] and len(y) == spec["rows"]
              and len(np.unique(trips)) == spec["trips"] and miss == spec["misses"]
              and empty == 0 and np.array_equal(np.unique(y), [0, 1])
              and X.shape[1] == 38 and str(X.dtype) == "float32")
        a.check(f"cache:{split}", ok,
                f"obs={len(offs)-1} rows={len(y)} trips={len(np.unique(trips))} "
                f"misses={miss} empty={empty} dtype={X.dtype}")
        caches[split] = arrays

    # 3. CV records: parameter bounds + OOF metric recomputation -------------
    for fam in FAMILIES:
        path = PHASE / "cv" / f"{fam}.json"
        if not path.exists():
            a.missing(f"cv:{fam}", "no CV record yet")
            continue
        rec = load_json(path)
        space = PHASE_SEARCH_SPACES.get(fam, {})
        in_bounds = all(spec[1] <= rec["parameters"][k] <= spec[2] for k, spec in space.items()
                        if k in rec["parameters"])
        a.check(f"cv:{fam}:parameters_in_bounds", in_bounds,
                f"parameters={rec['parameters']} space_dim={len(space)}")
        npz = PHASE / "predictions" / f"{fam}_oof.npz"
        shared_npz = ROOT / "experiments" / "predictions" / f"{fam}_oof.npz"
        source = npz if npz.exists() else (shared_npz if shared_npz.exists() else None)
        if source is None:
            a.missing(f"cv:{fam}:oof", "no OOF prediction file (runner may point elsewhere)")
            continue
        try:
            recomputed = recompute_metrics(source, caches.get("train"))
            deltas = {k: abs(recomputed[k] - rec["oof_metrics"][k]) for k in METRIC_KEYS}
            a.check(f"cv:{fam}:oof_recompute", max(deltas.values()) <= 1e-9,
                    f"source={source.name} max_delta={max(deltas.values()):.2e} "
                    f"macro_f1={recomputed['macro_f1']:.6f} trip_top1={recomputed['trip_macro_top1_accuracy']:.6f} "
                    f"misses_in_denominator={recomputed['observations'] == EXPECTED['train']['obs']}")
        except (AssertionError, KeyError) as exc:
            a.check(f"cv:{fam}:oof_recompute", False, str(exc))

    # 4. Selection ------------------------------------------------------------
    sel_path = PHASE / "selection.json"
    if not sel_path.exists():
        a.missing("selection", "run --stage select first")
    else:
        sel = load_json(sel_path)
        scores = sel.get("scores", {})
        def rank(v):
            return (v["objective"], v["macro_f1"], -v["log_loss"])
        expected_pick = max(FAMILIES, key=lambda n: rank(scores[n]))
        a.check("selection", (
            set(scores) == set(FAMILIES) and sel["selected_model"] == expected_pick
            and sel.get("student_test_consulted") is False),
            f"selected={sel['selected_model']} (recomputed argmax={expected_pick}) "
            f"over={sorted(scores)} student_consulted={sel.get('student_test_consulted')}")

    # 5. HPO records ----------------------------------------------------------
    trials_path = PHASE / "bayesian_trials.jsonl"
    hpo_path = PHASE / "bayesian_optimization.json"
    if not trials_path.exists() or not hpo_path.exists():
        a.missing("hpo", "run --stage hpo first")
    else:
        records = [json.loads(line) for line in trials_path.read_text().splitlines() if line.strip()]
        hist = {}
        for r in records:
            hist[r["proposal"]] = hist.get(r["proposal"], 0) + 1
        expected_hist = {"measured_baseline_prior": HPO_PRIOR,
                         "seeded_initial_design": HPO_SEEDED,
                         "gaussian_process_expected_improvement": HPO_TOTAL - HPO_PRIOR - HPO_SEEDED}
        a.check("hpo:budget", len(records) == HPO_TOTAL and hist == expected_hist,
                f"records={len(records)} histogram={hist} expected={expected_hist}")
        winner = max(records, key=lambda r: (r["objective"], r["cv_macro_f1"], -r["cv_log_loss"]))
        best = load_json(hpo_path)
        a.check("hpo:winner_consistent", (
            best["best_parameters"] == winner["parameters"]
            and abs(best["best_objective"] - winner["objective"]) <= 1e-12
            and abs(best["improvement"] - (winner["objective"] - best["baseline_objective"])) <= 1e-12),
            f"best={best['best_parameters']} objective={best['best_objective']:.6f} "
            f"improvement={best['improvement']:+.6f}")
        for r in records:
            for k, spec in PHASE_SEARCH_SPACES.get(best["model_name"], {}).items():
                if k in r["parameters"] and not (spec[1] <= r["parameters"][k] <= spec[2]):
                    a.check("hpo:parameters_in_bounds", False,
                            f"trial {r['trial']} {k}={r['parameters'][k]} outside {spec[1:]}")
                    break

    # 6. Freeze ---------------------------------------------------------------
    freeze_path = ARTIFACTS / "model_freeze.json"
    if not freeze_path.exists():
        a.missing("freeze", "run --stage freeze first")
    else:
        fr = load_json(freeze_path)
        model = ROOT / fr["model"]
        a.check("freeze:model_sha256", sha256(model) == fr["sha256"] and model.stat().st_size == fr["bytes"],
                f"sha={fr['sha256'][:16]}... bytes={fr['bytes']}")
        meta = load_json(ARTIFACTS / "model_metadata.json")
        hpo = load_json(hpo_path) if hpo_path.exists() else {}
        a.check("freeze:parameters_match_hpo", (
            meta["parameters"] == fr["parameters"]
            and (not hpo or fr["parameters"] == hpo.get("best_parameters"))),
            f"freeze={fr['parameters']} metadata=equal hpo_best={'equal' if hpo else 'n/a'}")
        clean = ARTIFACTS / "clean_load_report.json"
        if not clean.exists():
            a.missing("freeze:clean_load_report", "missing")
        else:
            cl = load_json(clean)
            a.check("freeze:clean_load", cl.get("passed") is True,
                    f"passed={cl.get('passed')} python={cl.get('python')} "
                    f"versions_numpy={cl.get('versions', {}).get('numpy')} "
                    "(PH3-D002: verify these versions are the CHILD's, not the parent's)")
        import joblib
        bundle = joblib.load(model)
        a.check("freeze:bundle", (
            hasattr(bundle.get("estimator"), "predict_proba")
            and bundle["feature_names"] == meta["feature_names"]
            and bundle.get("feature_version") == meta["feature_version"]),
            f"estimator={type(bundle['estimator']).__name__} features={len(bundle['feature_names'])}")
        replay = fr.get("final_oof_export") or meta.get("final_oof_export")
        if replay and replay.get("oof_predictions_file"):
            rp = ROOT / replay["oof_predictions_file"]
            try:
                recomputed = recompute_metrics(rp, caches.get("train"))
                tuned = [m for m in (PHASE / "cv").glob(f"{fr['model_name']}_tuned.json")]
                ref = load_json(tuned[0])["oof_metrics"] if tuned else None
                delta = max(abs(recomputed[k] - ref[k]) for k in METRIC_KEYS) if ref else None
                a.check("freeze:oof_replay", ref is None or delta <= 1e-6,
                        f"replayed agreement max_delta={delta} "
                        f"replayed={'confirmatory' if replay.get('replayed') else 'reused_baseline'}")
            except (AssertionError, KeyError) as exc:
                a.check("freeze:oof_replay", False, str(exc))

    # 7. Student stage: domain + metric recomputation vs results.json ---------
    results_path = PHASE / "results.json"
    student_npzs = sorted((PHASE / "predictions").glob("*student*.npz"))
    if results_path.exists():
        values = load_json(results_path).get("evaluations", {})
        values = values.values() if isinstance(values, dict) else values
        for evaluation in values:
            recorded = evaluation.get("student_predictions_file") or evaluation.get("provenance", {}).get("student_predictions_file")
            if recorded:
                path = (ROOT / recorded).resolve()
                if not path.is_relative_to(ROOT):
                    raise AssertionError("student prediction pointer escapes repository")
                if path not in [p.resolve() for p in student_npzs]:
                    student_npzs.append(path)
    if not results_path.exists() and not student_npzs:
        a.missing("student", "run --stage student first")
    else:
        recomputed: dict[str, dict] = {}
        for npz in student_npzs:
            ok, detail = domain_ok(npz)
            a.check(f"student:domain:{npz.name}", ok, detail)
            try:
                recomputed[npz.name] = recompute_metrics(npz, caches.get("student_test"))
            except (AssertionError, KeyError) as exc:
                a.check(f"student:recompute:{npz.name}", False, str(exc))
        if results_path.exists():
            res = load_json(results_path)
            rows = res.get("evaluations", [])
            rows = list(rows.values()) if isinstance(rows, dict) else rows
            matched = unmatched = 0
            for row in rows:
                sm = row.get("student_metrics")
                if not sm:
                    continue
                if any(all(abs(m[k] - sm[k]) <= 1e-9 for k in METRIC_KEYS)
                       for m in recomputed.values()):
                    matched += 1
                else:
                    unmatched += 1
            a.check("student:results_rows", unmatched == 0 and matched > 0,
                    f"rows={len(rows)} student_metric_rows={matched} unmatched={unmatched} "
                    f"recomputed_files={len(recomputed)}")
            comp = res.get("comparison", [])
            a.check("student:comparison_schema", len(comp) == len(rows),
                    f"comparison_rows={len(comp)} evaluation_rows={len(rows)}")

    # 8. Decoder freeze --------------------------------------------------------
    dec_freeze = DECODER / "freeze.json"
    if not dec_freeze.exists():
        a.missing("decoder", "decoder has not frozen a match config yet")
    else:
        dfr = load_json(dec_freeze)
        model = ROOT / dfr["model_path"]
        config = model.parent / "match_config.json"
        a.check("decoder:hashes", (
            sha256(model) == dfr["model_sha256"] and sha256(config) == dfr["config_file_sha256"]),
            f"model_sha={dfr['model_sha256'][:16]}... config_sha={dfr['config_file_sha256'][:16]}...")
        abl = DECODER / "student_ablation.json"
        if abl.exists():
            a.record("decoder:ablation", "PASS", f"{abl.name} present")
        else:
            a.missing("decoder:ablation", "student ablation not reported yet")

    # 9. Live evidence ---------------------------------------------------------
    live_files = sorted(LIVE.glob("*.json")) if LIVE.exists() else []
    if not live_files:
        a.missing("live", "reports/phase_three/live/ is empty; public proof pending")
    else:
        trip_status: dict[str, int] = {}
        browser_proof = False
        for f in live_files:
            try:
                doc = load_json(f)
            except json.JSONDecodeError:
                continue
            if isinstance(doc, dict):
                for m in doc.get("matches", []) or []:
                    trip_status[m.get("trip_id")] = m.get("http_status")
                browser_proof = browser_proof or bool(
                    doc.get("browser") or "screenshot" in json.dumps(doc)[:200000].lower())
        ok = len(trip_status) == 10 and all(v == 200 for v in trip_status.values())
        a.check("live:api_matches", ok,
                f"trips={len(trip_status)} statuses={sorted(set(trip_status.values()))}")
        a.check("live:browser", browser_proof, "browser evidence file present" if browser_proof
                else "no browser/screenshot evidence found in live JSON files")

    # 10. Historical root model untouched --------------------------------------
    hist_manifest = ROOT / "artifacts" / "artifact_manifest.json"
    hist_model = ROOT / "artifacts" / "model.joblib"
    try:
        hm = load_json(hist_manifest)
        want = hm.get("sha256") or hm["files"]["model.joblib"]["sha256"]
        a.check("historical_model_untouched", sha256(hist_model) == want,
                f"sha={want[:16]}... bytes={hm.get('bytes')}")
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        a.missing("historical_model_untouched", f"historical manifest/model unreadable: {exc}")

    return a


def main() -> int:
    a = audit()
    counts: dict[str, int] = {}
    for c in a.checks:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    report = {"generated_at_files": __file__, "counts": counts, "checks": a.checks}
    out = Path(__file__).with_name("audit_report.json")
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts,
                      "result": "PASS" if not counts.get("FAIL") and not counts.get("MISSING")
                                else "INCOMPLETE_OR_FAILED",
                      "report": str(out.relative_to(ROOT))}))
    return 0 if not counts.get("FAIL") and not counts.get("MISSING") else 1


if __name__ == "__main__":
    raise SystemExit(main())
