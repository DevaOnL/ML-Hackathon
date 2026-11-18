"""Fixture-based tests for the final-five figure generator and verifier.

These tests use small synthetic data fixtures only. They deliberately do NOT
render the real 23 final-report figures (22 model + one decoder) in CI (that takes minutes and needs
the real experiment outputs); they exercise the validation layer, the
formula identities the charts rely on, and the hash-pinning round trip.

Run: python3 -m unittest tests.test_final_figures
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "generate_final_five", ROOT / "visualizations" / "final_five" / "generate_final_five.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

_spec_v = importlib.util.spec_from_file_location(
    "verify_final_figures", ROOT / "visualizations" / "final_five" / "verify_final_figures.py")
ver = importlib.util.module_from_spec(_spec_v)
_spec_v.loader.exec_module(ver)


# ------------------------------------------------------------------ fixtures

FAMILIES_FIX = ("family_a", "family_b")
FINAL_RUN_FIX = "family_b_tuned"


def make_expected():
    """Small synthetic denominators, all identities the validator enforces."""
    return {
        "families": FAMILIES_FIX,
        "final_run": FINAL_RUN_FIX,
        "selected_model": "family_b",
        "cv_folds": 2,
        "train_trips": 4,
        "train_observations": 8,
        "train_observations_per_trip": 2,
        "train_candidate_rows": 40,
        "train_positive_rows": 8,
        "train_candidate_miss_count": 1,
        "student_trips": 2,
        "student_observations": 5,      # deliberately unequal trips: 2 + 3
        "student_candidate_rows": 20,
        "student_positive_rows": 5,
        "student_candidate_miss_count": 1,
        "selected_model_sha256": "0" * 64,
        "hpo_trials": 3,
        "hpo_measured_prior": 1,
        "hpo_seeded_initial_design": 1,
        "hpo_gp_expected_improvement": 1,
    }


def make_trip(trip_id, obs, correct, prefix="t"):
    """A per-trip record with a given top-1 correct count."""
    return {"trip_id": trip_id, "observations": obs,
            "top1_correct_observations": correct,
            "top1_accuracy": correct / obs}


def make_cv_block(correct_by_fold, miss=1):
    """2-fold CV block. correct_by_fold: [[per-trip correct], [per-trip correct]]."""
    folds, per_trip = [], []
    trip_ids = ["t0", "t1", "t2", "t3"]
    for k, fold_corrects in enumerate(correct_by_fold):
        val_ids = trip_ids[k * 2:(k + 1) * 2]
        folds.append({"fold": k,
                      "train_trip_ids": [t for t in trip_ids if t not in val_ids],
                      "validation_trip_ids": val_ids,
                      "metrics": {"macro_f1": 0.9, "top1_accuracy": 0.9, "log_loss": 0.1}})
        for tid, c in zip(val_ids, fold_corrects):
            per_trip.append(make_trip(tid, 2, c))
    obs = len(per_trip) * 2
    correct = sum(t["top1_correct_observations"] for t in per_trip)
    cm = [[30, 2], [1, 7]]  # sums to 40 rows, 8 positive
    return {"oof_metrics": {
        "observations": obs, "candidate_rows": 40, "positive_rows": 8,
        "candidate_present_observations": obs - miss, "candidate_miss_count": miss,
        "top1_correct_observations": correct, "top1_accuracy": correct / obs,
        "trip_macro_top1_accuracy": sum(t["top1_accuracy"] for t in per_trip) / len(per_trip),
        "macro_f1": 0.9, "log_loss": 0.1, "confusion_matrix": cm,
        "per_trip": per_trip},
        "folds": folds, "parameters": {"p": 1.0},
        "training_seconds_total": 1.0, "inference_seconds_total": 2.0,
        "elapsed_seconds": 3.0, "reused": False}


def make_eval_block():
    """Student block with unequal trip sizes (2 and 3 observations)."""
    per_trip = [make_trip("s0", 2, 2), make_trip("s1", 3, 2)]
    obs = 5
    correct = 4
    pooled = correct / obs
    trip_macro = sum(t["top1_accuracy"] for t in per_trip) / len(per_trip)
    return {"student_metrics": {
        "observations": obs, "candidate_rows": 20, "positive_rows": 5,
        "candidate_present_observations": 4, "candidate_miss_count": 1,
        "top1_correct_observations": correct, "top1_accuracy": pooled,
        "trip_macro_top1_accuracy": trip_macro,
        "macro_f1": 0.9, "log_loss": 0.1,
        "confusion_matrix": [[14, 1], [1, 4]], "per_trip": per_trip},
        "train_resubstitution_metrics": {"macro_f1": 0.95, "log_loss": 0.05},
        "training_seconds": 0.5, "student_inference_seconds": 0.25, "reused": False}


def make_hpo():
    trials = [
        {"trial": 0, "proposal": "measured_baseline_prior", "objective": 0.80,
         "parameters": {k: 1.0 for k in gen.HPO_PARAM_KEYS}},
        {"trial": 1, "proposal": "seeded_initial_design", "objective": 0.81,
         "parameters": {k: 2.0 for k in gen.HPO_PARAM_KEYS}},
        {"trial": 2, "proposal": "gaussian_process_expected_improvement", "objective": 0.82,
         "parameters": {k: 3.0 for k in gen.HPO_PARAM_KEYS}},
    ]
    return {"trials": trials, "trials_including_measured_prior": 3,
            "best_objective": 0.82, "best_parameters": {k: 3.0 for k in gen.HPO_PARAM_KEYS},
            "baseline_objective": 0.80, "seconds": 10.0, "method": "gp",
            "search_space": {}, "objective": "max",
            "selection_data": "TRAIN OOF only; student untouched"}


def make_view(expected=None):
    expected = expected or make_expected()
    runs = list(FAMILIES_FIX) + [FINAL_RUN_FIX]
    cv = {r: make_cv_block([[2, 2], [2, 1]]) for r in runs}
    cv[FINAL_RUN_FIX]["parameters"] = {k: 3.0 for k in gen.HPO_PARAM_KEYS}
    ev = {r: make_eval_block() for r in runs}
    results = {"families": list(FAMILIES_FIX), "family_count": 2,
               "selected_model": "family_b", "final_run_name": FINAL_RUN_FIX,
               "cv": {r: {"oof_metrics": {"macro_f1": cv[r]["oof_metrics"]["macro_f1"]}} for r in runs},
               "evaluations": {r: {"student_metrics": {"macro_f1": ev[r]["student_metrics"]["macro_f1"]}}
                               for r in runs},
               "bayesian": make_hpo(),
               "model_freeze": {"model_sha256": "0" * 64, "run_name": FINAL_RUN_FIX,
                                "student_used_for_selection": False,
                                "evaluator_used_for_selection": False}}
    seq = {r: {"observations": 5, "candidate_miss_count": 1,
               "top1_accuracy": ev[r]["student_metrics"]["top1_accuracy"],
               "observation_edge_macro_f1": 0.8, "trip_macro_observation_edge_f1": 0.8,
               "observation_categorical_log_loss_clipped": 0.3,
               "raw_classifier_continuity": 0.9, "edge_change_continuity": 0.7}
           for r in runs}
    selection = {"objectives": {r: cv[r]["oof_metrics"]["trip_macro_top1_accuracy"]
                                for r in FAMILIES_FIX},
                 "student_used": False, "evaluator_used": False,
                 "selected_model": "family_b"}
    view = gen.prepare(results, cv, ev, seq, selection, expected)
    return view, results, cv, ev, seq, selection, expected


# ------------------------------------------------------------------ tests

class TestFamilyValidation(unittest.TestCase):
    def test_real_constants_pin_the_five_required_families(self):
        self.assertEqual(gen.FAMILIES, ("logistic_regression", "knn_distance_weighted",
                                        "bagging", "gaussian_nb", "hist_gradient_boosting"))
        self.assertEqual(len(gen.FAMILIES), 5)
        self.assertEqual(gen.FINAL_RUN, "hist_gradient_boosting_tuned")
        self.assertNotIn(gen.FINAL_RUN, gen.FAMILIES)

    def test_extra_family_rejected(self):
        view, results, *_ = make_view()
        bad = json.loads(json.dumps(results))
        bad["families"] = list(FAMILIES_FIX) + ["family_c"]
        bad["family_count"] = 3
        with self.assertRaises(ValueError):
            gen.validate_family_sets(bad, make_expected())

    def test_missing_family_rejected(self):
        view, results, *_ = make_view()
        bad = json.loads(json.dumps(results))
        bad["families"] = [FAMILIES_FIX[0]]
        bad["family_count"] = 1
        with self.assertRaises(ValueError):
            gen.validate_family_sets(bad, make_expected())

    def test_tuned_run_must_extend_selected_family(self):
        view, results, *_ = make_view()
        bad = json.loads(json.dumps(results))
        bad["final_run_name"] = "family_a_tuned"
        with self.assertRaises(ValueError):
            gen.validate_family_sets(bad, make_expected())

    def test_six_runs_expected_for_real_data(self):
        self.assertEqual(len(gen.RUNS), 6)
        self.assertEqual(gen.RUNS[-1], gen.FINAL_RUN)


class TestDenominatorValidation(unittest.TestCase):
    def test_cv_denominators_enforced(self):
        _, _, cv, _, _, _, expected = make_view()
        bad = json.loads(json.dumps(cv[FAMILIES_FIX[0]]))
        bad["oof_metrics"]["candidate_rows"] += 1
        with self.assertRaises(ValueError):
            gen.validate_cv_block(FAMILIES_FIX[0], bad, expected)

    def test_cv_miss_count_enforced(self):
        _, _, cv, _, _, _, expected = make_view()
        bad = json.loads(json.dumps(cv[FAMILIES_FIX[0]]))
        bad["oof_metrics"]["candidate_miss_count"] = 99
        with self.assertRaises(ValueError):
            gen.validate_cv_block(FAMILIES_FIX[0], bad, expected)

    def test_cv_fold_trip_partition_must_cover_all_trips_disjointly(self):
        _, _, cv, _, _, _, expected = make_view()
        bad = json.loads(json.dumps(cv[FAMILIES_FIX[0]]))
        bad["folds"][1]["validation_trip_ids"] = ["t0", "t1"]  # overlaps fold 0
        with self.assertRaises(ValueError):
            gen.validate_cv_block(FAMILIES_FIX[0], bad, expected)

    def test_cv_confusion_must_sum_to_candidate_rows(self):
        _, _, cv, _, _, _, expected = make_view()
        bad = json.loads(json.dumps(cv[FAMILIES_FIX[0]]))
        bad["oof_metrics"]["confusion_matrix"] = [[1, 1], [1, 1]]
        with self.assertRaises(ValueError):
            gen.validate_cv_block(FAMILIES_FIX[0], bad, expected)

    def test_student_denominators_enforced(self):
        _, _, _, ev, _, _, expected = make_view()
        bad = json.loads(json.dumps(ev[FAMILIES_FIX[0]]))
        bad["student_metrics"]["observations"] = 999
        with self.assertRaises(ValueError):
            gen.validate_eval_block(FAMILIES_FIX[0], bad, expected)

    def test_student_confusion_positive_row_enforced(self):
        _, _, _, ev, _, _, expected = make_view()
        bad = json.loads(json.dumps(ev[FAMILIES_FIX[0]]))
        bad["student_metrics"]["confusion_matrix"] = [[13, 3], [3, 1]]
        with self.assertRaises(ValueError):
            gen.validate_eval_block(FAMILIES_FIX[0], bad, expected)


class TestMetricIdentities(unittest.TestCase):
    """The denominators that keep pooled and trip-macro top-1 distinct."""

    def test_equal_trip_sizes_make_pooled_equal_trip_macro(self):
        _, _, cv, *_ = make_view()
        oof = cv[FAMILIES_FIX[0]]["oof_metrics"]
        self.assertAlmostEqual(oof["top1_accuracy"], oof["trip_macro_top1_accuracy"], places=9)
        self.assertEqual(oof["observations"], 8)  # 4 trips x 2 obs each

    def test_unequal_trip_sizes_make_pooled_differ_from_trip_macro(self):
        st = make_eval_block()["student_metrics"]
        # 2 obs at 1.0 + 3 obs at 2/3: pooled 0.8, trip-macro (1.0 + 0.667)/2 = 0.833
        self.assertAlmostEqual(st["top1_accuracy"], 0.8, places=9)
        self.assertAlmostEqual(st["trip_macro_top1_accuracy"], (1.0 + 2 / 3) / 2, places=9)
        self.assertNotAlmostEqual(st["top1_accuracy"], st["trip_macro_top1_accuracy"], places=3)

    def test_trip_macro_is_unweighted_mean_of_per_trip_rates(self):
        st = make_eval_block()["student_metrics"]
        expected = sum(t["top1_accuracy"] for t in st["per_trip"]) / len(st["per_trip"])
        self.assertAlmostEqual(st["trip_macro_top1_accuracy"], expected, places=12)

    def test_validator_rejects_wrong_pooled_top1(self):
        _, _, _, ev, _, _, expected = make_view()
        bad = json.loads(json.dumps(ev[FAMILIES_FIX[0]]))
        bad["student_metrics"]["top1_accuracy"] = 0.5  # correct/observations disagrees
        with self.assertRaises(ValueError):
            gen.validate_eval_block(FAMILIES_FIX[0], bad, expected)

    def test_validator_rejects_cv_pooled_trip_macro_mismatch_under_equal_denominators(self):
        _, _, cv, _, _, _, expected = make_view()
        bad = json.loads(json.dumps(cv[FAMILIES_FIX[0]]))
        bad["oof_metrics"]["trip_macro_top1_accuracy"] = 0.123
        with self.assertRaises(ValueError):
            gen.validate_cv_block(FAMILIES_FIX[0], bad, expected)


class TestHpoValidation(unittest.TestCase):
    def test_trial_category_counts_enforced(self):
        view = make_view()[0]
        cv_view = view["cv"]
        results = {"selection_objectives": None}
        bayes = make_hpo()
        bayes["trials"][2]["proposal"] = "seeded_initial_design"
        with self.assertRaises(ValueError):
            gen.validate_hpo(bayes, cv_view, results, make_expected())

    def test_best_parameters_must_match_tuned_run(self):
        view = make_view()[0]
        bayes = make_hpo()
        bayes["best_parameters"] = {k: 99.0 for k in gen.HPO_PARAM_KEYS}
        with self.assertRaises(ValueError):
            gen.validate_hpo(bayes, view["cv"], {"selection_objectives": None}, make_expected())

    def test_selection_objectives_must_equal_cv_trip_macro(self):
        view = make_view()[0]
        bayes = make_hpo()
        sel = {"family_a": 0.5}  # not its CV trip-macro top-1
        with self.assertRaises(ValueError):
            gen.validate_hpo(bayes, view["cv"], {"selection_objectives": sel}, make_expected())

    def test_real_hpo_shape_is_1_prior_3_seeded_8_gp(self):
        self.assertEqual(gen.REAL_EXPECTED["hpo_trials"], 12)
        self.assertEqual(gen.REAL_EXPECTED["hpo_measured_prior"], 1)
        self.assertEqual(gen.REAL_EXPECTED["hpo_seeded_initial_design"], 3)
        self.assertEqual(gen.REAL_EXPECTED["hpo_gp_expected_improvement"], 8)
        self.assertEqual(gen.REAL_EXPECTED["train_observations"], 28000)
        self.assertEqual(gen.REAL_EXPECTED["train_candidate_rows"], 269463)
        self.assertEqual(gen.REAL_EXPECTED["train_candidate_miss_count"], 578)
        self.assertEqual(gen.REAL_EXPECTED["student_observations"], 19160)
        self.assertEqual(gen.REAL_EXPECTED["student_candidate_miss_count"], 19)


class TestScopeGuard(unittest.TestCase):
    def test_writes_outside_final_five_are_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            evil = Path(tmp) / "elsewhere.png"
            with self.assertRaises(ValueError):
                gen.safe_output(evil)

    def test_writes_inside_final_five_are_allowed(self):
        target = gen.FIVE_DIR / "plots" / "_scope_check.tmp.png"
        resolved = gen.safe_output(target)
        self.assertTrue(resolved.is_relative_to(gen.FIVE_DIR.resolve()))
        resolved.unlink(missing_ok=True)


class TestHashPinningRoundTrip(unittest.TestCase):
    def test_png_dpi_and_size_reader(self):
        import zlib
        # minimal PNG with IHDR + pHYs (300 dpi = 11811 pixels per metre)
        width, height = 4, 4
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
        phys = struct.pack(">IIB", 11811, 11811, 1)
        def chunk(tag, body):
            return struct.pack(">I", len(body)) + tag + body \
                + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"pHYs", phys)
               + chunk(b"IDAT", zlib.compress(b"\x00" * (height * (1 + width))))
               + chunk(b"IEND", b""))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tiny.png"
            p.write_bytes(png)
            dpi, size = ver.png_dpi_and_size(p)
            self.assertEqual(size, (4, 4))
            self.assertAlmostEqual(dpi[0], 299.97, places=1)

    def test_sha_round_trip_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.bin"
            p.write_bytes(b"original")
            digest = ver.sha256_file(p)
            self.assertEqual(digest, hashlib.sha256(b"original").hexdigest())
            p.write_bytes(b"tampered")
            self.assertNotEqual(ver.sha256_file(p), digest)

    def test_ocr_expected_token_normalization(self):
        self.assertIn("macro", ver.norm("Macro-F1 (0-1)!"))
        self.assertNotIn("macro f1", "")


class TestIncrementalManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / "visualizations" / "final_five"
        self.base.mkdir(parents=True)
        (self.base / "generate_final_five.py").write_text("# synthetic generator\n")
        self.patches = [patch.object(gen, "ROOT", self.root), patch.object(gen, "FIVE_DIR", self.base),
                        patch.object(gen, "MANIFEST", self.base / "figure_manifest.json"),
                        patch.object(gen, "EXPECTED_BY_STAGE", {"models": {"model.png"}, "decoder": {"decoder.png"}}),
                        patch.object(gen, "reused_eda_records", return_value=([], []))]
        for mocked in self.patches:
            mocked.start()
            self.addCleanup(mocked.stop)
        self.inputs = {}
        for key in ("results", "decoder_report", "decoder_source"):
            path = self.root / f"{key}.json"
            path.write_text(key)
            self.inputs[key] = {"path": path.name, "sha256": gen.sha256_file(path)}
        self.model = self.record("model", "models")
        self.decoder = self.record("decoder", "decoder")
        self.previous = {"figures": [self.model, self.decoder], "inputs": self.inputs}

    def record(self, name, stage):
        path = self.base / f"{name}.png"
        fig = gen.plt.figure(figsize=(1, 1))
        fig.savefig(path, dpi=300)
        fig.savefig(path.with_suffix(".pdf"))
        gen.plt.close(fig)
        return {"path": str(path.relative_to(self.root)), "sha256": gen.sha256_file(path),
                "pdf_path": str(path.with_suffix(".pdf").relative_to(self.root)),
                "pdf_sha256": gen.sha256_file(path.with_suffix(".pdf")), "dpi": 300,
                "pixels": {"width": 300, "height": 300}, "stage": stage}

    def test_all_manifest_lists_both_stages_with_total_count(self):
        with patch.object(gen, "REGISTRY", [self.model, self.decoder]):
            result = gen.build_manifest("all", self.inputs, {}, {"status": "charted"})
        self.assertEqual(result["figure_count"], 2)
        self.assertEqual(result["model_figure_count"], 1)
        self.assertEqual(result["decoder_figure_count"], 1)
        self.assertFalse(result["partial"])

    def test_decoder_mode_preserves_and_verifies_existing_models(self):
        kept, _, problems = gen.preserved_records("decoder", self.inputs, self.previous)
        self.assertEqual(kept, [self.model])
        self.assertEqual(problems, [])
        (self.root / self.model["path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "--mode all"):
            gen.preserved_records("decoder", self.inputs, self.previous)

    def test_decoder_mode_rejects_input_hash_or_dpi_drift(self):
        bad = json.loads(json.dumps(self.previous))
        bad["figures"][0]["dpi"] = 72
        with self.assertRaisesRegex(ValueError, "DPI"):
            gen.preserved_records("decoder", self.inputs, bad)
        (self.root / self.inputs["results"]["path"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "input pin"):
            gen.preserved_records("decoder", self.inputs, self.previous)

    def test_models_preserves_valid_decoder_otherwise_is_partial(self):
        kept, _, _ = gen.preserved_records("models", self.inputs, self.previous)
        self.assertEqual(kept, [self.decoder])
        (self.root / self.decoder["pdf_path"]).write_bytes(b"tampered")
        kept, _, problems = gen.preserved_records("models", self.inputs, self.previous)
        self.assertEqual(kept, [])
        self.assertTrue(problems)
        with patch.object(gen, "REGISTRY", [self.model]):
            result = gen.build_manifest("models", self.inputs, {}, {"status": "await_route"}, kept)
        self.assertTrue(result["partial"])

    def test_only_decoder_never_claims_complete_and_legacy_22_is_rejected(self):
        with patch.object(gen, "REGISTRY", [self.decoder]):
            result = gen.build_manifest("decoder", self.inputs, {}, {"status": "charted"})
        self.assertTrue(result["partial"])
        problems = []
        ver.validate_manifest_structure({"figures": [self.model], "figure_count": 22,
                                         "partial": False}, problems)
        self.assertTrue(any("23" in p for p in problems))


class TestManifestContents(unittest.TestCase):
    """The shipped manifest must describe the real generated artifacts."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(
            (ROOT / "visualizations" / "final_five" / "figure_manifest.json").read_text())

    def test_manifest_pins_all_required_inputs(self):
        inputs = json.dumps(self.manifest["inputs"])
        for key in ("results", "sequence_classification", "bayesian_optimization", "cv",
                    "evaluation", "selection", "model_acceptance"):
            self.assertIn(key, inputs)

    def test_every_figure_record_has_provenance_fields(self):
        for fig in self.manifest["figures"]:
            for field in ("path", "sha256", "title", "caption", "units", "source",
                          "created_utc", "dpi", "pixels", "pdf_path"):
                self.assertIn(field, fig, f"{fig.get('path')} lacks {field}")
            self.assertEqual(fig["dpi"], 300)

    def test_confusion_grid_covers_six_configs_not_five_models(self):
        grid = [f for f in self.manifest["figures"]
                if f["path"].endswith("confusion_matrices_all.png")][0]
        self.assertIn("six", grid["caption"].lower())

    def test_epoch_wise_loss_is_stated_na_for_every_family(self):
        epoch = self.manifest["epoch_wise_loss"]
        self.assertEqual(epoch["status"], "N/A for all five families")
        self.assertEqual(set(epoch["per_family"]), set(gen.FAMILIES))

    def test_decoder_and_all_22_models_are_in_the_complete_manifest(self):
        self.assertEqual(self.manifest["decoder"]["status"], "charted")
        self.assertEqual(self.manifest["model_figure_count"], 22)
        self.assertEqual(self.manifest["decoder_figure_count"], 1)
        self.assertEqual(self.manifest["figure_count"], 23)
        self.assertEqual(len(self.manifest["figures"]), 23)
        self.assertFalse(self.manifest["partial"])
        problems = []
        ver.validate_manifest_structure(self.manifest, problems)
        self.assertEqual(problems, [])
        self.assertIn("decoder_source", self.manifest["inputs"])
        self.assertIn("decoder_report", self.manifest["inputs"])

    def test_eda_reuse_records_are_read_only_pins(self):
        self.assertGreaterEqual(len(self.manifest["eda_reused_read_only"]), 12)
        for rec in self.manifest["eda_reused_read_only"]:
            self.assertTrue(rec["reused_read_only"])
            self.assertIn("sha256", rec)


if __name__ == "__main__":
    unittest.main()
