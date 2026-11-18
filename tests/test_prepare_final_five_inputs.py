"""Synthetic, temporary-only input-restoration tests; no real private data is read.

The fixture uses actual sklearn LR/KNN/Bagging fits and probability prediction.
Its tiny metrics are test expectations, never evidence for the measured project.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts import prepare_final_five_inputs as prep
from mapmatch.experiments import CandidateDataset, _metrics, build_group_folds
from mapmatch.features import FEATURE_NAMES, FEATURE_VERSION


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def dataset(trips=6, observations=5, seed=19):
    rng = np.random.default_rng(seed)
    n = trips * observations
    rows = n * 3
    y = np.tile(np.array([1, 0, 0], dtype=np.int8), n)
    X = rng.normal(size=(rows, len(FEATURE_NAMES))).astype(np.float32)
    X[:, 0] += y * 1.1
    return CandidateDataset(
        X=X, y=y, offsets=np.arange(0, rows + 1, 3, dtype=np.int64),
        observation_trip_ids=np.repeat([f"trip_{i:02d}" for i in range(trips)], observations),
        observation_indices=np.tile(np.arange(observations, dtype=np.int64), trips),
        reference_edges=np.full(n, 10, dtype=np.int64), candidate_edges=np.tile([10, 11, 12], n).astype(np.int64),
        candidate_fractions=np.full(rows, 0.5), candidate_distances=np.ones(rows), proxy_distances=np.ones(n),
        ambiguous=np.zeros(n, bool), expanded=np.zeros(n, bool), coverage={})


def write_cache(root, data, split):
    fingerprint = hashlib.sha256(("synthetic-" + split).encode()).hexdigest()
    name = split + "_" + fingerprint[:16]
    target = root / "experiments/cache" / (name + ".npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **{key: getattr(data, key) for key in CandidateDataset.__dataclass_fields__ if key != "coverage"})
    spec = {"split": split, "trips": len(np.unique(data.observation_trip_ids)), "observations": len(data.offsets) - 1,
            "rows": len(data.y), "misses": 0, "cache": name, "fingerprint": fingerprint,
            "sha256": prep.sha256(target), "array_sha256": prep.rawprep.npz_array_digests(target)}
    save_json(target.with_suffix(".json"), {"split": split, "fingerprint": fingerprint,
        "trip_count": spec["trips"], "selected_observations": spec["observations"], "candidate_rows": spec["rows"]})
    return spec


def write_predictions(root, family, split, data, probabilities):
    target = root / prep.prediction_path(family, split)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **prep.phase.prediction_arrays(data, probabilities))


def fixture(root, config):
    train, student = dataset(), dataset(trips=3, observations=4, seed=29)
    specs = {"train": write_cache(root, train, "train"), "student_test": write_cache(root, student, "student_test")}
    folds = build_group_folds(train, config)
    save_json(root / "experiments/cv_folds.json", [{
        "fold": fold["fold"], "train_trip_ids": fold["train_trip_ids"], "validation_trip_ids": fold["validation_trip_ids"],
        "train_candidate_rows": len(fold["train_rows"]), "validation_candidate_rows": len(fold["validation_rows"]),
        "train_observations": len(fold["train_observations"]), "validation_observations": len(fold["validation_observations"]),
    } for fold in folds])
    for name in prep.FAMILIES:
        parameters = prep.final.FINAL_DEFAULTS[name]
        oof = np.full(len(train.y), np.nan)
        original_folds = []
        for fold in folds:
            model, seconds, warnings, bank = prep.phase.phase_fit(name, parameters, config, train, fold["train_rows"], fold["fold"])
            p, inference = prep.phase.phase_predict(model, train.X[fold["validation_rows"]])
            oof[fold["validation_rows"]] = p
            original_folds.append({"fold": fold["fold"], "train_trip_ids": fold["train_trip_ids"],
                "validation_trip_ids": fold["validation_trip_ids"], "train_rows": len(fold["train_rows"]),
                "validation_rows": len(fold["validation_rows"]), "training_seconds": seconds,
                "inference_seconds": inference, "warnings": warnings, "knn_reference_bank": bank,
                "metrics": _metrics(train, p, fold["validation_observations"])})
        record = {"name": name, "model_name": name, "parameters": parameters, "folds": original_folds,
                  "oof_metrics": _metrics(train, oof), "oof_predictions_file": prep.prediction_path(name, "train")}
        source = "experiments/phase_three/cv/" + name + ".json"
        save_json(root / source, record)
        save_json(root / ("experiments/final_five/current/cv/" + name + ".json"), {
            **record, "reuse_provenance": {"cv_file": source, "cv_sha256": prep.sha256(root / source)}})
        write_predictions(root, name, "train", train, oof)
        model, _, _, _ = prep.phase.phase_fit(name, parameters, config, train, np.arange(len(train.y)), 3)
        bundle = {"estimator": model, "feature_names": list(FEATURE_NAMES), "feature_version": FEATURE_VERSION,
                  "metadata": {"model_name": name, "run_name": name, "seed": config.seed, "parameters": parameters,
                               "training_trip_ids": np.unique(train.observation_trip_ids).tolist()}}
        model_path = root / prep.BASE_PATHS[name]
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_path, compress=3)
        student_p, _ = prep.phase.phase_predict(model, student.X)
        evaluation = {"run_name": name, "model_name": name, "parameters": parameters,
                      "student_metrics": _metrics(student, student_p), "student_predictions_file": prep.prediction_path(name, "student_test")}
        source = "experiments/phase_three/evaluation/" + name + ".json"
        save_json(root / source, evaluation)
        save_json(root / ("experiments/final_five/current/evaluation/" + name + ".json"), {
            **evaluation, "model_path": prep.BASE_PATHS[name], "model_sha256": prep.sha256(model_path),
            "reuse_provenance": {"evaluation_file": source, "evaluation_sha256": prep.sha256(root / source)}})
        write_predictions(root, name, "student_test", student, student_p)
    base = root / "artifacts/final_five"
    base.mkdir(parents=True)
    (base / "model.joblib").write_bytes(b"synthetic-freeze-hash-only; never unpickled")
    save_json(base / "model_metadata.json", {"synthetic": True})
    save_json(base / "feature_schema.json", {"feature_names": list(FEATURE_NAMES), "feature_version": FEATURE_VERSION, "count": len(FEATURE_NAMES)})
    freeze = {"phase": "five-model-completion", "model_name": "hist_gradient_boosting", "run_name": "hist_gradient_boosting_tuned",
              "model": "artifacts/final_five/model.joblib", "student_used_for_selection": False, "evaluator_used_for_selection": False,
              "model_sha256": prep.sha256(base / "model.joblib"), "metadata_sha256": prep.sha256(base / "model_metadata.json"),
              "feature_schema_sha256": prep.sha256(base / "feature_schema.json")}
    save_json(base / "model_freeze.json", freeze)
    config_values = {"sigma_m": 7.1, "beta_m": 2.0}
    save_json(base / "match_config.json", config_values)
    save_json(root / "reports/final_five/decoder/freeze.json", {"phase": "five-model-completion", "config": config_values,
        "student_results_seen_for_selection": False, "student_results_used_for_selection": False, "evaluator_accessed": False,
        "model_path": freeze["model"], "model_sha256": freeze["model_sha256"], "config_file_sha256": prep.sha256(base / "match_config.json"),
        "effective_config_sha256": hashlib.sha256(json.dumps(config_values, sort_keys=True).encode()).hexdigest()})
    (root / "source.py").write_text("# synthetic source-hash fixture\n")
    return specs, train, student, folds


def snapshot(root):
    return {str(path.relative_to(root)): prep.sha256(path) for path in root.rglob("*") if path.is_file() and not path.is_symlink()}


class FinalInputPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template_tmp = tempfile.TemporaryDirectory(prefix="final-input-template-")
        cls.template = Path(cls.template_tmp.name)
        cls.config = replace(prep.phase.phase_config(), n_jobs=1)
        with threadpool_limits(limits=1):
            cls.specs, cls.train, cls.student, cls.folds = fixture(cls.template, cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.template_tmp.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="final-input-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "repo"
        shutil.copytree(self.template, self.root)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target, attr, value in ((prep, "ROOT", self.root), (prep, "CACHE_SPECS", self.specs),
                                    # Tiny fixtures are generated in this interpreter. Keep the
                                    # production 3.14.6 guard intact, but bind synthetic restoration
                                    # to the runtime that actually fitted the fixture.
                                    (prep, "PUBLISHED_RUNTIME", prep.final.versions()),
                                    (prep, "SOURCE_FILES", ("source.py",)), (prep.phase, "SHARED", self.root / "experiments"),
                                    (prep.phase, "EXPECTED", self.specs)):
            self.stack.enter_context(patch.object(target, attr, value))
        self.stack.enter_context(patch.object(prep.phase, "phase_config", return_value=self.config))
        self.stack.enter_context(threadpool_limits(limits=1))

    def command(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = prep.main(list(args))
        return code, json.loads(output.getvalue())

    def remove(self, name, split):
        (self.root / prep.prediction_path(name, split)).unlink()

    def edit_original(self, kind, name, change, *, update_pin=True):
        relative = "experiments/phase_three/" + kind + "/" + name + ".json"
        path = self.root / relative
        record = json.loads(path.read_text())
        change(record)
        save_json(path, record)
        if update_pin:
            target = self.root / "experiments/final_five/current" / kind / (name + ".json")
            pin = json.loads(target.read_text())
            pin.update(record)
            pin["reuse_provenance"]["cv_sha256" if kind == "cv" else "evaluation_sha256"] = prep.sha256(path)
            save_json(target, pin)

    def test_default_check_reads_full_arrays_without_writes_or_inference(self):
        before = snapshot(self.root)
        with patch.object(prep.phase, "phase_fit") as fit, patch.object(prep.phase, "phase_predict") as predict:
            code, report = self.command()
        self.assertEqual(code, 0, report)
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(report["fits_performed"], 0)
        self.assertEqual(report["prediction_calls"], 0)
        self.assertEqual(sum(c["kind"] == "prediction_archive" for c in report["checks"]), 6)
        fit.assert_not_called(); predict.assert_not_called()
        self.assertEqual(before, snapshot(self.root))

    def test_clean_missing_inventory_lists_all_prediction_inputs_without_building(self):
        for split in ("train", "student_test"):
            for name in prep.FAMILIES:
                self.remove(name, split)
        before = snapshot(self.root)
        with patch.object(prep.phase, "phase_fit") as fit, patch.object(prep.phase, "phase_predict") as predict:
            code, report = self.command()
        self.assertEqual(code, 2, report)
        archives = [item for item in report["inventory"] if item["path"].endswith(".npz") and "predictions/" in item["path"]]
        self.assertEqual(len(archives), 6)
        self.assertTrue(all(not item["present"] for item in archives))
        self.assertTrue(any("prepare_phase_three_inputs.py" in item for item in report["missing"]))
        fit.assert_not_called(); predict.assert_not_called()
        self.assertEqual(before, snapshot(self.root))

    def test_nondirectory_destination_parent_refuses_before_loading(self):
        parent = self.root / prep.REPLAY_PARENT
        parent.write_bytes(b"not a directory")
        self.remove("bagging", "train")
        with patch.object(prep.phase, "load_train_and_folds") as load, patch.object(prep.phase, "phase_fit") as fit:
            code, report = self.command("--stage", "train", "--run-id", "bad-parent")
        self.assertEqual(code, 1, report); load.assert_not_called(); fit.assert_not_called()
        self.assertIn("parent is not a directory", report["failures"][0])

    def test_existing_explicit_stages_validate_but_never_refit(self):
        before = snapshot(self.root)
        for stage in ("train", "student"):
            with patch.object(prep.phase, "phase_fit") as fit, patch.object(prep.phase, "phase_predict") as predict:
                code, report = self.command("--stage", stage)
            self.assertEqual(code, 0, report)
            fit.assert_not_called(); predict.assert_not_called()
        self.assertEqual(before, snapshot(self.root))

    def test_missing_lr_fails_before_dataset_load_or_fit(self):
        for stage, split in (("train", "train"), ("student", "student_test")):
            self.remove("logistic_regression", split)
            with patch.object(prep.phase, "load_train_and_folds") as load, patch.object(prep.phase, "phase_fit") as fit:
                code, report = self.command("--stage", stage, "--run-id", "missing-lr")
            self.assertEqual(code, 2, report)
            self.assertIn("prepare_phase_three_inputs.py", report["missing"][0])
            load.assert_not_called(); fit.assert_not_called()

    def test_missing_new_arrays_need_run_id_before_dataset_load(self):
        self.remove("bagging", "train")
        with patch.object(prep.phase, "load_train_and_folds") as load, patch.object(prep.phase, "phase_fit") as fit:
            code, report = self.command("--stage", "train")
        self.assertEqual(code, 1, report)
        self.assertIn("explicit new --run-id", report["failures"][0])
        load.assert_not_called(); fit.assert_not_called()

    def test_invalid_or_existing_run_id_refuses_before_dataset_load(self):
        self.remove("bagging", "train")
        existing = self.root / prep.REPLAY_PARENT / "used"
        existing.mkdir(parents=True)
        for run_id in ("../escape", "/tmp/oops", "A", "x/y", "a" * 49, "used"):
            with patch.object(prep.phase, "load_train_and_folds") as load, patch.object(prep.phase, "phase_fit") as fit:
                code, report = self.command("--stage", "train", "--run-id", run_id)
            self.assertEqual(code, 1, report)
            load.assert_not_called(); fit.assert_not_called()

    def test_symlink_prediction_and_parent_refused_before_loading(self):
        relative = prep.prediction_path("bagging", "train")
        target = self.root / relative
        target.unlink()
        target.symlink_to(self.root / prep.prediction_path("knn_distance_weighted", "train"))
        with patch.object(prep.phase, "load_train_and_folds") as load:
            code, report = self.command("--stage", "train", "--run-id", "symlink")
        self.assertEqual(code, 1, report); load.assert_not_called()
        self.assertIn("symlink", report["failures"][0])
        replay = self.root / prep.REPLAY_PARENT
        replay.symlink_to(self.root / "reports", target_is_directory=True)
        with self.assertRaises(prep.PrepError):
            prep.replay_path("safe-name")

    def test_original_parameter_and_fold_mismatches_refuse_before_fit(self):
        self.remove("bagging", "train")
        self.edit_original("cv", "bagging", lambda record: record["parameters"].update(n_estimators=41))
        with patch.object(prep.phase, "load_train_and_folds") as load, patch.object(prep.phase, "phase_fit") as fit:
            code, report = self.command("--stage", "train", "--run-id", "params")
        self.assertEqual(code, 1, report); load.assert_not_called(); fit.assert_not_called()

    def test_wrong_validation_trip_fold_fails_before_fit(self):
        self.remove("bagging", "train")
        self.edit_original("cv", "bagging", lambda record: record["folds"][0]["validation_trip_ids"].append("foreign"))
        with patch.object(prep.phase, "load_train_and_folds") as load, patch.object(prep.phase, "phase_fit") as fit:
            code, report = self.command("--stage", "train", "--run-id", "folds")
        self.assertEqual(code, 1, report); load.assert_not_called(); fit.assert_not_called()

    def test_unpinned_original_record_fails_without_repair(self):
        self.remove("bagging", "train")
        self.edit_original("cv", "bagging", lambda record: record.update(training_seconds_total=999), update_pin=False)
        before = snapshot(self.root)
        with patch.object(prep.phase, "phase_fit") as fit:
            code, report = self.command("--stage", "train", "--run-id", "history")
        self.assertEqual(code, 1, report); fit.assert_not_called()
        self.assertEqual(before, snapshot(self.root))

    def test_actual_small_oof_restoration_and_exclusive_reload_preserve_history(self):
        for name in prep.RESTORABLE:
            self.remove(name, "train")
        before = snapshot(self.root)
        real_fit = prep.phase.phase_fit
        with patch.object(prep.phase, "phase_fit", wraps=real_fit) as fit:
            code, report = self.command("--stage", "train", "--run-id", "tiny-train")
        self.assertEqual(code, 0, report)
        self.assertEqual(report["fits_performed"], 6)
        self.assertEqual(report["prediction_calls"], 6)
        self.assertEqual([call.args[-1] for call in fit.call_args_list], [0, 1, 2, 0, 1, 2])
        self.assertEqual([call.args[0] for call in fit.call_args_list], ["knn_distance_weighted"] * 3 + ["bagging"] * 3)
        for name in prep.RESTORABLE:
            reloaded = prep.phase.load_prediction_npz(self.root / prep.prediction_path(name, "train"))
            prep.phase.assert_prediction_identity(self.train, reloaded)
            self.assertTrue(np.isfinite(reloaded["probabilities"]).all())
        for relative, old_hash in before.items():
            self.assertEqual(old_hash, prep.sha256(self.root / relative), relative)
        self.assertTrue((self.root / prep.REPLAY_PARENT / "tiny-train/complete.json").is_file())

    def test_only_missing_family_is_fitted(self):
        self.remove("bagging", "train")
        existing_hash = prep.sha256(self.root / prep.prediction_path("knn_distance_weighted", "train"))
        with patch.object(prep.phase, "phase_fit", wraps=prep.phase.phase_fit) as fit:
            code, report = self.command("--stage", "train", "--run-id", "only-bag")
        self.assertEqual(code, 0, report)
        self.assertEqual(fit.call_count, 3)
        self.assertTrue(all(call.args[0] == "bagging" for call in fit.call_args_list))
        self.assertEqual(existing_hash, prep.sha256(self.root / prep.prediction_path("knn_distance_weighted", "train")))

    def test_wrong_recomputed_metric_has_no_prediction_output(self):
        self.remove("bagging", "train")
        self.edit_original("cv", "bagging", lambda record: record["folds"][0]["metrics"].update(log_loss=99.0))
        code, report = self.command("--stage", "train", "--run-id", "metric-fail")
        self.assertEqual(code, 1, report)
        self.assertFalse((self.root / prep.prediction_path("bagging", "train")).exists())
        work = self.root / prep.REPLAY_PARENT / "metric-fail"
        self.assertTrue((work / "bagging_fold_0.json").is_file())
        self.assertEqual(json.loads((work / "failed.json").read_text())["status"], "FAILED")

    def test_invalid_probability_and_row_identity_are_rejected(self):
        prediction = prep.phase.prediction_arrays(self.student, np.full(len(self.student.y), 0.2))
        for value in (np.nan, -0.1, 1.1):
            bad = deepcopy(prediction)
            bad["probabilities"][0] = value
            with self.assertRaises(prep.PrepError):
                prep.validate_probabilities(self.student, bad)
        prediction["candidate_edges"] = prediction["candidate_edges"].copy()
        prediction["candidate_edges"][0] += 1
        with self.assertRaises(AssertionError):
            prep.validate_probabilities(self.student, prediction)

    def test_all_saved_metric_fields_are_strictly_checked(self):
        metrics = _metrics(self.student, np.full(len(self.student.y), 0.2))
        for key, value in (("precision", 0.33), ("positive_rows", 999), ("log_loss", float("nan"))):
            bad = deepcopy(metrics); bad[key] = value
            with self.assertRaises((prep.PrepError, AssertionError)):
                prep.strict_metrics(bad, metrics)
        bad = deepcopy(metrics); bad["per_trip"][0]["top1_accuracy"] += 0.01
        with self.assertRaises(prep.PrepError):
            prep.strict_metrics(bad, metrics)
        bad = deepcopy(metrics); bad["confusion_matrix"][0][0] += 1
        with self.assertRaises((prep.PrepError, AssertionError)):
            prep.strict_metrics(bad, metrics)

    def test_student_missing_freeze_blocks_before_any_dataset_access(self):
        (self.root / "reports/final_five/decoder/freeze.json").unlink()
        with patch.object(prep.phase, "load_train_and_folds") as train, patch.object(prep.phase, "load_cached_dataset") as student, patch.object(prep.phase, "phase_predict") as predict:
            code, report = self.command("--stage", "student", "--run-id", "no-freeze")
        self.assertEqual(code, 2, report)
        train.assert_not_called(); student.assert_not_called(); predict.assert_not_called()

    def test_student_changed_model_or_config_freeze_blocks_before_data(self):
        (self.root / "artifacts/final_five/match_config.json").write_text("{}")
        with patch.object(prep.phase, "load_train_and_folds") as load, patch.object(prep.phase, "phase_predict") as predict:
            code, report = self.command("--stage", "student", "--run-id", "changed-config")
        self.assertEqual(code, 1, report); load.assert_not_called(); predict.assert_not_called()

    def test_decoder_selection_flags_must_explicitly_be_false(self):
        target = self.root / "reports/final_five/decoder/freeze.json"
        original = json.loads(target.read_text())
        for key in ("student_results_seen_for_selection", "student_results_used_for_selection", "evaluator_accessed"):
            for present in (True, False):
                bad = deepcopy(original)
                if present:
                    bad[key] = True
                else:
                    bad.pop(key)
                save_json(target, bad)
                with patch.object(prep.phase, "load_train_and_folds") as load:
                    code, report = self.command("--stage", "student")
                self.assertEqual(code, 1, report); load.assert_not_called()

    def test_runtime_mismatch_refuses_before_data_fit_or_prediction(self):
        mismatched = {**prep.final.versions(), "python": "synthetic-incompatible-version"}
        for stage, split in (("train", "train"), ("student", "student_test")):
            with self.subTest(stage=stage):
                self.remove("bagging", split)
                before = snapshot(self.root)
                with patch.object(prep, "PUBLISHED_RUNTIME", mismatched), \
                     patch.object(prep.phase, "load_train_and_folds") as load, \
                     patch.object(prep.phase, "phase_fit") as fit, \
                     patch.object(prep.phase, "phase_predict") as predict:
                    code, report = self.command("--stage", stage, "--run-id", "wrong-runtime-" + stage)
                self.assertEqual(code, 1, report)
                self.assertIn("pinned published runtime", report["failures"][0])
                load.assert_not_called(); fit.assert_not_called(); predict.assert_not_called()
                self.assertEqual(snapshot(self.root), before)

    def test_actual_student_inference_from_published_tiny_models_no_fit(self):
        for name in prep.RESTORABLE:
            self.remove(name, "student_test")
        before = snapshot(self.root)
        with patch.object(prep.phase, "phase_fit") as fit, patch.object(prep.phase, "phase_predict", wraps=prep.phase.phase_predict) as predict:
            code, report = self.command("--stage", "student", "--run-id", "tiny-student")
        self.assertEqual(code, 0, report)
        self.assertEqual(report["fits_performed"], 0)
        self.assertEqual(predict.call_count, 2); fit.assert_not_called()
        for relative, old_hash in before.items():
            self.assertEqual(old_hash, prep.sha256(self.root / relative), relative)
        for name in prep.RESTORABLE:
            reloaded = prep.phase.load_prediction_npz(self.root / prep.prediction_path(name, "student_test"))
            prep.phase.assert_prediction_identity(self.student, reloaded)
            self.assertEqual(reloaded["probabilities"].shape, self.student.y.shape)

    def test_student_wrong_metrics_publish_nothing(self):
        for name in prep.RESTORABLE:
            self.remove(name, "student_test")
        self.edit_original("evaluation", "bagging", lambda record: record["student_metrics"].update(log_loss=99.0))
        with patch.object(prep.phase, "phase_fit") as fit:
            code, report = self.command("--stage", "student", "--run-id", "student-fail")
        self.assertEqual(code, 1, report); fit.assert_not_called()
        for name in prep.RESTORABLE:
            self.assertFalse((self.root / prep.prediction_path(name, "student_test")).exists())
        self.assertTrue((self.root / prep.REPLAY_PARENT / "student-fail/failed.json").is_file())

    def test_baseline_sha_or_metadata_mismatch_refuses_before_inference(self):
        self.remove("bagging", "student_test")
        artifact = self.root / prep.BASE_PATHS["bagging"]
        bundle = joblib.load(artifact)
        bundle["metadata"]["training_trip_ids"] = bundle["metadata"]["training_trip_ids"][:-1]
        joblib.dump(bundle, artifact, compress=3)
        pin_path = self.root / "experiments/final_five/current/evaluation/bagging.json"
        pin = json.loads(pin_path.read_text()); pin["model_sha256"] = prep.sha256(artifact)
        save_json(pin_path, pin)
        with patch.object(prep.phase, "phase_predict") as predict:
            code, report = self.command("--stage", "student", "--run-id", "bad-metadata")
        self.assertEqual(code, 1, report); predict.assert_not_called()
        self.assertIn("all TRAIN trips", report["failures"][0])

    def test_destination_race_never_overwrites_existing_archive(self):
        self.remove("bagging", "train")
        records, _ = prep.original_records("train")
        candidate = prep.phase.load_prediction_npz(self.template / prep.prediction_path("bagging", "train"))["probabilities"]
        work = prep.replay_path("race")
        prep.allocate_replay(work, "train", ["bagging"])
        target = self.root / prep.prediction_path("bagging", "train")
        sentinel = b"concurrent-writer-sentinel"
        target.write_bytes(sentinel)
        with self.assertRaises(FileExistsError):
            prep.publish_prediction("bagging", "train", self.train, candidate, records["bagging"], self.folds, work)
        self.assertEqual(target.read_bytes(), sentinel)

    def test_report_is_opt_in_fixed_and_exclusive(self):
        for args in (("--report", "../arbitrary.json"), ("--stage", "train", "--report", prep.CHECK_REPORT)):
            code, report = self.command(*args)
            self.assertEqual(code, 1, report)
        code, report = self.command("--stage", "check", "--report", prep.CHECK_REPORT)
        self.assertEqual(code, 0, report)
        target = self.root / prep.CHECK_REPORT
        before = target.read_bytes()
        code, second = self.command("--stage", "check", "--report", prep.CHECK_REPORT)
        self.assertEqual(code, 1, second)
        self.assertEqual(target.read_bytes(), before)
        self.assertNotIn("probabilities", before.decode())
        self.assertNotIn("reference_edges", before.decode())


if __name__ == "__main__":
    unittest.main()
