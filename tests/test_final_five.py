from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.final_five import run_final_five as runner
from mapmatch.features import FEATURE_NAMES, FEATURE_VERSION
from mapmatch.final_models import FINAL_FAMILIES, FINAL_DEFAULTS, make_final_estimator
from mapmatch.learning import load_emission_model, predict_positive_proba


class FinalFiveTests(unittest.TestCase):
    def test_registry_actual_classes_and_fixed_configs(self):
        expected = {"logistic_regression": LogisticRegression, "knn_distance_weighted": KNeighborsClassifier,
                    "bagging": BaggingClassifier, "gaussian_nb": GaussianNB, "hist_gradient_boosting": HistGradientBoostingClassifier}
        self.assertEqual(set(FINAL_FAMILIES), set(expected))
        for name, cls in expected.items():
            model = make_final_estimator(name)
            classifier = model.named_steps["classifier"] if isinstance(model, Pipeline) else model
            self.assertIsInstance(classifier, cls)
        gnb = make_final_estimator("gaussian_nb")
        self.assertIsInstance(gnb.named_steps["scale"], StandardScaler)
        self.assertEqual(gnb.named_steps["classifier"].var_smoothing, 1e-7)
        self.assertFalse(make_final_estimator("hist_gradient_boosting").early_stopping)
        with self.assertRaises(ValueError):
            make_final_estimator("random_forest")

    def test_new_actual_models_fit_and_finite_probabilities(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(120, len(FEATURE_NAMES))).astype(np.float32)
        y = (X[:, 0] + .3 * X[:, 1] > 0).astype(int)
        for name in ("gaussian_nb", "hist_gradient_boosting"):
            params = dict(FINAL_DEFAULTS[name])
            if name == "hist_gradient_boosting":
                params.update(max_iter=4, min_samples_leaf=3)
            model = make_final_estimator(name, params)
            runner._fit(model, X, y, np.ones(len(y)))
            p = predict_positive_proba(model, X)
            self.assertEqual(p.shape, (120,))
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue(((p > 0) & (p < 1)).all())

    def test_gaussian_nb_prediction_promotes_only_its_input_precision(self):
        rng = np.random.default_rng(14)
        X = rng.normal(size=(100, len(FEATURE_NAMES))).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        model = make_final_estimator("gaussian_nb")
        runner._fit(model, X, y, np.ones(len(y)))
        expected = model.predict_proba(X.astype(np.float64))[:, 1]
        np.testing.assert_allclose(predict_positive_proba(model, X), np.clip(expected, 1e-7, 1-1e-7), atol=0, rtol=0)

    def test_select_only_exact_five_train_objectives(self):
        cvs = {n: {"objective": float(i), "oof_metrics": {"macro_f1": .5, "log_loss": .2}} for i, n in enumerate(FINAL_FAMILIES)}
        self.assertEqual(runner.select_family(cvs), "hist_gradient_boosting")
        cvs["gaussian_nb"]["objective"] = 100
        self.assertEqual(runner.select_family(cvs), "gaussian_nb")
        cvs["extra_trees"] = cvs["bagging"]
        with self.assertRaises(ValueError):
            runner.select_family(cvs)

    def test_cv_trip_isolation_validation_rejects_mismatch(self):
        folds = [{"train_trip_ids": ["a"], "validation_trip_ids": ["b"]}]
        runner.check_folds({"folds": deepcopy(folds)}, folds)
        with self.assertRaises(AssertionError):
            runner.check_folds({"folds": [{"train_trip_ids": ["a", "b"], "validation_trip_ids": ["b"]}]}, folds)

    def test_guards_precede_any_dataset_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, artifact = Path(tmp) / "output", Path(tmp) / "artifact"
            output.mkdir(); artifact.mkdir()
            (output / "selection.json").write_text("{}")
            with patch.object(runner, "OUT", output), patch.object(runner, "ART", artifact), patch.object(runner, "load_train_and_folds") as load:
                with self.assertRaises(FileExistsError):
                    runner.stage_train()
                load.assert_not_called()
            (output / "results.json").write_text("{}")
            with patch.object(runner, "OUT", output), patch.object(runner, "ART", artifact), patch.object(runner, "load_cached_dataset") as load:
                with self.assertRaises(FileExistsError):
                    runner.stage_student()
                load.assert_not_called()

    def test_actual_synthetic_export_reload_and_refusal(self):
        rng = np.random.default_rng(14)
        X = rng.normal(size=(80, len(FEATURE_NAMES)))
        y = (X[:, 0] > 0).astype(int)
        model = make_final_estimator("gaussian_nb")
        runner._fit(model, X, y, np.ones(len(y)))
        with tempfile.TemporaryDirectory() as tmp:
            md = {"model_name": "gaussian_nb", "run_name": "synthetic_unit", "feature_version": FEATURE_VERSION}
            record = runner.freeze_bundle(model, md, Path(tmp))
            loaded = load_emission_model(Path(tmp) / "model.joblib")
            np.testing.assert_allclose(predict_positive_proba(model, X), predict_positive_proba(loaded.estimator, X), atol=1e-12, rtol=0)
            self.assertEqual(record["model_sha256"], runner._sha256(Path(tmp) / "model.joblib"))
            with self.assertRaises(FileExistsError):
                runner.freeze_bundle(model, md, Path(tmp))

    def test_freeze_resolves_current_isolated_artifact_directory(self):
        rng = np.random.default_rng(18)
        X = rng.normal(size=(40, len(FEATURE_NAMES)))
        model = make_final_estimator("gaussian_nb")
        runner._fit(model, X, (X[:, 0] > 0).astype(int), np.ones(len(X)))
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "isolated-artifacts"
            md = {"model_name": "gaussian_nb", "run_name": "synthetic", "feature_version": FEATURE_VERSION}
            with patch.object(runner, "ART", target):
                record = runner.freeze_bundle(model, md)
            self.assertTrue((target / "model.joblib").is_file())
            self.assertEqual(record["model_sha256"], runner._sha256(target / "model.joblib"))

    def test_acceptance_and_partial_schema_guards_precede_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, art, report = (Path(tmp) / name for name in ("out", "art", "report"))
            art.mkdir(); report.mkdir()
            (art / "feature_schema.json").write_text("{}")
            with patch.object(runner, "OUT", out), patch.object(runner, "ART", art), patch.object(runner, "load_train_and_folds") as load:
                with self.assertRaises(FileExistsError):
                    runner.stage_train()
                load.assert_not_called()
            (report / "all_five_model_acceptance.json").write_text("{}")
            with patch.object(runner, "OUT", out), patch.object(runner, "REPORTS", report), patch.object(runner, "load_cached_dataset") as load:
                with self.assertRaises(FileExistsError):
                    runner.stage_student()
                load.assert_not_called()

    def test_isolated_paths_and_main_restore(self):
        a, b, c = runner.isolated_run_paths("reproduce-2026")
        self.assertEqual(a, runner.ROOT / "experiments/final_five_runs/reproduce-2026")
        self.assertEqual(b, runner.ROOT / "artifacts/final_five_runs/reproduce-2026")
        self.assertEqual(c, runner.ROOT / "reports/final_five_runs/reproduce-2026")
        for name in ("", "../oops", "A", "a/b", "a" * 49):
            with self.assertRaises(ValueError):
                runner.isolated_run_paths(name)
        old = runner.OUT, runner.ART, runner.REPORTS
        with patch.object(runner, "stage_train", side_effect=RuntimeError("synthetic")):
            with self.assertRaises(RuntimeError):
                runner.main(["--stage", "train", "--run-id", "unit-test"])
        self.assertEqual((runner.OUT, runner.ART, runner.REPORTS), old)


if __name__ == "__main__":
    unittest.main()
