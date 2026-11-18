"""Tests for scripts/prepare_phase_three_inputs.py.

Every fixture here is synthetic and lives in a temporary directory; nothing in
these tests touches the real project tree, the shared caches, or the raw
Student folder. Live read-only evidence for the actual project is documented in
docs/PHASE_REPRODUCTION.md and is deliberately kept separate from this file.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import prepare_phase_three_inputs as prep  # noqa: E402
from mapmatch.experiments import (  # noqa: E402
    CandidateDataset,
    ExperimentConfig,
    _fit,
    _metrics,
    build_group_folds,
    make_estimator,
    save_json,
)
from mapmatch.features import FEATURE_NAMES  # noqa: E402
from mapmatch.learning import predict_positive_proba  # noqa: E402

import joblib  # noqa: E402

DATASET_FIELDS = [f for f in CandidateDataset.__dataclass_fields__ if f != "coverage"]


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def make_raw_trips(data_root: Path, split: str, trips: int, payload_seed: int = 0) -> None:
    """Minimal raw trip layout: .track/.nodes/.arcs/.route byte files per trip."""
    directory = data_root / split
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(trips):
        trip_dir = directory / f"trip_{index:02d}"
        trip_dir.mkdir(parents=True, exist_ok=True)
        for suffix in (".track", ".nodes", ".arcs", ".route"):
            payload = f"{payload_seed}-{index}-{suffix}".encode()
            (trip_dir / (trip_dir.name + suffix)).write_bytes(payload)


def synthetic_dataset(trips: int = 3, obs_per_trip: int = 4, cands: int = 3,
                      seed: int = 7, split: str = "train") -> CandidateDataset:
    """A small but structurally valid CandidateDataset (one target per observation)."""
    rng = np.random.default_rng(seed)
    n_obs = trips * obs_per_trip
    rows = n_obs * cands
    X = rng.normal(size=(rows, len(FEATURE_NAMES))).astype(np.float32)
    y = np.zeros(rows, dtype=np.int8)
    offsets = [0]
    trip_ids: list[str] = []
    observation_indices: list[int] = []
    reference_edges: list[int] = []
    candidate_edges: list[int] = []
    candidate_fractions: list[float] = []
    candidate_distances: list[float] = []
    proxy_distances: list[float] = []
    ambiguous: list[bool] = []
    expanded: list[bool] = []
    row = 0
    for trip in range(trips):
        for observation in range(obs_per_trip):
            reference = (trip + observation) % cands
            for candidate in range(cands):
                edge = 1000 + 7 * candidate
                y[row] = 1 if candidate == reference else 0
                candidate_edges.append(edge)
                candidate_fractions.append(float(rng.uniform(0.0, 1.0)))
                candidate_distances.append(float(rng.uniform(1.0, 50.0)))
                row += 1
            reference_edges.append(1000 + 7 * reference)
            trip_ids.append(f"trip_{trip:02d}")
            observation_indices.append(observation)
            proxy_distances.append(float(rng.uniform(1.0, 30.0)))
            ambiguous.append(False)
            expanded.append(True)
            offsets.append(row)
    dataset = CandidateDataset(
        X=X,
        y=y,
        offsets=np.array(offsets, dtype=np.int64),
        observation_trip_ids=np.array(trip_ids),
        observation_indices=np.array(observation_indices, dtype=np.int64),
        reference_edges=np.array(reference_edges, dtype=np.int64),
        candidate_edges=np.array(candidate_edges, dtype=np.int64),
        candidate_fractions=np.array(candidate_fractions, dtype=np.float64),
        candidate_distances=np.array(candidate_distances, dtype=np.float64),
        proxy_distances=np.array(proxy_distances, dtype=np.float64),
        ambiguous=np.array(ambiguous, dtype=bool),
        expanded=np.array(expanded, dtype=bool),
        coverage={},
    )
    return dataset


def write_cache(root: Path, cache_name: str, dataset: CandidateDataset, fingerprint: str, split: str) -> None:
    """Write a cache npz + coverage json the way build_dataset does."""
    cache_dir = root / "experiments" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz = cache_dir / f"{cache_name}.npz"
    np.savez_compressed(npz, **{key: getattr(dataset, key) for key in DATASET_FIELDS})
    coverage = {
        "split": split,
        "trip_count": len(np.unique(dataset.observation_trip_ids)),
        "total_observations": len(dataset.observation_trip_ids),
        "selected_observations": len(dataset.offsets) - 1,
        "candidate_rows": len(dataset.y),
        "sampling": "synthetic test fixture",
        "maximum_observations_per_trip": 4,
        "trips": [],
        "input_manifest": [],
        "fingerprint": fingerprint,
        "seconds": 0.0,
    }
    save_json(npz.with_suffix(".json"), coverage)


def build_spec(root: Path, data_root: Path, split: str, trips: int, dataset: CandidateDataset) -> dict:
    """Spec pinned to the synthetic raw data and cache (mirrors the real specs)."""
    spec = {
        "split": split,
        "trips": trips,
        "observations": len(dataset.offsets) - 1,
        "rows": len(dataset.y),
        "misses": prep.observation_target_misses(dataset.y, dataset.offsets),
    }
    spec["fingerprint"] = prep.recompute_fingerprint(data_root, spec)
    spec["cache"] = f"{split}_{spec['fingerprint'][:16]}"
    write_cache(root, spec["cache"], dataset, spec["fingerprint"], split)
    npz = root / "experiments" / "cache" / f"{spec['cache']}.npz"
    spec["sha256"] = prep.sha256_file(npz)
    spec["array_sha256"] = prep.npz_array_digests(npz)
    return spec


class TempProject:
    """Temporary repo root + raw data root with synthetic trips for both splits."""

    def __init__(self, train_trips: int = 3, student_trips: int = 3):
        self.dir = Path(tempfile.mkdtemp(prefix="phase_prep_test_"))
        self.root = self.dir / "repo"
        self.data_root = self.dir / "data"
        self.root.mkdir()
        make_raw_trips(self.data_root, "train", train_trips)
        make_raw_trips(self.data_root, "student_test", student_trips)
        self.train_dataset = synthetic_dataset(trips=train_trips, split="train")
        self.student_dataset = synthetic_dataset(trips=student_trips, seed=11, split="student_test")
        self.train_spec = build_spec(self.root, self.data_root, "train", train_trips, self.train_dataset)
        self.student_spec = build_spec(self.root, self.data_root, "student_test", student_trips, self.student_dataset)

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def fake_build_dataset(self, datasets: dict, spec_by_split: dict):
        """Replacement for mapmatch.experiments.build_dataset that writes the
        synthetic cache exactly like the real builder (which rejects synthetic
        trip counts)."""
        def build(data_root, split, config, output):
            dataset = datasets[split]
            spec = spec_by_split[split]
            write_cache(Path(output).parent, spec["cache"], dataset, spec["fingerprint"], split)
            return dataset
        return build

    def fake_build_dataset_corrupted(self, datasets: dict, spec_by_split: dict):
        """Like fake_build_dataset, but perturbs X so verification must fail."""
        def build(data_root, split, config, output):
            dataset = datasets[split]
            spec = spec_by_split[split]
            corrupt = synthetic_dataset(trips=spec["trips"], seed=99, split=split)
            write_cache(Path(output).parent, spec["cache"], corrupt, spec["fingerprint"], split)
            return dataset
        return build


def tree_snapshot(root: Path) -> dict:
    snapshot = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            snapshot[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def fitted_lr_bundle(train: CandidateDataset) -> dict:
    estimator = make_estimator("logistic_regression", prep.LR_PARAMETERS, prep.SHARED_CONFIG)
    _fit(estimator, train.X, train.y, train.training_weights())
    return {
        "estimator": estimator,
        "feature_names": list(FEATURE_NAMES),
        "feature_version": __import__("mapmatch.features", fromlist=["FEATURE_VERSION"]).FEATURE_VERSION,
        "metadata": {
            "model_name": "logistic_regression",
            "run_name": "logistic_regression",
            "seed": prep.SHARED_CONFIG.seed,
            "parameters": dict(prep.LR_PARAMETERS),
            "training_trip_ids": np.unique(train.observation_trip_ids).tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Fingerprint and digest helpers
# ---------------------------------------------------------------------------

class TestFingerprint(unittest.TestCase):
    def setUp(self):
        self.project = TempProject()

    def tearDown(self):
        self.project.cleanup()

    def test_deterministic(self):
        first = prep.recompute_fingerprint(self.project.data_root, self.project.train_spec)
        second = prep.recompute_fingerprint(self.project.data_root, self.project.train_spec)
        self.assertEqual(first, second)
        self.assertEqual(first, self.project.train_spec["fingerprint"])

    def test_sensitive_to_raw_bytes(self):
        original = prep.recompute_fingerprint(self.project.data_root, self.project.train_spec)
        target = next((self.project.data_root / "train").glob("trip_00/*.arcs"))
        target.write_bytes(target.read_bytes() + b"x")
        tampered = prep.recompute_fingerprint(self.project.data_root, self.project.train_spec)
        self.assertNotEqual(original, tampered)
        # A byte change in an unlisted file (not part of the manifest) is invisible.
        untouched = prep.recompute_fingerprint(self.project.data_root, self.project.train_spec)
        (self.project.data_root / "train" / "trip_00" / "extra.txt").write_bytes(b"noise")
        self.assertEqual(
            untouched,
            prep.recompute_fingerprint(self.project.data_root, self.project.train_spec),
        )

    def test_sensitive_to_config_budget(self):
        config_8 = ExperimentConfig(
            seed=20261118, train_observations_per_trip=350, student_observations_per_trip=0,
            folds=3, bayesian_trials=8, n_jobs=2,
            candidate_radius_m=80.0, max_radius_m=1000.0, max_candidates=12, sample_spacing_m=40.0,
        )
        self.assertNotEqual(
            prep.recompute_fingerprint(self.project.data_root, self.project.train_spec, config_8),
            prep.recompute_fingerprint(self.project.data_root, self.project.train_spec, prep.SHARED_CONFIG),
        )

    def test_trip_count_guard(self):
        with self.assertRaises(prep.PrepError):
            prep.recompute_fingerprint(self.project.data_root, {**self.project.train_spec, "trips": 2})


class TestDigestHelpers(unittest.TestCase):
    def setUp(self):
        self.project = TempProject()

    def tearDown(self):
        self.project.cleanup()

    def test_npz_array_digests(self):
        npz = self.project.root / "experiments" / "cache" / f"{self.project.train_spec['cache']}.npz"
        digests = prep.npz_array_digests(npz)
        self.assertEqual(set(digests), set(DATASET_FIELDS))
        self.assertEqual(digests["X"]["shape"], [len(self.project.train_dataset.y), len(FEATURE_NAMES)])
        self.assertEqual(digests["X"]["dtype"], "float32")
        with np.load(npz, allow_pickle=False) as archive:
            expected = hashlib.sha256(np.ascontiguousarray(archive["y"]).tobytes()).hexdigest()
        self.assertEqual(digests["y"]["sha256"], expected)

    def test_observation_target_misses(self):
        y = np.array([0, 1, 0, 0, 0, 1], dtype=np.int8)
        offsets = np.array([0, 3, 6])
        self.assertEqual(prep.observation_target_misses(y, offsets), 0)
        y2 = np.array([0, 0, 0, 0, 0, 1], dtype=np.int8)
        self.assertEqual(prep.observation_target_misses(y2, offsets), 1)

    def test_compare_metric_records(self):
        computed = {"top1_accuracy": 0.5, "rows": 10}
        recorded = {"top1_accuracy": 0.5 + 1e-12, "rows": 10}
        self.assertEqual(prep.compare_metric_records("m", computed, recorded), [])
        recorded_bad = {"top1_accuracy": 0.6, "rows": 10, "per_trip": {"a": 1}}
        mismatches = prep.compare_metric_records("m", computed, recorded_bad)
        self.assertEqual(len(mismatches), 1)
        self.assertIn("top1_accuracy", mismatches[0])


# ---------------------------------------------------------------------------
# Check stage
# ---------------------------------------------------------------------------

class TestCheckStage(unittest.TestCase):
    def setUp(self):
        self.project = TempProject()

    def tearDown(self):
        self.project.cleanup()

    def test_missing_everything_exits_2_and_writes_nothing(self):
        empty_root = self.project.dir / "empty_repo"
        empty_root.mkdir()
        empty_data = self.project.dir / "empty_data"
        empty_data.mkdir()
        before = tree_snapshot(self.project.dir)
        exit_code = prep.check_stage(empty_root, empty_data,
                                     train_spec=self.project.train_spec,
                                     student_spec=self.project.student_spec)
        self.assertEqual(exit_code, prep.EXIT_MISSING)
        self.assertEqual(tree_snapshot(self.project.dir), before)

    def test_all_present_exits_0(self):
        root, data_root = self.project.root, self.project.data_root
        dataset = self.project.train_dataset
        folds = build_group_folds(dataset, prep.SHARED_CONFIG)
        oof, aggregate, _, _ = prep.lr_grouped_oof(dataset, folds, prep.LR_PARAMETERS, prep.SHARED_CONFIG)
        experiments = root / "experiments"
        save_json(experiments / "cv_folds.json", [
            {"fold": fold["fold"],
             "train_trip_ids": fold["train_trip_ids"],
             "validation_trip_ids": fold["validation_trip_ids"],
             "train_observations": len(fold["train_observations"]),
             "validation_observations": len(fold["validation_observations"]),
             "train_candidate_rows": len(fold["train_rows"]),
             "validation_candidate_rows": len(fold["validation_rows"])}
            for fold in folds
        ])
        cv_dir = experiments / "cv"
        cv_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = experiments / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        (cv_dir / "logistic_regression.json").write_text(json.dumps({
            "parameters": dict(prep.LR_PARAMETERS),
            "oof_metrics": aggregate,
        }))
        np.savez_compressed(pred_dir / "logistic_regression_oof.npz",
                            **prep.prediction_arrays(dataset, oof))
        np.savez_compressed(pred_dir / "logistic_regression_student.npz",
                            **prep.prediction_arrays(self.project.student_dataset,
                                                      np.zeros(len(self.project.student_dataset.y))))
        (experiments / "evaluation" / "logistic_regression.json").parent.mkdir(parents=True, exist_ok=True)
        (experiments / "evaluation" / "logistic_regression.json").write_text(json.dumps({
            "parameters": dict(prep.LR_PARAMETERS),
            "student_metrics": {"top1_accuracy": 0.5},
        }))
        (experiments / "results.json").write_text(json.dumps({
            "target_policy": "proxy", "comparison": [{"model": "logistic_regression"}],
        }))
        (root / "artifacts" / "baselines").mkdir(parents=True, exist_ok=True)
        (root / "artifacts" / "baselines" / "logistic_regression.joblib").write_bytes(b"fixture")
        before = tree_snapshot(root)
        exit_code = prep.check_stage(root, data_root,
                                     train_spec=self.project.train_spec,
                                     student_spec=self.project.student_spec)
        self.assertEqual(exit_code, prep.EXIT_OK)
        self.assertEqual(tree_snapshot(root), before)  # strictly read-only

    def test_tampered_cache_fails(self):
        root = self.project.root
        npz = root / "experiments" / "cache" / f"{self.project.train_spec['cache']}.npz"
        corrupt = synthetic_dataset(seed=99)
        write_cache(root, self.project.train_spec["cache"], corrupt,
                    self.project.train_spec["fingerprint"], "train")
        self.assertNotEqual(prep.sha256_file(npz), self.project.train_spec["sha256"])
        exit_code = prep.check_stage(root, self.project.data_root,
                                     train_spec=self.project.train_spec,
                                     student_spec=self.project.student_spec)
        self.assertEqual(exit_code, prep.EXIT_FAILURE)

    def test_wrong_raw_fingerprint_fails(self):
        target = next((self.project.data_root / "train").glob("trip_01/*.nodes"))
        target.write_bytes(b"tampered")
        exit_code = prep.check_stage(self.project.root, self.project.data_root,
                                     train_spec=self.project.train_spec,
                                     student_spec=self.project.student_spec)
        self.assertEqual(exit_code, prep.EXIT_FAILURE)


# ---------------------------------------------------------------------------
# LR mirror equivalence (the central reuse claim)
# ---------------------------------------------------------------------------

class TestLrMirrorEquivalence(unittest.TestCase):
    def setUp(self):
        self.project = TempProject()

    def tearDown(self):
        self.project.cleanup()

    def test_mirror_matches_cross_validate_model(self):
        from mapmatch.experiments import cross_validate_model

        dataset = self.project.train_dataset
        folds = build_group_folds(dataset, prep.SHARED_CONFIG)
        oof, aggregate, _, _ = prep.lr_grouped_oof(dataset, folds, prep.LR_PARAMETERS, prep.SHARED_CONFIG)
        record = cross_validate_model("logistic_regression", prep.LR_PARAMETERS, dataset, folds,
                                      prep.SHARED_CONFIG, self.project.root / "experiments",
                                      "logistic_regression")
        np.testing.assert_allclose(oof, np.load(self.project.root / "experiments" / "predictions" / "logistic_regression_oof.npz")["probabilities"], rtol=0, atol=1e-12)
        for key, value in aggregate.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.assertAlmostEqual(value, record["oof_metrics"][key], places=10,
                                       msg=f"metric {key} diverged between mirror and shared CV")


# ---------------------------------------------------------------------------
# Train stage
# ---------------------------------------------------------------------------

class TestTrainStage(unittest.TestCase):
    def setUp(self):
        self.project = TempProject()
        # The stages only fill gaps: remove the caches the spec builder wrote.
        spec = self.project.train_spec
        npz = self.project.root / "experiments" / "cache" / f"{spec['cache']}.npz"
        npz.unlink(missing_ok=True)
        npz.with_suffix(".json").unlink(missing_ok=True)

    def tearDown(self):
        self.project.cleanup()

    def run_stage(self, builder):
        with mock.patch.object(prep, "build_dataset", builder):
            return prep.train_stage(self.project.root, self.project.data_root,
                                    train_spec=self.project.train_spec)

    def test_refuses_to_overwrite_existing_cache(self):
        npz = self.project.root / "experiments" / "cache" / f"{self.project.train_spec['cache']}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        npz.write_bytes(b"historical")
        with self.assertRaises(prep.PrepError) as ctx:
            self.run_stage(self.project.fake_build_dataset(
                {"train": self.project.train_dataset}, {"train": self.project.train_spec}))
        self.assertIn("refusing", str(ctx.exception))

    def test_refuses_to_overwrite_existing_oof(self):
        oof = self.project.root / "experiments" / "predictions" / "logistic_regression_oof.npz"
        oof.parent.mkdir(parents=True, exist_ok=True)
        oof.write_bytes(b"historical")
        with self.assertRaises(prep.PrepError) as ctx:
            self.run_stage(self.project.fake_build_dataset(
                {"train": self.project.train_dataset}, {"train": self.project.train_spec}))
        self.assertIn("refusing", str(ctx.exception))

    def test_fingerprint_gate_before_build(self):
        target = next((self.project.data_root / "train").glob("trip_00/*.route"))
        target.write_bytes(b"tampered")
        builder = self.project.fake_build_dataset(
            {"train": self.project.train_dataset}, {"train": self.project.train_spec})
        spy = mock.MagicMock(side_effect=builder)
        with mock.patch.object(prep, "build_dataset", spy):
            with self.assertRaises(prep.PrepError) as ctx:
                self.run_stage(builder)
        spy.assert_not_called()
        self.assertIn("fingerprint", str(ctx.exception))

    def test_sha_honesty_on_corrupt_rebuild(self):
        builder = self.project.fake_build_dataset_corrupted(
            {"train": self.project.train_dataset}, {"train": self.project.train_spec})
        with self.assertRaises(prep.PrepError) as ctx:
            self.run_stage(builder)
        self.assertIn("pinned verification", str(ctx.exception))
        npz = self.project.root / "experiments" / "cache" / f"{self.project.train_spec['cache']}.npz"
        self.assertTrue(npz.exists())  # left for diagnosis, honestly reported
        oof = self.project.root / "experiments" / "predictions" / "logistic_regression_oof.npz"
        self.assertFalse(oof.exists())
        self.assertFalse((self.project.root / "experiments" / "cv_folds.json").exists())

    def test_success_creates_folds_and_cv_records(self):
        exit_code = self.run_stage(self.project.fake_build_dataset(
            {"train": self.project.train_dataset}, {"train": self.project.train_spec}))
        self.assertEqual(exit_code, prep.EXIT_OK)
        experiments = self.project.root / "experiments"
        cache = experiments / "cache" / f"{self.project.train_spec['cache']}.npz"
        self.assertTrue(cache.exists())
        self.assertTrue((experiments / "cv_folds.json").exists())
        record = json.loads((experiments / "cv" / "logistic_regression.json").read_text())
        self.assertEqual(record["parameters"], prep.LR_PARAMETERS)
        oof_npz = experiments / "predictions" / "logistic_regression_oof.npz"
        self.assertTrue(oof_npz.exists())
        dataset = self.project.train_dataset
        expected_members = set(prep.prediction_arrays(
            dataset, np.zeros(len(dataset.y))).keys())
        with np.load(oof_npz, allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), expected_members)
            self.assertTrue(np.isfinite(archive["probabilities"]).all())
        # Round-trip: the runner-style load works on the produced cache.
        dataset = prep.load_verified_dataset(self.project.root, self.project.train_spec)
        self.assertEqual(len(dataset.y), self.project.train_spec["rows"])

    def test_mirror_path_when_cv_record_exists(self):
        # Committed CV record present, OOF arrays missing: only the LR mirror runs.
        dataset = self.project.train_dataset
        folds = build_group_folds(dataset, prep.SHARED_CONFIG)
        _, aggregate, _, _ = prep.lr_grouped_oof(dataset, folds, prep.LR_PARAMETERS, prep.SHARED_CONFIG)
        cv_dir = self.project.root / "experiments" / "cv"
        cv_dir.mkdir(parents=True, exist_ok=True)
        (cv_dir / "logistic_regression.json").write_text(json.dumps({
            "parameters": dict(prep.LR_PARAMETERS),
            "oof_metrics": aggregate,
        }))
        exit_code = self.run_stage(self.project.fake_build_dataset(
            {"train": dataset}, {"train": self.project.train_spec}))
        self.assertEqual(exit_code, prep.EXIT_OK)
        oof_npz = self.project.root / "experiments" / "predictions" / "logistic_regression_oof.npz"
        self.assertTrue(oof_npz.exists())

    def test_mirror_refuses_divergent_cv_record(self):
        cv_dir = self.project.root / "experiments" / "cv"
        cv_dir.mkdir(parents=True, exist_ok=True)
        (cv_dir / "logistic_regression.json").write_text(json.dumps({
            "parameters": dict(prep.LR_PARAMETERS),
            "oof_metrics": {"top1_accuracy": 0.123, "macro_f1": 0.456, "log_loss": 1.0},
        }))
        with self.assertRaises(prep.PrepError) as ctx:
            self.run_stage(self.project.fake_build_dataset(
                {"train": self.project.train_dataset}, {"train": self.project.train_spec}))
        self.assertIn("disagree", str(ctx.exception))
        self.assertIn("Nothing was written", str(ctx.exception))
        self.assertFalse((self.project.root / "experiments" / "predictions" / "logistic_regression_oof.npz").exists())


# ---------------------------------------------------------------------------
# Student stage (gated)
# ---------------------------------------------------------------------------

class TestStudentStage(unittest.TestCase):
    def setUp(self):
        self.project = TempProject()
        for spec in (self.project.train_spec, self.project.student_spec):
            npz = self.project.root / "experiments" / "cache" / f"{spec['cache']}.npz"
            npz.unlink(missing_ok=True)
            npz.with_suffix(".json").unlink(missing_ok=True)

    def tearDown(self):
        self.project.cleanup()

    def write_gates(self, root: Path, selected: str = "logistic_regression", sha: str | None = None):
        (root / "experiments" / "phase_three").mkdir(parents=True, exist_ok=True)
        (root / "experiments" / "phase_three" / "selection.json").write_text(json.dumps({
            "phase": "three-model-first",
            "selected_model": selected,
            "families_considered": list(prep.PHASE_FAMILIES),
        }))
        artifacts = root / "artifacts" / "phase_three"
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "model.joblib").write_bytes(b"frozen model bytes")
        freeze = {
            "model": "artifacts/phase_three/model.joblib",
            "model_name": selected,
            "sha256": sha or hashlib.sha256(b"frozen model bytes").hexdigest(),
        }
        (artifacts / "model_freeze.json").write_text(json.dumps(freeze))

    def run_stage(self, builder, allow_lr_fit=False):
        with mock.patch.object(prep, "build_dataset", builder):
            return prep.student_stage(self.project.root, self.project.data_root,
                                      allow_lr_fit=allow_lr_fit,
                                      train_spec=self.project.train_spec,
                                      student_spec=self.project.student_spec)

    def test_gate_blocks_without_selection(self):
        self.write_gates(self.project.root)
        (self.project.root / "experiments" / "phase_three" / "selection.json").unlink()
        builder = self.project.fake_build_dataset(
            {"student_test": self.project.student_dataset}, {"student_test": self.project.student_spec})
        spy = mock.MagicMock(side_effect=builder)
        with mock.patch.object(prep, "build_dataset", spy):
            with self.assertRaises(prep.PrepError) as ctx:
                self.run_stage(builder)
        spy.assert_not_called()  # no student access before the gates pass
        self.assertIn("selection.json", str(ctx.exception))

    def test_gate_blocks_without_freeze(self):
        builder = self.project.fake_build_dataset(
            {"student_test": self.project.student_dataset}, {"student_test": self.project.student_spec})
        # selection.json exists (from a partial gate write) but no freeze record
        (self.project.root / "experiments" / "phase_three").mkdir(parents=True, exist_ok=True)
        (self.project.root / "experiments" / "phase_three" / "selection.json").write_text(json.dumps({
            "selected_model": "logistic_regression",
        }))
        spy = mock.MagicMock(side_effect=builder)
        with mock.patch.object(prep, "build_dataset", spy):
            with self.assertRaises(prep.PrepError) as ctx:
                self.run_stage(builder)
        spy.assert_not_called()
        self.assertIn("model_freeze.json", str(ctx.exception))

    def test_gate_blocks_on_freeze_sha_mismatch(self):
        self.write_gates(self.project.root, sha="0" * 64)
        builder = self.project.fake_build_dataset(
            {"student_test": self.project.student_dataset}, {"student_test": self.project.student_spec})
        spy = mock.MagicMock(side_effect=builder)
        with mock.patch.object(prep, "build_dataset", spy):
            with self.assertRaises(prep.PrepError) as ctx:
                self.run_stage(builder)
        spy.assert_not_called()
        self.assertIn("sha256", str(ctx.exception))

    def test_gate_blocks_out_of_scope_selection(self):
        self.write_gates(self.project.root, selected="gradient_boosting")
        builder = self.project.fake_build_dataset(
            {"student_test": self.project.student_dataset}, {"student_test": self.project.student_spec})
        spy = mock.MagicMock(side_effect=builder)
        with mock.patch.object(prep, "build_dataset", spy):
            with self.assertRaises(prep.PrepError):
                self.run_stage(builder)
        spy.assert_not_called()

    def prepare_baseline_and_eval(self):
        bundle = fitted_lr_bundle(self.project.train_dataset)
        baseline = self.project.root / "artifacts" / "baselines" / "logistic_regression.joblib"
        baseline.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, baseline, compress=3)
        probabilities = predict_positive_proba(bundle["estimator"], self.project.student_dataset.X)
        metrics = _metrics(self.project.student_dataset, probabilities)
        evaluation = {
            "parameters": dict(prep.LR_PARAMETERS),
            "student_metrics": metrics,
        }
        eval_path = self.project.root / "experiments" / "evaluation" / "logistic_regression.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps(evaluation))
        return probabilities

    def test_valid_gates_full_run(self):
        self.write_gates(self.project.root)
        expected_probabilities = self.prepare_baseline_and_eval()
        builder = self.project.fake_build_dataset(
            {"student_test": self.project.student_dataset}, {"student_test": self.project.student_spec})
        exit_code = self.run_stage(builder)
        self.assertEqual(exit_code, prep.EXIT_OK)
        student_npz = (self.project.root / "experiments" / "predictions"
                       / "logistic_regression_student.npz")
        self.assertTrue(student_npz.exists())
        with np.load(student_npz, allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), set(prep.prediction_arrays(self.project.student_dataset, expected_probabilities)))
            np.testing.assert_allclose(archive["probabilities"], expected_probabilities, rtol=0, atol=1e-12)

    def test_metric_mismatch_refuses_before_writing(self):
        self.write_gates(self.project.root)
        self.prepare_baseline_and_eval()
        eval_path = self.project.root / "experiments" / "evaluation" / "logistic_regression.json"
        record = json.loads(eval_path.read_text())
        record["student_metrics"]["top1_accuracy"] = record["student_metrics"]["top1_accuracy"] + 0.1
        eval_path.write_text(json.dumps(record))
        builder = self.project.fake_build_dataset(
            {"student_test": self.project.student_dataset}, {"student_test": self.project.student_spec})
        with self.assertRaises(prep.PrepError) as ctx:
            self.run_stage(builder)
        self.assertIn("disagree", str(ctx.exception))
        self.assertFalse((self.project.root / "experiments" / "predictions"
                          / "logistic_regression_student.npz").exists())

    def test_allow_lr_fit_explicit_regeneration(self):
        self.write_gates(self.project.root)
        self.prepare_baseline_and_eval()
        (self.project.root / "artifacts" / "baselines" / "logistic_regression.joblib").unlink()
        # The fresh all-80 fit reads the TRAIN cache; restore the one setUp removed.
        spec = self.project.train_spec
        write_cache(self.project.root, spec["cache"], self.project.train_dataset,
                    spec["fingerprint"], "train")
        builder = self.project.fake_build_dataset(
            {"student_test": self.project.student_dataset}, {"student_test": self.project.student_spec})
        exit_code = self.run_stage(builder, allow_lr_fit=True)
        self.assertEqual(exit_code, prep.EXIT_OK)
        baseline = self.project.root / "artifacts" / "baselines" / "logistic_regression.joblib"
        self.assertTrue(baseline.exists())
        reloaded = joblib.load(baseline)
        self.assertIn("regenerated_by", reloaded["metadata"])
        self.assertTrue((self.project.root / "experiments" / "predictions"
                         / "logistic_regression_student.npz").exists())

    def test_missing_baseline_refuses_without_opt_in(self):
        self.write_gates(self.project.root)
        self.prepare_baseline_and_eval()
        (self.project.root / "artifacts" / "baselines" / "logistic_regression.joblib").unlink()
        builder = self.project.fake_build_dataset(
            {"student_test": self.project.student_dataset}, {"student_test": self.project.student_spec})
        with self.assertRaises(prep.PrepError) as ctx:
            self.run_stage(builder)
        self.assertIn("--allow-lr-fit", str(ctx.exception))

    def test_missing_eval_record_refuses(self):
        self.write_gates(self.project.root)
        # No evaluation record: this tool does not fabricate measured records.
        builder = self.project.fake_build_dataset(
            {"student_test": self.project.student_dataset}, {"student_test": self.project.student_spec})
        with self.assertRaises(prep.PrepError) as ctx:
            self.run_stage(builder)
        self.assertIn("evaluation", str(ctx.exception))
        # Student raw data was fingerprint-gated first; the cache must not exist
        # because the gate on the committed evaluation record fires before build.
        self.assertFalse((self.project.root / "experiments" / "cache"
                          / f"{self.project.student_spec['cache']}.npz").exists())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCli(unittest.TestCase):
    def test_exit_codes_are_distinct(self):
        self.assertEqual(prep.EXIT_OK, 0)
        self.assertEqual(prep.EXIT_FAILURE, 1)
        self.assertEqual(prep.EXIT_MISSING, 2)

    def test_main_check_on_missing_project_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            data = Path(tmp) / "data"
            data.mkdir()
            exit_code = prep.main(["--stage", "check", "--root", str(root), "--data-root", str(data)])
            self.assertEqual(exit_code, prep.EXIT_MISSING)


if __name__ == "__main__":
    unittest.main()
