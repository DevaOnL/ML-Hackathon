"""Data-free safety tests for isolated TRAIN-only accuracy validation."""
from __future__ import annotations
from dataclasses import fields
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scripts import accuracy_common as common
from mapmatch.experiments import CandidateDataset


def tiny_dataset():
    n=30
    return CandidateDataset(X=np.arange(n*2*38,dtype=float).reshape(n*2,38)/100, y=np.tile([1,0],n).astype(np.int8), offsets=np.arange(0,n*2+1,2), observation_trip_ids=np.repeat([f"{i:08d}" for i in range(6)],5), observation_indices=np.tile(np.arange(5),6), reference_edges=np.zeros(n,dtype=np.int64), candidate_edges=np.tile([0,1],n), candidate_fractions=np.full(n*2,.5), candidate_distances=np.tile([1.,2.],n), proxy_distances=np.ones(n), ambiguous=np.zeros(n,bool), expanded=np.zeros(n,bool), coverage={"split":"train"})


class CommonSafetyTests(unittest.TestCase):
    def test_hash_split_independent_of_input_order_and_no_overlap(self):
        ids=[f"{i:08d}" for i in range(80)]
        a,b=common.split_trip_ids(ids)
        self.assertEqual((len(a),len(b)),(64,16))
        self.assertFalse(set(a)&set(b))
        self.assertEqual(set(a)|set(b),set(ids))
        self.assertEqual((a,b),common.split_trip_ids(ids[::-1]+ids))

    def test_subset_preserves_exact_proxy_candidate_groups(self):
        d=tiny_dataset(); obs=np.array([0,2,11])
        s=common.subset_dataset(d,obs)
        np.testing.assert_array_equal(s.offsets,[0,2,4,6])
        np.testing.assert_array_equal(s.X,d.X[[0,1,4,5,22,23]])
        np.testing.assert_array_equal(s.y,d.y[[0,1,4,5,22,23]])
        np.testing.assert_array_equal(s.reference_edges,d.reference_edges[obs])
        s.X[0,0]=999
        self.assertNotEqual(d.X[0,0],999)

    def test_invalid_subsets_fail_not_silently_sorted(self):
        d=tiny_dataset()
        for obs in ([2,1],[1,1],[-1],[30],[1.2]):
            with self.subTest(obs=obs),self.assertRaises(ValueError):common.subset_dataset(d,np.array(obs))

    def test_folds_whole_trip_and_partition_observations(self):
        d=tiny_dataset(); folds=common.get_folds(d)
        common.validate_folds(d,folds)
        for f in folds:self.assertFalse(set(f["train_trip_ids"])&set(f["validation_trip_ids"]))

    def test_folds_reject_one_missing_row_even_if_ids_match(self):
        d=tiny_dataset(); folds=common.get_folds(d); folds[0]["train_rows"]=folds[0]["train_rows"][1:]
        with self.assertRaisesRegex(ValueError,"exactly match"):common.validate_folds(d,folds)

    def test_strict_probabilities_handle_reversed_class_columns(self):
        class Estimator:
            classes_=np.array([1,0])
            def predict_proba(self,X):return np.tile([.8,.2],(len(X),1))
        p,checks=common.strict_positive_probabilities(Estimator(),np.ones((3,2),np.float32))
        np.testing.assert_array_equal(p,[.8]*3);self.assertLessEqual(checks["maximum_row_sum_error"],1e-10)

    def test_bad_probability_rows_are_not_repaired(self):
        class Estimator:
            classes_=np.array([0,1])
            def predict_proba(self,X):return self.values
        e=Estimator()
        for values in ([[.8,.3]], [[-1.,2.]], [[np.nan,1.]], [[.5,.50000001]], [[.2,.3,.5]]):
            e.values=np.array(values)
            with self.subTest(values=values),self.assertRaises(ValueError):common.strict_positive_probabilities(e,np.ones((1,2)))

    def test_write_new_refuses_overwrite_and_json_nan(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"out.json";common.write_new_json(p,{"a":1})
            with self.assertRaises(FileExistsError):common.write_new_json(p,{"a":2})
            self.assertIn("1",p.read_text())
            with self.assertRaises(ValueError):common.write_new_json(Path(td)/"nan.json",{"x":float("nan")})

    def test_canonical_output_and_private_output_are_refused_before_data(self):
        with patch.object(common,"load_train_dataset",side_effect=AssertionError("must not load")):
            for p in (common.ROOT/"experiments/final_five/current", common.PRIVATE_ROOT/"bad",common.EXPERIMENT_ROOT):
                with self.subTest(path=p),self.assertRaises(ValueError):common.run_cv_trial("bagging","safe",{},"v1",p,lambda *a:None)

    def test_trial_names_and_unknown_family_are_refused(self):
        with self.assertRaises(ValueError):common.run_cv_trial("bagging","../escape",{},"v1",common.EXPERIMENT_ROOT/"safe",lambda *a:None)
        with self.assertRaises(ValueError):common.run_cv_trial("not-a-family","safe",{},"v1",common.EXPERIMENT_ROOT/"safe",lambda *a:None)

    def test_preregistered_fold_or_parameter_drift_is_refused(self):
        d=tiny_dataset();folds=common.get_folds(d);p={"folds":[{k:f[k] for k in ("fold","train_trip_ids","validation_trip_ids")} for f in folds],"family_menus":{"bagging":[{"family":"bagging","run_name":"b","profile":"v1","parameters":{"n":2}}]},"source_TRAIN_cache_sha256":common.TRAIN_CACHE_SHA256}
        common.validate_registered_trial(p,"bagging","b_screen",{"n":2},"v1",folds)
        with self.assertRaisesRegex(ValueError,"parameters"):common.validate_registered_trial(p,"bagging","b_screen",{"n":3},"v1",folds)
        with self.assertRaisesRegex(ValueError,"menu"):common.validate_registered_trial(p,"bagging","unknown_screen",{"n":2},"v1",folds)
        import copy
        wrong=copy.deepcopy(p);wrong["folds"][0]["train_trip_ids"]=[]
        with self.assertRaisesRegex(ValueError,"folds"):common.validate_registered_trial(wrong,"bagging","b_screen",{"n":2},"v1",folds)

    def test_complete_tiny_oof_roundtrip_has_no_missing_or_mixed_rows(self):
        from scripts.accuracy_trees import make_bagging_estimator
        import json,joblib
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);exp=root/"experiments/accuracy_improvement";private=exp/"private";protocol=exp/"PROTOCOL.json"
            data=tiny_dataset(); declared=[{k:f[k] for k in ("fold","train_trip_ids","validation_trip_ids")} for f in common.get_folds(data)]
            params={"n_estimators":2,"max_samples":.7,"max_depth":2,"min_samples_leaf":1}
            common.write_new_json(protocol,{"training_authorized":True,"folds":declared,"family_menus":{"bagging":[{"family":"bagging","run_name":"tiny","profile":"v1","parameters":params}]},"source_TRAIN_cache_sha256":common.TRAIN_CACHE_SHA256})
            with patch.multiple(common,ROOT=root,EXPERIMENT_ROOT=exp,PRIVATE_ROOT=private,PROTOCOL_PATH=protocol),patch.object(common,"load_train_dataset",return_value=tiny_dataset()):
                result=common.run_cv_trial("bagging","tiny",{"n_estimators":2,"max_samples":.7,"max_depth":2,"min_samples_leaf":1},"v1",exp/"toy",make_bagging_estimator)
                self.assertTrue(result["full_three_fold"]);self.assertEqual(len(result["folds"]),3);self.assertEqual(result["oof_metrics"]["observations"],30)
                with np.load(private/"toy/tiny/probabilities.npz",allow_pickle=False) as a:
                    self.assertEqual(len(a["probabilities"]),60);np.testing.assert_array_equal(a["y"],tiny_dataset().y)
                b=joblib.load(private/"toy/tiny/fold_0.joblib");self.assertFalse(set(b["train_trip_ids"])&set(b["validation_trip_ids"]))
                self.assertEqual(b["base_feature_version"],common.FEATURE_VERSION)
                with self.assertRaises(FileExistsError):common.run_cv_trial("bagging","tiny",params,"v1",exp/"toy",make_bagging_estimator)

    def test_baseline_profile_preserves_values_float64(self):
        d=tiny_dataset()
        np.testing.assert_array_equal(common.profile_features(d.X,d.offsets,"v1"),d.X)
        self.assertEqual(common.profile_names("baseline38"),common.profile_names("v1"))


if __name__=="__main__":unittest.main()
