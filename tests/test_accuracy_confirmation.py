"""No-data tests for the one-shot internal confirmation guards."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import accuracy_confirmation as c

class ConfirmationTests(unittest.TestCase):
    def test_trip_delta_pairs_by_id_not_input_order(self):
        a={"per_trip":[{"trip_id":"a","top1_accuracy":.7},{"trip_id":"b","top1_accuracy":.8}]};b={"per_trip":[{"trip_id":"b","top1_accuracy":.9},{"trip_id":"a","top1_accuracy":.9}]}
        x=c.paired_trip_delta(a,b);self.assertAlmostEqual(x["mean_delta"],.15);self.assertEqual(x["improved_trips"],2);self.assertEqual(x,c.paired_trip_delta(a,b))
    def test_mismatched_trip_set_rejected(self):
        with self.assertRaises(ValueError):c.paired_trip_delta({"per_trip":[{"trip_id":"a","top1_accuracy":.5}]},{"per_trip":[{"trip_id":"b","top1_accuracy":.5}]})
    def test_duplicate_trip_rows_cannot_collapse_bootstrap_population(self):
        a={"per_trip":[{"trip_id":"a","top1_accuracy":.5}]*16}
        with self.assertRaisesRegex(ValueError,"unique"):c.paired_trip_delta(a,a)

    def test_empty_trip_bootstrap_refused(self):
        with self.assertRaisesRegex(ValueError,"nonempty"):c.paired_trip_delta({"per_trip":[]},{"per_trip":[]})

    def test_all_declines_remain_negative(self):
        a={"per_trip":[{"trip_id":"a","top1_accuracy":1.}]};b={"per_trip":[{"trip_id":"a","top1_accuracy":0.}]}
        d=c.paired_trip_delta(a,b);self.assertEqual(d["mean_delta"],-1);self.assertEqual(d["decreased_trips"],1);self.assertEqual(d["trip_bootstrap_95_percentile_interval"],[-1,-1])
    def test_transitive_trial_dependency_drift_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/"protocol.json";p.write_text("{}");dep=root/"scorer.py";dep.write_text("score=1\n");trial=root/"trial.json";trial.write_text(json.dumps({"source_sha256":{"scorer.py":c.common.sha256_file(dep)}}));f=root/"frozen.json"
            d={"protocol_sha256":c.common.sha256_file(p),"source_sha256":{},"families":{"bagging":{"baseline":{"trial_path":"trial.json","trial_sha256":c.common.sha256_file(trial)}}}};f.write_text(json.dumps(d))
            with patch.object(c,"FREEZE",f),patch.multiple(c.common,ROOT=root,PROTOCOL_PATH=p):
                c.verify_frozen();dep.write_text("score=2\n")
                with self.assertRaisesRegex(ValueError,"transitive"):c.verify_frozen()

    def test_source_and_protocol_drift_rejected_before_data(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/"protocol.json";p.write_text("{}\n");f=root/"frozen.json";source=root/"s.py";source.write_text("x=1\n")
            d={"protocol_sha256":c.common.sha256_file(p),"source_sha256":{"s.py":c.common.sha256_file(source)},"families":{}};f.write_text(json.dumps(d))
            with patch.object(c,"FREEZE",f),patch.multiple(c.common,ROOT=root,PROTOCOL_PATH=p):
                c.verify_frozen();source.write_text("x=2\n")
                with self.assertRaisesRegex(ValueError,"source changed"):c.verify_frozen()
                source.write_text("x=1\n");p.write_text("{ }\n")
                with self.assertRaisesRegex(ValueError,"protocol changed"):c.verify_frozen()
if __name__=="__main__":unittest.main()
