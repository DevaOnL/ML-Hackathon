"""Tests for scripts/build_phase_three_report.py.

The builder is the last line of defense between the evidence files and the
submitted PDF, so these tests pin the pieces most likely to silently rot:

* dig bracket selectors (comparison[model=bagging], trip=00000076,
  objectives[configuration=A_fitted]) - a selector that stops matching
  turns a bound number into a PENDING marker.
* the computed-delta family-prefix translation (sel_model -> bag_cv_f1),
  which broke when the selected family id and the table fact prefix
  diverged.
* the end-to-end --check contract: every declared fact resolves, every
  guard passes, and the readiness verdict is READY.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_phase_three_report.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_phase_three_report", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DigSelectorTests(unittest.TestCase):
    def test_bracket_selectors(self):
        mod = load_builder()
        results = json.loads(
            (ROOT / "experiments" / "phase_three" / "results.json").read_text()
        )
        comparison = results["comparison"]
        rec = mod.dig({"comparison": comparison}, "comparison[model=bagging].family")
        # the family field carries a human descriptor, not the family id
        self.assertIn("bootstrap", rec)
        # the tuned row is selected by absence from the phase families
        tuned = mod.dig({"comparison": comparison}, "comparison[model=~tuned].parameters")
        self.assertIsInstance(tuned, str)

        per_trip = json.loads(
            (ROOT / "analysis" / "data_audit" / "per_trip.json").read_text()
        )
        broken = mod.dig(
            per_trip, "trip=00000076.route_validity.arc_index_offset_0.broken_joins"
        )
        self.assertEqual(broken, 1)

        freeze = json.loads(
            (ROOT / "reports" / "phase_three" / "decoder" / "freeze.json").read_text()
        )
        self.assertEqual(mod.dig(freeze, "len:calibration_trip_ids"), 8)
        a_f1 = mod.dig(freeze, "objectives[configuration=A_fitted].mean_length_f1")
        self.assertAlmostEqual(a_f1, 0.9530675244860136, places=12)

    def test_generic_selector_matches_declared_field(self):
        mod = load_builder()
        rows = [{"model_id": "x", "v": 1}, {"model_id": "y", "v": 2}]
        # a generic key=value selector must match the field as written, not
        # silently fall back to a different key
        self.assertEqual(mod.dig(rows, "model_id=y.v"), 2)
        with self.assertRaises(KeyError):
            mod.dig(rows, "model=x.v")


class ComputedDeltaTests(unittest.TestCase):
    def test_right_from_translates_family_prefix(self):
        mod = load_builder()
        values = {
            "tuned_cv_f1": {"raw": 0.5},
            "sel_model": {"raw": "bagging"},
            "bag_cv_f1": {"raw": 0.4},
        }
        mod.resolve_computed(values)
        rec = values["tuned_gain_cv_f1"]
        self.assertIsNone(rec["error"])
        self.assertAlmostEqual(rec["raw"], 0.1, places=12)


class ReadinessContractTests(unittest.TestCase):
    def test_complete_delivery_and_new_evaluator_guards_fail_closed(self):
        mod=load_builder()
        inputs=mod.collect_inputs()
        data=mod.load_data(inputs)
        raw=mod.MD_PATH.read_text()
        values=mod.resolve_fact_values(data)
        resolved=mod.resolve_tokens(raw,values)
        def guard_for(payload,name):
            return next(x for x in mod.run_phase_guards(payload,resolved,raw) if x["name"]==name)["ok"]
        self.assertTrue(guard_for(data,"complete_delivery_reconciliation"))
        mutations=[("missing",None),("model", "0"*64),("config","0"*64),("global",False),("trip",False),("defect","bad")]
        for label,value in mutations:
            changed=copy.deepcopy(data)
            delivery=changed["delivery"]
            if label=="missing":changed.pop("delivery")
            elif label=="model":delivery["model_sha256"]=value
            elif label=="config":delivery["config_file_sha256"]=value
            elif label=="global":delivery["checks"][next(iter(delivery["checks"]))]=False
            elif label=="trip":
                trip=next(iter(delivery["trips"].values()))
                trip["checks"][next(iter(trip["checks"]))]=False
            else:delivery["defects"]=[value]
            with self.subTest(mutation=label):
                self.assertFalse(guard_for(changed,"complete_delivery_reconciliation"))
        changed=copy.deepcopy(data)
        changed["scope"]["deferred_evaluator_website_request"]["used_for_training_selection_or_decoder"]=True
        self.assertFalse(guard_for(changed,"new_evaluator_request_deferred"))

    def test_final_prose_distinguishes_misses_continuity_and_reported_new_data(self):
        mod=load_builder()
        path=mod.TEMPLATE_PATH if mod.TEMPLATE_PATH.exists() else mod.MD_PATH
        text=path.read_text()
        for stale in ("The reserved evaluator partition was never supplied", "No evaluator data is present anywhere", "up to {{fact:dec_retry}} attempts", "is a multinomial logistic classifier", "every number was typed"):
            self.assertNotIn(stale,text)
        for required in ("not listed, inspected or processed", "empty candidate set", "48 is a maximum width, not a retry count", "not membership in any part of the reference route", "Neither is observation-weighted", "not a globally blind final test"):
            self.assertIn(required,text)

    def test_check_reports_ready(self):
        proc = subprocess.run(
            [sys.executable, str(BUILDER), "--check"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("0 unresolved/error", proc.stdout)
        for line in proc.stdout.splitlines():
            if line.startswith("guards:"):
                self.assertIn("18/18 pass", line, line)
        self.assertIn("READY FOR --final", proc.stdout)


if __name__ == "__main__":
    unittest.main()
