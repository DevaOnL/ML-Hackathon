"""Data-free source-frozen dense route confirmation policy tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import asdict
from mapmatch.matcher import MatchConfig
from scripts import accuracy_route_confirmation as r

class RouteConfirmationTests(unittest.TestCase):
    def test_baseline_selects_own_best_decoder_on_same_tuning_routes(self):
        def item(name,beta,old,new,order):return {"name":name,"match_config":asdict(MatchConfig(beta_m=beta)),"transition_policy":{},"order":order,"candidate":{"length_f1":new,"edge_f1":new},"v1_control":{"length_f1":old,"edge_f1":old}}
        a=item("a",2,.99,.97,0);b=item("b",5,.95,.995,1)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/"selection.json";p.write_text(json.dumps({"all_configurations":[a,b]}));frozen={"decoder_selection":{"configuration":b,"path":"selection.json"}}
            with patch.object(r.common,"ROOT",root):c=r.frozen_configurations(frozen)
        self.assertEqual(c["baseline"]["name"],"a");self.assertEqual(c["candidate"]["name"],"b");self.assertEqual(c["baseline"]["config"].beta_m,2)
    def test_unknown_reserved_trip_fails_before_data_access(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"protocol.json";p.write_text(json.dumps({"confirmation_trip_ids":["yes"]}))
            with patch.object(r.confirmation,"verify_frozen",return_value={}),patch.object(r.common,"PROTOCOL_PATH",p),patch.object(r,"discover_trips",side_effect=AssertionError("must not open")):
                with self.assertRaisesRegex(ValueError,"reserved"):r.run_trip("no")
if __name__=="__main__":unittest.main()
