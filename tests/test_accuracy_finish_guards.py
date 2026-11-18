"""Strict nonfinite/duplicate evidence guard regressions, no private data."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from scripts import accuracy_confirmation as c
from scripts import accuracy_route_confirmation as r

class FinalGateTests(unittest.TestCase):
    def test_duplicate_same_freeze_route_files_fail_population_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);output=root/"routes";output.mkdir();freeze=root/"freeze.json";freeze.write_text("{}");protocol=root/"protocol.json";ids=[str(i) for i in range(16)];protocol.write_text(json.dumps({"confirmation_trip_ids":ids}));paired=root/"hist_gradient_boosting.json";paired.write_text("{}");f={"decoder_selection":{"emission_family":"hist_gradient_boosting"}}
            for t in ids:(output/(t+".json")).write_text(json.dumps({"trip_id":"0","family":"hist_gradient_boosting","paired_fit_record_sha256":c.common.sha256_file(paired),"frozen_selection_sha256":c.common.sha256_file(freeze),"reference_loaded_after_both_predictions":True,"student_used":False,"evaluator_used":False,"records":{"baseline":{"configuration":"b","observations":1},"candidate":{"configuration":"b","observations":1}},"fulltrack_observations":1}))
            with patch.object(r,"OUTPUT",output),patch.object(c,"FREEZE",freeze),patch.object(c,"CONFIRMATION",root),patch.object(c,"verify_frozen",return_value=f),patch.object(r,"frozen_configurations",return_value={"baseline":{"name":"b"},"candidate":{"name":"b"}}),patch.object(r.common,"PROTOCOL_PATH",protocol):
                with self.assertRaisesRegex(ValueError,"population"):r.aggregate()
if __name__=="__main__":unittest.main()
