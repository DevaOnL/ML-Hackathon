"""Opt-in runtime refuses artifacts and acceptance mismatches before inference."""
from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch
from backend import accuracy_runtime as r

class AccuracyRuntimeTests(unittest.TestCase):
    def test_invalid_file_allowlist_rejected_without_loading_model(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"release_manifest.json";p.write_text(json.dumps({"schema":"pathsnap-accuracy-export-v1","files":{"../x":"a"*64}}))
            with self.assertRaisesRegex(ValueError,"allowlist"):r.release_components(p)
    def test_bad_artifact_hash_rejected_before_pickle(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);names=("model.joblib","match_config.json","transition_policy.json","model_metadata.json")
            for n in names:(root/n).write_text("notanartifact")
            p=root/"release_manifest.json";p.write_text(json.dumps({"schema":"pathsnap-accuracy-export-v1","files":{n:"a"*64 for n in names}}))
            with self.assertRaisesRegex(ValueError,"hash differs"):r.release_components(p)
    def test_acceptance_required_before_runtime_loading(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/"release_manifest.json";p.write_text("{}")
            (root/"acceptance.json").write_text(json.dumps({"accepted":False}))
            with patch.object(r,"release_components",side_effect=AssertionError("must not load")):
                with self.assertRaisesRegex(ValueError,"not accepted"):r.create_accuracy_app(settings=object(),manifest_path=p)
if __name__=="__main__":unittest.main()
