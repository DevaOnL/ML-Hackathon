"""No alternate service: same existing entrypoint opts into accepted release only."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from backend import entrypoint as e

class AccuracyEntrypointTests(unittest.TestCase):
    def test_original_release_uses_unchanged_factory(self):
        settings=SimpleNamespace(model_path=e.ROOT/"artifacts/final_five/model.joblib")
        with patch.object(e.Settings,"from_env",return_value=settings),patch("backend.app.create_app",return_value="v1") as old,patch("backend.accuracy_runtime.create_accuracy_app",side_effect=AssertionError("no optin")):
            self.assertEqual(e.create_configured_app(),"v1");old.assert_called_once_with()
    def test_only_explicit_accuracy_path_uses_acceptance_guarded_factory(self):
        settings=SimpleNamespace(model_path=e.ROOT/"artifacts/accuracy_improvement/model.joblib")
        with patch.object(e.Settings,"from_env",return_value=settings),patch("backend.app.create_app",side_effect=AssertionError("no fallback")),patch("backend.accuracy_runtime.create_accuracy_app",return_value="v2") as new:
            self.assertEqual(e.create_configured_app(),"v2");new.assert_called_once_with()
if __name__=="__main__":unittest.main()
