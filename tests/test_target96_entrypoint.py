"""Only explicitly requested accepted route96 artifacts select the new runtime."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from backend import entrypoint as e

class Target96EntrypointTests(unittest.TestCase):
    def test_explicit_path_invokes_acceptance_guarded_factory_with_same_settings(self):
        s=SimpleNamespace(model_path=e.ROOT/'artifacts/target96/model.joblib')
        with patch.object(e.Settings,'from_env',return_value=s),patch('backend.target96_runtime.create_target96_app',return_value='accepted') as new,patch('backend.app.create_app',side_effect=AssertionError('no fallback')):
            self.assertEqual(e.create_configured_app(),'accepted');new.assert_called_once_with(s)
    def test_guard_failure_does_not_fallback(self):
        s=SimpleNamespace(model_path=e.ROOT/'artifacts/target96/model.joblib')
        with patch.object(e.Settings,'from_env',return_value=s),patch('backend.target96_runtime.create_target96_app',side_effect=ValueError('not accepted')),patch('backend.app.create_app',side_effect=AssertionError('no fallback')):
            with self.assertRaisesRegex(ValueError,'not accepted'):e.create_configured_app()
if __name__=='__main__':unittest.main()
