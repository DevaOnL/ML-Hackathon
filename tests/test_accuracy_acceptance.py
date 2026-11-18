"""Independent final gate rejects poisoned numerical evidence before inference."""
from pathlib import Path
import tempfile
import unittest
from scripts import accuracy_acceptance as a

class AccuracyAcceptanceTests(unittest.TestCase):
    def test_nonfinite_outofrange_bool_refused(self):
        for value in (float("nan"),float("inf"),-.001,1.001,True,".9"):
            with self.subTest(value=value),self.assertRaises(ValueError):a.finite_fraction(value)
    def test_fraction_endpoints_and_real_measurement_allowed(self):
        for value in (0,1,.923):self.assertEqual(a.finite_fraction(value),value)
    def test_nonstandard_json_nonfinite_tokens_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"fake.json"
            for token in ("NaN","Infinity","-Infinity"):
                p.write_text("{\"score\":"+token+"}")
                with self.assertRaisesRegex(ValueError,"nonfinite"):a.load(p)
if __name__=="__main__":unittest.main()
