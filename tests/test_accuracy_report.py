"""Build the addendum on synthetic numbers only to catch layout code errors."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pypdf import PdfReader
from scripts import build_accuracy_report as r

class AccuracyReportTests(unittest.TestCase):
    def test_synthetic_pdf_has_all_six_sections_and_readable_text(self):
        family={"family":"hist_gradient_boosting","selected_run":"synthetic_context_full","feature_profile":"context","confirmation_baseline_top1":.88,"confirmation_candidate_top1":.90,"confirmation_delta_top1":.02,"confirmation_baseline_binary_macro_f1":.93,"confirmation_candidate_binary_macro_f1":.94,"confirmation_baseline_binary_log_loss":.06,"confirmation_candidate_binary_log_loss":.05,"tuning_baseline_top1":.88,"tuning_candidate_top1":.89,"paired_trip_bootstrap_interval":[.001,.03],"improved_trips":12,"decreased_trips":3}
        metrics={k:.96 for k in ("length_f1","edge_f1","proxy_edge_accuracy","proxy_edge_macro_f1","route_continuity")};metrics["elapsed_seconds"]=2.
        data={"frozen_selection_sha256":"a"*64,"families":[{**family,"family":f} for f in ("logistic_regression","knn_distance_weighted","bagging","gaussian_nb","hist_gradient_boosting")],"route_confirmation":{"baseline":metrics,"candidate":metrics},"student_development_comparison":{"baseline":metrics,"candidate":metrics},"test_results":{"python_tests":6,"hosting_tests":2,"node_tests":2,"total_tests":10,"static_smoke":"pass"},"release_decision":{"accepted":False,"reason":"synthetic_fixture_only"},"limitations":["Synthetic fixture, not measured evidence."]}
        with tempfile.TemporaryDirectory() as td:
            with patch.object(r,"OUTPUT",Path(td)):p=r.build_pdf(data);r.build_markdown(data)
            doc=PdfReader(p);text="\n".join(page.extract_text() or "" for page in doc.pages)
            self.assertGreaterEqual(len(doc.pages),6);self.assertIn("Genuine learned-model",text);self.assertIn("paired internal classifier",text.lower());self.assertIn("not a new blind test",text);self.assertNotIn("\u25a0",text)
if __name__=="__main__":unittest.main()
