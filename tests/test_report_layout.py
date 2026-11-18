"""Synthetic content-profile checks; no private data, PDF rasterization or OCR."""
import re
import unittest

from scripts.verify_report_layout import content_text, required_topics


class ReportLayoutProfileTests(unittest.TestCase):
    def test_historical_profile_retains_old_acceptance_topics(self):
        topics = required_topics()
        self.assertIn("deferred five-model requirement", topics)
        self.assertIn("absent evaluator limitation", topics)
        self.assertNotIn("Gaussian Naive Bayes", topics)

    def test_final_profile_requires_both_added_models_and_actual_evidence(self):
        topics = required_topics("final-five")
        self.assertNotIn("deferred five-model requirement", topics)
        self.assertIn("Gaussian Naive Bayes", topics)
        self.assertIn("Histogram Gradient Boosting", topics)
        self.assertIn("geographical implementation map", topics)
        self.assertIn("historical RandomForest", topics)
        self.assertIn("historical ExtraTrees", topics)
        self.assertIn("human academic defence", topics)
        for name, pattern in topics.items():
            re.compile(pattern, re.IGNORECASE)

    def test_final_profile_matches_actual_model_name_variants(self):
        topics = required_topics("final-five")
        for value in ("Gaussian Naive Bayes", "GaussianNB"):
            self.assertRegex(value.lower(), topics["Gaussian Naive Bayes"])
        for value in ("HistGradientBoosting", "Histogram gradient boosting"):
            self.assertRegex(value.lower(), topics["Histogram Gradient Boosting"])

    def test_watermark_and_footer_cannot_mask_empty_page(self):
        text = "PathSnap Final Technical Report\nPage 13 of 63\nDRAFT - NOT FINAL\n"
        self.assertEqual(content_text(text), "")
        self.assertEqual(content_text("A real paragraph.\n" + text), "A real paragraph.")

    def test_unknown_profile_refused(self):
        with self.assertRaises(ValueError):
            required_topics("invented")


if __name__ == "__main__":
    unittest.main()
