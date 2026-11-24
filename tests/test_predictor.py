"""
Tests for agents.url_agent.predictor and feature_extractor apex normalization.
=============================================================================
Verifies:
1. Apex domain normalization prevents false alarms on legitimate domains (github.com).
2. Phishing URLs remain classified as potentially phishing.
3. predictor.py functions correctly when called programmatically.
"""

import unittest
from pathlib import Path
import joblib

from agents.url_agent.feature_extractor import extract_features
from agents.url_agent.predictor import MODEL_PATH, predict_url


class TestFeatureExtractorApexNormalization(unittest.TestCase):
    """Verify apex domain normalization aligns with training distribution."""

    def test_apex_domain_normalized_to_www(self):
        feats = extract_features("https://github.com")
        self.assertEqual(feats["subdomain_count"], 1)
        self.assertEqual(feats["dot_count"], 2)

    def test_already_www_domain_unchanged(self):
        feats = extract_features("https://www.github.com")
        self.assertEqual(feats["subdomain_count"], 1)
        self.assertEqual(feats["dot_count"], 2)

    def test_ip_address_not_normalized(self):
        feats = extract_features("http://192.168.1.1/admin")
        self.assertEqual(feats["has_ip_address"], 1)
        # IP has 4 dot-separated octets, yielding max(4-2, 0) = 2 in legacy extractor
        self.assertEqual(feats["subdomain_count"], 2)
        # Ensure it was not falsely prepended with 'www.'
        self.assertEqual(feats["dot_count"], 3)

    def test_subdomain_url_retains_subdomain(self):
        feats = extract_features("http://paypal.verify-account.com/login")
        self.assertEqual(feats["subdomain_count"], 1)
        self.assertGreaterEqual(feats["suspicious_word_count"], 2)


class TestPredictorWithModel(unittest.TestCase):
    """End-to-end inference tests with the trained model."""

    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists():
            raise unittest.SkipTest("Trained model not found")
        bundle = joblib.load(MODEL_PATH)
        cls.model = bundle["model"]
        cls.feature_names = bundle["feature_names"]

    def test_github_classified_legitimate(self):
        res = predict_url("https://github.com", self.model, self.feature_names)
        self.assertEqual(res["prediction"], "Likely Legitimate")
        self.assertEqual(res["predicted_label"], 1)
        self.assertLess(res["phishing_model_score_percent"], 5.0)

    def test_google_classified_legitimate(self):
        res = predict_url("https://google.com", self.model, self.feature_names)
        self.assertEqual(res["prediction"], "Likely Legitimate")
        self.assertEqual(res["predicted_label"], 1)

    def test_phishing_url_classified_phishing(self):
        res = predict_url(
            "http://paypal-secure-login.verify-account.com/confirm",
            self.model,
            self.feature_names,
        )
        self.assertEqual(res["prediction"], "Potentially Phishing")
        self.assertEqual(res["predicted_label"], 0)
        self.assertGreater(res["phishing_model_score_percent"], 90.0)


if __name__ == "__main__":
    unittest.main()
