"""
PhishGuard AI – Multi-Category URL Evaluation Suite
===================================================
Tests the improved URL Analysis Agent across all 12 required categories:
  1. Popular legitimate websites
  2. Educational websites (.edu)
  3. Government websites (.gov)
  4. Banking websites (with paths)
  5. E-commerce websites (with IDs/queries)
  6. Legitimate login pages
  7. Legitimate URLs with numbers, hyphens, and long queries
  8. Known / synthetic phishing URLs
  9. Phishing websites using HTTPS
 10. Brand impersonation URLs (brand in subdomain)
 11. Newly registered domains (< 30 days)
 12. URLs with missing / unavailable RDAP data
"""

import sys
from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.url_agent.feature_extractor import extract_features
from agents.url_agent.risk_scorer import compute_combined_risk, risk_label

TEST_CASES = [
    # (URL, Expected Class, Category Description, Mock RDAP Age, Mock OpenPhish)
    # 1. Popular legitimate websites
    ("https://www.google.com", "Legitimate", "Popular legitimate homepage", 9500, False),
    ("https://github.com", "Legitimate", "Popular developer platform", 6200, False),
    ("https://en.wikipedia.org/wiki/Phishing", "Legitimate", "Popular content site with path", 7500, False),

    # 2. Educational websites
    ("https://www.stanford.edu/academics/", "Legitimate", "Educational (.edu) portal", 11000, False),
    ("https://mit.edu/research", "Legitimate", "Educational (.edu) research", 11500, False),

    # 3. Government websites
    ("https://www.usa.gov/benefits", "Legitimate", "Government (.gov) portal", 8500, False),
    ("https://cisa.gov/cybersecurity", "Legitimate", "Government (.gov) cybersecurity", 3000, False),

    # 4. Banking websites
    ("https://www.chase.com/personal/banking", "Legitimate", "Legitimate banking with paths", 9000, False),
    ("https://www.wellsfargo.com/checking-and-savings/savings/", "Legitimate", "Legitimate banking multi-segment", 9500, False),

    # 5. E-commerce websites
    ("https://www.amazon.com/dp/B09X7MVJNH?ref=ppx_yo2ov", "Legitimate", "E-commerce deep path with query", 10000, False),
    ("https://www.ebay.com/itm/1234567890?hash=item1", "Legitimate", "E-commerce with item ID and query", 9800, False),

    # 6. Legitimate login pages
    ("https://accounts.google.com/signin", "Legitimate", "Google account login page", 9500, False),
    ("https://login.microsoftonline.com/common/oauth2/v2.0/authorize", "Legitimate", "Microsoft login/OAuth page", 8000, False),

    # 7. Legitimate URLs with numbers, hyphens, and long queries
    ("https://support.apple.com/en-us/HT213604", "Legitimate", "Apple support with article ID and hyphens", 9800, False),
    ("https://aws.amazon.com/blogs/aws/category/artificial-intelligence/?blog-posts-cards.sort-by=item.additionalFields.createdDate", "Legitimate", "Long query params and hyphens", 10000, False),

    # 8. Known / synthetic phishing URLs
    ("http://paypal-secure-login.verify-account.com/confirm", "Phishing", "Phishing: fake verification domain", 15, False),
    ("http://amazon-update-billing.tk/account/login", "Phishing", "Phishing: Freenom TLD + fake brand", 5, False),

    # 9. Phishing websites using HTTPS
    ("https://secure-login-chase-update.com/signin/index.php", "Phishing", "Phishing: HTTPS with spoofed bank brand", 12, False),
    ("https://verify-apple-id-suspended.top/auth/login", "Phishing", "Phishing: HTTPS on suspicious .top TLD", 8, False),

    # 10. Brand impersonation URLs
    ("http://paypal.evil-domain.com/webscr/login", "Phishing", "Brand in subdomain (paypal.evil-domain)", 20, False),
    ("http://microsoft.account-security-alert.xyz/verify", "Phishing", "Brand in subdomain + .xyz", 10, False),

    # 11. Newly registered / suspicious domains
    ("http://login-update-security-alert-2026.cf/verify", "Phishing", "Suspicious TLD .cf with urgency", 3, False),

    # 12. URLs with missing / unavailable RDAP data & OpenPhish match
    ("http://192.168.1.1/admin/login.php", "Suspicious", "Raw IP address URL (no RDAP domain)", None, False),
    ("http://confirmed-phish-sample.net/login", "Phishing", "Threat Feed Match: OpenPhish confirmed match", 25, True),
]


def run_suite():
    model_path = PROJECT_ROOT / "models" / "url_random_forest.joblib"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    print("=" * 85)
    print("        PHISHGUARD AI - COMPREHENSIVE URL EVALUATION SUITE")
    print("=" * 85)
    print(f"Model version: {bundle.get('version', 'unknown')} | Features: {len(feature_names)}")
    print(f"Total test cases: {len(TEST_CASES)}\n")

    results = []

    for idx, (url, expected, desc, rdap_age, op_match) in enumerate(TEST_CASES, 1):
        # 1. Feature extraction & RF prediction
        feats = pd.DataFrame([extract_features(url)]).reindex(columns=feature_names, fill_value=0)
        pred = model.predict(feats)[0]
        classes = list(model.classes_)
        rf_phish_pct = float(model.predict_proba(feats)[0][classes.index(0)]) * 100

        # 2. Combined risk scoring
        url_res = {
            "predicted_label": int(pred),
            "phishing_model_score_percent": rf_phish_pct,
            "features": feats.iloc[0].to_dict(),
        }
        dom_res = {
            "domain_age_days": rdap_age,
            "registration_date": "2020-01-01T00:00:00Z" if rdap_age else None,
            "status": "Active" if rdap_age else "Unavailable",
        }
        op_res = {
            "found": op_match,
            "status": "Matched" if op_match else ("Not Found" if op_match is False else "Unavailable"),
            "error": None,
        }

        cr = compute_combined_risk(url_res, dom_res, op_res)

        # 3. Status check
        actual = cr.classification
        score = cr.risk_score

        if expected == "Legitimate":
            passed = actual in {"Legitimate"} or (actual == "Suspicious" and score <= 45)
        elif expected == "Phishing":
            passed = actual in {"Phishing"} or (actual == "Suspicious" and score >= 45)
        elif expected == "Suspicious":
            passed = actual in {"Suspicious", "Phishing"}
        else:
            passed = actual == expected

        status = "PASS" if passed else "FAIL"

        results.append({
            "idx": idx,
            "url": url,
            "expected": expected,
            "actual": actual,
            "score": score,
            "rf_pct": rf_phish_pct,
            "rf_pred": "Phish" if pred == 0 else "Legit",
            "desc": desc,
            "status": status,
        })

    # Summary table
    print(f"{'#':<3} {'Category / Description':<38} {'Expected':<11} {'Actual':<11} {'Score':<7} {'RF %':<7} {'Status'}")
    print("-" * 88)
    pass_count = 0
    for r in results:
        desc_short = r['desc'][:36]
        rf_str = f"{r['rf_pct']:.1f}%"
        print(f"{r['idx']:<3} {desc_short:<38} {r['expected']:<11} {r['actual']:<11} {r['score']:<7} {rf_str:<7} {r['status']}")
        if r['status'] == "PASS":
            pass_count += 1

    print("=" * 88)
    print(f"OVERALL RESULT: {pass_count}/{len(results)} PASSED ({pass_count/len(results)*100:.1f}%)\n")

    # Detailed spot checks
    print("--- Detailed Highlights ---")
    print(f"  • accounts.google.com/signin       : RF={results[11]['rf_pct']:.1f}%, Final Score={results[11]['score']}/100 -> {results[11]['actual']}")
    print(f"  • login.microsoftonline.com/...    : RF={results[12]['rf_pct']:.1f}%, Final Score={results[12]['score']}/100 -> {results[12]['actual']}")
    print(f"  • chase.com/personal/banking       : RF={results[7]['rf_pct']:.1f}%, Final Score={results[7]['score']}/100 -> {results[7]['actual']}")
    print(f"  • amazon.com/dp/B09X7MVJNH?...     : RF={results[9]['rf_pct']:.1f}%, Final Score={results[9]['score']}/100 -> {results[9]['actual']}")
    print(f"  • paypal.evil-domain.com/...       : RF={results[19]['rf_pct']:.1f}%, Final Score={results[19]['score']}/100 -> {results[19]['actual']}")
    print(f"  • confirmed-phish-sample (OpenPhish): RF={results[23]['rf_pct']:.1f}%, Final Score={results[23]['score']}/100 -> {results[23]['actual']}")


if __name__ == "__main__":
    run_suite()
