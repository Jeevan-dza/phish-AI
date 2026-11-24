
from pathlib import Path

import joblib
import pandas as pd
from agents.url_agent.feature_extractor import extract_features
from agents.url_agent.openphish_client import check_url as openphish_check
from agents.url_agent.risk_scorer import compute_combined_risk, risk_label
from agents.url_agent.whois_analyzer import analyze_domain


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "url_random_forest.joblib"

PHISHING_LABEL = 0
LEGITIMATE_LABEL = 1


def analyze_url(url: str, model, feature_names: list) -> dict:
    """Analyze URL patterns, domain registration, and threat intelligence."""

    # ------------------------------------------------------------------
    # 1. Random Forest model prediction (unchanged)
    # ------------------------------------------------------------------
    features = pd.DataFrame([extract_features(url)])
    features = features.reindex(columns=feature_names)

    prediction = model.predict(features)[0]
    classes = list(model.classes_)
    probabilities = model.predict_proba(features)[0]

    phishing_score = (
        float(probabilities[classes.index(PHISHING_LABEL)]) * 100
        if PHISHING_LABEL in classes
        else None
    )

    if prediction == PHISHING_LABEL:
        prediction_text = "Potentially Phishing"
    elif prediction == LEGITIMATE_LABEL:
        prediction_text = "Likely Legitimate"
    else:
        prediction_text = f"Unknown label: {prediction}"

    # ------------------------------------------------------------------
    # 2. RDAP domain registration analysis (unchanged)
    # ------------------------------------------------------------------
    domain_info = analyze_domain(url)

    # ------------------------------------------------------------------
    # 3. OpenPhish threat intelligence lookup
    #    NOTE: The API result is informational only. It does NOT change
    #    the Random Forest prediction above.
    # ------------------------------------------------------------------
    openphish_result = openphish_check(url)

    # ------------------------------------------------------------------
    # 4. Combined risk scoring (URL + RDAP + OpenPhish → one score)
    # ------------------------------------------------------------------
    url_result = {
        "predicted_label": int(prediction),
        "phishing_model_score_percent": phishing_score,
        "features": features.iloc[0].to_dict(),
    }
    combined_risk = compute_combined_risk(url_result, domain_info, openphish_result)

    return {
        "url": url,
        # --- Model signals ---
        "prediction": prediction_text,
        "predicted_label": prediction,
        "phishing_model_score_percent": phishing_score,
        # --- RDAP signals ---
        "domain_registration": domain_info,
        # --- Threat intelligence ---
        "openphish": openphish_result,
        # --- Combined risk assessment ---
        "combined_risk": combined_risk,
    }


def print_report(report: dict) -> None:
    """Print the combined investigation report."""

    print("\n" + "=" * 60)
    print("         URL INVESTIGATION REPORT")
    print("=" * 60)

    print(f"URL: {report['url']}")

    # ------------------------------------------------------------------
    # Section 1: Random Forest Model
    # ------------------------------------------------------------------
    print("\n--- RANDOM FOREST MODEL ---")
    print(f"Prediction:          {report['prediction']}")

    score = report["phishing_model_score_percent"]
    if score is not None:
        print(f"Phishing Score:      {score:.2f}%")

    # ------------------------------------------------------------------
    # Section 2: Domain Registration
    # ------------------------------------------------------------------
    print("\n--- DOMAIN REGISTRATION ---")

    domain_info = report.get("domain_registration") or {}
    domain_age = domain_info.get("domain_age_days")
    if domain_age is None and "domain_age" in domain_info:
        domain_age = domain_info.get("domain_age")

    fields = [
        ("Domain", domain_info.get("domain")),
        ("Registration Date", domain_info.get("registration_date")),
        ("Domain Age", domain_age),
        ("Registrar", domain_info.get("registrar")),
        ("Status", domain_info.get("status")),
    ]
    for label, value in fields:
        print(f"{label}: {value}")

    # ------------------------------------------------------------------
    # Section 3: Threat Intelligence
    # ------------------------------------------------------------------
    print("\n--- THREAT INTELLIGENCE ---")

    op = report.get("openphish", {})
    found = op.get("found")

    if found is True:
        print("OpenPhish Match: YES - URL found in OpenPhish phishing database")
    elif found is False:
        print("OpenPhish Match: NO  - URL not currently listed in OpenPhish")
    else:
        print("OpenPhish Match: UNAVAILABLE")

    print(f"Status:          {op.get('status', 'Unavailable')}")

    # ------------------------------------------------------------------
    # Section 4: Combined Risk Assessment
    # ------------------------------------------------------------------
    cr = report.get("combined_risk")
    if cr is not None:
        # CombinedRiskScore dataclass – access as attributes
        # Fields: risk_score, classification, components, all_findings,
        #         recommendations, scoring_explanation, evidence_coverage
        print("\n--- COMBINED RISK ASSESSMENT ---")
        print(f"Final Score:         {cr.risk_score}/100  ({risk_label(cr.risk_score)})")
        print(f"Classification:      {cr.classification.upper()}")

        # Component breakdown (components is a list of ComponentResult dataclasses)
        if cr.components:
            print("\n  Component Breakdown:")
            total_w = sum(c.weight_assigned for c in cr.components)
            for comp in cr.components:
                eff_pct = (comp.weight_assigned / total_w * 100) if total_w > 0 else 0
                avail = comp.availability in {"available", "partial"}
                if avail:
                    avail_flag = "(partial)" if comp.availability == "partial" else ""
                    print(
                        f"    {comp.name:<36} "
                        f"evidence={comp.evidence_score:>3.0f}/100  "
                        f"weight={eff_pct:.0f}% {avail_flag}"
                    )
                else:
                    print(f"    {comp.name:<36} UNAVAILABLE")

        # Key findings
        if cr.all_findings:
            print("\n  Key Findings:")
            for finding in cr.all_findings[:6]:   # cap at 6 for readability
                print(f"    • {finding}")

        # Recommendations
        if cr.recommendations:
            print("\n  Recommendations:")
            for rec in cr.recommendations[:3]:
                print(f"    → {rec}")

        # Evidence coverage note
        if cr.evidence_coverage:
            print(f"\n  Coverage: {cr.evidence_coverage}")

    print("=" * 60 + "\n")


def main():
    if not MODEL_PATH.exists():
        print("Trained model not found.")
        print("Run: python agents/url_agent/train_model_v2.py")
        return

    print("Loading Random Forest model...")
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    feature_names = bundle["feature_names"]

    version = bundle.get("version", "v1")
    print(f"Model loaded (version: {version}, features: {len(feature_names)})")
    print("URL Investigator is ready.")
    print("Enter 'quit' to exit.\n")

    while True:
        url = input("Enter a URL to investigate: ").strip()

        if url.lower() in {"quit", "exit", "q"}:
            print("Investigation finished.")
            break

        if not url:
            print("Please enter a URL.\n")
            continue

        try:
            report = analyze_url(url, model, feature_names)
            print_report(report)
        except Exception as exc:
            print(f"Investigation failed: {exc}\n")


if __name__ == "__main__":
    main()
