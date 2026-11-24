"""
PhishGuard AI – Combined Risk Scorer
======================================
Combines three detection components into one final risk score (0–100):

  Component A – URL Analysis (Random Forest ML prediction)        weight: 50%
  Component B – WHOIS/RDAP Domain Registration Analysis           weight: 20%
  Component C – OpenPhish Threat Intelligence Feed                weight: 30%

Scoring methodology (documented and transparent):
─────────────────────────────────────────────────
  1. Each component contributes an EVIDENCE SCORE (0–100) independently.
  2. Available components are weighted and summed to produce the composite.
  3. If a component is unavailable, its weight is redistributed proportionally
     across available components (renormalization).
  4. The final score is clamped to [0, 100].
  5. Classification thresholds (adjustable):
       Phishing:    combined_score ≥ 65
       Suspicious:  combined_score ≥ 30
       Legitimate:  combined_score <  30

Important limitations:
  - The Random Forest phishing_model_score_percent is NOT a calibrated
    probability.  It is used as an INDICATOR, not a probability.
  - The domain age signal is a risk indicator, NOT proof of phishing.
  - An OpenPhish match is strong evidence from an external threat feed.
    It does NOT constitute independent verification of malicious behaviour.
  - The combined score has NOT been validated against an independent
    labelled dataset.  Treat it as a decision-support tool.

Risk level display (approximate – NOT hard-coded):
  0–9:   Very low risk
  10–29: Low risk
  30–49: Some concerns (Suspicious)
  50–64: Elevated risk (Suspicious)
  65–84: High risk (Phishing)
  85–99: Very high risk (Phishing)
  100:   Confirmed phishing evidence (OpenPhish match + ML + young domain)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Default component weights (must sum to 1.0)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "url_analysis": 0.50,
    "domain_registration": 0.20,
    "openphish": 0.30,
}

# Classification thresholds
PHISHING_THRESHOLD = 65
SUSPICIOUS_THRESHOLD = 30

# Domain age thresholds
_DOMAIN_NEW_DAYS = 30       # very new → high risk signal
_DOMAIN_YOUNG_DAYS = 180    # young → moderate risk signal
_DOMAIN_MATURE_DAYS = 365   # 1 year+  → low risk from age alone


# ---------------------------------------------------------------------------
# Component scoring functions
# ---------------------------------------------------------------------------

def score_url_analysis(
    predicted_label: Optional[int],
    phishing_score_pct: Optional[float],
    features: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, str, List[str]]:
    """
    Compute URL Analysis component evidence score.

    Combines:
      1. Random Forest prediction & phishing probability
      2. Suspicious URL patterns & lexical indicators

    Returns (evidence_score 0–100, raw_component_weight, label, findings).

    Scoring rationale:
      - If model predicts PHISHING: scale evidence score by the uncalibrated
        model probability.
      - If model predicts LEGITIMATE: low base score (never 0 — URL analysis
        was performed and returned a result).
      - Suspicious patterns (brand impersonation in subdomain, suspicious TLD,
        IP in hostname, '@' symbol, double slash) elevate the evidence score.
      - Max contribution from URL Analysis alone is capped at 90 to prevent a
        single component from producing a 100 final score.
      - Unavailable: return neutral score 50 and zero weight.
    """
    findings: List[str] = []

    if predicted_label is None or phishing_score_pct is None:
        findings.append(
            "URL analysis result unavailable; component excluded from scoring."
        )
        return 50.0, 0.0, "unavailable", findings

    if predicted_label == 0:  # phishing prediction
        # Map [50%, 100%] model score → [30, 80] evidence score
        pct = max(0.0, min(100.0, phishing_score_pct))
        evidence_score = 30.0 + (pct / 100.0) * 50.0  # 30..80
        findings.append(
            f"Random Forest model: Potentially Phishing "
            f"(uncalibrated phishing score {pct:.1f}%)"
        )
        if pct >= 90:
            findings.append(
                "High model confidence in phishing classification "
                "(note: score is not a calibrated probability)."
            )
    else:  # legitimate prediction
        pct = max(0.0, min(100.0, phishing_score_pct))
        # Even with a legitimate prediction, a moderate phishing score is a mild signal
        evidence_score = max(2.0, pct * 0.25)   # 0..25 from residual probability
        findings.append(
            f"Random Forest model: Likely Legitimate "
            f"(phishing score {pct:.1f}%)"
        )

    # ── Suspicious URL pattern checks ─────────────────────────────────
    if features and isinstance(features, dict):
        if features.get("brand_in_subdomain"):
            evidence_score = max(evidence_score + 35.0, 65.0)
            findings.append(
                "Suspicious URL pattern: Well-known brand token detected in a subdomain "
                "of an unrelated domain (possible brand impersonation / spoofing)."
            )
        if features.get("has_suspicious_tld"):
            evidence_score = max(evidence_score + 20.0, 50.0)
            findings.append(
                "Suspicious URL pattern: High-abuse top-level domain detected."
            )
        if features.get("has_ip_address"):
            evidence_score = max(evidence_score + 30.0, 60.0)
            findings.append(
                "Suspicious URL pattern: Host is a raw IP address rather than a domain name."
            )
        if features.get("has_at_symbol"):
            evidence_score = max(evidence_score + 25.0, 55.0)
            findings.append(
                "Suspicious URL pattern: '@' symbol in URL can obscure the actual destination."
            )
        if features.get("has_double_slash_in_path"):
            evidence_score = max(evidence_score + 15.0, 45.0)
            findings.append(
                "Suspicious URL pattern: Double slash '//' in path (redirect evasion pattern)."
            )

    # Cap URL Analysis component at 90 (URL alone cannot confirm 100% phishing)
    evidence_score = min(90.0, evidence_score)

    return round(evidence_score, 1), DEFAULT_WEIGHTS["url_analysis"], "available", findings


def score_domain_registration(
    domain_age_days: Optional[int],
    registration_date: Optional[str],
    rdap_status: Optional[str],
) -> Tuple[float, float, str, List[str]]:
    """
    Compute Domain Registration component evidence score.

    Returns (evidence_score 0–100, weight, label, findings).

    Scoring rationale:
      - Very new domain (< 30 days): 70 pts — fresh domains are strongly
        associated with phishing infrastructure.
      - Young domain (30–180 days): 40 pts.
      - Moderate age (180–365 days): 20 pts.
      - Established (> 365 days): 5 pts — age alone does not prove safety.
      - Unavailable: return 35 (slight concern, not neutral) and reduced weight.

    Domain age is NOT sufficient to classify phishing alone.  It is an
    INPUT to the combined score, not a standalone verdict.
    """
    findings: List[str] = []

    # RDAP lookup failed or domain not found
    rdap_ok = rdap_status and "successful" in rdap_status.lower()
    rdap_no_record = rdap_status and "no rdap" in rdap_status.lower()

    if domain_age_days is None:
        if rdap_no_record:
            findings.append(
                "No RDAP registration record found. Some privacy-protected or "
                "country-code domains may not appear in RDAP."
            )
            return 35.0, DEFAULT_WEIGHTS["domain_registration"] * 0.5, "partial", findings
        elif rdap_status and "timed out" in rdap_status.lower():
            findings.append("RDAP lookup timed out; domain age could not be determined.")
            return 35.0, DEFAULT_WEIGHTS["domain_registration"] * 0.3, "partial", findings
        else:
            findings.append(
                f"Domain registration data unavailable "
                f"(RDAP status: {rdap_status or 'unknown'}). "
                "Component weight reduced."
            )
            return 35.0, DEFAULT_WEIGHTS["domain_registration"] * 0.3, "partial", findings

    if domain_age_days < _DOMAIN_NEW_DAYS:
        evidence_score = 70.0
        findings.append(
            f"Domain is very newly registered ({domain_age_days} days old). "
            f"Domains registered within {_DOMAIN_NEW_DAYS} days are "
            "commonly used for phishing campaigns."
        )
    elif domain_age_days < _DOMAIN_YOUNG_DAYS:
        evidence_score = 40.0
        findings.append(
            f"Domain is relatively young ({domain_age_days} days old). "
            "Young domains carry elevated phishing risk."
        )
    elif domain_age_days < _DOMAIN_MATURE_DAYS:
        evidence_score = 20.0
        findings.append(
            f"Domain is moderately established ({domain_age_days} days old)."
        )
    else:
        evidence_score = 5.0
        years = domain_age_days // 365
        findings.append(
            f"Domain is well-established (~{years} year(s) old, {domain_age_days} days). "
            "Long-registered domains are less commonly used for phishing, "
            "but age alone does not guarantee safety."
        )

    if registration_date:
        findings.append(f"Registration date: {registration_date[:10]}.")

    return round(evidence_score, 1), DEFAULT_WEIGHTS["domain_registration"], "available", findings


def score_openphish(
    found: Optional[bool],
    openphish_status: Optional[str],
    error: Optional[str],
) -> Tuple[float, float, str, List[str]]:
    """
    Compute OpenPhish Threat Intelligence component evidence score.

    Returns (evidence_score 0–100, weight, label, findings).

    Scoring rationale:
      - MATCH (found=True): 95 pts — a live threat-feed match is strong
        evidence.  NOT treated as 100 because feed matches can be stale
        or contain false positives.
      - NO MATCH (found=False): 5 pts — absence from the feed does NOT
        confirm safety.  Minimum score (not 0) because the check was
        performed and returned a usable answer.
      - UNAVAILABLE: 30 pts neutral and reduced weight.
    """
    findings: List[str] = []

    if found is True:
        evidence_score = 95.0
        findings.append(
            "CONFIRMED: URL is currently listed in the OpenPhish active phishing database."
        )
        findings.append(
            "OpenPhish is an external threat feed. This match is strong evidence "
            "of phishing activity but is distinct from independent behavioural verification."
        )
    elif found is False:
        evidence_score = 5.0
        findings.append(
            "URL is NOT currently listed in OpenPhish. "
            "Absence from the database does NOT confirm the URL is safe — "
            "phishing sites often operate before being indexed."
        )
    else:
        # Unavailable
        err_msg = error or openphish_status or "unknown error"
        findings.append(
            f"OpenPhish threat intelligence is unavailable ({err_msg}). "
            "This component is excluded from the combined score."
        )
        return 30.0, 0.0, "unavailable", findings

    return round(evidence_score, 1), DEFAULT_WEIGHTS["openphish"], "available", findings


# ---------------------------------------------------------------------------
# Combined Risk Score
# ---------------------------------------------------------------------------

@dataclass
class ComponentResult:
    name: str
    evidence_score: float
    weight_assigned: float
    availability: str          # "available" | "partial" | "unavailable"
    findings: List[str] = field(default_factory=list)


@dataclass
class CombinedRiskScore:
    # Final outputs
    risk_score: int                      # 0–100
    classification: str                  # "Legitimate" | "Suspicious" | "Phishing"
    # Components
    components: List[ComponentResult] = field(default_factory=list)
    # Evidence
    all_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    # Transparency
    scoring_explanation: str = ""
    evidence_coverage: str = ""
    confidence_note: str = (
        "This score is a heuristic composite that has NOT been validated "
        "against an independent labelled dataset. Use it as decision support, "
        "not as a definitive verdict."
    )


def compute_combined_risk(
    url_result: Dict[str, Any],
    domain_result: Dict[str, Any],
    openphish_result: Dict[str, Any],
) -> CombinedRiskScore:
    """
    Compute the combined risk score from all three detection components.

    Parameters
    ----------
    url_result:
        Output of analyze_url() — contains prediction, predicted_label,
        phishing_model_score_percent.
    domain_result:
        Output of analyze_domain() — contains domain_age_days,
        registration_date, status.
    openphish_result:
        Output of check_url() — contains found (bool|None), status, error.

    Returns
    -------
    CombinedRiskScore dataclass with all scoring details.
    """

    components: List[ComponentResult] = []
    all_findings: List[str] = []
    recommendations: List[str] = []

    # ── Component A: URL Analysis ─────────────────────────────────────
    ml_label = url_result.get("predicted_label")
    ml_pct = url_result.get("phishing_model_score_percent")
    ml_feats = url_result.get("features")
    url_evidence, url_weight, url_avail, url_findings = score_url_analysis(
        ml_label, ml_pct, ml_feats
    )
    components.append(ComponentResult(
        name="URL Analysis (Random Forest)",
        evidence_score=url_evidence,
        weight_assigned=url_weight,
        availability=url_avail,
        findings=url_findings,
    ))
    all_findings.extend(url_findings)

    # ── Component B: Domain Registration ──────────────────────────────
    domain_age = domain_result.get("domain_age_days")
    reg_date = domain_result.get("registration_date")
    rdap_status = domain_result.get("status")
    dom_evidence, dom_weight, dom_avail, dom_findings = score_domain_registration(
        domain_age, reg_date, rdap_status
    )
    components.append(ComponentResult(
        name="Domain Registration (RDAP)",
        evidence_score=dom_evidence,
        weight_assigned=dom_weight,
        availability=dom_avail,
        findings=dom_findings,
    ))
    all_findings.extend(dom_findings)

    # ── Component C: OpenPhish ─────────────────────────────────────────
    op_found = openphish_result.get("found")
    op_status = openphish_result.get("status")
    op_error = openphish_result.get("error")
    op_evidence, op_weight, op_avail, op_findings = score_openphish(
        op_found, op_status, op_error
    )
    components.append(ComponentResult(
        name="Threat Intelligence (OpenPhish)",
        evidence_score=op_evidence,
        weight_assigned=op_weight,
        availability=op_avail,
        findings=op_findings,
    ))
    all_findings.extend(op_findings)

    # ── Weight renormalization ─────────────────────────────────────────
    total_weight = url_weight + dom_weight + op_weight
    if total_weight <= 0.0:
        # All components unavailable – default to 50 (unknown)
        final_score = 50
    else:
        weighted_sum = (
            url_evidence * url_weight
            + dom_evidence * dom_weight
            + op_evidence * op_weight
        )
        final_score_raw = weighted_sum / total_weight
        final_score = max(0, min(100, int(round(final_score_raw))))

    # ── Coverage summary ──────────────────────────────────────────────
    avail_names = [c.name.split("(")[0].strip() for c in components if c.availability == "available"]
    partial_names = [c.name.split("(")[0].strip() for c in components if c.availability == "partial"]
    unavail_names = [c.name.split("(")[0].strip() for c in components if c.availability == "unavailable"]

    if unavail_names or partial_names:
        coverage_parts = []
        if avail_names:
            coverage_parts.append(f"Full data: {', '.join(avail_names)}")
        if partial_names:
            coverage_parts.append(f"Partial data: {', '.join(partial_names)}")
        if unavail_names:
            coverage_parts.append(f"Unavailable: {', '.join(unavail_names)}")
        evidence_coverage = "Incomplete — " + "; ".join(coverage_parts) + "."
    else:
        evidence_coverage = "Complete — all three detection components contributed."

    # ── Classification ─────────────────────────────────────────────────
    if final_score >= PHISHING_THRESHOLD:
        classification = "Phishing"
        recommendations.append(
            "Do not enter credentials, personal information, or payment details on this page."
        )
        recommendations.append(
            "Report this URL to your organisation's security team or use "
            "Google Safe Browsing report at safebrowsing.google.com/safebrowsing/report_phish/."
        )
    elif final_score >= SUSPICIOUS_THRESHOLD:
        classification = "Suspicious"
        recommendations.append(
            "Proceed with caution. Verify the URL directly by navigating to the "
            "organisation's official website rather than clicking this link."
        )
        recommendations.append(
            "Do not enter sensitive information if you have any doubt."
        )
    else:
        classification = "Legitimate"
        recommendations.append(
            "URL appears low risk based on available signals. Continue to apply "
            "good security hygiene when entering credentials or personal information."
        )

    # OpenPhish override: if confirmed in feed AND score is Suspicious, upgrade to Phishing
    if op_found is True and classification == "Suspicious":
        classification = "Phishing"
        recommendations.insert(
            0,
            "OpenPhish database match overrides Suspicious classification → Phishing."
        )

    # ── Scoring explanation ────────────────────────────────────────────
    comp_lines = []
    for c in components:
        comp_lines.append(
            f"  • {c.name}: evidence score {c.evidence_score:.0f}/100 "
            f"(weight {c.weight_assigned:.2f}, status: {c.availability})"
        )
    renorm_note = (
        f"Weights renormalized from {total_weight:.2f} (some components unavailable)."
        if abs(total_weight - 1.0) > 0.01 else
        "All component weights fully applied."
    )
    scoring_explanation = (
        f"Combined risk score = weighted average of three detection components.\n"
        + "\n".join(comp_lines) + "\n"
        f"  {renorm_note}\n"
        f"  Thresholds: Phishing ≥ {PHISHING_THRESHOLD}, "
        f"Suspicious ≥ {SUSPICIOUS_THRESHOLD}, Legitimate < {SUSPICIOUS_THRESHOLD}."
    )

    return CombinedRiskScore(
        risk_score=final_score,
        classification=classification,
        components=components,
        all_findings=all_findings,
        recommendations=list(dict.fromkeys(recommendations)),
        scoring_explanation=scoring_explanation,
        evidence_coverage=evidence_coverage,
    )


# ---------------------------------------------------------------------------
# Human-readable risk label
# ---------------------------------------------------------------------------

def risk_label(score: int) -> str:
    if score <= 9:
        return "Very Low Risk"
    elif score <= 29:
        return "Low Risk"
    elif score <= 49:
        return "Some Concerns"
    elif score <= 64:
        return "Elevated Risk"
    elif score <= 84:
        return "High Risk"
    elif score <= 99:
        return "Very High Risk"
    else:
        return "Confirmed Phishing Evidence"
