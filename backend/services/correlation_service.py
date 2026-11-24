"""
PhishGuard AI – Evidence Correlation Service
==============================================
Collects evidence from all available agent results, identifies cross-agent
relationships, and produces:
  - A composite RiskAssessment (score 0–100, classification, key findings)
  - A list of CorrelatedEvidence items

Scoring methodology (documented and transparent):
  ----------------------------------------------------------------
  IMPORTANT CAVEATS
  - The Random Forest model's phishing_model_score_percent is NOT a
    calibrated probability. It reflects the proportion of trees that
    voted "phishing" in the ensemble. It may be overconfident.
  - The composite risk score produced here has NOT been validated
    against an independent, labelled hold-out dataset. It should be
    treated as a heuristic indicator, not a ground-truth probability.
  - Domain age alone is not proof of phishing. Very new domains are a
    risk signal, not a verdict.
  - An OpenPhish "not found" result does NOT mean the URL is safe.
  ----------------------------------------------------------------

  Scoring approach:
  1. If the ML model predicts "Phishing" → base contribution of up to 60 pts
     scaled by the model's (uncalibrated) phishing score percentage.
  2. If the domain is newly registered (< 30 days) → up to +15 pts.
  3. If the domain is very young (30–180 days) → up to +5 pts.
  4. If OpenPhish reports a match → +25 pts (hard evidence; capped at 100).
  5. Caps and floors are applied: score is always in [0, 100].
  6. Classification thresholds: Phishing ≥ 70, Suspicious 40–69, Legitimate < 40.

  Related signals (ML prediction + OpenPhish match) are intentionally NOT
  simply summed – the OpenPhish match is weighted independently to avoid
  double-counting correlated observations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.schemas.response import (
    AgentStatus,
    CorrelatedEvidence,
    RiskAssessment,
    RiskClassification,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights (documented here, not buried in magic numbers)
# ---------------------------------------------------------------------------

# Maximum contribution of the ML model vote (scaled by uncalibrated score)
_ML_PHISHING_MAX_PTS = 60

# Domain age thresholds (days) and their point contributions
_DOMAIN_NEW_THRESHOLD_DAYS = 30     # very new domain
_DOMAIN_YOUNG_THRESHOLD_DAYS = 180  # moderately young domain
_DOMAIN_NEW_PTS = 15
_DOMAIN_YOUNG_PTS = 5

# OpenPhish hard-match contribution
_OPENPHISH_MATCH_PTS = 25

# Classification thresholds
_PHISHING_THRESHOLD = 70
_SUSPICIOUS_THRESHOLD = 40


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def correlate(
    url: str,
    message: Optional[str],
    agent_results: Dict[str, Any],
) -> Tuple[RiskAssessment, List[CorrelatedEvidence]]:
    """
    Correlate evidence from all agents and return a RiskAssessment.

    Parameters
    ----------
    url:        The investigated URL (used for logging).
    message:    Optional message text (reserved for context agent).
    agent_results:  Dict of agent_id → agent result dict.

    Returns
    -------
    Tuple of (RiskAssessment, list[CorrelatedEvidence])
    """

    evidence_items: List[CorrelatedEvidence] = []
    score: float = 0.0
    key_findings: List[str] = []
    recommendations: List[str] = []

    url_result = agent_results.get("url_analysis_agent", {})

    # ------------------------------------------------------------------ #
    # Signal 1 – Random Forest ML model                                   #
    # ------------------------------------------------------------------ #

    ml_score_pct: Optional[float] = url_result.get("phishing_model_score_percent")
    ml_prediction: Optional[str] = url_result.get("prediction")
    ml_label: Optional[int] = url_result.get("predicted_label")

    if url_result.get("status") == AgentStatus.COMPLETED and ml_prediction is not None:
        if ml_label == 0:  # Phishing prediction
            # Scale contribution: max _ML_PHISHING_MAX_PTS at 100% model score
            if ml_score_pct is not None:
                ml_contribution = (ml_score_pct / 100.0) * _ML_PHISHING_MAX_PTS
            else:
                ml_contribution = _ML_PHISHING_MAX_PTS * 0.5  # assume moderate
            score += ml_contribution

            key_findings.append(
                f"Random Forest model classified the URL as potentially phishing "
                f"(uncalibrated model score: {ml_score_pct:.1f}% phishing)."
                if ml_score_pct is not None else
                "Random Forest model classified the URL as potentially phishing."
            )
            recommendations.append(
                "Do not enter credentials or personal information on this page."
            )
            evidence_items.append(
                CorrelatedEvidence(
                    source_agents=["url_analysis_agent"],
                    finding=(
                        f"ML model prediction: Phishing "
                        f"(uncalibrated score {ml_score_pct:.1f}%)"
                        if ml_score_pct is not None else
                        "ML model prediction: Phishing"
                    ),
                    severity="high",
                    weight=round(ml_contribution / 100.0, 3),
                )
            )
        else:  # Legitimate prediction
            key_findings.append(
                f"Random Forest model classified the URL as likely legitimate "
                f"(phishing model score: {ml_score_pct:.1f}%)."
                if ml_score_pct is not None else
                "Random Forest model classified the URL as likely legitimate."
            )
            evidence_items.append(
                CorrelatedEvidence(
                    source_agents=["url_analysis_agent"],
                    finding=(
                        f"ML model prediction: Legitimate "
                        f"(phishing score {ml_score_pct:.1f}%)"
                        if ml_score_pct is not None else
                        "ML model prediction: Legitimate"
                    ),
                    severity="info",
                    weight=0.0,
                )
            )
    elif url_result.get("status") in (AgentStatus.FAILED, AgentStatus.UNAVAILABLE):
        key_findings.append(
            "URL analysis agent was unavailable; ML prediction could not be obtained."
        )

    # ------------------------------------------------------------------ #
    # Signal 2 – Domain age (RDAP)                                        #
    # ------------------------------------------------------------------ #

    domain_info: Dict[str, Any] = url_result.get("domain_registration") or {}
    domain_age_days: Optional[int] = domain_info.get("domain_age_days")

    if domain_age_days is not None:
        if domain_age_days < _DOMAIN_NEW_THRESHOLD_DAYS:
            score += _DOMAIN_NEW_PTS
            finding = (
                f"Domain is very newly registered ({domain_age_days} days old), "
                "which is a common characteristic of phishing infrastructure."
            )
            key_findings.append(finding)
            recommendations.append(
                "Exercise extreme caution with newly registered domains."
            )
            evidence_items.append(
                CorrelatedEvidence(
                    source_agents=["url_analysis_agent"],
                    finding=finding,
                    severity="high",
                    weight=round(_DOMAIN_NEW_PTS / 100.0, 3),
                )
            )
        elif domain_age_days < _DOMAIN_YOUNG_THRESHOLD_DAYS:
            score += _DOMAIN_YOUNG_PTS
            finding = (
                f"Domain is relatively young ({domain_age_days} days old)."
            )
            key_findings.append(finding)
            evidence_items.append(
                CorrelatedEvidence(
                    source_agents=["url_analysis_agent"],
                    finding=finding,
                    severity="medium",
                    weight=round(_DOMAIN_YOUNG_PTS / 100.0, 3),
                )
            )
        else:
            key_findings.append(
                f"Domain has been registered for {domain_age_days} days "
                "(established domain)."
            )

    # ------------------------------------------------------------------ #
    # Signal 3 – OpenPhish threat intelligence                            #
    # ------------------------------------------------------------------ #

    openphish: Dict[str, Any] = url_result.get("openphish") or {}
    openphish_found: Optional[bool] = openphish.get("found")

    if openphish_found is True:
        score += _OPENPHISH_MATCH_PTS
        finding = "URL is listed in the OpenPhish phishing database (active threat feed match)."
        key_findings.append(finding)
        recommendations.append(
            "This URL has been confirmed as a phishing URL by OpenPhish. Avoid it completely."
        )
        evidence_items.append(
            CorrelatedEvidence(
                source_agents=["url_analysis_agent"],
                finding=finding,
                severity="high",
                weight=round(_OPENPHISH_MATCH_PTS / 100.0, 3),
            )
        )
    elif openphish_found is False:
        evidence_items.append(
            CorrelatedEvidence(
                source_agents=["url_analysis_agent"],
                finding=(
                    "URL is NOT currently listed in OpenPhish. "
                    "Note: absence from the database does NOT confirm the URL is safe."
                ),
                severity="info",
                weight=0.0,
            )
        )
    else:
        key_findings.append(
            "OpenPhish threat intelligence was unavailable for this investigation."
        )

    # ------------------------------------------------------------------ #
    # Cross-agent correlation: ML phishing + OpenPhish match              #
    # ------------------------------------------------------------------ #

    if ml_label == 0 and openphish_found is True:
        # Both signals agree – note the convergence but do NOT double-add pts
        evidence_items.append(
            CorrelatedEvidence(
                source_agents=["url_analysis_agent"],
                finding=(
                    "CORRELATED: Both the ML model and OpenPhish independently "
                    "identify this URL as phishing. This convergence significantly "
                    "increases confidence in the phishing assessment."
                ),
                severity="high",
                weight=0.5,
            )
        )
        recommendations.append(
            "Report this URL to your organisation's security team immediately."
        )

    # ------------------------------------------------------------------ #
    # Clamp score and classify                                            #
    # ------------------------------------------------------------------ #

    final_score = max(0, min(100, int(round(score))))

    if final_score >= _PHISHING_THRESHOLD:
        classification = RiskClassification.PHISHING
    elif final_score >= _SUSPICIOUS_THRESHOLD:
        classification = RiskClassification.SUSPICIOUS
    else:
        classification = RiskClassification.LEGITIMATE

    # Always add general recommendations
    if not recommendations:
        recommendations.append(
            "Continue to apply good security hygiene when visiting unfamiliar URLs."
        )

    if classification in (RiskClassification.PHISHING, RiskClassification.SUSPICIOUS):
        recommendations.append(
            "Do not provide personal information, login credentials, or payment details on this page."
        )

    risk_assessment = RiskAssessment(
        risk_score=final_score,
        classification=classification,
        confidence_note=(
            "This risk score is a heuristic composite derived from an uncalibrated "
            "Random Forest model, RDAP domain age data, and the OpenPhish threat feed. "
            "It has NOT been validated against an independent labelled dataset. "
            "Treat it as a decision-support tool, not a ground-truth verdict."
        ),
        scoring_method=(
            f"Composite score = ML model contribution (max {_ML_PHISHING_MAX_PTS} pts, "
            f"scaled by uncalibrated model score) "
            f"+ domain age bonus (new <{_DOMAIN_NEW_THRESHOLD_DAYS}d: +{_DOMAIN_NEW_PTS}pts, "
            f"young <{_DOMAIN_YOUNG_THRESHOLD_DAYS}d: +{_DOMAIN_YOUNG_PTS}pts) "
            f"+ OpenPhish match (+{_OPENPHISH_MATCH_PTS}pts). "
            f"Related signals are NOT double-counted. "
            f"Classification: Phishing ≥{_PHISHING_THRESHOLD}, "
            f"Suspicious ≥{_SUSPICIOUS_THRESHOLD}, Legitimate <{_SUSPICIOUS_THRESHOLD}."
        ),
        key_findings=key_findings,
        recommendations=list(dict.fromkeys(recommendations)),  # deduplicate, preserve order
    )

    return risk_assessment, evidence_items
