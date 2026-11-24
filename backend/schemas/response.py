"""
PhishGuard AI – Response Schemas
==================================
Pydantic v2 models for all API responses.
Every response the React frontend receives is typed here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations (plain strings – avoids enum import overhead on older Pydantic)
# ---------------------------------------------------------------------------

class InvestigationStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"   # some agents failed but report was still generated


class RiskClassification:
    LEGITIMATE = "Legitimate"
    SUSPICIOUS = "Suspicious"
    PHISHING = "Phishing"
    UNKNOWN = "Unknown"


class AgentStatus:
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_IMPLEMENTED = "not_implemented"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Domain Registration sub-model
# ---------------------------------------------------------------------------

class DomainRegistrationResult(BaseModel):
    domain: Optional[str] = None
    registration_date: Optional[str] = None
    domain_age_days: Optional[int] = None
    registrar: Optional[str] = None
    status: Optional[str] = None


# ---------------------------------------------------------------------------
# OpenPhish sub-model
# ---------------------------------------------------------------------------

class OpenPhishResult(BaseModel):
    queried_url: Optional[str] = None
    found: Optional[bool] = None
    status: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# URL Agent result
# ---------------------------------------------------------------------------

class URLAgentResult(BaseModel):
    agent_name: str = "url_analysis_agent"
    status: str = AgentStatus.COMPLETED
    prediction: Optional[str] = None
    predicted_label: Optional[int] = None
    phishing_model_score_percent: Optional[float] = None
    domain_registration: Optional[DomainRegistrationResult] = None
    openphish: Optional[OpenPhishResult] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Generic stub result for agents not yet implemented
# ---------------------------------------------------------------------------

class AgentStubResult(BaseModel):
    agent_name: str
    status: str = AgentStatus.NOT_IMPLEMENTED
    message: str = "This agent is not yet implemented."
    findings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Correlated evidence item
# ---------------------------------------------------------------------------

class CorrelatedEvidence(BaseModel):
    source_agents: List[str]
    finding: str
    severity: str                      # "high" | "medium" | "low" | "info"
    weight: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------

class RiskAssessment(BaseModel):
    risk_score: int = Field(ge=0, le=100, description="Composite risk score 0–100.")
    classification: str = Field(
        description="Overall classification: Legitimate, Suspicious, or Phishing."
    )
    confidence_note: str = Field(
        description="Honest note about confidence / calibration status."
    )
    scoring_method: str = Field(
        description="Human-readable explanation of how the score was derived."
    )
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Full investigation report
# ---------------------------------------------------------------------------

class InvestigationReport(BaseModel):
    investigation_id: str
    url: str
    message_provided: bool = False
    timestamp: datetime
    investigation_status: str
    risk_assessment: Optional[RiskAssessment] = None
    correlated_evidence: List[CorrelatedEvidence] = Field(default_factory=list)
    agent_results: Dict[str, Any] = Field(default_factory=dict)
    execution_summary: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of agent_name → status for the frontend agent-activity view.",
    )
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# API response envelopes
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    """Returned by POST /api/analyze and POST /api/analyze/url."""
    success: bool
    investigation_id: str
    report: Optional[InvestigationReport] = None
    error: Optional[str] = None


class InvestigationStatusResponse(BaseModel):
    """Returned by GET /api/analyze/{investigation_id}."""
    investigation_id: str
    status: str
    report: Optional[InvestigationReport] = None


class ReportsListResponse(BaseModel):
    """Returned by GET /api/reports."""
    total: int
    reports: List[InvestigationReport]


class AgentStatusInfo(BaseModel):
    agent_name: str
    display_name: str
    implemented: bool
    status: str
    description: str


class AgentsStatusResponse(BaseModel):
    """Returned by GET /api/agents/status."""
    agents: List[AgentStatusInfo]


class HealthResponse(BaseModel):
    """Returned by GET /api/health."""
    status: str
    version: str
    model_loaded: bool
    agents_available: int
    agents_implemented: int
    message: str
