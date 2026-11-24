"""
PhishGuard AI – Analysis Routes
=================================
POST /api/analyze         – Full multi-agent investigation
POST /api/analyze/url     – URL-only analysis (same pipeline, cleaner response)
GET  /api/analyze/{id}    – Retrieve an investigation by ID
GET  /api/agents/status   – Agent availability and implementation status
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from backend.schemas.request import AnalyzeRequest, URLOnlyRequest
from backend.schemas.response import (
    AgentStatusInfo,
    AgentsStatusResponse,
    AgentStatus,
    AnalyzeResponse,
    InvestigationStatus,
    InvestigationStatusResponse,
)
from backend.services import investigation_service
from backend.services.agent_orchestrator import AGENT_REGISTRY
from backend.utils.validators import URLValidationError, validate_url

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /api/analyze
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a full multi-agent phishing investigation",
    description=(
        "Accepts a URL and optional message text, runs all implemented agents "
        "concurrently, correlates the evidence, and returns the full investigation report."
    ),
    tags=["Analysis"],
)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    # Validate URL (SSRF protection + structural validation)
    try:
        validated_url = validate_url(request.url)
    except URLValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    try:
        report = await investigation_service.run_full_investigation(
            url=validated_url,
            message=request.message,
        )
        return AnalyzeResponse(
            success=True,
            investigation_id=report.investigation_id,
            report=report,
        )
    except Exception as exc:
        logger.exception("Investigation failed for URL %r", validated_url)
        return AnalyzeResponse(
            success=False,
            investigation_id="",
            error=f"Investigation failed: {exc}",
        )


# ---------------------------------------------------------------------------
# POST /api/analyze/url   (URL-only fast path)
# ---------------------------------------------------------------------------

@router.post(
    "/analyze/url",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a URL-only phishing investigation",
    description=(
        "Runs the full investigation pipeline on the supplied URL only. "
        "Identical to POST /api/analyze without an optional message."
    ),
    tags=["Analysis"],
)
async def analyze_url_only(request: URLOnlyRequest) -> AnalyzeResponse:
    try:
        validated_url = validate_url(request.url)
    except URLValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    try:
        report = await investigation_service.run_full_investigation(url=validated_url)
        return AnalyzeResponse(
            success=True,
            investigation_id=report.investigation_id,
            report=report,
        )
    except Exception as exc:
        logger.exception("URL-only investigation failed for %r", validated_url)
        return AnalyzeResponse(
            success=False,
            investigation_id="",
            error=f"Investigation failed: {exc}",
        )


# ---------------------------------------------------------------------------
# GET /api/analyze/{investigation_id}
# ---------------------------------------------------------------------------

@router.get(
    "/analyze/{investigation_id}",
    response_model=InvestigationStatusResponse,
    summary="Retrieve an investigation by ID",
    description="Fetches the status and results of a previously run investigation.",
    tags=["Analysis"],
)
async def get_investigation(investigation_id: str) -> InvestigationStatusResponse:
    response = investigation_service.get_investigation(investigation_id)
    if response.status == "not_found":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Investigation '{investigation_id}' not found.",
        )
    return response


# ---------------------------------------------------------------------------
# GET /api/agents/status
# ---------------------------------------------------------------------------

@router.get(
    "/agents/status",
    response_model=AgentsStatusResponse,
    summary="Get implementation status of all agents",
    description=(
        "Returns the name, description, and implementation status of all six agents "
        "so the frontend can display which agents are active."
    ),
    tags=["Agents"],
)
async def agents_status() -> AgentsStatusResponse:
    agent_infos = [
        AgentStatusInfo(
            agent_name=agent["id"],
            display_name=agent["display_name"],
            implemented=agent["implemented"],
            status=AgentStatus.COMPLETED if agent["implemented"] else AgentStatus.NOT_IMPLEMENTED,
            description=agent["description"],
        )
        for agent in AGENT_REGISTRY
    ]
    return AgentsStatusResponse(agents=agent_infos)
