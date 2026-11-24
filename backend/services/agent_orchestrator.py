"""
PhishGuard AI – Agent Orchestrator
=====================================
Central coordinator that:
  1. Accepts an investigation request.
  2. Runs the implemented URL analysis agent (and any future agents concurrently).
  3. Stubs out unimplemented agents with honest "not_implemented" markers.
  4. Collects all results and passes them to the correlation engine.
  5. Returns a fully assembled InvestigationReport.

Design rules enforced here:
  - Each agent runs in its own try/except; a single agent failure must NOT
    abort the investigation.
  - Stubs are never presented as real results.
  - asyncio.gather() is used for concurrent execution of independent agents.
  - The correlation engine runs AFTER all agent evidence is collected.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from backend.config import AGENT_TIMEOUT_SECONDS
from backend.schemas.response import (
    AgentStatus,
    InvestigationReport,
    InvestigationStatus,
)
from backend.services import agent_service, correlation_service, report_service
from backend.utils.helpers import generate_investigation_id, utc_now

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent registry – lists every agent and whether it is implemented.
# When teammates add new agents, register them here.
# ---------------------------------------------------------------------------

AGENT_REGISTRY: list[dict] = [
    {
        "id": "url_analysis_agent",
        "display_name": "URL Analysis Agent",
        "implemented": True,
        "description": "URL feature extraction, Random Forest ML prediction, RDAP domain registration, OpenPhish threat intelligence.",
    },
    {
        "id": "webpage_analysis_agent",
        "display_name": "Webpage Analysis Agent",
        "implemented": False,
        "description": "HTML/DOM analysis, login form detection, suspicious link analysis.",
    },
    {
        "id": "visual_analysis_agent",
        "display_name": "Visual Analysis Agent",
        "implemented": False,
        "description": "Screenshot analysis, brand impersonation detection, visual similarity.",
    },
    {
        "id": "behaviour_analysis_agent",
        "display_name": "Behaviour Analysis Agent",
        "implemented": False,
        "description": "Redirect behaviour, JavaScript analysis, external request tracking.",
    },
    {
        "id": "context_analysis_agent",
        "display_name": "Message & Context Analysis Agent",
        "implemented": False,
        "description": "Phishing language detection, urgency indicators, social engineering analysis.",
    },
    {
        "id": "correlation_engine",
        "display_name": "Evidence Correlation Engine",
        "implemented": True,
        "description": "Cross-agent evidence correlation and final risk assessment.",
    },
]


# ---------------------------------------------------------------------------
# Stub result for unimplemented agents
# ---------------------------------------------------------------------------

def _stub_result(agent_id: str, description: str) -> Dict[str, Any]:
    return {
        "agent_name": agent_id,
        "status": AgentStatus.NOT_IMPLEMENTED,
        "message": (
            f"Agent '{agent_id}' is not yet implemented. "
            "This field will be populated when the agent is integrated."
        ),
        "description": description,
        "findings": [],
        "error": None,
    }


# ---------------------------------------------------------------------------
# Individual agent runners (with per-agent timeout + error isolation)
# ---------------------------------------------------------------------------

async def _run_url_agent(url: str) -> Dict[str, Any]:
    try:
        return await asyncio.wait_for(
            agent_service.run(url),
            timeout=AGENT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("URL agent timed out for %r", url)
        return {
            "agent_name": "url_analysis_agent",
            "status": AgentStatus.FAILED,
            "error": f"Agent timed out after {AGENT_TIMEOUT_SECONDS} seconds.",
        }
    except Exception as exc:
        logger.exception("URL agent raised an unexpected error for %r", url)
        return {
            "agent_name": "url_analysis_agent",
            "status": AgentStatus.FAILED,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Main orchestration entry point
# ---------------------------------------------------------------------------

async def run_investigation(
    url: str,
    message: str | None = None,
) -> InvestigationReport:
    """
    Orchestrate the full PhishGuard AI investigation pipeline.

    Parameters
    ----------
    url:
        The URL to investigate (already validated by the route handler).
    message:
        Optional message text supplied by the user.

    Returns
    -------
    InvestigationReport
        A fully populated Pydantic report model.
    """
    investigation_id = generate_investigation_id(url)
    started_at: datetime = utc_now()

    logger.info(
        "Investigation %s started for URL: %s", investigation_id, url
    )

    # ------------------------------------------------------------------ #
    # Phase 1 – Run all INDEPENDENT implemented agents concurrently.      #
    # Currently only the URL agent is implemented; the others return stubs.#
    # ------------------------------------------------------------------ #

    # Gather implemented agents concurrently
    url_result_coro = _run_url_agent(url)

    # For agents not yet implemented, generate stubs (synchronously – no I/O)
    webpage_stub = _stub_result(
        "webpage_analysis_agent",
        "HTML/DOM analysis, login form detection, suspicious link analysis.",
    )
    visual_stub = _stub_result(
        "visual_analysis_agent",
        "Screenshot analysis, brand impersonation, visual similarity.",
    )
    behaviour_stub = _stub_result(
        "behaviour_analysis_agent",
        "Redirect behaviour, JavaScript analysis, external requests.",
    )
    context_stub = _stub_result(
        "context_analysis_agent",
        "Phishing language detection, urgency indicators, social engineering.",
    )

    # Await all concurrent real agents
    (url_result,) = await asyncio.gather(url_result_coro)

    agent_results: Dict[str, Any] = {
        "url_analysis_agent": url_result,
        "webpage_analysis_agent": webpage_stub,
        "visual_analysis_agent": visual_stub,
        "behaviour_analysis_agent": behaviour_stub,
        "context_analysis_agent": context_stub,
    }

    # Build execution summary for the frontend agent-activity panel
    execution_summary: Dict[str, str] = {
        name: result.get("status", AgentStatus.UNAVAILABLE)
        for name, result in agent_results.items()
    }
    execution_summary["correlation_engine"] = "pending"

    # ------------------------------------------------------------------ #
    # Phase 2 – Evidence Correlation & Risk Assessment                    #
    # ------------------------------------------------------------------ #

    try:
        risk_assessment, correlated_evidence = correlation_service.correlate(
            url=url,
            message=message,
            agent_results=agent_results,
        )
        execution_summary["correlation_engine"] = AgentStatus.COMPLETED
    except Exception as exc:
        logger.exception(
            "Correlation engine failed for investigation %s", investigation_id
        )
        risk_assessment = None
        correlated_evidence = []
        execution_summary["correlation_engine"] = AgentStatus.FAILED

    # ------------------------------------------------------------------ #
    # Phase 3 – Determine overall investigation status                    #
    # ------------------------------------------------------------------ #

    any_real_agent_succeeded = url_result.get("status") == AgentStatus.COMPLETED
    any_real_agent_failed = url_result.get("status") in (
        AgentStatus.FAILED, AgentStatus.UNAVAILABLE
    )

    if any_real_agent_succeeded:
        inv_status = InvestigationStatus.COMPLETED
    elif any_real_agent_failed:
        inv_status = InvestigationStatus.PARTIAL
    else:
        inv_status = InvestigationStatus.FAILED

    # ------------------------------------------------------------------ #
    # Phase 4 – Assemble and persist the report                          #
    # ------------------------------------------------------------------ #

    report = InvestigationReport(
        investigation_id=investigation_id,
        url=url,
        message_provided=bool(message),
        timestamp=started_at,
        investigation_status=inv_status,
        risk_assessment=risk_assessment,
        correlated_evidence=correlated_evidence,
        agent_results=agent_results,
        execution_summary=execution_summary,
    )

    # Persist the report (fire-and-forget; failures are logged, not raised)
    try:
        report_service.save_report(report)
    except Exception as exc:
        logger.warning(
            "Could not persist report %s: %s", investigation_id, exc
        )

    logger.info(
        "Investigation %s completed. Status: %s. Risk score: %s.",
        investigation_id,
        inv_status,
        risk_assessment.risk_score if risk_assessment else "N/A",
    )

    return report
