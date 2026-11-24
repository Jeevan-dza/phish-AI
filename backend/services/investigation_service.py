"""
PhishGuard AI – Investigation Service
=======================================
High-level service layer consumed by route handlers.
Keeps route handlers thin by delegating all business logic here.
"""

from __future__ import annotations

import logging

from backend.schemas.response import InvestigationReport, InvestigationStatusResponse
from backend.services import agent_orchestrator, report_service

logger = logging.getLogger(__name__)


async def run_full_investigation(
    url: str,
    message: str | None = None,
) -> InvestigationReport:
    """
    Run the full multi-agent investigation pipeline and return a report.
    Delegates entirely to the orchestrator; exists to give route handlers
    a single, stable import point.
    """
    return await agent_orchestrator.run_investigation(url=url, message=message)


def get_investigation(investigation_id: str) -> InvestigationStatusResponse:
    """
    Retrieve a previously run investigation by its ID.
    Returns a status response with the report if available.
    """
    report = report_service.load_report(investigation_id)
    if report is None:
        return InvestigationStatusResponse(
            investigation_id=investigation_id,
            status="not_found",
            report=None,
        )
    return InvestigationStatusResponse(
        investigation_id=investigation_id,
        status=report.investigation_status,
        report=report,
    )
