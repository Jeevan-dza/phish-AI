"""
PhishGuard AI – Reports Routes
================================
GET /api/reports                      – List persisted investigation reports
GET /api/reports/{investigation_id}   – Retrieve a specific report
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, status

from backend.schemas.response import InvestigationReport, ReportsListResponse
from backend.services import report_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/reports",
    response_model=ReportsListResponse,
    summary="List investigation reports",
    description="Returns the most recent investigation reports (newest first).",
    tags=["Reports"],
)
async def list_reports(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of reports to return."),
) -> ReportsListResponse:
    reports = report_service.list_reports(limit=limit)
    return ReportsListResponse(total=len(reports), reports=reports)


@router.get(
    "/reports/{investigation_id}",
    response_model=InvestigationReport,
    summary="Retrieve a specific investigation report",
    description="Fetches a full investigation report by its unique ID.",
    tags=["Reports"],
)
async def get_report(investigation_id: str) -> InvestigationReport:
    report = report_service.load_report(investigation_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{investigation_id}' not found.",
        )
    return report
