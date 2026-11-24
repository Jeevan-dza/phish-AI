"""
PhishGuard AI – Report Persistence Service
===========================================
Provides simple local file-system persistence for investigation reports.

Storage format: one JSON file per investigation, stored in
  backend/reports_store/{investigation_id}.json

This module is intentionally simple. The persistence layer is fully
isolated behind this interface so that migrating to Firebase, PostgreSQL,
or any other backend later requires changes to only this file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from backend.config import REPORTS_DIR
from backend.schemas.response import InvestigationReport

logger = logging.getLogger(__name__)


def _ensure_store() -> Path:
    """Create the reports directory if it does not exist."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def save_report(report: InvestigationReport) -> None:
    """
    Persist an investigation report to disk as JSON.
    Silently logs and returns if saving fails (non-critical path).
    """
    store = _ensure_store()
    path = store / f"{report.investigation_id}.json"
    try:
        path.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.debug("Report saved: %s", path)
    except Exception as exc:
        logger.warning("Failed to save report %s: %s", report.investigation_id, exc)


def load_report(investigation_id: str) -> Optional[InvestigationReport]:
    """
    Load a single report by investigation ID.
    Returns None if the report does not exist or cannot be parsed.
    """
    store = _ensure_store()
    path = store / f"{investigation_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return InvestigationReport.model_validate(data)
    except Exception as exc:
        logger.warning("Failed to load report %s: %s", investigation_id, exc)
        return None


def list_reports(limit: int = 50) -> List[InvestigationReport]:
    """
    Return up to *limit* most-recent reports, sorted newest first.
    Reports that fail to parse are skipped (and logged).
    """
    store = _ensure_store()
    files = sorted(store.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    reports: List[InvestigationReport] = []
    for path in files[:limit]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            reports.append(InvestigationReport.model_validate(data))
        except Exception as exc:
            logger.warning("Skipping unreadable report file %s: %s", path.name, exc)
    return reports


def delete_report(investigation_id: str) -> bool:
    """Delete a report file. Returns True if deleted, False if not found."""
    store = _ensure_store()
    path = store / f"{investigation_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False
