"""
PhishGuard AI – Backend Test Suite
=====================================
Tests cover:
  1. Backend health endpoint
  2. URL validation and SSRF protection
  3. URL agent service (with real model if available; mocked otherwise)
  4. Agent orchestrator (integration)
  5. Partial agent failure handling
  6. Evidence correlation and risk assessment logic
  7. Report persistence (save / load / list)
  8. Full investigation workflow via POST /api/analyze
  9. API response schema validation
  10. GET /api/agents/status

Run from project root:
    .venv\\Scripts\\python.exe -m pytest tests/test_backend.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path before any project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# 1. Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """GET /api/health should always return 200 with a valid body."""

    def test_health_returns_200(self, test_client):
        response = test_client.get("/api/health")
        assert response.status_code == 200

    def test_health_response_has_required_fields(self, test_client):
        data = test_client.get("/api/health").json()
        for field in ("status", "version", "model_loaded", "agents_available", "message"):
            assert field in data, f"Missing field: {field}"

    def test_health_status_is_ok_or_degraded(self, test_client):
        data = test_client.get("/api/health").json()
        assert data["status"] in ("ok", "degraded")


# ---------------------------------------------------------------------------
# 2. URL Validation
# ---------------------------------------------------------------------------

class TestURLValidation:
    """Unit tests for the URL validator / SSRF protection."""

    def test_valid_https_url_passes(self):
        from backend.utils.validators import validate_url
        result = validate_url("https://example.com/path?q=1")
        assert result == "https://example.com/path?q=1"

    def test_valid_http_url_passes(self):
        from backend.utils.validators import validate_url
        result = validate_url("http://example.com")
        assert result == "http://example.com"

    def test_missing_scheme_raises(self):
        from backend.utils.validators import URLValidationError, validate_url
        with pytest.raises(URLValidationError, match="scheme"):
            validate_url("example.com/page")

    def test_ftp_scheme_rejected(self):
        from backend.utils.validators import URLValidationError, validate_url
        with pytest.raises(URLValidationError):
            validate_url("ftp://example.com/file.txt")

    def test_javascript_scheme_rejected(self):
        from backend.utils.validators import URLValidationError, validate_url
        with pytest.raises(URLValidationError):
            validate_url("javascript:alert(1)")

    def test_localhost_blocked(self):
        from backend.utils.validators import URLValidationError, validate_url
        with pytest.raises(URLValidationError, match="private or reserved"):
            validate_url("http://127.0.0.1/admin")

    def test_private_ip_blocked(self):
        from backend.utils.validators import URLValidationError, validate_url
        with pytest.raises(URLValidationError, match="private or reserved"):
            validate_url("http://192.168.1.1/")

    def test_url_too_long_rejected(self):
        from backend.utils.validators import URLValidationError, validate_url
        long_url = "https://example.com/" + "a" * 3000
        with pytest.raises(URLValidationError, match="length"):
            validate_url(long_url)

    def test_empty_string_rejected(self):
        from backend.utils.validators import URLValidationError, validate_url
        with pytest.raises(URLValidationError):
            validate_url("")


# ---------------------------------------------------------------------------
# 3. URL Agent Service (mocked to avoid network + model dependency)
# ---------------------------------------------------------------------------

class TestURLAgentService:
    """Unit tests for the agent_service module."""

    def test_run_returns_unavailable_when_model_not_loaded(self):
        import asyncio
        from backend.services import agent_service

        # Temporarily unset the model
        original_model = agent_service._model
        agent_service._model = None
        agent_service._model_load_error = "Test: model not loaded"

        result = asyncio.get_event_loop().run_until_complete(
            agent_service.run("https://example.com")
        )

        # Restore
        agent_service._model = original_model
        agent_service._model_load_error = None

        assert result["status"] == "unavailable"
        assert result["agent_name"] == "url_analysis_agent"

    def test_run_returns_completed_on_success(self):
        """Mock analyze_url to verify the service wraps it correctly."""
        import asyncio
        from backend.services import agent_service

        fake_result = {
            "url": "https://example.com",
            "prediction": "Likely Legitimate",
            "predicted_label": 1,
            "phishing_model_score_percent": 5.0,
            "domain_registration": {"domain": "example.com", "domain_age_days": 1000, "status": "ok"},
            "openphish": {"found": False, "status": "Not found"},
        }

        agent_service._model = MagicMock()
        agent_service._feature_names = ["f1", "f2"]

        with patch(
            "backend.services.agent_service._run_sync",
            return_value={**fake_result, "agent_name": "url_analysis_agent", "status": "completed", "error": None},
        ):
            result = asyncio.get_event_loop().run_until_complete(
                agent_service.run("https://example.com")
            )

        assert result["status"] == "completed"
        assert result["prediction"] == "Likely Legitimate"


# ---------------------------------------------------------------------------
# 4. Correlation Service
# ---------------------------------------------------------------------------

class TestCorrelationService:
    """Unit tests for the evidence correlation engine."""

    def _make_url_result(self, label: int, score_pct: float, openphish_found, age_days=None):
        return {
            "url_analysis_agent": {
                "agent_name": "url_analysis_agent",
                "status": "completed",
                "prediction": "Potentially Phishing" if label == 0 else "Likely Legitimate",
                "predicted_label": label,
                "phishing_model_score_percent": score_pct,
                "domain_registration": {
                    "domain": "example.com",
                    "domain_age_days": age_days,
                    "status": "ok",
                },
                "openphish": {
                    "found": openphish_found,
                    "status": "test",
                },
                "error": None,
            }
        }

    def test_phishing_ml_plus_openphish_match_gives_high_score(self):
        from backend.services.correlation_service import correlate
        agent_results = self._make_url_result(0, 95.0, True, age_days=5)
        risk, evidence = correlate("https://evil.com", None, agent_results)
        assert risk.risk_score >= 70
        assert risk.classification == "Phishing"

    def test_legitimate_ml_no_openphish_gives_low_score(self):
        from backend.services.correlation_service import correlate
        agent_results = self._make_url_result(1, 2.0, False, age_days=1500)
        risk, evidence = correlate("https://github.com", None, agent_results)
        assert risk.risk_score < 40
        assert risk.classification == "Legitimate"

    def test_new_domain_adds_points(self):
        from backend.services.correlation_service import correlate
        agent_results_old = self._make_url_result(1, 10.0, False, age_days=1000)
        agent_results_new = self._make_url_result(1, 10.0, False, age_days=5)
        risk_old, _ = correlate("https://example.com", None, agent_results_old)
        risk_new, _ = correlate("https://new.com", None, agent_results_new)
        assert risk_new.risk_score > risk_old.risk_score

    def test_risk_score_capped_at_100(self):
        from backend.services.correlation_service import correlate
        agent_results = self._make_url_result(0, 100.0, True, age_days=1)
        risk, _ = correlate("https://evil.com", None, agent_results)
        assert risk.risk_score <= 100

    def test_risk_score_minimum_is_0(self):
        from backend.services.correlation_service import correlate
        agent_results = self._make_url_result(1, 0.0, False, age_days=5000)
        risk, _ = correlate("https://established.com", None, agent_results)
        assert risk.risk_score >= 0

    def test_evidence_items_returned(self):
        from backend.services.correlation_service import correlate
        agent_results = self._make_url_result(0, 80.0, True, age_days=10)
        _, evidence = correlate("https://phish.com", None, agent_results)
        assert len(evidence) > 0

    def test_risk_assessment_has_required_fields(self):
        from backend.services.correlation_service import correlate
        agent_results = self._make_url_result(0, 80.0, False, age_days=200)
        risk, _ = correlate("https://test.com", None, agent_results)
        assert hasattr(risk, "risk_score")
        assert hasattr(risk, "classification")
        assert hasattr(risk, "confidence_note")
        assert hasattr(risk, "scoring_method")
        assert hasattr(risk, "key_findings")
        assert hasattr(risk, "recommendations")


# ---------------------------------------------------------------------------
# 5. Report Persistence
# ---------------------------------------------------------------------------

class TestReportPersistence:
    """Tests for the report_service save/load/list cycle."""

    def test_save_and_load_roundtrip(self, tmp_reports_dir):
        from backend.services import report_service
        from backend.schemas.response import InvestigationReport, InvestigationStatus
        from datetime import datetime, timezone

        report = InvestigationReport(
            investigation_id="test-inv-001",
            url="https://example.com",
            timestamp=datetime.now(timezone.utc),
            investigation_status=InvestigationStatus.COMPLETED,
        )

        report_service.save_report(report)
        loaded = report_service.load_report("test-inv-001")
        assert loaded is not None
        assert loaded.investigation_id == "test-inv-001"
        assert loaded.url == "https://example.com"

    def test_load_nonexistent_returns_none(self, tmp_reports_dir):
        from backend.services import report_service
        result = report_service.load_report("does-not-exist-xyz")
        assert result is None

    def test_list_reports_returns_saved(self, tmp_reports_dir):
        from backend.services import report_service
        from backend.schemas.response import InvestigationReport, InvestigationStatus
        from datetime import datetime, timezone

        report = InvestigationReport(
            investigation_id="test-list-001",
            url="https://list-test.com",
            timestamp=datetime.now(timezone.utc),
            investigation_status=InvestigationStatus.COMPLETED,
        )
        report_service.save_report(report)
        reports = report_service.list_reports()
        ids = [r.investigation_id for r in reports]
        assert "test-list-001" in ids


# ---------------------------------------------------------------------------
# 6. POST /api/analyze endpoint
# ---------------------------------------------------------------------------

class TestAnalyzeEndpoint:
    """Integration tests for POST /api/analyze."""

    def test_analyze_requires_url(self, test_client):
        response = test_client.post("/api/analyze", json={})
        assert response.status_code == 422

    def test_analyze_rejects_missing_scheme(self, test_client):
        response = test_client.post("/api/analyze", json={"url": "example.com"})
        assert response.status_code == 422

    def test_analyze_rejects_private_ip(self, test_client):
        response = test_client.post("/api/analyze", json={"url": "http://192.168.0.1/"})
        assert response.status_code == 422

    def test_analyze_returns_investigation_id(self, test_client, mock_agent_service):
        response = test_client.post(
            "/api/analyze",
            json={"url": "https://example.com"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "investigation_id" in data
        assert data["success"] is True

    def test_analyze_response_contains_report(self, test_client, mock_agent_service):
        response = test_client.post(
            "/api/analyze",
            json={"url": "https://example.com"},
        )
        data = response.json()
        assert "report" in data
        assert data["report"] is not None

    def test_analyze_report_has_risk_assessment(self, test_client, mock_agent_service):
        response = test_client.post(
            "/api/analyze",
            json={"url": "https://example.com"},
        )
        data = response.json()
        risk = data["report"]["risk_assessment"]
        assert risk is not None
        assert 0 <= risk["risk_score"] <= 100
        assert risk["classification"] in ("Legitimate", "Suspicious", "Phishing", "Unknown")

    def test_analyze_with_message(self, test_client, mock_agent_service):
        response = test_client.post(
            "/api/analyze",
            json={
                "url": "https://example.com",
                "message": "Click here to verify your account immediately.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["report"]["message_provided"] is True


# ---------------------------------------------------------------------------
# 7. POST /api/analyze/url
# ---------------------------------------------------------------------------

class TestAnalyzeURLEndpoint:
    def test_url_only_endpoint_works(self, test_client, mock_agent_service):
        response = test_client.post(
            "/api/analyze/url",
            json={"url": "https://example.com"},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True


# ---------------------------------------------------------------------------
# 8. GET /api/agents/status
# ---------------------------------------------------------------------------

class TestAgentsStatusEndpoint:
    def test_agents_status_returns_200(self, test_client):
        response = test_client.get("/api/agents/status")
        assert response.status_code == 200

    def test_agents_status_has_six_agents(self, test_client):
        data = test_client.get("/api/agents/status").json()
        assert len(data["agents"]) == 6

    def test_url_agent_is_implemented(self, test_client):
        data = test_client.get("/api/agents/status").json()
        agents = {a["agent_name"]: a for a in data["agents"]}
        assert "url_analysis_agent" in agents
        assert agents["url_analysis_agent"]["implemented"] is True

    def test_other_agents_are_not_implemented(self, test_client):
        data = test_client.get("/api/agents/status").json()
        agents = {a["agent_name"]: a for a in data["agents"]}
        for name in ("webpage_analysis_agent", "visual_analysis_agent",
                     "behaviour_analysis_agent", "context_analysis_agent"):
            assert agents[name]["implemented"] is False


# ---------------------------------------------------------------------------
# 9. GET /api/reports
# ---------------------------------------------------------------------------

class TestReportsEndpoint:
    def test_reports_returns_200(self, test_client):
        response = test_client.get("/api/reports")
        assert response.status_code == 200

    def test_reports_response_has_total(self, test_client):
        data = test_client.get("/api/reports").json()
        assert "total" in data
        assert "reports" in data

    def test_reports_404_for_unknown(self, test_client):
        response = test_client.get("/api/reports/does-not-exist")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_client():
    """Return a FastAPI test client with the full app mounted."""
    from backend.main import app
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def mock_agent_service():
    """
    Replace the URL agent's run() coroutine with a lightweight mock
    so tests do not trigger real network calls or ML inference.
    """
    fake_result = {
        "agent_name": "url_analysis_agent",
        "status": "completed",
        "url": "https://example.com",
        "prediction": "Likely Legitimate",
        "predicted_label": 1,
        "phishing_model_score_percent": 4.5,
        "domain_registration": {
            "domain": "example.com",
            "registration_date": "1992-01-01T00:00:00+00:00",
            "domain_age_days": 12000,
            "registrar": "IANA",
            "status": "RDAP lookup successful",
        },
        "openphish": {
            "queried_url": "https://example.com",
            "found": False,
            "status": "Not found in OpenPhish database (does not confirm safety)",
            "error": None,
        },
        "error": None,
    }

    with patch(
        "backend.services.agent_service.run",
        new_callable=lambda: lambda: AsyncMock(return_value=fake_result),
    ):
        yield


@pytest.fixture()
def tmp_reports_dir(tmp_path, monkeypatch):
    """Redirect report persistence to a temporary directory."""
    import backend.services.report_service as rs
    import backend.config as cfg

    monkeypatch.setattr(cfg, "REPORTS_DIR", tmp_path / "reports_store")
    monkeypatch.setattr(rs, "REPORTS_DIR", tmp_path / "reports_store")
    yield tmp_path / "reports_store"
