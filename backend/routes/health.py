"""
PhishGuard AI – Health Route
==============================
GET /api/health
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.response import HealthResponse
from backend.services import agent_service
from backend.services.agent_orchestrator import AGENT_REGISTRY

router = APIRouter()

_VERSION = "1.0.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Backend health check",
    description=(
        "Returns the operational status of the PhishGuard AI backend, "
        "including whether the ML model is loaded and how many agents are available."
    ),
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    model_loaded = agent_service.is_model_loaded()
    implemented = [a for a in AGENT_REGISTRY if a["implemented"]]
    total_agents = len([a for a in AGENT_REGISTRY if a["id"] != "correlation_engine"])

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        version=_VERSION,
        model_loaded=model_loaded,
        agents_available=total_agents,
        agents_implemented=len([a for a in implemented if a["id"] != "correlation_engine"]),
        message=(
            "PhishGuard AI backend is running. All core services are operational."
            if model_loaded
            else (
                "PhishGuard AI backend is running but the ML model could not be loaded. "
                f"Model error: {agent_service.model_load_error()}"
            )
        ),
    )
