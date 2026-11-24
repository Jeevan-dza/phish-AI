"""
PhishGuard AI – FastAPI Application Entry Point
================================================
Starts the FastAPI application, registers all routers, configures CORS,
and loads the ML model at startup.

Run the server:
    .venv\\Scripts\\python.exe -m uvicorn backend.main:app --reload --port 8000

Or using the helper at the project root:
    .venv\\Scripts\\python.exe backend/main.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that
#   from agents.url_agent.xxx import yyy   works correctly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import CORS_ORIGINS, LOG_LEVEL
from backend.routes import analyze, health, reports
from backend.services import agent_service

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("phishguard")

# ---------------------------------------------------------------------------
# Lifespan – load the ML model once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PhishGuard AI backend starting …")
    success = agent_service.load_model()
    if success:
        logger.info("URL analysis agent: Random Forest model loaded successfully.")
    else:
        logger.warning(
            "URL analysis agent: model could not be loaded. "
            "URL analysis will return 'unavailable'. "
            "Error: %s",
            agent_service.model_load_error(),
        )
    logger.info("PhishGuard AI backend ready. Docs: http://localhost:8000/docs")
    yield

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PhishGuard AI",
    description=(
        "An AI-Based Multi-Agent Phishing Detection and Attack Correlation System. "
        "Combines Random Forest URL analysis, domain registration intelligence, "
        "OpenPhish threat feeds, and a cross-agent evidence correlation engine."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS – allow the React development server
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global exception handler – never expose stack traces to the client
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An internal server error occurred. Please try again.",
        },
    )



# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------

app.include_router(health.router, prefix="/api")
app.include_router(analyze.router, prefix="/api")
app.include_router(reports.router, prefix="/api")

# ---------------------------------------------------------------------------
# Root redirect to docs
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    return JSONResponse(
        content={
            "message": "PhishGuard AI Backend is running.",
            "docs": "/docs",
            "health": "/api/health",
        }
    )

# ---------------------------------------------------------------------------
# Dev server launcher
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from backend.config import HOST, PORT, RELOAD
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=RELOAD)
