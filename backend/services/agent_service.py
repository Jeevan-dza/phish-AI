"""
PhishGuard AI – URL Agent Service
===================================
Wraps the existing url_investigator.analyze_url() function for use inside
the backend orchestrator.

Design principles:
  - Load the ML model ONCE at module import and cache it.
  - Expose a single async-compatible run() coroutine so the orchestrator
    can await it alongside other concurrent tasks.
  - Never raise unhandled exceptions to the orchestrator; always return a
    structured result dict with an "error" key on failure.
  - Do NOT modify the agent source files in agents/url_agent/.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

_model: Optional[Any] = None
_feature_names: Optional[list] = None
_model_load_error: Optional[str] = None

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODEL_PATH = _PROJECT_ROOT / "models" / "url_random_forest.joblib"


def load_model() -> bool:
    """
    Load the Random Forest model bundle from disk into module-level cache.
    Returns True on success, False on failure.
    Called once at application startup.
    """
    global _model, _feature_names, _model_load_error

    if not _MODEL_PATH.exists():
        _model_load_error = f"Model file not found at: {_MODEL_PATH}"
        logger.error("URL agent: %s", _model_load_error)
        return False

    try:
        import joblib  # Already installed in the venv
        bundle = joblib.load(_MODEL_PATH)
        _model = bundle["model"]
        _feature_names = bundle["feature_names"]
        logger.info("URL agent: Random Forest model loaded from %s", _MODEL_PATH)
        return True
    except Exception as exc:
        _model_load_error = f"Failed to load model: {exc}"
        logger.error("URL agent: %s", _model_load_error)
        return False


def is_model_loaded() -> bool:
    return _model is not None and _feature_names is not None


def model_load_error() -> Optional[str]:
    return _model_load_error


# ---------------------------------------------------------------------------
# Public coroutine used by the orchestrator
# ---------------------------------------------------------------------------

async def run(url: str) -> Dict[str, Any]:
    """
    Run the URL analysis agent asynchronously.

    Runs the synchronous analyze_url() call in the default thread executor
    so it does not block the FastAPI event loop.

    Returns a dict that always contains:
      - agent_name  : str
      - status      : "completed" | "failed" | "unavailable"
      - error       : str | None
    Plus all keys from analyze_url() on success.
    """
    if not is_model_loaded():
        return {
            "agent_name": "url_analysis_agent",
            "status": "unavailable",
            "error": _model_load_error or "Model not loaded.",
        }

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _run_sync, url)
        return result
    except Exception as exc:
        logger.exception("URL agent: unexpected error for URL %r", url)
        return {
            "agent_name": "url_analysis_agent",
            "status": "failed",
            "error": str(exc),
        }


def _run_sync(url: str) -> Dict[str, Any]:
    """
    Synchronous wrapper around analyze_url().
    Executed in a thread pool by run().
    """
    try:
        from agents.url_agent.url_investigator import analyze_url  # noqa: PLC0415
        result = analyze_url(url, _model, _feature_names)
        result["agent_name"] = "url_analysis_agent"
        result["status"] = "completed"
        result["error"] = None
        return result
    except Exception as exc:
        logger.error("URL agent sync execution failed for %r: %s", url, exc)
        return {
            "agent_name": "url_analysis_agent",
            "status": "failed",
            "error": str(exc),
        }
