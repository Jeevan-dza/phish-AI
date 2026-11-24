"""
PhishGuard AI – Backend Configuration
======================================
Central configuration loaded from environment variables with safe defaults.
All sensitive values should be supplied via a .env file or environment variables
in production. Never commit secrets to source control.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# Two levels up from this file  →  project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Default model used by the URL analysis agent
DEFAULT_MODEL_PATH: Path = PROJECT_ROOT / "models" / "url_random_forest.joblib"

# Directory where investigation reports are persisted as JSON files
REPORTS_DIR: Path = PROJECT_ROOT / "backend" / "reports_store"

# ---------------------------------------------------------------------------
# Server settings
# ---------------------------------------------------------------------------

HOST: str = os.getenv("PHISHGUARD_HOST", "0.0.0.0")
PORT: int = int(os.getenv("PHISHGUARD_PORT", "8000"))
RELOAD: bool = os.getenv("PHISHGUARD_RELOAD", "true").lower() == "true"

# ---------------------------------------------------------------------------
# CORS – allow the React dev server and any explicit override
# ---------------------------------------------------------------------------

_cors_env: str = os.getenv(
    "PHISHGUARD_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
)
CORS_ORIGINS: list[str] = [origin.strip() for origin in _cors_env.split(",") if origin.strip()]

# ---------------------------------------------------------------------------
# Request / investigation limits
# ---------------------------------------------------------------------------

# Maximum URL length accepted from the client
MAX_URL_LENGTH: int = int(os.getenv("PHISHGUARD_MAX_URL_LENGTH", "2048"))

# Maximum message text length accepted from the client
MAX_MESSAGE_LENGTH: int = int(os.getenv("PHISHGUARD_MAX_MESSAGE_LENGTH", "10000"))

# Maximum uploaded image size in bytes  (default: 5 MB)
MAX_IMAGE_BYTES: int = int(os.getenv("PHISHGUARD_MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))

# Allowed image MIME types
ALLOWED_IMAGE_TYPES: set[str] = {"image/png", "image/jpeg", "image/webp", "image/gif"}

# Per-agent execution timeout in seconds
AGENT_TIMEOUT_SECONDS: int = int(os.getenv("PHISHGUARD_AGENT_TIMEOUT", "30"))

# ---------------------------------------------------------------------------
# SSRF protection – blocked private/reserved CIDRs
# ---------------------------------------------------------------------------

BLOCKED_NETWORKS: list[str] = [
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "127.0.0.0/8",
    "169.254.0.0/16",   # link-local
    "::1/128",           # IPv6 loopback
    "fc00::/7",          # IPv6 ULA
    "0.0.0.0/8",
    "100.64.0.0/10",    # shared address space (RFC 6598)
    "198.18.0.0/15",    # benchmark testing
    "240.0.0.0/4",      # reserved
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.getenv("PHISHGUARD_LOG_LEVEL", "INFO").upper()
