"""
PhishGuard AI – General Helpers
=================================
Small, self-contained utilities used across the backend.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone


def generate_investigation_id(url: str) -> str:
    """
    Generate a unique, deterministic-looking investigation ID.

    Format:  inv-<short_url_hash>-<uuid4_hex[:8]>
    The hash prefix makes IDs human-scannable in logs; the UUID suffix
    ensures uniqueness even for repeated analyses of the same URL.
    """
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:8]
    unique_part = uuid.uuid4().hex[:8]
    return f"inv-{url_hash}-{unique_part}"


def utc_now() -> datetime:
    """Return the current UTC datetime with timezone info."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return the current UTC datetime as an ISO 8601 string."""
    return utc_now().isoformat()


def safe_round(value: float | None, ndigits: int = 2) -> float | None:
    """Round a float to *ndigits* decimal places, or return None if value is None."""
    if value is None:
        return None
    return round(value, ndigits)
