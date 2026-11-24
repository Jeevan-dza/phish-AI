"""
PhishGuard AI – URL Validators & SSRF Protection
==================================================
Input validation and SSRF-prevention utilities.

SSRF protection strategy:
  1. Parse and validate URL structure.
  2. Refuse schemes other than http/https.
  3. Resolve the hostname to its IP address.
  4. Reject the request if the resolved IP falls inside any private /
     reserved CIDR block.
  5. Checks are synchronous; they intentionally do NOT follow redirects
     (redirect targets are checked at fetch time by the agent, not here).

Do NOT use user-supplied URLs to make unchecked outbound connections.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlsplit

from backend.config import BLOCKED_NETWORKS, MAX_URL_LENGTH

logger = logging.getLogger(__name__)

# Pre-parse the blocked networks once at import time.
_BLOCKED_NETWORKS = [ipaddress.ip_network(n, strict=False) for n in BLOCKED_NETWORKS]


class URLValidationError(ValueError):
    """Raised when a submitted URL fails validation."""


def _resolve_hostname(hostname: str) -> list[str]:
    """Return all IP addresses the hostname resolves to."""
    try:
        results = socket.getaddrinfo(hostname, None)
        return [r[4][0] for r in results]
    except socket.gaierror:
        return []


def _ip_is_blocked(ip_str: str) -> bool:
    """Return True if the IP address falls inside a blocked private/reserved range."""
    try:
        addr = ipaddress.ip_address(ip_str)
        return any(addr in network for network in _BLOCKED_NETWORKS)
    except ValueError:
        # Unparseable address – treat as blocked to be safe.
        return True


def validate_url(url: str) -> str:
    """
    Validate and sanitise a URL string.

    Raises
    ------
    URLValidationError
        If the URL is structurally invalid, uses a disallowed scheme,
        is too long, or resolves to a private / reserved IP address.

    Returns
    -------
    str
        The cleaned URL (stripped of surrounding whitespace).
    """
    if not url or not isinstance(url, str):
        raise URLValidationError("URL must be a non-empty string.")

    url = url.strip()

    if len(url) > MAX_URL_LENGTH:
        raise URLValidationError(
            f"URL exceeds maximum length of {MAX_URL_LENGTH} characters."
        )

    # --- Scheme check -------------------------------------------------------
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise URLValidationError(
            f"URL scheme '{parsed.scheme}' is not allowed. Only http and https are accepted."
        )

    # --- Hostname must be present -------------------------------------------
    hostname = parsed.hostname
    if not hostname:
        raise URLValidationError("URL does not contain a valid hostname.")

    # --- Reject bare IP literals that are private/reserved ------------------
    try:
        addr_obj = ipaddress.ip_address(hostname)
        if _ip_is_blocked(str(addr_obj)):
            raise URLValidationError(
                "URL resolves to a private or reserved IP address (SSRF protection)."
            )
    except ValueError:
        # Not a bare IP literal – fall through to DNS resolution check.
        pass

    # --- DNS resolution + SSRF check ----------------------------------------
    # Note: resolution can fail in sandboxed / offline environments.
    # We log the failure but do not block the investigation, because
    # the agent-level HTTP client will enforce its own timeout.
    resolved_ips = _resolve_hostname(hostname)
    if resolved_ips:
        for ip in resolved_ips:
            if _ip_is_blocked(ip):
                raise URLValidationError(
                    f"URL hostname '{hostname}' resolves to a private or reserved "
                    f"IP address ({ip}). Request blocked (SSRF protection)."
                )
    else:
        logger.warning(
            "Could not resolve hostname '%s' during validation. "
            "The agent will attempt the connection and apply its own timeout.",
            hostname,
        )

    return url
