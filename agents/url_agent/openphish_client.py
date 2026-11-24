

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urlencode

import requests


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENPHISH_API_URL = "https://openphish.eu/api/check"

# Seconds to wait for a response before giving up.
_REQUEST_TIMEOUT = 10

# Human-readable status strings used throughout the project.
STATUS_FOUND = "Known phishing URL (matched in OpenPhish database)"
STATUS_NOT_FOUND = "Not found in OpenPhish database (does not confirm safety)"
STATUS_UNAVAILABLE = "Unavailable"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_url(url: str, timeout: int = _REQUEST_TIMEOUT) -> dict:
    """Query the OpenPhish Community API and return a structured result.

    Parameters
    ----------
    url:
        The URL to check against the OpenPhish database.
    timeout:
        Maximum seconds to wait for a server response (default 10 s).

    Returns
    -------
    dict with keys:
        queried_url  : str         - the URL that was submitted.
        found        : bool | None - True/False when the API responded,
                                     None when the API was unreachable.
        status       : str         - human-readable summary, always present.
        error        : str | None  - error description or None on success.

    Notes
    -----
    This function **never raises**. All exceptions are caught and
    reflected in the ``found=None`` / ``status="Unavailable"`` result so
    that the caller's investigation can always continue.

    The ``found=False`` result does **not** confirm that a URL is safe;
    it only means the URL is not currently in the OpenPhish database.
    """
    base_result = {
        "queried_url": url,
        "found": None,
        "status": STATUS_UNAVAILABLE,
        "error": None,
    }

    query_string = urlencode({"url": url})
    endpoint = f"{OPENPHISH_API_URL}?{query_string}"

    try:
        response = requests.get(endpoint, timeout=timeout)
        response.raise_for_status()

    except requests.Timeout:
        msg = f"Request timed out after {timeout} s"
        logger.warning("OpenPhish: %s for URL %r", msg, url)
        base_result["error"] = msg
        return base_result

    except requests.ConnectionError as exc:
        msg = f"Connection failed: {exc}"
        logger.warning("OpenPhish: %s", msg)
        base_result["error"] = msg
        return base_result

    except requests.HTTPError as exc:
        msg = f"HTTP error {exc.response.status_code}: {exc}"
        logger.warning("OpenPhish: %s", msg)
        base_result["error"] = msg
        return base_result

    except requests.RequestException as exc:
        msg = f"Request error: {exc}"
        logger.warning("OpenPhish: %s", msg)
        base_result["error"] = msg
        return base_result

    # --- Parse JSON body -----------------------------------------------
    try:
        data = response.json()
    except ValueError as exc:
        msg = f"Invalid JSON in response: {exc}"
        logger.warning("OpenPhish: %s -- raw body: %r", msg, response.text[:200])
        base_result["error"] = msg
        return base_result

    if not isinstance(data, dict) or "found" not in data:
        msg = (
            'Unexpected response structure (expected {"found": bool}), '
            f"got: {str(data)[:200]}"
        )
        logger.warning("OpenPhish: %s", msg)
        base_result["error"] = msg
        return base_result

    found_value = data["found"]

    if not isinstance(found_value, bool):
        msg = f'"found" field is not a boolean: {found_value!r}'
        logger.warning("OpenPhish: %s", msg)
        base_result["error"] = msg
        return base_result

    return {
        "queried_url": url,
        "found": found_value,
        "status": STATUS_FOUND if found_value else STATUS_NOT_FOUND,
        "error": None,
    }
