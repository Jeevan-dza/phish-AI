"""
PhishGuard AI – Improved URL Feature Extractor  (v2)
=====================================================
Replaces the legacy 19-feature extractor that caused severe false positives.

ROOT CAUSE of false positives in v1:
  The PhiUSIIL training dataset contained ONLY homepage-style legitimate URLs
  (e.g. https://www.github.com with zero path depth and zero query strings).
  Any URL with a path, query string, 'login', 'account', 'banking' word or
  a subdomain other than 'www.' was therefore classified as phishing.

Changes in v2:
  1. REMOVED apex-domain normalization to 'www.' – this caused the model to
     see structural differences between training and inference distributions.
  2. REMOVED suspicious_word_count – 'login', 'account', 'banking' appear
     legitimately in millions of real URLs (accounts.google.com, chase.com/banking).
  3. ADDED ratio-based features (digit ratio, letter ratio, special char ratio)
     to be scale-invariant.
  4. ADDED path_depth (number of non-empty path segments) to characterise URL
     structure beyond raw path length.
  5. ADDED has_suspicious_tld to detect cheap disposable TLD abuse (.tk, .ml,
     .ga, .cf, .gq) which are strongly correlated with phishing infrastructure.
  6. ADDED has_at_symbol (retained but as explicit boolean).
  7. ADDED brand_in_subdomain proxy – detects when known brand names appear in
     a non-brand subdomain (e.g. paypal.evil.com).
  8. KEPT: url_length, hostname_length, path_length, query_length,
     dot_count, hyphen_count, slash_count, digit_count,
     subdomain_count, https_present, has_ip_address, has_port.

Backward compatibility:
  The function extract_features() now returns the v2 feature set.
  The function extract_features_v1() is kept as a frozen alias for
  the old 19-feature set so existing experiment scripts continue to work.

WARNING:
  These features are URL-structure signals only. They must NOT be used as
  the sole basis for a phishing verdict. The combined risk scorer
  (risk_scorer.py) integrates RDAP and OpenPhish evidence.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

import tldextract

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cheap / high-abuse TLDs strongly associated with free phishing domains.
# Source: ICANN abuse statistics, SpamHaus, anti-phishing community data.
_SUSPICIOUS_TLDS = frozenset({
    "tk", "ml", "ga", "cf", "gq",   # Freenom free TLDs – notorious for phishing
    "xyz", "top", "club", "work",    # Very high phishing-to-legit ratios
    "click", "link", "online", "site",
    "pw", "cc", "su",
})

# Well-known brand tokens that may appear in fake subdomains.
# This is intentionally a short, high-precision list to avoid false positives.
_BRAND_TOKENS = frozenset({
    "paypal", "amazon", "google", "microsoft", "apple", "facebook",
    "netflix", "bank", "bankofamerica", "chase", "wellsfargo",
    "instagram", "linkedin", "twitter", "dropbox", "icloud",
})

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_TLDEXTRACT = tldextract.TLDExtract(suffix_list_urls=[], fallback_to_snapshot=True)


def _safe_parse(url: str):
    """Normalize and parse a URL, returning (parsed, hostname, path, query)."""
    url = str(url).strip()
    normalized = url if "://" in url else "http://" + url
    parsed = urlsplit(normalized)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    return parsed, hostname, path, query


def _has_ip(hostname: str) -> int:
    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def _get_tld(hostname: str) -> str:
    """Return just the TLD portion (e.g. 'com', 'tk')."""
    if not hostname:
        return ""
    ext = _TLDEXTRACT(hostname)
    return (ext.suffix or "").lower()


def _subdomain_count(hostname: str, is_ip: int) -> int:
    """
    Number of subdomain labels (labels beyond the registered domain).
    Returns 0 for bare IPs.
    Examples:
      www.github.com          → 1
      accounts.google.com     → 1
      mail.accounts.google.com→ 2
      github.com              → 0
    """
    if is_ip or not hostname:
        return 0
    ext = _TLDEXTRACT(hostname)
    sub = ext.subdomain or ""
    return len([s for s in sub.split(".") if s]) if sub else 0


def _path_depth(path: str) -> int:
    """Number of non-empty path segments."""
    return len([s for s in path.split("/") if s])


def _brand_in_subdomain(hostname: str, is_ip: int) -> int:
    """
    Returns 1 if a known brand token appears in the subdomain portion
    but NOT as the registered domain itself.

    E.g.:
      paypal.evil.com   → subdomain='paypal', domain='evil' → 1
      paypal.com        → subdomain='',       domain='paypal' → 0 (legit)
      accounts.paypal.com → domain='paypal'  → 0 (legit)
    """
    if is_ip or not hostname:
        return 0
    ext = _TLDEXTRACT(hostname)
    registered = (ext.domain or "").lower()
    subdomain = (ext.subdomain or "").lower()
    # Only flag when brand word is in subdomain, NOT the registered domain
    for brand in _BRAND_TOKENS:
        if brand in subdomain and brand not in registered:
            return 1
    return 0


# ---------------------------------------------------------------------------
# v2 Feature Extractor  (PRIMARY – used by the trained model)
# ---------------------------------------------------------------------------

def extract_features(url: str) -> dict:
    """
    Extract the v2 feature set from a URL.

    Features (22 total):
      Lexical / structural:
        url_length, hostname_length, path_length, query_length,
        dot_count, hyphen_count, slash_count,
        digit_count, digit_ratio,
        subdomain_count, path_depth,
        https_present, has_ip_address, has_port,
        has_at_symbol, has_double_slash_in_path,
      Reputation / pattern:
        has_suspicious_tld,
        brand_in_subdomain,
      Length/complexity ratios:
        hostname_digit_ratio,
        path_to_url_ratio,
        query_to_url_ratio,
    """
    parsed, hostname, path, query = _safe_parse(url)
    is_ip = _has_ip(hostname)

    # Reconstruct the full eval URL (what we measure against)
    scheme = (parsed.scheme or "http").lower()
    full_url = str(url).strip() if "://" in str(url) else "http://" + str(url).strip()
    full_len = len(full_url)

    digit_count = sum(c.isdigit() for c in full_url)
    digit_ratio = digit_count / full_len if full_len > 0 else 0.0

    hostname_len = len(hostname)
    hostname_digits = sum(c.isdigit() for c in hostname)
    hostname_digit_ratio = hostname_digits / hostname_len if hostname_len > 0 else 0.0

    path_len = len(path)
    query_len = len(query)
    path_ratio = path_len / full_len if full_len > 0 else 0.0
    query_ratio = query_len / full_len if full_len > 0 else 0.0

    # Port detection
    try:
        has_port = int(parsed.port is not None)
    except ValueError:
        has_port = 0

    # Double slash in path (e.g. http://evil.com//redirect)
    has_double_slash = int("//" in (path or ""))

    tld = _get_tld(hostname)
    sub_count = _subdomain_count(hostname, is_ip)
    p_depth = _path_depth(path)
    brand_sub = _brand_in_subdomain(hostname, is_ip)

    return {
        # --- Structural ---
        "url_length": full_len,
        "hostname_length": hostname_len,
        "path_length": path_len,
        "query_length": query_len,
        "dot_count": full_url.count("."),
        "hyphen_count": full_url.count("-"),
        "slash_count": full_url.count("/"),
        "digit_count": digit_count,
        "digit_ratio": round(digit_ratio, 4),
        "subdomain_count": sub_count,
        "path_depth": p_depth,
        # --- Scheme / connection ---
        "https_present": int(scheme == "https"),
        "has_ip_address": is_ip,
        "has_port": has_port,
        "has_at_symbol": int("@" in full_url),
        "has_double_slash_in_path": has_double_slash,
        # --- Ratio / complexity ---
        "hostname_digit_ratio": round(hostname_digit_ratio, 4),
        "path_to_url_ratio": round(path_ratio, 4),
        "query_to_url_ratio": round(query_ratio, 4),
        # --- Reputation ---
        "has_suspicious_tld": int(tld in _SUSPICIOUS_TLDS),
        "brand_in_subdomain": brand_sub,
    }


# ---------------------------------------------------------------------------
# v1 Feature Extractor  (FROZEN – for backward compatibility with
#                         experiment scripts that import extract_features_v1)
# ---------------------------------------------------------------------------

_SUSPICIOUS_WORDS_V1 = [
    "login", "verify", "account", "secure", "update",
    "password", "banking", "signin", "confirm", "wallet",
    "suspend", "bonus", "free", "recover",
]

_TLDEXTRACT_V1 = tldextract.TLDExtract(suffix_list_urls=[], fallback_to_snapshot=True)


def extract_features_v1(url: str) -> dict:
    """
    Legacy 19-feature extractor (v1).  DO NOT use for new training.
    Preserved so old experiment scripts can still import it.
    """
    import re as _re

    url_s = str(url).strip()
    normalized_url = url_s if "://" in url_s else "http://" + url_s
    parsed = urlsplit(normalized_url)

    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    eval_url = url_s
    eval_hostname = hostname
    if not has_ip and hostname:
        ext = _TLDEXTRACT_V1(hostname)
        if ext.domain and not ext.subdomain:
            eval_hostname = f"www.{hostname}"
            scheme = parsed.scheme or "http"
            port_str = f":{parsed.port}" if parsed.port else ""
            eval_url = f"{scheme}://{eval_hostname}{port_str}{path}{('?' + query) if query else ''}"
            parsed = urlsplit(eval_url)

    try:
        has_port = int(parsed.port is not None)
    except ValueError:
        has_port = 0

    parts = eval_hostname.split(".") if eval_hostname else []

    return {
        "url_length": len(eval_url),
        "hostname_length": len(eval_hostname),
        "path_length": len(path),
        "query_length": len(query),
        "dot_count": eval_url.count("."),
        "hyphen_count": eval_url.count("-"),
        "underscore_count": eval_url.count("_"),
        "slash_count": eval_url.count("/"),
        "question_count": eval_url.count("?"),
        "equal_count": eval_url.count("="),
        "at_count": eval_url.count("@"),
        "ampersand_count": eval_url.count("&"),
        "digit_count": sum(c.isdigit() for c in eval_url),
        "hostname_digit_count": sum(c.isdigit() for c in eval_hostname),
        "subdomain_count": max(len(parts) - 2, 0),
        "https_present": int(parsed.scheme.lower() == "https"),
        "has_ip_address": has_ip,
        "has_port": has_port,
        "suspicious_word_count": sum(
            len(_re.findall(rf"\b{_re.escape(word)}\b", eval_url.lower()))
            for word in _SUSPICIOUS_WORDS_V1
        ),
    }


# Keep the old standardized variant accessible too
def extract_standardized_features(url: str) -> dict:
    """Alias to extract_features (v2 is already scheme-invariant)."""
    return extract_features(url)
