"""
PhishGuard AI – Request Schemas
=================================
Pydantic v2 models validating all data coming in from the React frontend.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    """Full investigation request – URL is mandatory; message is optional."""

    url: str = Field(
        ...,
        min_length=4,
        max_length=2048,
        description="The URL to investigate. Must start with http:// or https://.",
        examples=["https://example.com/login"],
    )
    message: Optional[str] = Field(
        default=None,
        max_length=10000,
        description=(
            "Optional message text (e.g. phishing e-mail body) for context analysis."
        ),
    )

    @field_validator("url")
    @classmethod
    def url_must_have_scheme(cls, v: str) -> str:
        stripped = v.strip()
        if not (stripped.startswith("http://") or stripped.startswith("https://")):
            raise ValueError("URL must begin with http:// or https://")
        return stripped


class URLOnlyRequest(BaseModel):
    """Lightweight request for URL-only fast prediction."""

    url: str = Field(
        ...,
        min_length=4,
        max_length=2048,
        description="URL to run ML prediction on.",
        examples=["https://example.com"],
    )

    @field_validator("url")
    @classmethod
    def url_must_have_scheme(cls, v: str) -> str:
        stripped = v.strip()
        if not (stripped.startswith("http://") or stripped.startswith("https://")):
            raise ValueError("URL must begin with http:// or https://")
        return stripped
