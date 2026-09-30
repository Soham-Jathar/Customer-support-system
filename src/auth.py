"""Minimal server-side agent authorization for the local prototype."""
from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException, status


def verify_agent_key(access_key: str | None) -> None:
    expected = os.getenv("AGENT_ACCESS_KEY")
    if not expected:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent access is not configured on this server.")
    if not access_key or not secrets.compare_digest(access_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Valid agent access is required.")


def require_agent(x_agent_key: str | None = Header(default=None)) -> None:
    verify_agent_key(x_agent_key)
