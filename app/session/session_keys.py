# app/session/session_keys.py
from __future__ import annotations

from typing import Any

ONE_HOUR = 3600
DEFAULT_SESSION_TTL = 86400  # 24h


def sess_key(session_id: str) -> str:
    return f"sess:{session_id}"


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def weather_key(location: str) -> str:
    loc = _norm(location) or "unknown"
    return f"cache:weather:{loc}"


def news_key(location: str) -> str:
    loc = _norm(location) or "unknown"
    return f"cache:news:{loc}"


def to_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v).strip()
        return int(s) if s else default
    except (TypeError, ValueError, OverflowError):
        return default
