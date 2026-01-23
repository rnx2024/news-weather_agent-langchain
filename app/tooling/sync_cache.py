# app/tooling/sync_cache.py
from __future__ import annotations

import json
import os
from typing import Any, Optional

import redis

CACHE_TTL_SECONDS_DEFAULT = 3600

_sync_redis: Optional[redis.Redis] = None


def _get_sync_redis() -> Optional[redis.Redis]:
    """
    Sync Redis client for LangGraph sync tool calls.
    Uses REDIS_URL. Safe to return None if misconfigured.
    """
    global _sync_redis
    if _sync_redis is not None:
        return _sync_redis

    url = os.environ.get("REDIS_URL")
    if not url:
        return None

    try:
        _sync_redis = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            health_check_interval=30,
        )
        _sync_redis.ping()
        return _sync_redis
    except Exception:
        _sync_redis = None
        return None


def norm(s: str) -> str:
    return (s or "").strip().lower()


def cache_get_str(key: str) -> Optional[str]:
    r = _get_sync_redis()
    if r is None:
        return None
    try:
        v = r.get(key)
        return v if v is not None else None
    except Exception:
        return None


def cache_set_str(key: str, value: str, ttl: int = CACHE_TTL_SECONDS_DEFAULT) -> None:
    r = _get_sync_redis()
    if r is None:
        return
    try:
        r.set(key, value, ex=ttl)
    except Exception:
        return


def cache_get_json(key: str) -> Optional[Any]:
    raw = cache_get_str(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def cache_set_json(key: str, obj: Any, ttl: int = CACHE_TTL_SECONDS_DEFAULT) -> None:
    try:
        raw = json.dumps(obj, ensure_ascii=False)
    except Exception:
        return
    cache_set_str(key, raw, ttl=ttl)
