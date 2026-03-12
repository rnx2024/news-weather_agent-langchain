# app/redis_client.py
from __future__ import annotations

"""
Redis client lifecycle for SmartNews.

- One Redis connection per FastAPI process
- Initialized on startup
- Closed on shutdown
- Used ONLY for short-lived session + cache data
"""

import logging
from typing import Optional
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.settings import settings

log = logging.getLogger(__name__)

redis: Optional[Redis] = None


async def init_redis() -> None:
    """
    Initialize Redis client once per process.
    Must be called on FastAPI startup.
    """
    global redis

    if redis is not None:
        return

    redis_url = settings.redis_url
    if not redis_url:
        if settings.redis_required:
            raise RuntimeError("REDIS_URL is not set")
        log.warning("REDIS_URL is not set. Continuing without Redis.")
        return

    client = Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        health_check_interval=30,
    )

    try:
        # Health check connection eagerly to avoid hidden runtime failures.
        await client.ping()
    except (RedisError, OSError) as exc:
        await client.aclose()
        if settings.redis_required:
            raise RuntimeError(f"Redis connection failed: {exc}") from exc
        log.warning("Redis unavailable at startup. Continuing without Redis: %s", exc)
        return

    redis = client


async def close_redis() -> None:
    """
    Cleanly close Redis connection.
    Must be called on FastAPI shutdown.
    """
    global redis

    if redis is not None:
        await redis.aclose()
        redis = None
