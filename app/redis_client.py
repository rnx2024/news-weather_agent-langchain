# app/redis_client.py
from __future__ import annotations

"""
Redis client lifecycle for SmartNews.

- One Redis connection per FastAPI process
- Initialized on startup
- Closed on shutdown
- Used ONLY for short-lived session + cache data
"""

import os
from typing import Optional
from redis.asyncio import Redis

redis: Optional[Redis] = None


async def init_redis() -> None:
    """
    Initialize Redis client once per process.
    Must be called on FastAPI startup.
    """
    global redis

    if redis is not None:
        return

    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        raise RuntimeError("REDIS_URL is not set")

    redis = Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        health_check_interval=30,
    )

    # Fail fast if credentials / TLS are wrong
    await redis.ping()


async def close_redis() -> None:
    """
    Cleanly close Redis connection.
    Must be called on FastAPI shutdown.
    """
    global redis

    if redis is not None:
        await redis.aclose()
        redis = None
