# app/session_cache.py
from __future__ import annotations

import time
from typing import Awaitable, Callable, Tuple

from app.redis_client import redis

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


async def should_include(session_id: str, force_weather: bool, force_news: bool) -> Tuple[bool, bool]:
    """
    Session-level suppression rule:
    - If user explicitly asks (force_*), allow.
    - Else allow only if last sent >= 1 hour ago.
    """
    if redis is None:
        # fail-open: app still works even if Redis is down
        return True, True

    now = int(time.time())
    data = await redis.hgetall(sess_key(session_id))

    # Safe casts
    try:
        last_w = int(data.get("last_weather_sent_at", "0") or "0")
    except ValueError:
        last_w = 0
    try:
        last_n = int(data.get("last_news_sent_at", "0") or "0")
    except ValueError:
        last_n = 0

    include_weather = force_weather or (now - last_w >= ONE_HOUR)
    include_news = force_news or (now - last_n >= ONE_HOUR)
    return include_weather, include_news


async def mark_sent(
    session_id: str,
    *,
    weather_sent: bool,
    news_sent: bool,
    user_message: str | None = None,
    agent_reply: str | None = None,
    ttl_seconds: int = DEFAULT_SESSION_TTL,
) -> None:
    """
    Persist minimal per-session state:
    - last_*_sent_at timestamps
    - last exchange (one turn)
    """
    if redis is None:
        return

    now = int(time.time())
    mapping: dict[str, str] = {}

    if weather_sent:
        mapping["last_weather_sent_at"] = str(now)
    if news_sent:
        mapping["last_news_sent_at"] = str(now)

    mapping["last_chat_sent_at"] = str(now)

    if user_message:
        mapping["last_user_message"] = user_message[:500]
    if agent_reply:
        mapping["last_agent_reply"] = agent_reply[:1000]

    sk = sess_key(session_id)
    await redis.hset(sk, mapping=mapping)
    await redis.expire(sk, ttl_seconds)


async def get_last_exchange(session_id: str) -> tuple[str | None, str | None]:
    """
    Fetch exactly one previous exchange.
    Returns (last_user_message, last_agent_reply).
    """
    if redis is None:
        return None, None

    last_user, last_reply = await redis.hmget(
        sess_key(session_id),
        "last_user_message",
        "last_agent_reply",
    )
    return last_user, last_reply


async def get_or_set(
    key: str,
    ttl_seconds: int,
    compute_fn: Callable[[], Awaitable[str] | str],
) -> str:
    """
    Global cache (shared across sessions).
    compute_fn can be sync or async.
    """
    if redis is None:
        v = compute_fn()
        return await v if hasattr(v, "__await__") else v  # type: ignore[misc]

    cached = await redis.get(key)
    if cached is not None:
        return cached

    v = compute_fn()
    value = await v if hasattr(v, "__await__") else v  # type: ignore[misc]
    await redis.set(key, value, ex=ttl_seconds)
    return value


async def prepare_weather_news(
    *,
    session_id: str,
    user_text: str,
    location: str,
    fetch_weather_fn: Callable[[], Awaitable[str] | str],
    fetch_news_fn: Callable[[], Awaitable[str] | str],
) -> tuple[str, str, bool, bool]:
    """
    Endpoint-facing helper.

    Returns:
      (weather_text, news_text, include_weather, include_news)

    Empty strings mean "do not include in prompt".
    include_* flags show what was allowed by policy.
    """
    text_lc = (user_text or "").lower()
    force_weather = "weather" in text_lc
    force_news = "news" in text_lc

    include_weather, include_news = await should_include(session_id, force_weather, force_news)

    weather_text = ""
    news_text = ""

    if include_weather:
        weather_text = await get_or_set(weather_key(location), ONE_HOUR, fetch_weather_fn)

    if include_news:
        news_text = await get_or_set(news_key(location), ONE_HOUR, fetch_news_fn)

    return weather_text, news_text, include_weather, include_news
