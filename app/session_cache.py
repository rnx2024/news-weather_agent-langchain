# app/session_cache.py
from __future__ import annotations

import time
from typing import Awaitable, Callable, Tuple, Optional, Dict, Any, Iterable

from app.redis_client import redis

ONE_HOUR = 3600
DEFAULT_SESSION_TTL = 86400  # 24h


# -----------------------------
# Keys
# -----------------------------
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


# -----------------------------
# Session state helpers
# -----------------------------
def _to_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v).strip()
        return int(s) if s else default
    except Exception:
        return default


async def get_session_state(session_id: str) -> Dict[str, Any]:
    """
    Fetch the full per-session hash.
    Returns {} if Redis is unavailable.
    """
    if redis is None:
        return {}

    try:
        data = await redis.hgetall(sess_key(session_id))
        # data is usually Dict[str,str] (decode_responses), but keep defensive
        return data or {}
    except Exception:
        return {}


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

    try:
        data = await redis.hgetall(sess_key(session_id))
    except Exception:
        return True, True

    last_w = _to_int((data or {}).get("last_weather_sent_at"), 0)
    last_n = _to_int((data or {}).get("last_news_sent_at"), 0)

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
    try:
        await redis.hset(sk, mapping=mapping)
        await redis.expire(sk, ttl_seconds)
    except Exception:
        return


async def mark_tools_called(
    session_id: str,
    *,
    tool_names: Iterable[str],
    user_message: str | None = None,
    agent_reply: str | None = None,
    ttl_seconds: int = DEFAULT_SESSION_TTL,
) -> None:
    """
    Convenience wrapper for LangGraph/LangChain runs:
    - tool_names: names actually invoked (e.g., {"weather_tool","news_tool","city_risk_tool"}).
    - Updates last_weather_sent_at / last_news_sent_at based on actual calls.
    - Also stores one-turn exchange like mark_sent().
    """
    tset = {str(t or "").strip() for t in (tool_names or []) if str(t or "").strip()}
    await mark_sent(
        session_id,
        weather_sent=("weather_tool" in tset),
        news_sent=("news_tool" in tset),
        user_message=user_message,
        agent_reply=agent_reply,
        ttl_seconds=ttl_seconds,
    )


async def get_last_exchange(session_id: str) -> tuple[str | None, str | None]:
    """
    Fetch exactly one previous exchange.
    Returns (last_user_message, last_agent_reply).
    """
    if redis is None:
        return None, None

    try:
        last_user, last_reply = await redis.hmget(
            sess_key(session_id),
            "last_user_message",
            "last_agent_reply",
        )
        return last_user, last_reply
    except Exception:
        return None, None


async def get_last_sent_timestamps(session_id: str) -> tuple[int, int, int]:
    """
    Returns (last_weather_sent_at, last_news_sent_at, last_chat_sent_at) as ints.
    0 means "never" or "unavailable".
    """
    if redis is None:
        return 0, 0, 0

    try:
        w, n, c = await redis.hmget(
            sess_key(session_id),
            "last_weather_sent_at",
            "last_news_sent_at",
            "last_chat_sent_at",
        )
        return _to_int(w, 0), _to_int(n, 0), _to_int(c, 0)
    except Exception:
        return 0, 0, 0


# -----------------------------
# Global cache (shared across sessions)
# -----------------------------
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

    try:
        cached = await redis.get(key)
        if cached is not None:
            return cached
    except Exception:
        # fail-open: compute and return
        v = compute_fn()
        return await v if hasattr(v, "__await__") else v  # type: ignore[misc]

    v = compute_fn()
    value = await v if hasattr(v, "__await__") else v  # type: ignore[misc]
    try:
        await redis.set(key, value, ex=ttl_seconds)
    except Exception:
        pass
    return value


# -----------------------------
# Endpoint-facing helper
# -----------------------------
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
