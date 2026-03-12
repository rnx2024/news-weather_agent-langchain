# app/session/session_store.py
from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable, Dict, Iterable, Tuple

from app.redis_client import redis
from app.session.session_keys import DEFAULT_SESSION_TTL, ONE_HOUR, news_key, sess_key, to_int, weather_key
from redis.exceptions import RedisError

log = logging.getLogger(__name__)


async def get_session_state(session_id: str) -> Dict[str, Any]:
    if redis is None:
        return {}
    try:
        data = await redis.hgetall(sess_key(session_id))
        return data or {}
    except RedisError as exc:
        log.warning("Redis hgetall failed in get_session_state [session_id=%s]: %s", session_id, exc)
        return {}


async def should_include(session_id: str, force_weather: bool, force_news: bool) -> Tuple[bool, bool]:
    if redis is None:
        return True, True

    now = int(time.time())
    try:
        data = await redis.hgetall(sess_key(session_id))
    except RedisError as exc:
        log.warning("Redis hgetall failed in should_include [session_id=%s]: %s", session_id, exc)
        return True, True

    last_w = to_int((data or {}).get("last_weather_sent_at"), 0)
    last_n = to_int((data or {}).get("last_news_sent_at"), 0)

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
    except RedisError as exc:
        log.warning("Redis write failed in mark_sent [session_id=%s]: %s", session_id, exc)
        return


async def mark_tools_called(
    session_id: str,
    *,
    tool_names: Iterable[str],
    user_message: str | None = None,
    agent_reply: str | None = None,
    ttl_seconds: int = DEFAULT_SESSION_TTL,
) -> None:
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
    if redis is None:
        return None, None
    try:
        last_user, last_reply = await redis.hmget(
            sess_key(session_id),
            "last_user_message",
            "last_agent_reply",
        )
        return last_user, last_reply
    except RedisError as exc:
        log.warning("Redis hmget failed in get_last_exchange [session_id=%s]: %s", session_id, exc)
        return None, None


async def get_last_sent_timestamps(session_id: str) -> tuple[int, int, int]:
    if redis is None:
        return 0, 0, 0
    try:
        w, n, c = await redis.hmget(
            sess_key(session_id),
            "last_weather_sent_at",
            "last_news_sent_at",
            "last_chat_sent_at",
        )
        return to_int(w, 0), to_int(n, 0), to_int(c, 0)
    except RedisError as exc:
        log.warning("Redis hmget failed in get_last_sent_timestamps [session_id=%s]: %s", session_id, exc)
        return 0, 0, 0


async def get_or_set(
    key: str,
    ttl_seconds: int,
    compute_fn: Callable[[], Awaitable[str] | str],
) -> str:
    if redis is None:
        v = compute_fn()
        return await v if hasattr(v, "__await__") else v  # type: ignore[misc]

    try:
        cached = await redis.get(key)
        if cached is not None:
            return cached
    except RedisError as exc:
        log.warning("Redis get failed in get_or_set [key=%s]: %s", key, exc)
        v = compute_fn()
        return await v if hasattr(v, "__await__") else v  # type: ignore[misc]

    v = compute_fn()
    value = await v if hasattr(v, "__await__") else v  # type: ignore[misc]
    try:
        await redis.set(key, value, ex=ttl_seconds)
    except RedisError as exc:
        log.warning("Redis set failed in get_or_set [key=%s]: %s", key, exc)
    return value


async def prepare_weather_news(
    *,
    session_id: str,
    user_text: str,
    location: str,
    fetch_weather_fn: Callable[[], Awaitable[str] | str],
    fetch_news_fn: Callable[[], Awaitable[str] | str],
) -> tuple[str, str, bool, bool]:
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
