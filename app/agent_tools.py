# app/agent_tools.py
from __future__ import annotations

import os, json, time
from typing import Callable, Any, Optional
from pydantic import BaseModel
from langchain_core.tools import tool
from app.session_cache import get_last_exchange, mark_sent
import redis  # <-- UPDATED: force sync redis client (prevents "coroutine ... was never awaited")

from app.weather_service import (
    get_weather_line,
    get_weather_summary,
    classify_weather_code,
)
from app.news_service import get_news_items


# -----------------------------
# Global cache (Redis) for tools
# -----------------------------
CACHE_TTL_SECONDS = 3600
_sync_redis: Optional[redis.Redis] = None  # <-- UPDATED


def _get_sync_redis() -> Optional[redis.Redis]:  # <-- UPDATED
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
        _sync_redis = redis.Redis.from_url(  # <-- UPDATED
            url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            health_check_interval=30,
        )
        # Fail fast at init time (but still non-fatal)
        _sync_redis.ping()
        return _sync_redis
    except Exception:
        _sync_redis = None
        return None


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _cache_get_str(key: str) -> Optional[str]:
    r = _get_sync_redis()
    if r is None:
        return None
    try:
        v = r.get(key)
        return v if v is not None else None
    except Exception:
        return None


def _cache_set_str(key: str, value: str, ttl: int = CACHE_TTL_SECONDS) -> None:
    r = _get_sync_redis()
    if r is None:
        return
    try:
        r.set(key, value, ex=ttl)
    except Exception:
        return


def _cache_get_json(key: str) -> Optional[Any]:
    raw = _cache_get_str(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _cache_set_json(key: str, obj: Any, ttl: int = CACHE_TTL_SECONDS) -> None:
    try:
        raw = json.dumps(obj, ensure_ascii=False)
    except Exception:
        return
    _cache_set_str(key, raw, ttl=ttl)


# -----------------------------
# Retry wrapper
# -----------------------------
def retry(fn: Callable[[], Any], retries: int = 3, base_delay: float = 0.5):
    """Run a function with exponential backoff retries."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            if attempt > retries:
                return f"ERROR: {str(e)}"
            time.sleep(base_delay * (2 ** (attempt - 1)))


# -----------------------------
# Token-bucket rate limiter
# -----------------------------
class RateLimiter:
    """Simple token-bucket rate limiter to throttle API/tool usage."""

    def __init__(self, max_per_interval: int, interval_seconds: float):
        self.max_per_interval = max_per_interval
        self.interval_seconds = interval_seconds
        self.tokens = max_per_interval
        self.last_refill = time.time()

    def acquire(self):
        """Acquire a token, waiting if necessary."""
        now = time.time()
        elapsed = now - self.last_refill

        intervals = int(elapsed // self.interval_seconds)
        if intervals > 0:
            self.tokens = min(
                self.max_per_interval,
                self.tokens + intervals * self.max_per_interval,
            )
            self.last_refill = now

        if self.tokens == 0:
            sleep_for = self.interval_seconds - (now - self.last_refill)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self.tokens = self.max_per_interval
            self.last_refill = time.time()

        self.tokens -= 1


# -----------------------------
# Initialize rate limiters
# -----------------------------
weather_rate = RateLimiter(5, 1.0)
news_rate = RateLimiter(2, 1.0)


# -----------------------------
# Tool Schemas
# -----------------------------
class WeatherInput(BaseModel):
    place: str
    horizon: Optional[str] = "today"


class NewsInput(BaseModel):
    place: str


class RiskInput(BaseModel):
    place: str
    horizon: Optional[str] = "today"
    activity: Optional[str] = None


# -----------------------------
# Helpers to reduce cognitive complexity
# -----------------------------
def _score_weather_risk(summary: Any) -> tuple[int, list[str]]:
    """Return (risk_score_delta, reasons) from weather summary. No side effects."""
    if not summary:
        return 0, []

    risk_score = 0
    reasons: list[str] = []

    cur = summary["current"]
    day = summary["day"]

    code = cur.get("weather_code")
    cat = classify_weather_code(code)

    cat_score: dict[str, tuple[int, str]] = {
        "thunderstorm": (3, "thunderstorms expected"),
        "heavy_rain": (2, "heavy rain expected"),
        "rain": (1, "rain likely"),
        "snow": (2, "snow or icy conditions"),
        "fog": (1, "fog reducing visibility"),
    }
    if cat in cat_score:
        s, r = cat_score[cat]
        risk_score += s
        reasons.append(r)

    wind_max = day.get("wind_speed_max_kmh") or 0
    if wind_max >= 70:
        risk_score += 3
        reasons.append(f"very strong winds (~{wind_max} km/h)")
    elif wind_max >= 50:
        risk_score += 2
        reasons.append(f"strong winds (~{wind_max} km/h)")
    elif wind_max >= 30:
        risk_score += 1
        reasons.append(f"gusty winds (~{wind_max} km/h)")

    precip = day.get("precip_mm") or 0
    if precip >= 30:
        risk_score += 2
        reasons.append(f"heavy precipitation (~{precip} mm)")
    elif precip >= 5:
        risk_score += 1
        reasons.append(f"rainfall (~{precip} mm)")

    tmax = day.get("tmax_c")
    tmin = day.get("tmin_c")
    if tmax is not None and tmax >= 35:
        risk_score += 2
        reasons.append(f"very hot (up to {tmax}°C)")
    if tmin is not None and tmin <= -5:
        risk_score += 2
        reasons.append(f"very cold (down to {tmin}°C)")

    return risk_score, reasons


def _score_news_risk(place: str, headlines: Any) -> tuple[int, list[str]]:
    """Return (risk_score_delta, reasons) from local news signals. No side effects."""
    if not headlines:
        return 0, []

    risk_score = 0
    reasons: list[str] = []

    place_l = (place or "").lower()
    relevant_blobs: list[str] = []

    for h in headlines or []:
        blob = ((h.get("title") or "") + " " + (h.get("snippet") or "")).lower()
        if place_l in blob:
            relevant_blobs.append(blob)

    titles = " ".join(relevant_blobs)
    if not titles:
        return 0, []

    severe = ("flood", "landslide", "evacuation", "emergency")
    disruption = ("protest", "strike", "closure", "outage", "traffic")

    if any(k in titles for k in severe):
        risk_score += 3
        reasons.append("severe local incident reported")
    elif any(k in titles for k in disruption):
        risk_score += 2
        reasons.append("local disruption reported")

    return risk_score, reasons


def _classify_risk_level(score: int) -> str:
    if score >= 5:
        return "HIGH"
    if score >= 2:
        return "MEDIUM"
    return "LOW"


# -----------------------------
# Tools
# -----------------------------
@tool(args_schema=WeatherInput)
def weather_tool(place: str, horizon: Optional[str] = "today") -> str:
    """Get a concise weather summary for a specific place and time horizon."""
    # Cache key includes horizon for safety (even if get_weather_line ignores it today).
    cache_key = f"cache:tool:weather_line:{_norm(place)}:{_norm(horizon or 'today')}"
    cached = _cache_get_str(cache_key)
    if cached is not None:
        return cached

    weather_rate.acquire()

    def call():
        line, err = get_weather_line(place)
        if err:
            raise RuntimeError(err)
        return line or "No weather data."

    out = retry(call)
    # Cache even "No weather data." to reduce repeated calls
    if isinstance(out, str):
        _cache_set_str(cache_key, out, ttl=CACHE_TTL_SECONDS)
    return out


@tool(args_schema=NewsInput)
def news_tool(place: str) -> str:
    """Fetch recent news headlines for a specific city or region."""
    cache_key = f"cache:tool:news_lines:{_norm(place)}"
    cached = _cache_get_str(cache_key)
    if cached is not None:
        return cached

    news_rate.acquire()

    def call():
        headlines, err = get_news_items(place)
        if err:
            raise RuntimeError(err)
        if not headlines:
            return "No recent news."

        return "\n".join(
            f"- {h['title']} ({h['source']}, {h['date']})"
            + (f" — {h['snippet'][:160].strip()}" if h.get("snippet") else "")
            + (f" -> {h['link']}" if h.get("link") else "")
            for h in headlines
        )

    out = retry(call)
    if isinstance(out, str):
        _cache_set_str(cache_key, out, ttl=CACHE_TTL_SECONDS)
    return out


@tool(args_schema=RiskInput)
def city_risk_tool(
    place: str,
    horizon: str = "today",
    activity: Optional[str] = None,
) -> str:
    """
    Assess city risk level (LOW, MEDIUM, HIGH) for outdoor activity and also consider travel conditions
    based on forecasted weather and recent local news.
    """

    w_key = f"cache:tool:weather_summary:{_norm(place)}:{_norm(horizon)}"
    n_key = f"cache:tool:news_items:{_norm(place)}"

    def _get_or_fetch_weather() -> Any:
        cached = _cache_get_json(w_key)
        if cached is not None:
            return cached

        summary, werr = get_weather_summary(place, horizon)
        if werr and not summary:
            raise RuntimeError(werr)
        _cache_set_json(w_key, summary, ttl=CACHE_TTL_SECONDS)
        return summary

    def _get_or_fetch_news() -> Any:
        cached = _cache_get_json(n_key)
        if cached is not None:
            return cached

        headlines, _nerr = get_news_items(place)
        # keep prior behavior: cache [] even on error/None
        _cache_set_json(n_key, headlines or [], ttl=CACHE_TTL_SECONDS)
        return headlines or []

    def _build_message(level: str, reasons: list[str]) -> str:
        msg = f"Risk level: {level}. "
        if reasons:
            msg += "Key factors: " + "; ".join(dict.fromkeys(reasons)) + "."
        if activity:
            msg += f" Activity: {activity}."
        return msg

    def call() -> str:
        weather_rate.acquire()
        news_rate.acquire()

        summary = _get_or_fetch_weather()
        headlines = _get_or_fetch_news()

        risk_score = 0
        reasons: list[str] = []

        w_score, w_reasons = _score_weather_risk(summary)
        risk_score += w_score
        reasons.extend(w_reasons)

        n_score, n_reasons = _score_news_risk(place, headlines)
        risk_score += n_score
        reasons.extend(n_reasons)

        level = _classify_risk_level(risk_score)
        return _build_message(level, reasons)

    return retry(call)
