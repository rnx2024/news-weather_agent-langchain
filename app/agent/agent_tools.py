# app/agent_tools.py (UPDATED: delegates helpers; public tool names/signatures unchanged)
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel
from langchain_core.tools import tool

from app.weather.weather_service import get_weather_line, get_weather_summary
from app.news.news_service import get_news_items
from app.travel_intelligence import classify_risk_level, score_news_risk, score_weather_risk

from app.tooling.sync_cache import (
    CACHE_TTL_SECONDS_DEFAULT,
    cache_get_json,
    cache_get_str,
    cache_set_json,
    cache_set_str,
)
from app.tooling.text_normalize import normalize_text
from app.tooling.retry_rate_limit import RateLimiter, is_error_result, retry


# -----------------------------
# Global cache (Redis) for tools
# -----------------------------
CACHE_TTL_SECONDS = CACHE_TTL_SECONDS_DEFAULT

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
# Tools
# -----------------------------
@tool(args_schema=WeatherInput)
def weather_tool(place: str, horizon: Optional[str] = "today") -> str:
    """Get a concise weather summary for a specific place and time horizon."""
    hz = (horizon or "today").strip().lower()
    cache_key = f"cache:tool:weather_line:{normalize_text(place)}:{normalize_text(hz)}"
    cached = cache_get_str(cache_key)
    if cached is not None:
        return cached

    weather_rate.acquire()

    def call():
        if hz in ("today", "now"):
            line, err = get_weather_line(place)
            if err:
                raise RuntimeError(err)
            return line or "No weather data."

        summary, err = get_weather_summary(place, hz)
        if err or not summary:
            raise RuntimeError(err or "No forecast data.")

        cur = summary.get("current") or {}
        day = summary.get("day") or {}
        place_label = summary.get("place_label") or place
        label = day.get("label") or hz
        wx = cur.get("weather_text") or "n/a"
        tmin = day.get("tmin_c")
        tmax = day.get("tmax_c")
        precip = day.get("precip_mm")
        wind = day.get("wind_speed_max_kmh")

        parts = [f"{place_label} ({label}): {wx}"]
        if tmin is not None and tmax is not None:
            parts.append(f"{tmin}–{tmax}°C")
        if precip is not None:
            parts.append(f"precip ~{precip} mm")
        if wind is not None:
            parts.append(f"wind max ~{wind} km/h")
        return ", ".join(parts)

    out = retry(call)
    if isinstance(out, str) and not is_error_result(out):
        cache_set_str(cache_key, out, ttl=CACHE_TTL_SECONDS)
    return out


@tool(args_schema=NewsInput)
def news_tool(place: str) -> str:
    """Fetch recent news headlines for a specific city or region."""
    cache_key = f"cache:tool:news_lines:{normalize_text(place)}"
    cached = cache_get_str(cache_key)
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
    if isinstance(out, str) and not is_error_result(out):
        cache_set_str(cache_key, out, ttl=CACHE_TTL_SECONDS)
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
    w_key = f"cache:tool:weather_summary:{normalize_text(place)}:{normalize_text(horizon)}"
    n_key = f"cache:tool:news_items:{normalize_text(place)}"

    def _get_or_fetch_weather() -> Any:
        cached = cache_get_json(w_key)
        if cached is not None:
            return cached

        summary, werr = get_weather_summary(place, horizon)
        if werr and not summary:
            raise RuntimeError(werr)
        cache_set_json(w_key, summary, ttl=CACHE_TTL_SECONDS)
        return summary

    def _get_or_fetch_news() -> Any:
        cached = cache_get_json(n_key)
        if cached is not None:
            return cached

        headlines, _nerr = get_news_items(place)
        cache_set_json(n_key, headlines or [], ttl=CACHE_TTL_SECONDS)
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

        w_score, w_reasons = score_weather_risk(summary)
        risk_score += w_score
        reasons.extend(w_reasons)

        n_score, n_reasons = score_news_risk(place, headlines)
        risk_score += n_score
        reasons.extend(n_reasons)

        level = classify_risk_level(risk_score, uppercase=True)
        return _build_message(level, reasons)

    return retry(call)
