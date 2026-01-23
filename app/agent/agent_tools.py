# app/agent_tools.py (UPDATED: delegates helpers; public tool names/signatures unchanged)
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel
from langchain_core.tools import tool

from app.session.session_cache import get_last_exchange, mark_sent  # unchanged imports (even if unused)
from app.weather.weather_service import get_weather_line, get_weather_summary, classify_weather_code
from app.news.news_service import get_news_items

from app.tooling.sync_cache import (
    CACHE_TTL_SECONDS_DEFAULT,
    cache_get_json,
    cache_get_str,
    cache_set_json,
    cache_set_str,
    norm,
)
from app.tooling.retry_rate_limit import RateLimiter, retry


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
# Helpers to reduce cognitive complexity
# -----------------------------
def _score_weather_risk(summary: Any) -> tuple[int, list[str]]:
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
    hz = (horizon or "today").strip().lower()
    cache_key = f"cache:tool:weather_line:{norm(place)}:{norm(hz)}"
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
    if isinstance(out, str):
        cache_set_str(cache_key, out, ttl=CACHE_TTL_SECONDS)
    return out


@tool(args_schema=NewsInput)
def news_tool(place: str) -> str:
    """Fetch recent news headlines for a specific city or region."""
    cache_key = f"cache:tool:news_lines:{norm(place)}"
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
    if isinstance(out, str):
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
    w_key = f"cache:tool:weather_summary:{norm(place)}:{norm(horizon)}"
    n_key = f"cache:tool:news_items:{norm(place)}"

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

        w_score, w_reasons = _score_weather_risk(summary)
        risk_score += w_score
        reasons.extend(w_reasons)

        n_score, n_reasons = _score_news_risk(place, headlines)
        risk_score += n_score
        reasons.extend(n_reasons)

        level = _classify_risk_level(risk_score)
        return _build_message(level, reasons)

    return retry(call)
