# app/agent_tools.py
from __future__ import annotations

import time
from typing import Callable, Any, Optional
from pydantic import BaseModel
from langchain_core.tools import tool

from app.weather_service import (
    get_weather_line,
    get_weather_summary,
    classify_weather_code,
)
from app.news_service import get_news_items


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
# Tools
# -----------------------------
@tool(args_schema=WeatherInput)
def weather_tool(place: str, horizon: Optional[str] = "today") -> str:
    """Get a concise weather summary for a specific place and time horizon."""
    weather_rate.acquire()

    def call():
        line, err = get_weather_line(place)
        if err:
            raise RuntimeError(err)
        return line or "No weather data."

    return retry(call)


@tool(args_schema=NewsInput)
def news_tool(place: str) -> str:
    """Fetch recent news headlines for a specific city or region."""
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

    return retry(call)


@tool(args_schema=RiskInput)
def city_risk_tool(
    place: str,
    horizon: str = "today",
    activity: Optional[str] = None,
) -> str:
    """
    Assess city risk level (LOW, MEDIUM, HIGH) for outdoor activity
    based on forecasted weather and recent local news.
    """
    weather_rate.acquire()
    news_rate.acquire()

    def call():
        summary, werr = get_weather_summary(place, horizon)
        headlines, nerr = get_news_items(place)

        if werr and not summary:
            raise RuntimeError(werr)

        risk_score = 0
        reasons = []

        # -------------------------
        # Weather-based risk
        # -------------------------
        if summary:
            cur = summary["current"]
            day = summary["day"]

            code = cur.get("weather_code")
            cat = classify_weather_code(code)

            if cat == "thunderstorm":
                risk_score += 3
                reasons.append("thunderstorms expected")
            elif cat == "heavy_rain":
                risk_score += 2
                reasons.append("heavy rain expected")
            elif cat == "rain":
                risk_score += 1
                reasons.append("rain likely")
            elif cat == "snow":
                risk_score += 2
                reasons.append("snow or icy conditions")
            elif cat == "fog":
                risk_score += 1
                reasons.append("fog reducing visibility")

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

        # -------------------------
        # News-based risk (LOCAL ONLY)
        # -------------------------
        place_l = place.lower()
        relevant_blobs = []

        for h in headlines or []:
            blob = (
                (h.get("title") or "")
                + " "
                + (h.get("snippet") or "")
            ).lower()

            if place_l in blob:
                relevant_blobs.append(blob)

        titles = " ".join(relevant_blobs)

        if titles:
            if any(k in titles for k in ("flood", "landslide", "evacuation", "emergency")):
                risk_score += 3
                reasons.append("severe local incident reported")
            elif any(k in titles for k in ("protest", "strike", "closure", "outage", "traffic")):
                risk_score += 2
                reasons.append("local disruption reported")

        # -------------------------
        # Final classification
        # -------------------------
        if risk_score >= 5:
            level = "HIGH"
        elif risk_score >= 2:
            level = "MEDIUM"
        else:
            level = "LOW"

        msg = f"Risk level: {level}. "
        if reasons:
            msg += "Key factors: " + "; ".join(dict.fromkeys(reasons)) + "."
        if activity:
            msg += f" Activity: {activity}."

        return msg

    return retry(call)
