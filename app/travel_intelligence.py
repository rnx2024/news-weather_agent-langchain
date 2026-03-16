from __future__ import annotations

from typing import Any, List

from app.weather.weather_service import classify_weather_code


def score_weather_risk(summary: Any) -> tuple[int, List[str]]:
    if not summary:
        return 0, []

    current = summary.get("current") or {}
    day = summary.get("day") or {}

    score = 0
    reasons: List[str] = []

    weather_category = classify_weather_code(current.get("weather_code"))
    if weather_category == "thunderstorm":
        score += 3
        reasons.append("thunderstorms are expected")
    elif weather_category == "heavy_rain":
        score += 2
        reasons.append("heavy rain is likely")
    elif weather_category == "rain":
        score += 1
        reasons.append("scattered rain is possible")
    elif weather_category == "snow":
        score += 2
        reasons.append("snow or icy conditions may affect movement")
    elif weather_category == "fog":
        score += 1
        reasons.append("reduced visibility is possible")

    wind_max = day.get("wind_speed_max_kmh") or 0
    if wind_max >= 70:
        score += 3
        reasons.append("very strong winds may disrupt transport")
    elif wind_max >= 50:
        score += 2
        reasons.append("strong winds may affect outdoor plans")
    elif wind_max >= 30:
        score += 1
        reasons.append("gusty conditions are expected")

    precip = day.get("precip_mm") or 0
    if precip >= 30:
        score += 2
        reasons.append("heavy rainfall could slow local travel")
    elif precip >= 5:
        score += 1
        reasons.append("rain may affect outdoor timing")

    tmax = day.get("tmax_c")
    tmin = day.get("tmin_c")
    if tmax is not None and tmax >= 35:
        score += 2
        reasons.append("very hot conditions are expected")
    if tmin is not None and tmin <= -5:
        score += 2
        reasons.append("very cold conditions are expected")

    return score, reasons


def classify_risk_level(score: int, *, uppercase: bool = False) -> str:
    if score >= 5:
        return "HIGH" if uppercase else "high"
    if score >= 2:
        return "MEDIUM" if uppercase else "medium"
    return "LOW" if uppercase else "low"
