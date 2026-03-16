from __future__ import annotations

from typing import Any, Dict, List

from app.weather.weather_service import classify_weather_code


_SEVERE_NEWS_TERMS = ("flood", "landslide", "evacuation", "emergency", "airport closure", "airport closed")
_DISRUPTION_NEWS_TERMS = (
    "protest",
    "strike",
    "closure",
    "closed",
    "outage",
    "traffic",
    "delay",
    "delays",
    "cancelled",
    "canceled",
    "ferry",
    "airport",
    "rail",
)


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


def filter_relevant_news_items(place: str, headlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not headlines:
        return []

    place_lc = (place or "").strip().lower()
    if not place_lc:
        return headlines

    relevant_items: List[Dict[str, Any]] = []
    for item in headlines:
        title = str(item.get("title") or "")
        snippet = str(item.get("snippet") or "")
        blob = f"{title} {snippet}".strip().lower()
        if place_lc in blob:
            relevant_items.append(item)

    # SerpAPI results were already queried for the selected place. If the snippet
    # omits the exact place string, keep the original results rather than losing
    # potentially relevant disruption details.
    return relevant_items or headlines


def score_news_risk(place: str, headlines: List[Dict[str, Any]]) -> tuple[int, List[str]]:
    relevant_items = filter_relevant_news_items(place, headlines)
    if not relevant_items:
        return 0, []

    combined = " ".join(
        f"{str(item.get('title') or '')} {str(item.get('snippet') or '')}".strip().lower()
        for item in relevant_items
    )

    score = 0
    reasons: List[str] = []
    if any(term in combined for term in _SEVERE_NEWS_TERMS):
        score += 3
        reasons.append("recent reports suggest a significant local disruption")
    elif any(term in combined for term in _DISRUPTION_NEWS_TERMS):
        score += 2
        reasons.append("recent reports suggest possible transport or access disruptions")

    return score, reasons


def classify_risk_level(score: int, *, uppercase: bool = False) -> str:
    if score >= 5:
        return "HIGH" if uppercase else "high"
    if score >= 2:
        return "MEDIUM" if uppercase else "medium"
    return "LOW" if uppercase else "low"
