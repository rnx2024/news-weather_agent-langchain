from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

from app.news.news_service import get_news_items
from app.weather.weather_service import classify_weather_code, get_weather_line, get_weather_summary
from app.travel_intelligence import classify_risk_level, filter_relevant_news_items, score_news_risk, score_weather_risk


RiskLevel = Literal["low", "medium", "high"]
SourceType = Literal["weather", "news"]


class BriefSource(TypedDict):
    type: SourceType


class TravelBrief(TypedDict):
    place: str
    final: str
    risk_level: RiskLevel
    travel_advice: List[str]
    sources: List[BriefSource]
    weather_summary: Dict[str, Any] | None
    weather_reasons: List[str]
    news_reasons: List[str]
    news_items: List[Dict[str, Any]]


def _score_weather(summary: Dict[str, Any] | None) -> tuple[int, List[str], List[str]]:
    if not summary:
        return 0, [], []

    current = summary.get("current") or {}
    day = summary.get("day") or {}

    score, reasons = score_weather_risk(summary)
    advice: List[str] = []

    weather_category = classify_weather_code(current.get("weather_code"))
    if weather_category == "thunderstorm":
        advice.append("Avoid exposed outdoor activities during storm windows")
    elif weather_category == "heavy_rain":
        advice.append("Carry rain protection and allow extra transit time")
    elif weather_category == "rain":
        advice.append("Carry light rain protection")
    elif weather_category == "snow":
        advice.append("Check road and transit conditions before departure")
    elif weather_category == "fog":
        advice.append("Allow extra time for driving or transfers")

    wind_max = day.get("wind_speed_max_kmh") or 0
    if wind_max >= 70:
        advice.append("Monitor ferry, flight, and exposed-route conditions")
    elif wind_max >= 50:
        advice.append("Secure flexible transport plans if traveling later")

    tmax = day.get("tmax_c")
    if tmax is not None and tmax >= 35:
        advice.append("Plan strenuous outdoor activity earlier or later in the day")

    uv_index = day.get("uv_index_max")
    if uv_index is not None and uv_index >= 8:
        advice.append("Use sun protection during midday hours")

    return score, reasons, advice


def _format_temp_range(day: Dict[str, Any]) -> str:
    tmin = day.get("tmin_c")
    tmax = day.get("tmax_c")
    if tmin is None or tmax is None:
        return ""
    return f"{tmin}-{tmax}°C"


def _trim_text(text: str, limit: int = 140) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _build_news_detail(item: Dict[str, Any]) -> str:
    title = _trim_text(str(item.get("title") or ""), 110)
    snippet = _trim_text(str(item.get("snippet") or ""), 140)
    if snippet and snippet.lower() not in title.lower():
        return f"{title} ({snippet})"
    return title


def _infer_news_travel_impact(item: Dict[str, Any]) -> str:
    blob = f"{item.get('title') or ''} {item.get('snippet') or ''}".lower()
    if any(term in blob for term in ("airport", "flight", "ferry", "rail", "train", "bus", "traffic", "road")):
        return "which could affect transfers or arrival timing"
    if any(term in blob for term in ("protest", "strike", "closure", "closed", "cancelled", "canceled", "delay")):
        return "which could affect access, opening status, or schedules"
    if any(term in blob for term in ("flood", "landslide", "evacuation", "emergency", "storm", "fire")):
        return "which may create safety or routing issues"
    if any(term in blob for term in ("festival", "concert", "event", "crowd")):
        return "which could increase crowding or change local access"
    return "which may affect local plans"


def _baseline_weather_advice(summary: Dict[str, Any] | None) -> str:
    if not summary:
        return ""
    current = summary.get("current") or {}
    day = summary.get("day") or {}
    weather_text = str(current.get("weather_text") or "").strip().lower()
    temp_range = _format_temp_range(day)
    if weather_text and temp_range:
        return f"Forecast is {weather_text} around {temp_range}, so routine transfers and outdoor plans look manageable"
    if weather_text:
        return f"Forecast is {weather_text}, so normal planning looks reasonable if live conditions stay steady"
    return ""


def _build_news_advice(news_items: List[Dict[str, Any]], score: int) -> List[str]:
    if not news_items:
        return []

    detail = _build_news_detail(news_items[0])
    impact = _infer_news_travel_impact(news_items[0])
    prefix = "Recent local reporting highlights"
    if score >= 3:
        prefix = "A significant recent local report highlights"
    return [f"{prefix} {detail}, {impact}; verify the latest official status before departure"]


def _score_news(place: str, headlines: List[Dict[str, Any]]) -> tuple[int, List[str], List[str], List[Dict[str, Any]]]:
    score, reasons = score_news_risk(place, headlines)
    relevant_items = filter_relevant_news_items(place, headlines)
    advice = _build_news_advice(relevant_items, score)
    return score, reasons, advice, relevant_items


def _dedupe(items: List[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _compose_final(
    place: str,
    risk_level: RiskLevel,
    weather_summary: Dict[str, Any] | None,
    news_items: List[Dict[str, Any]],
    *,
    news_scan_available: bool,
) -> str:
    intro_map = {
        "low": f"{place} looks generally fine for travel today.",
        "medium": f"{place} is manageable for travel today, but some caution is warranted.",
        "high": f"{place} may be challenging for travel today.",
    }
    parts = [intro_map[risk_level]]

    if weather_summary:
        current = weather_summary.get("current") or {}
        day = weather_summary.get("day") or {}
        weather_text = current.get("weather_text")
        precip = day.get("precip_mm")
        wind = day.get("wind_speed_max_kmh")
        temp_range = _format_temp_range(day)

        if weather_text:
            sentence = f"Expect {str(weather_text).lower()} conditions"
            if temp_range:
                sentence += f" with temperatures around {temp_range}"
            if precip and precip >= 5:
                sentence += " and some rain-related delays possible"
            if wind and wind >= 50:
                sentence += " with wind that may affect exposed routes"
            parts.append(sentence + ".")

    if news_items:
        top_item = news_items[0]
        parts.append(
            f"Recent local reporting mentions {_build_news_detail(top_item)}, {_infer_news_travel_impact(top_item)}."
        )
    elif news_scan_available:
        parts.append("No major recent local disruptions were identified from the current news scan.")
    else:
        parts.append("Recent local news could not be confirmed from the current scan.")

    return " ".join(parts)


def build_travel_brief(place: str) -> tuple[TravelBrief, str]:
    weather_summary, weather_summary_err = get_weather_summary(place, "today")
    weather_line = ""
    weather_line_err = ""
    if not weather_summary:
        weather_line, weather_line_err = get_weather_line(place)

    headlines, news_err = get_news_items(place)

    weather_score, weather_reasons, weather_advice = _score_weather(weather_summary)
    news_score, news_reasons, news_advice, relevant_news_items = _score_news(place, headlines)

    if not headlines and not news_err:
        news_advice = ["Current news scan did not surface a major traveler-facing disruption"]

    total_score = weather_score + news_score
    risk_level = classify_risk_level(total_score)
    advice = _dedupe(weather_advice + news_advice)

    if not advice:
        baseline_weather = _baseline_weather_advice(weather_summary)
        advice = [baseline_weather] if baseline_weather else []

    if weather_summary and not weather_advice:
        baseline_weather = _baseline_weather_advice(weather_summary)
        advice = _dedupe([baseline_weather] + advice) if baseline_weather else advice

    sources: List[BriefSource] = []
    if weather_summary or weather_line:
        sources.append({"type": "weather"})
    if headlines:
        sources.append({"type": "news"})

    final = _compose_final(
        place,
        risk_level,
        weather_summary,
        relevant_news_items,
        news_scan_available=not bool(news_err),
    )

    if weather_line and not weather_summary:
        final = f"{final} Weather snapshot: {weather_line}."

    errors = [err for err in (weather_summary_err or weather_line_err, news_err) if err]
    if not final:
        final = f"{place} travel conditions could not be summarized right now."

    brief: TravelBrief = {
        "place": place,
        "final": final,
        "risk_level": risk_level,
        "travel_advice": advice[:3],
        "sources": sources,
        "weather_summary": weather_summary,
        "weather_reasons": weather_reasons,
        "news_reasons": news_reasons,
        "news_items": relevant_news_items[:3],
    }

    if not sources:
        brief["travel_advice"] = ["Retry shortly because current weather and news sources were unavailable"]

    if not weather_reasons and not news_reasons and sources:
        if risk_level == "low" and not news_err:
            quiet_news = "Current news scan did not surface a major traveler-facing disruption"
            brief["travel_advice"] = _dedupe(brief["travel_advice"] + [quiet_news])[:3]

    return brief, "; ".join(errors)
