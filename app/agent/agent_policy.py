# app/agent_policy.py
from __future__ import annotations

from typing import Optional, Tuple


def is_trip_planning_question(q: Optional[str]) -> bool:
    """
    Heuristic: identify trip/planning go/no-go questions that should trigger both
    weather + news signals even without explicit 'weather'/'news' keywords.
    """
    if not q:
        return False

    s = q.lower()

    trip_terms = (
        "trip",
        "travel",
        "go on",
        "should i go",
        "should we go",
        "visit",
        "outing",
        "hike",
        "beach",
        "roadtrip",
        "commute",
        "drive",
        "fly",
        "ferry",
    )

    date_terms = (
        "today",
        "tomorrow",
        "tonight",
        "weekend",
        "next week",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "next ",
        "this ",
    )

    intent_terms = ("safe", "risk", "cancel", "postpone", "worth it")

    return any(t in s for t in trip_terms) and (any(t in s for t in date_terms) or any(t in s for t in intent_terms))


def decide_tool_includes(question: Optional[str]) -> Tuple[bool, bool]:
    """
    Decide whether the agent should be allowed to call weather_tool/news_tool.

    Returns: (include_weather, include_news)
    """
    if not question:
        return True, True

    if is_trip_planning_question(question):
        return True, True

    q = question.lower()

    inc_w = any(
        t in q
        for t in (
            "weather",
            "forecast",
            "temperature",
            "rain",
            "storm",
            "wind",
            "uv",
            "heat",
            "snow",
            "fog",
            "typhoon",
            "hurricane",
        )
    )

    inc_n = any(
        t in q
        for t in (
            "news",
            "headline",
            "headlines",
            "update",
            "updates",
            "disruption",
            "disruptions",
            "incident",
            "incidents",
            "protest",
            "strike",
            "closure",
            "closures",
            "outage",
            "traffic",
            "where",
            "location",
            "locations",
            "area",
            "areas",
        )
    )

    return (inc_w, inc_n) if (inc_w or inc_n) else (False, False)


def detect_force_signals(question: str) -> Tuple[bool, bool]:
    """
    Decide whether the request should bypass session suppression (force include).
    Trip-planning questions force both.
    """
    q = (question or "")
    q_lc = q.lower()

    if is_trip_planning_question(q):
        return True, True

    return ("weather" in q_lc, "news" in q_lc)
