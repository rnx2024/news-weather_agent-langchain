# app/agent_policy.py
from __future__ import annotations

from typing import Literal, Optional, Tuple


AnswerMode = Literal["travel_brief", "news_followup", "weather_followup"]


_WEATHER_TERMS = (
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
    "humid",
    "humidity",
    "clear",
    "cloud",
)

_NEWS_TERMS = (
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
    "closed",
    "outage",
    "traffic",
    "where",
    "location",
    "locations",
    "area",
    "areas",
    "reported",
    "reporting",
    "article",
)

_TRAVEL_BRIEF_TERMS = (
    "travel brief",
    "travel advice",
    "practical advice",
    "should i go",
    "should we go",
    "fine for travel",
    "safe",
    "risk",
    "go/no-go",
    "worth it",
)


def _has_any_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


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

    inc_w = _has_any_term(q, _WEATHER_TERMS)
    inc_n = _has_any_term(q, _NEWS_TERMS)

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


def classify_answer_mode(question: Optional[str], last_reply: Optional[str] = None) -> AnswerMode:
    if not question:
        return "travel_brief"

    q = question.lower()
    last = (last_reply or "").lower()

    if is_trip_planning_question(question) or _has_any_term(q, _TRAVEL_BRIEF_TERMS):
        return "travel_brief"

    has_weather = _has_any_term(q, _WEATHER_TERMS)
    has_news = _has_any_term(q, _NEWS_TERMS)

    # Support short follow-ups like "is that in Vigan?" by inheriting the prior topic.
    if not has_weather and not has_news and any(token in q for token in ("that", "this", "it", "there")):
        if _has_any_term(last, _NEWS_TERMS):
            has_news = True
        elif _has_any_term(last, _WEATHER_TERMS):
            has_weather = True

    if has_news and not has_weather:
        return "news_followup"
    if has_weather and not has_news:
        return "weather_followup"
    return "travel_brief"
