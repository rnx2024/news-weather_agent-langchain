# app/agent_policy.py
from __future__ import annotations

import re
from typing import Literal, Optional, Tuple


AnswerMode = Literal["travel_brief", "news_followup", "weather_followup", "journey_planning"]


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

_JOURNEY_TERMS = (
    "continue my trip",
    "continue the trip",
    "continue travelling",
    "continue traveling",
    "on the way",
    "along the way",
    "getting there",
    "get there",
    "travel to ",
    "trip to ",
    "best route",
    "best way",
    "best transpo",
    "best transport",
    "how should i get",
    "how do i get",
    "which route",
    "which transport",
)

_ROUTE_TRANSPORT_TERMS = (
    "best route",
    "best way",
    "best transpo",
    "best transport",
    "which route",
    "which transport",
    "how should i get",
    "how do i get",
)

_ORIGIN_QUESTION_TERMS = (
    "where are you traveling from",
    "where are you coming from",
    "what is your departure location",
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

    if is_trip_planning_question(q) or is_journey_planning_question(q):
        return True, True

    return ("weather" in q_lc, "news" in q_lc)


def classify_answer_mode(question: Optional[str], last_reply: Optional[str] = None) -> AnswerMode:
    if not question:
        return "travel_brief"

    q = question.lower()
    last = (last_reply or "").lower()

    if _has_any_term(last, _ORIGIN_QUESTION_TERMS) and extract_origin(question):
        return "journey_planning"

    if is_journey_planning_question(question):
        return "journey_planning"

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
        elif _has_any_term(last, _ORIGIN_QUESTION_TERMS):
            return "journey_planning"

    if has_news and not has_weather:
        return "news_followup"
    if has_weather and not has_news:
        return "weather_followup"
    return "travel_brief"


def is_journey_planning_question(q: Optional[str]) -> bool:
    if not q:
        return False

    s = q.lower()
    if _has_any_term(s, _JOURNEY_TERMS):
        return True

    has_route_phrase = bool(re.search(r"\bfrom\s+\S+", s)) and bool(re.search(r"\bto\s+\S+", s))
    return has_route_phrase and any(token in s for token in ("trip", "travel", "route", "transport", "transpo"))


def asks_route_or_transport(question: Optional[str]) -> bool:
    if not question:
        return False
    return _has_any_term(question.lower(), _ROUTE_TRANSPORT_TERMS)


def needs_origin_clarification(question: Optional[str], last_reply: Optional[str] = None) -> bool:
    if not question:
        return False
    q = question.lower()
    last = (last_reply or "").lower()
    if not is_journey_planning_question(question) and not _has_any_term(last, _ORIGIN_QUESTION_TERMS):
        return False
    return extract_origin(question) is None


def extract_origin(question: Optional[str]) -> str | None:
    if not question:
        return None

    q = " ".join(question.strip().split())
    if not q:
        return None

    patterns = (
        r"\bfrom\s+(.+?)\s+\bto\b",
        r"\btraveling from\s+(.+?)(?:\s+\bto\b|$)",
        r"\btravelling from\s+(.+?)(?:\s+\bto\b|$)",
        r"\bcoming from\s+(.+?)(?:\s+\bto\b|$)",
        r"\bdeparting from\s+(.+?)(?:\s+\bto\b|$)",
        r"^\s*from\s+(.+)$",
    )

    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if not match:
            continue
        origin = _clean_location_fragment(match.group(1))
        if origin:
            return origin

    return None


def _clean_location_fragment(fragment: str) -> str | None:
    cleaned = re.split(r"[?.!,;]| by | via | using ", fragment, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    cleaned = re.sub(r"^(the)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or None
