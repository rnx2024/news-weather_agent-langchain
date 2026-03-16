# app/session_cache.py (UPDATED: facade; public function names unchanged)
from __future__ import annotations

from app.session.session_store import (
    get_last_exchange,
    get_pending_journey_question,
    get_last_sent_timestamps,
    get_or_set,
    get_session_state,
    mark_sent,
    mark_tools_called,
    prepare_weather_news,
    set_pending_journey_question,
    should_include,
)

# Keep public names stable
__all__ = [
    "get_session_state",
    "should_include",
    "mark_sent",
    "mark_tools_called",
    "get_last_exchange",
    "get_pending_journey_question",
    "get_last_sent_timestamps",
    "get_or_set",
    "prepare_weather_news",
    "set_pending_journey_question",
]
