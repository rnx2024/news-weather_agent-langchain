# routes.py
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, Header, Depends, Request
from pydantic import BaseModel

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.agent_service import run_agent
from app.weather_service import get_weather_line
from app.news_service import get_news_items
from app.settings import settings
from app.session_auth import require_session, sign_session

# NEW: session cache helpers used by /chat
from app.session_cache import should_include, get_last_exchange, mark_sent


router = APIRouter()

# -----------------------------------------------------------
# Rate limiting (per-IP)
# -----------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

# -----------------------------------------------------------
# API KEY VALIDATION
# -----------------------------------------------------------
def require_api_key(x_api_key: str = Header(..., alias="x-api-key")):
    """
    Validates that the incoming request supplies a correct x-api-key header.
    """
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


# -----------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------
class AgentRequest(BaseModel):
    place: str
    question: Optional[str] = None


class AgentResponse(BaseModel):
    final: str


# -----------------------------------------------------------
# Helpers (to reduce cognitive complexity in /chat)
# -----------------------------------------------------------
def _detect_explicit_asks(question: str) -> Tuple[bool, bool]:
    text_lc = (question or "").lower()
    return ("weather" in text_lc, "news" in text_lc)


def _build_context_lines(last_user: Optional[str], last_reply: Optional[str]) -> List[str]:
    lines: List[str] = []
    if last_user or last_reply:
        lines.append("Previous exchange (most recent):")
        if last_user:
            lines.append(f"- User: {last_user}")
        if last_reply:
            lines.append(f"- Assistant: {last_reply}")
    return lines


def _build_suppression_lines(
    force_weather: bool,
    force_news: bool,
    include_weather: bool,
    include_news: bool,
) -> List[str]:
    lines: List[str] = []
    if force_weather and not include_weather:
        lines.append(
            "Session policy: Weather was already provided recently. Do not repeat weather details; "
            "say it was recently provided and offer to refresh after some time."
        )
    if force_news and not include_news:
        lines.append(
            "Session policy: News was already provided recently. Do not repeat news details; "
            "say it was recently provided and offer to refresh after some time."
        )
    return lines


def _augment_question(question: str, context_lines: List[str], suppression_lines: List[str]) -> str:
    if not context_lines and not suppression_lines:
        return question

    parts: List[str] = []
    if context_lines:
        parts.append("\n".join(context_lines))
    if suppression_lines:
        parts.append("\n".join(suppression_lines))
    parts.append(question)
    return "\n\n".join(parts)


async def _persist_session_state(
    *,
    session_id: str,
    question: str,
    final_text: str,
    force_weather: bool,
    force_news: bool,
    include_weather: bool,
    include_news: bool,
) -> None:
    try:
        await mark_sent(
            session_id,
            weather_sent=bool(force_weather and include_weather),
            news_sent=bool(force_news and include_news),
            user_message=question,
            agent_reply=final_text,
        )
    except Exception:
        # fail-open: chat response should not fail because Redis write failed
        return


# -----------------------------------------------------------
# Endpoints
# -----------------------------------------------------------
@router.get("/health")
@limiter.limit("30/minute")
async def health(request: Request) -> Dict[str, str]:
    return {"status": "ok"}


@router.post("/session", tags=["session"], dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def create_session(request: Request) -> Dict[str, str]:
    from uuid import uuid4

    session_id = str(uuid4())
    session_token = sign_session(session_id)
    return {"session_id": session_id, "session_token": session_token}


@router.post("/chat", response_model=AgentResponse, tags=["agent"], dependencies=[Depends(require_api_key)])
@limiter.limit("15/minute")
async def agent_endpoint(
    request: Request,
    payload: AgentRequest,
    session_id: str = Depends(require_session),
) -> AgentResponse:
    """
    Chat endpoint with Redis-backed session memory:
    - Loads last exchange (one-turn) and injects it as context.
    - Applies session-level suppression policy for explicit weather/news requests.
    - Saves the latest exchange + sent timestamps back to Redis.
    """
    question = payload.question or ""

    force_weather, force_news = _detect_explicit_asks(question)
    include_weather, include_news = await should_include(session_id, force_weather, force_news)

    last_user, last_reply = await get_last_exchange(session_id)
    context_lines = _build_context_lines(last_user, last_reply)

    suppression_lines = _build_suppression_lines(
        force_weather=force_weather,
        force_news=force_news,
        include_weather=include_weather,
        include_news=include_news,
    )

    augmented_question = _augment_question(question, context_lines, suppression_lines)

    result = await run_agent(
        session_id=session_id,
        place=payload.place,
        question=augmented_question,
    )

    await _persist_session_state(
        session_id=session_id,
        question=question,
        final_text=str(result.get("final") or ""),
        force_weather=force_weather,
        force_news=force_news,
        include_weather=include_weather,
        include_news=include_news,
    )

    return AgentResponse(**result)


@router.get("/weather", tags=["weather"], dependencies=[Depends(require_api_key)])
@limiter.limit("15/minute")
async def weather_endpoint(
    request: Request,
    place: str = Query(..., description="City or place name"),
) -> Dict[str, str]:
    line, err = get_weather_line(place)
    if err:
        raise HTTPException(status_code=502, detail=err)
    return {"place": place, "summary": line}


@router.get("/news", tags=["news"], dependencies=[Depends(require_api_key)])
@limiter.limit("15/minute")
async def news_endpoint(
    request: Request,
    place: str = Query(..., description="City or topic for news search"),
) -> Dict[str, Any]:
    """
    Returns: latest (≤7 days) top 3 headlines for the given place,
    using global location-derived country codes for better relevance.
    """
    headlines, err = get_news_items(place)

    if err:
        raise HTTPException(status_code=502, detail="News retrieval failed")

    return {
        "place": place,
        "recent_count": len(headlines),
        "items": headlines,
        "note": "Showing only top 3 headlines from last 7 days",
    }
