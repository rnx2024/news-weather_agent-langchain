from __future__ import annotations

from typing import Annotated, Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Header, Depends, Request
from pydantic import BaseModel

from app.agent.agent_service import run_agent
from app.weather.weather_service import get_weather_line
from app.news.news_service import get_news_items
from app.settings import settings
from app.session.session_auth import require_session, sign_session
from app.tooling.ratelimit import limiter


router = APIRouter()


def require_api_key(
    x_api_key: Annotated[str, Header(..., alias="x-api-key")],
) -> None:
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


class AgentRequest(BaseModel):
    place: str
    question: Optional[str] = None


class AgentResponse(BaseModel):
    final: str


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


@router.post("/chat", tags=["agent"], dependencies=[Depends(require_api_key)])
@limiter.limit("15/minute")
async def agent_endpoint(
    request: Request,
    payload: AgentRequest,
    session_id: Annotated[str, Depends(require_session)],
) -> AgentResponse:
    question = payload.question or ""

    result = await run_agent(
        session_id=session_id,
        place=payload.place,
        question=question,
    )

    return AgentResponse(**result)


@router.get(
    "/weather",
    tags=["weather"],
    dependencies=[Depends(require_api_key)],
    responses={502: {"description": "Weather provider request failed"}},
)
@limiter.limit("15/minute")
async def weather_endpoint(
    request: Request,
    place: Annotated[str, Query(..., description="City or place name")],
) -> Dict[str, str]:
    line, err = get_weather_line(place)
    if err:
        raise HTTPException(status_code=502, detail=err)
    return {"place": place, "summary": line}


@router.get(
    "/news",
    tags=["news"],
    dependencies=[Depends(require_api_key)],
    responses={502: {"description": "News retrieval failed"}},
)
@limiter.limit("15/minute")
async def news_endpoint(
    request: Request,
    place: Annotated[str, Query(..., description="City or topic for news search")],
) -> Dict[str, Any]:
    headlines, err = get_news_items(place)
    if err:
        raise HTTPException(status_code=502, detail="News retrieval failed")

    return {
        "place": place,
        "recent_count": len(headlines),
        "items": headlines,
        "note": "Showing only top 3 headlines from last 7 days",
    }
