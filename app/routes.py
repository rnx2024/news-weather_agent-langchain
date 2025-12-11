# routes.py
from __future__ import annotations

from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query, Header, Depends
from pydantic import BaseModel

from app.agent_service import run_agent
from app.weather_service import get_weather_line
from app.agent_tools import city_risk_tool
from app.news_service import get_news_items
from app.settings import settings

router = APIRouter()

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
# Endpoints 
# -----------------------------------------------------------

@router.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=AgentResponse, tags=["agent"], dependencies=[Depends(require_api_key)])
async def agent_endpoint(payload: AgentRequest) -> AgentResponse:
    result = run_agent(payload.place, payload.question)
    return AgentResponse(**result)


@router.get("/weather", tags=["weather"], dependencies=[Depends(require_api_key)])
async def weather_endpoint(
    place: str = Query(..., description="City or place name")
) -> Dict[str, str]:
    line, err = get_weather_line(place)
    if err:
        raise HTTPException(status_code=502, detail=err)
    return {"place": place, "summary": line}

@router.get("/news", tags=["news"], dependencies=[Depends(require_api_key)])
async def news_endpoint(
    place: str = Query(..., description="City or topic for news search")
) -> Dict[str, Any]:
    """
    Returns: latest (≤7 days) top 3 headlines for the given place,
    using global location-derived country codes for better relevance.
    """
    headlines, err = get_news_items(place)

    if err:
        # SerpAPI or networking error
        raise HTTPException(status_code=502, detail=f"News retrieval failed: {err}")

    return {
        "place": place,
        "recent_count": len(headlines),
        "items": headlines,
        "note": "Showing only top 3 headlines from last 7 days",
    }

