# routes.py
from __future__ import annotations

from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query, Header, Depends, Request
from pydantic import BaseModel

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.agent_service import run_agent
from app.weather_service import get_weather_line
from app.agent_tools import city_risk_tool
from app.news_service import get_news_items
from app.settings import settings
from app.session_auth import require_session, sign_session 
from app.admin_clean import purge_by_patterns, should_purge

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
    session_id: str = Depends(require_session),  # <-- ADD THIS
) -> AgentResponse:
    result = await run_agent(                     # <-- AWAIT and pass session_id
        session_id=session_id,
        place=payload.place,
        question=payload.question,
    )
    return AgentResponse(**result)


@router.get("/weather", tags=["weather"], dependencies=[Depends(require_api_key)])
@limiter.limit("15/minute")
async def weather_endpoint(
    request: Request,
    place: str = Query(..., description="City or place name")
) -> Dict[str, str]:
    line, err = get_weather_line(place)
    if err:
        raise HTTPException(status_code=502, detail=err)
    return {"place": place, "summary": line}

@router.get("/news", tags=["news"], dependencies=[Depends(require_api_key)])
@limiter.limit("15/minute")
async def news_endpoint(
    request: Request,
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

@router.post("/admin/purge-cache", tags=["admin"], dependencies=[Depends(require_api_key)])
@limiter.limit("1/minute")
async def purge_cache(request: Request):
    from app.admin_clean import purge_by_patterns
    deleted = await purge_by_patterns()
    return {"status": "ok", "deleted": deleted}

@router.get("/admin/should-purge", tags=["admin"], dependencies=[Depends(require_api_key)])
@limiter.limit("5/minute")
async def get_should_purge(request: Request):
    from app.admin_clean import should_purge
    return {"should_purge": await should_purge()}

@router.get("/admin/memory", tags=["admin"], dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def admin_memory(
    request: Request,
    threshold_mb: float = Query(28.0, ge=1.0, description="Soft limit for UI status")
) -> Dict[str, Any]:
    if redis is None:
        raise HTTPException(status_code=500, detail="Redis not initialized")

    # 1) Authoritative server memory from INFO MEMORY
    info = await redis.info(section="memory")
    used_bytes = int(info.get("used_memory", 0))                 # includes overhead
    dataset_bytes = int(info.get("used_memory_dataset", 0))      # data (less overhead)
    peak_bytes = int(info.get("used_memory_peak", 0))

    # 2) Optional breakdown for your namespaces via MEMORY USAGE (can be slower on large DBs)
    total_app_bytes = 0
    total_keys = 0
    for pattern in ("sess:*", "cache:*"):
        cursor = 0
        while True:
            cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=200)
            total_keys += len(keys)
            if keys:
                # MEMORY USAGE returns bytes for key+value object (excludes global allocator frag)
                usages = await redis.pipeline().execute_command(*sum([["MEMORY", "USAGE", k] for k in keys], []))  # type: ignore
                # The pipeline trick above may not be supported by all clients; safe fallback:
                # for k in keys:
                #     b = await redis.memory_usage(k)
                #     total_app_bytes += int(b or 0)
                if isinstance(usages, list):
                    for b in usages:
                        total_app_bytes += int(b or 0)
            if cursor == 0:
                break

    def _human(b: int) -> str:
        for unit in ("B","KB","MB","GB","TB"):
            if b < 1024.0:
                return f"{b:.2f} {unit}"
            b /= 1024.0
        return f"{b:.2f} PB"

    # Use INFO used_memory for % since it matches the dashboard
    limit_bytes = int(threshold_mb * 1024 * 1024)
    pct = (used_bytes / limit_bytes) * 100.0 if limit_bytes > 0 else 0.0
    status = "Free Memory" if pct < 70 else ("Stable" if pct < 90 else "High")

    return {
        "used_bytes": used_bytes,
        "used_human": _human(used_bytes),
        "dataset_bytes": dataset_bytes,
        "dataset_human": _human(dataset_bytes),
        "peak_bytes": peak_bytes,
        "peak_human": _human(peak_bytes),
        "app_keys_bytes": total_app_bytes,
        "app_keys_human": _human(total_app_bytes),
        "app_keys_count": total_keys,
        "threshold_mb": threshold_mb,
        "percent_of_threshold": round(pct, 2),
        "status": status,
        "note": "percent_of_threshold uses INFO used_memory to match Redis Cloud dashboard."
    }