# main.py
from __future__ import annotations

from fastapi import FastAPI

from app.routes import router as api_router


app = FastAPI(
    title="News & Weather Agent API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# Mount the API routes
app.include_router(api_router)


@app.get("/", tags=["meta"])
async def root():
    """
    Root/landing endpoint
    """
    return {
        "name": "News & Weather Agent API",
        "status": "ok",
        "docs": "/docs",
    }
