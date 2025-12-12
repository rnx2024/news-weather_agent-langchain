# main.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.settings import settings
from app.routes import router as api_router


app = FastAPI(
    title="News & Weather Agent API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

allowed_origins = [origin.strip() for origin in settings.frontend_cors_origin.split(',')]
# ---------------------------------------------------------
# CORS CONFIGURATION
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Mount API routes AFTER adding CORS
# ---------------------------------------------------------
app.include_router(api_router)


@app.get("/", tags=["meta"])
async def root():
    return {
        "name": "News & Weather Agent API",
        "status": "ok",
        "docs": "/docs",
    }
