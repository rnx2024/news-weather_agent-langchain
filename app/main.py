# main.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router as api_router


app = FastAPI(
    title="News & Weather Agent API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------
# CORS CONFIGURATION
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # allow all origins (local + deployed)
    allow_credentials=False,          # safer since you use x-api-key, not cookies
    allow_methods=["*"],              # allow POST /chat
    allow_headers=["*"],              # allow x-api-key, content-type…
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
