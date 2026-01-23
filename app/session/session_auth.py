# app/session_auth.py
from __future__ import annotations

from itsdangerous import URLSafeSerializer, BadSignature
from fastapi import Header, HTTPException
from app.settings import settings


_serializer = URLSafeSerializer(secret_key=settings.session_secret, salt="smartnews-session-v1")


def sign_session(session_id: str) -> str:
    return _serializer.dumps({"sid": session_id})


def verify_session(session_id: str, session_token: str) -> None:
    try:
        payload = _serializer.loads(session_token)
    except BadSignature:
        raise HTTPException(status_code=401, detail="Invalid session token")

    if not isinstance(payload, dict) or payload.get("sid") != session_id:
        raise HTTPException(status_code=401, detail="Session token mismatch")


def require_session(
    x_session_id: str = Header(..., alias="x-session-id"),
    x_session_token: str = Header(..., alias="x-session-token"),
) -> str:
    verify_session(x_session_id, x_session_token)
    return x_session_id
