# app/db.py
from __future__ import annotations

import os
from typing import Optional
import asyncio
import libsql

_LIBSQL_URL = os.environ.get("LIBSQL_URL")
_LIBSQL_TOKEN = os.environ.get("LIBSQL_AUTH_TOKEN")

_client: Optional[libsql.Connection] = None


def get_client() -> libsql.Connection:
    """
    Singleton libSQL client (remote-only).
    """
    global _client

    if _client is None:
        if not _LIBSQL_URL or not _LIBSQL_TOKEN:
            raise RuntimeError("LIBSQL_URL or LIBSQL_AUTH_TOKEN is not set")

        # libsql Python SDK uses a local replica file and syncs with Turso.
        # Keep your env var names unchanged.
        _client = libsql.connect(
            "smartnews.db",
            sync_url=_LIBSQL_URL,
            auth_token=_LIBSQL_TOKEN,
        )

    return _client


async def execute(query: str, args: list | None = None):
    client = get_client()

    def _run():
        return client.execute(query, tuple(args or []))

    return await asyncio.to_thread(_run)


async def fetch_all(query: str, args: list | None = None):
    client = get_client()

    def _run():
        cur = client.execute(query, tuple(args or []))
        return cur.fetchall()

    return await asyncio.to_thread(_run)
