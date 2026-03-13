from __future__ import annotations


def normalize_text(value: str | None) -> str:
    return (value or "").strip().lower()
