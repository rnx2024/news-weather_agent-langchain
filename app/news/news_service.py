from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.news.serpapi_news_fetcher import fetch_news_items, search_news_items


def get_news_items(place: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Backwards-compatible facade for news retrieval.
    """
    return fetch_news_items(place)


def search_news(query: str, place_hint: str | None = None) -> Tuple[List[Dict[str, Any]], str]:
    """
    Targeted follow-up news search that keeps the same normalized response
    shape as the default place-based news lookup.
    """
    return search_news_items(query, place_hint)
