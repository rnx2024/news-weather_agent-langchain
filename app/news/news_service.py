from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.news.serpapi_news_fetcher import fetch_news_items


def get_news_items(place: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Backwards-compatible facade for news retrieval.
    """
    return fetch_news_items(place)
