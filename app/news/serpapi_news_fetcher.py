# app/news/serpapi_news_fetcher.py
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

from app.http.http_utils import get_json_with_retry
from app.location.resolve_country import resolve_country_code
from app.news.serpapi_date_parser import parse_serpapi_date
from app.settings import settings

log = logging.getLogger(__name__)


def fetch_news_items(place: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Fetch SerpAPI Google News results:
    - dynamically compute GL (country code)
    - filter only last 7 days
    - return max 3 headlines
    """
    gl_code = resolve_country_code(place)
    gl_code = gl_code.lower() if gl_code else "us"

    params = {
        "engine": "google_news",
        "q": place,
        "hl": "en",
        "gl": gl_code,
        "api_key": settings.serp_api_key,
    }

    data, err = get_json_with_retry(settings.serpapi_search_url, params)
    if err:
        log.error("SerpAPI request failed")
        return [], err

    results = data.get("news_results") or data.get("organic_results") or []

    max_age_days = 7
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    filtered: List[Dict[str, Any]] = []

    for item in results:
        date_raw = item.get("date") or item.get("published")
        parsed_date = parse_serpapi_date(date_raw)

        if not parsed_date or parsed_date < cutoff:
            continue

        filtered.append(
            {
                "title": item.get("title", "Untitled"),
                "source": item.get("source", {}).get("name")
                if isinstance(item.get("source"), dict)
                else item.get("source"),
                "date": parsed_date.isoformat(),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            }
        )

    filtered.sort(key=lambda x: x["date"], reverse=True)
    return filtered[:3], ""
