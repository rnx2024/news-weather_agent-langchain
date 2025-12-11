# app/news_service.py
from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta

from app.http_utils import get_json_with_retry
from app.settings import settings
from app.location_resolver import resolve_country_code

log = logging.getLogger(__name__)


# --------------------------
# Date Parsing
# --------------------------
def _parse_serpapi_date(date_str: str) -> datetime | None:
    """
    Parse SerpAPI Google News dates:
    - '1 hour ago'
    - '3 days ago'
    - '12/11/2025'
    - '12/11/2025, 3:00 PM'
    - 'Dec 11, 2025'
    - 'December 11, 2025'
    - '2025-12-11'
    Returns None if parsing fails.
    """

    if not date_str:
        return None

    s = date_str.lower().replace(",", "").strip()
    now = datetime.utcnow()

    # -----------------------------
    # 1. Absolute date formats (try several)
    # -----------------------------
    absolute_formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%b %d %Y",
        "%B %d %Y",
    ]

    for fmt in absolute_formats:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass

    # -----------------------------
    # 2. Absolute date with time (strip time part)
    # -----------------------------
    try:
        # example: "12/11/2025 3:00 PM"
        core = s.split(" ")[0]  # keep only date component
        for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(core, fmt)
            except Exception:
                pass
    except Exception:
        pass

    # -----------------------------
    # 3. Relative formats (X hours ago, X days ago)
    # -----------------------------
    parts = s.split()
    if len(parts) >= 2:
        qty_str, unit = parts[0], parts[1]

        if qty_str.isdigit():
            qty = int(qty_str)

            if "hour" in unit:
                return now - timedelta(hours=qty)
            if "minute" in unit:
                return now - timedelta(minutes=qty)
            if "day" in unit:
                return now - timedelta(days=qty)

    # -----------------------------
    # Unknown format
    # -----------------------------
    return None

# --------------------------
# Fetch News
# --------------------------
def get_news_items(place: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Fetch SerpAPI Google News results:
    - dynamically compute GL (country code)
    - filter only last 7 days
    - return max 3 headlines
    """

    gl_code = resolve_country_code(place)  # e.g., PH, JP, DE, GB, BR
    if gl_code:
        gl_code = gl_code.lower()  # SerpAPI expects lowercase
    else:
        gl_code = "us"  # Safe fallback so it still returns something

    params = {
        "engine": "google_news",
        "q": place,
        "hl": "en",
        "gl": gl_code,                 # <-- dynamic, global
        "api_key": settings.serp_api_key,
    }

    data, err = get_json_with_retry(settings.serpapi_search_url, params)
    if err:
        log.error("SerpAPI error for '%s': %s", place, err)
        return [], err

    results = data.get("news_results") or data.get("organic_results") or []
    max_age_days = 7
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)

    filtered: List[Dict[str, Any]] = []

    for item in results:
        date_raw = item.get("date") or item.get("published")
        parsed_date = _parse_serpapi_date(date_raw)

        if not parsed_date:
            continue

        if parsed_date < cutoff:
            continue

        filtered.append({
            "title": item.get("title", "Untitled"),
            "source": item.get("source", {}).get("name") if isinstance(item.get("source"), dict) else item.get("source"),
            "date": parsed_date.isoformat(),
            "link": item.get("link"),
        })

    # Sort by newest
    filtered.sort(key=lambda x: x["date"], reverse=True)

    # Return only top 3
    return filtered[:3], ""
