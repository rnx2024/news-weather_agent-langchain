# app/news_service.py
from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta, timezone

from app.http_utils import get_json_with_retry
from app.settings import settings
from app.location_resolver import resolve_country_code

log = logging.getLogger(__name__)


# --------------------------
# Date Parsing
# --------------------------
_ABSOLUTE_DATE_FORMATS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%b %d %Y",
    "%B %d %Y",
)

_ABSOLUTE_DATE_CORE_FORMATS = (
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y-%m-%d",
)


def _try_parse_absolute_date(s_clean: str) -> datetime | None:
    for fmt in _ABSOLUTE_DATE_FORMATS:
        try:
            dt = datetime.strptime(s_clean, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _try_parse_absolute_date_with_time(s_clean: str) -> datetime | None:
    # handles: "12/11/2025 3:00 PM" by extracting date part only
    try:
        core = s_clean.split(" ")[0]
    except Exception:
        return None

    for fmt in _ABSOLUTE_DATE_CORE_FORMATS:
        try:
            dt = datetime.strptime(core, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _try_parse_relative(s_lower: str, now: datetime) -> datetime | None:
    # handles: "1 hour ago", "3 days ago", "15 minutes ago"
    parts = s_lower.split()
    if len(parts) < 2:
        return None

    qty_str, unit = parts[0], parts[1]
    if not qty_str.isdigit():
        return None

    qty = int(qty_str)
    if "hour" in unit:
        return now - timedelta(hours=qty)
    if "minute" in unit:
        return now - timedelta(minutes=qty)
    if "day" in unit:
        return now - timedelta(days=qty)
    return None


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

    Returns a timezone-aware UTC datetime, or None if parsing fails.
    """
    if not date_str:
        return None

    s_clean = date_str.replace(",", "").strip()
    if not s_clean:
        return None

    s_lower = s_clean.lower()
    now = datetime.now(timezone.utc)

    dt = _try_parse_absolute_date(s_clean)
    if dt is not None:
        return dt

    dt = _try_parse_absolute_date_with_time(s_clean)
    if dt is not None:
        return dt

    return _try_parse_relative(s_lower, now)


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
        log.error("SerpAPI error for '%s': %s", place, err)
        return [], err

    results = data.get("news_results") or data.get("organic_results") or []

    max_age_days = 7
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    filtered: List[Dict[str, Any]] = []

    for item in results:
        date_raw = item.get("date") or item.get("published")
        parsed_date = _parse_serpapi_date(date_raw)

        if not parsed_date:
            continue
        if parsed_date < cutoff:
            continue

        filtered.append(
            {
                "title": item.get("title", "Untitled"),
                "source": item.get("source", {}).get("name")
                if isinstance(item.get("source"), dict)
                else item.get("source"),
                "date": parsed_date.isoformat(),
                "link": item.get("link"),
            }
        )

    filtered.sort(key=lambda x: x["date"], reverse=True)
    return filtered[:3], ""
