# app/location_resolver.py
from __future__ import annotations
import httpx
from typing import Optional
from app.settings import settings

def resolve_country_code(place: str) -> Optional[str]:
    """
    Resolve a city/place name into a 2-letter country code (ISO-3166)
    using Open-Meteo geocoding.
    Returns None if the place cannot be resolved.
    """
    try:
        r = httpx.get(
            settings.openmeteo_geocode_url,
            params={"name": place, "count": 1},
            timeout=5.0
        )
        if r.status_code != 200:
            return None

        data = r.json()
        results = data.get("results")
        if not results:
            return None

        return results[0].get("country_code", None)
    except Exception:
        return None
