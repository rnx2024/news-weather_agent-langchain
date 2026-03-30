from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import httpx

from app.settings import settings
from app.weather.openmeteo_provider import geocode_place

log = logging.getLogger(__name__)

DEFAULT_PROFILES = (
    "driving-car",
    "cycling-regular",
    "foot-walking",
)

PROFILE_LABELS = {
    "driving-car": "car",
    "cycling-regular": "bike",
    "foot-walking": "walk",
}


def _build_point_label(loc: Dict[str, Any], fallback: str) -> str:
    name = str(loc.get("name") or "").strip()
    country = str(loc.get("country") or "").strip()
    if name and country:
        return f"{name}, {country}"
    if name:
        return name
    return fallback


def _midpoint(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
    return (lat1 + lat2) / 2.0, (lon1 + lon2) / 2.0


def _fetch_route(profile: str, start: Tuple[float, float], end: Tuple[float, float]) -> Tuple[Dict[str, Any] | None, str]:
    if not settings.ors_api:
        return None, "missing_ors_api"

    url = f"{settings.ors_directions_url.rstrip('/')}/{profile}"
    params = {
        "start": f"{start[1]},{start[0]}",
        "end": f"{end[1]},{end[0]}",
        "format": "geojson",
    }
    headers = {"Authorization": settings.ors_api}

    try:
        response = httpx.get(url, params=params, headers=headers, timeout=10.0)
        response.raise_for_status()
    except httpx.TimeoutException:
        return None, "timeout"
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else "http_error"
        return None, str(status)
    except httpx.RequestError as exc:
        return None, str(exc)

    try:
        data = response.json()
    except ValueError:
        return None, "invalid_json"

    features = data.get("features") or []
    if not features:
        return None, "no_route"
    summary = (features[0].get("properties") or {}).get("summary") or {}
    distance_m = summary.get("distance")
    duration_s = summary.get("duration")
    if not isinstance(distance_m, (int, float)) or not isinstance(duration_s, (int, float)):
        return None, "invalid_summary"

    return {
        "profile": profile,
        "mode": PROFILE_LABELS.get(profile, profile),
        "distance_km": round(distance_m / 1000.0, 2),
        "duration_min": round(duration_s / 60.0, 1),
        "raw_distance_m": distance_m,
        "raw_duration_s": duration_s,
    }, ""


def plan_route(
    origin: str,
    destination: str,
    profiles: Tuple[str, ...] | None = None,
) -> Tuple[Dict[str, Any] | None, str]:
    if not settings.ors_api:
        return None, "missing_ors_api"

    origin_loc, oerr = geocode_place(origin)
    if oerr or not origin_loc:
        return None, f"origin_geocode_failed:{oerr or 'unknown'}"

    dest_loc, derr = geocode_place(destination)
    if derr or not dest_loc:
        return None, f"destination_geocode_failed:{derr or 'unknown'}"

    lat1 = origin_loc.get("latitude")
    lon1 = origin_loc.get("longitude")
    lat2 = dest_loc.get("latitude")
    lon2 = dest_loc.get("longitude")
    if not all(isinstance(v, (int, float)) for v in (lat1, lon1, lat2, lon2)):
        return None, "invalid_geocoding_coordinates"

    start = (float(lat1), float(lon1))
    end = (float(lat2), float(lon2))
    midpoint_lat, midpoint_lon = _midpoint(start[0], start[1], end[0], end[1])

    chosen_profiles = profiles or DEFAULT_PROFILES
    routes: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}

    for profile in chosen_profiles:
        route, err = _fetch_route(profile, start, end)
        if route:
            routes.append(route)
        else:
            errors[profile] = err or "route_error"

    if not routes:
        return None, "no_routes"

    best = min(routes, key=lambda r: r.get("raw_duration_s") or float("inf"))
    return {
        "origin": {
            "label": _build_point_label(origin_loc, origin),
            "lat": start[0],
            "lon": start[1],
        },
        "destination": {
            "label": _build_point_label(dest_loc, destination),
            "lat": end[0],
            "lon": end[1],
        },
        "routes": routes,
        "best_profile": best.get("profile"),
        "best_mode": best.get("mode"),
        "best_distance_km": best.get("distance_km"),
        "best_duration_min": best.get("duration_min"),
        "midpoint": {"lat": midpoint_lat, "lon": midpoint_lon},
        "errors": errors,
        "source": "openrouteservice",
    }, ""
