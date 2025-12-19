# weather_service.py
from __future__ import annotations

import logging, requests
from typing import Tuple, Dict, Any, Optional

from app.http_utils import get_json_with_retry
from app.settings import settings

log = logging.getLogger(__name__)


# -------------------------------------------------------------------
# PART 1 — OpenWeather (simple current weather line)
# -------------------------------------------------------------------

def get_weather_raw(place: str) -> Tuple[Dict[str, Any], str]:
    """
    Call OpenWeather current weather endpoint.
    URL is loaded from settings.openweather_current_url.
    """
    return get_json_with_retry(
        settings.openweather_current_url,
        {"q": place, "appid": settings.openweather_api_key, "units": "metric"},
    )


def get_weather_line(place: str) -> Tuple[str, str]:
    """
    Backwards-compatible one-line weather summary.
    """
    data, err = get_weather_raw(place)
    if err:
        # Do not include user-controlled data in returned error strings
        return ("", f"Weather error: {err}")

    name = data.get("name") or place
    wx = (data.get("weather") or [{}])[0]
    desc = wx.get("description") or "n/a"
    temp = (data.get("main") or {}).get("temp", "n/a")

    return (f"{name}: {desc}, {temp}°C", "")


# -------------------------------------------------------------------
# PART 2 — Open-Meteo (rich forecast + risk analysis)
# -------------------------------------------------------------------

# WMO code → readable description
WEATHER_CODE_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

_THUNDERSTORM_CODES = {95, 96, 99}
_HEAVY_RAIN_CODES = {82, 65, 67, 75, 86}
_RAIN_CODES = {51, 53, 55, 56, 57, 61, 63, 66, 71, 73, 77, 80, 81, 85}
_SNOW_CODES = {71, 73, 75, 77, 85, 86}
_FOG_CODES = {45, 48}
_CLEAR_OR_CLOUDY_CODES = {0, 1, 2, 3}


def weather_code_to_text(code: Optional[int]) -> str:
    return WEATHER_CODE_DESCRIPTIONS.get(code, "Unknown conditions")


def classify_weather_code(code: Optional[int]) -> str:
    if code is None:
        return "unknown"
    if code in _THUNDERSTORM_CODES:
        return "thunderstorm"
    if code in _HEAVY_RAIN_CODES:
        return "heavy_rain"
    if code in _RAIN_CODES:
        return "rain"
    if code in _SNOW_CODES:
        return "snow"
    if code in _FOG_CODES:
        return "fog"
    if code in _CLEAR_OR_CLOUDY_CODES:
        return "clear_or_cloudy"
    return "unknown"


def geocode_place(place: str, language: str = "en"):
    try:
        r = requests.get(
            settings.openmeteo_geocode_url,
            params={"name": place, "count": 1, "language": language, "format": "json"},
            timeout=5,
        )
        r.raise_for_status()
        data = r.json() or {}
        results = data.get("results") or []
        if not results:
            # Do not include user-controlled data in returned error strings
            return None, "No geocoding results."

        loc = results[0]
        return {
            "name": loc.get("name"),
            "country": loc.get("country"),
            "latitude": loc.get("latitude"),
            "longitude": loc.get("longitude"),
            "timezone": loc.get("timezone") or "auto",
        }, None
    except Exception as e:
        return None, str(e)


def fetch_openmeteo_forecast(lat: float, lon: float, timezone: str = "auto"):
    try:
        r = requests.get(
            settings.openmeteo_forecast_url,
            params={
                "latitude": lat,
                "longitude": lon,
                "timezone": timezone,
                "current": ",".join(
                    [
                        "temperature_2m",
                        "relative_humidity_2m",
                        "apparent_temperature",
                        "precipitation",
                        "wind_speed_10m",
                        "weather_code",
                        "is_day",
                    ]
                ),
                "daily": ",".join(
                    [
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_sum",
                        "uv_index_max",
                        "wind_speed_10m_max",
                    ]
                ),
                "forecast_days": 2,
            },
            timeout=8,
        )
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def _pick_daily_value(daily: Dict[str, Any], key: str, idx: int):
    arr = daily.get(key) or []
    if isinstance(arr, list) and 0 <= idx < len(arr):
        return arr[idx]
    return None


def get_weather_summary(place: str, horizon: str = "today", language: str = "en"):
    loc, err = geocode_place(place, language)
    if err or not loc:
        return None, err

    raw, werr = fetch_openmeteo_forecast(
        lat=loc["latitude"], lon=loc["longitude"], timezone=loc["timezone"]
    )
    if werr or not raw:
        return None, werr

    current = raw.get("current") or {}
    daily = raw.get("daily") or {}

    idx = 0 if horizon in ("now", "today") else 1

    return {
        "place_label": f"{loc['name']}, {loc['country']}",
        "current": {
            "temp_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "precip_mm": current.get("precipitation"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "weather_code": current.get("weather_code"),
            "weather_text": weather_code_to_text(current.get("weather_code")),
        },
        "day": {
            "label": "today" if idx == 0 else "tomorrow",
            "tmin_c": _pick_daily_value(daily, "temperature_2m_min", idx),
            "tmax_c": _pick_daily_value(daily, "temperature_2m_max", idx),
            "precip_mm": _pick_daily_value(daily, "precipitation_sum", idx),
            "uv_index_max": _pick_daily_value(daily, "uv_index_max", idx),
            "wind_speed_max_kmh": _pick_daily_value(daily, "wind_speed_10m_max", idx),
        },
    }, None
