# weather_service.py (UPDATED: facade; public function names unchanged)
from __future__ import annotations

from typing import Optional

from app.weather.openmeteo_provider import (
    classify_weather_code,
    get_weather_summary,
    weather_code_to_text,
)
from app.weather.openweather_provider import get_weather_line, get_weather_raw

# Re-exported names remain available to callers:
# - get_weather_raw
# - get_weather_line
# - get_weather_summary
# - classify_weather_code
# - weather_code_to_text

__all__ = [
    "get_weather_raw",
    "get_weather_line",
    "get_weather_summary",
    "classify_weather_code",
    "weather_code_to_text",
]
