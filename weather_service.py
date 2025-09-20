from __future__ import annotations
from typing import Tuple, Dict, Any
import streamlit as st
from http_utils import get_json_with_retry
from settings import OPENWEATHER_API_KEY

@st.cache_data(ttl=180)
def get_weather_raw(place: str) -> Tuple[Dict[str, Any], str]:
    return get_json_with_retry(
        "https://api.openweathermap.org/data/2.5/weather",
        {"q": place, "appid": OPENWEATHER_API_KEY, "units": "metric"},
    )

def get_weather_line(place: str) -> Tuple[str, str]:
    data, err = get_weather_raw(place)
    if err:
        return ("", f"Weather error for '{place}': {err}")
    name = data.get("name") or place
    wx = (data.get("weather") or [{}])[0]
    desc = wx.get("description") or "n/a"
    temp = (data.get("main") or {}).get("temp", "n/a")
    return (f"{name}: {desc}, {temp}°C", "")
