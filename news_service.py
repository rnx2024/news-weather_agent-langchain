from __future__ import annotations
from typing import Dict, List, Tuple
import streamlit as st
from http_utils import get_json_with_retry
from settings import SERPAPI_API_KEY

@st.cache_data(ttl=180)
def get_news_raw(place: str) -> Tuple[Dict, str]:
    return get_json_with_retry(
        "https://serpapi.com/search.json",
        {"engine": "google_news", "q": place, "hl": "en", "gl": "ph", "api_key": SERPAPI_API_KEY},
    )

def get_news_items(place: str) -> Tuple[List[Dict], str]:
    data, err = get_news_raw(place)
    if err:
        return ([], f"News error for '{place}': {err}")
    items = (data.get("news_results") or [])[:5]
    out: List[Dict] = []
    for n in items:
        out.append({
            "title": n.get("title") or "Untitled",
            "source": n.get("source") or "",
            "date": n.get("date") or "",
            "link": n.get("link") or "",
        })
    return (out, "")
