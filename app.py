import time
from typing import Dict, List, Tuple
import requests
from requests.exceptions import RequestException
import streamlit as st
from langchain_openai import ChatOpenAI

# ---- Page ----
st.set_page_config(page_title="Weather + News", page_icon="🛰")
st.markdown(
    """
    <style>
    .block-container { max-width: 820px; margin: auto; padding-top: 1rem; }
    .dim { opacity:.8; font-size:0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Secrets (fail-fast, but with user-friendly message) ----
def _require_secret(name: str) -> str:
    val = st.secrets.get(name)
    if not val:
        st.error(f"Missing secret: {name}")
        st.stop()
    return val

OPENAI_API_KEY     = _require_secret("OPENAI_API_KEY")
OPENWEATHER_API_KEY = _require_secret("OPENWEATHER_API_KEY")
SERPAPI_API_KEY     = _require_secret("SERPAPI_API_KEY")

# ---- HTTP helper with retry/backoff ----
def _get_json_with_retry(url: str, params: Dict, tries: int = 2, timeout: int = 15) -> Tuple[Dict, str]:
    """
    Returns (json, error). On success: (dict, ""). On failure: ({}, message).
    """
    last_err = ""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return (r.json() or {}, "")
            last_err = f"HTTP {r.status_code}"
        except RequestException as e:
            last_err = str(e)
        time.sleep(0.5 * (i + 1))
    return ({}, last_err)

# ---- Data sources ----
@st.cache_data(ttl=180)
def get_weather(place: str) -> Tuple[str, str]:
    data, err = _get_json_with_retry(
        "https://api.openweathermap.org/data/2.5/weather",
        {"q": place, "appid": OPENWEATHER_API_KEY, "units": "metric"},
    )
    if err:
        return ("", f"Weather error for '{place}': {err}")
    name = data.get("name") or place
    wx = (data.get("weather") or [{}])[0]
    desc = wx.get("description") or "n/a"
    temp = (data.get("main") or {}).get("temp", "n/a")
    return (f"{name}: {desc}, {temp}°C", "")

@st.cache_data(ttl=180)
def get_news(place: str) -> Tuple[List[Dict], str]:
    data, err = _get_json_with_retry(
        "https://serpapi.com/search.json",
        {"engine": "google_news", "q": place, "hl": "en", "gl": "ph", "api_key": SERPAPI_API_KEY},
    )
    if err:
        return ([], f"News error for '{place}': {err}")
    items = (data.get("news_results") or [])[:5]
    results = []
    for n in items:
        results.append({
            "title": n.get("title") or "Untitled",
            "source": n.get("source") or "",
            "date": n.get("date") or "",
            "link": n.get("link") or "",
        })
    return (results, "")

# ---- Optional LLM summary (deterministic) ----
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

def summarize(place: str, weather_line: str, headlines: List[Dict]) -> str:
    head_txt = "\n".join([f"- {h['title']} ({h['source']}, {h['date']})" for h in headlines]) or "No recent news."
    prompt = (
        "Summarize concisely:\n"
        f"Location: {place}\n"
        f"Weather: {weather_line or 'n/a'}\n"
        "News:\n"
        f"{head_txt}\n"
        "Output: One paragraph, plain text."
    )
    return llm.invoke(prompt).content

# ---- UI ----
st.title("🛰 Weather + News")

LOCATIONS = [
    "Vigan City", "Laoag City", "Candon City", "San Fernando City, La Union", "Dagupan City", "Lingayen",
    "Manila", "Cebu City", "Davao City", "Baguio City", "Texas", "India"
]

col1, col2 = st.columns([2,1])
with col1:
    loc = st.selectbox("Choose location", LOCATIONS, index=0)
with col2:
    do_summary = st.toggle("LLM summary", value=True)

if st.button("Get Updates", type="primary"):
    loc_sanitized = (loc or "").strip()
    if not loc_sanitized:
        st.warning("Select a location.")
        st.stop()

    with st.spinner("Fetching..."):
        weather_line, w_err = get_weather(loc_sanitized)
        headlines, n_err = get_news(loc_sanitized)

    if w_err:
        st.error(w_err)
    if n_err:
        st.error(n_err)

    st.subheader("Weather")
    if weather_line:
        st.write(weather_line)
    else:
        st.write("n/a")

    st.subheader("News")
    if headlines:
        for h in headlines:
            # Markdown link + meta
            st.markdown(f"- [{h['title']}]({h['link']})  \n  <span class='dim'>{h['source']} • {h['date']}</span>", unsafe_allow_html=True)
    else:
        st.write("No recent news.")

    if do_summary:
        try:
            st.subheader("Summary")
            st.write(summarize(loc_sanitized, weather_line, headlines))
        except Exception as e:
            st.info("Summary unavailable.")
            # Optionally log e on server side
