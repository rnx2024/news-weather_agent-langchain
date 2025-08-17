# app_geo_weather.py
import os, requests, re
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Geo → Weather + News", page_icon="🛰")
load_dotenv("secret_key")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# --- tools ---
@tool
def geocode_place(place: str) -> dict:
    """Return coordinates via OpenWeather Geocoding. Input: place string."""
    r = requests.get(
        "https://api.openweathermap.org/geo/1.0/direct",
        params={"q": place, "limit": 1, "appid": OPENWEATHER_API_KEY},
        timeout=15,
    )
    arr = r.json() if r.status_code == 200 else []
    if not arr:
        return {"error": f"no geocode match for '{place}'", "query": place}
    top = arr[0]
    return {
        "lat": float(top["lat"]),
        "lon": float(top["lon"]),
        "name": top.get("name", place),
        "country": top.get("country", "")
    }


@tool
def weather_by_coords(lat: float, lon: float) -> str:
    """Weather by coordinates using OpenWeather. Inputs: lat, lon."""
    r = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"},
        timeout=15,
    )
    j = r.json()
    name = j.get("name", f"{lat},{lon}")
    desc = j["weather"][0]["description"]
    temp = j["main"]["temp"]
    return f"{name}: {desc}, {temp}°C"

@tool
def news_by_place(place: str) -> str:
    """Latest headlines via SERPAPI Google News. Input: place string."""
    q = _clean_place(place)
    r = requests.get(
        "https://serpapi.com/search.json",
        params={"engine": "google_news", "q": q, "hl": "en", "gl": "ph", "api_key": SERPAPI_API_KEY},
        timeout=15,
    )
    data = r.json()
    items = (data.get("news_results") or [])[:5]
    if not items:
        return f"No recent news for {q}."
    lines = []
    for n in items:
        title = n.get("title", "Untitled")
        src = n.get("source", "")
        date = n.get("date", "")
        link = n.get("link", "")
        lines.append(f"- {title} ({src}, {date}) -> {link}")
    return f"Latest news for {q}:\n" + "\n".join(lines)

tools = [geocode_place, weather_by_coords, news_by_place]

# --- agent ---
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Do this every time: "
     "1) call geocode_place with the user text to get lat and lon; "
     "2) call weather_by_coords with lat and lon; "
     "3) call news_by_place with the original user text; "
     "then summarize briefly."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- UI ---
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🛰 Geocode → Weather + News")

LOCATIONS = [
    "Ilocos Sur", "Vigan City", "Laoag City", "La Union", "Dagupan City", "Manila", "Cebu City", "Davao City", "Baguio",
    "Texas", "India"
]
loc = st.selectbox("Choose location", LOCATIONS, index=0)
q = f"Weather in {loc}"

if st.button("Get Weather Update"):
    st.session_state.history.append(HumanMessage(content=q))
    out = executor.invoke({"input": q, "chat_history": st.session_state.history})
    ans = out["output"]
    st.session_state.history.append(AIMessage(content=ans))
    st.subheader("Answer")
    st.write(ans)
