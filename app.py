import requests
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Weather + News", page_icon="🛰")

# --- Secrets (Streamlit) ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]

# --- LLM ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# --- Tools (no geocoding) ---
@tool
def weather_by_place(place: str) -> str:
    """Current weather using OpenWeather by place name."""
    r = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": place, "appid": OPENWEATHER_API_KEY, "units": "metric"},
        timeout=15,
    )
    if r.status_code != 200:
        return f"Weather error HTTP {r.status_code} for {place}"
    j = r.json() or {}
    name = j.get("name", place)
    wx = (j.get("weather") or [{}])[0]
    main = j.get("main") or {}
    desc = wx.get("description", "n/a")
    temp = main.get("temp", "n/a")
    return f"{name}: {desc}, {temp}°C"

@tool
def news_by_place(place: str) -> str:
    """Latest headlines via SERPAPI Google News."""
    r = requests.get(
        "https://serpapi.com/search.json",
        params={"engine": "google_news", "q": place, "hl": "en", "gl": "ph", "api_key": SERPAPI_API_KEY},
        timeout=15,
    )
    if r.status_code != 200:
        return f"News error HTTP {r.status_code} for {place}"
    data = r.json() or {}
    items = (data.get("news_results") or [])[:5]
    if not items:
        return f"No recent news for {place}."
    lines = []
    for n in items:
        title = n.get("title", "Untitled")
        src = n.get("source", "")
        date = n.get("date", "")
        link = n.get("link", "")
        lines.append(f"- {title} ({src}, {date}) -> {link}")
    return f"Latest news for {place}:\n" + "\n".join(lines)

tools = [weather_by_place, news_by_place]

# --- Agent ---
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Do this every time:\n"
     "1) call weather_by_place with the user text;\n"
     "2) call news_by_place with the same text;\n"
     "Then summarize briefly."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- UI ---
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🛰 Weather + News")
LOCATIONS = [
    "Ilocos Sur", "Vigan City", "Laoag City", "La Union", "Dagupan City",
    "Manila", "Cebu City", "Davao City", "Baguio", "Texas", "India"
]
loc = st.selectbox("Choose location", LOCATIONS, index=0)

if st.button("Get Update"):
    st.session_state.history.append(HumanMessage(content=loc))
    out = executor.invoke({"input": loc, "chat_history": st.session_state.history})
    ans = out["output"]
    st.session_state.history.append(AIMessage(content=ans))
    st.subheader("Answer")
    st.write(ans)
