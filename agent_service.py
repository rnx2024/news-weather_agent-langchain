from __future__ import annotations
from typing import Dict, Any, List
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from weather_service import get_weather_line
from news_service import get_news_items
from settings import OPENAI_API_KEY

@tool("get_weather", return_direct=False)
def get_weather_tool(place: str) -> str:
    """Get a concise weather line for a place. Input: place name."""
    line, err = get_weather_line(place)
    if err:
        return f"ERROR: {err}"
    return line

@tool("get_news", return_direct=False)
def get_news_tool(place: str) -> str:
    """Get top 5 headlines for a place as bullet lines with source and date. Input: place name."""
    items, err = get_news_items(place)
    if err:
        return f"ERROR: {err}"
    if not items:
        return "No recent news."
    bullets = []
    for h in items:
        bullets.append(f"- {h['title']} ({h['source']}, {h['date']}) -> {h['link']}")
    return "\n".join(bullets)

_SYSTEM = """You are a precise news+weather assistant.
Use tools to fetch current weather and top headlines for the requested place.
Return a concise paragraph that merges weather and the most relevant 2–3 headlines.
If a tool returns ERROR, state which source failed and still answer with what you have.
No markdown lists in the final answer; plain text only."""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "Place: {place}\nTask: Produce a single-paragraph brief."),
])

def _build_agent() -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    tools = [get_weather_tool, get_news_tool]
    agent = create_react_agent(llm, tools, _PROMPT)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, return_intermediate_steps=True)

_EXECUTOR = None

def run_agent(place: str) -> Dict[str, Any]:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = _build_agent()
    result = _EXECUTOR.invoke({"place": place})
    final = result.get("output", "")
    steps = result.get("intermediate_steps", [])
    trace: List[str] = []
    for action, observation in steps:
        trace.append(f"Tool: {getattr(action, 'tool', '?')} | Input: {getattr(action, 'tool_input', '')}\nObs: {observation}")
    return {"final": final, "trace": "\n\n".join(trace)}
