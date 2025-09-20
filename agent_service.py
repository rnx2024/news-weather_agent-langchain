from __future__ import annotations
from typing import Dict, Any, List
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from weather_service import get_weather_line
from news_service import get_news_items
from settings import OPENAI_API_KEY

@tool("get_weather")
def get_weather_tool(place: str) -> str:
    """Get a concise weather line for a place. Input: place name."""
    line, err = get_weather_line(place)
    return line if not err else f"ERROR: {err}"

@tool("get_news")
def get_news_tool(place: str) -> str:
    """Get top 5 headlines for a place as bullet lines with source and date. Input: place name."""
    items, err = get_news_items(place)
    if err:
        return f"ERROR: {err}"
    if not items:
        return "No recent news."
    return "\n".join(f"- {h['title']} ({h['source']}, {h['date']}) -> {h['link']}" for h in items)

def _build_agent() -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    tools = [get_weather_tool, get_news_tool]
    agent = create_react_agent(llm, tools)  # use default, compatible prompt
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

_EXECUTOR: AgentExecutor | None = None

def run_agent(place: str) -> Dict[str, Any]:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = _build_agent()
    result = _EXECUTOR.invoke({"input": place})
    return {"final": result, "trace": ""}  # trace omitted per your UI
