from __future__ import annotations
from typing import Dict, Any
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from weather_service import get_weather_line
from news_service import get_news_items
from settings import OPENAI_API_KEY

@tool("get_weather")
def get_weather_tool(place: str) -> str:
    """Return concise weather line for a place."""
    line, err = get_weather_line(place)
    return line if not err else f"ERROR: {err}"

@tool("get_news")
def get_news_tool(place: str) -> str:
    """Return top headlines for a place as bullets with source/date/link."""
    items, err = get_news_items(place)
    if err:
        return f"ERROR: {err}"
    if not items:
        return "No recent news."
    return "\n".join(f"- {h['title']} ({h['source']}, {h['date']}) -> {h['link']}" for h in items)

# Prompt must include {tools}, {input}, and MessagesPlaceholder("agent_scratchpad")
_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise news+weather assistant.\n"
     "You can use these tools:\n{tools}\n"
     "Fetch weather and top headlines for the place and return ONE plain-text paragraph. "
     "If a tool errors, note it briefly and continue."),
    ("human", "Place: {input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

_EXECUTOR: AgentExecutor | None = None

def _build_agent() -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    tools = [get_weather_tool, get_news_tool]
    agent = create_react_agent(llm, tools, _PROMPT)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

def run_agent(place: str) -> Dict[str, Any]:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = _build_agent()
    res = _EXECUTOR.invoke({"input": place})
    final = res["output"] if isinstance(res, dict) and "output" in res else str(res)
    return {"final": final}
