# app/agent_service.py
from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple, Set

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from app.settings import settings
from app.agent_tools import weather_tool, news_tool, city_risk_tool
from app.agent_prompts import LOCAL_INTELLIGENCE_SYSTEM_PROMPT

# NEW: session memory (Redis-backed)
from app.session_cache import get_last_exchange, should_include, mark_tools_called


# -----------------------------------------------------
# LLM + tools
# -----------------------------------------------------
_llm = ChatOpenAI(
    model=settings.openrouter_model,
    temperature=settings.openrouter_temperature,
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url,
)

tools = [weather_tool, news_tool, city_risk_tool]

# -----------------------------------------------------
# Tool-gated helpers
# -----------------------------------------------------
# cache of gated agents keyed by (include_weather, include_news)
_REACT_APP_CACHE: Dict[Tuple[bool, bool], Any] = {}


def _get_react_app(include_weather: bool, include_news: bool):
    key = (include_weather, include_news)
    app = _REACT_APP_CACHE.get(key)
    if app is not None:
        return app

    gated = [city_risk_tool]  # always available (forces grounding)
    if include_weather:
        gated.append(weather_tool)
    if include_news:
        gated.append(news_tool)

    app = create_react_agent(model=_llm, tools=gated, prompt=LOCAL_INTELLIGENCE_SYSTEM_PROMPT)
    _REACT_APP_CACHE[key] = app
    return app


def _decide_includes(question: Optional[str]) -> Tuple[bool, bool]:
    if not question:
        return True, True

    q = question.lower()
    inc_w = any(
        t in q
        for t in (
            "weather",
            "forecast",
            "temperature",
            "rain",
            "storm",
            "wind",
            "uv",
            "heat",
            "snow",
            "fog",
            "typhoon",
            "hurricane",
        )
    )
    inc_n = any(
        t in q
        for t in (
            "news",
            "headline",
            "headlines",
            "update",
            "updates",
            "disruption",
            "disruptions",
            "incident",
            "incidents",
            "protest",
            "strike",
            "closure",
            "closures",
            "outage",
            "traffic",
            "where",
            "location",
            "locations",
            "area",
            "areas",
        )
    )
    return (inc_w, inc_n) if (inc_w or inc_n) else (False, False)


# -----------------------------------------------------
# Build LangGraph ReAct agent
# -----------------------------------------------------
react_app = create_react_agent(
    model=_llm,
    tools=tools,
    prompt=LOCAL_INTELLIGENCE_SYSTEM_PROMPT,
)


def _build_user_prompt(place: str, question: Optional[str]) -> str:
    if not question:
        return (
            "Provide a concise one-paragraph summary of the current weather and recent news "
            f"for the location: {place}."
        )
    return (
        f"Location: {place}\n"
        f"Question: {question}\n"
        "Answer as ONE concise paragraph, plain text."
    )


# -----------------------------------------------------
# Helper: extract final assistant message from LangGraph state
# -----------------------------------------------------
def _extract_final_message(messages: List[BaseMessage]) -> str:
    final_text = ""
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            final_text = str(msg.content)
    return final_text or ""


# -----------------------------------------------------
# Helper: build debug info from messages (tool calls + results)
# -----------------------------------------------------
def _build_debug(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    debug_steps: List[Dict[str, Any]] = []
    pending_tools: Dict[str, Dict[str, Any]] = {}

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                call_id = tc.get("id")
                if not call_id:
                    continue
                pending_tools[call_id] = {
                    "tool": tc.get("name"),
                    "tool_input": tc.get("args"),
                    "observation": None,
                }
        elif isinstance(msg, ToolMessage):
            call_id = getattr(msg, "tool_call_id", None)
            if call_id and call_id in pending_tools:
                pending_tools[call_id]["observation"] = msg.content

    debug_steps.extend(pending_tools.values())
    return debug_steps


def _extract_called_tools(messages: List[BaseMessage]) -> Set[str]:
    called: Set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name")
                if isinstance(name, str) and name:
                    called.add(name)
    return called


# -----------------------------------------------------
# Public function: run_agent
# -----------------------------------------------------
async def run_agent(
    *,
    session_id: str,
    place: str,
    question: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run the LangGraph ReAct agent with tool gating per request.
    Signature matches your router call and fixes the TypeError.
    """
    user_prompt = _build_user_prompt(place, question)

    # decide tool availability for this turn
    include_weather, include_news = _decide_includes(question)

    # session-aware suppression (Redis)
    q_lc = (question or "").lower()
    force_weather = "weather" in q_lc
    force_news = "news" in q_lc
    allow_weather, allow_news = await should_include(session_id, force_weather, force_news)

    if include_weather and not allow_weather:
        include_weather = False
    if include_news and not allow_news:
        include_news = False

    # small policy hint
    policy_lines: List[str] = ["Policy:", f"- Selected location: {place}"]
    if not include_weather:
        policy_lines.append("- Do NOT call weather_tool or include weather unless explicitly asked.")
    if not include_news:
        policy_lines.append("- Do NOT call news_tool or include news unless explicitly asked.")

    # prior exchange context (1 turn)
    last_user, last_reply = await get_last_exchange(session_id)
    if last_user or last_reply:
        policy_lines.append("- Prior exchange context (most recent only):")
        if last_user:
            policy_lines.append(f"  - User: {last_user}")
        if last_reply:
            policy_lines.append(f"  - Assistant: {last_reply}")

    # ADD additional policies
    policy_lines.extend(
        [
            "- Always produce a one-paragraph risk recommendation for the specified location.",
            "- Use the city_risk_tool each turn to ground your answer on current weather/news signals.",
            "- Do NOT include explicit weather/news text in the final paragraph unless the user asked for it or a new update is available.",
            "- If the user asks about disruptions or 'where' they are, ground the answer using recent news: list up to 3 named places if present, otherwise say 'no specific locations reported'.",
            f"- If the user's question mentions a different place than '{place}', begin with: \"You asked about <other place> but your selected location is {place}. To get updates for <other place>, change the Location.\" Then provide the recommendation for {place} only.",
        ]
    )

    user_prompt = "\n".join(policy_lines) + "\n\n---\n\n" + user_prompt

    # use a gated agent (hard enforcement)
    app = _get_react_app(include_weather=include_weather, include_news=include_news)

    # invoke remains synchronous; awaiting here is fine because function is async
    state: Dict[str, Any] = app.invoke({"messages": [{"role": "user", "content": user_prompt}]})

    messages = state.get("messages", []) or []
    final_text = _extract_final_message(messages)

    # persist session state based on actual tool calls
    called_tools = _extract_called_tools(messages)
    await mark_tools_called(
        session_id,
        tool_names=called_tools,
        user_message=question,
        agent_reply=final_text,
    )

    result: Dict[str, Any] = {"final": final_text}
    if debug:
        result["debug"] = _build_debug(messages)
    return result
