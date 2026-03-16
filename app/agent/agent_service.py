# app/agent_service.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from app.settings import settings
from app.agent.agent_tools import weather_tool, news_tool, city_risk_tool, travel_brief_tool
from app.agent.agent_prompts import LOCAL_INTELLIGENCE_SYSTEM_PROMPT

# session memory (Redis-backed)
from app.session.session_cache import get_last_exchange, should_include, mark_tools_called
from app.agent.agent_policy import decide_tool_includes, detect_force_signals


# -----------------------------------------------------
# LLM + tools
# -----------------------------------------------------
_llm = ChatOpenAI(
    model=settings.openrouter_model,
    temperature=settings.openrouter_temperature,
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url,
)

# -----------------------------------------------------
# Tool-gated helpers
# -----------------------------------------------------
_REACT_APP_CACHE: Dict[Tuple[bool, bool], Any] = {}


def _get_react_app(include_weather: bool, include_news: bool):
    key = (include_weather, include_news)
    app = _REACT_APP_CACHE.get(key)
    if app is not None:
        return app

    gated = [travel_brief_tool, city_risk_tool]
    if include_weather:
        gated.append(weather_tool)
    if include_news:
        gated.append(news_tool)

    app = create_react_agent(model=_llm, tools=gated, prompt=LOCAL_INTELLIGENCE_SYSTEM_PROMPT)
    _REACT_APP_CACHE[key] = app
    return app


def _build_user_prompt(place: str, question: Optional[str]) -> str:
    if not question:
        return (
            "Provide a concise travel brief for the destination below. Focus on travel conditions, likely disruptions, "
            f"and what matters most for someone going there today: {place}."
        )
    return (
        f"Location: {place}\n"
        f"Question: {question}\n"
        "Answer as ONE concise travel-oriented paragraph, plain text."
    )


def _extract_final_message(messages: List[BaseMessage]) -> str:
    final_text = ""
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            final_text = str(msg.content)
    return final_text or ""


def _collect_tool_calls(messages: List[BaseMessage]) -> Dict[str, Dict[str, Any]]:
    pending: Dict[str, Dict[str, Any]] = {}
    for msg in messages:
        if not (isinstance(msg, AIMessage) and msg.tool_calls):
            continue
        for tc in msg.tool_calls:
            call_id = tc.get("id")
            if not call_id:
                continue
            pending[call_id] = {
                "tool": tc.get("name"),
                "tool_input": tc.get("args"),
                "observation": None,
            }
    return pending


def _attach_tool_observations(messages: List[BaseMessage], pending: Dict[str, Dict[str, Any]]) -> None:
    if not pending:
        return
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        call_id = getattr(msg, "tool_call_id", None)
        if call_id and call_id in pending:
            pending[call_id]["observation"] = msg.content


def _build_debug(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    pending_tools = _collect_tool_calls(messages)
    _attach_tool_observations(messages, pending_tools)
    return list(pending_tools.values())


def _extract_called_tools(messages: List[BaseMessage]) -> Set[str]:
    called: Set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name")
                if isinstance(name, str) and name:
                    called.add(name)
    return called


def _extract_tool_outputs(messages: List[BaseMessage]) -> Dict[str, str]:
    tool_names_by_call_id: Dict[str, str] = {}
    outputs: Dict[str, str] = {}

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                call_id = tc.get("id")
                name = tc.get("name")
                if isinstance(call_id, str) and isinstance(name, str):
                    tool_names_by_call_id[call_id] = name

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        call_id = getattr(msg, "tool_call_id", None)
        if not isinstance(call_id, str):
            continue
        tool_name = tool_names_by_call_id.get(call_id)
        if tool_name:
            outputs[tool_name] = str(msg.content)

    return outputs


def _extract_structured_brief(messages: List[BaseMessage], place: str) -> Dict[str, Any]:
    tool_outputs = _extract_tool_outputs(messages)
    raw_brief = tool_outputs.get("travel_brief_tool")
    if raw_brief:
        try:
            payload = json.loads(raw_brief)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

    risk_output = tool_outputs.get("city_risk_tool", "")
    risk_level = "low"
    if "Risk level: HIGH" in risk_output:
        risk_level = "high"
    elif "Risk level: MEDIUM" in risk_output:
        risk_level = "medium"

    sources: list[dict[str, str]] = []
    if "travel_brief_tool" in tool_outputs or "weather_tool" in tool_outputs:
        sources.append({"type": "weather"})
    if "travel_brief_tool" in tool_outputs or "news_tool" in tool_outputs:
        sources.append({"type": "news"})

    return {
        "place": place,
        "final": "",
        "risk_level": risk_level,
        "travel_advice": [],
        "sources": sources,
    }


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
    """
    user_prompt = _build_user_prompt(place, question)

    # decide tool availability for this turn
    include_weather, include_news = decide_tool_includes(question)

    # session-aware suppression (Redis)
    force_weather, force_news = detect_force_signals(question or "")
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

    policy_lines.extend(
        [
            "- Always produce a one-paragraph travel brief for the specified location.",
            "- You MUST call travel_brief_tool exactly once before writing the final answer.",
            "- Use the travel_brief_tool result as the primary source for risk level, travel advice, and supporting travel context.",
            "- Ground the answer in the concrete travel_brief_tool evidence: weather_summary, weather_reasons, news_items, and news_reasons when available.",
            "- Use the city_risk_tool only when the user explicitly asks about safety level, risk, or go/no-go judgment.",
            "- Explicitly frame the answer around travel conditions, likely disruptions, and practical planning impact.",
            "- Do NOT give generic advice. If weather data is available, mention the material weather signal driving the advice. If news_items are available, mention the most relevant reported issue from the title/snippet.",
            "- If news_items is empty, say that the current news scan did not identify a major local traveler-facing disruption.",
            "- If the user asks for news details, answer only from the retrieved titles/snippets/links. If that detail is absent, say it is not specified in the retrieved news.",
            "- If the user asks about disruptions or 'where' they are, ground the answer using recent news: list up to 3 named places if present, otherwise say 'no specific locations reported'.",
            f"- If the user's question mentions a different place than '{place}', begin with: \"You asked about <other place> but your selected location is {place}. To get updates for <other place>, change the Location.\" Then provide the recommendation for {place} only.",
        ]
    )

    user_prompt = "\n".join(policy_lines) + "\n\n---\n\n" + user_prompt

    # use a gated agent (hard enforcement)
    app = _get_react_app(include_weather=include_weather, include_news=include_news)

    state: Dict[str, Any] = await app.ainvoke({"messages": [{"role": "user", "content": user_prompt}]})
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

    brief = _extract_structured_brief(messages, place)
    result: Dict[str, Any] = {
        "place": str(brief.get("place") or place),
        "final": final_text or str(brief.get("final") or ""),
        "risk_level": str(brief.get("risk_level") or "low"),
        "travel_advice": cast(list[str], brief.get("travel_advice") or []),
        "sources": cast(list[dict[str, str]], brief.get("sources") or []),
    }
    if debug:
        result["debug"] = _build_debug(messages)
    return result
