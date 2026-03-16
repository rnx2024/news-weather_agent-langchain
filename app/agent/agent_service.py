# app/agent_service.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from app.settings import settings
from app.agent.agent_tools import weather_tool, news_tool, news_search_tool, city_risk_tool, travel_brief_tool
from app.agent.agent_prompts import ANSWER_MODE_ROUTER_SYSTEM_PROMPT, LOCAL_INTELLIGENCE_SYSTEM_PROMPT
from app.agent.followup_qa import (
    answer_general_followup as _answer_general_followup,
    answer_journey_question as _answer_journey_question,
    answer_news_followup as _answer_news_followup,
    answer_weather_followup as _answer_weather_followup,
)

# session memory (Redis-backed)
from app.session.session_cache import (
    get_active_destination,
    get_last_exchange,
    get_pending_agent_context,
    get_recent_turns,
    get_pending_journey_question,
    mark_tools_called,
    set_active_destination,
    set_pending_agent_context,
    set_pending_journey_question,
    should_include,
)
from app.agent.agent_policy import (
    AnswerMode,
    asks_route_or_transport,
    classify_answer_mode,
    decide_tool_includes,
    detect_force_signals,
    extract_origin,
    needs_followup_reference_clarification,
    needs_origin_clarification,
)


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
        gated.append(news_search_tool)

    app = create_react_agent(model=_llm, tools=gated, prompt=LOCAL_INTELLIGENCE_SYSTEM_PROMPT)
    _REACT_APP_CACHE[key] = app
    return app


def _build_user_prompt(place: str, question: Optional[str], origin: Optional[str] = None) -> str:
    if not question:
        return (
            "Provide a concise travel brief for the destination below. Focus on travel conditions, likely disruptions, "
            f"and what matters most for someone going there today: {place}."
        )
    parts = [
        f"Location: {place}\n"
        f"Question: {question}\n"
    ]
    if origin:
        parts.append(f"Journey origin: {origin}\n")
    parts.append("Answer as ONE concise travel-oriented paragraph, plain text.")
    return "".join(parts)


def _format_recent_turns(recent_turns: List[Dict[str, str]]) -> List[str]:
    if not recent_turns:
        return []

    lines = ["- Recent conversation context:"]
    for turn in recent_turns[-4:]:
        user_text = str(turn.get("user") or "").strip()
        assistant_text = str(turn.get("assistant") or "").strip()
        if user_text:
            lines.append(f"  - User: {user_text}")
        if assistant_text:
            lines.append(f"  - Assistant: {assistant_text}")
    return lines


async def _resolve_answer_mode(
    *,
    question: Optional[str],
    last_reply: Optional[str],
    recent_turns: List[Dict[str, str]],
    pending_agent_context: Optional[Dict[str, str]],
    place: str,
) -> AnswerMode:
    fallback = classify_answer_mode(question, last_reply)
    if not question or not recent_turns:
        return fallback

    evidence = {
        "selected_location": place,
        "latest_question": question,
        "last_reply": last_reply,
        "recent_turns": recent_turns[-4:],
        "pending_agent_context": pending_agent_context or {},
        "allowed_modes": ["travel_brief", "news_followup", "weather_followup", "journey_planning"],
    }
    try:
        response = await _llm.ainvoke(
            [
                {"role": "system", "content": ANSWER_MODE_ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(evidence, ensure_ascii=True, indent=2)},
            ]
        )
        payload = json.loads(str(getattr(response, "content", "") or "").strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback

    mode = str(payload.get("mode") or "").strip()
    if mode in {"travel_brief", "news_followup", "weather_followup", "journey_planning"}:
        return cast(AnswerMode, mode)
    return fallback


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


def _build_policy_lines(
    *,
    place: str,
    answer_mode: AnswerMode,
    include_weather: bool,
    include_news: bool,
    last_user: Optional[str],
    last_reply: Optional[str],
    recent_turns: List[Dict[str, str]],
    origin: Optional[str] = None,
    route_or_transport: bool = False,
) -> List[str]:
    policy_lines: List[str] = ["Policy:", f"- Selected location: {place}"]
    if not include_weather:
        policy_lines.append("- Do NOT call weather_tool or include weather unless explicitly asked.")
    if not include_news:
        policy_lines.append("- Do NOT call news_tool, news_search_tool, or include news unless explicitly asked.")

    if last_user or last_reply:
        policy_lines.append("- Prior exchange context (most recent only):")
        if last_user:
            policy_lines.append(f"  - User: {last_user}")
        if last_reply:
            policy_lines.append(f"  - Assistant: {last_reply}")
    policy_lines.extend(_format_recent_turns(recent_turns))

    common_lines = [
        "- Mention specific locations only if they are explicitly stated in the retrieved news or weather context.",
        "- If evidence is missing or inconclusive, say it is not specified instead of guessing.",
    ]
    if answer_mode != "journey_planning":
        common_lines.append(
            f"- If the user's question mentions a different place than '{place}', begin with: \"You asked about <other place> but your selected location is {place}. To get updates for <other place>, change the Location.\" Then answer for {place} only."
        )

    if answer_mode == "news_followup":
        policy_lines.extend(
            [
                "- Answer the user's specific news question directly in 1-3 sentences. Do NOT produce a generic travel brief.",
                "- You MUST call travel_brief_tool exactly once first to inspect current news_items for the selected location.",
                "- If the current news_items already answer the question, answer directly from those titles/snippets and do NOT call any extra search tool.",
                "- If the current news_items do not answer the question, you MUST call news_search_tool exactly once using a short targeted query composed from the issue/topic and the selected location, such as 'PISTON strike Vigan'.",
                "- If the targeted search still does not confirm the answer, say the retrieved news does not specify it.",
                "- Do NOT include generic travel advice, risk level, or weather unless the user explicitly asked for them.",
            ]
        )
    elif answer_mode == "weather_followup":
        policy_lines.extend(
            [
                "- Answer the user's specific weather question directly in 1-3 sentences. Do NOT produce a generic travel brief.",
                "- You MUST call travel_brief_tool exactly once first to inspect current weather_summary for the selected location.",
                "- If weather_summary already answers the question, answer directly from it and do NOT call weather_tool.",
                "- If weather_summary does not answer the question, you MAY call weather_tool once using the narrowest relevant horizon from the question.",
                "- If the current forecast still does not specify the requested detail, say the current weather data does not specify it.",
                "- Do NOT include generic travel advice, risk level, or unrelated news unless the user explicitly asked for them.",
            ]
        )
    elif answer_mode == "journey_planning":
        policy_lines.extend(
            [
                "- Answer as a journey assessment, not as a destination-only travel brief.",
                "- You MUST call travel_brief_tool exactly once first for the selected destination.",
                f"- Treat '{origin or 'the departure location'}' as the trip origin and '{place}' as the destination.",
                "- If origin is available, inspect origin-side conditions with weather_tool and/or news_tool when they are needed to answer the journey question.",
                "- Distinguish departure conditions, destination conditions, and unknown route conditions.",
                "- If the user asks whether they should continue or postpone the trip, state clearly what is known for the departure point and destination, then note any unknowns along the route.",
                "- Do NOT claim a best route or best transport option from weather/news alone. If asked, say you can comment on likely disruptions and conditions, but not optimize the route without dedicated routing or transport data.",
                "- Keep the answer concise and practical. Do NOT include generic travel-advice bullets or a risk label unless the user explicitly asks for a broad travel brief.",
            ]
        )
        if route_or_transport:
            policy_lines.append(
                "- The user is asking about route or transport choice. Provide only limited guidance from weather/news at the origin and destination, and explicitly say dedicated routing data is not available."
            )
    else:
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
            ]
        )

    policy_lines.extend(common_lines)
    return policy_lines


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
    last_user, last_reply = await get_last_exchange(session_id)
    recent_turns = await get_recent_turns(session_id)
    active_destination = await get_active_destination(session_id)
    pending_agent_context = await get_pending_agent_context(session_id)
    pending_journey_question = await get_pending_journey_question(session_id)
    if active_destination and active_destination != place:
        recent_turns = []
        pending_agent_context = None
        pending_journey_question = None
        await set_pending_agent_context(session_id, None)
        await set_pending_journey_question(session_id, None)
    effective_question = question
    origin = extract_origin(question, last_reply)
    pending_question = (pending_agent_context or {}).get("question")
    same_destination_session = bool(question and active_destination == place and recent_turns)

    awaiting_origin = (pending_agent_context or {}).get("awaiting") == "origin"
    if awaiting_origin:
        origin = origin or extract_origin(question, "Where are you traveling from?")
        if origin:
            effective_question = pending_question or pending_journey_question or last_user or question

    if origin and "where are you traveling from" in (last_reply or "").lower() and not effective_question:
        effective_question = pending_journey_question or last_user or question

    answer_mode = await _resolve_answer_mode(
        question=effective_question,
        last_reply=last_reply,
        recent_turns=recent_turns,
        pending_agent_context=pending_agent_context,
        place=place,
    )
    if awaiting_origin and origin and (pending_agent_context or {}).get("mode") == "journey_planning":
        answer_mode = "journey_planning"

    origin = origin or extract_origin(effective_question, last_reply)
    route_or_transport = asks_route_or_transport(effective_question)

    if needs_followup_reference_clarification(question, last_reply):
        clarification = "I need the specific news item or the previous message to answer that follow-up directly."
        await mark_tools_called(
            session_id,
            tool_names=[],
            user_message=question,
            agent_reply=clarification,
        )
        result: Dict[str, Any] = {
            "place": place,
            "final": clarification,
            "risk_level": None,
            "travel_advice": [],
            "sources": [],
        }
        if debug:
            result["debug"] = []
        return result

    if answer_mode == "news_followup":
        result = await _answer_news_followup(
            _llm,
            place,
            question or "",
            last_reply,
            conversation_history=recent_turns,
        )
        await mark_tools_called(
            session_id,
            tool_names=[],
            user_message=question,
            agent_reply=result["final"],
        )
        await set_active_destination(session_id, place)
        if debug:
            result["debug"] = []
        return result

    if answer_mode == "weather_followup":
        result = await _answer_weather_followup(
            _llm,
            place,
            question or "",
            conversation_history=recent_turns,
        )
        await mark_tools_called(
            session_id,
            tool_names=[],
            user_message=question,
            agent_reply=result["final"],
        )
        await set_active_destination(session_id, place)
        if debug:
            result["debug"] = []
        return result

    if same_destination_session and answer_mode == "travel_brief":
        result = await _answer_general_followup(
            _llm,
            place,
            question or "",
            last_reply,
            conversation_history=recent_turns,
        )
        await mark_tools_called(
            session_id,
            tool_names=[],
            user_message=question,
            agent_reply=result["final"],
        )
        await set_active_destination(session_id, place)
        if debug:
            result["debug"] = []
        return result

    if answer_mode == "journey_planning" and needs_origin_clarification(question, last_reply):
        clarification = (
            f"I can assess conditions in {place}, but I need your departure location to judge the trip itself. "
            "Where are you traveling from?"
        )
        await set_pending_agent_context(
            session_id,
            {
                "mode": "journey_planning",
                "awaiting": "origin",
                "question": question or "",
                "destination": place,
            },
        )
        await set_pending_journey_question(session_id, question)
        await mark_tools_called(
            session_id,
            tool_names=[],
            user_message=question,
            agent_reply=clarification,
        )
        await set_active_destination(session_id, place)
        result: Dict[str, Any] = {
            "place": place,
            "final": clarification,
            "risk_level": None,
            "travel_advice": [],
            "sources": [],
        }
        if debug:
            result["debug"] = []
        return result

    if answer_mode == "journey_planning" and origin:
        await set_pending_agent_context(session_id, None)
        await set_pending_journey_question(session_id, None)
        result = await _answer_journey_question(
            _llm,
            place,
            effective_question or question or "",
            origin,
            route_or_transport=route_or_transport,
            latest_user_message=question or "",
            conversation_history=recent_turns,
            pending_question=pending_question,
        )
        await mark_tools_called(
            session_id,
            tool_names=[],
            user_message=question,
            agent_reply=result["final"],
        )
        await set_active_destination(session_id, place)
        if debug:
            result["debug"] = []
        return result

    user_prompt = _build_user_prompt(place, effective_question, origin)

    # decide tool availability for this turn
    include_weather, include_news = decide_tool_includes(effective_question)

    # session-aware suppression (Redis)
    force_weather, force_news = detect_force_signals(effective_question or "")
    allow_weather, allow_news = await should_include(session_id, force_weather, force_news)

    # Follow-up questions should still be allowed to inspect current evidence.
    if answer_mode == "news_followup":
        include_news = True
    elif answer_mode == "weather_followup":
        include_weather = True
    elif answer_mode == "journey_planning":
        include_weather = True
        include_news = True

    if include_weather and not allow_weather and answer_mode == "travel_brief":
        include_weather = False
    if include_news and not allow_news and answer_mode == "travel_brief":
        include_news = False

    policy_lines = _build_policy_lines(
        place=place,
        answer_mode=answer_mode,
        include_weather=include_weather,
        include_news=include_news,
        last_user=last_user,
        last_reply=last_reply,
        recent_turns=recent_turns,
        origin=origin,
        route_or_transport=route_or_transport,
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
    await set_active_destination(session_id, place)

    brief = _extract_structured_brief(messages, place)
    result: Dict[str, Any] = {
        "place": str(brief.get("place") or place),
        "final": final_text or str(brief.get("final") or ""),
        "risk_level": (
            str(brief.get("risk_level") or "low") if answer_mode == "travel_brief" else None
        ),
        "travel_advice": cast(list[str], brief.get("travel_advice") or []) if answer_mode == "travel_brief" else [],
        "sources": cast(list[dict[str, str]], brief.get("sources") or []),
    }
    if debug:
        result["debug"] = _build_debug(messages)
    return result
