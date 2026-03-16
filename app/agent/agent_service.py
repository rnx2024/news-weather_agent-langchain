# app/agent_service.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from app.settings import settings
from app.agent.agent_tools import weather_tool, news_tool, news_search_tool, city_risk_tool, travel_brief_tool
from app.agent.agent_prompts import FOLLOWUP_QA_SYSTEM_PROMPT, JOURNEY_QA_SYSTEM_PROMPT, LOCAL_INTELLIGENCE_SYSTEM_PROMPT
from app.news.news_service import get_news_items, search_news
from app.travel_brief import build_travel_brief
from app.weather.weather_service import get_weather_line, get_weather_summary
from app.travel_intelligence import score_news_risk

# session memory (Redis-backed)
from app.session.session_cache import get_last_exchange, should_include, mark_tools_called
from app.agent.agent_policy import (
    AnswerMode,
    asks_route_or_transport,
    classify_answer_mode,
    decide_tool_includes,
    detect_force_signals,
    extract_origin,
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
_FOLLOWUP_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "any",
    "are",
    "campaign",
    "city",
    "does",
    "going",
    "have",
    "into",
    "just",
    "know",
    "last",
    "local",
    "news",
    "question",
    "recent",
    "reported",
    "risk",
    "saturday",
    "should",
    "still",
    "that",
    "there",
    "this",
    "until",
    "visit",
    "what",
    "when",
    "where",
    "will",
    "with",
}


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


def _tokenize_for_followup(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z]{4,}", (text or "").lower())
        if token not in _FOLLOWUP_STOPWORDS
    }


def _match_news_item(question: str, last_reply: Optional[str], items: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not items:
        return None

    q_tokens = _tokenize_for_followup(question)
    last_tokens = _tokenize_for_followup(last_reply or "")
    best_item: Dict[str, Any] | None = None
    best_score = -1

    for item in items:
        blob = f"{item.get('title') or ''} {item.get('snippet') or ''}".lower()
        item_tokens = _tokenize_for_followup(blob)
        score = len(q_tokens & item_tokens) * 3 + len(last_tokens & item_tokens)
        if score > best_score:
            best_score = score
            best_item = item

    return best_item or items[0]


def _news_text(item: Dict[str, Any] | None) -> str:
    if not item:
        return ""
    return " ".join(
        part.strip()
        for part in (str(item.get("title") or ""), str(item.get("snippet") or ""))
        if part and str(part).strip()
    ).strip()


def _gather_place_evidence(place: str) -> Dict[str, Any]:
    brief, err = build_travel_brief(place)
    return {
        "place": place,
        "risk_level": brief.get("risk_level"),
        "weather_summary": brief.get("weather_summary"),
        "weather_reasons": brief.get("weather_reasons") or [],
        "news_items": brief.get("news_items") or [],
        "news_reasons": brief.get("news_reasons") or [],
        "source_error": err or None,
    }


async def _run_followup_reasoner(
    *,
    place: str,
    question: str,
    evidence: Dict[str, Any],
) -> str:
    payload = json.dumps(evidence, ensure_ascii=True, indent=2)
    response = await _llm.ainvoke(
        [
            {"role": "system", "content": FOLLOWUP_QA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Destination: {place}\nQuestion: {question}\nEvidence:\n{payload}",
            },
        ]
    )
    return str(getattr(response, "content", "") or "").strip()


async def _run_journey_reasoner(
    *,
    place: str,
    question: str,
    evidence: Dict[str, Any],
) -> str:
    payload = json.dumps(evidence, ensure_ascii=True, indent=2)
    response = await _llm.ainvoke(
        [
            {"role": "system", "content": JOURNEY_QA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Destination: {place}\nQuestion: {question}\nEvidence:\n{payload}",
            },
        ]
    )
    return str(getattr(response, "content", "") or "").strip()


def _soften_followup_tone(final: str, place: str) -> str:
    text = (final or "").strip()
    if not text:
        return text

    replacements = (
        (
            re.compile(rf"^The retrieved news for {re.escape(place)} does not specify the answer to that question\.?$", re.IGNORECASE),
            f"I don't see anything in the current news for {place} that answers that directly.",
        ),
        (
            re.compile(rf"^The current weather data for {re.escape(place)} does not specify that detail\.?$", re.IGNORECASE),
            f"The current forecast for {place} doesn't spell that out.",
        ),
        (
            re.compile(r"^The retrieved reporting does not specify any possible disruptions\.?\s*", re.IGNORECASE),
            "I don't see any confirmed disruptions in the current reporting. ",
        ),
        (
            re.compile(r"^The retrieved reporting does not confirm that ", re.IGNORECASE),
            "I don't see anything in the current reporting that confirms ",
        ),
        (
            re.compile(r"^The retrieved reporting does not specify ", re.IGNORECASE),
            "I don't see anything in the current reporting that specifies ",
        ),
        (
            re.compile(r"^The retrieved weather data for ", re.IGNORECASE),
            "The current weather for ",
        ),
        (
            re.compile(r" does not specify any possible risks or weather disturbances\.?$", re.IGNORECASE),
            " doesn't point to any specific weather disruptions right now.",
        ),
        (
            re.compile(r" does not specify that detail\.?$", re.IGNORECASE),
            " doesn't spell that out.",
        ),
    )

    for pattern, replacement in replacements:
        text = pattern.sub(replacement, text)

    return re.sub(r"\s{2,}", " ", text).strip()


def _is_duration_question(question: str) -> bool:
    q = (question or "").lower()
    return any(term in q for term in ("until", "last", "still", "continue", "ongoing", "on saturday", "on sunday"))


def _is_location_question(question: str) -> bool:
    q = (question or "").lower()
    return any(term in q for term in ("where", "in ", "location", "area", "which part"))


def _is_disruption_question(question: str) -> bool:
    q = (question or "").lower()
    return any(term in q for term in ("disruption", "problem", "issue", "closure", "strike", "does this affect"))


def _contains_schedule_signal(text: str) -> bool:
    t = (text or "").lower()
    return any(
        token in t
        for token in (
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "week",
            "weekend",
            "until",
            "through",
        )
    ) or bool(re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b", t))


def _extract_place_phrase(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"\bin\s+([A-Z][A-Za-z .-]+)", text)
    if match:
        return match.group(1).strip()
    return None


def _build_news_targeted_query(place: str, question: str, item: Dict[str, Any] | None, last_reply: Optional[str]) -> str:
    if item and item.get("title"):
        return f"{item['title']} {place}"

    tokens = list(_tokenize_for_followup(question))
    if not tokens and last_reply:
        tokens = list(_tokenize_for_followup(last_reply))
    query_core = " ".join(tokens[:6]).strip()
    return f"{query_core} {place}".strip() or place


def _build_news_followup_final(place: str, question: str, item: Dict[str, Any] | None) -> str | None:
    text = _news_text(item)
    if not text:
        return None

    title = str(item.get("title") or "the retrieved report").strip()
    snippet = str(item.get("snippet") or "").strip()

    if _is_duration_question(question):
        if _contains_schedule_signal(text):
            return f"The retrieved news says {title}. Based on the available report, it appears to include timing details: {snippet or title}"
        return f"The retrieved news mentions {title}, but the available snippets do not say whether it will still be running on Saturday."

    if _is_location_question(question):
        specific_place = _extract_place_phrase(text)
        if specific_place:
            return f"The retrieved news places it in {specific_place}."
        return f"The retrieved news mentions {title}, but the available snippets do not identify a more specific location."

    if _is_disruption_question(question):
        score, _ = score_news_risk(place, [item])
        if score >= 2:
            return f"The retrieved news suggests this could affect travel convenience, but it does not read like a major emergency-level disruption."
        return f"The retrieved news mentions {title}, but it does not read like a major traveler-facing disruption."

    return f"The retrieved news says {title}" + (f": {snippet}" if snippet else ".")


def _news_item_answers_question(question: str, item: Dict[str, Any] | None) -> bool:
    text = _news_text(item)
    if not text:
        return False

    if _is_duration_question(question):
        return _contains_schedule_signal(text)
    if _is_location_question(question):
        return _extract_place_phrase(text) is not None
    if _is_disruption_question(question):
        return True
    return bool(_tokenize_for_followup(question) & _tokenize_for_followup(text))


def _extract_best_news_link(evidence: Dict[str, Any]) -> str | None:
    for key in ("matched_targeted_item", "matched_current_item"):
        item = evidence.get(key)
        if isinstance(item, dict):
            link = str(item.get("link") or "").strip()
            if link:
                return link
    for key in ("targeted_news_items",):
        items = evidence.get(key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    link = str(item.get("link") or "").strip()
                    if link:
                        return link
    for key in ("destination_evidence", "origin_evidence"):
        block = evidence.get(key)
        if isinstance(block, dict):
            items = block.get("news_items")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        link = str(item.get("link") or "").strip()
                        if link:
                            return link
    return None


def _answer_mentions_article_or_source(text: str) -> bool:
    lower = (text or "").lower()
    return any(term in lower for term in ("article", "source", "details", "read more", "here", "link"))


def _contains_url(text: str) -> bool:
    return bool(re.search(r"https?://\S+", text or ""))


def _append_followup_link_if_needed(final: str, evidence: Dict[str, Any]) -> str:
    if not final or _contains_url(final) or not _answer_mentions_article_or_source(final):
        return final
    link = _extract_best_news_link(evidence)
    if not link:
        return final
    separator = "" if final.endswith((".", "!", "?")) else "."
    return f"{final}{separator} Source: {link}"


def _build_journey_targeted_query(origin: str, destination: str, question: str, route_or_transport: bool) -> str:
    if route_or_transport:
        return f"{origin} {destination} road closure traffic bus transport"

    q = (question or "").lower()
    if any(term in q for term in ("continue", "delay", "closure", "disruption", "strike", "cancel", "postpone")):
        return f"{origin} {destination} travel disruption closure"

    tokens = list(_tokenize_for_followup(question))
    core = " ".join(tokens[:6]).strip()
    return f"{core} {origin} {destination}".strip()


async def _answer_journey_question(
    place: str,
    question: str,
    origin: str,
    *,
    route_or_transport: bool,
) -> Dict[str, Any]:
    destination_evidence = _gather_place_evidence(place)
    origin_evidence = _gather_place_evidence(origin)

    targeted_query = _build_journey_targeted_query(origin, place, question, route_or_transport)
    targeted_news_items, targeted_err = search_news(targeted_query, place)

    evidence = {
        "mode": "journey_planning",
        "question": question,
        "origin": origin,
        "destination": place,
        "route_or_transport": route_or_transport,
        "destination_evidence": destination_evidence,
        "origin_evidence": origin_evidence,
        "targeted_search_query": targeted_query,
        "targeted_search_error": targeted_err or None,
        "targeted_news_items": targeted_news_items[:3],
        "limitations": [
            "No dedicated routing engine or live transport schedule data is available in this workflow."
        ],
    }
    final = await _run_journey_reasoner(place=place, question=question, evidence=evidence)
    if not final:
        if route_or_transport:
            final = "I can't tell you the best transport confidently from the data I gathered so far."
        else:
            final = "I can't answer that confidently from the data I gathered so far."
    final = _soften_followup_tone(final, place)
    final = _append_followup_link_if_needed(final, evidence)
    sources = [{"type": "weather"}, {"type": "news"}]
    return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": sources}


async def _answer_news_followup(place: str, question: str, last_reply: Optional[str]) -> Dict[str, Any]:
    items, err = get_news_items(place)
    if err:
        final = f"I could not retrieve current news for {place} right now."
        return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": []}

    matched_item = _match_news_item(question, last_reply, items)
    need_targeted_search = not _news_item_answers_question(question, matched_item)

    targeted_query = None
    targeted_items: List[Dict[str, Any]] = []
    targeted_match: Dict[str, Any] | None = None
    targeted_query = _build_news_targeted_query(place, question, matched_item, last_reply)
    targeted_err = ""
    if need_targeted_search:
        targeted_items, targeted_err = search_news(targeted_query, place)
        targeted_match = _match_news_item(question, last_reply, targeted_items)

    evidence = {
        "mode": "news_followup",
        "question": question,
        "current_news_items": items[:3],
        "matched_current_item": matched_item,
        "used_targeted_search": need_targeted_search,
        "targeted_search_query": targeted_query if need_targeted_search else None,
        "targeted_search_error": targeted_err or None,
        "targeted_news_items": targeted_items[:3],
        "matched_targeted_item": targeted_match,
        "current_item_answers_question": _news_item_answers_question(question, matched_item),
        "targeted_item_answers_question": _news_item_answers_question(question, targeted_match),
    }
    final = await _run_followup_reasoner(place=place, question=question, evidence=evidence)
    if not final:
        final = f"The retrieved news for {place} does not specify the answer to that question."
    final = _soften_followup_tone(final, place)
    final = _append_followup_link_if_needed(final, evidence)
    return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": [{"type": "news"}]}


def _detect_weather_horizon(question: str) -> str:
    q = (question or "").lower()
    for day in ("today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
        if day in q:
            return day
    if "next week" in q:
        return "next week"
    return "today"


async def _answer_weather_followup(place: str, question: str) -> Dict[str, Any]:
    horizon = _detect_weather_horizon(question)
    summary, err = get_weather_summary(place, horizon)
    sources = [{"type": "weather"}]

    if not summary:
        line, line_err = get_weather_line(place)
        if line:
            evidence = {
                "mode": "weather_followup",
                "question": question,
                "requested_horizon": horizon,
                "weather_snapshot": line,
                "weather_summary": None,
            }
            final = await _run_followup_reasoner(place=place, question=question, evidence=evidence)
            if not final:
                final = f"The current weather snapshot for {place} is {line}."
            final = _soften_followup_tone(final, place)
            return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": sources}
        final = f"I could not retrieve current weather data for {place} right now."
        if err or line_err:
            final = final
        return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": []}

    evidence = {
        "mode": "weather_followup",
        "question": question,
        "requested_horizon": horizon,
        "weather_summary": summary,
        "weather_snapshot": None,
    }
    final = await _run_followup_reasoner(place=place, question=question, evidence=evidence)
    if not final:
        final = f"The current weather data for {place} does not specify that detail."
    final = _soften_followup_tone(final, place)
    return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": sources}


def _build_policy_lines(
    *,
    place: str,
    answer_mode: AnswerMode,
    include_weather: bool,
    include_news: bool,
    last_user: Optional[str],
    last_reply: Optional[str],
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

    common_lines = [
        "- Mention specific locations only if they are explicitly stated in the retrieved news or weather context.",
        "- If evidence is missing or inconclusive, say it is not specified instead of guessing.",
        f"- If the user's question mentions a different place than '{place}', begin with: \"You asked about <other place> but your selected location is {place}. To get updates for <other place>, change the Location.\" Then answer for {place} only.",
    ]

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
    resumed_origin = extract_origin(question, last_reply)
    effective_question = last_user if resumed_origin and "where are you traveling from" in (last_reply or "").lower() and last_user else question
    answer_mode = classify_answer_mode(effective_question, last_reply)
    origin = resumed_origin or extract_origin(effective_question, last_reply)
    route_or_transport = asks_route_or_transport(effective_question)

    if answer_mode == "news_followup":
        result = await _answer_news_followup(place, question or "", last_reply)
        await mark_tools_called(
            session_id,
            tool_names=[],
            user_message=question,
            agent_reply=result["final"],
        )
        if debug:
            result["debug"] = []
        return result

    if answer_mode == "weather_followup":
        result = await _answer_weather_followup(place, question or "")
        await mark_tools_called(
            session_id,
            tool_names=[],
            user_message=question,
            agent_reply=result["final"],
        )
        if debug:
            result["debug"] = []
        return result

    if answer_mode == "journey_planning" and needs_origin_clarification(question, last_reply):
        clarification = (
            f"I can assess conditions in {place}, but I need your departure location to judge the trip itself. "
            "Where are you traveling from?"
        )
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

    if answer_mode == "journey_planning" and origin:
        result = await _answer_journey_question(
            place,
            effective_question or question or "",
            origin,
            route_or_transport=route_or_transport,
        )
        await mark_tools_called(
            session_id,
            tool_names=[],
            user_message=question,
            agent_reply=result["final"],
        )
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
