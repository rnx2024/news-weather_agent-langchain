from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from app.agent.agent_prompts import (
    FOLLOWUP_ACTION_SYSTEM_PROMPT,
    FOLLOWUP_QA_SYSTEM_PROMPT,
    JOURNEY_ACTION_SYSTEM_PROMPT,
    JOURNEY_QA_SYSTEM_PROMPT,
)
from app.news.news_service import get_news_items, search_news
from app.travel_brief import build_travel_brief
from app.weather.weather_service import get_weather_line, get_weather_summary


def _extract_text_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{4,}", (text or "").lower())}


def _match_news_item(question: str, last_reply: Optional[str], items: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not items:
        return None

    reference_text = " ".join(
        part for part in (question or "", last_reply or "") if part and part.strip()
    ).strip()
    reference_tokens = _extract_text_tokens(reference_text)
    if not reference_tokens:
        return items[0]

    best_item = items[0]
    best_score = -1
    for item in items:
        item_tokens = _extract_text_tokens(_news_text(item))
        score = len(reference_tokens & item_tokens)
        if score > best_score:
            best_item = item
            best_score = score
    return best_item


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
        "weather_summary": brief.get("weather_summary"),
        "weather_reasons": brief.get("weather_reasons") or [],
        "news_items": brief.get("news_items") or [],
        "news_reasons": brief.get("news_reasons") or [],
        "source_error": err or None,
    }


async def _invoke_reasoner(llm: Any, system_prompt: str, *, place: str, question: str, evidence: Dict[str, Any]) -> str:
    payload = json.dumps(evidence, ensure_ascii=True, indent=2)
    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Destination: {place}\nQuestion: {question}\nEvidence:\n{payload}",
            },
        ]
    )
    return str(getattr(response, "content", "") or "").strip()


async def _run_followup_reasoner(llm: Any, *, place: str, question: str, evidence: Dict[str, Any]) -> str:
    return await _invoke_reasoner(
        llm,
        FOLLOWUP_QA_SYSTEM_PROMPT,
        place=place,
        question=question,
        evidence=evidence,
    )


async def _run_journey_reasoner(llm: Any, *, place: str, question: str, evidence: Dict[str, Any]) -> str:
    return await _invoke_reasoner(
        llm,
        JOURNEY_QA_SYSTEM_PROMPT,
        place=place,
        question=question,
        evidence=evidence,
    )


async def _plan_followup_action(llm: Any, *, place: str, question: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    raw = await _invoke_reasoner(
        llm,
        FOLLOWUP_ACTION_SYSTEM_PROMPT,
        place=place,
        question=question,
        evidence=evidence,
    )
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {"answered": False, "answer": "", "search_query": ""}

    return {
        "answered": bool(parsed.get("answered")),
        "answer": str(parsed.get("answer") or "").strip(),
        "search_query": str(parsed.get("search_query") or "").strip(),
    }


async def _plan_journey_action(llm: Any, *, place: str, question: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    raw = await _invoke_reasoner(
        llm,
        JOURNEY_ACTION_SYSTEM_PROMPT,
        place=place,
        question=question,
        evidence=evidence,
    )
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {"answered": False, "answer": "", "search_query": ""}

    return {
        "answered": bool(parsed.get("answered")),
        "answer": str(parsed.get("answer") or "").strip(),
        "search_query": str(parsed.get("search_query") or "").strip(),
    }


def _soften_followup_tone(final: str, place: str) -> str:
    text = (final or "").strip()
    if not text:
        return text

    replacements = (
        (
            re.compile(
                rf"^The retrieved news for {re.escape(place)} does not specify the answer to that question\.?$",
                re.IGNORECASE,
            ),
            f"I don't see anything in the current news for {place} that answers that directly.",
        ),
        (
            re.compile(
                rf"^The current weather data for {re.escape(place)} does not specify that detail\.?$",
                re.IGNORECASE,
            ),
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


def _condense_direct_answer(final: str) -> str:
    text = (final or "").strip()
    if not text:
        return text

    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(parts) <= 1:
        return text

    generic_patterns = (
        re.compile(r"\blooks generally fine for travel\b", re.IGNORECASE),
        re.compile(r"\blow risk level\b", re.IGNORECASE),
        re.compile(r"\brecent local reporting highlights\b", re.IGNORECASE),
        re.compile(r"\bcurrent news scan did not surface\b", re.IGNORECASE),
        re.compile(r"\brisk level\b", re.IGNORECASE),
    )
    answer_patterns = (
        re.compile(r"\b(?:not specified|doesn't say|does not say|doesn't confirm|does not confirm)\b", re.IGNORECASE),
        re.compile(r"\b(?:through|until|scheduled|expected|continues?|lasting|runs?)\b", re.IGNORECASE),
        re.compile(r"\b(?:yes|no)\b", re.IGNORECASE),
        re.compile(r"\b(?:i don't see|i can'?t|can'?t answer|cannot answer)\b", re.IGNORECASE),
    )

    non_generic = [part for part in parts if not any(pattern.search(part) for pattern in generic_patterns)]
    direct = [part for part in non_generic if any(pattern.search(part) for pattern in answer_patterns)]

    if direct:
        return direct[0].strip()
    if non_generic:
        return non_generic[0].strip()
    return parts[0].strip()


def _normalize_search_query(query: str, *extras: str) -> str:
    parts = [" ".join((query or "").split())]
    parts.extend(" ".join((extra or "").split()) for extra in extras if extra and extra.strip())
    return " ".join(part for part in parts if part).strip()


def _build_news_targeted_query(place: str, question: str, item: Dict[str, Any] | None, _last_reply: Optional[str]) -> str:
    if question and question.strip():
        return _normalize_search_query(question, place)
    if item and item.get("title"):
        return _normalize_search_query(str(item["title"]), place)
    return place


def _extract_best_news_link(evidence: Dict[str, Any]) -> str | None:
    for key in ("matched_targeted_item", "matched_current_item"):
        item = evidence.get(key)
        if isinstance(item, dict):
            link = str(item.get("link") or "").strip()
            if link:
                return link

    targeted_news_items = evidence.get("targeted_news_items")
    if isinstance(targeted_news_items, list):
        for item in targeted_news_items:
            if isinstance(item, dict):
                link = str(item.get("link") or "").strip()
                if link:
                    return link

    for key in ("destination_evidence", "origin_evidence"):
        block = evidence.get(key)
        if not isinstance(block, dict):
            continue
        news_items = block.get("news_items")
        if not isinstance(news_items, list):
            continue
        for item in news_items:
            if isinstance(item, dict):
                link = str(item.get("link") or "").strip()
                if link:
                    return link

    return None


def _answer_mentions_article_or_source(text: str) -> bool:
    lowered = (text or "").lower()
    return any(term in lowered for term in ("article", "source", "details", "read more", "here", "link"))


def _contains_url(text: str) -> bool:
    return bool(re.search(r"https?://\S+", text or ""))


def _append_followup_link_if_needed(final: str, evidence: Dict[str, Any], original_text: str | None = None) -> str:
    source_text = original_text or final
    if not final or _contains_url(final) or not _answer_mentions_article_or_source(source_text):
        return final

    link = _extract_best_news_link(evidence)
    if not link:
        return final

    separator = "" if final.endswith((".", "!", "?")) else "."
    return f"{final}{separator} Source: {link}"


def _build_journey_targeted_query(origin: str, destination: str, question: str, pending_question: Optional[str]) -> str:
    return _normalize_search_query(pending_question or question, origin, destination)


def _detect_weather_horizon(question: str) -> str:
    lowered = (question or "").lower()
    for day in ("today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
        if day in lowered:
            return day
    if "next week" in lowered:
        return "next week"
    return "today"


async def answer_journey_question(
    llm: Any,
    place: str,
    question: str,
    origin: str,
    *,
    route_or_transport: bool,
    latest_user_message: str = "",
    conversation_history: Optional[List[Dict[str, str]]] = None,
    pending_question: Optional[str] = None,
) -> Dict[str, Any]:
    destination_evidence = _gather_place_evidence(place)
    origin_evidence = _gather_place_evidence(origin)

    current_evidence = {
        "mode": "journey_planning",
        "question": question,
        "latest_user_message": latest_user_message or question,
        "pending_question": pending_question,
        "conversation_history": conversation_history or [],
        "origin": origin,
        "destination": place,
        "route_or_transport": route_or_transport,
        "destination_evidence": destination_evidence,
        "origin_evidence": origin_evidence,
        "targeted_search_query": None,
        "targeted_search_error": None,
        "targeted_news_items": [],
        "limitations": [
            "No dedicated routing engine or live transport schedule data is available in this workflow."
        ],
    }
    plan = await _plan_journey_action(llm, place=place, question=question, evidence=current_evidence)
    if plan["answered"] and plan["answer"]:
        final = plan["answer"]
        raw_final = final
        evidence = current_evidence
    else:
        targeted_query = plan["search_query"] or _build_journey_targeted_query(origin, place, question, pending_question)
        targeted_news_items, targeted_err = search_news(targeted_query, place)
        evidence = {
            **current_evidence,
            "targeted_search_query": targeted_query,
            "targeted_search_error": targeted_err or None,
            "targeted_news_items": targeted_news_items[:3],
        }
        final = await _run_journey_reasoner(llm, place=place, question=question, evidence=evidence)
        raw_final = final

    if not final:
        if route_or_transport:
            final = "I couldn't find a confirmed answer about the best transport from the data I gathered so far."
        else:
            final = "I couldn't find a confirmed answer from the data I gathered so far."

    final = _soften_followup_tone(final, place)
    final = _condense_direct_answer(final)
    final = _append_followup_link_if_needed(final, evidence, raw_final)
    sources = [{"type": "weather"}, {"type": "news"}]
    return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": sources}


async def answer_news_followup(
    llm: Any,
    place: str,
    question: str,
    last_reply: Optional[str],
    *,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    items, err = get_news_items(place)
    if err:
        final = f"I could not retrieve current news for {place} right now."
        return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": []}

    matched_item = _match_news_item(question, last_reply, items)
    current_evidence = {
        "mode": "news_followup",
        "question": question,
        "conversation_history": conversation_history or [],
        "current_news_items": items[:3],
        "matched_current_item": matched_item,
        "used_targeted_search": False,
        "targeted_search_query": None,
        "targeted_search_error": None,
        "targeted_news_items": [],
        "matched_targeted_item": None,
    }

    plan = await _plan_followup_action(llm, place=place, question=question, evidence=current_evidence)
    if plan["answered"] and plan["answer"]:
        final = plan["answer"]
        raw_final = final
        evidence = current_evidence
    else:
        targeted_query = plan["search_query"] or _build_news_targeted_query(place, question, matched_item, last_reply)
        targeted_items, targeted_err = search_news(targeted_query, place)
        targeted_match = _match_news_item(question, last_reply, targeted_items)
        evidence = {
            **current_evidence,
            "used_targeted_search": True,
            "targeted_search_query": targeted_query,
            "targeted_search_error": targeted_err or None,
            "targeted_news_items": targeted_items[:3],
            "matched_targeted_item": targeted_match,
        }
        final = await _run_followup_reasoner(llm, place=place, question=question, evidence=evidence)
        raw_final = final
        if not final:
            final = f"I couldn't find a confirmed answer in the current news for {place}."

    final = _soften_followup_tone(final, place)
    final = _condense_direct_answer(final)
    final = _append_followup_link_if_needed(final, evidence, raw_final)
    return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": [{"type": "news"}]}


async def answer_general_followup(
    llm: Any,
    place: str,
    question: str,
    last_reply: Optional[str],
    *,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    place_evidence = _gather_place_evidence(place)
    matched_item = _match_news_item(question, last_reply, list(place_evidence.get("news_items") or []))
    current_evidence = {
        "mode": "general_followup",
        "question": question,
        "conversation_history": conversation_history or [],
        "place_evidence": place_evidence,
        "matched_current_item": matched_item,
        "used_targeted_search": False,
        "targeted_search_query": None,
        "targeted_search_error": None,
        "targeted_news_items": [],
        "matched_targeted_item": None,
    }

    plan = await _plan_followup_action(llm, place=place, question=question, evidence=current_evidence)
    if plan["answered"] and plan["answer"]:
        final = plan["answer"]
        raw_final = final
        evidence = current_evidence
    else:
        targeted_query = plan["search_query"] or _build_news_targeted_query(place, question, matched_item, last_reply)
        targeted_items, targeted_err = search_news(targeted_query, place)
        targeted_match = _match_news_item(question, last_reply, targeted_items)
        evidence = {
            **current_evidence,
            "used_targeted_search": True,
            "targeted_search_query": targeted_query,
            "targeted_search_error": targeted_err or None,
            "targeted_news_items": targeted_items[:3],
            "matched_targeted_item": targeted_match,
        }
        final = await _run_followup_reasoner(llm, place=place, question=question, evidence=evidence)
        raw_final = final
        if not final:
            final = f"I couldn't find a confirmed answer from the data I gathered so far for {place}."

    final = _soften_followup_tone(final, place)
    final = _condense_direct_answer(final)
    final = _append_followup_link_if_needed(final, evidence, raw_final)
    return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": [{"type": "weather"}, {"type": "news"}]}


async def answer_weather_followup(
    llm: Any,
    place: str,
    question: str,
    *,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    horizon = _detect_weather_horizon(question)
    summary, err = get_weather_summary(place, horizon)
    sources = [{"type": "weather"}]

    if not summary:
        line, line_err = get_weather_line(place)
        if line:
            evidence = {
                "mode": "weather_followup",
                "question": question,
                "conversation_history": conversation_history or [],
                "requested_horizon": horizon,
                "weather_snapshot": line,
                "weather_summary": None,
            }
            final = await _run_followup_reasoner(llm, place=place, question=question, evidence=evidence)
            if not final:
                final = f"The current weather snapshot for {place} is {line}."

            final = _soften_followup_tone(final, place)
            final = _condense_direct_answer(final)
            return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": sources}

        final = f"I could not retrieve current weather data for {place} right now."
        return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": []}

    evidence = {
        "mode": "weather_followup",
        "question": question,
        "conversation_history": conversation_history or [],
        "requested_horizon": horizon,
        "weather_summary": summary,
        "weather_snapshot": None,
    }
    final = await _run_followup_reasoner(llm, place=place, question=question, evidence=evidence)
    if not final:
        final = f"I couldn't find a confirmed weather answer for {place} from the data I gathered so far."

    final = _soften_followup_tone(final, place)
    final = _condense_direct_answer(final)
    return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": sources}
