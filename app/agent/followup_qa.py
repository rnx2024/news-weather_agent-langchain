from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from app.agent.agent_prompts import FOLLOWUP_QA_SYSTEM_PROMPT, JOURNEY_QA_SYSTEM_PROMPT
from app.news.news_service import get_news_items, search_news
from app.travel_brief import build_travel_brief
from app.travel_intelligence import score_news_risk
from app.weather.weather_service import get_weather_line, get_weather_summary

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
    lowered = (text or "").lower()
    day_tokens = (
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
    has_named_time = any(token in lowered for token in day_tokens)
    has_calendar_date = bool(re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b", lowered))
    return has_named_time or has_calendar_date


def _extract_place_phrase(text: str) -> str | None:
    if not text:
        return None

    match = re.search(r"\bin\s+([A-Z][A-Za-z .-]+)", text)
    if match:
        return match.group(1).strip()
    return None


def _build_news_targeted_query(place: str, question: str, item: Dict[str, Any] | None, last_reply: Optional[str]) -> str:
    question_terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*", question or "")
    filtered_terms = [
        term
        for term in question_terms
        if term.lower() not in _FOLLOWUP_STOPWORDS and len(term) > 2
    ]
    if filtered_terms:
        return f"{' '.join(filtered_terms[:10])} {place}".strip()

    if item and item.get("title"):
        return f"{item['title']} {place}"

    tokens = list(_tokenize_for_followup(question))
    if not tokens and last_reply:
        tokens = list(_tokenize_for_followup(last_reply))
    query_core = " ".join(tokens[:6]).strip()
    return f"{query_core} {place}".strip() or place


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


def _build_journey_targeted_query(origin: str, destination: str, question: str, route_or_transport: bool) -> str:
    if route_or_transport:
        question_terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*", question or "")
        filtered_terms = [
            term
            for term in question_terms
            if term.lower() not in _FOLLOWUP_STOPWORDS and len(term) > 2
        ]
        if filtered_terms:
            return f"{' '.join(filtered_terms[:10])} {origin} {destination}".strip()
        return f"{origin} {destination} transport route travel"

    lowered = (question or "").lower()
    if any(term in lowered for term in ("continue", "delay", "closure", "disruption", "strike", "cancel", "postpone")):
        return f"{origin} {destination} travel disruption closure"

    tokens = list(_tokenize_for_followup(question))
    core = " ".join(tokens[:6]).strip()
    return f"{core} {origin} {destination}".strip()


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


async def answer_news_followup(llm: Any, place: str, question: str, last_reply: Optional[str]) -> Dict[str, Any]:
    items, err = get_news_items(place)
    if err:
        final = f"I could not retrieve current news for {place} right now."
        return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": []}

    matched_item = _match_news_item(question, last_reply, items)
    need_targeted_search = not _news_item_answers_question(question, matched_item)

    targeted_query = _build_news_targeted_query(place, question, matched_item, last_reply)
    targeted_items: List[Dict[str, Any]] = []
    targeted_match: Dict[str, Any] | None = None
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
    final = await _run_followup_reasoner(llm, place=place, question=question, evidence=evidence)
    raw_final = final
    if not final:
        final = f"I couldn't find a confirmed answer in the current news for {place}."

    final = _soften_followup_tone(final, place)
    final = _condense_direct_answer(final)
    final = _append_followup_link_if_needed(final, evidence, raw_final)
    return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": [{"type": "news"}]}


async def answer_weather_followup(llm: Any, place: str, question: str) -> Dict[str, Any]:
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
