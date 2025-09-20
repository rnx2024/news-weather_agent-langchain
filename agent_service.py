from __future__ import annotations
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from settings import OPENAI_API_KEY
from weather_service import get_weather_line
from news_service import get_news_items

def run_agent(place: str) -> Dict[str, Any]:
    # 1) fetch
    weather_line, w_err = get_weather_line(place)
    news, n_err = get_news_items(place)

    # 2) build summary prompt
    head_txt = "\n".join(
        f"- {h['title']} ({h['source']}, {h['date']}) -> {h['link']}"
        for h in (news or [])
    ) or "No recent news."

    err_notes: List[str] = []
    if w_err: err_notes.append(f"Weather source failed: {w_err}")
    if n_err: err_notes.append(f"News source failed: {n_err}")
    notes = ("\n\nNotes: " + "; ".join(err_notes)) if err_notes else ""

    prompt = (
        "Summarize concisely as one paragraph, plain text only.\n"
        f"Location: {place}\n"
        f"Weather: {weather_line or 'n/a'}\n"
        "News:\n"
        f"{head_txt}"
        f"{notes}"
    )

    # 3) LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    final = llm.invoke(prompt).content

    return {"final": final}
