from __future__ import annotations
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from settings import OPENAI_API_KEY
from weather_service import get_weather_line
from news_service import get_news_items

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
_parser = StrOutputParser()

_prompt = PromptTemplate.from_template(
    "Summarize as ONE concise paragraph, plain text.\n"
    "Location: {place}\n"
    "Weather: {weather_line}\n"
    "News:\n{headlines}\n"
    "{notes}"
)

def run_agent(place: str) -> Dict[str, Any]:
    weather_line, w_err = get_weather_line(place)
    headlines, n_err = get_news_items(place)

    head_txt = "\n".join(
        f"- {h['title']} ({h['source']}, {h['date']}) -> {h['link']}"
        for h in (headlines or [])
    ) or "No recent news."

    errs: List[str] = []
    if w_err: errs.append(f"Weather source failed: {w_err}")
    if n_err: errs.append(f"News source failed: {n_err}")
    notes = ("Notes: " + "; ".join(errs)) if errs else ""

    chain = _prompt | _llm | _parser
    final = chain.invoke({
        "place": place,
        "weather_line": weather_line or "n/a",
        "headlines": head_txt,
        "notes": notes
    })
    return {"final": final}
