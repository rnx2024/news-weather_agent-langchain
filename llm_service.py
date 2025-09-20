from __future__ import annotations
from typing import Dict, List
from langchain_openai import ChatOpenAI
from settings import OPENAI_API_KEY

def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

def summarize_plain(place: str, weather_line: str, headlines: List[Dict]) -> str:
    head_txt = "\n".join([f"- {h['title']} ({h['source']}, {h['date']})" for h in headlines]) or "No recent news."
    prompt = (
        "Summarize concisely:\n"
        f"Location: {place}\n"
        f"Weather: {weather_line or 'n/a'}\n"
        "News:\n"
        f"{head_txt}\n"
        "Output: One paragraph, plain text."
    )
    return get_llm().invoke(prompt).content
