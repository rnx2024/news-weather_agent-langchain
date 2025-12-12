# app/agent_service.py
from __future__ import annotations

from typing import Dict, Any, Optional, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from app.settings import settings
from app.agent_tools import weather_tool, news_tool, city_risk_tool
from app.agent_prompts import LOCAL_INTELLIGENCE_SYSTEM_PROMPT


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
# Build LangGraph ReAct agent
# -----------------------------------------------------
# Prebuilt ReAct agent: loops Thought → Action(tool) → Observation until done.
# State schema is {"messages": [...]} where messages is a list of BaseMessage.
react_app = create_react_agent(
    model=_llm,
    tools=tools,
    prompt=LOCAL_INTELLIGENCE_SYSTEM_PROMPT,
)

# -----------------------------------------------------
# Helper: build user prompt string
# -----------------------------------------------------
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
    """
    Given the list of messages from the ReAct agent state,
    return the last non-tool AI assistant message content.
    """
    final_text = ""

    for msg in messages:
        # We only care about AI messages that are not pure tool messages.
        if isinstance(msg, AIMessage):
            # AIMessage may contain tool_calls (for tools) or plain content (final answer)
            if msg.content:
                final_text = str(msg.content)

    return final_text or ""


# -----------------------------------------------------
# Helper: build debug info from messages (tool calls + results)
# -----------------------------------------------------
def _build_debug(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """
    Very lightweight debugging info: pairs tool calls (AIMessage with tool_calls)
    with following ToolMessage observations.
    """
    debug_steps: List[Dict[str, Any]] = []

    # We walk messages and link AI tool calls -> ToolMessages.
    pending_tools: Dict[str, Dict[str, Any]] = {}

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Each tool_call: {"name": ..., "args": {...}, "id": ...}
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
            # ToolMessage usually has metadata about which tool_call it answers.
            call_id = msg.tool_call_id if hasattr(msg, "tool_call_id") else None
            if call_id and call_id in pending_tools:
                pending_tools[call_id]["observation"] = msg.content

    # Normalize into list
    for step in pending_tools.values():
        debug_steps.append(step)

    return debug_steps


# -----------------------------------------------------
# Public function: run_agent
# -----------------------------------------------------
def run_agent(
    place: str,
    question: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run the LangGraph ReAct agent.

    Returns:
        {
          "final": "<one-paragraph answer>",
          "debug": [ {tool, tool_input, observation}, ... ]  # only if debug=True
        }
    """
    user_prompt = _build_user_prompt(place, question)

    # LangGraph ReAct expects {"messages": [ {...} ]}
    state: Dict[str, Any] = react_app.invoke(
        {
            "messages": [
                {"role": "user", "content": user_prompt},
            ]
        }
    )

    messages = state.get("messages", []) or []
    final_text = _extract_final_message(messages)

    result: Dict[str, Any] = {"final": final_text}

    if debug:
        result["debug"] = _build_debug(messages)

    return result
