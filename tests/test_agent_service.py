import json
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, ToolMessage

from app.agent.agent_service import run_agent


class _FakeAgentApp:
    async def ainvoke(self, _payload):
        brief = {
            "place": "Vigan",
            "final": "Vigan is manageable for travel today.",
            "risk_level": "medium",
            "travel_advice": [
                "Recent local reporting highlights a PISTON strike; verify the latest official status before departure"
            ],
            "sources": [{"type": "weather"}, {"type": "news"}],
            "weather_summary": {
                "current": {"weather_text": "Clear sky"},
                "day": {"tmin_c": 24, "tmax_c": 31},
            },
            "weather_reasons": [],
            "news_reasons": ["recent reports suggest possible transport or access disruptions"],
            "news_items": [
                {
                    "title": "PISTON strike affects transport routes",
                    "snippet": "The report does not specify whether the strike is in Vigan.",
                    "source": "Local News",
                    "date": "2026-03-16T05:00:00+00:00",
                    "link": "https://example.com/piston-strike",
                }
            ],
        }
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"id": "call1", "name": "travel_brief_tool", "args": {"place": "Vigan"}}],
                ),
                ToolMessage(content=json.dumps(brief), tool_call_id="call1"),
                AIMessage(
                    content="The retrieved Vigan news mentions a PISTON strike, but the available snippets do not specify that it is in Vigan.",
                ),
            ]
        }


class AgentServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_news_followup_hides_brief_metadata(self) -> None:
        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=(None, None))):
            with patch("app.agent.agent_service.should_include", new=AsyncMock(return_value=(True, True))):
                with patch("app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)):
                    with patch("app.agent.agent_service._get_react_app", return_value=_FakeAgentApp()):
                        result = await run_agent(
                            session_id="session-1",
                            place="Vigan",
                            question="Can you check if the PISTON strike is in Vigan?",
                        )

        self.assertIsNone(result["risk_level"])
        self.assertEqual(result["travel_advice"], [])
        self.assertEqual(result["sources"], [{"type": "weather"}, {"type": "news"}])
        self.assertIn("do not specify", result["final"].lower())


if __name__ == "__main__":
    unittest.main()
