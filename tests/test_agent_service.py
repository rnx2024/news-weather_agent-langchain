import unittest
from unittest.mock import AsyncMock, patch

from app.agent.agent_service import run_agent


class AgentServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_news_followup_hides_brief_metadata(self) -> None:
        initial_items = [
            {
                "title": "PISTON strike affects transport routes",
                "snippet": "The report does not specify whether the strike is in Vigan.",
                "source": "Local News",
                "date": "2026-03-16T05:00:00+00:00",
                "link": "https://example.com/piston-strike",
            }
        ]
        targeted_items = [
            {
                "title": "PISTON strike update",
                "snippet": "The updated report still does not confirm that the strike is in Vigan.",
                "source": "Local News",
                "date": "2026-03-16T06:00:00+00:00",
                "link": "https://example.com/piston-strike-update",
            }
        ]
        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=(None, None))):
            with patch("app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)):
                with patch("app.agent.agent_service.get_news_items", return_value=(initial_items, "")):
                    with patch("app.agent.agent_service.search_news", return_value=(targeted_items, "")) as search_mock:
                        with patch(
                            "app.agent.agent_service._run_followup_reasoner",
                            new=AsyncMock(
                                return_value="The retrieved reporting does not confirm that the PISTON strike is in Vigan."
                            ),
                        ) as reasoner_mock:
                            with patch("app.agent.agent_service._get_react_app") as react_mock:
                                result = await run_agent(
                                    session_id="session-1",
                                    place="Vigan",
                                    question="Can you check if the PISTON strike is in Vigan?",
                                )

        self.assertIsNone(result["risk_level"])
        self.assertEqual(result["travel_advice"], [])
        self.assertEqual(result["sources"], [{"type": "news"}])
        self.assertIn("confirms the piston strike is in vigan", result["final"].lower())
        self.assertNotIn("the retrieved reporting does not confirm", result["final"].lower())
        search_mock.assert_not_called()
        reasoner_mock.assert_awaited()
        react_mock.assert_not_called()

    async def test_news_followup_softens_robotic_language(self) -> None:
        initial_items = [
            {
                "title": "Hit-and-run chase reported in La Union",
                "snippet": "Initial reporting mentions an incident in the area.",
                "source": "Local News",
                "date": "2026-03-16T05:00:00+00:00",
                "link": "https://example.com/hit-and-run",
            }
        ]
        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=(None, None))):
            with patch("app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)):
                with patch("app.agent.agent_service.get_news_items", return_value=(initial_items, "")):
                    with patch(
                        "app.agent.agent_service._run_followup_reasoner",
                        new=AsyncMock(
                            return_value=(
                                "The retrieved reporting does not specify any possible disruptions. "
                                "For more details, you can check the news article here."
                            )
                        ),
                    ):
                        result = await run_agent(
                            session_id="session-tone-news",
                            place="La Union",
                            question="I plan to go here by Saturday? Any possible disruptions?",
                        )

        self.assertIn("i don't see any confirmed disruptions", result["final"].lower())
        self.assertNotIn("the retrieved reporting does not specify", result["final"].lower())

    async def test_journey_question_without_origin_asks_for_clarification(self) -> None:
        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=(None, None))):
            with patch("app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)) as mark_mock:
                result = await run_agent(
                    session_id="session-2",
                    place="Vigan",
                    question="Should I continue my trip?",
                )

        self.assertIsNone(result["risk_level"])
        self.assertEqual(result["travel_advice"], [])
        self.assertEqual(result["sources"], [])
        self.assertIn("where are you traveling from", result["final"].lower())
        mark_mock.assert_awaited()

    async def test_news_followup_duration_uses_targeted_search_and_direct_answer(self) -> None:
        initial_items = [
            {
                "title": "La Union city launches free vaccination campaign to prevent rabies",
                "snippet": "The initial item announces the campaign but does not include an end date.",
                "source": "Local News",
                "date": "2026-03-16T05:00:00+00:00",
                "link": "https://example.com/vaccine-campaign",
            }
        ]
        targeted_items = [
            {
                "title": "Vaccination campaign runs through Saturday in San Fernando",
                "snippet": "City officials said the free vaccination drive will continue through Saturday afternoon.",
                "source": "Local News",
                "date": "2026-03-16T06:00:00+00:00",
                "link": "https://example.com/vaccine-campaign-saturday",
            }
        ]
        prior_reply = (
            "San Fernando La Union has a low travel risk level. Recent local reporting highlights "
            "La Union city launches free vaccination campaign to prevent rabies."
        )
        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=(None, prior_reply))):
            with patch("app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)):
                with patch("app.agent.agent_service.get_news_items", return_value=(initial_items, "")):
                    with patch("app.agent.agent_service.search_news", return_value=(targeted_items, "")) as search_mock:
                        with patch(
                            "app.agent.agent_service._run_followup_reasoner",
                            new=AsyncMock(
                                return_value="Based on the retrieved update, the vaccination drive is scheduled to continue through Saturday afternoon."
                            ),
                        ):
                            result = await run_agent(
                                session_id="session-3",
                                place="San Fernando La Union",
                                question="I plan to visit on Saturday. Will the vaccination last until Saturday?",
                            )

        self.assertIsNone(result["risk_level"])
        self.assertEqual(result["travel_advice"], [])
        self.assertEqual(result["sources"], [{"type": "news"}])
        self.assertIn("through saturday afternoon", result["final"].lower())
        search_mock.assert_called_once()

    async def test_news_followup_appends_source_link_when_answer_references_article(self) -> None:
        initial_items = [
            {
                "title": "Hit-and-run chase reported in La Union",
                "snippet": "Initial reporting mentions an incident in the area.",
                "source": "Local News",
                "date": "2026-03-16T05:00:00+00:00",
                "link": "https://example.com/hit-and-run",
            }
        ]
        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=(None, None))):
            with patch("app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)):
                with patch("app.agent.agent_service.get_news_items", return_value=(initial_items, "")):
                    with patch(
                        "app.agent.agent_service._run_followup_reasoner",
                        new=AsyncMock(
                            return_value=(
                                "The retrieved reporting does not specify any possible disruptions. "
                                "For more details, you can check the news article here."
                            )
                        ),
                    ):
                        result = await run_agent(
                            session_id="session-4",
                            place="La Union",
                            question="I plan to go here by Saturday? Any possible disruptions?",
                        )

        self.assertIn("source: https://example.com/hit-and-run", result["final"].lower())

    async def test_weather_followup_softens_robotic_language(self) -> None:
        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=(None, None))):
            with patch("app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)):
                with patch("app.agent.agent_service.get_weather_summary", return_value=("Broken clouds with a temperature of 25.78C.", "")):
                    with patch(
                        "app.agent.agent_service._run_followup_reasoner",
                        new=AsyncMock(
                            return_value=(
                                "The retrieved weather data for La Union shows broken clouds with a temperature of 25.78C, "
                                "but it does not specify any possible risks or weather disturbances."
                            )
                        ),
                    ):
                        result = await run_agent(
                            session_id="session-tone-weather",
                            place="La Union",
                            question="So no possible risks? or weather disturbances?",
                        )

        self.assertIn("the current weather for la union", result["final"].lower())
        self.assertIn("doesn't point to any specific weather disruptions right now", result["final"].lower())
        self.assertNotIn("the retrieved weather data", result["final"].lower())


if __name__ == "__main__":
    unittest.main()
