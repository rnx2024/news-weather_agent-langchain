import unittest

from app.agent.agent_policy import classify_answer_mode


class AgentPolicyTests(unittest.TestCase):
    def test_classify_news_followup(self) -> None:
        mode = classify_answer_mode("Can you check if the PISTON strike is in Vigan?")
        self.assertEqual(mode, "news_followup")

    def test_classify_weather_followup(self) -> None:
        mode = classify_answer_mode("Will it rain tomorrow morning in Vigan?")
        self.assertEqual(mode, "weather_followup")

    def test_classify_travel_brief_request(self) -> None:
        mode = classify_answer_mode("Is Vigan fine for travel today?")
        self.assertEqual(mode, "travel_brief")

    def test_classify_short_followup_from_last_reply(self) -> None:
        mode = classify_answer_mode(
            "Is that in Vigan?",
            "Recent local reporting mentions a PISTON transport strike and possible disruptions.",
        )
        self.assertEqual(mode, "news_followup")

    def test_classify_journey_planning(self) -> None:
        mode = classify_answer_mode("Should I continue my trip from Manila to Vigan today?")
        self.assertEqual(mode, "journey_planning")

    def test_classify_origin_reply_after_clarification(self) -> None:
        mode = classify_answer_mode("From Manila", "Where are you traveling from?")
        self.assertEqual(mode, "journey_planning")


if __name__ == "__main__":
    unittest.main()
