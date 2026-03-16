import unittest

from app.travel_intelligence import filter_relevant_news_items


class TravelIntelligenceTests(unittest.TestCase):
    def test_filter_relevant_news_items_drops_low_signal_program_news(self) -> None:
        headlines = [
            {
                "title": "Cebu City Launches Job Program For Senior Citizens",
                "snippet": "A local livelihood initiative will support senior citizens in Cebu City.",
                "link": "https://example.com/job-program",
            },
            {
                "title": "Airport shuttle delays after runway works",
                "snippet": "Mactan terminal transfers are taking longer than usual during overnight repairs.",
                "link": "https://example.com/airport-delays",
            },
        ]

        filtered = filter_relevant_news_items("Cebu", headlines)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["title"], "Airport shuttle delays after runway works")

    def test_filter_relevant_news_items_keeps_event_news_when_place_matches(self) -> None:
        headlines = [
            {
                "title": "Foreign, local aces gear up for fierce challenge at IRONMAN 70.3 Davao",
                "snippet": "The race weekend is expected to bring extra road activity in Davao.",
                "link": "https://example.com/ironman",
            }
        ]

        filtered = filter_relevant_news_items("Davao", headlines)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["title"], headlines[0]["title"])


if __name__ == "__main__":
    unittest.main()
