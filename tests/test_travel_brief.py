import unittest
from unittest.mock import patch

from app.travel_brief import build_travel_brief


class TravelBriefTests(unittest.TestCase):
    def test_brief_is_grounded_in_weather_and_news_details(self) -> None:
        weather_summary = {
            "place_label": "Cebu, Philippines",
            "current": {
                "weather_code": 63,
                "weather_text": "Moderate rain",
            },
            "day": {
                "label": "2026-03-16",
                "tmin_c": 24,
                "tmax_c": 30,
                "precip_mm": 12,
                "uv_index_max": 6,
                "wind_speed_max_kmh": 32,
            },
        }
        headlines = [
            {
                "title": "Airport shuttle delays after runway works",
                "snippet": "Mactan terminal transfers are taking longer than usual during overnight repairs.",
                "source": "Local News",
                "date": "2026-03-16T05:00:00+00:00",
                "link": "https://example.com/airport-delays",
            }
        ]

        with patch("app.travel_brief.get_weather_summary", return_value=(weather_summary, None)):
            with patch("app.travel_brief.get_news_items", return_value=(headlines, "")):
                brief, err = build_travel_brief("Cebu")

        self.assertEqual(err, "")
        self.assertEqual(brief["risk_level"], "high")
        self.assertIn("moderate rain", brief["final"].lower())
        self.assertIn("24-30", brief["final"])
        self.assertIn("Airport shuttle delays after runway works", brief["final"])
        self.assertEqual(brief["news_items"][0]["snippet"], headlines[0]["snippet"])
        self.assertTrue(
            any("Airport shuttle delays after runway works" in item for item in brief["travel_advice"]),
            brief["travel_advice"],
        )
        self.assertTrue(brief["weather_reasons"])
        self.assertTrue(brief["news_reasons"])

    def test_brief_does_not_claim_quiet_news_when_news_fetch_failed(self) -> None:
        weather_summary = {
            "place_label": "Cebu, Philippines",
            "current": {
                "weather_code": 1,
                "weather_text": "Mainly clear",
            },
            "day": {
                "label": "2026-03-16",
                "tmin_c": 25,
                "tmax_c": 31,
                "precip_mm": 0,
                "uv_index_max": 5,
                "wind_speed_max_kmh": 12,
            },
        }

        with patch("app.travel_brief.get_weather_summary", return_value=(weather_summary, None)):
            with patch("app.travel_brief.get_news_items", return_value=([], "provider unavailable")):
                brief, err = build_travel_brief("Cebu")

        self.assertEqual(err, "provider unavailable")
        self.assertIn("could not be confirmed", brief["final"].lower())
        self.assertFalse(
            any("did not surface a major traveler-facing disruption" in item.lower() for item in brief["travel_advice"]),
            brief["travel_advice"],
        )
        self.assertTrue(
            any("forecast is mainly clear around 25-31" in item.lower() for item in brief["travel_advice"]),
            brief["travel_advice"],
        )


if __name__ == "__main__":
    unittest.main()
