import unittest

from app.travel_intelligence import classify_risk_level, score_weather_risk


class TravelIntelligenceTests(unittest.TestCase):
    def test_score_weather_risk_flags_strong_signals(self) -> None:
        summary = {
            "current": {"weather_code": 63},
            "day": {
                "wind_speed_max_kmh": 55,
                "precip_mm": 12,
                "tmax_c": 33,
                "tmin_c": 24,
            },
        }

        score, reasons = score_weather_risk(summary)

        self.assertGreaterEqual(score, 4)
        self.assertTrue(reasons)

    def test_classify_risk_level_returns_expected_band(self) -> None:
        self.assertEqual(classify_risk_level(0), "low")
        self.assertEqual(classify_risk_level(2), "medium")
        self.assertEqual(classify_risk_level(5), "high")


if __name__ == "__main__":
    unittest.main()
