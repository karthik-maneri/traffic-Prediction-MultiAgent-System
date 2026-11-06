"""Tests for the individual agents and the pipeline controller."""

from __future__ import annotations

import unittest

from agents.decision_agent import DecisionAgent
from agents.route_agent import RouteAgent
from agents.traffic_agent import TrafficAgent
from agents.weather_agent import WeatherAgent
from main import Controller


class StubPredictionAgent:
    """Deterministic stand-in so controller tests do not need a trained model."""

    name = "Stub Prediction Agent"

    def predict(self, traffic, weather):  # noqa: ANN001 - mirrors the real agent
        return {
            "level": "Medium",
            "confidence": 80.0,
            "probabilities": {"Low": 10.0, "Medium": 80.0, "High": 10.0},
        }


class TrafficAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = TrafficAgent()

    def test_collects_and_flags_peak_hour(self) -> None:
        report = self.agent.collect(150, 20, 8)
        self.assertEqual(report["vehicle_count"], 150)
        self.assertTrue(report["peak_hour"])

    def test_non_peak_hour(self) -> None:
        self.assertFalse(self.agent.collect(50, 60, 2)["peak_hour"])

    def test_rejects_out_of_range_values(self) -> None:
        with self.assertRaises(ValueError):
            self.agent.collect(501, 20, 8)
        with self.assertRaises(ValueError):
            self.agent.collect(50, 20, 24)

    def test_rejects_non_numeric_values(self) -> None:
        with self.assertRaises(TypeError):
            self.agent.collect("150", 20, 8)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.agent.collect(True, 20, 8)  # type: ignore[arg-type]


class WeatherAgentTests(unittest.TestCase):
    def test_encodes_every_supported_condition(self) -> None:
        agent = WeatherAgent()
        self.assertEqual(agent.process("Clear")["weather_code"], 0)
        self.assertEqual(agent.process("Rainy")["weather_code"], 2)

    def test_rejects_unknown_weather(self) -> None:
        with self.assertRaises(ValueError):
            WeatherAgent().process("Snowy")


class RouteAgentTests(unittest.TestCase):
    def test_suggests_route_for_each_level(self) -> None:
        agent = RouteAgent()
        for level in ("Low", "Medium", "High"):
            self.assertIn("route", agent.recommend(level))
        self.assertFalse(agent.recommend("Low")["avoid_main_road"])
        self.assertTrue(agent.recommend("High")["avoid_main_road"])

    def test_rejects_unknown_level(self) -> None:
        with self.assertRaises(ValueError):
            RouteAgent().recommend("Extreme")


class DecisionAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = DecisionAgent()
        self.traffic = TrafficAgent().collect(150, 20, 8)
        self.weather = WeatherAgent().process("Rainy")
        self.prediction = {"level": "High", "confidence": 90.0, "probabilities": {}}
        self.route = RouteAgent().recommend("High")

    def test_adds_rain_warning(self) -> None:
        decision = self.agent.decide(self.traffic, self.weather, self.prediction, self.route)
        self.assertEqual(decision["final_status"], "High")
        self.assertIn("Rain", decision["recommendation"])

    def test_adds_peak_hour_warning_when_dry(self) -> None:
        clear = WeatherAgent().process("Clear")
        decision = self.agent.decide(self.traffic, clear, self.prediction, self.route)
        self.assertIn("peak hour", decision["recommendation"])

    def test_low_traffic_message(self) -> None:
        prediction = {"level": "Low", "confidence": 70.0, "probabilities": {}}
        route = RouteAgent().recommend("Low")
        decision = self.agent.decide(self.traffic, self.weather, prediction, route)
        self.assertIn("smooth", decision["recommendation"])


class ControllerTests(unittest.TestCase):
    def test_pipeline_returns_all_agent_outputs_and_log(self) -> None:
        controller = Controller(prediction_agent=StubPredictionAgent())
        result = controller.run(150, 20, 8, "Rainy")

        self.assertEqual(result["prediction"]["level"], "Medium")
        self.assertEqual(result["route"]["avoid_main_road"], True)
        self.assertEqual(len(result["log"]), 5)
        self.assertEqual(result["decision"]["final_status"], "Medium")

    def test_log_is_reset_between_runs(self) -> None:
        controller = Controller(prediction_agent=StubPredictionAgent())
        controller.run(150, 20, 8, "Rainy")
        second = controller.run(50, 60, 2, "Clear")
        self.assertEqual(len(second["log"]), 5)


if __name__ == "__main__":
    unittest.main()
