"""Orchestrates the five agents and exposes the ``Controller`` used by the app."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from agents.decision_agent import DecisionAgent
from agents.prediction_agent import PredictionAgent
from agents.route_agent import RouteAgent
from agents.traffic_agent import TrafficAgent
from agents.types import ControllerResult
from agents.weather_agent import WeatherAgent


class Controller:
    """Runs the agents in order and records their message log.

    ``run`` is stateless: every call starts a fresh log, which makes it safe to
    share a single cached controller across Streamlit sessions.
    """

    def __init__(
        self,
        traffic_agent: TrafficAgent | None = None,
        weather_agent: WeatherAgent | None = None,
        prediction_agent: PredictionAgent | None = None,
        route_agent: RouteAgent | None = None,
        decision_agent: DecisionAgent | None = None,
    ) -> None:
        self.traffic = traffic_agent or TrafficAgent()
        self.weather = weather_agent or WeatherAgent()
        self.prediction = prediction_agent or PredictionAgent()
        self.route = route_agent or RouteAgent()
        self.decision = decision_agent or DecisionAgent()

    @staticmethod
    def _message(agent: object, output: object) -> str:
        name = getattr(agent, "name", type(agent).__name__)
        return f"[{datetime.now():%H:%M:%S}] {name}: {output}"

    def run(
        self,
        vehicle_count: int,
        avg_speed: float,
        hour: int,
        weather: str,
    ) -> ControllerResult:
        """Execute the full agent pipeline and return every intermediate result."""
        log: list[str] = []

        traffic = self.traffic.collect(vehicle_count, avg_speed, hour)
        log.append(self._message(self.traffic, traffic))

        weather_report = self.weather.process(weather)
        log.append(self._message(self.weather, weather_report))

        prediction = self.prediction.predict(traffic, weather_report)
        log.append(self._message(self.prediction, prediction))

        route = self.route.recommend(prediction["level"])
        log.append(self._message(self.route, route))

        decision = self.decision.decide(traffic, weather_report, prediction, route)
        log.append(self._message(self.decision, decision))

        return {
            "traffic": traffic,
            "weather": weather_report,
            "prediction": prediction,
            "route": route,
            "decision": decision,
            "log": log,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the traffic prediction agent pipeline once from the terminal."
    )
    parser.add_argument("--vehicle-count", type=int, default=150, help="vehicles observed (0-500)")
    parser.add_argument("--avg-speed", type=float, default=20, help="average speed in km/h (0-150)")
    parser.add_argument("--hour", type=int, default=8, help="hour of day (0-23)")
    parser.add_argument(
        "--weather",
        choices=("Clear", "Cloudy", "Rainy"),
        default="Rainy",
        help="weather condition",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        result = Controller().run(
            vehicle_count=args.vehicle_count,
            avg_speed=args.avg_speed,
            hour=args.hour,
            weather=args.weather,
        )
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\n".join(result["log"]))
    print("\nFINAL:", json.dumps(result["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
