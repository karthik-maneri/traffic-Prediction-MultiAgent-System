"""Agent 5: combines every agent output into a final recommendation."""

from __future__ import annotations

from agents.base import BaseAgent
from agents.types import (
    DecisionResult,
    PredictionResult,
    RouteRecommendation,
    TrafficReport,
    WeatherReport,
)

_MESSAGES = {
    "Low": "Traffic is smooth. Travel normally.",
    "Medium": "Moderate traffic. Allow extra travel time.",
    "High": "Heavy congestion! Delay travel or use the alternative route.",
}


class DecisionAgent(BaseAgent):
    """Produces the final status and traveller-facing recommendation."""

    name = "Decision Making Agent"

    def decide(
        self,
        traffic: TrafficReport,
        weather: WeatherReport,
        prediction: PredictionResult,
        route: RouteRecommendation,
    ) -> DecisionResult:
        """Combine traffic, weather, prediction and route into one decision."""
        level = prediction["level"]
        message = _MESSAGES.get(level, "Traffic status unknown. Travel with caution.")

        if weather["weather"] == "Rainy" and level != "Low":
            message += " Rain is worsening conditions - drive carefully."
        elif traffic["peak_hour"] and level == "High":
            message += " It is also a peak hour, so consider postponing the trip."

        return {
            "final_status": level,
            "recommendation": message,
            "route": route["route"],
        }
