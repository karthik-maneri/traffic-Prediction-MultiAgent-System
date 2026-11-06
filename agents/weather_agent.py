"""Agent 2: turns a weather label into model features and a human impact."""

from __future__ import annotations

from agents.base import BaseAgent
from agents.types import WeatherReport
from config import WEATHER_CODES, WEATHER_IMPACTS


class WeatherAgent(BaseAgent):
    """Encodes the weather condition and describes its traffic impact."""

    name = "Weather Monitoring Agent"

    def process(self, weather: str) -> WeatherReport:
        """Return the encoded weather report.

        Raises:
            ValueError: if ``weather`` is not one of the supported conditions.
        """
        if weather not in WEATHER_CODES:
            supported = ", ".join(WEATHER_CODES)
            raise ValueError(f"weather must be one of: {supported}. Got {weather!r}.")

        return {
            "weather": weather,
            "weather_code": WEATHER_CODES[weather],
            "impact": WEATHER_IMPACTS[weather],
        }
