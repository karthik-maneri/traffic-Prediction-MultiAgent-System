"""Agent 1: validates and normalises the raw traffic measurements."""

from __future__ import annotations

from numbers import Real

from agents.base import BaseAgent
from agents.types import TrafficReport
from config import RUSH_HOURS


class TrafficAgent(BaseAgent):
    """Collects the raw traffic input and annotates peak-hour status."""

    name = "Traffic Monitoring Agent"

    @staticmethod
    def _validate(name: str, value: Real, minimum: Real, maximum: Real) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a number, got {type(value).__name__}")
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")

    def collect(self, vehicle_count: int, avg_speed: float, hour: int) -> TrafficReport:
        """Validate measurements and return a traffic report.

        Raises:
            TypeError: if a value is not numeric.
            ValueError: if a value is outside its accepted range.
        """
        self._validate("vehicle_count", vehicle_count, 0, 500)
        self._validate("avg_speed", avg_speed, 0, 150)
        self._validate("hour", hour, 0, 23)

        peak_hour = any(start <= hour <= end for start, end in RUSH_HOURS)
        return {
            "vehicle_count": int(vehicle_count),
            "avg_speed": float(avg_speed),
            "hour": int(hour),
            "peak_hour": peak_hour,
        }
