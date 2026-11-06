"""Typed dictionaries describing the data exchanged between agents."""

from __future__ import annotations

from typing import TypedDict


class TrafficReport(TypedDict):
    vehicle_count: int
    avg_speed: float
    hour: int
    peak_hour: bool


class WeatherReport(TypedDict):
    weather: str
    weather_code: int
    impact: str


class PredictionResult(TypedDict):
    level: str
    confidence: float
    probabilities: dict[str, float]


class RouteRecommendation(TypedDict):
    route: str
    avoid_main_road: bool


class DecisionResult(TypedDict):
    final_status: str
    recommendation: str
    route: str


class ControllerResult(TypedDict):
    traffic: TrafficReport
    weather: WeatherReport
    prediction: PredictionResult
    route: RouteRecommendation
    decision: DecisionResult
    log: list[str]
