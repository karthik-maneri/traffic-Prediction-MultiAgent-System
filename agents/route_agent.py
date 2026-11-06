"""Agent 4: maps a congestion level to an example alternative route."""

from __future__ import annotations

from agents.base import BaseAgent
from agents.types import RouteRecommendation
from config import LEVELS, ROUTES


class RouteAgent(BaseAgent):
    """Suggests a route for the predicted congestion level.

    The routes are examples only; no live map or road-network data is used.
    """

    name = "Route Optimization Agent"

    def recommend(self, level: str) -> RouteRecommendation:
        """Return a route recommendation for ``level``.

        Raises:
            ValueError: if ``level`` is not a known congestion level.
        """
        if level not in ROUTES:
            raise ValueError(f"level must be one of: {', '.join(LEVELS)}. Got {level!r}.")

        return {
            "route": ROUTES[level],
            "avoid_main_road": level != "Low",
        }
