"""Small shared base class for the cooperating agents."""

from __future__ import annotations

from abc import ABC


class BaseAgent(ABC):
    """Base class that gives every agent a human-readable name.

    Agents intentionally expose different methods (``collect``, ``process``,
    ``predict`` ...), so no abstract method is declared here.
    """

    name: str = "Base Agent"

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{type(self).__name__}(name={self.name!r})"
