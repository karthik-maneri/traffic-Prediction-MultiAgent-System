"""Agent 3: runs the trained Random Forest model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from agents.base import BaseAgent
from agents.types import PredictionResult, TrafficReport, WeatherReport
from config import FEATURES, MODEL_PATH


class PredictionAgent(BaseAgent):
    """Loads the persisted model once and predicts congestion for new inputs."""

    name = "Traffic Prediction Agent"

    def __init__(self, model_path: str | Path = MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Run `python train_model.py` (or start the Streamlit app) first."
            )

        artifact: Any = joblib.load(self.model_path)
        if isinstance(artifact, dict) and "model" in artifact:
            # Current artifact format: keeps metadata next to the estimator.
            self.model = artifact["model"]
            self.features: list[str] = list(artifact.get("features", FEATURES))
        else:
            # Backwards compatibility with the original bare-estimator pickle.
            self.model = artifact
            self.features = list(FEATURES)

    def predict(self, traffic: TrafficReport, weather: WeatherReport) -> PredictionResult:
        """Predict the congestion level plus per-class probabilities (percent)."""
        row = pd.DataFrame(
            [
                {
                    "vehicle_count": traffic["vehicle_count"],
                    "avg_speed": traffic["avg_speed"],
                    "hour": traffic["hour"],
                    "weather_code": weather["weather_code"],
                }
            ],
            columns=self.features,
        )

        probabilities = self.model.predict_proba(row)[0]
        classes = [str(label) for label in self.model.classes_]
        best_index = int(probabilities.argmax())

        return {
            "level": classes[best_index],
            "confidence": round(float(probabilities[best_index]) * 100, 1),
            "probabilities": {
                label: round(float(probability) * 100, 1)
                for label, probability in zip(classes, probabilities)
            },
        }
