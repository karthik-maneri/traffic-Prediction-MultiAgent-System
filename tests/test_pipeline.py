"""Tests for dataset generation, training and the prediction agent."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib

from agents.prediction_agent import PredictionAgent
from config import LEVELS
from generate_data import generate_dataset
from train_model import train_model


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.dataset_path = root / "dataset" / "traffic_data.csv"
        self.model_path = root / "models" / "traffic_model.pkl"
        self.metrics_path = root / "models" / "metrics.json"
        self.matrix_path = root / "models" / "confusion_matrix.png"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_generate_dataset_creates_expected_columns(self) -> None:
        dataframe = generate_dataset(self.dataset_path, n_samples=100, seed=7)
        self.assertTrue(self.dataset_path.exists())
        self.assertEqual(len(dataframe), 100)
        self.assertEqual(
            set(dataframe.columns),
            {"vehicle_count", "avg_speed", "hour", "weather", "congestion_level"},
        )

    def test_training_and_prediction_round_trip(self) -> None:
        generate_dataset(self.dataset_path, n_samples=600, seed=7)
        metrics = train_model(
            dataset_path=self.dataset_path,
            model_path=self.model_path,
            metrics_path=self.metrics_path,
            confusion_matrix_path=self.matrix_path,
        )

        self.assertTrue(self.model_path.exists())
        self.assertTrue(self.metrics_path.exists())
        self.assertTrue(self.matrix_path.exists())
        self.assertIn("accuracy", metrics)
        self.assertEqual(json.loads(self.metrics_path.read_text())["train_rows"], 480)

        artifact = joblib.load(self.model_path)
        self.assertEqual(artifact["features"][-1], "weather_code")

        agent = PredictionAgent(self.model_path)
        traffic = {"vehicle_count": 150, "avg_speed": 20, "hour": 8, "peak_hour": True}
        weather = {"weather": "Rainy", "weather_code": 2, "impact": "High impact"}
        prediction = agent.predict(traffic, weather)

        self.assertIn(prediction["level"], LEVELS)
        self.assertGreaterEqual(prediction["confidence"], 0)
        self.assertLessEqual(prediction["confidence"], 100)
        self.assertEqual(set(prediction["probabilities"]), set(LEVELS))

    def test_prediction_agent_reports_missing_model(self) -> None:
        with self.assertRaises(FileNotFoundError):
            PredictionAgent(Path(self.temp_dir.name) / "missing.pkl")


if __name__ == "__main__":
    unittest.main()
