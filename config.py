"""Shared configuration for the traffic prediction multi-agent system.

Every path is resolved relative to this file so the scripts and the Streamlit
app behave the same no matter which directory they are launched from.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
MODEL_DIR = ROOT_DIR / "models"

DATASET_PATH = DATASET_DIR / "traffic_data.csv"
MODEL_PATH = MODEL_DIR / "traffic_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"

# ---------------------------------------------------------------------------
# Model / data schema
# ---------------------------------------------------------------------------
FEATURES = ["vehicle_count", "avg_speed", "hour", "weather_code"]

WEATHER_OPTIONS = ("Clear", "Cloudy", "Rainy")
WEATHER_CODES = {weather: code for code, weather in enumerate(WEATHER_OPTIONS)}
WEATHER_IMPACTS = {
    "Clear": "No impact",
    "Cloudy": "Slight impact",
    "Rainy": "High impact (slower, wet roads)",
}

LEVELS = ("Low", "Medium", "High")
LEVEL_ICONS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

ROUTES = {
    "Low": "Main Road (direct route) - clear",
    "Medium": "Ring Road - slightly longer but steadier flow",
    "High": "Outer Bypass / Metro - avoid Main Road and Ring Road",
}

# ---------------------------------------------------------------------------
# Dashboard defaults
# ---------------------------------------------------------------------------
APP_TITLE = "Traffic Prediction and Congestion Management"
DEFAULT_INPUTS = {
    "vehicle_count": 80,
    "avg_speed": 40,
    "hour": 9,
    "weather": "Clear",
}

# ---------------------------------------------------------------------------
# Dataset generation / training defaults
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_SAMPLES = 2500
TEST_SIZE = 0.2
N_ESTIMATORS = 200

RUSH_HOURS = ((7, 10), (16, 19))
