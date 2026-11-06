"""Generate the SIMULATED traffic dataset used to train the model.

The data is synthetic and must not be presented as real traffic observations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import DATASET_PATH, N_SAMPLES, RANDOM_SEED, WEATHER_OPTIONS

WEATHER_PROBABILITIES = [0.5, 0.3, 0.2]
WEATHER_IMPACT = {"Clear": 0.0, "Cloudy": 0.03, "Rainy": 0.1}


def generate_dataset(
    output_path: str | Path = DATASET_PATH,
    n_samples: int = N_SAMPLES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create a synthetic traffic dataset and write it to ``output_path``."""
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than zero")

    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, n_samples)
    weather = rng.choice(WEATHER_OPTIONS, n_samples, p=WEATHER_PROBABILITIES)

    rush = ((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 19))
    vehicle_count = np.clip(rng.normal(70, 35, n_samples) + rush * 50, 5, 220).astype(int)
    speed = np.clip(70 - vehicle_count * 0.25 + rng.normal(0, 8, n_samples), 5, 90).round(1)

    weather_impact = np.array([WEATHER_IMPACT[value] for value in weather])
    score = (
        0.5 * vehicle_count / 200
        + 0.4 * (1 - speed / 90)
        + 0.12 * rush
        + weather_impact
        + rng.normal(0, 0.04, n_samples)
    )
    level = np.where(score < 0.35, "Low", np.where(score < 0.55, "Medium", "High"))

    dataframe = pd.DataFrame(
        {
            "vehicle_count": vehicle_count,
            "avg_speed": speed,
            "hour": hour,
            "weather": weather,
            "congestion_level": level,
        }
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(destination, index=False)
    return dataframe


def main() -> int:
    dataframe = generate_dataset()
    print(f"Saved {len(dataframe)} rows (simulated) to {DATASET_PATH}")
    print(dataframe["congestion_level"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
