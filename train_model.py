"""Train the Random Forest congestion classifier and save all artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from config import (
    CONFUSION_MATRIX_PATH,
    DATASET_PATH,
    FEATURES,
    LEVELS,
    METRICS_PATH,
    MODEL_PATH,
    N_ESTIMATORS,
    RANDOM_SEED,
    TEST_SIZE,
    WEATHER_CODES,
)


def load_training_data(dataset_path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """Read and validate the simulated dataset, returning model-ready features."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python generate_data.py` first."
        )

    required = {"vehicle_count", "avg_speed", "hour", "weather", "congestion_level"}
    dataframe = pd.read_csv(path)
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    dataframe = dataframe.drop_duplicates().dropna().copy()
    unknown_weather = set(dataframe["weather"]).difference(WEATHER_CODES)
    if unknown_weather:
        raise ValueError(f"Unknown weather values: {', '.join(sorted(unknown_weather))}")

    unknown_levels = set(dataframe["congestion_level"]).difference(LEVELS)
    if unknown_levels:
        raise ValueError(f"Unknown congestion levels: {', '.join(sorted(unknown_levels))}")

    dataframe["weather_code"] = dataframe["weather"].map(WEATHER_CODES)
    return dataframe


def train_model(
    dataset_path: str | Path = DATASET_PATH,
    model_path: str | Path = MODEL_PATH,
    metrics_path: str | Path = METRICS_PATH,
    confusion_matrix_path: str | Path = CONFUSION_MATRIX_PATH,
) -> dict[str, object]:
    """Train the model, persist it with metrics, and return the metrics."""
    dataframe = load_training_data(dataset_path)
    features = dataframe[FEATURES]
    target = dataframe["congestion_level"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=target,
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ).fit(x_train, y_train)

    predictions = model.predict(x_test)
    labels = list(LEVELS)

    metrics: dict[str, object] = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_weighted": round(
            float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 4
        ),
        "recall_weighted": round(
            float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 4
        ),
        "f1_weighted": round(
            float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4
        ),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "features": FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    model_file = Path(model_path)
    metrics_file = Path(metrics_path)
    matrix_file = Path(confusion_matrix_path)
    for file in (model_file, metrics_file, matrix_file):
        file.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "classes": labels,
            "trained_at": metrics["trained_at"],
        },
        model_file,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix(y_test, predictions, labels=labels),
        display_labels=labels,
    )
    display.plot(cmap="Blues", colorbar=False)
    plt.title("Congestion level confusion matrix")
    plt.tight_layout()
    plt.savefig(matrix_file, dpi=120)
    plt.close()

    metrics_file.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    metrics = train_model()
    print("Training complete")
    print(f"  Accuracy : {metrics['accuracy']}")
    print(f"  Precision: {metrics['precision_weighted']} (weighted)")
    print(f"  Recall   : {metrics['recall_weighted']} (weighted)")
    print(f"  F1-score : {metrics['f1_weighted']} (weighted)")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
