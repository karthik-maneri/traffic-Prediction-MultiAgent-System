"""Streamlit dashboard for the traffic prediction multi-agent system.

Run with:

    streamlit run app.py

Do not run this file with ``python app.py``: Streamlit scripts must be started
through the Streamlit CLI.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from config import (
    APP_TITLE,
    CONFUSION_MATRIX_PATH,
    DATASET_PATH,
    DEFAULT_INPUTS,
    LEVEL_ICONS,
    METRICS_PATH,
    MODEL_PATH,
    WEATHER_OPTIONS,
)
from agents.types import ControllerResult
from generate_data import generate_dataset
from main import Controller
from train_model import train_model


@st.cache_resource(show_spinner=False)
def get_controller() -> Controller:
    """Load the agents (and the model) once per Streamlit server process."""
    return Controller()


def ensure_assets() -> None:
    """Create the simulated dataset and model on first run if they are missing."""
    if not DATASET_PATH.exists():
        with st.spinner("Dataset not found - generating simulated data..."):
            generate_dataset()
    if not MODEL_PATH.exists():
        with st.spinner("Model not found - training the Random Forest..."):
            train_model()


def load_controller() -> Controller | None:
    """Return a ready controller or show an actionable error and stop."""
    try:
        ensure_assets()
        return get_controller()
    except Exception as exc:  # Streamlit should show a friendly message, not a traceback.
        st.error("The application could not start because its model assets are unavailable.")
        st.info(
            "Run `python generate_data.py` and `python train_model.py` from the project "
            "root, then reload this page."
        )
        with st.expander("Error details"):
            st.exception(exc)
        return None


def load_metrics() -> dict[str, Any] | None:
    """Read the training metrics written by ``train_model.py``, if present."""
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return metrics if isinstance(metrics, dict) else None


def render_result(result: ControllerResult) -> None:
    """Render one controller result: headline, advice, agent outputs and log."""
    prediction = result["prediction"]
    traffic = result["traffic"]
    weather = result["weather"]
    decision = result["decision"]
    level = prediction["level"]
    icon = LEVEL_ICONS.get(level, "⚪")

    st.subheader(f"{icon} Predicted traffic: {level} ({prediction['confidence']}% confidence)")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Congestion level", f"{icon} {level}")
    metric_columns[1].metric("Confidence", f"{prediction['confidence']:.1f}%")
    metric_columns[2].metric("Peak hour", "Yes" if traffic["peak_hour"] else "No")
    metric_columns[3].metric("Weather impact", weather["impact"])

    alert = {"High": st.error, "Medium": st.warning}.get(level, st.success)
    alert(decision["recommendation"])

    st.info(f"🛣️ Alternative route: {result['route']['route']}")

    overview_tab, agents_tab, log_tab = st.tabs(["Overview", "Agent outputs", "Communication log"])

    with overview_tab:
        probabilities = prediction.get("probabilities", {})
        if probabilities:
            probability_frame = pd.DataFrame(
                {
                    "Congestion level": list(probabilities),
                    "Probability (%)": list(probabilities.values()),
                }
            ).set_index("Congestion level")
            st.bar_chart(probability_frame)
        st.caption(
            "Predictions come from a Random Forest trained on simulated data. "
            "They are for demonstration only, not live traffic advice."
        )

    with agents_tab:
        traffic_column, weather_column, route_column = st.columns(3)
        traffic_column.markdown("**Traffic Agent**")
        traffic_column.json(traffic)
        weather_column.markdown("**Weather Agent**")
        weather_column.json(weather)
        route_column.markdown("**Route Agent**")
        route_column.json(result["route"])
        st.markdown("**Prediction Agent**")
        st.json(prediction)
        st.markdown("**Decision Agent**")
        st.json(decision)

    with log_tab:
        st.code("\n".join(result["log"]), language="text")


def render_history() -> None:
    """Show previous predictions and a small distribution chart."""
    history: list[dict[str, Any]] = st.session_state.history
    if not history:
        return

    st.divider()
    header, clear_column = st.columns([4, 1])
    header.subheader("Prediction history")
    if clear_column.button("Clear history"):
        history.clear()
        st.rerun()

    frame = pd.DataFrame(history)
    st.dataframe(frame, hide_index=True)

    counts = (
        frame["Prediction"]
        .value_counts()
        .rename_axis("Congestion level")
        .reset_index(name="Count")
    )
    st.bar_chart(counts, x="Congestion level", y="Count")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🚦", layout="wide")
    st.title(f"🚦 {APP_TITLE} - Multi-Agent System")
    st.caption("Random Forest model trained on SIMULATED data. Not live traffic.")

    if "history" not in st.session_state:
        st.session_state.history = []

    controller = load_controller()

    with st.sidebar:
        st.header("Traffic input")
        with st.form("traffic_form"):
            vehicle_count = st.slider(
                "Vehicle count", 0, 250, DEFAULT_INPUTS["vehicle_count"], step=1
            )
            avg_speed = st.slider(
                "Average speed (km/h)", 0, 100, DEFAULT_INPUTS["avg_speed"], step=1
            )
            hour = st.slider("Hour of day", 0, 23, DEFAULT_INPUTS["hour"], step=1)
            weather = st.selectbox(
                "Weather",
                WEATHER_OPTIONS,
                index=WEATHER_OPTIONS.index(DEFAULT_INPUTS["weather"]),
            )
            submitted = st.form_submit_button("🔮 Predict traffic", type="primary")

        metrics = load_metrics()
        if metrics:
            with st.expander("Model information"):
                st.write(f"Accuracy: **{metrics.get('accuracy', 'n/a')}**")
                st.write(f"F1 (weighted): **{metrics.get('f1_weighted', 'n/a')}**")
                st.write(f"Trained: {metrics.get('trained_at', 'unknown')}")
                st.caption(f"Features: {', '.join(metrics.get('features', []))}")
                if CONFUSION_MATRIX_PATH.exists():
                    st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion matrix")

    if controller is None:
        st.stop()

    if submitted:
        try:
            result = controller.run(vehicle_count, avg_speed, hour, weather)
        except (ValueError, TypeError, FileNotFoundError) as exc:
            st.error(f"Could not run the prediction: {exc}")
        else:
            st.session_state.history.append(
                {
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Vehicles": vehicle_count,
                    "Speed (km/h)": avg_speed,
                    "Hour": hour,
                    "Weather": weather,
                    "Prediction": result["prediction"]["level"],
                    "Confidence (%)": result["prediction"]["confidence"],
                }
            )
            render_result(result)
    elif not st.session_state.history:
        st.info("Set the traffic inputs in the sidebar, then click **Predict traffic**.")

    render_history()


def _running_inside_streamlit() -> bool:
    """Return True when this module is executed by the Streamlit runtime."""
    try:
        from streamlit import runtime
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:  # pragma: no cover - Streamlit is a requirement
        return False

    if not runtime.exists():
        return False
    return get_script_run_ctx() is not None


if __name__ == "__main__":
    if _running_inside_streamlit():
        main()
    else:
        print(
            "This is a Streamlit application.\n"
            "Start it from the project root with:\n\n"
            "    streamlit run app.py\n"
        )
        raise SystemExit(1)
