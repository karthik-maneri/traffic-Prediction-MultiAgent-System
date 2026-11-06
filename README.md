# 🚦 Traffic Prediction and Congestion Management (Multi-Agent System)

A self-contained demo that predicts traffic congestion with a Random Forest
classifier and passes the result through five cooperating agents. The dashboard
is built with Streamlit.

> **Important:** all traffic data is **simulated** (`generate_data.py`). This
> project does not use live traffic, maps, or weather feeds and must not be used
> for real travel decisions.

## Features

- **Random Forest classifier** for `Low` / `Medium` / `High` congestion.
- **Five cooperating agents:** traffic, weather, prediction, route, decision.
- **Streamlit dashboard** with confidence, per-class probabilities, agent
  outputs, communication log, and prediction history.
- **Automatic first-run setup:** if the dataset or model is missing, the
  dashboard generates/trains them on launch.
- **Path-independent scripts:** everything resolves from the project root, even
  when launched from another directory.
- **Typed agent contracts, input validation, and a 17-test suite.**

## Quick start

Requires **Python 3.10+** (tested on Python 3.15).

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the dashboard
streamlit run app.py
```

Open the URL Streamlit prints (usually <http://localhost:8501>). On first launch
the app creates `dataset/traffic_data.csv` and `models/traffic_model.pkl` if they
are missing.

> ⚠️ Run the dashboard with `streamlit run app.py`, **not** `python app.py`.
> Running a Streamlit file as a normal Python script is the most common setup
> error.

### Manual pipeline (optional)

```bash
python generate_data.py     # create the simulated dataset
python train_model.py       # train, evaluate, and save the model + metrics
python main.py              # run all five agents once in the terminal
streamlit run app.py        # launch the dashboard
```

## Using the dashboard

1. Set **vehicle count**, **average speed**, **hour**, and **weather** in the
   sidebar.
2. Click **🔮 Predict traffic**.
3. Review the congestion level, confidence, recommendation, alternate route,
   agent outputs, and communication log.
4. Re-run predictions with different inputs; history is kept for the session and
   can be cleared from the history section.

## Command-line usage

`main.py` accepts the same inputs as the dashboard:

```bash
python main.py --vehicle-count 150 --avg-speed 20 --hour 8 --weather Rainy
```

| Argument | Range | Default |
| --- | --- | --- |
| `--vehicle-count` | 0–500 | 150 |
| `--avg-speed` | 0–150 km/h | 20 |
| `--hour` | 0–23 | 8 |
| `--weather` | `Clear`, `Cloudy`, `Rainy` | `Rainy` |

## Model and artifacts

The classifier uses four features:

`vehicle_count`, `avg_speed`, `hour`, `weather_code` (Clear=0, Cloudy=1, Rainy=2).

Current evaluation on a stratified 20% hold-out split of the simulated data:

| Metric | Score |
| --- | --- |
| Accuracy | 0.8540 |
| Precision (weighted) | 0.8534 |
| Recall (weighted) | 0.8540 |
| F1 (weighted) | 0.8536 |

Generated files:

| Path | Description |
| --- | --- |
| `dataset/traffic_data.csv` | 2,500 simulated traffic rows |
| `models/traffic_model.pkl` | Model bundle (estimator + feature metadata) |
| `models/metrics.json` | Latest evaluation metrics and training timestamp |
| `models/confusion_matrix.png` | Hold-out confusion matrix |

Regenerate everything with:

```bash
python generate_data.py && python train_model.py
```

## Project structure

```text
.
├── app.py                  # Streamlit dashboard
├── main.py                 # Controller + CLI entry point
├── config.py               # Paths, schema, levels, routes, defaults
├── generate_data.py        # Simulated dataset generator
├── train_model.py          # Training, evaluation, artifact persistence
├── agents/
│   ├── base.py             # Shared agent base class
│   ├── types.py            # TypedDict contracts between agents
│   ├── traffic_agent.py    # Validates traffic measurements
│   ├── weather_agent.py    # Encodes weather + impact
│   ├── prediction_agent.py # Loads model and predicts
│   ├── route_agent.py      # Example route recommendations
│   └── decision_agent.py   # Final traveller recommendation
├── tests/                  # unittest suite (agents, pipeline, dashboard)
├── dataset/                # Generated data
├── models/                 # Generated model and metrics
├── AGENTS.md               # Contributor / coding-agent guide
└── GUIDE.md                # Detailed user guide & troubleshooting
```

## Agent flow

```text
User input
   ↓
TrafficAgent → WeatherAgent → PredictionAgent → RouteAgent → DecisionAgent
   ↓              ↓                ↓                ↓             ↓
validated      encoded        congestion       alternative    final advice
measurements   weather        level + probs    route          + warnings
                                   ↓
                         Controller returns all
                         outputs + communication log
```

`Controller.run()` is stateless: it returns a fresh log on every call, which
makes it safe to cache a single controller across Streamlit sessions.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers agent validation, the controller pipeline, dataset generation,
training/prediction round-trips, and a Streamlit smoke test that submits a
prediction.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `streamlit: command not found` | Activate the virtual environment and run `pip install -r requirements.txt`. |
| App shows "model assets are unavailable" | Run `python generate_data.py && python train_model.py`, then reload. |
| `ModuleNotFoundError` for `agents`/`config` | Run commands from the project root. |
| Nothing happens with `python app.py` | Use `streamlit run app.py`. |
| Port 8501 is busy | `streamlit run app.py --server.port 8502`. |
| Stale model after changing features | Retrain with `python train_model.py` and restart Streamlit. |

See [GUIDE.md](GUIDE.md) for a longer walkthrough and [AGENTS.md](AGENTS.md) for
the architecture and contribution conventions.

## Limitations

- Data, routes, and weather impacts are simulated/illustrative.
- No live map integration or real-time traffic feed.
- The model is a demo classifier, not a safety-critical forecasting system.
