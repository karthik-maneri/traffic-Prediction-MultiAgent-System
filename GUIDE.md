# GUIDE.md — User & Operations Guide

A practical walkthrough of running, using, retraining, and troubleshooting the
traffic prediction multi-agent system.

## 1. What the system does

You provide four inputs:

| Input | Meaning | Typical range |
| --- | --- | --- |
| Vehicle count | Number of vehicles observed | 0–250 on the slider (agent accepts 0–500) |
| Average speed | Average speed in km/h | 0–100 on the slider (agent accepts 0–150) |
| Hour of day | Local hour, 0–23 | 0–23 |
| Weather | `Clear`, `Cloudy`, or `Rainy` | — |

The pipeline then:

1. **Traffic Agent** validates the measurements and flags rush hours
   (07:00–10:00 and 16:00–19:00).
2. **Weather Agent** encodes weather for the model and describes its impact.
3. **Prediction Agent** predicts `Low`, `Medium`, or `High` congestion with a
   confidence score and per-class probabilities.
4. **Route Agent** suggests an example route for that level.
5. **Decision Agent** combines everything into a final recommendation, adding
   rain and peak-hour warnings when relevant.

All data is simulated. Routes are illustrative and no live map or traffic service
is used.

## 2. Installation

Requires Python 3.10 or newer.

### macOS / Linux

```bash
cd Traffic_Prediction_MultiAgent_System
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell / CMD)

```bat
cd Traffic_Prediction_MultiAgent_System
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the install:

```bash
python -c "import pandas, sklearn, streamlit; print('dependencies ready')"
```

## 3. Running the dashboard

```bash
streamlit run app.py
```

Streamlit prints a local URL, normally <http://localhost:8501>. Press `Ctrl+C` in
the terminal to stop it.

### First-run behaviour

- If `dataset/traffic_data.csv` is missing, the app generates 2,500 simulated
  rows.
- If `models/traffic_model.pkl` is missing, the app trains the Random Forest and
  saves the model, metrics, and confusion matrix.
- This can take a few seconds on the first launch. Later launches reuse the
  cached model.

### Using the dashboard

1. Adjust the sidebar sliders/select box.
2. Click **🔮 Predict traffic**.
3. Read the headline and the four metrics (level, confidence, peak hour, weather
   impact).
4. Open **Overview** for the probability chart, **Agent outputs** for each
   agent's JSON, and **Communication log** for the ordered message trace.
5. Repeat predictions; each one is appended to the session's prediction history.
6. Use **Clear history** to reset the history table and chart.

> **Do not use `python app.py`.** Streamlit apps must be launched with
> `streamlit run app.py`. If you run `python app.py`, the file now prints a clear
> reminder instead of failing with Streamlit warnings.

## 4. Running the terminal pipeline

```bash
python main.py --vehicle-count 150 --avg-speed 20 --hour 8 --weather Rainy
```

Defaults are used when flags are omitted:

```bash
python main.py
```

The output contains five timestamped agent messages followed by a JSON final
decision. Invalid values exit with a clear error.

## 5. Manual data and training workflow

Use this if you want to inspect or regenerate artifacts:

```bash
python generate_data.py
python train_model.py
```

`generate_data.py` writes `dataset/traffic_data.csv` and prints the class
distribution. `train_model.py` writes:

- `models/traffic_model.pkl` — model bundle with feature metadata;
- `models/metrics.json` — accuracy, weighted precision/recall/F1, row counts,
  timestamp;
- `models/confusion_matrix.png` — hold-out confusion matrix.

Expected current metrics on the default simulated data:

```text
Accuracy : 0.854
Precision: 0.8534 (weighted)
Recall   : 0.854 (weighted)
F1-score : 0.8536 (weighted)
```

Metrics are deterministic for the default seed and split.

### Retraining after changing data or features

1. Update `config.py` (`FEATURES`, encodings, or data constants).
2. Update `generate_data.py` / agents as needed.
3. Run `python generate_data.py` if the dataset schema changed.
4. Run `python train_model.py`.
5. Restart the Streamlit server (cached resources are cleared on restart).
6. Update the metric table in `README.md` if numbers changed.

## 6. Tests

```bash
python -m unittest discover -s tests -v
```

What is covered:

- `test_agents.py` — validation, weather encoding, route mapping, decision
  warnings, controller logging/statelessness.
- `test_pipeline.py` — dataset generation, model training, artifact round-trip,
  prediction agent, missing-model error.
- `test_app.py` — boots the real Streamlit app with `AppTest`, submits a
  prediction, and asserts the UI renders without exceptions.

Run the tests after any change. They use temporary directories for training and
do not modify the checked-in artifacts.

## 7. Troubleshooting

### `streamlit: command not found`

The virtual environment is not active or dependencies are not installed.

```bash
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'config'` (or `agents`)

Run commands from the project root (the directory containing `app.py`). The
scripts use absolute paths internally, but Python still needs the project root on
`sys.path`. If needed:

```bash
cd /path/to/Traffic_Prediction_MultiAgent_System
```

### "The application could not start because its model assets are unavailable"

The app could not create the dataset/model. Run the manual workflow and read the
error shown under **Error details**:

```bash
python generate_data.py
python train_model.py
```

Common causes: read-only project directory, incompatible scikit-learn version
for an old pickle, or a corrupted dataset. Deleting `models/traffic_model.pkl`
and retraining fixes the last one.

### `python app.py` prints a message and exits

That is intentional. Use:

```bash
streamlit run app.py
```

### Port already in use

```bash
streamlit run app.py --server.port 8502
```

### The page opens but prediction never runs

Click the **🔮 Predict traffic** button inside the sidebar form. Sliders inside a
form only take effect after submission.

### Predictions look stale

Restart Streamlit and/or retrain:

```bash
python train_model.py
streamlit run app.py
```

### Matplotlib font-cache warning on first training

Harmless. It happens once while Matplotlib builds its font cache.

## 8. Project layout and extension points

See [AGENTS.md](AGENTS.md) for the architecture, data-flow contract, invariants,
and step-by-step recipes for adding features or agents.

Key files:

- `config.py` — all shared constants and paths.
- `agents/` — one file per agent; outputs typed in `agents/types.py`.
- `main.py` — orchestration and CLI.
- `app.py` — dashboard.
- `tests/` — automated checks.

## 9. Limitations and responsible use

- Simulated data only; no live traffic, map, or weather API.
- Example routes are strings, not navigation instructions.
- Accuracy reflects synthetic patterns and will not transfer to real roads
  without real data and validation.
- Not suitable for safety-critical or production traffic management.
