# AGENTS.md — Contributor & Coding-Agent Guide

This file explains how the project is organised, which invariants must be kept,
and how to verify changes. Read it before modifying the pipeline or dashboard.

## Purpose

A teaching/demo multi-agent traffic congestion system:

- Synthetic data → Random Forest classifier → five cooperating agents →
  Streamlit dashboard and CLI.
- All data is **simulated**. Never describe it as real or live traffic.
- Routes are example strings; no map/geospatial service is connected.

## Commands

Run from the project root.

```bash
# Install
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Full pipeline
python generate_data.py
python train_model.py
python main.py --vehicle-count 150 --avg-speed 20 --hour 8 --weather Rainy

# Dashboard (NOT `python app.py`)
streamlit run app.py

# Tests
python -m unittest discover -s tests -v
```

The dashboard auto-creates the dataset and model when they are missing, so a
fresh clone can be started with only `streamlit run app.py`.

## Repository map

| File | Responsibility |
| --- | --- |
| `config.py` | Single source of truth: paths, feature names, weather codes, levels, routes, defaults, seeds. |
| `generate_data.py` | Deterministic (`seed=42`) simulated dataset with rush-hour and weather effects. |
| `train_model.py` | Data validation, stratified split, Random Forest training, metrics JSON, confusion matrix, model bundle. |
| `main.py` | `Controller` orchestration plus CLI (`argparse`). |
| `agents/base.py` | `BaseAgent` with shared `name`. |
| `agents/types.py` | `TypedDict` contracts exchanged between agents. |
| `agents/traffic_agent.py` | Validates ranges/types and flags peak hours. |
| `agents/weather_agent.py` | Maps weather to a model code and impact description. |
| `agents/prediction_agent.py` | Loads the persisted model and returns level, confidence, probabilities. |
| `agents/route_agent.py` | Maps congestion level to an example route. |
| `agents/decision_agent.py` | Produces final status, advice, and warnings. |
| `app.py` | Streamlit UI, auto-bootstrap, cached controller, history. |
| `tests/` | `unittest` suite, including a Streamlit `AppTest` smoke test. |

## Data-flow contract

```text
TrafficAgent.collect(vehicle_count, avg_speed, hour) -> TrafficReport
WeatherAgent.process(weather)                       -> WeatherReport
PredictionAgent.predict(traffic, weather)           -> PredictionResult
RouteAgent.recommend(level)                         -> RouteRecommendation
DecisionAgent.decide(traffic, weather, prediction,
                     route)                         -> DecisionResult
Controller.run(...)                                 -> ControllerResult (+ log)
```

The key names in `agents/types.py` and the dashboard are part of the public
contract. If you rename a key, update:

1. the `TypedDict`;
2. the producing agent;
3. every consumer (`main.py`, `app.py`, tests);
4. docs that mention the field.

## Invariants

- **Paths come from `config.py`.** Do not add relative paths such as
  `"models/traffic_model.pkl"`; scripts must work from any working directory.
- **`Controller.run()` must stay stateless.** Build the log locally and return
  it. The Streamlit app caches one controller process-wide with
  `st.cache_resource`; storing per-run state on the controller would leak data
  between sessions.
- **Feature order matches training.** `FEATURES` in `config.py` is used by both
  training and inference.
- **Weather codes stay aligned.** `WEATHER_CODES` must match what generated the
  dataset.
- **Prediction confidence is a percentage** (`0–100`, one decimal). The field is
  displayed directly in the UI.
- **All agent outputs must be JSON-serialisable** because the dashboard uses
  `st.json`.
- **Validate at the boundary.** Agents raise `ValueError`/`TypeError` with
  actionable messages; the app catches these and renders a friendly error.
- **Model artifacts are replaceable.** Loading supports both the current bundle
  (`{"model", "features", ...}`) and the legacy bare estimator pickle.
- **Do not call `st.set_page_config` at import time.** It belongs at the start of
  `main()` so `python app.py` can print a helpful message without Streamlit
  warnings.
- **`python app.py` must not look like a working launch.** The bottom guard
  detects the Streamlit runtime and otherwise prints the correct command.

## Coding conventions

- Python 3.10+ with `from __future__ import annotations`.
- Type hints on public functions; docstrings on modules and non-trivial methods.
- `pathlib.Path` for filesystem work; `encoding="utf-8"` for text writes.
- Keep functions small and side effects explicit.
- Prefer configuration over duplicated constants.
- Streamlit: use `st.form` for multi-input submission, `st.cache_resource` for
  model/agent loading, and always provide an empty-state message.

## Change recipes

### Add or change a model feature

1. Add it to `FEATURES` and any encoding map in `config.py`.
2. Produce it in `generate_data.py` (or the relevant agent).
3. Pass it through `agents/types.py` and `PredictionAgent.predict`.
4. Retrain (`python train_model.py`).
5. Update tests and docs (`README.md`, `GUIDE.md`).

### Add an agent

1. Create `agents/<name>_agent.py` subclassing `BaseAgent`.
2. Add a `TypedDict` output in `agents/types.py`.
3. Instantiate it in `Controller.__init__` (with dependency injection support)
   and call it in `run`, appending to the local log.
4. Render its output in the app and CLI.
5. Add unit tests and update `agents/__init__.py`, this file, and docs.

### Change UI copy

Update `app.py` labels/explanations and the relevant docs. Keep the simulated-data
disclaimer visible and avoid implying real-world accuracy.

## Testing expectations

- Run `python -m unittest discover -s tests -v` before finishing.
- Add tests for new validation branches and any change to the pipeline contract.
- The Streamlit smoke test (`tests/test_app.py`) must keep passing; it boots the
  real app, submits a form, and asserts the result renders without exceptions.
- Training tests use temporary directories and must not overwrite `models/`.

## Troubleshooting for contributors

| Symptom | Likely cause / fix |
| --- | --- |
| `ModuleNotFoundError: config` / `agents` | Run from the project root. |
| `FileNotFoundError` for the model | Run `python train_model.py` or start the app to auto-train. |
| `NameError` after refactoring imports | Check `main.py` imports and `agents/__init__.py`. |
| Streamlit form button not found in tests | Use `AppTest.from_file(...)`, then `.button[0].click().run()`. |
| Stale predictions | Delete `models/` and rerun training, or retrain after feature changes. |

## Definition of done

- [ ] `python -m unittest discover -s tests -v` passes.
- [ ] `python main.py` produces a full 5-agent log and final decision.
- [ ] `streamlit run app.py` renders and predicts (or `AppTest` passes).
- [ ] No relative asset paths or mutable controller state were introduced.
- [ ] README/GUIDE/AGENTS remain accurate if behaviour or commands changed.
