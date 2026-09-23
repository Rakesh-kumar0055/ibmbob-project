# 🚦 Road Accident Prediction — India

a full end-to-end **Machine Learning** project that predicts road accidents for
any Indian State/UT and Year using the `only_road_accidents_data3.csv` dataset
from [Road Accidents in India](https://www.kaggle.com/datasets/manugupta/road-accidents-in-india)
(or synthetic data generated from its schema).

---

## Tech Stack

| Layer | Library |
|---|---|
| Data wrangling | `pandas`, `numpy` |
| ML models | `scikit-learn` (RandomForest, GradientBoosting, Ridge) |
| Visualisation | `plotly`, `matplotlib`, `seaborn` |
| **Frontend** | **Streamlit** (`app.py`) |
| API backend | **Flask** (`backend.py`) + Streamlit API client (`ui.py`) |
| Persistence | `joblib` |
| Reports | `python-docx` |

---

## Project Structure

```
road_accident_prediction/
├── data/
│   └── road_accidents.csv         ← dataset (created by _complete_project.py)
├── models/                        ← auto-created by train_model.py
│   ├── rf_model.pkl               ← best trained model
│   ├── label_encoder.pkl          ← state label encoder
│   └── feature_stats.json         ← feature metadata for the UI
├── eda_plots/                     ← auto-created by eda.py
│   ├── 01_total_by_year.png
│   ├── 02_top10_states.png
│   ├── 03_avg_by_timeslot.png
│   ├── 04_correlation_heatmap.png
│   └── 05_distribution.png
├── reports/
│   ├── road_accident_prediction_report.docx
│   └── api_outputs/
│       └── api_snapshot.json
├── _complete_project.py           ← ONE-STEP setup (installs + data + train)
├── eda.py                         ← EDA + static chart generation
├── train_model.py                 ← Model training & comparison
├── setup_data.py                  ← Helper: copies CSV from Downloads
├── backend.py                     ← Flask REST API
├── ui.py                          ← Streamlit UI backed by Flask API
├── app.py                         ← Streamlit UI (standalone, no API needed)
└── requirements.txt
```

---

## Quick Start (Recommended — One Step)

```bash
# Run this once to install packages, generate data, train model, make reports:
.venv\Scripts\python.exe _complete_project.py

# Then launch the Streamlit app:
.venv\Scripts\streamlit.exe run app.py
```

---

## Manual Step-by-Step

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2a. Use real dataset (copy from Downloads)
python setup_data.py

# 2b. OR generate synthetic dataset (if real CSV not available)
#     (handled automatically by _complete_project.py)

# 3. Optional: generate EDA plots (saved to eda_plots/)
python eda.py

# 4. Train the model (REQUIRED before launching UI)
python train_model.py

# 5a. Launch standalone Streamlit app (no API required)
streamlit run app.py

# 5b. OR launch Flask API + API-backed Streamlit UI
python backend.py          # Terminal 1 — starts on port 5000
streamlit run ui.py        # Terminal 2 — connects to Flask
```

---

## Pages (app.py — Standalone)

| Page | Description |
|---|---|
| 🏠 Home | KPI cards — total accidents, avg/year, highest-risk state, peak slot |
| 🔮 Predict | Enter time-slot inputs → ML prediction + historical comparison chart |
| 📊 EDA | Trend lines, state breakdown, correlation heatmap, scatter plot |
| 📋 Dataset | Filterable raw data table with CSV download + descriptive stats |
| ℹ️ About | Architecture, pipeline details, run instructions |

## Pages (ui.py — API-backed)

| Page | Description |
|---|---|
| Overview | KPI cards + national trend chart pulled from Flask API |
| Predict | Prediction via POST `/api/predict` |
| Dataset | Dataset preview from GET `/api/dataset` |
| Project report | Live API health + model info JSON |

---

## Flask API Endpoints (backend.py)

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness check — returns `{"status":"ok"}` |
| GET | `/api/model` | Model info, states list, feature stats |
| GET | `/api/dataset` | Dataset preview (params: `limit`, `state`) |
| POST | `/api/predict` | Predict total accidents (body: `{state, year, time_slots}`) |

---

## ML Pipeline

1. **Load & clean** — drop NaN rows
2. **Label encode** — STATE/UT → integer (35 classes)
3. **Feature engineer** — `NIGHT_TOTAL`, `DAY_TOTAL`, `PEAK_RATIO`
4. **Compare** — Random Forest · Gradient Boosting · Ridge Regression
5. **Select best** — auto-selected by R² on 20% hold-out set
6. **Cross-validate** — 5-fold CV
7. **Save** — `rf_model.pkl`, `label_encoder.pkl`, `feature_stats.json`

---

## Environment

- Python 3.14 (`.venv/`)
- All packages installed into `.venv/`
