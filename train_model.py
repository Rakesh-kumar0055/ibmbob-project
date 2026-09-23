"""
Model Training Script — Road Accident Prediction Project
Run: python train_model.py
Outputs: models/rf_model.pkl, models/label_encoder.pkl, models/feature_stats.json
"""

import os
import json
import joblib  # pyright: ignore[reportMissingImports]
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
from sklearn.ensemble import (  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import Ridge  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
from sklearn.preprocessing import LabelEncoder, StandardScaler  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
from sklearn.model_selection import (  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
    train_test_split,
    cross_val_score,
)
from sklearn.metrics import (  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "road_accidents.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── feature & target columns ──────────────────────────────────────────────────
TIME_COLS = [
    "0-3 hrs. (Night)", "3-6 hrs. (Night)", "6-9 hrs (Day)",
    "9-12 hrs (Day)",   "12-15 hrs (Day)",  "15-18 hrs (Day)",
    "18-21 hrs (Night)", "21-24 hrs (Night)"
]
TARGET = "Total"

# ── 1. Load & clean ───────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df = df.dropna()

print(f"Loaded {len(df)} rows — {df['STATE/UT'].nunique()} states, "
      f"years {df['YEAR'].min()}–{df['YEAR'].max()}")

# ── 2. Encode state label ─────────────────────────────────────────────────────
le = LabelEncoder()
df["STATE_ENC"] = np.asarray(le.fit_transform(df["STATE/UT"]), dtype=np.int64)

# ── 3. Feature engineering ────────────────────────────────────────────────────
df["NIGHT_TOTAL"] = (df["0-3 hrs. (Night)"] + df["3-6 hrs. (Night)"] +
                     df["18-21 hrs (Night)"] + df["21-24 hrs (Night)"])
df["DAY_TOTAL"]   = (df["6-9 hrs (Day)"]    + df["9-12 hrs (Day)"] +
                     df["12-15 hrs (Day)"]   + df["15-18 hrs (Day)"])
df["PEAK_RATIO"]  = df["DAY_TOTAL"] / (df["NIGHT_TOTAL"] + 1)

FEATURE_COLS = TIME_COLS + ["YEAR", "STATE_ENC", "NIGHT_TOTAL", "DAY_TOTAL", "PEAK_RATIO"]

X = df[FEATURE_COLS].values
y = df[TARGET].values

# ── 4. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 5. Train models & compare ─────────────────────────────────────────────────
models = {
    "Random Forest"       : RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting"   : GradientBoostingRegressor(n_estimators=200, random_state=42),
    "Ridge Regression"    : Ridge(alpha=1.0),
}

print("\n── Model Comparison ─────────────────────────────────────────")
print(f"{'Model':<25} {'R²':>6}  {'MAE':>10}  {'RMSE':>10}")
print("-" * 55)

best_r2    = -np.inf
best_name  = None
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2    = r2_score(y_test, preds)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    print(f"{name:<25} {r2:>6.4f}  {mae:>10.1f}  {rmse:>10.1f}")
    if r2 > best_r2:
        best_r2    = r2
        best_name  = name
        best_model = model

print("-" * 55)
print(f"\nBest model : {best_name}  (R² = {best_r2:.4f})")

# ── 6. Cross-validation on best model ─────────────────────────────────────────
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="r2") # pyright: ignore[reportArgumentType]
print(f"5-fold CV R² : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 7. Feature importances (if available) ────────────────────────────────────
if best_model is not None and hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
    print("\nTop feature importances:")
    print(imp.sort_values(ascending=False).to_string())

# ── 8. Save artefacts ─────────────────────────────────────────────────────────
joblib.dump(best_model, os.path.join(MODELS_DIR, "rf_model.pkl"))
joblib.dump(le,         os.path.join(MODELS_DIR, "label_encoder.pkl"))

# persist feature stats for the UI
feature_stats = {
    "feature_cols"  : FEATURE_COLS,
    "time_cols"     : TIME_COLS,
    "states"        : list(le.classes_),
    "year_min"      : int(df["YEAR"].min()),
    "year_max"      : int(df["YEAR"].max()),
    "target_col"    : TARGET,
    "best_model"    : best_name,
    "test_r2"       : round(best_r2, 4),
    "time_col_stats": {
        col: {
            "min"  : int(df[col].min()),
            "max"  : int(df[col].max()),
            "mean" : round(float(df[col].mean()), 1),
        }
        for col in TIME_COLS
    },
}
with open(os.path.join(MODELS_DIR, "feature_stats.json"), "w") as f:
    json.dump(feature_stats, f, indent=2)

print("\n✅  Model saved → models/rf_model.pkl")
print("✅  Encoder saved → models/label_encoder.pkl")
print("✅  Stats saved  → models/feature_stats.json")
