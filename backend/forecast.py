"""Module 3 -> Module 4 forecasting adapter for the MVP.

The REAL Module 3 model is expected at backend/artifacts/forecast_model.pkl.
The model is loaded with joblib and is never retrained by Module 4.

Public output is limited to the agreed MVP fields:
  forecast_probability, risk_level, trend, forecast_window_minutes
"""
from pathlib import Path
from typing import Any, Mapping, Optional
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "forecast_model.pkl"
SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"
LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.70


def load_schema() -> dict:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Missing Module 1 artifact: {SCHEMA_PATH}")
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        return json.load(f)

def load_model(model_path: Optional[str] = None):
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Module 3 model: {path}. "
            "Copy the real forecast_model.pkl from Module 3 into backend/artifacts/."
        )
    return joblib.load(path)

def get_model_features(model) -> list[str]:
    # XGBoost saved through its sklearn wrapper
    try:
        names = model.get_booster().feature_names
        if names:
            return list(names)
    except Exception:
        pass
    # sklearn models trained from a DataFrame
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)
    # Last-resort contract for the agreed M1/M3 78-feature aggregate design.
    schema = load_schema()
    raw = list(schema["feature_columns"])
    expected = [f"{c}_mean" for c in raw] + [f"{c}_std" for c in raw] + [
        "lookback_attack_rate", "lookback_attack_rate_lag1",
        "lookback_attack_rate_lag2", "lookback_attack_rate_lag3",
    ]
    n = getattr(model, "n_features_in_", None)
    if n == len(expected):
        return expected
    raise ValueError("Module 3 model does not expose its feature names and its feature count is not the agreed MVP schema")

def risk_from_probability(probability: float) -> str:
    if probability < LOW_THRESHOLD:
        return "Low"
    if probability < HIGH_THRESHOLD:
        return "Medium"
    return "High"

def forecast_from_row(features: Mapping[str, Any], model_path: Optional[str] = None) -> dict[str, Any]:
    model = load_model(model_path)
    model_features = get_model_features(model)
    missing = [name for name in model_features if name not in features]
    if missing:
        raise ValueError("Missing required Module 3 features: " + ", ".join(missing[:10]))
    row = pd.DataFrame([[features[name] for name in model_features]], columns=model_features)
    row = row.apply(pd.to_numeric, errors="coerce")
    if row.isna().any().any():
        bad = row.columns[row.isna().any()].tolist()
        raise ValueError("Missing or non-numeric forecast features: " + ", ".join(bad[:10]))
    probability = float(model.predict_proba(row)[0, 1])
    schema = load_schema()
    # M1 must explicitly declare a real minute horizon. We do not invent one.
    horizon = schema.get("forecast_window_minutes")
    if horizon is None:
        horizon = schema.get("forecast_horizon_minutes")
    if horizon is None:
        raise ValueError("feature_schema.json must declare forecast_window_minutes for the MVP")
    return {
        "forecast_probability": round(float(np.clip(probability, 0.0, 1.0)), 6),
        "risk_level": risk_from_probability(probability),
        "trend": "stable",  # main.py computes trend from successive forecasts
        "forecast_window_minutes": int(horizon),
    }
