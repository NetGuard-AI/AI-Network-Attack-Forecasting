"""
Module 4 - Forecast inference for SIH-153

Loads the trained XGBoost model produced by Module 3 and generates
a forecast probability, predicted attack flag, and risk level.

Expected model file:
    forecast_model.json

Usage from backend:
    from forecast import forecast_from_row

    result = forecast_from_row(feature_dict)

Periodic execution:
    Call forecast_from_row(...) whenever new network-traffic features
    for a forecast window are available. This script DOES NOT retrain
    the model.

Important:
- The input feature names/order must match the features used when the
  Module 3 model was trained.
- The current Module 3 MVP used 160 model features:
    78 *_mean features
    78 *_std features
    lookback_attack_rate
    lookback_attack_rate_lag1
    lookback_attack_rate_lag2
    lookback_attack_rate_lag3
- If raw traffic data is used, Module 4 should first calculate those
  same aggregate features and lookback features before calling this
  function.
"""

from pathlib import Path
from typing import Mapping, Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb


# Keep forecast.py beside forecast_model.json by default.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "forecast_model.json"

# MVP risk thresholds used by Module 3.
LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.70


def load_model(model_path: Optional[str] = None) -> xgb.XGBClassifier:
    """Load the trained XGBoost model."""
    path = Path(model_path) if model_path else MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Put forecast_model.json in the same folder as forecast.py "
            "or pass model_path explicitly."
        )

    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


def get_model_features(model: xgb.XGBClassifier) -> list[str]:
    """Return the exact feature names stored in the trained model."""
    booster = model.get_booster()
    feature_names = booster.feature_names

    if not feature_names:
        raise ValueError(
            "The model does not contain feature names. "
            "The backend must provide the exact Module 3 feature order."
        )

    return list(feature_names)


def risk_from_probability(probability: float) -> str:
    """Convert forecast probability to the MVP risk level."""
    if probability < LOW_THRESHOLD:
        return "LOW"
    if probability < HIGH_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


def forecast_from_row(
    features: Mapping[str, Any],
    model_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate one forecast from one already-prepared feature row.

    Parameters
    ----------
    features:
        Dictionary containing the Module 3 model features.
        It may contain extra fields such as window_end_seq; only model
        features are selected.

    model_path:
        Optional path to forecast_model.json.

    Returns
    -------
    dict:
        forecast_probability
        predicted_attack
        risk_level
    """
    model = load_model(model_path)
    model_features = get_model_features(model)

    missing = [name for name in model_features if name not in features]
    if missing:
        raise ValueError(
            "Missing required model features: "
            + ", ".join(missing)
        )

    # Build the row in the exact feature order stored in the model.
    row = pd.DataFrame(
        [[features[name] for name in model_features]],
        columns=model_features,
    )

    # Convert values to numeric and reject missing/non-numeric values.
    row = row.apply(pd.to_numeric, errors="coerce")

    if row.isna().any().any():
        bad = row.columns[row.isna().any()].tolist()
        raise ValueError(
            "Missing or non-numeric values in model features: "
            + ", ".join(bad)
        )

    probability = float(model.predict_proba(row)[0, 1])
    predicted_attack = int(probability >= 0.50)
    risk_level = risk_from_probability(probability)

    return {
        "forecast_probability": round(probability, 6),
        "predicted_attack": predicted_attack,
        "risk_level": risk_level,
    }


def forecast_from_dataframe(
    df: pd.DataFrame,
    model_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate forecasts for multiple already-prepared feature rows.

    Useful for testing or batch dashboard updates.
    """
    model = load_model(model_path)
    model_features = get_model_features(model)

    missing = [name for name in model_features if name not in df.columns]
    if missing:
        raise ValueError(
            "Missing required model features: "
            + ", ".join(missing)
        )

    X = df.loc[:, model_features].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "Missing or non-numeric values in model features: "
            + ", ".join(bad)
        )

    probabilities = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["forecast_probability"] = probabilities
    result["predicted_attack"] = (probabilities >= 0.50).astype(int)
    result["risk_level"] = [
        risk_from_probability(float(p)) for p in probabilities
    ]

    return result


if __name__ == "__main__":
    print("Module 4 forecast.py")
    print(f"Model path: {MODEL_PATH}")

    model = load_model()
    features = get_model_features(model)

    print(f"Model loaded successfully.")
    print(f"Required model features: {len(features)}")
    print("forecast.py is ready for backend integration.")
