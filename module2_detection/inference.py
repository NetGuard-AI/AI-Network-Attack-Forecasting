"""
Module 2 — Attack Detection ML Model
Inference wrapper for Module 4.

Loads once at import time:
    feature_schema.json
    scaler.pkl
    trained_model.pkl   (the XGBoost model from train_xgboost.py)

Exposes:
    detect(record: dict) -> {"label": str, "confidence": float}

This matches the frozen contract from the build plan:
    detect() contract (Module 2 -> Module 4): {label: str, confidence: float} in
    is a dict, out is a dict, JSON-serializable.

Module 4 usage:
    from inference import detect
    result = detect(record_dict)   # {"label": "DDoS", "confidence": 0.97}
"""

import json

import joblib
import numpy as np

SCHEMA_PATH = "feature_schema.json"
SCALER_PATH = "scaler.pkl"
MODEL_PATH = "trained_model.pkl"

with open(SCHEMA_PATH) as _f:
    _schema = json.load(_f)

_FEATURE_COLUMNS = _schema["feature_columns"]
_LABEL_CLASSES = _schema["label_classes"]  # name -> int
_IDX_TO_LABEL = {idx: name for name, idx in _LABEL_CLASSES.items()}

_scaler = joblib.load(SCALER_PATH)
_model = joblib.load(MODEL_PATH)


class MissingFeatureError(ValueError):
    """Raised when an incoming record is missing a feature the model needs."""


def _record_to_vector(record: dict) -> np.ndarray:
    """
    Pull features out of an incoming record dict, in the exact order
    feature_schema.json defines. Do not re-derive or reorder — Module 1's
    scaler was fit on this exact column order.
    """
    missing = [c for c in _FEATURE_COLUMNS if c not in record]
    if missing:
        raise MissingFeatureError(
            f"Record is missing {len(missing)} required feature(s): {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )
    values = [record[c] for c in _FEATURE_COLUMNS]
    return np.asarray(values, dtype=np.float32).reshape(1, -1)


def detect(record: dict) -> dict:
    """
    Classify a single traffic record as benign or a specific attack type.

    Args:
        record: dict mapping each feature name in feature_schema.json's
            feature_columns to its numeric value. Extra keys are ignored.

    Returns:
        {"label": str, "confidence": float}
        label is one of feature_schema.json's label_classes keys (e.g. "BENIGN", "DDoS").
        confidence is the model's predicted probability for that label, in [0, 1].

    Raises:
        MissingFeatureError: if a required feature is absent from `record`.
    """
    x = _record_to_vector(record)
    x_scaled = _scaler.transform(x).astype(np.float32)

    probs = _model.predict_proba(x_scaled)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    label = _IDX_TO_LABEL[pred_idx]

    return {"label": label, "confidence": confidence}


if __name__ == "__main__":
    # Quick smoke test: build one fabricated all-zero record from the schema
    # and confirm detect() runs end-to-end. Not a real accuracy check —
    # Module 2/4 pre-sync should test with real sample records.
    dummy_record = {col: 0.0 for col in _FEATURE_COLUMNS}
    result = detect(dummy_record)
    print("Smoke test result:", result)
    assert set(result.keys()) == {"label", "confidence"}
    assert isinstance(result["label"], str)
    assert isinstance(result["confidence"], float)
    print("OK — detect() returns the expected {label, confidence} shape.")
