"""Module 2 -> Module 4 detection adapter.

Loads the REAL Module 1 preprocessing artifacts and the REAL Module 2 model.
Place these in backend/artifacts/:
  feature_schema.json, scaler.pkl, encoders.pkl, trained_model.pkl

Public function:
  detect(record) -> {"label": str, "confidence": float}
"""
from pathlib import Path
import json
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
MODEL_PATH = ARTIFACT_DIR / "trained_model.pkl"

if not SCHEMA_PATH.exists():
    raise FileNotFoundError(f"Missing Module 1 artifact: {SCHEMA_PATH}")
if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Missing Module 1 artifact: {SCALER_PATH}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing Module 2 artifact: {MODEL_PATH}")

with SCHEMA_PATH.open(encoding="utf-8") as f:
    _schema = json.load(f)
_FEATURE_COLUMNS = list(_schema["feature_columns"])
_LABEL_CLASSES = _schema["label_classes"]
_IDX_TO_LABEL = {int(idx): name for name, idx in _LABEL_CLASSES.items()}
_scaler = joblib.load(SCALER_PATH)
_model = joblib.load(MODEL_PATH)

class MissingFeatureError(ValueError):
    pass

def _record_to_matrix(record: dict) -> np.ndarray:
    missing = [c for c in _FEATURE_COLUMNS if c not in record]
    if missing:
        raise MissingFeatureError(
            f"Record is missing {len(missing)} required feature(s): {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )
    try:
        return np.asarray([[record[c] for c in _FEATURE_COLUMNS]], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("One or more traffic features are not numeric") from exc

def scale_record(record: dict) -> np.ndarray:
    return _scaler.transform(_record_to_matrix(record)).astype(np.float32)[0]

def detect(record: dict) -> dict:
    x_scaled = scale_record(record)
    probs = _model.predict_proba(x_scaled.reshape(1, -1))[0]
    pred_idx = int(np.argmax(probs))
    return {"label": _IDX_TO_LABEL[pred_idx], "confidence": float(probs[pred_idx])}
