"""Module 1 schema loader and live forecast-window factory."""
import json
from pathlib import Path
from live_forecast_window import LiveForecastWindow

BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR / "artifacts" / "feature_schema.json"

def load_feature_columns() -> list[str]:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Missing Module 1 artifact: {SCHEMA_PATH}")
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        return list(json.load(f)["feature_columns"])

def make_live_forecast_window(model_feature_order: list[str]) -> LiveForecastWindow:
    return LiveForecastWindow(load_feature_columns(), model_feature_order)
