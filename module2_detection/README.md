# Module 2 — Attack Detection ML Model

**SIH Problem Statement:** AI-Based Network Attack Forecasting from Network Traffic Data

## Objective

Classify each incoming network traffic record as **benign** or a specific **attack type**, in real time, with a confidence score. This is the detection layer that Modules 3 (attack-rate time windows / forecasting) and 4 (live ingestion pipeline) depend on downstream.

## Inputs (from Module 1)

Module 1 provides the following — **do not re-derive preprocessing**, use these exactly as given:

| File | Purpose |
|---|---|
| `train.csv`, `val.csv`, `test.csv` | Pre-split, pre-processed traffic data |
| `feature_schema.json` | Exact feature order + target column + class list |
| `scaler.pkl` | Fitted scaler (numeric features already scaled in the CSVs) |
| `encoders.pkl` | Fitted encoders for categorical features |

> ⚠️ Known issue: `encoders.pkl` loaded as a duplicate `StandardScaler` rather than an actual encoder — flagged to Module 1 owner, awaiting fix/confirmation before final retrain.

## Pipeline (`train_model.py`)

1. Load Module 1's splits + schema + encoders
2. **Stage 1 — Baseline:** RandomForest (`class_weight="balanced"`) → evaluate on validation set
3. **Stage 2 — Tuned model:** XGBoost with class-weighted sampling, small grid search over `max_depth` / `n_estimators` / `learning_rate` → best model selected on validation F1, prioritizing **attack-class recall** (not accuracy)
4. **Stage 3 — Final check:** single held-out test evaluation of the selected model (whichever of baseline/tuned scores higher on validation)
5. Serialize the shipped model → `trained_model.pkl` (joblib)
6. Write evaluation report → `metrics.json`, `confusion_matrix.png`, `feature_importance.png`

### Run it

```bash
python train_model.py --data-dir ./data --out-dir ./outputs
```

Expects `./data/` to contain `train.csv`, `val.csv`, `test.csv`, `feature_schema.json` (schema format: `{"features": [...], "target": "label", "classes": [...]}`). Adjust column names in `load_split()` if Module 1's naming differs.

## Files in this module

| File | Status | Description |
|---|---|---|
| `train_model.py` | ✅ done | Full training pipeline (baseline → tuned → eval → serialize) |
| `trained_model.pkl` | ✅ done | Serialized model bundle: `{model, model_type, feature_order, classes}` |
| `inference.py` | — | Wrapper exposing `detect()`, called by Module 4 on every ingested record |
| `metrics.json` | ✅ done | Precision/recall/F1 per class, confusion matrix, top-10 feature importances |
| `confusion_matrix.png` | ✅ done | Visual confusion matrix (test set) |
| `feature_importance.png` | ✅ done | Top 10 feature importance chart |

## Contract — `detect()` (Module 2 → Module 4)

```python
detect(record: dict) -> {"label": str, "confidence": float}
```

Output must be JSON-serializable. Field names must match byte-for-byte with what Module 4 expects — confirmed via test call at the Hour 10 pre-sync.

## Contract — Module 2 → Module 3 handoff

Module 3 consumes Module 2's per-record labels + confidence to build attack-rate time windows and produce:

```python
get_risk_score() -> {
    "risk_score": float,
    "trend": "rising" | "stable" | "falling",
    "next_window_prediction": bool
}
```

Eval metrics (`metrics.json`) should be shared with the Module 3 owner so they understand detection reliability per class.

## Forecasting definition (agreed framing, for context)

Given traffic/features from a preceding window `t-k → t`, predict whether an attack will occur in the future window `t+1 → t+h`. An attack "occurs" in the forecast horizon when the number/rate of attack-labeled flows exceeds a predefined threshold within that future window.

## Status

- [x] Module 2 → Module 3 handoff shared (`inference.py`, `trained_model.pkl`, `results.csv`)
- [x] Module 2 → Module 4 handoff shared (`inference.py`, `trained_model.pkl`)
- [x] `train_model.py` + `metrics.json` built and tested end-to-end (synthetic data — rerun on real Module 1 output before final submission)
- [ ] Awaiting Module 1 owner's reply on the `encoders.pkl` mixup (currently loads as a duplicate `StandardScaler`)

## Stack

scikit-learn (RandomForest baseline), XGBoost (tuned model), `sklearn.metrics`, `joblib`. CPU-only, no GPU required.
