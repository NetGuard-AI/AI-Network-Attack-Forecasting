# Module 3 — AI-Based Network Attack Forecasting

## What this module does

Module 3 trains an XGBoost binary classification model to forecast whether
a future network-traffic window will contain an attack.

### Model input

The trained model expects **160 features**:
- 78 traffic `_mean` features
- 78 traffic `_std` features
- `lookback_attack_rate`
- `attack_lag_1`
- `attack_lag_2`
- `attack_lag_3`

Target: `future_attack_flag`

## Files

- `forecast_model.json` — native XGBoost model.
- `forecast_model.pkl` — Python pickle copy of the same XGBoost model.
- `forecast.py` — inference helper used by the backend.
- `train_forecaster.py` — training/reproduction script.
- `metrics.json` — recorded MVP validation metrics.
- `README.md` — module documentation.

## MVP training setup

The MVP used a **200,000-row chronological slice** from the training data,
followed by a chronological 80/20 train/validation split. The data was not
shuffled.

After removing rows affected by the three lag features:
- 199,997 usable rows
- 159,997 training rows
- 40,000 validation rows

XGBoost parameters:
- `n_estimators=100`
- `max_depth=4`
- `learning_rate=0.05`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `tree_method=hist`
- `objective=binary:logistic`
- `random_state=42`

## Recorded MVP metrics

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9935 |
| PR-AUC | 0.9998 |
| F1 | 0.9936 |
| Precision | 0.9920 |
| Recall | 0.9952 |

**Important:** these are metrics for the selected chronological MVP
validation window. They are not a full-dataset benchmark.

## Backend integration

The backend should use `forecast.py` for inference rather than retraining
the model. It must prepare the same 160 features in the same
order as the trained model.

The forecast contract includes:
- `forecast_probability`
- `risk_level`
- `trend`
- `forecast_window_minutes`

The MVP forecast window is **15 minutes**.

Risk thresholds:
- `< 0.30` → LOW
- `0.30` to `< 0.70` → MEDIUM
- `>= 0.70` → HIGH

These thresholds are MVP heuristics and can be recalibrated later.
