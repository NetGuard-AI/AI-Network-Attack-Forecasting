"""
Module 3 - AI Network Attack Forecasting
Training script for the XGBoost MVP model.

This script documents/reproduces the training setup used for the MVP.
It uses a chronological slice of forecast_train.csv and never shuffles
the time-series data.
"""

from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

TRAIN_CSV = Path("forecast_train.csv")
TARGET = "future_attack_flag"
ORDER_COL = "window_end_seq"

START_ROW = 900_000
ROWS_TO_USE = 200_000
CHUNKSIZE = 50_000

MODEL_PARAMS = dict(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

def prepare_features(df):
    mean_cols = sorted([c for c in df.columns if c.endswith("_mean")])
    std_cols = sorted([c for c in df.columns if c.endswith("_std")])

    base_names = sorted(set(c[:-5] for c in mean_cols) & set(c[:-4] for c in std_cols))

    X = df[[f"{b}_mean" for b in base_names] + [f"{b}_std" for b in base_names]].copy()
    X["lookback_attack_rate"] = (
        df[TARGET].rolling(10, min_periods=1).mean().shift(1)
    )
    for lag in (1, 2, 3):
        X[f"attack_lag_{lag}"] = df[TARGET].shift(lag)

    return X

def main():
    # Read only the chronological MVP slice.
    parts = []
    remaining = ROWS_TO_USE
    skip = START_ROW

    while remaining > 0:
        n = min(CHUNKSIZE, remaining)
        chunk = pd.read_csv(TRAIN_CSV, skiprows=range(1, skip + 1), nrows=n)
        if chunk.empty:
            break
        parts.append(chunk)
        remaining -= len(chunk)
        skip += len(chunk)

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(ORDER_COL).reset_index(drop=True)

    X = prepare_features(df)
    y = df[TARGET].astype(int)

    valid = X.notna().all(axis=1)
    X, y = X.loc[valid], y.loc[valid]

    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        **MODEL_PARAMS,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    p = model.predict_proba(X_val)[:, 1]
    pred = (p >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_val, p),
        "pr_auc": average_precision_score(y_val, p),
        "f1": f1_score(y_val, pred),
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
    }

    model.save_model("forecast_model.json")
    print(metrics)
    print("Saved forecast_model.json")

if __name__ == "__main__":
    main()
