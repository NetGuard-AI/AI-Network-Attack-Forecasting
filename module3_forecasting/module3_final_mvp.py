import pandas as pd
import numpy as np
import os
import xgboost as xgb

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# ============================================================
# MODULE 3 - FINAL MVP
# AI-BASED NETWORK ATTACK FORECASTING
# ============================================================

BASE_DIR = r"C:\sih153\mod1 for shrawani"

TRAIN_CSV = os.path.join(
    BASE_DIR,
    r"train_extracted\forecast_train.csv"
)

OUTPUT_MODEL = os.path.join(
    BASE_DIR,
    "forecast_model.json"
)

OUTPUT_PREDICTIONS = os.path.join(
    BASE_DIR,
    "forecast_predictions.csv"
)

TARGET = "future_attack_flag"
ORDER = "window_end_seq"

# Use a manageable chronological MVP sample
START_ROW = 900000
ROWS_TO_USE = 200000
CHUNK_SIZE = 50000

print("=" * 65)
print("MODULE 3 - FINAL MVP")
print("AI-BASED NETWORK ATTACK FORECASTING")
print("=" * 65)

print("\nLoading chronological MVP data...")
print("Start row:", START_ROW)
print("Rows:", ROWS_TO_USE)

# ------------------------------------------------------------
# READ REQUIRED SECTION
# ------------------------------------------------------------

parts = []
current = 0
end_row = START_ROW + ROWS_TO_USE

for chunk in pd.read_csv(
    TRAIN_CSV,
    chunksize=CHUNK_SIZE
):

    chunk_start = current
    chunk_end = current + len(chunk)

    if chunk_end <= START_ROW:
        current = chunk_end
        continue

    if chunk_start >= end_row:
        break

    local_start = max(
        0,
        START_ROW - chunk_start
    )

    local_end = min(
        len(chunk),
        end_row - chunk_start
    )

    parts.append(
        chunk.iloc[local_start:local_end]
    )

    current = chunk_end

    if current >= end_row:
        break

data = pd.concat(
    parts,
    ignore_index=True
)

print("\nData loaded:", data.shape)

# ------------------------------------------------------------
# SORT CHRONOLOGICALLY
# ------------------------------------------------------------

data = data.sort_values(
    ORDER
).reset_index(drop=True)

# ------------------------------------------------------------
# FIND 78 TRAFFIC FEATURES
# ------------------------------------------------------------

mean_columns = [
    c for c in data.columns
    if c.endswith("_mean")
]

raw_features = []

for col in mean_columns:

    base = col[:-5]

    if base + "_std" in data.columns:
        raw_features.append(base)

print("\nTraffic features:", len(raw_features))

# ------------------------------------------------------------
# BUILD FEATURES
# ------------------------------------------------------------

features = []

for base in raw_features:

    features.append(base + "_mean")
    features.append(base + "_std")

if "lookback_attack_rate" in data.columns:

    features.append(
        "lookback_attack_rate"
    )

# ------------------------------------------------------------
# LAG FEATURES
# ------------------------------------------------------------

for lag in [1, 2, 3]:

    col = f"lookback_attack_rate_lag_{lag}"

    data[col] = data[
        "lookback_attack_rate"
    ].shift(lag)

    features.append(col)

# ------------------------------------------------------------
# CLEAN
# ------------------------------------------------------------

data = data.dropna(
    subset=features + [TARGET]
).reset_index(drop=True)

print("Rows after preprocessing:", len(data))

# ------------------------------------------------------------
# TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\nTarget distribution:")

print(
    data[TARGET].value_counts()
)

# ------------------------------------------------------------
# CHRONOLOGICAL SPLIT
# 80% TRAIN / 20% VALIDATION
# ------------------------------------------------------------

split = int(
    len(data) * 0.8
)

train = data.iloc[:split]

validation = data.iloc[split:]

print("\nTraining rows:", len(train))
print("Validation rows:", len(validation))

print(
    "Training attacks:",
    int((train[TARGET] == 1).sum())
)

print(
    "Validation attacks:",
    int((validation[TARGET] == 1).sum())
)

# ------------------------------------------------------------
# X / Y
# ------------------------------------------------------------

X_train = train[features]
y_train = train[TARGET].astype(int)

X_val = validation[features]
y_val = validation[TARGET].astype(int)

# ------------------------------------------------------------
# CLASS WEIGHT
# ------------------------------------------------------------

positive = max(
    int((y_train == 1).sum()),
    1
)

negative = int(
    (y_train == 0).sum()
)

scale_pos_weight = negative / positive

print(
    "\nScale positive weight:",
    scale_pos_weight
)

# ------------------------------------------------------------
# XGBOOST
# ------------------------------------------------------------

print("\nTraining XGBoost...")

model = xgb.XGBClassifier(

    n_estimators=100,

    max_depth=4,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    tree_method="hist",

    scale_pos_weight=scale_pos_weight,

    random_state=42,

    n_jobs=-1
)

model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_val, y_val)
    ],

    verbose=False
)

print("XGBoost training complete.")

# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------

print("\nGenerating attack forecasts...")

probability = model.predict_proba(
    X_val
)[:, 1]

prediction = (
    probability >= 0.5
).astype(int)

# ------------------------------------------------------------
# RISK LEVEL
# ------------------------------------------------------------

def risk_level(p):

    if p >= 0.7:
        return "HIGH"

    elif p >= 0.3:
        return "MEDIUM"

    else:
        return "LOW"


risk = [
    risk_level(p)
    for p in probability
]

# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FINAL MVP RESULTS")
print("=" * 65)

if len(np.unique(y_val)) == 2:

    roc = roc_auc_score(
        y_val,
        probability
    )

    pr = average_precision_score(
        y_val,
        probability
    )

    f1 = f1_score(
        y_val,
        prediction
    )

    precision = precision_score(
        y_val,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        prediction,
        zero_division=0
    )

    print("\nROC-AUC:", round(roc, 4))
    print("PR-AUC:", round(pr, 4))
    print("F1:", round(f1, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_val,
            prediction
        )
    )

# ------------------------------------------------------------
# SAVE PREDICTIONS
# ------------------------------------------------------------

output = pd.DataFrame({

    "window_end_seq":
        validation[ORDER].values,

    "actual_attack":
        y_val.values,

    "attack_probability":
        probability,

    "predicted_attack":
        prediction,

    "risk_level":
        risk
})

output.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)

# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

model.save_model(
    OUTPUT_MODEL
)

# ------------------------------------------------------------
# FINISHED
# ------------------------------------------------------------

print("\nPrediction file saved:")
print(OUTPUT_PREDICTIONS)

print("\nModel saved:")
print(OUTPUT_MODEL)

print("\nSample predictions:")

print(
    output.head(10).to_string(
        index=False
    )
)

print("\n" + "=" * 65)
print("MODULE 3 MVP COMPLETE")
print("=" * 65)