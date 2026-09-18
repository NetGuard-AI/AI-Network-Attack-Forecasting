"""
train_model.py
Module 2 — Attack Detection ML Model
SIH Problem Statement: AI-Based Network Attack Forecasting from Network Traffic Data

Pipeline:
  1. Load Module 1 outputs: train/val/test splits, feature_schema.json, scaler.pkl, encoders.pkl
  2. Train RandomForest baseline -> evaluate on validation set
  3. Train XGBoost with class weighting + small grid search -> select best on validation F1
     (prioritizing attack-class recall)
  4. Single held-out test evaluation of the selected model
  5. Serialize model (joblib) -> trained_model.pkl
  6. Write evaluation report -> metrics.json, confusion_matrix.png, feature_importance.png

Expected input directory layout (default: ./data):
  data/
    train.csv
    val.csv
    test.csv
    feature_schema.json   # {"features": [...], "target": "label", "classes": [...]}
    scaler.pkl            # fitted on Module 1 side, already applied to CSVs if numeric cols pre-scaled
    encoders.pkl           # fitted encoders for categorical cols (if any)

Usage:
  python train_model.py --data-dir ./data --out-dir ./outputs
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)
from xgboost import XGBClassifier


def load_schema(data_dir: Path):
    schema_path = data_dir / "feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"feature_schema.json not found in {data_dir}. "
            "This must come from Module 1 — do not re-derive feature order."
        )
    with open(schema_path) as f:
        return json.load(f)


def load_split(data_dir: Path, name: str, feature_cols, target_col):
    path = data_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Expected Module 1 to provide this split.")
    df = pd.read_csv(path)
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{name}.csv is missing expected columns: {missing}")
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y


def evaluate(y_true, y_pred, classes):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    per_class = {}
    for i, c in enumerate(classes):
        per_class[str(c)] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=support)) if support.sum() > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [str(c) for c in classes],
    }


def attack_recall_score(metrics_dict, classes, benign_label="benign"):
    """Recall averaged over attack classes only (excludes benign), used for model selection."""
    attack_recalls = [
        v["recall"] for k, v in metrics_dict["per_class"].items() if k.lower() != str(benign_label).lower()
    ]
    return float(np.mean(attack_recalls)) if attack_recalls else metrics_dict["macro_f1"]


def plot_confusion_matrix(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set)")
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importances, feature_names, out_path, top_n=10):
    idx = np.argsort(importances)[::-1][:top_n]
    top_feats = [feature_names[i] for i in idx]
    top_vals = [importances[i] for i in idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(top_feats)), top_vals[::-1])
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return list(zip(top_feats, [round(float(v), 4) for v in top_vals]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="./outputs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = load_schema(data_dir)
    feature_cols = schema["features"]
    target_col = schema.get("target", "label")
    classes = schema.get("classes")

    X_train, y_train = load_split(data_dir, "train", feature_cols, target_col)
    X_val, y_val = load_split(data_dir, "val", feature_cols, target_col)
    X_test, y_test = load_split(data_dir, "test", feature_cols, target_col)

    if classes is None:
        classes = sorted(pd.unique(np.concatenate([y_train, y_val, y_test])).tolist())

    metrics_report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "feature_count": len(feature_cols)}

    # ---------- Stage 1: RandomForest baseline ----------
    print("[Stage 1] Training RandomForest baseline...")
    baseline = RandomForestClassifier(
        n_estimators=200, max_depth=None, class_weight="balanced", random_state=42, n_jobs=-1
    )
    baseline.fit(X_train, y_train)
    baseline_val_pred = baseline.predict(X_val)
    baseline_val_metrics = evaluate(y_val, baseline_val_pred, classes)
    metrics_report["baseline_rf"] = {
        "params": {"n_estimators": 200, "class_weight": "balanced"},
        "validation": baseline_val_metrics,
    }
    print(f"  Baseline val macro F1: {baseline_val_metrics['macro_f1']}")

    # ---------- Stage 2: XGBoost with small grid search ----------
    print("[Stage 2] Training XGBoost with grid search...")
    class_counts = pd.Series(y_train).value_counts()
    # crude class weighting via sample_weight (works for multi-class, unlike scale_pos_weight)
    weight_map = {c: len(y_train) / (len(class_counts) * cnt) for c, cnt in class_counts.items()}
    sample_weight = np.array([weight_map[y] for y in y_train])

    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_train_enc = np.array([label_to_idx[y] for y in y_train])
    y_val_enc = np.array([label_to_idx[y] for y in y_val])
    y_test_enc = np.array([label_to_idx[y] for y in y_test])

    grid = {
        "max_depth": [4, 6, 8],
        "n_estimators": [150, 300],
        "learning_rate": [0.05, 0.1],
    }

    best_score = -1.0
    best_model = None
    best_params = None
    best_val_metrics = None

    for max_depth in grid["max_depth"]:
        for n_estimators in grid["n_estimators"]:
            for lr in grid["learning_rate"]:
                model = XGBClassifier(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    learning_rate=lr,
                    objective="multi:softmax",
                    num_class=len(classes),
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train_enc, sample_weight=sample_weight)
                val_pred_enc = model.predict(X_val)
                val_pred = np.array([classes[i] for i in val_pred_enc])
                val_metrics = evaluate(y_val, val_pred, classes)

                # selection: weighted F1 combined with attack-class recall
                score = 0.5 * val_metrics["weighted_f1"] + 0.5 * attack_recall_score(val_metrics, classes)

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = {"max_depth": max_depth, "n_estimators": n_estimators, "learning_rate": lr}
                    best_val_metrics = val_metrics

    print(f"  Best XGBoost params: {best_params} (selection score: {round(best_score, 4)})")
    metrics_report["tuned_xgboost"] = {
        "params": best_params,
        "selection_score": round(float(best_score), 4),
        "validation": best_val_metrics,
    }

    # ---------- Stage 3: single held-out test evaluation ----------
    print("[Stage 3] Evaluating on held-out test set...")
    test_pred_enc = best_model.predict(X_test)
    test_pred = np.array([classes[i] for i in test_pred_enc])
    test_metrics = evaluate(y_test, test_pred, classes)
    metrics_report["test"] = test_metrics
    print(f"  Test macro F1: {test_metrics['macro_f1']}, accuracy: {test_metrics['accuracy']}")

    # decide which model ships: whichever scored higher on validation selection criteria
    baseline_score = 0.5 * baseline_val_metrics["weighted_f1"] + 0.5 * attack_recall_score(baseline_val_metrics, classes)
    shipped_model_name = "tuned_xgboost" if best_score >= baseline_score else "baseline_rf"
    metrics_report["shipped_model"] = shipped_model_name

    shipped_model = best_model if shipped_model_name == "tuned_xgboost" else baseline
    importances = shipped_model.feature_importances_

    # ---------- Plots ----------
    cm = np.array(test_metrics["confusion_matrix"])
    plot_confusion_matrix(cm, test_metrics["confusion_matrix_labels"], out_dir / "confusion_matrix.png")
    top_features = plot_feature_importance(importances, feature_cols, out_dir / "feature_importance.png", top_n=10)
    metrics_report["top_10_feature_importance"] = top_features

    # ---------- Serialize ----------
    model_bundle = {
        "model": shipped_model,
        "model_type": shipped_model_name,
        "feature_order": feature_cols,
        "classes": classes,
    }
    joblib.dump(model_bundle, out_dir / "trained_model.pkl")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_report, f, indent=2)

    print(f"\nDone. Shipped model: {shipped_model_name}")
    print(f"  -> {out_dir / 'trained_model.pkl'}")
    print(f"  -> {out_dir / 'metrics.json'}")
    print(f"  -> {out_dir / 'confusion_matrix.png'}")
    print(f"  -> {out_dir / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
