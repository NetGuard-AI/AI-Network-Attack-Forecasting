"""
Module 1 — Data Acquisition & Preprocessing (memory-safe MVP)
AI-Based Network Attack Forecasting System

This version keeps the original Module 1 outputs/contracts but avoids the
largest memory spikes in the original script.

Key fixes:
- Numeric feature data is stored as float32 instead of float64.
- The scaler is fitted ONLY on the chronological forecasting training slice.
- Scaling is applied split-by-split instead of making one huge scaled copy.
- Forecast windows are built with NumPy rolling calculations rather than
  millions of Python dictionaries.
- SMOTE is applied only to the detection training data and only when useful.
  For very large datasets, the training data can be capped before SMOTE to
  prevent RAM exhaustion. The cap is configurable below.
- The common CICIDS2017 CSV distribution has no native Timestamp column, so
  flow_seq/file order remains the chronological proxy described in the
  original Module 1 specification.
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

FILE_ORDER = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

LABEL_COL = "Label"

DETECTION_TRAIN_FRAC = 0.70
DETECTION_VAL_FRAC = 0.15

FORECAST_TRAIN_FRAC = 0.70
FORECAST_VAL_FRAC = 0.15

# Keep the agreed small MVP window definition from the source file.
LOOKBACK_K = 20
HORIZON_H = 10

# Memory guard for SMOTE.
# CICIDS2017 is already very large. SMOTE on the complete 1.1M+ row
# training set can itself exhaust RAM. If detection training is larger than
# this value, a stratified sample is used for the SMOTE operation.
SMOTE_MAX_TRAIN_ROWS = 300_000


def load_and_clean(raw_dir: str, nrows_per_file=None) -> pd.DataFrame:
    """Load the 8 CICIDS2017 CSVs in canonical capture-day order."""

    frames = []

    for fname in FILE_ORDER:
        path = os.path.join(raw_dir, fname)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected CICIDS2017 file not found:\n{path}\n"
                f"Check --raw-dir points to the folder containing the 8 CSVs."
            )

        df = pd.read_csv(
            path,
            nrows=nrows_per_file,
            low_memory=False,
        )

        df.columns = df.columns.str.strip()
        df["capture_day"] = fname.split(".")[0]
        frames.append(df)

        print(f"  loaded {fname}: {len(df):,} rows")

    data = pd.concat(frames, ignore_index=True)
    del frames

    data["flow_seq"] = np.arange(len(data), dtype=np.int64)

    print(f"Raw concatenated shape: {data.shape}")

    data.columns = data.columns.str.strip()

    if LABEL_COL not in data.columns:
        raise KeyError(
            f"Expected label column '{LABEL_COL}' not found: {list(data.columns)}"
        )

    # Convert numeric columns to float32 to substantially reduce RAM.
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    data[numeric_cols] = data[numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    before = len(data)
    data = data.dropna(subset=numeric_cols)

    print(
        f"Dropped {before - len(data):,} rows with NaN/Inf "
        f"-> {len(data):,} remain"
    )

    before = len(data)
    data = data.drop_duplicates()

    print(
        f"Dropped {before - len(data):,} exact duplicate rows "
        f"-> {len(data):,} remain"
    )

    data[LABEL_COL] = data[LABEL_COL].astype(str).str.strip()

    # Downcast numeric columns after cleaning.
    for col in numeric_cols:
        if col in data.columns and col != "flow_seq":
            data[col] = pd.to_numeric(data[col], errors="coerce").astype(
                np.float32
            )

    return data.reset_index(drop=True)


def get_feature_columns(data: pd.DataFrame):
    """Every column except label and provenance/order columns."""
    exclude = {LABEL_COL, "capture_day", "flow_seq"}
    return [c for c in data.columns if c not in exclude]


def fit_shared_pipeline(data: pd.DataFrame, feature_cols):
    """
    Fit the label encoder and scaler without using validation/test data.

    The scaler is fitted on the chronological forecasting TRAIN slice.
    This gives one shared scaler that can safely be reused by Modules 2 and 3.
    """

    label_encoder = LabelEncoder()
    label_encoder.fit(data[LABEL_COL])

    n = len(data)
    train_end = int(n * FORECAST_TRAIN_FRAC)

    scaler_train = data.iloc[:train_end][feature_cols]

    scaler = StandardScaler()
    scaler.fit(scaler_train.astype(np.float32))

    return label_encoder, scaler


def transform_split(
    data: pd.DataFrame,
    feature_cols,
    label_encoder,
    scaler,
):
    """Transform one split only, keeping the resulting numeric matrix float32."""

    out = data[["flow_seq", "capture_day", LABEL_COL]].copy()

    values = scaler.transform(
        data[feature_cols].to_numpy(dtype=np.float32, copy=False)
    ).astype(np.float32)

    scaled_features = pd.DataFrame(
        values,
        columns=feature_cols,
        index=data.index,
    )

    out = pd.concat([scaled_features, out], axis=1)

    out["Label_enc"] = label_encoder.transform(out[LABEL_COL]).astype(np.int32)

    return out.reset_index(drop=True)


def chronological_split(data: pd.DataFrame):
    n = len(data)

    train_end = int(n * FORECAST_TRAIN_FRAC)
    val_end = train_end + int(n * FORECAST_VAL_FRAC)

    train = data.iloc[:train_end].reset_index(drop=True)
    val = data.iloc[train_end:val_end].reset_index(drop=True)
    test = data.iloc[val_end:].reset_index(drop=True)

    print(
        f"Forecasting time-split sizes -> "
        f"train {len(train):,} / val {len(val):,} / test {len(test):,}"
    )

    return train, val, test


def build_detection_fork(
    data: pd.DataFrame,
    feature_cols,
    label_encoder,
    scaler,
    out_dir,
):
    """
    Random/stratified detection split.

    Important: the scaler was fitted only on the chronological training
    portion in fit_shared_pipeline().
    """

    labels = data[LABEL_COL]

    counts = labels.value_counts()

    keep_labels = counts[counts >= 6].index
    dropped = sorted(set(labels.unique()) - set(keep_labels))

    if dropped:
        print(
            "WARNING: dropping label class(es) with <6 rows: "
            f"{dropped}"
        )

        data = data[data[LABEL_COL].isin(keep_labels)].reset_index(drop=True)

    indices = np.arange(len(data))
    y = data[LABEL_COL].to_numpy()

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=DETECTION_TRAIN_FRAC,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    rel_val = DETECTION_VAL_FRAC / (1 - DETECTION_TRAIN_FRAC)

    temp_y = y[temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=rel_val,
        stratify=temp_y,
        random_state=RANDOM_STATE,
    )

    det_train_raw = data.iloc[train_idx].reset_index(drop=True)
    det_val_raw = data.iloc[val_idx].reset_index(drop=True)
    det_test_raw = data.iloc[test_idx].reset_index(drop=True)

    print(
        f"Detection split sizes -> "
        f"train {len(det_train_raw):,} / "
        f"val {len(det_val_raw):,} / "
        f"test {len(det_test_raw):,}"
    )

    # Transform each split independently instead of scaling the entire
    # 1.6M-row dataset at once.
    det_train = transform_split(
        det_train_raw, feature_cols, label_encoder, scaler
    )
    det_val = transform_split(
        det_val_raw, feature_cols, label_encoder, scaler
    )
    det_test = transform_split(
        det_test_raw, feature_cols, label_encoder, scaler
    )

    X_train = det_train[feature_cols].to_numpy(dtype=np.float32)
    y_train = det_train["Label_enc"].to_numpy(dtype=np.int32)

    # SMOTE on training only.
    # For this very large dataset, cap the data supplied to SMOTE so the
    # operation remains practical on a normal laptop.
    if len(X_train) > SMOTE_MAX_TRAIN_ROWS:
        print(
            f"Detection training set is {len(X_train):,} rows. "
            f"Sampling {SMOTE_MAX_TRAIN_ROWS:,} rows for SMOTE to limit RAM use."
        )

        sample_idx, _ = train_test_split(
            np.arange(len(X_train)),
            train_size=SMOTE_MAX_TRAIN_ROWS,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )

        X_smote = X_train[sample_idx]
        y_smote = y_train[sample_idx]
    else:
        X_smote = X_train
        y_smote = y_train

    min_class = np.bincount(y_smote).min()

    if min_class >= 2:
        k_neighbors = max(1, min(5, min_class - 1))

        print(
            f"Running SMOTE on {len(X_smote):,} training rows "
            f"with k_neighbors={k_neighbors}..."
        )

        smote = SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=k_neighbors,
        )

        X_bal, y_bal = smote.fit_resample(X_smote, y_smote)

        print(
            f"After SMOTE: {len(X_bal):,} rows "
            f"(SMOTE working set was {len(X_smote):,})"
        )

        det_train_bal = pd.DataFrame(
            X_bal.astype(np.float32),
            columns=feature_cols,
        )
        det_train_bal["Label_enc"] = y_bal.astype(np.int32)

        # Keep only model columns in detection files, matching the original
        # contract.
        det_train = det_train_bal

    else:
        print("SMOTE skipped: a class has fewer than 2 training samples.")

        det_train = pd.DataFrame(
            X_train,
            columns=feature_cols,
        )
        det_train["Label_enc"] = y_train

    return det_train, det_val, det_test


def build_windows(
    segment: pd.DataFrame,
    feature_cols,
    k: int,
    h: int,
) -> pd.DataFrame:
    """
    Build forecasting windows efficiently.

    For each position:
      - mean and std of each feature over the previous k rows
      - attack rate over the previous k rows
      - fraction of attacks in the next h rows
      - future_attack_flag = 1 if any future attack exists
    """

    if len(segment) <= k + h:
        return pd.DataFrame()

    feats = segment[feature_cols].to_numpy(
        dtype=np.float32,
        copy=False,
    )

    is_attack = (
        segment[LABEL_COL].to_numpy() != "BENIGN"
    ).astype(np.float32)

    seq = segment["flow_seq"].to_numpy(dtype=np.int64)

    n = len(segment)
    m = n - k - h

    # The number of windows can be very large. Build the result directly
    # as a NumPy matrix instead of a Python list of dictionaries.
    n_features = len(feature_cols)

    result = np.empty(
        (m, n_features * 2 + 4),
        dtype=np.float32,
    )

    # Rolling calculations using cumulative sums.
    # This avoids creating a separate k-row array for every window.
    csum = np.vstack(
        [
            np.zeros((1, n_features), dtype=np.float64),
            np.cumsum(feats.astype(np.float64), axis=0),
        ]
    )

    csum_sq = np.vstack(
        [
            np.zeros((1, n_features), dtype=np.float64),
            np.cumsum(
                np.square(feats.astype(np.float64)),
                axis=0,
            ),
        ]
    )

    attack_csum = np.concatenate(
        [[0.0], np.cumsum(is_attack, dtype=np.float64)]
    )

    # Forecast horizon attack counts.
    future_attack_counts = (
        attack_csum[k + h : n]
        - attack_csum[k : n - h]
    )

    for row_idx, i in enumerate(range(k, n - h)):
        start = i - k
        end = i

        window_sum = csum[end] - csum[start]
        window_sq_sum = csum_sq[end] - csum_sq[start]

        mean = window_sum / k
        variance = np.maximum(
            (window_sq_sum / k) - np.square(mean),
            0.0,
        )
        std = np.sqrt(variance)

        result[row_idx, :n_features] = mean.astype(np.float32)
        result[row_idx, n_features : 2 * n_features] = std.astype(
            np.float32
        )

        past_attack_count = (
            attack_csum[i] - attack_csum[i - k]
        )

        result[row_idx, 2 * n_features] = (
            past_attack_count / k
        )

        result[row_idx, 2 * n_features + 1] = (
            future_attack_counts[row_idx] / h
        )

        result[row_idx, 2 * n_features + 2] = (
            1.0 if future_attack_counts[row_idx] > 0 else 0.0
        )

        result[row_idx, 2 * n_features + 3] = seq[i - 1]

    columns = (
        [f"{c}_mean" for c in feature_cols]
        + [f"{c}_std" for c in feature_cols]
        + [
            "lookback_attack_rate",
            "future_attack_fraction",
            "future_attack_flag",
            "window_end_seq",
        ]
    )

    return pd.DataFrame(result, columns=columns)


def build_replay_stream(
    cleaned_chrono: pd.DataFrame,
    feature_cols,
) -> pd.DataFrame:
    """Use the final chronological 15% as the human-readable replay stream."""

    n = len(cleaned_chrono)

    test_start = n - int(
        n * (1 - FORECAST_TRAIN_FRAC - FORECAST_VAL_FRAC)
    )

    stream = cleaned_chrono.iloc[test_start:].reset_index(drop=True)

    cols = (
        ["flow_seq", "capture_day"]
        + feature_cols
        + [LABEL_COL]
    )

    return stream[cols]


def write_csv_chunked(df: pd.DataFrame, path: str, chunk_size=100_000):
    """
    Write CSV in chunks so pandas does not need another giant temporary
    representation during output.
    """

    first = True

    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start : start + chunk_size]

        chunk.to_csv(
            path,
            mode="w" if first else "a",
            header=first,
            index=False,
        )

        first = False


def write_schema(
    out_dir,
    feature_cols,
    label_encoder,
):
    schema = {
        "dataset": "CICIDS2017",
        "storage_format": "csv",
        "chronology_note": (
            "Source CSVs have no native timestamp column. Chronological order "
            "is a proxy: files concatenated in real capture-day order "
            "(Monday..Friday-afternoon per FILE_ORDER), then a monotonic "
            "'flow_seq' integer stands in for a timestamp. Within-file order "
            "reflects CICFlowMeter's write order, not a captured timestamp."
        ),
        "file_order": FILE_ORDER,
        "order_column": "flow_seq",
        "provenance_column": "capture_day",
        "feature_columns": feature_cols,
        "label_column": LABEL_COL,
        "label_encoded_column": "Label_enc",
        "label_classes": {
            str(cls): int(idx)
            for idx, cls in enumerate(label_encoder.classes_)
        },
        "lookback_k": LOOKBACK_K,
        "horizon_h": HORIZON_H,
        "detection_split": {
            "train_frac": DETECTION_TRAIN_FRAC,
            "val_frac": DETECTION_VAL_FRAC,
            "test_frac": round(
                1 - DETECTION_TRAIN_FRAC - DETECTION_VAL_FRAC,
                2,
            ),
            "method": "random, stratified by label",
            "smote_applied_to": "training split only",
        },
        "forecast_split": {
            "train_frac": FORECAST_TRAIN_FRAC,
            "val_frac": FORECAST_VAL_FRAC,
            "test_frac": round(
                1 - FORECAST_TRAIN_FRAC - FORECAST_VAL_FRAC,
                2,
            ),
            "method": "chronological, no shuffle, windows built within each split",
            "smote_applied_to": "none — SMOTE is never applied to this path",
        },
        "artifacts": {
            "scaler": "scaler.pkl",
            "encoders": "encoders.pkl",
            "detection_files": [
                "detection_train.csv",
                "detection_val.csv",
                "detection_test.csv",
            ],
            "forecast_window_files": [
                "forecast_train.csv",
                "forecast_val.csv",
                "forecast_test.csv",
            ],
            "replay_stream_file": "replay_stream.csv",
        },
    }

    path = os.path.join(out_dir, "feature_schema.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        required=True,
        help="Folder containing the 8 CICIDS2017 CSVs",
    )

    parser.add_argument(
        "--out-dir",
        default="./output",
        help="Folder for Module 1 outputs",
    )

    parser.add_argument(
        "--nrows-per-file",
        type=int,
        default=None,
        help="Debug only: cap rows loaded per file for a quick smoke test",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("MODULE 1 — MEMORY-SAFE PREPROCESSING")
    print("=" * 70)

    print("\n== Step 1-2: load + clean ==")

    data = load_and_clean(
        args.raw_dir,
        nrows_per_file=args.nrows_per_file,
    )

    feature_cols = get_feature_columns(data)

    print(f"{len(feature_cols)} feature columns")

    # ------------------------------------------------------------------
    # Fit shared preprocessing artifacts.
    # ------------------------------------------------------------------
    print(
        "\n== Step 3: fit shared encoder + scaler "
        "on chronological TRAIN data only =="
    )

    label_encoder, scaler = fit_shared_pipeline(
        data,
        feature_cols,
    )

    with open(
        os.path.join(args.out_dir, "encoders.pkl"),
        "wb",
    ) as f:
        pickle.dump(
            {
                "label_encoder": label_encoder,
                "feature_columns": feature_cols,
            },
            f,
        )

    with open(
        os.path.join(args.out_dir, "scaler.pkl"),
        "wb",
    ) as f:
        pickle.dump(scaler, f)

    print("Saved scaler.pkl and encoders.pkl")

    # ------------------------------------------------------------------
    # Forecast split first because it defines the leakage-safe scaler fit.
    # ------------------------------------------------------------------
    print(
        "\n== Step 4: chronological forecasting split =="
    )

    fc_train_raw, fc_val_raw, fc_test_raw = chronological_split(data)

    print("\nScaling forecasting splits...")

    fc_train_scaled = transform_split(
        fc_train_raw,
        feature_cols,
        label_encoder,
        scaler,
    )

    fc_val_scaled = transform_split(
        fc_val_raw,
        feature_cols,
        label_encoder,
        scaler,
    )

    fc_test_scaled = transform_split(
        fc_test_raw,
        feature_cols,
        label_encoder,
        scaler,
    )

    # ------------------------------------------------------------------
    # Detection split.
    # ------------------------------------------------------------------
    print(
        "\n== Step 5: detection fork "
        "(random, stratified, SMOTE on train only) =="
    )

    det_train, det_val, det_test = build_detection_fork(
        data,
        feature_cols,
        label_encoder,
        scaler,
        args.out_dir,
    )

    print("Writing detection files...")

    det_train.to_csv(
        os.path.join(args.out_dir, "detection_train.csv"),
        index=False,
    )

    det_val.to_csv(
        os.path.join(args.out_dir, "detection_val.csv"),
        index=False,
    )

    det_test.to_csv(
        os.path.join(args.out_dir, "detection_test.csv"),
        index=False,
    )

    print(
        "Wrote detection_train.csv / "
        "detection_val.csv / detection_test.csv"
    )

    del det_train, det_val, det_test

    # ------------------------------------------------------------------
    # Forecast windows.
    # ------------------------------------------------------------------
    print(
        "\n== Step 6: forecasting windows "
        "(chronological, no SMOTE) =="
    )

    print("Building forecast TRAIN windows...")
    fc_train = build_windows(
        fc_train_scaled,
        feature_cols,
        LOOKBACK_K,
        HORIZON_H,
    )

    print("Building forecast VAL windows...")
    fc_val = build_windows(
        fc_val_scaled,
        feature_cols,
        LOOKBACK_K,
        HORIZON_H,
    )

    print("Building forecast TEST windows...")
    fc_test = build_windows(
        fc_test_scaled,
        feature_cols,
        LOOKBACK_K,
        HORIZON_H,
    )

    print(
        f"Forecast windows -> "
        f"train {len(fc_train):,} / "
        f"val {len(fc_val):,} / "
        f"test {len(fc_test):,}"
    )

    print("Writing forecast files...")

    fc_train.to_csv(
        os.path.join(args.out_dir, "forecast_train.csv"),
        index=False,
    )

    fc_val.to_csv(
        os.path.join(args.out_dir, "forecast_val.csv"),
        index=False,
    )

    fc_test.to_csv(
        os.path.join(args.out_dir, "forecast_test.csv"),
        index=False,
    )

    del fc_train, fc_val, fc_test
    del fc_train_scaled, fc_val_scaled, fc_test_scaled

    # ------------------------------------------------------------------
    # Replay stream.
    # ------------------------------------------------------------------
    print(
        "\n== Step 7: replay stream for live demo =="
    )

    replay = build_replay_stream(
        data,
        feature_cols,
    )

    replay_path = os.path.join(
        args.out_dir,
        "replay_stream.csv",
    )

    replay.to_csv(
        replay_path,
        index=False,
    )

    print(
        f"Wrote replay_stream.csv "
        f"({len(replay):,} rows)"
    )

    del replay

    # ------------------------------------------------------------------
    # Schema.
    # ------------------------------------------------------------------
    print("\n== Step 8: freeze schema ==")

    write_schema(
        args.out_dir,
        feature_cols,
        label_encoder,
    )

    print("\n" + "=" * 70)
    print("DONE — Module 1 artifacts generated.")
    print("=" * 70)

    print(f"\nOutput folder:\n{os.path.abspath(args.out_dir)}")

    print(
        """
Expected files:
  feature_schema.json
  scaler.pkl
  encoders.pkl
  detection_train.csv
  detection_val.csv
  detection_test.csv
  forecast_train.csv
  forecast_val.csv
  forecast_test.csv
  replay_stream.csv
"""
    )


if __name__ == "__main__":
    main()
