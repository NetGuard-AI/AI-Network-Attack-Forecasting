"""Rolling chronological window used by the MVP backend.

This mirrors the current M1/M3 row-window design: 20 flows per lookback and
three prior window attack-rate lags. It is flow-count based; the minute horizon
is reported only from feature_schema.json and is never invented here.
"""
from collections import deque
import numpy as np

LOOKBACK_K = 20

class LiveForecastWindow:
    def __init__(self, feature_cols, model_feature_order):
        self.feature_cols = feature_cols
        self.model_feature_order = model_feature_order
        self.flow_buffer = deque(maxlen=LOOKBACK_K)
        self.is_attack_buffer = deque(maxlen=LOOKBACK_K)
        self.rate_history = deque(maxlen=3)

    def add_flow(self, scaled_feature_vector, is_attack: int):
        self.flow_buffer.append(np.asarray(scaled_feature_vector, dtype=np.float32))
        self.is_attack_buffer.append(int(is_attack))

    def build_features(self):
        if len(self.flow_buffer) < LOOKBACK_K:
            return None
        arr = np.stack(self.flow_buffer)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)
        rate = float(np.mean(self.is_attack_buffer))
        features = {}
        for i, name in enumerate(self.feature_cols):
            features[f"{name}_mean"] = float(means[i])
            features[f"{name}_std"] = float(stds[i])
        history = list(self.rate_history)
        # Early windows are padded with the current rate because no older
        # windows exist yet. This is deterministic and keeps the MVP live.
        while len(history) < 3:
            history.append(rate)
        features["lookback_attack_rate"] = rate
        features["lookback_attack_rate_lag1"] = history[-1]
        features["lookback_attack_rate_lag2"] = history[-2]
        features["lookback_attack_rate_lag3"] = history[-3]
        self.rate_history.append(rate)
        missing = [n for n in self.model_feature_order if n not in features]
        if missing:
            raise ValueError("Live forecast feature mismatch: " + ", ".join(missing[:10]))
        return {n: features[n] for n in self.model_feature_order}
