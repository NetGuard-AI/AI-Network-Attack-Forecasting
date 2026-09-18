import numpy as np
from live_forecast_window import LiveForecastWindow

RAW = [f"f{i}" for i in range(78)]
MODEL = [f"f{i}_mean" for i in range(78)] + [f"f{i}_std" for i in range(78)] + [
    "lookback_attack_rate",
    "lookback_attack_rate_lag1",
    "lookback_attack_rate_lag2",
    "lookback_attack_rate_lag3",
]

w = LiveForecastWindow(RAW, MODEL)
for i in range(19):
    w.add_flow(np.full(78, i, dtype=float), i % 2)
    assert w.build_features() is None

w.add_flow(np.full(78, 19, dtype=float), 1)
row = w.build_features()
assert row is not None
assert len(row) == 160
assert row["f0_mean"] == 9.5
assert row["f0_std"] > 0
assert row["lookback_attack_rate"] == 0.5
print("OK: first feature row is produced on flow 20 and contains 160 features.")
