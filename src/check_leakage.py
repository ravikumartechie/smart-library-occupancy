"""Verify lag features only use past data."""
import numpy as np
from features import load_and_clean, build_features

df = load_and_clean()
d = build_features(df, horizon=1)

for zone in df["zone"].unique():
    z = d[d.zone == zone].reset_index(drop=True)
    raw = df[df.zone == zone].reset_index(drop=True)
    for i in [10, 500, 3000]:
        ts = z.loc[i, "timestamp"]
        prev = raw[raw.timestamp < ts].iloc[-1]                       # last known value
        day_ago = raw[raw.timestamp == ts - np.timedelta64(1, "D")]   # same hour yesterday
        assert abs(z.loc[i, "lag_recent"] - prev.rate) < 1e-9, (zone, ts)
        assert abs(z.loc[i, "lag_day"] - day_ago.rate.iloc[0]) < 1e-9, (zone, ts)

print("Leakage check passed: lag_recent and lag_day only use past values.")