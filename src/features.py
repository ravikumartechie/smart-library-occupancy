"""Cleaning, feature engineering and chronological split.

Run:  python src/features.py
Writes train/val/test CSVs to data/processed/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/library_occupancy.csv")
OUT = Path("data/processed")
ROWS_PER_DAY = 14            # library open 08:00..21:00 -> 14 hourly rows
TRAIN_END = pd.Timestamp("2025-12-31")
VAL_END = pd.Timestamp("2026-03-31")
PEAK_THRESHOLD = 0.70        # occupancy rate >= 70% counts as a peak hour

CALENDAR_FEATURES = ["is_weekend", "is_holiday", "is_exam", "is_vacation", "days_to_exam"]
TIME_FEATURES = ["hour", "dow", "month", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
LAG_FEATURES = ["lag_recent", "lag_day", "lag_week", "roll_mean_3", "roll_mean_day"]
FEATURES = ["zone", "capacity"] + CALENDAR_FEATURES + TIME_FEATURES + LAG_FEATURES


def load_and_clean(path: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["zone", "timestamp"]).reset_index(drop=True)
    # impossible readings (more people than capacity) are sensor glitches
    df.loc[df["occupancy"] > df["capacity"], "occupancy"] = np.nan
    # forward-fill within each zone: uses past values only (no leakage)
    df["occupancy"] = df.groupby("zone")["occupancy"].ffill()
    df["rate"] = df["occupancy"] / df["capacity"]
    return df


def check_full_calendar(df: pd.DataFrame) -> None:
    """Row-based lags are only valid if every zone has 14 rows on every day."""
    per_day = df.groupby(["zone", df["timestamp"].dt.date]).size()
    assert (per_day == ROWS_PER_DAY).all(), "calendar has gaps: row-based lags invalid"


def build_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Features for predicting `rate` `horizon` rows ahead (horizon <= 14)."""
    assert 1 <= horizon <= ROWS_PER_DAY
    d = df.copy()
    d["hour"] = d["timestamp"].dt.hour
    d["dow"] = d["timestamp"].dt.dayofweek
    d["month"] = d["timestamp"].dt.month
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["dow_sin"] = np.sin(2 * np.pi * d["dow"] / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["dow"] / 7)

    g = d.groupby("zone")["rate"]
    d["lag_recent"] = g.shift(horizon)                      # latest value we would know
    d["lag_day"] = g.shift(ROWS_PER_DAY)                    # same hour yesterday
    d["lag_week"] = g.shift(7 * ROWS_PER_DAY)               # same hour last week
    d["roll_mean_3"] = g.transform(lambda s: s.shift(horizon).rolling(3).mean())
    d["roll_mean_day"] = g.transform(lambda s: s.shift(horizon).rolling(ROWS_PER_DAY).mean())

    d["is_peak"] = (d["rate"] >= PEAK_THRESHOLD).astype(int)
    # first week per zone has no lag_week; rows with missing target are dropped too
    d = d.dropna(subset=LAG_FEATURES + ["rate"]).reset_index(drop=True)
    return d


def chronological_split(d: pd.DataFrame):
    day = d["timestamp"].dt.normalize()
    train = d[day <= TRAIN_END]
    val = d[(day > TRAIN_END) & (day <= VAL_END)]
    test = d[day > VAL_END]
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()
    return train, val, test


def main(horizon: int = 1):
    df = load_and_clean()
    check_full_calendar(df)
    d = build_features(df, horizon=horizon)
    train, val, test = chronological_split(d)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(OUT / f"{name}_h{horizon}.csv", index=False)
        print(f"{name:5s} rows={len(part):6d}  "
              f"{part['timestamp'].min().date()} -> {part['timestamp'].max().date()}  "
              f"peak share={part['is_peak'].mean():.3f}")
    print("features used:", FEATURES)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)