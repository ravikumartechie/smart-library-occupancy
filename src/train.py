"""Compare baselines and ML models on the validation set.

Run:  python src/train.py        (1-hour horizon)
      python src/train.py 4      (4-hour horizon)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from features import FEATURES, PEAK_THRESHOLD

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
CATEGORICAL = ["zone"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
SEED = 42


def load(horizon: int):
    train = pd.read_csv(PROCESSED / f"train_h{horizon}.csv", parse_dates=["timestamp"])
    val = pd.read_csv(PROCESSED / f"val_h{horizon}.csv", parse_dates=["timestamp"])
    return train, val


def make_pipeline(model, scale: bool = False) -> Pipeline:
    pre = ColumnTransformer([
        ("zone", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler() if scale else "passthrough", NUMERIC),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def get_models() -> dict:
    return {
        "Ridge": make_pipeline(Ridge(alpha=1.0), scale=True),
        "Random Forest": make_pipeline(RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=SEED)),
        "XGBoost": make_pipeline(XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED)),
        "LightGBM": make_pipeline(LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=SEED, verbose=-1)),
    }


def score(name: str, y_true, y_pred, fit_seconds: float = 0.0) -> dict:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, 1)
    true_peak = (y_true >= PEAK_THRESHOLD).astype(int)
    pred_peak = (y_pred >= PEAK_THRESHOLD).astype(int)
    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "peak_precision": precision_score(true_peak, pred_peak, zero_division=0),
        "peak_recall": recall_score(true_peak, pred_peak, zero_division=0),
        "peak_F1": f1_score(true_peak, pred_peak, zero_division=0),
        "fit_sec": round(fit_seconds, 1),
    }


def baseline_scores(train: pd.DataFrame, val: pd.DataFrame) -> list:
    y = val["rate"]
    out = [
        score("Baseline: persistence (lag_recent)", y, val["lag_recent"]),
        score("Baseline: same hour last week", y, val["lag_week"]),
    ]
    # historical average per zone x weekday x hour, learned on TRAIN only
    profile = (train.groupby(["zone", "dow", "hour"])["rate"].mean()
               .rename("hist_avg").reset_index())
    merged = val.merge(profile, on=["zone", "dow", "hour"], how="left")
    merged["hist_avg"] = merged["hist_avg"].fillna(train["rate"].mean())
    out.append(score("Baseline: historical avg (zone/weekday/hour)", y, merged["hist_avg"]))
    return out


def main(horizon: int = 1):
    train, val = load(horizon)
    X_tr, y_tr = train[FEATURES], train["rate"]
    X_va, y_va = val[FEATURES], val["rate"]

    rows = baseline_scores(train, val)
    for name, pipe in get_models().items():
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        rows.append(score(name, y_va, pipe.predict(X_va), time.time() - t0))

    results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    REPORTS.mkdir(exist_ok=True)
    results.to_csv(REPORTS / f"model_comparison_val_h{horizon}.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"\nValidation results, horizon = {horizon} hour(s)\n")
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)