"""Book-borrowing demand forecast per category, 7 days ahead.

Run:  python src/resources.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_generation import build_calendar

RAW = Path("data/raw/book_borrowing.csv")
MODELS, REPORTS = Path("models"), Path("reports")
HORIZON = 7                                   # forecast one week ahead
TRAIN_END, VAL_END = pd.Timestamp("2025-12-31"), pd.Timestamp("2026-03-31")
CAL = ["day_of_week", "is_weekend", "is_holiday", "is_exam", "is_vacation", "days_to_exam"]
LAGS = ["lag_h", "lag_h14", "roll_mean_7"]
NUMERIC = CAL + LAGS
FEATURES = ["category"] + NUMERIC


def build_dataset() -> pd.DataFrame:
    b = pd.read_csv(RAW, parse_dates=["date"]).sort_values(["category", "date"])
    d = b.merge(build_calendar(), on="date")
    g = d.groupby("category")["books_borrowed"]
    d["lag_h"] = g.shift(HORIZON)             # same weekday last week
    d["lag_h14"] = g.shift(HORIZON + 7)       # same weekday two weeks ago
    d["roll_mean_7"] = g.transform(lambda s: s.shift(HORIZON).rolling(7).mean())
    return d.dropna(subset=LAGS).reset_index(drop=True)


def make_pipe(model, scale=False):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]),
        ("num", StandardScaler() if scale else "passthrough", NUMERIC)])
    return Pipeline([("prep", pre), ("model", model)])


def score(name, y, pred):
    pred = np.clip(np.asarray(pred, dtype=float), 0, None)
    return {"model": name, "MAE": mean_absolute_error(y, pred),
            "WAPE": np.abs(y.values - pred).sum() / y.sum()}


def baseline_rows(fit_df, eval_df):
    prof = (fit_df.groupby(["category", "day_of_week"])["books_borrowed"].mean()
            .rename("hist_avg").reset_index())
    m = eval_df.merge(prof, on=["category", "day_of_week"], how="left")
    y = eval_df["books_borrowed"]
    return [score("Baseline: same weekday last week", y, eval_df["lag_h"]),
            score("Baseline: historical avg (category/weekday)", y, m["hist_avg"])]


def lgbm():
    return make_pipe(LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                   min_child_samples=20, random_state=42, verbose=-1))


def main():
    d = build_dataset()
    train = d[d.date <= TRAIN_END]
    val = d[(d.date > TRAIN_END) & (d.date <= VAL_END)]
    test = d[d.date > VAL_END]
    pd.set_option("display.width", 200)

    # ---- model comparison on validation ----
    rows = baseline_rows(train, val)
    for name, pipe in {"Ridge": make_pipe(Ridge(alpha=1.0), scale=True),
                       "LightGBM": lgbm()}.items():
        pipe.fit(train[FEATURES], train["books_borrowed"])
        rows.append(score(name, val["books_borrowed"], pipe.predict(val[FEATURES])))
    res = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    res.to_csv(REPORTS / "borrowing_comparison_val.csv", index=False)
    print("VALIDATION (7-day-ahead books borrowed per category/day)\n", res.round(3).to_string(index=False))

    # ---- final: refit on train+val, score once on test ----
    full = pd.concat([train, val])
    final = lgbm().fit(full[FEATURES], full["books_borrowed"])
    joblib.dump(final, MODELS / "borrowing_lightgbm.joblib")
    test = test.copy()
    test["pred"] = np.clip(final.predict(test[FEATURES]), 0, None)
    rows = baseline_rows(full, test) + [score("FINAL: LightGBM", test["books_borrowed"], test["pred"])]
    res = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    res.to_csv(REPORTS / "borrowing_test_results.csv", index=False)
    print("\nTEST\n", res.round(3).to_string(index=False))
    by_cat = test.assign(ae=(test.pred - test.books_borrowed).abs()).groupby("category").agg(
        MAE=("ae", "mean"), avg_daily=("books_borrowed", "mean")).round(2)
    print("\nTest error by category\n", by_cat)


if __name__ == "__main__":
    main()