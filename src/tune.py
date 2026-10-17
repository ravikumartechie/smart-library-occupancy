"""Hyperparameter tuning (Optuna, time-aware CV) and a stacking ensemble.

Run:  python src/tune.py        (1-hour horizon)
      python src/tune.py 4      (4-hour horizon)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from features import FEATURES
from train import SEED, load, make_pipeline, score

optuna.logging.set_verbosity(optuna.logging.WARNING)
MODELS_DIR = Path("models")
REPORTS = Path("reports")
N_TRIALS = 30


# ---------- model builders ----------
def build_lgbm(p):
    return make_pipeline(LGBMRegressor(**p, subsample_freq=1, random_state=SEED, verbose=-1))


def build_xgb(p):
    return make_pipeline(XGBRegressor(**p, random_state=SEED))


def build_rf():
    return make_pipeline(RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=SEED))


# ---------- search spaces ----------
def suggest_lgbm(t):
    return dict(
        n_estimators=t.suggest_int("n_estimators", 200, 800),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=t.suggest_int("num_leaves", 15, 127),
        min_child_samples=t.suggest_int("min_child_samples", 10, 60),
        subsample=t.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )


def suggest_xgb(t):
    return dict(
        n_estimators=t.suggest_int("n_estimators", 200, 800),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=t.suggest_int("max_depth", 3, 8),
        min_child_weight=t.suggest_int("min_child_weight", 1, 10),
        subsample=t.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )


def tune(name, suggest, build, X, y, horizon, n_trials=N_TRIALS):
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = suggest(trial)
        maes = []
        for tr, te in tscv.split(X):
            m = build(params)
            m.fit(X.iloc[tr], y.iloc[tr])
            maes.append(mean_absolute_error(y.iloc[te], m.predict(X.iloc[te])))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    study.trials_dataframe().to_csv(REPORTS / f"optuna_{name}_h{horizon}.csv", index=False)
    print(f"{name}: best CV MAE = {study.best_value:.4f}")
    return study.best_params


# ---------- stacking with out-of-fold predictions ----------
def oof_predictions(builders, X, y, n_splits=4):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.full((len(X), len(builders)), np.nan)
    for tr, te in tscv.split(X):
        for j, build in enumerate(builders.values()):
            m = build()
            m.fit(X.iloc[tr], y.iloc[tr])
            oof[te, j] = m.predict(X.iloc[te])
    return oof, ~np.isnan(oof).any(axis=1)   # first block has no OOF prediction


def main(horizon=1):
    train, val = load(horizon)
    train = train.sort_values("timestamp").reset_index(drop=True)   # time order for CV
    val = val.sort_values("timestamp").reset_index(drop=True)
    X_tr, y_tr = train[FEATURES], train["rate"]
    X_va, y_va = val[FEATURES], val["rate"]
    MODELS_DIR.mkdir(exist_ok=True)

    t0 = time.time()
    lgb_best = tune("LightGBM", suggest_lgbm, build_lgbm, X_tr, y_tr, horizon)
    xgb_best = tune("XGBoost", suggest_xgb, build_xgb, X_tr, y_tr, horizon)
    print(f"tuning took {time.time() - t0:.0f}s")
    (MODELS_DIR / f"best_params_h{horizon}.json").write_text(
        json.dumps({"lightgbm": lgb_best, "xgboost": xgb_best}, indent=2))

    builders = {
        "LightGBM (tuned)": lambda: build_lgbm(lgb_best),
        "XGBoost (tuned)": lambda: build_xgb(xgb_best),
        "Random Forest": build_rf,
    }

    rows, val_preds = [], []
    for name, build in builders.items():
        t1 = time.time()
        m = build()
        m.fit(X_tr, y_tr)
        pred = m.predict(X_va)
        val_preds.append(pred)
        rows.append(score(name, y_va, pred, time.time() - t1))

    oof, mask = oof_predictions(builders, X_tr, y_tr)
    meta = LinearRegression(positive=True).fit(oof[mask], y_tr[mask])
    stack_pred = meta.predict(np.column_stack(val_preds))
    rows.append(score("Stacking ensemble", y_va, stack_pred))
    print("stacking weights:", dict(zip(builders, meta.coef_.round(3))),
          "| intercept:", round(float(meta.intercept_), 4))

    base = pd.read_csv(REPORTS / f"model_comparison_val_h{horizon}.csv")
    final = pd.concat([base, pd.DataFrame(rows)]).sort_values("MAE").reset_index(drop=True)
    final.to_csv(REPORTS / f"model_comparison_tuned_val_h{horizon}.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"\nValidation results incl. tuned models, horizon = {horizon}\n")
    print(final.round(4).to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)