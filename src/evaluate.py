"""Final test-set evaluation and error analysis.

Run:  python src/evaluate.py        (1-hour horizon)
      python src/evaluate.py 4      (4-hour horizon)
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score

from features import FEATURES, PEAK_THRESHOLD
from train import baseline_scores, score
from tune import build_lgbm

PROCESSED, MODELS, REPORTS = Path("data/processed"), Path("models"), Path("reports")
FIG = REPORTS / "figures"


def peak_metrics(y_true, pred, thr):
    t = (np.asarray(y_true) >= PEAK_THRESHOLD).astype(int)
    p = (np.asarray(pred) >= thr).astype(int)
    return (precision_score(t, p, zero_division=0), recall_score(t, p, zero_division=0),
            f1_score(t, p, zero_division=0), fbeta_score(t, p, beta=2, zero_division=0))


def read(name, horizon):
    return pd.read_csv(PROCESSED / f"{name}_h{horizon}.csv", parse_dates=["timestamp"])


def main(horizon=1):
    FIG.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    pd.set_option("display.width", 200, "display.max_columns", 20)
    train, val, test = read("train", horizon), read("val", horizon), read("test", horizon)
    params = json.loads((MODELS / f"best_params_h{horizon}.json").read_text())["lightgbm"]

    # 1) choose the alert threshold on VALIDATION (model trained on train only)
    m_tr = build_lgbm(params).fit(train[FEATURES], train["rate"])
    val_pred = np.clip(m_tr.predict(val[FEATURES]), 0, 1)
    sweep = pd.DataFrame(
        [(round(t, 2), *peak_metrics(val["rate"], val_pred, t)) for t in np.arange(0.40, 0.71, 0.05)],
        columns=["alert_threshold", "precision", "recall", "F1", "F2"])
    best_thr = float(sweep.loc[sweep["F2"].idxmax(), "alert_threshold"])
    print("Alert-threshold sweep on VALIDATION:\n", sweep.round(3).to_string(index=False))
    print(f"-> chosen alert threshold: {best_thr}\n")

    # 2) final model on train + validation, scored once on TEST
    full = pd.concat([train, val]).sort_values("timestamp").reset_index(drop=True)
    final = build_lgbm(params).fit(full[FEATURES], full["rate"])
    joblib.dump(final, MODELS / f"final_lightgbm_h{horizon}.joblib")
    test = test.sort_values(["zone", "timestamp"]).reset_index(drop=True)
    test["pred"] = np.clip(final.predict(test[FEATURES]), 0, 1)

    rows = baseline_scores(full, test) + [score("FINAL: tuned LightGBM", test["rate"], test["pred"])]
    res = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    res.to_csv(REPORTS / f"test_results_h{horizon}.csv", index=False)
    print(f"TEST results, horizon = {horizon}\n", res.round(4).to_string(index=False), "\n")

    for thr in sorted({PEAK_THRESHOLD, best_thr}):
        p, r, f1, f2 = peak_metrics(test["rate"], test["pred"], thr)
        print(f"Peak detection on TEST at alert threshold {thr}: "
              f"precision={p:.3f} recall={r:.3f} F1={f1:.3f}")

    # 3) error analysis
    test["err"] = test["pred"] - test["rate"]
    test["abs_err"] = test["err"].abs()
    test["period"] = np.select(
        [test.is_exam == 1, test.is_vacation == 1, test.is_holiday == 1, test.is_weekend == 1],
        ["Exam", "Vacation", "Holiday", "Weekend"], default="Regular")
    print("\nMAE and bias (pred - actual) by zone:\n",
          test.groupby("zone").agg(MAE=("abs_err", "mean"), bias=("err", "mean")).round(4))
    print("\nMAE and bias by period type:\n",
          test.groupby("period").agg(rows=("abs_err", "size"), MAE=("abs_err", "mean"),
                                     bias=("err", "mean")).round(4))
    print("\nMAE by hour:\n", test.groupby("hour")["abs_err"].mean().round(4).to_string())
    daily = test.groupby(test["timestamp"].dt.date)["abs_err"].mean()
    print("\n8 worst days (mean absolute error):\n", daily.nlargest(8).round(4).to_string())
    test.to_csv(REPORTS / f"test_predictions_h{horizon}.csv", index=False)

    # 4) figures
    plt.figure(figsize=(5.5, 5))
    sns.scatterplot(data=test, x="rate", y="pred", hue="zone", s=10, alpha=0.5)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("Actual occupancy rate"); plt.ylabel("Predicted"); plt.title("Predicted vs actual (test)")
    plt.tight_layout(); plt.savefig(FIG / f"eval_pred_vs_actual_h{horizon}.png", dpi=150); plt.close()

    plt.figure(figsize=(9, 4))
    sns.barplot(data=test, x="hour", y="abs_err", hue="zone", errorbar=None)
    plt.ylabel("MAE"); plt.title("Test error by hour and zone")
    plt.tight_layout(); plt.savefig(FIG / f"eval_error_by_hour_h{horizon}.png", dpi=150); plt.close()

    wk = test[(test.zone == "Reading Hall") & (test.timestamp >= "2026-05-04") & (test.timestamp < "2026-05-11")]
    plt.figure(figsize=(11, 4))
    plt.plot(wk["timestamp"], wk["rate"], label="actual")
    plt.plot(wk["timestamp"], wk["pred"], label="predicted")
    plt.axhline(PEAK_THRESHOLD, color="red", ls=":", label="peak threshold")
    plt.legend(); plt.title("Reading Hall, one exam week (test)")
    plt.tight_layout(); plt.savefig(FIG / f"eval_week_h{horizon}.png", dpi=150); plt.close()
    print("\nSaved figures to", FIG)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)