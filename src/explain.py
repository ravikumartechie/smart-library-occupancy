"""SHAP explainability for the final LightGBM model.

Run:  python src/explain.py        (1-hour horizon)
      python src/explain.py 4
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from features import FEATURES

PROCESSED, MODELS, REPORTS = Path("data/processed"), Path("models"), Path("reports")
FIG = REPORTS / "figures"
GROUPS = {
    "recent history (lags, rolling means)": ["lag_recent", "lag_day", "lag_week",
                                             "roll_mean_3", "roll_mean_day"],
    "calendar (exam / holiday / vacation)": ["is_weekend", "is_holiday", "is_exam",
                                             "is_vacation", "days_to_exam"],
    "time of day / week": ["hour", "dow", "month", "hour_sin", "hour_cos",
                           "dow_sin", "dow_cos"],
}


def to_frame(pipe, df):
    """Apply the fitted preprocessing and return a DataFrame with readable names."""
    pre = pipe.named_steps["prep"]
    X = pre.transform(df[FEATURES])
    if hasattr(X, "toarray"):
        X = X.toarray()
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    return pd.DataFrame(X, columns=names, index=df.index)


def group_of(name):
    for group, cols in GROUPS.items():
        if name in cols:
            return group
    return "zone / capacity"


def explain_row(pipe, explainer, row_df, top=4):
    """Top contributions for one row: list of (feature, value, shap_value)."""
    X = to_frame(pipe, row_df)
    sv = explainer.shap_values(X)[0]
    order = np.argsort(-np.abs(sv))[:top]
    return [(X.columns[i], float(X.iloc[0, i]), float(sv[i])) for i in order]


def main(horizon=1):
    FIG.mkdir(parents=True, exist_ok=True)
    pipe = joblib.load(MODELS / f"final_lightgbm_h{horizon}.joblib")
    test = pd.read_csv(PROCESSED / f"test_h{horizon}.csv", parse_dates=["timestamp"])
    sample = test.sample(n=min(1500, len(test)), random_state=42).sort_index()

    X = to_frame(pipe, sample)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    sv = explainer.shap_values(X)
    base = float(np.ravel(explainer.expected_value)[0])

    # ---- global importance ----
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
    print(f"Mean |SHAP| per feature (horizon {horizon}):\n", imp.round(4).head(12).to_string())
    grp = imp.groupby(imp.index.map(group_of)).sum()
    print("\nShare of total importance by feature family:\n", (grp / grp.sum()).round(3).to_string())
    imp.to_csv(REPORTS / f"shap_importance_h{horizon}.csv", header=["mean_abs_shap"])

    # ---- plots ----
    shap.summary_plot(sv, X, plot_type="bar", show=False, max_display=12)
    plt.savefig(FIG / f"shap_bar_h{horizon}.png", dpi=150, bbox_inches="tight"); plt.close()

    shap.summary_plot(sv, X, show=False, max_display=12)
    plt.savefig(FIG / f"shap_beeswarm_h{horizon}.png", dpi=150, bbox_inches="tight"); plt.close()

    for feat in ["lag_recent", "hour", "days_to_exam"]:
        shap.dependence_plot(feat, sv, X, show=False)
        plt.savefig(FIG / f"shap_dependence_{feat}_h{horizon}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ---- one local explanation: the busiest predicted exam-period hour ----
    pred = np.clip(pipe.predict(sample), 0, 1)
    cand = np.where(sample["is_exam"].values == 1)[0]
    i = int(cand[np.argmax(pred[cand])]) if len(cand) else int(np.argmax(pred))
    exp = shap.Explanation(values=sv[i], base_values=base,
                           data=X.iloc[i].values, feature_names=list(X.columns))
    shap.plots.waterfall(exp, max_display=10, show=False)
    plt.savefig(FIG / f"shap_waterfall_h{horizon}.png", dpi=150, bbox_inches="tight"); plt.close()

    row = sample.iloc[i]
    print(f"\nLocal explanation: {row['timestamp']} | {row['zone']} | "
          f"actual={row['rate']:.2f} predicted={pred[i]:.2f} (base value {base:.2f})")
    for name, val, contrib in explain_row(pipe, explainer, sample.iloc[[i]], top=5):
        print(f"  {name:15s} = {val:8.3f}  ->  {contrib:+.3f}")
    print("\nSaved SHAP figures to", FIG)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)