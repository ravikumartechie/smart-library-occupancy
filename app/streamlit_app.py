"""Streamlit dashboard: Smart Library Occupancy & Resource Intelligence System.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import datetime as dt
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                        # src modules use paths relative to the repo root
sys.path.insert(0, str(ROOT / "src"))

import resources                                   # noqa: E402
from explain import explain_row                    # noqa: E402
from features import FEATURES, PEAK_THRESHOLD      # noqa: E402
from recommend import ALERT, recommend, status     # noqa: E402

st.set_page_config(page_title="Smart Library Occupancy", layout="wide")

REPORTS, MODELS = ROOT / "reports", ROOT / "models"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
COLOR = {"Free": "green", "Moderate": "orange", "Crowded": "red"}
NICE = {
    "lag_recent": "Latest known occupancy", "lag_day": "Same hour yesterday",
    "lag_week": "Same hour last week", "roll_mean_3": "Average of last 3 readings",
    "roll_mean_day": "Average of last day", "hour": "Hour of day",
    "hour_sin": "Hour of day (cyclic)", "hour_cos": "Hour of day (cyclic)",
    "dow": "Day of week", "dow_sin": "Day of week (cyclic)", "dow_cos": "Day of week (cyclic)",
    "month": "Month", "is_vacation": "Vacation period", "is_exam": "Exam period",
    "is_holiday": "Holiday", "is_weekend": "Weekend", "days_to_exam": "Days to next exam",
    "capacity": "Zone capacity",
}
SHAP_NOTES = {
    1: ("At a 1-hour horizon the model mostly continues the latest reading: recent-history "
        "features carry about 78% of the total importance, and the calendar only about 5%. "
        "Exam and vacation effects are already visible in the previous hour's occupancy."),
    4: ("At a 4-hour horizon the latest reading matters much less. The same hour last week "
        "and yesterday, plus the vacation flag, take over, and the calendar's share rises "
        "to about 24%. The further ahead we forecast, the more the model relies on routine."),
}


def nice(f: str) -> str:
    return "Zone: " + f[5:] if f.startswith("zone_") else NICE.get(f, f)


@st.cache_resource
def load_model(h):
    return joblib.load(MODELS / f"final_lightgbm_h{h}.joblib")


@st.cache_resource
def load_explainer(h):
    return shap.TreeExplainer(load_model(h).named_steps["model"])


@st.cache_data
def load_predictions(h):
    return pd.read_csv(REPORTS / f"test_predictions_h{h}.csv", parse_dates=["timestamp"])


@st.cache_data
def load_train_profile():
    t = pd.read_csv(ROOT / "data" / "processed" / "train_h1.csv")
    return t[["zone", "dow", "hour", "rate"]]


@st.cache_data
def borrowing_forecast():
    d = resources.build_dataset()
    avg = d.groupby("category")["books_borrowed"].mean()
    test = d[d["date"] > resources.VAL_END].copy()
    model = joblib.load(MODELS / "borrowing_lightgbm.joblib")
    test["pred"] = np.clip(model.predict(test[resources.FEATURES]), 0, None)
    return test, avg


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Controls")
h = 1 if st.sidebar.radio("Forecast horizon", ["1 hour ahead", "4 hours ahead"]).startswith("1") else 4
df = load_predictions(h)
model, explainer, alert = load_model(h), load_explainer(h), ALERT[h]

dates = sorted(df["timestamp"].dt.date.unique())
d = st.sidebar.date_input("Date", value=dt.date(2026, 4, 27), min_value=dates[0], max_value=dates[-1])
hours = [int(x) for x in sorted(df[df["timestamp"].dt.date == d]["timestamp"].dt.hour.unique())]
hour = st.sidebar.select_slider("Hour", options=hours, value=17 if 17 in hours else hours[0])
zone = st.sidebar.selectbox("Zone (for details)", sorted(df["zone"].unique()))
show_actual = st.sidebar.checkbox("Show actual occupancy (held-out test data)", value=False)
st.sidebar.caption(f"Alert threshold: {alert:.0%} (chosen on validation data)")

st.title("Smart Library Occupancy & Resource Intelligence")
st.caption("Demo on the held-out test period (Apr-Jun 2026) of a synthetic dataset. Each forecast uses only "
           f"information available {h} slot(s) earlier.")

tab_live, tab_day, tab_pat, tab_books, tab_model = st.tabs(
    ["Live forecast", "Day view", "Typical patterns", "Resource demand", "Model & explainability"])

# ------------------------------------------------------------------ live forecast
with tab_live:
    ts = pd.Timestamp(dt.datetime.combine(d, dt.time(int(hour))))
    snap = df[df["timestamp"] == ts].sort_values("zone").copy()
    if snap.empty:
        st.warning("No data for this date and hour.")
        st.stop()
    snap["pred"] = np.clip(model.predict(snap[FEATURES]), 0, 1)

    st.subheader(f"Forecast for {ts:%a %d %b %Y, %H:00}")
    cols = st.columns(len(snap))
    for col, (_, r) in zip(cols, snap.iterrows()):
        s = status(r["pred"], alert)
        col.metric(r["zone"], f"{r['pred']:.0%}")
        col.markdown(f":{COLOR[s]}[**{s}**]")
        col.caption(f"about {int(round((1 - r['pred']) * r['capacity']))} of {int(r['capacity'])} seats free")
        if show_actual:
            col.caption(f"Actual (test data): {r['rate']:.0%}")

    st.subheader("Recommendations")
    crowded = snap[snap["pred"] >= alert]["zone"].tolist()
    if not crowded:
        st.success("No zone is forecast to be crowded at this hour.")
    for z in crowded:
        opts = recommend(snap[["zone", "capacity", "pred"]], z, alert)
        if opts:
            txt = ", ".join(f"{o['zone']} (about {o['est_free_seats']} seats free)" for o in opts)
            st.warning(f"{z} is forecast to be crowded. Try: {txt}")
        else:
            st.error(f"{z} is forecast to be crowded and no alternative zone suits it. "
                     "Try a later slot (see the Typical patterns tab).")

    st.subheader(f"Why this forecast for {zone}?")
    row = snap[snap["zone"] == zone]
    contrib = explain_row(model, explainer, row, top=6)
    cdf = pd.DataFrame(contrib, columns=["feature", "value", "shap"])
    cdf["label"] = [f"{nice(f)} = {v:.2f}" for f, v in zip(cdf["feature"], cdf["value"])]
    rev = cdf.iloc[::-1]
    fig = go.Figure(go.Bar(x=rev["shap"], y=rev["label"], orientation="h",
                           marker_color=["#d62728" if v > 0 else "#1f77b4" for v in rev["shap"]]))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Effect on the forecast (occupancy rate, 0-1 scale)")
    st.plotly_chart(fig)
    base = float(np.ravel(explainer.expected_value)[0])
    top = cdf.iloc[0]
    st.write(f"The average forecast is {base:.2f}. The biggest driver here is **{nice(top['feature'])}** "
             f"({top['value']:.2f}), which {'raises' if top['shap'] > 0 else 'lowers'} the forecast "
             f"by {abs(top['shap']):.2f}. Red bars push the forecast up, blue bars push it down.")

# ------------------------------------------------------------------ day view
with tab_day:
    day = df[(df["zone"] == zone) & (df["timestamp"].dt.date == d)].sort_values("timestamp")
    st.subheader(f"{zone}: forecast vs actual on {d:%a %d %b %Y}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=day["timestamp"].dt.hour, y=day["rate"], name="Actual", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=day["timestamp"].dt.hour, y=day["pred"], name="Forecast", mode="lines+markers"))
    fig.add_hline(y=alert, line_dash="dot", annotation_text="alert threshold")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.05],
                      xaxis_title="Hour of day", yaxis_title="Occupancy rate")
    st.plotly_chart(fig)
    mae = (day["pred"] - day["rate"]).abs().mean()
    st.caption(f"Mean absolute error for this day: {mae:.3f} (occupancy rate). "
               f"A peak hour is one with occupancy of at least {PEAK_THRESHOLD:.0%}.")

# ------------------------------------------------------------------ typical patterns
with tab_pat:
    tp = load_train_profile()
    st.subheader(f"Typical occupancy in {zone} (training data)")
    z = tp[tp["zone"] == zone].pivot_table(index="dow", columns="hour", values="rate", aggfunc="mean")
    z.index = [DAYS[i] for i in z.index]
    fig = px.imshow(z, color_continuous_scale="YlOrRd", aspect="auto", zmin=0, zmax=1, text_auto=".0%",
                    labels=dict(x="Hour", y="Day", color="Occupancy"))
    st.plotly_chart(fig)
    wd = d.weekday()
    q = tp[(tp["zone"] == zone) & (tp["dow"] == wd)].groupby("hour")["rate"].mean().nsmallest(3)
    st.info(f"Typically quietest hours for {zone} on {DAYS[wd]}s: "
            + ", ".join(f"{hh}:00 ({v:.0%})" for hh, v in sorted(q.items())))
    st.markdown("**Frequent peak windows** (at least 70% full on 25% or more of training days)")
    pw = pd.read_csv(REPORTS / "peak_windows_h1.csv")
    sub = pw[pw["zone"] == zone]
    st.dataframe(sub if len(sub) else pd.DataFrame({"note": ["No frequent peak windows for this zone"]}),
                 hide_index=True)

# ------------------------------------------------------------------ resource demand
with tab_books:
    bt, avg = borrowing_forecast()
    st.subheader(f"Books borrowed, 7-day-ahead forecast for {d:%a %d %b %Y}")
    today = bt[bt["date"] == pd.Timestamp(d)].copy()

    def level(row):
        r = row["pred"] / avg[row["category"]]
        return "High" if r >= 1.25 else ("Low" if r <= 0.6 else "Normal")

    today["Demand level"] = today.apply(level, axis=1)
    out = today[["category", "pred", "books_borrowed", "Demand level"]].rename(
        columns={"category": "Category", "pred": "Forecast", "books_borrowed": "Actual (test data)"})
    out["Forecast"] = out["Forecast"].round(1)
    st.dataframe(out, hide_index=True)
    st.caption("Demand level compares the forecast with the category's long-run daily average "
               "(High: 25% or more above, Low: 40% or more below).")

    cat = st.selectbox("Category", sorted(bt["category"].unique()))
    c = bt[bt["category"] == cat].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c["date"], y=c["books_borrowed"], name="Actual"))
    fig.add_trace(go.Scatter(x=c["date"], y=c["pred"], name="Forecast"))
    fig.update_layout(yaxis_title="Books borrowed per day", xaxis_title="Date")
    st.plotly_chart(fig)
    err = (c["pred"] - c["books_borrowed"]).abs()
    st.caption(f"{cat}: MAE {err.mean():.2f} books/day, WAPE {err.sum() / c['books_borrowed'].sum():.1%} (test period).")

# ------------------------------------------------------------------ model & explainability
with tab_model:
    st.subheader("Test-set results (final model vs baselines)")
    st.dataframe(pd.read_csv(REPORTS / f"test_results_h{h}.csv").round(4), hide_index=True)
    st.subheader("Validation comparison incl. tuned models")
    st.dataframe(pd.read_csv(REPORTS / f"model_comparison_tuned_val_h{h}.csv").round(4), hide_index=True)

    st.subheader("What drives the forecasts? (SHAP)")
    st.write(SHAP_NOTES[h])
    c1, c2 = st.columns(2)
    c1.image(str(REPORTS / "figures" / f"shap_bar_h{h}.png"), caption="Global importance (mean |SHAP|)")
    c2.image(str(REPORTS / "figures" / f"shap_beeswarm_h{h}.png"), caption="Direction and size of effects")

    with st.expander("Limitations"):
        st.markdown(
            "- The data is **synthetic**, so accuracy is probably higher than on real sensor data.\n"
            "- Occupancy is capped at capacity, so demand above the seat count is invisible (censored data).\n"
            "- Random outage or event days cannot be predicted from the features.\n"
            "- SHAP shows what the model uses, not what causes crowding.\n"
            "- Alternative zones are a domain assumption; the Computer Lab has no substitute.")