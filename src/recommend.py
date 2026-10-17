"""Peak-hour detection and alternative-zone recommender.

Run:  python src/recommend.py        (uses test_predictions_h1.csv)
      python src/recommend.py 4
"""
import sys
from pathlib import Path

import pandas as pd

from features import PEAK_THRESHOLD

PROCESSED, REPORTS = Path("data/processed"), Path("reports")
ALERT = {1: 0.65, 4: 0.60}        # chosen on validation in evaluate.py
FREE_BELOW = 0.40
# Domain assumption: which zones can sensibly substitute for each other
ALTERNATIVES = {
    "Reading Hall": ["Silent Study", "Group Discussion"],
    "Silent Study": ["Reading Hall", "Group Discussion"],
    "Group Discussion": ["Reading Hall"],
    "Computer Lab": [],
}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def status(rate: float, alert: float) -> str:
    if rate >= alert:
        return "Crowded"
    return "Moderate" if rate >= FREE_BELOW else "Free"


def recommend(snapshot: pd.DataFrame, zone: str, alert: float) -> list:
    """snapshot: one row per zone at one timestamp (columns zone, capacity, pred)."""
    snap = snapshot.set_index("zone")
    if snap.loc[zone, "pred"] < alert:
        return []
    options = []
    for alt in ALTERNATIVES.get(zone, []):
        if alt in snap.index and snap.loc[alt, "pred"] < alert:
            r = float(snap.loc[alt, "pred"])
            options.append({"zone": alt, "pred_rate": round(r, 3),
                            "est_free_seats": int(round((1 - r) * snap.loc[alt, "capacity"]))})
    return sorted(options, key=lambda o: o["pred_rate"])


def ranges(hours) -> str:
    hours, out, start, prev = sorted(hours), [], None, None
    for h in hours:
        if start is None:
            start = prev = h
        elif h == prev + 1:
            prev = h
        else:
            out.append((start, prev)); start = prev = h
    if start is not None:
        out.append((start, prev))
    return ", ".join(f"{a}:00-{b + 1}:00" for a, b in out) or "-"


def peak_windows(horizon: int = 1) -> pd.DataFrame:
    train = pd.read_csv(PROCESSED / f"train_h{horizon}.csv")
    share = train.groupby(["zone", "dow", "hour"])["is_peak"].mean().reset_index()
    frequent = share[share["is_peak"] >= 0.25]
    rows = [{"zone": z, "day": DAYS[dow], "peak_hours": ranges(g["hour"])}
            for (z, dow), g in frequent.groupby(["zone", "dow"])]
    return pd.DataFrame(rows)


def evaluate_recommender(test: pd.DataFrame, alert: float):
    actual = test.pivot(index="timestamp", columns="zone", values="rate")
    n_alerts = n_alt = n_ok = 0
    for ts, snap in test.groupby("timestamp"):
        snap = snap[["zone", "capacity", "pred"]]
        for zone in snap["zone"]:
            if snap.loc[snap.zone == zone, "pred"].iloc[0] < alert:
                continue
            n_alerts += 1
            opts = recommend(snap, zone, alert)
            if opts:
                n_alt += 1
                n_ok += actual.loc[ts, opts[0]["zone"]] < PEAK_THRESHOLD
    print(f"Alerts raised (zone forecast crowded):   {n_alerts}")
    print(f"  ...with an alternative suggested:      {n_alt} ({n_alt / max(n_alerts, 1):.1%})")
    print(f"  ...suggestion actually < {PEAK_THRESHOLD:.0%} full: {n_ok} ({n_ok / max(n_alt, 1):.1%})")


def main(horizon: int = 1):
    alert = ALERT[horizon]
    pw = peak_windows(horizon)
    pw.to_csv(REPORTS / f"peak_windows_h{horizon}.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 100)
    print("Frequent peak windows (peak on >= 25% of training days):\n", pw.to_string(index=False))

    test = pd.read_csv(REPORTS / f"test_predictions_h{horizon}.csv", parse_dates=["timestamp"])
    print(f"\nRecommender evaluation on TEST (horizon {horizon}h, alert threshold {alert})")
    evaluate_recommender(test, alert)

    crowded = test[test["pred"] >= alert]
    if len(crowded):
        ts = crowded["timestamp"].iloc[0]
        snap = test[test["timestamp"] == ts][["zone", "capacity", "pred"]]
        zone = crowded["zone"].iloc[0]
        print(f"\nExample at {ts}:\n", snap.round(3).to_string(index=False))
        print(f"Recommendation for {zone}:", recommend(snap, zone, alert) or "no alternative; try a later slot")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)