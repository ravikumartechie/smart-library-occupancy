"""Synthetic data generator for the Smart Library Occupancy project.

Produces:
  data/raw/library_occupancy.csv  - hourly occupancy per library zone
  data/raw/book_borrowing.csv     - daily book borrowing per category
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START, END = "2024-07-01", "2026-06-30"
OPEN_HOUR, CLOSE_HOUR = 8, 21          # hourly slots 08:00 .. 21:00
OUT_DIR = Path("data/raw")

# Each peak is (centre hour, width, height as a fraction of capacity)
ZONES = {
    "Reading Hall":     dict(capacity=300, base=0.10, exam_boost=1.45,
                             peaks=[(11.5, 2.0, 0.45), (17.5, 2.5, 0.55)]),
    "Silent Study":     dict(capacity=150, base=0.08, exam_boost=1.60,
                             peaks=[(14.0, 2.5, 0.40), (20.0, 2.5, 0.70)]),
    "Computer Lab":     dict(capacity=80,  base=0.10, exam_boost=1.15,
                             peaks=[(11.0, 2.0, 0.50), (15.0, 2.0, 0.55)]),
    "Group Discussion": dict(capacity=60,  base=0.05, exam_boost=1.20,
                             peaks=[(13.0, 2.0, 0.40), (16.5, 2.0, 0.45)]),
}

# Monday=0 ... Sunday=6
WEEKDAY_FACTOR = [0.95, 1.00, 1.00, 0.95, 0.85, 0.60, 0.40]

# Approximate dates, documented as assumptions in the README
HOLIDAYS = [
    "2024-08-15", "2024-10-02", "2024-10-12", "2024-11-01", "2024-12-25",
    "2025-01-26", "2025-03-14", "2025-08-15", "2025-10-02", "2025-10-20",
    "2025-12-25", "2026-01-26", "2026-03-04",
]
EXAM_WINDOWS = [
    ("2024-09-23", "2024-09-28"), ("2024-11-25", "2024-12-14"),
    ("2025-02-24", "2025-03-01"), ("2025-04-28", "2025-05-17"),
    ("2025-09-22", "2025-09-27"), ("2025-11-24", "2025-12-13"),
    ("2026-02-23", "2026-02-28"), ("2026-04-27", "2026-05-16"),
]
VACATIONS = [
    ("2024-07-01", "2024-07-20"), ("2024-12-15", "2025-01-05"),
    ("2025-05-18", "2025-07-20"), ("2025-12-14", "2026-01-04"),
    ("2026-05-17", "2026-06-30"),
]
CATEGORIES = {  # average books borrowed per day
    "Computer Science": 60, "Engineering": 80, "Management": 40,
    "Science": 50, "Literature": 25, "Competitive Exams": 45,
}


def build_calendar() -> pd.DataFrame:
    cal = pd.DataFrame({"date": pd.date_range(START, END, freq="D")})
    cal["day_of_week"] = cal["date"].dt.dayofweek
    cal["is_weekend"] = (cal["day_of_week"] >= 5).astype(int)
    cal["is_holiday"] = cal["date"].isin(pd.to_datetime(HOLIDAYS)).astype(int)
    cal["is_exam"] = 0
    cal["is_vacation"] = 0
    for s, e in EXAM_WINDOWS:
        cal.loc[cal["date"].between(pd.Timestamp(s), pd.Timestamp(e)), "is_exam"] = 1
    for s, e in VACATIONS:
        cal.loc[cal["date"].between(pd.Timestamp(s), pd.Timestamp(e)), "is_vacation"] = 1

    starts = pd.to_datetime([s for s, _ in EXAM_WINDOWS])

    def days_to_next_exam(d):
        future = starts[starts >= d]
        return (future[0] - d).days if len(future) else 60

    cal["days_to_exam"] = cal["date"].apply(days_to_next_exam).clip(upper=60)
    return cal


def hour_profile(hour: float, cfg: dict) -> float:
    """Typical share of capacity in use at a given hour (sum of bumps)."""
    p = cfg["base"]
    for centre, width, height in cfg["peaks"]:
        p += height * np.exp(-((hour - centre) ** 2) / (2 * width ** 2))
    return p


def day_multiplier(row, cfg: dict) -> float:
    m = WEEKDAY_FACTOR[row.day_of_week]
    if row.is_holiday:
        m *= 0.30
    if row.is_vacation:
        m *= 0.30
    if row.is_exam:
        m *= cfg["exam_boost"]
    elif row.days_to_exam <= 7:                     # pre-exam ramp-up
        m *= 1 + 0.05 * (7 - row.days_to_exam)
    return m


def generate_occupancy(cal: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    # hidden day-level events (outage / big event); NOT saved as a column
    events = rng.choice([1.0, 0.5, 1.3], size=len(cal), p=[0.96, 0.02, 0.02])
    rows = []
    for zone, cfg in ZONES.items():
        noise = 0.0
        for i, row in enumerate(cal.itertuples(index=False)):
            base_mult = day_multiplier(row, cfg) * events[i]
            for h in range(OPEN_HOUR, CLOSE_HOUR + 1):
                noise = 0.7 * noise + rng.normal(0, 0.07)   # AR(1) noise
                util = min(hour_profile(h, cfg) * base_mult * np.exp(noise), 1.0)
                rows.append({
                    "timestamp": row.date + pd.Timedelta(hours=h),
                    "zone": zone,
                    "capacity": cfg["capacity"],
                    "occupancy": int(round(util * cfg["capacity"])),
                    "is_weekend": row.is_weekend,
                    "is_holiday": row.is_holiday,
                    "is_exam": row.is_exam,
                    "is_vacation": row.is_vacation,
                    "days_to_exam": row.days_to_exam,
                })
    return pd.DataFrame(rows)


def add_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df["occupancy"] = df["occupancy"].astype(float)
    n = len(df)
    missing = rng.choice(n, size=int(0.005 * n), replace=False)   # sensor dropouts
    glitch = rng.choice(n, size=int(0.001 * n), replace=False)    # impossible readings
    df.loc[glitch, "occupancy"] = (df.loc[glitch, "capacity"] * 1.8).round()
    df.loc[missing, "occupancy"] = np.nan
    return df


def generate_borrowing(cal: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    first_day = cal["date"].iloc[0]
    for cat, avg in CATEGORIES.items():
        for row in cal.itertuples(index=False):
            lam = avg * WEEKDAY_FACTOR[row.day_of_week]
            if row.is_holiday:
                lam *= 0.20
            if row.is_vacation:
                lam *= 0.35
            if row.is_exam:
                lam *= 1.40
            elif row.days_to_exam <= 7:
                lam *= 1 + 0.04 * (7 - row.days_to_exam)
            if cat == "Competitive Exams":              # slow upward trend
                lam *= 1 + 0.0005 * (row.date - first_day).days
            rows.append({"date": row.date, "category": cat,
                         "books_borrowed": int(rng.poisson(lam))})
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cal = build_calendar()
    occ = add_data_quality_issues(generate_occupancy(cal, rng), rng)
    occ.to_csv(OUT_DIR / "library_occupancy.csv", index=False)
    generate_borrowing(cal, rng).to_csv(OUT_DIR / "book_borrowing.csv", index=False)
    print(f"occupancy rows: {len(occ)} | missing: {occ['occupancy'].isna().sum()}")
    print("saved to", OUT_DIR.resolve())


if __name__ == "__main__":
    main()