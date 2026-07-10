"""
Feature Engineering for Electricity Theft Detection
=====================================================
Raw half-hourly readings are useless to a classifier directly (too granular,
different scale per household). We aggregate each METER into a feature
vector per WEEK, then summarize into per-meter features. The features below
are specifically designed to expose the 4 theft signatures we injected:

- night_to_day_ratio      -> low if bypass/reversal happening at night
- zero_reading_pct        -> high for METER_BYPASS (near-zero windows)
- coefficient_of_variation-> high for ILLEGAL_HOOKING (random spikes)
- flat_line_pct           -> high for METER_REVERSAL (frozen readings)
- weekday_weekend_ratio   -> deviates for tampered meters
- avg_daily_consumption   -> general usage level
- rolling_std_of_daily_sum-> instability in day-to-day totals
- morning_evening_peak_ratio -> normal households have a clear double-peak;
                                tampered ones flatten or distort this
"""

import pandas as pd
import numpy as np

print("Loading raw data...")
df = pd.read_parquet("./smart_meter_data.parquet")
df["hour"] = df["timestamp"].dt.hour
df["date"] = df["timestamp"].dt.date
df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5

print("Aggregating per-meter features...")

feature_rows = []

for meter_id, g in df.groupby("meter_id"):
    label = g["label"].iloc[0]
    theft_type = g["theft_type"].iloc[0]

    night_mask = g["hour"].isin([0, 1, 2, 3, 4])
    day_mask = g["hour"].isin([9, 10, 11, 12, 13, 14, 15, 16])
    morning_mask = g["hour"].isin([7, 8, 9])
    evening_mask = g["hour"].isin([19, 20, 21])

    night_avg = g.loc[night_mask, "consumption_kwh"].mean()
    day_avg = g.loc[day_mask, "consumption_kwh"].mean()
    morning_avg = g.loc[morning_mask, "consumption_kwh"].mean()
    evening_avg = g.loc[evening_mask, "consumption_kwh"].mean()

    daily_totals = g.groupby("date")["consumption_kwh"].sum()
    weekend_avg = g.loc[g["is_weekend"], "consumption_kwh"].mean()
    weekday_avg = g.loc[~g["is_weekend"], "consumption_kwh"].mean()

    zero_pct = (g["consumption_kwh"] < 0.05).mean()

    # flat-line detection: consecutive identical readings within a day
    flat_count = 0
    for _, day_group in g.groupby("date"):
        vals = day_group["consumption_kwh"].values
        flat_count += (np.diff(vals) == 0).sum()
    flat_pct = flat_count / len(g)

    features = {
        "meter_id": meter_id,
        "avg_daily_consumption": daily_totals.mean(),
        "std_daily_consumption": daily_totals.std(),
        "cv_consumption": g["consumption_kwh"].std() / (g["consumption_kwh"].mean() + 1e-6),
        "night_to_day_ratio": night_avg / (day_avg + 1e-6),
        "morning_evening_peak_ratio": morning_avg / (evening_avg + 1e-6),
        "weekday_weekend_ratio": weekday_avg / (weekend_avg + 1e-6),
        "zero_reading_pct": zero_pct,
        "flat_line_pct": flat_pct,
        "rolling_std_of_daily_sum": daily_totals.rolling(7).mean().std(),
        "max_single_reading": g["consumption_kwh"].max(),
        "min_single_reading": g["consumption_kwh"].min(),
        "label": label,
        "theft_type": theft_type,
    }
    feature_rows.append(features)

feat_df = pd.DataFrame(feature_rows)
feat_df.to_csv("./meter_features.csv", index=False)
print(f"\nFeature dataset created: {feat_df.shape}")
print(feat_df.groupby("theft_type")[["night_to_day_ratio", "zero_reading_pct", "flat_line_pct", "cv_consumption"]].mean())
