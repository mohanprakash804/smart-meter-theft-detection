"""
Smart Meter Data Generator
==========================
Most student projects just download a ready-made Kaggle dataset (SGCC) and
plug it into a model. Here we instead SIMULATE realistic household smart
meter data ourselves — this is closer to what a real energy-sector data
scientist would need to do (since real utility data is private/confidential).

We simulate 365 days of HALF-HOURLY readings (48 readings/day) for:
- N_HONEST honest households
- N_THEFT households committing one of 4 real-world theft types

Theft types modeled (these are the actual categories power companies track):
1. METER_BYPASS      -> Illegal wire routes AROUND the meter for part of the day
                         (consumption suddenly drops to near-zero for hours,
                          then returns to normal — classic bypass signature)
2. PARTIAL_TAMPER    -> Meter under-reports a FIXED percentage of real usage
                         (e.g. only 60% of true consumption is recorded)
3. ILLEGAL_HOOKING   -> A neighbour's load is hooked onto this meter's line
                         at random hours (consumption spikes appear that
                         don't match the household's own usage profile)
4. METER_REVERSAL    -> Meter tampered to run backward / freeze during
                         specific hours (usually night) -> flat-line pattern
                         at unusual hours
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_HONEST = 300
N_THEFT_PER_TYPE = 45          # 45 x 4 = 180 theft households
DAYS = 365
READINGS_PER_DAY = 48          # every 30 minutes
START_DATE = datetime(2025, 1, 1)


def base_daily_profile():
    """
    Typical Indian household load curve (kW) across 48 half-hour slots:
    low at night, morning peak (6-9am), afternoon dip, evening peak (6-10pm).
    """
    hours = np.arange(0, 24, 0.5)
    morning_peak = 1.2 * np.exp(-((hours - 7.5) ** 2) / (2 * 1.2 ** 2))
    evening_peak = 1.8 * np.exp(-((hours - 20) ** 2) / (2 * 1.5 ** 2))
    base_load = 0.25
    profile = base_load + morning_peak + evening_peak
    return profile


def generate_household_year(household_id, label, theft_type=None, avg_scale=1.0):
    profile = base_daily_profile() * avg_scale
    records = []

    # season multiplier: higher AC/fan usage in summer (Apr-Jun), heater in winter nights
    for day in range(DAYS):
        date = START_DATE + timedelta(days=day)
        month = date.month
        is_weekend = date.weekday() >= 5

        season_mult = 1.0
        if month in [4, 5, 6]:          # summer
            season_mult = 1.35
        elif month in [12, 1, 2]:       # winter
            season_mult = 1.15

        weekend_mult = 1.1 if is_weekend else 1.0
        noise = np.random.normal(1.0, 0.08, size=READINGS_PER_DAY)

        day_readings = profile * season_mult * weekend_mult * noise
        day_readings = np.clip(day_readings, 0.02, None)

        # ---- inject theft signatures (kept SUBTLE/realistic on purpose --
        # real theft doesn't scream "I'm theft!" in the data, that's why
        # detection is a hard problem worth solving with ML) ----
        if theft_type == "METER_BYPASS":
            # only ~12% of days show a short bypass window, and it's a
            # partial (not near-zero) drop, so it blends with normal variance
            if np.random.rand() < 0.12:
                start_slot = np.random.randint(0, READINGS_PER_DAY - 8)
                dur = np.random.randint(4, 8)
                day_readings[start_slot:start_slot + dur] *= np.random.uniform(0.35, 0.55)

        elif theft_type == "PARTIAL_TAMPER":
            # subtler under-reporting, closer to normal household variance
            under_report_factor = np.random.uniform(0.75, 0.88)
            day_readings *= under_report_factor

        elif theft_type == "ILLEGAL_HOOKING":
            # smaller, less frequent spikes -- easy to confuse with a
            # guest visiting / appliance use
            if np.random.rand() < 0.5:
                n_spikes = np.random.randint(1, 3)
                spike_slots = np.random.choice(READINGS_PER_DAY, n_spikes, replace=False)
                day_readings[spike_slots] += np.random.uniform(0.5, 1.3, n_spikes)

        elif theft_type == "METER_REVERSAL":
            # shorter freeze window, and only on some days (intermittent tampering)
            if np.random.rand() < 0.6:
                day_readings[0:4] = day_readings[0]  # freeze midnight-2am only

        for slot in range(READINGS_PER_DAY):
            ts = date + timedelta(minutes=30 * slot)
            records.append({
                "meter_id": household_id,
                "timestamp": ts,
                "consumption_kwh": round(day_readings[slot], 4),
                "label": label,          # 0 = honest, 1 = theft
                "theft_type": theft_type if theft_type else "NONE",
            })

    return records


def main():
    all_records = []
    hid = 1

    print("Generating honest households...")
    for _ in range(N_HONEST):
        scale = np.random.uniform(0.6, 1.6)  # household size variation
        all_records.extend(generate_household_year(f"M{hid:04d}", label=0, avg_scale=scale))
        hid += 1

    for t_type in ["METER_BYPASS", "PARTIAL_TAMPER", "ILLEGAL_HOOKING", "METER_REVERSAL"]:
        print(f"Generating theft households: {t_type}...")
        for _ in range(N_THEFT_PER_TYPE):
            scale = np.random.uniform(0.6, 1.6)
            all_records.extend(
                generate_household_year(f"M{hid:04d}", label=1, theft_type=t_type, avg_scale=scale)
            )
            hid += 1

    df = pd.DataFrame(all_records)
    df.to_parquet("./smart_meter_data.parquet", index=False)
    print(f"\nDone. Total rows: {len(df):,}")
    print(f"Total meters: {df['meter_id'].nunique()}")
    print(df.groupby('theft_type')['meter_id'].nunique())


if __name__ == "__main__":
    main()
