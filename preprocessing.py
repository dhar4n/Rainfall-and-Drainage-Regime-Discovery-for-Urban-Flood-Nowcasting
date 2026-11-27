
import os
import pandas as pd
import numpy as np

BASE = "data-set"
OUTPUT = "processed"

os.makedirs(OUTPUT, exist_ok=True)

# -----------------------------------
# 1. PROCESS RAINFALL SENSOR DATA
# -----------------------------------

rain_files = {
    "0033900601.csv": "rain_gauge_1",
    "0033901401.csv": "rain_gauge_2",
    "0033901601.csv": "rain_gauge_3"
}

rain_frames = []

for filename, sensor_name in rain_files.items():

    path = os.path.join(
        BASE, "precipitation data", filename
    )

    df = pd.read_csv(path)

    df["time"] = pd.to_datetime(
        df["time"], utc=True
    )

    df[sensor_name] = pd.to_numeric(
        df[filename.replace(".csv", "")],
        errors="coerce"
    )

    df = df.dropna(subset=["time", sensor_name])
    df = df.sort_values("time")

    # Remove duplicate timestamps
    df = df.groupby("time")[sensor_name].mean()

    # Aggregate recorded rainfall events into 5-minute bins.
    # Missing bins remain NaN, NOT zero.
    rain = df.resample("5min").sum(min_count=1)

    rain_frames.append(rain.rename(sensor_name))

rainfall = pd.concat(rain_frames, axis=1)

print("Rainfall processed:", rainfall.shape)


# -----------------------------------
# 2. PROCESS HYDRAULIC DATA
# -----------------------------------

def load_hydraulic(filename, prefix):

    path = os.path.join(
        BASE, "hydraulic data", filename
    )

    df = pd.read_csv(path)

    df["time"] = pd.to_datetime(
        df["time"], utc=True
    )

    df = df.dropna(subset=["time"])
    df = df.set_index("time").sort_index()

    df = df.apply(pd.to_numeric, errors="coerce")

    df = df.groupby(level=0).mean()

    df.columns = [
        prefix + "_" + col for col in df.columns
    ]

    # Average measurements in 5-minute intervals
    return df.resample("5min").mean()


chamber = load_hydraulic(
    "chamber.csv", "chamber"
)

inflow = load_hydraulic(
    "inflow-analogue.csv", "inflow"
)

overflow = load_hydraulic(
    "overflow.csv", "overflow"
)


# -----------------------------------
# 3. COMBINE HYDRAULIC DATA
# -----------------------------------

hydraulic = pd.concat(
    [chamber, inflow, overflow],
    axis=1
)

# -----------------------------------
# 4. ALIGN RAINFALL AND HYDRAULIC
# -----------------------------------

combined = rainfall.join(
    hydraulic,
    how="inner"
)

print("\nCombined shape:", combined.shape)

# -----------------------------------
# 5. HANDLE MISSING DATA
# -----------------------------------


# -----------------------------------
# 5. HANDLE MISSING DATA - IMPROVED
# -----------------------------------

# Interpolate only short gaps in hydraulic readings
hydraulic_cols = [
    "chamber_water_level",
    "inflow_flow_rate",
    "inflow_flow_velocity",
    "inflow_water_level",
    "overflow_flow_rate",
    "overflow_water_level"
]

hydraulic[hydraulic_cols] = hydraulic[
    hydraulic_cols
].interpolate(
    method="time",
    limit=2,
    limit_area="inside"
)

# Recombine rainfall and hydraulic measurements
combined = rainfall.join(
    hydraulic,
    how="inner"
)

# Keep records with complete core hydraulic readings
required_hydraulic = [
    "chamber_water_level",
    "inflow_flow_rate",
    "inflow_water_level",
    "overflow_flow_rate",
    "overflow_water_level"
]

cleaned = combined.dropna(
    subset=required_hydraulic
).copy()

# Calculate mean rainfall only when at least
# one sensor has an actual observation
rain_cols = [
    "rain_gauge_1",
    "rain_gauge_2",
    "rain_gauge_3"
]

cleaned["rainfall_mean"] = cleaned[
    rain_cols
].mean(axis=1)

# Keep a record of how many sensors reported rainfall
cleaned["rain_sensors_available"] = (
    cleaned[rain_cols].notna().sum(axis=1)
)

# Mark rainfall observations as incomplete
# when no sensor has reported a measurement
cleaned["rainfall_available"] = (
    cleaned["rain_sensors_available"] > 0
)

# Remove physically impossible negative rainfall
for col in rain_cols + ["rainfall_mean"]:
    cleaned[col] = cleaned[col].clip(lower=0)

cleaned = cleaned.replace(
    [np.inf, -np.inf], np.nan
)

print("\nHydraulic records retained:", len(cleaned))
print("\nRainfall availability:")
print(cleaned["rainfall_available"].value_counts())

print("\nMissing values:")
print(cleaned.isnull().sum())

# -----------------------------------
# 6. SAVE FINAL DATASET
# -----------------------------------

cleaned = cleaned.reset_index()

cleaned.to_csv(
    os.path.join(
        OUTPUT, "urban_flood_cleaned.csv"
    ),
    index=False
)

print("\n========== PREPROCESSING RESULT ==========")
print("Final records:", len(cleaned))
print("Final columns:", list(cleaned.columns))
print("\nMissing values:")
print(cleaned.isnull().sum())
print("\nSaved successfully!")
