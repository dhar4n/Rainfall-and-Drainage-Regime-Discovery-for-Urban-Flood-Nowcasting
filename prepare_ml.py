
import pandas as pd
import numpy as np

df = pd.read_csv(
    "processed/urban_flood_cleaned.csv"
)

# Convert timestamp
df["time"] = pd.to_datetime(df["time"], utc=True)

# Features for drainage regime discovery
hydraulic_features = [
    "chamber_water_level",
    "inflow_flow_rate",
    "inflow_flow_velocity",
    "inflow_water_level",
    "overflow_flow_rate",
    "overflow_water_level"
]

# Features for combined rainfall and drainage discovery
rain_features = [
    "rain_gauge_1",
    "rain_gauge_2",
    "rain_gauge_3"
]

# Remove remaining invalid hydraulic observations
hydraulic = df.dropna(
    subset=hydraulic_features
).copy()

# Only use actual rainfall observations
combined = df[
    df["rainfall_available"] == True
].dropna(
    subset=rain_features + hydraulic_features
).copy()

# Save both datasets
hydraulic.to_csv(
    "processed/hydraulic_ml.csv",
    index=False
)

combined.to_csv(
    "processed/rainfall_drainage_ml.csv",
    index=False
)

print("HYDRAULIC ML DATASET:", hydraulic.shape)
print("RAINFALL + DRAINAGE ML DATASET:", combined.shape)

print("\nBoth ML datasets saved successfully!")
