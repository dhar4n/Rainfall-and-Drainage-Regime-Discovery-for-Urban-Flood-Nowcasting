
import pandas as pd
import os

os.makedirs("results", exist_ok=True)

# ======================================
# 1. LOAD CLUSTERED DATASETS
# ======================================

hydraulic = pd.read_csv(
    "results/hydraulic_kmeans.csv"
)

rainfall = pd.read_csv(
    "results/rainfall_drainage_kmeans.csv"
)

# ======================================
# 2. HYDRAULIC CLUSTER ANALYSIS
# ======================================

hydraulic_features = [
    "chamber_water_level",
    "inflow_flow_rate",
    "inflow_flow_velocity",
    "inflow_water_level",
    "overflow_flow_rate",
    "overflow_water_level"
]

print("\n======================================")
print("HYDRAULIC CLUSTER INTERPRETATION")
print("======================================")

hydraulic_profile = hydraulic.groupby(
    "kmeans_cluster"
)[hydraulic_features].agg(
    ["mean", "median"]
).round(4)

print("\nCluster feature averages and medians:")
print(hydraulic_profile)

print("\nCluster sizes:")
print(hydraulic["kmeans_cluster"].value_counts().sort_index())

hydraulic_profile.to_csv(
    "results/hydraulic_cluster_profiles.csv"
)

# ======================================
# 3. RAINFALL + DRAINAGE ANALYSIS
# ======================================

combined_features = [
    "rain_gauge_1",
    "rain_gauge_2",
    "rain_gauge_3",
    "chamber_water_level",
    "inflow_flow_rate",
    "inflow_flow_velocity",
    "inflow_water_level",
    "overflow_flow_rate",
    "overflow_water_level"
]

print("\n======================================")
print("RAINFALL + DRAINAGE INTERPRETATION")
print("======================================")

rainfall_profile = rainfall.groupby(
    "kmeans_cluster"
)[combined_features].agg(
    ["mean", "median"]
).round(4)

print("\nCluster feature averages and medians:")
print(rainfall_profile)

print("\nCluster sizes:")
print(rainfall["kmeans_cluster"].value_counts().sort_index())

rainfall_profile.to_csv(
    "results/rainfall_cluster_profiles.csv"
)

# ======================================
# 4. FINAL SUMMARY
# ======================================

print("\n======================================")
print("CLUSTER INTERPRETATION COMPLETED")
print("======================================")

print("Generated files:")
print("1. hydraulic_cluster_profiles.csv")
print("2. rainfall_cluster_profiles.csv")
