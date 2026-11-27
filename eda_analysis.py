
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project folders
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "processed"
OUTPUT = ROOT / "eda_results"
OUTPUT.mkdir(exist_ok=True)

# Load datasets
hyd = pd.read_csv(DATA / "hydraulic_ml.csv")
rain = pd.read_csv(DATA / "rainfall_drainage_ml.csv")

# Convert timestamps
for df in [hyd, rain]:
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df.dropna(subset=["time"], inplace=True)
    df.sort_values("time", inplace=True)

# Convert available measurement columns to numeric
for df in [hyd, rain]:
    for col in df.columns:
        if col != "time":
            df[col] = pd.to_numeric(df[col], errors="coerce")

def analyse_dataset(df, name):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Shape:", df.shape)
    print("Date range:", df["time"].min(), "to", df["time"].max())
    print("Duplicate timestamps:", df["time"].duplicated().sum())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDescriptive statistics:")
    stats = df.select_dtypes(include=np.number).describe().T
    print(stats)

    stats.to_csv(OUTPUT / f"{name}_statistics.csv")

    # Sampling interval and regularity
    gaps = df["time"].sort_values().diff().dropna().dt.total_seconds() / 60

    if len(gaps):
        print("\nTimestamp gap analysis (minutes):")
        print("Median gap:", gaps.median())
        print("Mean gap:", gaps.mean())
        print("Minimum gap:", gaps.min())
        print("Maximum gap:", gaps.max())
        print("Gaps over 5 minutes:", (gaps > 5).sum())
        print("Gaps over 30 minutes:", (gaps > 30).sum())

        pd.DataFrame({
            "gap_minutes": gaps
        }).to_csv(OUTPUT / f"{name}_timestamp_gaps.csv", index=False)

    # Missing-value chart
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing):
        plt.figure(figsize=(10, 5))
        missing.sort_values().plot(kind="bar")
        plt.title(f"{name}: Missing Values")
        plt.ylabel("Missing observations")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT / f"{name}_missing_values.png", dpi=200)
        plt.close()

    # Histograms
    numeric = df.select_dtypes(include=np.number)
    if not numeric.empty:
        numeric.hist(figsize=(14, 10), bins=30)
        plt.suptitle(f"{name}: Variable Distributions")
        plt.tight_layout()
        plt.savefig(OUTPUT / f"{name}_distributions.png", dpi=200)
        plt.close()

        # Correlation matrix
        corr = numeric.corr()
        corr.to_csv(OUTPUT / f"{name}_correlation.csv")

        plt.figure(figsize=(11, 8))
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(label="Correlation")
        plt.xticks(range(len(corr.columns)), corr.columns,
                   rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title(f"{name}: Correlation Matrix")
        plt.tight_layout()
        plt.savefig(OUTPUT / f"{name}_correlation.png", dpi=200)
        plt.close()

    print("\nAnalysis completed for", name)


# Run analysis
analyse_dataset(hyd, "hydraulic")
analyse_dataset(rain, "rainfall_drainage")


# -------------------------------------------------
# RAINFALL VS DRAINAGE RELATIONSHIP
# -------------------------------------------------

print("\nRAIN FALL AND DRAINAGE RELATIONSHIP")

relationships = [
    ("rainfall_mean", "inflow_water_level"),
    ("rainfall_mean", "overflow_flow_rate"),
    ("rainfall_mean", "overflow_water_level"),
    ("rainfall_mean", "inflow_flow_rate")
]

available = []

for x, y in relationships:
    if x in rain.columns and y in rain.columns:
        pair = rain[[x, y]].dropna()

        if len(pair) > 2:
            correlation = pair[x].corr(pair[y])
            print(f"{x} vs {y}:")
            print("Observations:", len(pair))
            print("Correlation:", round(correlation, 4))

            available.append({
                "rainfall_variable": x,
                "drainage_variable": y,
                "observations": len(pair),
                "correlation": correlation
            })

            plt.figure(figsize=(7, 5))
            plt.scatter(pair[x], pair[y], alpha=0.5)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f"{x} vs {y}")
            plt.tight_layout()
            plt.savefig(
                OUTPUT / f"relationship_{y}.png",
                dpi=200
            )
            plt.close()

if available:
    pd.DataFrame(available).to_csv(
        OUTPUT / "rainfall_drainage_relationships.csv",
        index=False
    )


# -------------------------------------------------
# FORECASTING FEASIBILITY CHECK
# -------------------------------------------------

print("\n" + "=" * 60)
print("SHORT-TERM NOWCASTING FEASIBILITY")
print("=" * 60)

for name, df in [
    ("Hydraulic", hyd),
    ("Rainfall + drainage", rain)
]:
    gaps = df["time"].sort_values().diff().dropna().dt.total_seconds() / 60

    print(f"\n{name} dataset")
    print("Observations:", len(df))
    print("Unique timestamps:", df["time"].nunique())

    if len(gaps):
        median_gap = gaps.median()
        regularity = (gaps <= median_gap * 1.5).mean() * 100

        print("Median sampling interval:", round(median_gap, 3), "minutes")
        print("Intervals within 1.5x median:", round(regularity, 2), "%")

        if regularity >= 80:
            print("Sampling regularity: PROMISING")
        else:
            print("Sampling regularity: IRREGULAR - needs investigation")

print("\nEDA COMPLETED.")
print("All reports and graphs saved in:", OUTPUT)