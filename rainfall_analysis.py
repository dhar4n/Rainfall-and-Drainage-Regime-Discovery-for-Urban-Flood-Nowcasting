
import pandas as pd
import os

folder = "data-set/precipitation data"

files = [
    "0033900601.csv",
    "0033901401.csv",
    "0033901601.csv"
]

for file in files:

    path = os.path.join(folder, file)

    df = pd.read_csv(path)

    sensor = file.replace(".csv", "")

    df["time"] = pd.to_datetime(
        df["time"], utc=True
    )

    df[sensor] = pd.to_numeric(
        df[sensor], errors="coerce"
    )

    df = df.sort_values("time")

    print("\n" + "=" * 55)
    print("SENSOR:", sensor)
    print("=" * 55)

    print("Total observations:", len(df))
    print("Valid readings:", df[sensor].count())
    print("Zero readings:", (df[sensor] == 0).sum())
    print("Non-zero readings:", (df[sensor] > 0).sum())

    print("\nDate range:")
    print(df["time"].min())
    print(df["time"].max())

    print("\nTime interval in minutes:")
    print(
        df["time"].diff()
        .dt.total_seconds()
        .div(60)
        .describe()
    )

    print("\nFirst 10 observations:")
    print(df.head(10).to_string(index=False))

print("\nRAINFALL ANALYSIS COMPLETED")
