
import pandas as pd

df = pd.read_csv(
    "processed/combined_before_cleaning.csv"
)

print("\nTOTAL ROWS:", len(df))

print("\nNON-MISSING VALUES:")
print(df.count())

print("\nMISSING PERCENTAGE:")
print((df.isnull().mean() * 100).round(2))

print("\nDATE RANGE:")
print(df["time"].min())
print(df["time"].max())

print("\nROWS WITH ALL FEATURES AVAILABLE:")
print(df.dropna().shape[0])
