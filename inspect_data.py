
import pandas as pd
import os

BASE_PATH = "data-set"

folders = [
    "precipitation data",
    "hydraulic data"
]

print("\nURBAN FLOOD NOWCASTING")
print("=" * 65)

for folder in folders:

    folder_path = os.path.join(BASE_PATH, folder)

    print(f"\n\nFOLDER: {folder}")
    print("=" * 65)

    for file in os.listdir(folder_path):

        if file.endswith(".csv"):

            file_path = os.path.join(folder_path, file)

            try:
                df = pd.read_csv(
                    file_path,
                    sep=None,
                    engine="python",
                    nrows=5
                )

                full_df = pd.read_csv(
                    file_path,
                    sep=None,
                    engine="python"
                )

                print("\nFILE:", file)
                print("ROWS:", len(full_df))
                print("COLUMNS:", list(full_df.columns))

                print("\nDATA TYPES:")
                print(full_df.dtypes)

                print("\nSAMPLE DATA:")
                print(df.to_string(index=False))

                print("\nMISSING VALUES:")
                print(full_df.isnull().sum())

                print("-" * 65)

            except Exception as e:
                print(f"Error reading {file}: {e}")

print("\nDATASET INSPECTION COMPLETED!")
