
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

INPUT_DIR = "forecast_results"
OUTPUT_DIR = "forecast_evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

horizons = ["5min", "15min", "30min"]
summary = []

for horizon in horizons:

    path = f"{INPUT_DIR}/predictions_{horizon}.csv"

    if not os.path.exists(path):
        print(f"Missing file: {path}")
        continue

    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    actual = df["actual_overflow"]
    predicted = df["predicted_overflow"]
    baseline = df["baseline_prediction"]

    errors = actual - predicted

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    baseline_mae = mean_absolute_error(actual, baseline)
    baseline_rmse = np.sqrt(
        mean_squared_error(actual, baseline)
    )

    summary.append({
        "Horizon": horizon,
        "MAE": mae,
        "RMSE": rmse,
        "Baseline_MAE": baseline_mae,
        "Baseline_RMSE": baseline_rmse,
        "Mean_Error": errors.mean(),
        "Median_Absolute_Error": errors.abs().median()
    })

    # Actual vs predicted: chronological sample
    plot_df = df.head(min(1000, len(df)))

    plt.figure(figsize=(14, 5))
    plt.plot(
        plot_df["time"],
        plot_df["actual_overflow"],
        label="Actual",
        linewidth=1
    )
    plt.plot(
        plot_df["time"],
        plot_df["predicted_overflow"],
        label="Predicted",
        linewidth=1
    )
    plt.title(f"Actual vs Predicted Overflow - {horizon}")
    plt.xlabel("Time")
    plt.ylabel("Overflow Flow Rate")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/actual_vs_predicted_{horizon}.png"
    )
    plt.close()

    # Residual distribution
    plt.figure(figsize=(9, 5))
    plt.hist(errors, bins=50)
    plt.title(f"Forecast Residual Distribution - {horizon}")
    plt.xlabel("Actual - Predicted")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/residuals_{horizon}.png"
    )
    plt.close()

    # Absolute error vs actual value
    plt.figure(figsize=(9, 5))
    plt.scatter(
        actual,
        errors.abs(),
        alpha=0.3,
        s=8
    )
    plt.title(f"Absolute Error vs Actual Overflow - {horizon}")
    plt.xlabel("Actual Overflow Flow Rate")
    plt.ylabel("Absolute Prediction Error")
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/error_vs_actual_{horizon}.png"
    )
    plt.close()

    # Inspect largest prediction errors
    worst = df.copy()
    worst["absolute_error"] = errors.abs()
    worst["signed_error"] = errors

    worst.sort_values(
        "absolute_error",
        ascending=False
    ).head(20).to_csv(
        f"{OUTPUT_DIR}/largest_errors_{horizon}.csv",
        index=False
    )

    print(f"\n{horizon} evaluation")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Baseline MAE: {baseline_mae:.4f}")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Median absolute error: {errors.abs().median():.4f}")

if summary:
    result = pd.DataFrame(summary)
    result.to_csv(
        f"{OUTPUT_DIR}/evaluation_summary.csv",
        index=False
    )

    print("\nFINAL EVALUATION")
    print(result.to_string(index=False))

    print("\nEvaluation files saved in:", OUTPUT_DIR)
else:
    print("\nNo prediction files found. Run forecasting.py first.")
