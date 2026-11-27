
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)

os.makedirs("final_results", exist_ok=True)

# Load original hydraulic dataset
data = pd.read_csv("processed/hydraulic_ml.csv")
data["time"] = pd.to_datetime(data["time"], utc=True)
data = data.sort_values("time").reset_index(drop=True)

# Define high overflow using the training-period 95th percentile
split = int(len(data) * 0.80)
threshold = data.iloc[:split]["overflow_flow_rate"].quantile(0.95)

# Load 5-minute predictions
df = pd.read_csv("forecast_results/predictions_5min.csv")

actual = df["actual_overflow"]
predicted = df["predicted_overflow"]
baseline = df["baseline_prediction"]

# High-overflow classification
actual_event = actual >= threshold
predicted_event = predicted >= threshold
baseline_event = baseline >= threshold

def calculate_metrics(y_true, y_pred, name):
    return {
        "Method": name,
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        )
    }

results = [
    calculate_metrics(
        actual_event, predicted_event, "Random Forest"
    ),
    calculate_metrics(
        actual_event, baseline_event, "Baseline"
    )
]

result_df = pd.DataFrame(results)
result_df.to_csv(
    "final_results/high_overflow_metrics.csv",
    index=False
)

# Confusion matrix
cm = confusion_matrix(
    actual_event, predicted_event, labels=[False, True]
)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("5-Minute High-Overflow Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Normal", "High overflow"])
plt.yticks([0, 1], ["Normal", "High overflow"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]),
                 ha="center", va="center")

plt.tight_layout()
plt.savefig("final_results/high_overflow_confusion_matrix.png")
plt.close()

# Actual vs predicted around the 20 largest actual events
largest = df.nlargest(20, "actual_overflow").copy()
largest = largest.sort_values("time")

plt.figure(figsize=(12, 5))
plt.plot(
    largest["time"], largest["actual_overflow"],
    marker="o", label="Actual"
)
plt.plot(
    largest["time"], largest["predicted_overflow"],
    marker="o", label="Predicted"
)
plt.title("Actual vs Predicted: 20 Largest Overflow Events")
plt.xlabel("Time")
plt.ylabel("Overflow Flow Rate")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.savefig("final_results/extreme_event_predictions.png")
plt.close()

# Save final summary
with open("final_results/final_summary.txt", "w") as f:
    f.write("FINAL URBAN FLOOD NOWCASTING EVALUATION\n")
    f.write("=" * 45 + "\n")
    f.write(f"High-overflow threshold (training 95th percentile): {threshold:.4f}\n")
    f.write(f"Test observations: {len(df)}\n")
    f.write(f"Actual high-overflow observations: {actual_event.sum()}\n")
    f.write(f"Predicted high-overflow observations: {predicted_event.sum()}\n\n")
    f.write(result_df.to_string(index=False))
    f.write("\n\n5-minute regression metrics\n")
    f.write(f"MAE: {mean_absolute_error(actual, predicted):.4f}\n")
    f.write(
        f"RMSE: {np.sqrt(mean_squared_error(actual, predicted)):.4f}\n"
    )
    f.write("\nInterpretation:\n")
    f.write(
        "High-overflow detection is evaluated using a statistical "
        "training-period threshold. It is not a validated flood-warning threshold.\n"
    )

print("HIGH-OVERFLOW THRESHOLD:", round(threshold, 4))
print("\nHIGH-OVERFLOW DETECTION RESULTS")
print(result_df.to_string(index=False))
print("\nConfusion matrix:")
print(cm)
print("\nFinal evaluation saved in final_results/")
