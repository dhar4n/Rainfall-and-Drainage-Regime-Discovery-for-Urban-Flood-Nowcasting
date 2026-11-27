
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

os.makedirs("results", exist_ok=True)

# ==========================================
# 1. LOAD DATASETS
# ==========================================

hydraulic = pd.read_csv(
    "processed/hydraulic_ml.csv"
)

combined = pd.read_csv(
    "processed/rainfall_drainage_ml.csv"
)

hydraulic_features = [
    "chamber_water_level",
    "inflow_flow_rate",
    "inflow_flow_velocity",
    "inflow_water_level",
    "overflow_flow_rate",
    "overflow_water_level"
]

rainfall_features = [
    "rain_gauge_1",
    "rain_gauge_2",
    "rain_gauge_3"
]

combined_features = (
    rainfall_features + hydraulic_features
)

print("Hydraulic dataset:", hydraulic.shape)
print("Rainfall dataset:", combined.shape)


# ==========================================
# 2. K-MEANS CLUSTERING FUNCTION
# ==========================================

def run_kmeans(df, features, name):

    X = df[features].copy()

    X = X.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    print("\n", "=" * 50)
    print(name, "K-MEANS RESULTS")
    print("=" * 50)

    print("Records:", len(X))
    print("Clusters:", np.unique(labels))
    print("Cluster distribution:")
    print(pd.Series(labels).value_counts().sort_index())

    # Silhouette score using a sample
    sample_size = min(5000, len(X))

    score = silhouette_score(
        X_scaled,
        labels,
        sample_size=sample_size,
        random_state=42
    )

    print("Silhouette score:", round(score, 4))

    # Save labelled dataset
    result = df.loc[X.index].copy()
    result["kmeans_cluster"] = labels

    result.to_csv(
        f"results/{name}_kmeans.csv",
        index=False
    )

    # PCA visualization
    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    # Plot a manageable sample
    rng = np.random.RandomState(42)

    plot_size = min(10000, len(X_pca))

    indices = rng.choice(
        len(X_pca),
        size=plot_size,
        replace=False
    )

    plt.figure(figsize=(9, 6))

    scatter = plt.scatter(
        X_pca[indices, 0],
        X_pca[indices, 1],
        c=labels[indices],
        cmap="viridis",
        s=8,
        alpha=0.6
    )

    plt.colorbar(scatter, label="Cluster")

    plt.title(f"{name}: K-Means Clustering with PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.tight_layout()

    plt.savefig(
        f"results/{name}_kmeans_pca.png",
        dpi=150
    )

    plt.close()

    print("K-Means results saved.")

    return result, X_scaled


# ==========================================
# 3. HYDRAULIC CLUSTERING
# ==========================================

hydraulic_result, hydraulic_scaled = run_kmeans(
    hydraulic,
    hydraulic_features,
    "hydraulic"
)


# ==========================================
# 4. RAINFALL + DRAINAGE CLUSTERING
# ==========================================

combined_result, combined_scaled = run_kmeans(
    combined,
    combined_features,
    "rainfall_drainage"
)


# ==========================================
# 5. DBSCAN ANOMALY DETECTION
# ==========================================

print("\nDBSCAN ANOMALY DETECTION")
print("=" * 50)

# DBSCAN is computationally expensive on large datasets.
# Use a reproducible sample of 5,000 observations.

sample_size = min(5000, len(hydraulic_scaled))

rng = np.random.RandomState(42)

sample_indices = rng.choice(
    len(hydraulic_scaled),
    size=sample_size,
    replace=False
)

X_sample = hydraulic_scaled[sample_indices]

dbscan = DBSCAN(
    eps=0.8,
    min_samples=15
)

db_labels = dbscan.fit_predict(X_sample)

print("DBSCAN sample size:", sample_size)
print("Clusters found:", len(set(db_labels)) - (1 if -1 in db_labels else 0))
print("Noise observations:", np.sum(db_labels == -1))

# Save DBSCAN sample results
dbscan_result = hydraulic.iloc[
    sample_indices
].copy()

dbscan_result["dbscan_cluster"] = db_labels

dbscan_result["is_anomaly"] = (
    db_labels == -1
)

dbscan_result.to_csv(
    "results/hydraulic_dbscan_sample.csv",
    index=False
)


# ==========================================
# 6. DBSCAN PCA VISUALIZATION
# ==========================================

pca = PCA(n_components=2)

X_dbscan_pca = pca.fit_transform(X_sample)

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    X_dbscan_pca[:, 0],
    X_dbscan_pca[:, 1],
    c=db_labels,
    cmap="viridis",
    s=10,
    alpha=0.6
)

plt.colorbar(scatter, label="DBSCAN cluster")

plt.title("DBSCAN Anomaly Detection - PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()

plt.savefig(
    "results/dbscan_pca.png",
    dpi=150
)

plt.close()


# ==========================================
# 7. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 55)
print("UNSUPERVISED ML COMPLETED")
print("=" * 55)

print("Hydraulic records:", len(hydraulic_result))
print("Rainfall records:", len(combined_result))

print("\nGenerated files:")
print("1. hydraulic_kmeans.csv")
print("2. rainfall_drainage_kmeans.csv")
print("3. hydraulic_dbscan_sample.csv")
print("4. hydraulic_kmeans_pca.png")
print("5. rainfall_drainage_kmeans_pca.png")
print("6. dbscan_pca.png")

print("\nAll results are saved in the results folder.")
