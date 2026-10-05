# visualize_clustering.py
# shows how well K-Means and Isolation Forest separated Benign vs Attack,
# visually, using a SMALL sample of real training points (not all of them).
#
# unlike visualize_separation.py (which shows a TRAINED CLASSIFIER's decision
# on test data), this shows the CLUSTERING step itself - comparing what each
# clustering method guessed against the true label, on the same small sample.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

PROCESSED_DIR = "data/processed"
PLOT_DIR = "outputs/plots"
RANDOM_STATE = 42

N_SAMPLES_PER_CLASS = 5  # 5 benign + 5 attack = 10 total points shown
CONTAMINATION = 0.245


def load_train_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train_binary.csv").squeeze()
    return X_train, y_train


def pick_small_sample(X_train, y_train):
    benign_idx = y_train[y_train == 0].sample(N_SAMPLES_PER_CLASS, random_state=RANDOM_STATE).index
    attack_idx = y_train[y_train == 1].sample(N_SAMPLES_PER_CLASS, random_state=RANDOM_STATE).index

    sample_idx = benign_idx.append(attack_idx)
    return X_train.loc[sample_idx], y_train.loc[sample_idx]


def plot_clustering_result(X_sample, y_sample, X_full, cluster_labels_sample, method_name):
    """
    cluster_labels_sample: the clustering method's 0/1 guess for just the
    10 sampled points (already computed on the full training set, then
    sliced down to match our sample).
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca.fit(X_full)  # fit on full data for a meaningful projection
    X_sample_2d = pca.transform(X_sample)

    fig, ax = plt.subplots(figsize=(8, 6))

    for true_label, label_name, color in [(0, "Benign", "tab:blue"), (1, "Attack", "tab:red")]:
        mask = y_sample.values == true_label
        ax.scatter(
            X_sample_2d[mask, 0], X_sample_2d[mask, 1],
            c=color, label=f"True: {label_name}",
            s=150, edgecolors="black", linewidths=1.5, alpha=0.85,
        )


    wrong_mask = cluster_labels_sample != y_sample.values
    if wrong_mask.any():
        ax.scatter(
            X_sample_2d[wrong_mask, 0], X_sample_2d[wrong_mask, 1],
            c="none", edgecolors="black", linewidths=2.5, s=300,
            marker="X", label="Clustering disagreed with true label",
        )

    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"Clustering Separation — {method_name}\n"
                 f"({N_SAMPLES_PER_CLASS} Benign + {N_SAMPLES_PER_CLASS} Attack samples, PCA-reduced)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_name = method_name.lower().replace(" ", "_")
    out_path = f"{PLOT_DIR}/clustering_separation_{safe_name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close()


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    X_train, y_train = load_train_data()
    X_sample, y_sample = pick_small_sample(X_train, y_train)

    print("Running K-Means (for visualization only)...")
    km = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
    km_all = km.fit_predict(X_train)
    km_counts = pd.Series(km_all).value_counts()
    km_anomaly_cluster = km_counts.idxmin()
    km_labels_all = (km_all == km_anomaly_cluster).astype(int)
    km_labels_sample = pd.Series(km_labels_all, index=X_train.index).loc[X_sample.index].values

    print("Running Isolation Forest (for visualization only)...")
    iso = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1)
    iso_all = iso.fit_predict(X_train)
    iso_labels_all = (iso_all == -1).astype(int)
    iso_labels_sample = pd.Series(iso_labels_all, index=X_train.index).loc[X_sample.index].values

    plot_clustering_result(X_sample, y_sample, X_train, km_labels_sample, "K-Means")
    plot_clustering_result(X_sample, y_sample, X_train, iso_labels_sample, "Isolation Forest")

    print("\nSample points used (for reference in your report):")
    print(pd.DataFrame({
        "True Label": y_sample.map({0: "Benign", 1: "Attack"}).values,
        "K-Means Guess": pd.Series(km_labels_sample).map({0: "Benign", 1: "Attack"}).values,
        "Isolation Forest Guess": pd.Series(iso_labels_sample).map({0: "Benign", 1: "Attack"}).values,
    }, index=y_sample.index))


if __name__ == "__main__":
    main()