# run k means clustering - it partitions traffic into groups typically anomaly and fine
# run isolation forest - also unsupervised, but isolates anomalies by how FEW random
# splits it takes to separate a point from the rest (anomalies isolate faster)
# after comparing the 2 the mutual voted row is selected and other is dropped
# they vote on whether network traffic is anomalous or normal
# this would help us in robustness
# once this file runs: k means precision/recall vs real labels and
# isolation forest's precision/recall vs real labels

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

PROCESSED_DIR = "data/processed"
RANDOM_STATE = 42

# Rough expected anomaly proportion in the data — adjust this based on the
# actual binary class distribution printed during preprocessing.
# Our data came out ~24.5% attack, so we set contamination close to that.
CONTAMINATION = 0.245


def load_processed():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    y_train_binary = pd.read_csv(f"{PROCESSED_DIR}/y_train_binary.csv").squeeze()

    return X_train, y_train_binary


def run_kmeans(X: pd.DataFrame, n_clusters: int = 2) -> np.ndarray:
    """since attacks are less likely to occur, treating the smaller cluster as the attack one"""
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    raw_labels = km.fit_predict(X)

    counts = pd.Series(raw_labels).value_counts()
    anomaly_cluster = counts.idxmin()
    pseudo_labels = (raw_labels == anomaly_cluster).astype(int)

    print(f"K-Means cluster sizes: {dict(counts)}")
    print(f"K-Means treating cluster {anomaly_cluster} as anomalous "
          f"({pseudo_labels.sum()} points flagged)")
    return pseudo_labels


def run_isolation_forest(X: pd.DataFrame, contamination: float = CONTAMINATION) -> np.ndarray:
    """
    isolation forest natively outputs -1 for anomalies, 1 for normal points.
    we convert that to 0/1 so it matches k-means' convention (1 = anomalous).
    scales fine to millions of rows, unlike dbscan - no subsampling needed.
    """
    iso = IsolationForest(
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    raw_labels = iso.fit_predict(X)
    pseudo_labels = (raw_labels == -1).astype(int)

    print(f"Isolation Forest flagged {pseudo_labels.sum()} points as anomalous "
          f"out of {len(X)}")
    return pseudo_labels


def evaluate_against_ground_truth(pseudo_labels: np.ndarray, y_true: pd.Series, method_name: str):
    print(f"\n--- {method_name} vs. ground truth ---")
    print(confusion_matrix(y_true, pseudo_labels))
    print(classification_report(y_true, pseudo_labels, target_names=["Benign", "Attack"]))


def main():
    X_train, y_train_binary = load_processed()
    print(f"Loaded X_train: {X_train.shape}\n")

    print("=" * 60)
    print("Running K-Means")
    print("=" * 60)
    km_labels = run_kmeans(X_train)
    evaluate_against_ground_truth(km_labels, y_train_binary, "K-Means")

    print("\n" + "=" * 60)
    print("Running Isolation Forest")
    print("=" * 60)
    iso_labels = run_isolation_forest(X_train)
    evaluate_against_ground_truth(iso_labels, y_train_binary, "Isolation Forest")

    # now we have to combine rows where both methods are mutual
    agreement_mask = km_labels == iso_labels
    combined_labels = km_labels

    print("\n" + "=" * 60)
    print("Combined results")
    print("=" * 60)
    print(f"Agreement between K-Means and Isolation Forest: "
          f"{agreement_mask.sum()} / {len(agreement_mask)} "
          f"({100 * agreement_mask.mean():.1f}%)")

    filtered_true = y_train_binary[agreement_mask]
    filtered_pseudo = combined_labels[agreement_mask]
    evaluate_against_ground_truth(filtered_pseudo, filtered_true, "Agreement-filtered subset")

    out = pd.DataFrame({
        "kmeans_label": km_labels,
        "isoforest_label": iso_labels,
        "pseudo_label": combined_labels,
        "agree": agreement_mask,
    })
    out.to_csv(f"{PROCESSED_DIR}/pseudo_labels.csv", index=False)
    print(f"\nSaved pseudo-labels to {PROCESSED_DIR}/pseudo_labels.csv")


if __name__ == "__main__":
    main()