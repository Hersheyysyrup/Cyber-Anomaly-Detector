# run k means clustering - it partitions traffic into groups typically anonmaly and fine
# run dbscan - it also does the same job but intead its working on density here
# after comparing the 2 the mutual voted row is selected and other is dropped
# they vote on whether network traffic anomalous or normal
# this would help us in robustness 
# once this files runs k means precision/recall vs real labels and 
# Dbscan;s precision/recall vs real labels 

import os 
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import classification_report, confusion_matrix

PROCESSED_DIR = "data/processed"
RANDOM_STATE = 42

DBSCAN_SUBSAMPLE = None
DBSCAN_EPS = 1.5
DBSCAN_MIN_SAMPLES = 10


def load_processed():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    y_train_binary = pd.read_csv(f"{PROCESSED_DIR}/y_train_binary.csv").squeeze()

    return X_train, y_train_binary


def run_kmeans(X: pd.DataFrame, n_clusters: int = 2) -> np.ndarray:
    """since attacks are very less likely to occur using small clusters to represent attack"""
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    raw_labels = km.fit_predict(X)

    counts = pd.Series(raw_labels).value_counts()
    anomaly_cluster = counts.idxmin()
    pseudo_labels = (raw_labels == anomaly_cluster).astype(int)

    print(f"K-Means cluster sizes: {dict(counts)}")
    print(f"K-Means treating cluster {anomaly_cluster} as anomalous "
          f"({pseudo_labels.sum()} points flagged)")
    return pseudo_labels


def run_dbscan(X: pd.DataFrame, eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES) -> np.ndarray:
    """since db scan points as -1 we will be treating anomalities as noice cuz the point doesnt fit densely 
    into any normal traffic cluster"""
    data = X
    if DBSCAN_SUBSAMPLE and len(X) > DBSCAN_SUBSAMPLE:
        print(f"Subsampling {DBSCAN_SUBSAMPLE} rows for DBSCAN (full data too slow)")

        data = X.sample(DBSCAN_SUBSAMPLE, random_state=RANDOM_STATE)

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    raw_labels = db.fit_predict(data)
    pseudo_labels = pd.Series((raw_labels == -1).astype(int), index=data.index)

    n_noise = pseudo_labels.sum()
    n_clusters = len(set(raw_labels)) - (1 if -1 in raw_labels else 0)
    print(f"DBSCAN found {n_clusters} clusters, flagged {n_noise} noise/anomalous points "
          f"out of {len(data)}")

    if DBSCAN_SUBSAMPLE and len(X) > DBSCAN_SUBSAMPLE:
        full_labels = pd.Series(0, index=X.index)
        full_labels.loc[pseudo_labels.index] = pseudo_labels
        return full_labels.values

    return pseudo_labels.values


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
    print("Running DBSCAN")
    print("=" * 60)
    db_labels = run_dbscan(X_train)
    evaluate_against_ground_truth(db_labels, y_train_binary, "DBSCAN")

    # now we have to combine rows where both methods are mutual
    agreement_mask = km_labels == db_labels
    combined_labels = km_labels

    print("\n" + "=" * 60)
    print("Combined results")
    print("=" * 60)
    print(f"Agreement between K-Means and DBSCAN: "
          f"{agreement_mask.sum()} / {len(agreement_mask)} "
          f"({100 * agreement_mask.mean():.1f}%)")

    filtered_true = y_train_binary[agreement_mask]
    filtered_pseudo = combined_labels[agreement_mask]
    evaluate_against_ground_truth(filtered_pseudo, filtered_true, "Agreement-filtered subset")

    out = pd.DataFrame({
        "kmeans_label": km_labels,
        "dbscan_label": db_labels,
        "pseudo_label": combined_labels,
        "agree": agreement_mask,
    })
    out.to_csv(f"{PROCESSED_DIR}/pseudo_labels.csv", index=False)
    print(f"\nSaved pseudo-labels to {PROCESSED_DIR}/pseudo_labels.csv")


if __name__ == "__main__":
    main()