# visualize_separation.py
# shows how well-separated Benign vs Attack classes are, visually, using a
# SMALL sample of real test points (not all of them - a handful is clearer
# to look at and is what was asked for).
#
# since we have 102 features and can't plot that directly, PCA compresses
# everything down to 2 dimensions (the 2 directions that capture the most
# variance) so it can be plotted on a normal x/y scatter plot.

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

PROCESSED_DIR = "data/processed"
MODEL_DIR = "outputs/models"
PLOT_DIR = "outputs/plots"
RANDOM_STATE = 42

N_SAMPLES_PER_CLASS = 5  # 5 benign + 5 attack = 10 total points shown


def load_test_data():
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test_binary.csv").squeeze()
    return X_test, y_test


def pick_small_sample(X_test, y_test, rng):
    """Pick a small, balanced sample - some benign, some attack - so the
    plot clearly shows both classes rather than being dominated by one."""
    benign_idx = y_test[y_test == 0].sample(N_SAMPLES_PER_CLASS, random_state=RANDOM_STATE).index
    attack_idx = y_test[y_test == 1].sample(N_SAMPLES_PER_CLASS, random_state=RANDOM_STATE).index

    sample_idx = benign_idx.append(attack_idx)
    return X_test.loc[sample_idx], y_test.loc[sample_idx]


def plot_separation(X_sample, y_sample, model, model_name):

    X_test, _ = load_test_data()
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca.fit(X_test)

    X_sample_2d = pca.transform(X_sample)
    y_pred = model.predict(X_sample)

    fig, ax = plt.subplots(figsize=(8, 6))

    for true_label, label_name, color in [(0, "Benign", "tab:blue"), (1, "Attack", "tab:red")]:
        mask = y_sample.values == true_label
        ax.scatter(
            X_sample_2d[mask, 0], X_sample_2d[mask, 1],
            c=color, label=f"True: {label_name}",
            s=150, edgecolors="black", linewidths=1.5, alpha=0.85,
        )


    wrong_mask = y_pred != y_sample.values
    if wrong_mask.any():
        ax.scatter(
            X_sample_2d[wrong_mask, 0], X_sample_2d[wrong_mask, 1],
            c="none", edgecolors="black", linewidths=2.5, s=300,
            marker="X", label="Misclassified",
        )

    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"Class Separation — {model_name}\n"
                 f"({N_SAMPLES_PER_CLASS} Benign + {N_SAMPLES_PER_CLASS} Attack samples, PCA-reduced)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    out_path = f"{PLOT_DIR}/class_separation_{safe_name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close()


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    rng = np.random.RandomState(RANDOM_STATE)

    X_test, y_test = load_test_data()
    X_sample, y_sample = pick_small_sample(X_test, y_test, rng)

    rf = joblib.load(f"{MODEL_DIR}/random_forest.joblib")
    xgb = joblib.load(f"{MODEL_DIR}/xgboost.joblib")

    plot_separation(X_sample, y_sample, rf, "Random Forest")
    plot_separation(X_sample, y_sample, xgb, "XGBoost")

    print("\nSample points used (for reference in your report):")
    print(pd.DataFrame({
        "True Label": y_sample.map({0: "Benign", 1: "Attack"}).values
    }, index=y_sample.index))


if __name__ == "__main__":
    main()