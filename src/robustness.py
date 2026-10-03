# robustness test 1: noise injection
# we take the already-trained models and test set, then add increasing amounts of gaussian noise to
# the test features to see how much
# performance degrades - this simulates sensor noise, measurement error,
# or an attacker slightly perturbing traffic to evade detection

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve, auc

PROCESSED_DIR = "data/processed"
MODEL_DIR = "outputs/models"
PLOT_DIR = "outputs/plots"
RANDOM_STATE = 42

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)

NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def load_test_data():
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test_binary.csv").squeeze()
    return X_test, y_test


def load_models():
    rf = joblib.load(f"{MODEL_DIR}/random_forest.joblib")
    xgb = joblib.load(f"{MODEL_DIR}/xgboost.joblib")
    return {"Random Forest": rf, "XGBoost": xgb}


def add_noise(X: pd.DataFrame, noise_level: float, rng: np.random.RandomState) -> pd.DataFrame:
    
    if noise_level == 0.0:
        return X.copy()

    stds = X.std(axis=0)
    noise = rng.normal(loc=0.0, scale=stds.values * noise_level, size=X.shape)
    return X + noise


def evaluate_at_noise_level(model, X_test, y_test, noise_level, rng):
    X_noisy = add_noise(X_test, noise_level, rng)

    y_pred = model.predict(X_noisy)
    y_proba = model.predict_proba(X_noisy)[:, 1]

    f1 = f1_score(y_test, y_pred, pos_label=1)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    return f1, pr_auc


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    rng = np.random.RandomState(RANDOM_STATE)

    X_test, y_test = load_test_data()
    models = load_models()

    results = {name: {"f1": [], "pr_auc": []} for name in models}

    for noise_level in NOISE_LEVELS:
        print(f"\n--- Noise level: {noise_level} ---")
        for name, model in models.items():
            f1, pr_auc = evaluate_at_noise_level(model, X_test, y_test, noise_level, rng)
            results[name]["f1"].append(f1)
            results[name]["pr_auc"].append(pr_auc)
            print(f"{name}: F1={f1:.4f}, PR-AUC={pr_auc:.4f}")

    out_rows = []
    for name in models:
        for i, noise_level in enumerate(NOISE_LEVELS):
            out_rows.append({
                "model": name,
                "noise_level": noise_level,
                "f1": results[name]["f1"][i],
                "pr_auc": results[name]["pr_auc"][i],
            })
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv("outputs/metrics/noise_injection_results.csv", index=False)
    print("\nSaved results to outputs/metrics/noise_injection_results.csv")

 ## F1 degratation
    plt.figure(figsize=(8, 5))
    for name in models:
        plt.plot(NOISE_LEVELS, results[name]["f1"], marker="o", label=name)
    plt.xlabel("Noise level (fraction of feature std dev)")
    plt.ylabel("F1 score (Attack class)")
    plt.title("Model Performance Degradation Under Noise Injection")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/noise_injection_f1.png", dpi=150)
    print(f"Saved plot to {PLOT_DIR}/noise_injection_f1.png")

    # PR-AUC degradation
    plt.figure(figsize=(8, 5))
    for name in models:
        plt.plot(NOISE_LEVELS, results[name]["pr_auc"], marker="o", label=name)
    plt.xlabel("Noise level (fraction of feature std dev)")
    plt.ylabel("Precision-Recall AUC")
    plt.title("PR-AUC Degradation Under Noise Injection")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/noise_injection_prauc.png", dpi=150)
    print(f"Saved plot to {PLOT_DIR}/noise_injection_prauc.png")


if __name__ == "__main__":
    main()