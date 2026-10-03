# robustness test 1 noise injection
# we take the already-trained models and test set, then add increasing amounts
# of gaussian noise to the test features to see how much performance degrades -
# this simulates sensor noise, measurement error, or an attacker slightly
# perturbing traffic to evade detection

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

PROCESSED_DIR = "data/processed"
MODEL_DIR = "outputs/models"
PLOT_DIR = "outputs/plots"
METRICS_DIR = "outputs/metrics"
RANDOM_STATE = 42

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
CORRUPTION_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
N_ESTIMATORS = 100


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


def run_noise_injection_test():
    print("\n" + "#" * 60)
    print("# TEST 1: NOISE INJECTION")
    print("#" * 60)

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
    out_df.to_csv(f"{METRICS_DIR}/noise_injection_results.csv", index=False)
    print(f"\nSaved results to {METRICS_DIR}/noise_injection_results.csv")

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


# TEST 2 Label Noise Sensitivity 
# we deliberately corrupt (flip) a percentage of the pseudo-labels that
# classify.py trains on, retrain at each corruption level, and see how much
# performance against the REAL test labels degrades. this tells us how much
# the whole pipeline depends on clustering having produced good labels.

def load_training_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    pseudo = pd.read_csv(f"{PROCESSED_DIR}/pseudo_labels.csv")

    agree_mask = pseudo["agree"].astype(bool)
    X_train_filtered = X_train[agree_mask].reset_index(drop=True)
    y_train_filtered = pseudo.loc[agree_mask, "pseudo_label"].reset_index(drop=True)

    return X_train_filtered, y_train_filtered


def corrupt_labels(y: pd.Series, corruption_level: float, rng: np.random.RandomState) -> pd.Series:
    """Randomly flip a fraction of labels (0 -> 1, 1 -> 0)."""
    if corruption_level == 0.0:
        return y.copy()

    y_corrupted = y.copy()
    n_flip = int(len(y) * corruption_level)
    flip_idx = rng.choice(y.index, size=n_flip, replace=False)
    y_corrupted.loc[flip_idx] = 1 - y_corrupted.loc[flip_idx]

    return y_corrupted


def train_and_evaluate_rf(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred, pos_label=1)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    return f1, pr_auc


def train_and_evaluate_xgb(X_train, y_train, X_test, y_test):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    xgb = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred, pos_label=1)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    return f1, pr_auc


def run_label_noise_test():
    print("\n" + "#" * 60)
    print("# TEST 2: LABEL NOISE SENSITIVITY")
    print("#" * 60)

    rng = np.random.RandomState(RANDOM_STATE)

    X_train, y_train = load_training_data()
    X_test, y_test = load_test_data()

    results = {"Random Forest": {"f1": [], "pr_auc": []},
               "XGBoost": {"f1": [], "pr_auc": []}}

    for corruption_level in CORRUPTION_LEVELS:
        print(f"\n{'=' * 60}")
        print(f"Corruption level: {corruption_level * 100:.0f}%")
        print(f"{'=' * 60}")

        y_train_corrupted = corrupt_labels(y_train, corruption_level, rng)

        print("Training Random Forest...")
        rf_f1, rf_pr_auc = train_and_evaluate_rf(X_train, y_train_corrupted, X_test, y_test)
        results["Random Forest"]["f1"].append(rf_f1)
        results["Random Forest"]["pr_auc"].append(rf_pr_auc)
        print(f"Random Forest: F1={rf_f1:.4f}, PR-AUC={rf_pr_auc:.4f}")

        print("Training XGBoost...")
        xgb_f1, xgb_pr_auc = train_and_evaluate_xgb(X_train, y_train_corrupted, X_test, y_test)
        results["XGBoost"]["f1"].append(xgb_f1)
        results["XGBoost"]["pr_auc"].append(xgb_pr_auc)
        print(f"XGBoost: F1={xgb_f1:.4f}, PR-AUC={xgb_pr_auc:.4f}")

    out_rows = []
    for name in results:
        for i, corruption_level in enumerate(CORRUPTION_LEVELS):
            out_rows.append({
                "model": name,
                "corruption_level": corruption_level,
                "f1": results[name]["f1"][i],
                "pr_auc": results[name]["pr_auc"][i],
            })
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(f"{METRICS_DIR}/label_noise_results.csv", index=False)
    print(f"\nSaved results to {METRICS_DIR}/label_noise_results.csv")

    plt.figure(figsize=(8, 5))
    for name in results:
        plt.plot([c * 100 for c in CORRUPTION_LEVELS], results[name]["f1"], marker="o", label=name)
    plt.xlabel("Label corruption (%)")
    plt.ylabel("F1 score (Attack class)")
    plt.title("Model Performance vs. Pseudo-Label Corruption")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/label_noise_f1.png", dpi=150)
    print(f"Saved plot to {PLOT_DIR}/label_noise_f1.png")

    plt.figure(figsize=(8, 5))
    for name in results:
        plt.plot([c * 100 for c in CORRUPTION_LEVELS], results[name]["pr_auc"], marker="o", label=name)
    plt.xlabel("Label corruption (%)")
    plt.ylabel("Precision-Recall AUC")
    plt.title("PR-AUC vs. Pseudo-Label Corruption")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/label_noise_prauc.png", dpi=150)
    print(f"Saved plot to {PLOT_DIR}/label_noise_prauc.png")


if __name__ == "__main__":
    run_noise_injection_test()
    run_label_noise_test()