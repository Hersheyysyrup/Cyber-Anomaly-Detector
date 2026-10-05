# train supervised classifiers on the pseudo-labels produced by clustering.py
# we only train on rows where kmeans and isolation forest agreed (the "clean" subset)
# then we test against the REAL test set labels - this tells us if pseudo-labeling
# actually worked well enough to build a usable classifier
#
# metrics for BOTH classes (Benign and Attack), for each algorithm separately,
# are saved to disk so demo.py can display them later WITHOUT retraining anything

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    accuracy_score,
    auc,
)
from xgboost import XGBClassifier

PROCESSED_DIR = "data/processed"
MODEL_DIR = "outputs/models"
RANDOM_STATE = 42


def load_training_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    pseudo = pd.read_csv(f"{PROCESSED_DIR}/pseudo_labels.csv")

    agree_mask = pseudo["agree"].astype(bool)
    X_train_filtered = X_train[agree_mask].reset_index(drop=True)
    y_train_filtered = pseudo.loc[agree_mask, "pseudo_label"].reset_index(drop=True)

    print(f"Training on agreement-filtered subset: {X_train_filtered.shape[0]} rows "
          f"out of {X_train.shape[0]} total "
          f"({100 * agree_mask.mean():.1f}% kept)")

    return X_train_filtered, y_train_filtered


def load_test_data():
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test_binary.csv").squeeze()
    return X_test, y_test


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(X_train, y_train):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    xgb = XGBClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    return xgb


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {model_name} on real test set ---")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"]))

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    print(f"{model_name} Precision-Recall AUC: {pr_auc:.4f}")

    return pr_auc, y_pred


def get_metrics_rows(y_true, y_pred, pr_auc, model_name: str) -> list:
    """Extracts precision/recall/F1 for BOTH classes, plus accuracy and
    PR-AUC, as plain numbers - one row per class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    rows = []
    for i, class_name in enumerate(["Benign", "Attack"]):
        rows.append({
            "model": model_name,
            "class": class_name,
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
            "accuracy": round(float(acc), 4),
            "pr_auc": round(float(pr_auc), 4),
        })
    return rows


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, y_train = load_training_data()
    X_test, y_test = load_test_data()

    print("\n" + "=" * 60)
    print("Training Random Forest")
    print("=" * 60)
    rf_model = train_random_forest(X_train, y_train)
    rf_pr_auc, rf_pred = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    print("\n" + "=" * 60)
    print("Training XGBoost")
    print("=" * 60)
    xgb_model = train_xgboost(X_train, y_train)
    xgb_pr_auc, xgb_pred = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Random Forest PR-AUC: {rf_pr_auc:.4f}")
    print(f"XGBoost PR-AUC:       {xgb_pr_auc:.4f}")

    
    metrics_rows = []
    metrics_rows += get_metrics_rows(y_test, rf_pred, rf_pr_auc, "Random Forest")
    metrics_rows += get_metrics_rows(y_test, xgb_pred, xgb_pr_auc, "XGBoost")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(f"{PROCESSED_DIR}/classification_metrics.csv", index=False)
    print(f"\nSaved classification metrics to {PROCESSED_DIR}/classification_metrics.csv")
    print(metrics_df.to_string(index=False))

    joblib.dump(rf_model, f"{MODEL_DIR}/random_forest.joblib")
    joblib.dump(xgb_model, f"{MODEL_DIR}/xgboost.joblib")
    print(f"\nSaved models to {MODEL_DIR}/")


if __name__ == "__main__":
    main()