#training the labels mutually decided by k_means and dbscan here
#using random_forest and xgboost here 
#since clustering doesnt scale well for real time detection so 
# we use classification to train a model that can label new traffic instantly

import os 
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
)
from xgboost import XGBClassifer

PROCESSED_DIR = "data/processed"
MODEL_DIR = "outputs/models"
RANDOM_STATE = 42

def load_training_data():
    X_train =pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
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
    randomforest = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",  # helps with the imbalance we still have even after filtering
    )
    randomforest.fit(X_train. y_train)
    return randomforest

def train_xgboost(X_train, y_train):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg /max(n_pos , 1)

    xgb = XGBClassifer(
    n_estimators = 200,
    random_state = RANDOM_STATE,
    scale_pos_weight = scale_pos_weight,
    eval_metric = "logloss",
    n_jobs = -1,
    )
    xgb.fit (X_train, y_train)
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
 
    return pr_auc

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, y_train = load_training_data()
    X_test, y_test = load_test_data()

    print("\n" + "=" * 60)
    print("Training Random Forest")
    print("=" * 60)
    rf_model = train_random_forest(X_train, y_train)
    rf_pr_auc = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    print("\n" + "=" * 60)
    print("Training XGBoost")
    print("=" * 60)
    xgb_model = train_xgboost(X_train, y_train)
    xgb_pr_auc = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Random Forest PR-AUC: {rf_pr_auc:.4f}")
    print(f"XGBoost PR-AUC:       {xgb_pr_auc:.4f}")

    """save both these modesl for robustness check later """

    joblib.dump(rf_model, f"{MODEL_DIR}/random_forest.joblib")
    joblib.dump(xgb_model, f"{MODEL_DIR}/xgboost.joblib")
    print(f"\nSaved models to {MODEL_DIR}/")

if __name__ == "__main__":
    main()