##Multiclass has been created for only rows of actual attack types.
##I'd be using it to tell the type of the attack instead of just predicting between normal vs anomalous 
##this is what actually answers the project's SQL Injection requirement directly

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
 
PROCESSED_DIR = "data/processed"
MODEL_DIR = "outputs/models"
RANDOM_STATE = 42
 
MIN_SAMPLES_WARNING = 20
 
 
def load_attack_only_data():
    """loads only attacked row's train/test data and drops benign rows"""

    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    y_train_multiclass = pd.read_csv(f"{PROCESSED_DIR}/y_train_multiclass.csv").squeeze()
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test_multiclass = pd.read_csv(f"{PROCESSED_DIR}/y_test_multiclass.csv").squeeze()
 
    train_mask = y_train_multiclass.str.upper() != "BENIGN"
    test_mask = y_test_multiclass.str.upper() != "BENIGN"
 
    X_train_attacks = X_train[train_mask].reset_index(drop=True)
    y_train_attacks = y_train_multiclass[train_mask].reset_index(drop=True)
    X_test_attacks = X_test[test_mask].reset_index(drop=True)
    y_test_attacks = y_test_multiclass[test_mask].reset_index(drop=True)
 
    print(f"Attack-only training rows: {X_train_attacks.shape[0]}")
    print(f"Attack-only test rows: {X_test_attacks.shape[0]}")
 
    print("\nTraining set attack-type distribution:")
    counts = y_train_attacks.value_counts()
    print(counts)
 
    small_classes = counts[counts < MIN_SAMPLES_WARNING]
    if not small_classes.empty:
        print(f"\nWarning: these attack types have very few training samples "
              f"(below {MIN_SAMPLES_WARNING}):")
        print(small_classes)
        print("Expect unstable/unreliable scores for these classes - this is "
              "worth noting as a limitation, not a bug.")
 
    return X_train_attacks, y_train_attacks, X_test_attacks, y_test_attacks
 
 
def encode_labels(y_train, y_test, encoder: LabelEncoder):
    
    y_train_enc = encoder.fit_transform(y_train)
 
    known_classes = set(encoder.classes_)
    unseen_mask = ~y_test.isin(known_classes)
    if unseen_mask.any():
        print(f"\nDropping {unseen_mask.sum()} test rows with attack types "
              f"never seen in training: {y_test[unseen_mask].unique()}")
 
    y_test_known = y_test[~unseen_mask]
    y_test_enc = encoder.transform(y_test_known)
 
    return y_train_enc, y_test_enc, unseen_mask
 
 
def train_random_forest(X_train, y_train_enc):
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",  
    )
    rf.fit(X_train, y_train_enc)
    return rf
 
 
def train_xgboost(X_train, y_train_enc, n_classes):
    xgb = XGBClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    ###since xgboost doesn't have a built in class weight for multiclass like random forest
    ###therefore we are computing per row sample weights manually to achieve the same balancing
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_enc)
    xgb.fit(X_train, y_train_enc, sample_weight=sample_weights)
    return xgb
 
 
def evaluate_model(model, X_test, y_test_enc, encoder: LabelEncoder, model_name):
    y_pred_enc = model.predict(X_test)
 
  
    present_labels = sorted(set(y_test_enc) | set(y_pred_enc))
    target_names = encoder.inverse_transform(present_labels)
 
    print(f"\n--- {model_name}: attack-type classification ---")
    print(classification_report(
        y_test_enc, y_pred_enc,
        labels=present_labels,
        target_names=target_names,
        zero_division=0,
    ))
 
    sql_classes = [c for c in encoder.classes_ if "sql" in c.lower()]
    for sql_class in sql_classes:
        if sql_class in target_names:
            idx = list(target_names).index(sql_class)
            label_id = present_labels[idx]
            mask = y_test_enc == label_id
            if mask.sum() > 0:
                correct = (y_pred_enc[mask] == label_id).sum()
                print(f"SQL Injection check -> '{sql_class}': "
                      f"{correct}/{mask.sum()} correctly identified in test set")
            else:
                print(f"SQL Injection check -> '{sql_class}': "
                      f"0 test samples available (too rare to evaluate)")
 
 
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
 
    X_train, y_train, X_test, y_test = load_attack_only_data()
 
    encoder = LabelEncoder()
    y_train_enc, y_test_enc, unseen_mask = encode_labels(y_train, y_test, encoder)
    X_test_known = X_test[~unseen_mask].reset_index(drop=True)
 
    n_classes = len(encoder.classes_)
    print(f"\nTotal attack types (classes): {n_classes}")
 
    print("\n" + "=" * 60)
    print("Training Random Forest (multi-class)")
    print("=" * 60)
    rf_model = train_random_forest(X_train, y_train_enc)
    evaluate_model(rf_model, X_test_known, y_test_enc, encoder, "Random Forest")
 
    print("\n" + "=" * 60)
    print("Training XGBoost (multi-class)")
    print("=" * 60)
    xgb_model = train_xgboost(X_train, y_train_enc, n_classes)
    evaluate_model(xgb_model, X_test_known, y_test_enc, encoder, "XGBoost")
 
    joblib.dump(rf_model, f"{MODEL_DIR}/random_forest_multiclass.joblib")
    joblib.dump(xgb_model, f"{MODEL_DIR}/xgboost_multiclass.joblib")
    joblib.dump(encoder, f"{MODEL_DIR}/label_encoder_multiclass.joblib")
    print(f"\nSaved multi-class models and label encoder to {MODEL_DIR}/")
 
 
if __name__ == "__main__":
    main()
