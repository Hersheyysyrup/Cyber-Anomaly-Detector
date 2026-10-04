###This file is the actual working that has to be presented in front of the viewer
### ensure running the previous files first beforehand 

import joblib
import pandas as pd
from PIL import Image
 
PROCESSED_DIR = "data/processed"
MODEL_DIR = "outputs/models"
METRICS_DIR = "outputs/metrics"
PLOT_DIR = "outputs/plots"
 
 
def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
 
 
def show_clustering_results():
    section("STAGE 1: CLUSTERING (K-MEANS + ISOLATION FOREST PSEUDO-LABELS)")
    pseudo = pd.read_csv(f"{PROCESSED_DIR}/pseudo_labels.csv")
 
    agree_pct = 100 * pseudo["agree"].mean()
    print(f"Agreement between K-Means and Isolation Forest: {agree_pct:.1f}%")
    print(f"Rows kept for training after agreement filtering: "
          f"{pseudo['agree'].sum():,} / {len(pseudo):,}")
 
 
def show_binary_classification():
    section("STAGE 2: BINARY CLASSIFICATION (ANOMALOUS vs NORMAL)")
 
    rf = joblib.load(f"{MODEL_DIR}/random_forest.joblib")
    xgb = joblib.load(f"{MODEL_DIR}/xgboost.joblib")
 
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test_binary.csv").squeeze()
 
    sample = X_test.sample(10, random_state=1)
    true_labels = y_test.loc[sample.index]
 
    rf_preds = rf.predict(sample)
    xgb_preds = xgb.predict(sample)
 
    results = pd.DataFrame({
        "True Label": true_labels.map({0: "Benign", 1: "Attack"}).values,
        "Random Forest Prediction": pd.Series(rf_preds).map({0: "Benign", 1: "Attack"}).values,
        "XGBoost Prediction": pd.Series(xgb_preds).map({0: "Benign", 1: "Attack"}).values,
    })
    print("\nLive predictions on 10 random real test rows:\n")
    print(results.to_string(index=False))
 
 
def show_multiclass_results():
    section("STAGE 3: MULTI-CLASS ATTACK-TYPE CLASSIFICATION")
 
    try:
        rf_multi = joblib.load(f"{MODEL_DIR}/random_forest_multiclass.joblib")
        encoder = joblib.load(f"{MODEL_DIR}/label_encoder_multiclass.joblib")
    except FileNotFoundError:
        print("Multi-class models not found - skipping this section.")
        return
 
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test_multi = pd.read_csv(f"{PROCESSED_DIR}/y_test_multiclass.csv").squeeze()
 
    sql_mask = y_test_multi.str.contains("sql", case=False, na=False)
    sql_rows = X_test[sql_mask]
 
    if len(sql_rows) == 0:
        print("No SQL Injection rows available in this test split to demo live.")
        return
 
    preds_enc = rf_multi.predict(sql_rows)
    preds = encoder.inverse_transform(preds_enc)
 
    print(f"\nLive predictions on {len(sql_rows)} real SQL Injection test rows:\n")
    demo_df = pd.DataFrame({
        "True Label": y_test_multi[sql_mask].values,
        "Predicted": preds,
    })
    print(demo_df.to_string(index=False))
 
 
def show_robustness_summary():
    section("STAGE 4: ROBUSTNESS TESTING SUMMARY")
 
    noise = pd.read_csv(f"{METRICS_DIR}/noise_injection_results.csv")
    label_noise = pd.read_csv(f"{METRICS_DIR}/label_noise_results.csv")
    drift = pd.read_csv(f"{METRICS_DIR}/concept_drift_results.csv")
 
    print("\n--- Noise Injection (baseline vs highest noise level) ---")
    print(noise[noise["noise_level"].isin([noise["noise_level"].min(),
                                             noise["noise_level"].max()])]
          .to_string(index=False))
 
    print("\n--- Label Noise Sensitivity (0% vs 30% corruption) ---")
    print(label_noise[label_noise["corruption_level"].isin([0.0, 0.30])]
          .to_string(index=False))
 
    print("\n--- Concept Drift (baseline vs highest drift level) ---")
    print(drift[drift["drift_level"].isin([drift["drift_level"].min(),
                                             drift["drift_level"].max()])]
          .to_string(index=False))
 
 
def show_plots():
    section("ROBUSTNESS PLOTS")
    plot_files = [
        "noise_injection_f1.png",
        "label_noise_f1.png",
        "concept_drift_f1.png",
    ]
    for fname in plot_files:
        path = f"{PLOT_DIR}/{fname}"
        print(f"Opening {path} ...")
        try:
            Image.open(path).show()
        except FileNotFoundError:
            print(f"  (not found - run robustness.py first)")
 
 
def main():
    print("CYBER ANOMALY DETECTION - LIVE DEMO")
    print("(all models and results below were pre-computed - nothing is being trained now)")
 
    show_clustering_results()
    show_binary_classification()
    show_multiclass_results()
    show_robustness_summary()
    show_plots()
 
 
if __name__ == "__main__":
    main()
 