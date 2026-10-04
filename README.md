# Cyber Anomaly Detector

A hybrid anomaly detection system for network intrusion detection, combining unsupervised clustering (to generate training labels without ground truth) with supervised classification (for fast, generalizable detection), evaluated for robustness under noise, label corruption, and simulated concept drift.

Built on the **CICIDS2017** dataset.

## How It Works

1. **Clustering generates pseudo-labels.** K-Means and Isolation Forest independently flag traffic as normal or anomalous, with no access to real labels. Rows where both methods agree become the training labels for the next stage (84.2% agreement in our run).
2. **Classification learns from those pseudo-labels.** Random Forest and XGBoost are trained on the agreement-filtered subset, then evaluated against the real, held-out test labels — the real test of whether unsupervised labeling was good enough to bootstrap a usable classifier.
3. **A second classifier identifies attack type.** For traffic flagged as anomalous, a separate Random Forest/XGBoost pair (trained on real multi-class labels) predicts the specific attack type — DDoS, Port Scan, SQL Injection, etc.
4. **Robustness testing** measures how performance degrades under three realistic failure conditions: random feature noise, corrupted training labels, and simulated distribution drift.
5. **A Streamlit app** lets you upload new traffic (single file or bulk CSV) and get live predictions with an attack-type breakdown, without retraining anything.

## Key Results

- **Clustering agreement:** 84.2% between K-Means and Isolation Forest (1.41M of 1.68M training rows kept)
- **Binary classification (real test set):** Random Forest F1=0.47, XGBoost F1=0.48 — both models performed *better* than the pseudo-labels they trained on, suggesting supervised learning partially corrected clustering noise
- **Multi-class attack-type ID:** Random Forest and XGBoost both achieve ~0.95+ macro F1 across 26 attack categories, including correctly identifying SQL Injection samples — but only after XGBoost was given explicit sample weighting to handle severe class imbalance (without it, XGBoost silently failed on every rare class while still reporting 99% overall accuracy)
- **Robustness:** PR-AUC drops sharply under even small amounts of noise; XGBoost is notably more resistant to label noise than Random Forest; synthetic concept drift causes both models' predictions to saturate at high drift magnitudes

Full plots and metrics are in `outputs/plots/` and `outputs/metrics/`.

## Project Structure

```
├── data/
│   ├── raw/                  # CICIDS2017 _plus.csv files (not committed — see .gitignore)
│   └── processed/            # cleaned/scaled train-test splits, scaler, pseudo-labels
├── src/
│   ├── preprocessed.py       # load, clean, scale, split CICIDS2017 data
│   ├── clustering.py         # K-Means + Isolation Forest pseudo-labeling
│   ├── classification.py     # binary classification (anomalous vs normal)
│   ├── multiclass.py         # attack-type classification
│   ├── robustness.py         # noise, label-noise, and drift testing
│   ├── sample.py             # generate a sample CSV (from real data) for testing the app
│   ├── demo.py                # presentation-safe script: loads saved results only, no retraining
│   └── app.py                  # Streamlit web app: live single-row demo + bulk CSV upload
├── outputs/
│   ├── models/                # trained model artifacts (.joblib)
│   ├── metrics/                # robustness results (.csv)
│   ├── plots/                  # robustness degradation plots (.png)
│   └── sample_upload.csv       # generated sample file for testing uploads
├── requirements.txt
└── README.md
```

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Download CICIDS2017 (the corrected `_plus.csv` files) and place them in `data/raw/`.

## Running the Pipeline

Run once, in order, to generate all models and results:

```powershell
python src/preprocessed.py
python src/clustering.py
python src/classification.py
python src/multiclass.py
python src/robustness.py
```

Then, for a quick look at everything without re-running anything:

```powershell
python src/demo.py
```

To generate a sample CSV (pulled from real data) for testing the app's upload feature:

```powershell
python src/sample.py
```

To launch the web app:

```powershell
streamlit run src/app.py
```

The app has two tabs:
- **Try a Real Example** — one click, instantly classifies a random row from the real held-out test set
- **Uploaded File Results** — upload a CSV of new traffic (e.g. `outputs/sample_upload.csv`), see total flows analyzed, normal/flagged counts, an attack-type breakdown chart, per-row predictions, and a downloadable results CSV. Any label columns in the upload are automatically stripped before prediction so they can never leak into the result.

## Methodology Notes

- **DBSCAN was evaluated and dropped** in favor of Isolation Forest after repeated scalability failures (extremely slow, then a memory error, even after PCA dimensionality reduction) on the full 1.67M-row training set — a documented, real limitation of density-based clustering in high-dimensional space, not an oversight.
- **SVM was considered and excluded** from the classification stage — standard kernel SVM does not scale to datasets of this size (O(n²)-O(n³) complexity), making Random Forest and XGBoost the practical choice.
- **Concept drift is simulated synthetically**, not derived from real timestamps, since the raw `Timestamp` column was dropped during preprocessing and train/test splitting shuffled row order. This is a standard, accepted approach when real longitudinal data isn't available, and is stated as a limitation rather than presented as real-world drift.
- **The binary decision rule in the app** flags a row as "Attack" based on the Random Forest prediction specifically; XGBoost's prediction is shown alongside as a secondary reference, not used for flagging — documented explicitly rather than left ambiguous.

## Limitations

- Binary classification F1 (~0.47-0.48) reflects the inherent noise of training on unsupervised pseudo-labels rather than ground truth — this is expected and is the central question the project investigates, not a flaw to hide.
- Several attack types (SQL Injection, Heartbleed) have extremely few samples in CICIDS2017, making their individual metrics less statistically stable.
- Robustness tests use a reduced `n_estimators` (100 vs. 200) for the label-noise test specifically, to keep retraining time reasonable across corruption levels.