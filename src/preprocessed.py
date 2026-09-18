import glob
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FILE_PATTERN = "*_plus.csv"

def load_all_csvs(raw_dir: str) -> pd.DataFrame:
    """Load and concatenate all the files into one DataFrame"""

    csv_paths = glob.glob(os.path.join(raw_dir, FILE_PATTERN))
    if not csv_paths:
        raise FileNotFoundError(
            f"No files matching {FILE_PATTERN} found in {raw_dir}. "
            f"Check that your '_plus' CSVs are in that folder.")

    print(
        f"Found {len(csv_paths)} files:")
    for p in csv_paths:
        print(f"  - {os.path.basename(p)}")

    frames = []
    for path in csv_paths:
        print(f"Loading {path}...")
        df = pd.read_csv(path, low_memory= False)
        frames.append(df)

    full_df = df.concat(frames, ignore_index = True)
    print(f"\nCombined shape: {full_df.shape}")
    return full_df

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """remove whitespaces from column names"""

    df.columns = [c.strip() for c in df.columns]
    return df

def clean_values (df: pd.DataFrame) -> pd.DataFrame:
    """Removes infinity, null values and duplicate rows"""

    df = df.replac([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows")

    return df

def build_labels(df: pd.DataFrame):
    """create binary and multiclass columns"""
    label_col_candidates = [c for c in df.columns if c.lower() == "label"]
    if not label_col_candidates:
        raise KeyError(
            f"Couldn't find a 'Label' column. Columns present: {list(df.columns)}"
        )

    label_col = label_col_candidates[0]

    df["label_col"] = df[label_col].astype(str).str.strip()
    df["label_multiclass"] = df[label_col]
    df["label_binary"] = (df[label_col].str.upper() != "BENIGN").astype(int)

    print("\nClass distribution (binary):")
    print(df["label_binary"].value_counts(normalize = True))

    print("\nClass distribution (binary):")
    print(df["label_multiclass"].value_counts())

    #Flag id SQL injection is present 

    sql_mask = df["label_multiclass"].str.contauns("sql", case = False, na=False)
    print(f"\nSQL Injection rows found: {sql_mask.sum()}")

    return df, label_col

def split_features_labels(df: pd.DataFrame, label_col: str):
    """to seperate features from feature columns"""

    label_cols = [label_col, "label_multiclass", "label_binary"]
    feature_cols = [c for c in df.columns if c not in label_cols]

    numeric_cols = df[feature_cols].select_dtypes(include =[np.number]).columns.tolist()
    dropped = set(feature_cols) - set(numeric_cols)
    if dropped:
        print(f"\mDropping non numeric feature columns: {dropped}")


    X = df[numeric_cols].copy()
    y_binary = df["label_binary"].copy()
    y_multiclass = df["label_multiclass"].copy()

    return X, y_binary, y_multiclass


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_all_csvs(RAW_DIR)
    df = clean_columns(df)
    df = clean_values(df)
    df , label_col = build_labels(df)

    X, y_binary, y_multiclass = split_features_labels(df, label_col)
 
    X_train, X_test, yb_train, yb_test, ym_train, ym_test = train_test_split(
        X, y_binary, y_multiclass,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_binary,
    )

    #Feature Scaling(Train only )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns = X_train.columns, index = X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), column = X_test.columns, index = X_test.index)
    
    X_train_scaled.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
    X_test_scaled.to_csv(f"{PROCESSED_DIR}/X_test.csv", index=False)
    yb_train.to_csv(f"{PROCESSED_DIR}/y_train_binary.csv", index=False)
    yb_test.to_csv(f"{PROCESSED_DIR}/y_test_binary.csv", index=False)
    ym_train.to_csv(f"{PROCESSED_DIR}/y_train_multiclass.csv", index=False)
    ym_test.to_csv(f"{PROCESSED_DIR}/y_test_multiclass.csv", index=False)

    print(f"\nSaved processed data to {PROCESSED_DIR}/")
    print(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
 
 
if __name__ == "__main__":
    main()

