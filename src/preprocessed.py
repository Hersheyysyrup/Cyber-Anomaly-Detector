import os
import warnings
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

random_state = 42
test_size = 0.20
project_root = Path(__file__).resolve().parent.parent

raw_dir = project_root/"data"/"processed"
processed_dir = project_root/ "data" / "processed"

file_pattern = "*_plus.csv"

Identifier_Columns = {
    "flow id",
    "source ip",
    "destination ip",
    "timestamp",
}

Target_Columns = {
    "label",
    "label_binary",
    "label_multiclass",
}

def normalize_column_name(column: str) -> str:

    ### To normalize a column make everything equal
    return " ".join(str(column).strip().lower().split())

def print_section(title: str):
    """Print a readable section heading."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


###Data Loading###

def load_csv(raw_dir: Path) -> pd.DataFrame:

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory does not exist:\n{raw_dir}"
        )

    csv_files = sorted(raw_dir.glob(file_pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"No files matching '{file_pattern}' found in:\n{raw_dir}\n\n"
            "data/raw/."
        )

    print_section("LOADING DATA")
    print(f"Found {len(csv_files)} dataset files:")

    frames = []

    for file in csv_files:

        try:
            df = pd.read_csv(file, low_memory = False)

        except Exception as exc:
            raise RuntimeError(
                f"Unable to read file: {file}\n{exc}"
            )from exc

        print(
            f"  {file.name:<30} "
            f"{len(df):>10,} rows | "
            f"{len(df.columns):>3} columns"
        )

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index= True,
        sort = False,
    )

    print(f"\nCombined dataset shape: {combined.shape}")

    return combined

### Column Name Cleaning ###
def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    remove duplicate rows,
    remove empty columns,
    convert infinity values to NA,
    remove missing values
    """

    print_section("CLEANING DATA")

    df = df.copy()

    original_rows = len(df)
    original_columns = len(df.columns)

    df = df.replace(
        [np.inf, -np.inf], #Infinity to NA
        np.nan
    )

    empty_columns = [
        column                      #removes empty columns
        for column in df.columns
        if df[column].isna().all()
    ]

    if empty_columns:

        print(
            f"Dropping {len(empty_columns)} completely empty columns:"
        )

        for column in empty_columns:
            print(f"  - {column}")

        df = df.drop(columns=empty_columns)

    before = len(df)        #remove missing values
    df = df.dropna()

    dropped_nan = before - len(df)

    print(
        f"Dropped rows containing NaN/inf: "
        f"{dropped_nan:,}"
    )

    #remove duplicate rows

    before = len(df)
    df = df.drop_duplicates()

    dropped_duplicates = before - len(df)

    print(
        f"Dropped duplicate rows: "
        f"{dropped_duplicates:,}"
    )

    print(
        f"\nRows: "
        f"{original_rows:,} -> {len(df):,}"
    )

    print(
        f"Columns: "
        f"{original_columns} -> {len(df.columns)}"
    )

    return df

