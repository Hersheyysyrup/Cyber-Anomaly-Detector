##This is a sample csv from the present dataset that ive created so that
##the user doesnt have to look around to find new dataset 
##and easily run and check the model on the streamlit app

import glob
import os
import pandas as pd

RAW_DIR = "data/raw"
OUTPUT_PATH = "outputs/sample_upload.csv"
N_ROWS = 50
RANDOM_STATE = 1


def main():
    os.makedirs("outputs", exist_ok=True)

    csv_paths = glob.glob(os.path.join(RAW_DIR, "*_plus.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No *_plus.csv files found in {RAW_DIR}")

    ##pick a random file
    preferred = [p for p in csv_paths if "thursday" in p.lower()]
    source_path = preferred[0] if preferred else csv_paths[0]

    print(f"Sampling from: {source_path}")
    df = pd.read_csv(source_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    ## grabs a mix data like begin attck ddos etc
    label_col = next(c for c in df.columns if c.lower() == "label")
    df[label_col] = df[label_col].astype(str).str.strip()

    benign = df[df[label_col].str.upper() == "BENIGN"]
    attacks = df[df[label_col].str.upper() != "BENIGN"]

    n_attack = min(N_ROWS // 2, len(attacks))
    n_benign = N_ROWS - n_attack

    sample = pd.concat([
        benign.sample(n=min(n_benign, len(benign)), random_state=RANDOM_STATE),
        attacks.sample(n=n_attack, random_state=RANDOM_STATE),
    ]).sample(frac=1, random_state=RANDOM_STATE)  # shuffle row order

    sample.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(sample)} sample rows to {OUTPUT_PATH}")
    print(f"Label breakdown:\n{sample[label_col].value_counts()}")
    print(f"\nThis file INCLUDES the real Label column, so you can verify "
          f"the app correctly strips it before prediction, and compare "
          f"predictions against the true answer afterward.")


if __name__ == "__main__":
    main()