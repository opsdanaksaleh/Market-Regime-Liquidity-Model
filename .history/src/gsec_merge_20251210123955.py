# src/gsec_merge.py
"""
Merges Investing.com G-Sec 1Y, 3Y, 5Y, 10Y CSVs into one clean dataset.
Output -> data/clean/gsec_tenor_clean.csv
Run: python src/gsec_merge.py
"""

import os
import pandas as pd

RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
os.makedirs(CLEAN_DIR, exist_ok=True)

FILES = {
    "1Y": "gsec_1y.csv",
    "3Y": "gsec_3y.csv",
    "5Y": "gsec_5y.csv",
    "10Y": "gsec_10y.csv"
}

def load_series(path, tenor):
    df = pd.read_csv(path)

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Keep only Date and Price
    df = df[['Date', 'Price']].rename(columns={"Price": tenor})

    # Convert yield to float
    df[tenor] = (
        df[tenor]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df[tenor] = pd.to_numeric(df[tenor], errors='coerce')

    return df

def merge_all():
    merged = None

    for tenor, filename in FILES.items():
        path = os.path.join(RAW_DIR, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        df = load_series(path, tenor)

        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="Date", how="outer")

    # Sort by date
    merged = merged.sort_values("Date").set_index("Date")

    # Resample to Business Days and forward-fill
    merged = merged.asfreq("B")
    merged = merged.ffill(limit=7)

    # Save cleaned output
    out_path = os.path.join(CLEAN_DIR, "gsec_tenor_clean.csv")
    merged.to_csv(out_path, float_format="%.4f")

    print("\nSaved cleaned G-Sec dataset ->", out_path)
    print("\nPreview:\n", merged.tail())

if __name__ == "__main__":
    merge_all()