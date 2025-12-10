# src/gsec_merge.py
"""
Merges Investing.com G-Sec 1Y, 3Y, 5Y, 10Y CSVs into one clean dataset.
Fixes duplicate-date errors by aggregating duplicates (mean).
Output -> data/clean/gsec_tenor_clean.csv
Run: python src/gsec_merge.py
"""

import os
import pandas as pd
from pathlib import Path

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
    if 'Date' not in df.columns:
        raise KeyError(f"'Date' column not found in {path}. Columns: {list(df.columns)}")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Keep only Date and Price (some CSVs may call it 'Price' or 'Value')
    price_col = None
    for candidate in ['Price','price','Close','close','Last','last','Value','value']:
        if candidate in df.columns:
            price_col = candidate
            break
    if price_col is None:
        # fallback: try second column if Date is first
        possible_cols = [c for c in df.columns if c != 'Date']
        if not possible_cols:
            raise KeyError(f"No price-like column found in {path}. Columns: {list(df.columns)}")
        price_col = possible_cols[0]

    df = df[['Date', price_col]].rename(columns={price_col: tenor})

    # Clean numeric strings
    df[tenor] = (
        df[tenor]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df[tenor] = pd.to_numeric(df[tenor], errors='coerce')

    # Drop rows with NaT dates
    df = df.dropna(subset=['Date'])
    return df

def merge_all():
    merged = None
    duplicate_summary = {}

    for tenor, filename in FILES.items():
        path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        df = load_series(path, tenor)

        # Check for duplicate dates within this file
        dup_dates = df['Date'][df['Date'].duplicated(keep=False)]
        if not dup_dates.empty:
            dup_count = dup_dates.nunique()
            duplicate_summary[filename] = dup_count
            print(f"Warning: {filename} contains {dup_count} unique duplicated date(s). They will be aggregated by mean.")

        # Aggregate duplicate dates within the file by mean (if any)
        df = df.groupby('Date', as_index=False).mean()

        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="Date", how="outer")

    # Final duplicate check on merged dates
    if merged['Date'].duplicated().any():
        print("Warning: merged DataFrame has duplicate Date rows. Aggregating by mean across all columns.")
        merged = merged.groupby('Date', as_index=False).mean()

    # Sort by date and set index
    merged = merged.sort_values("Date").set_index("Date")

    # Now safe to reindex to business days
    try:
        merged = merged.asfreq("B")
    except ValueError as e:
        # extra safety: remove any remaining duplicate index labels
        if merged.index.has_duplicates:
            print("Removing duplicate index labels by aggregating (mean) — extra safety step.")
            merged = merged.groupby(level=0).mean()
            merged = merged.asfreq("B")
        else:
            raise

    # Forward fill short gaps
    merged = merged.ffill(limit=7)

    # Save cleaned output
    out_path = os.path.join(CLEAN_DIR, "gsec_tenor_clean.csv")
    merged.to_csv(out_path, float_format="%.4f")

    print("\nSaved cleaned G-Sec dataset ->", out_path)
    print("\nPreview (last 8 rows):\n", merged.tail(8).to_string())

    if duplicate_summary:
        print("\nDuplicate summary (per-file):")
        for f, cnt in duplicate_summary.items():
            print(f" - {f}: {cnt} duplicate date(s) aggregated")

    return merged

if __name__ == "__main__":
    merged = merge_all()