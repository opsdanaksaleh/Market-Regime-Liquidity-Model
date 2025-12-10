# src/data_download.py
"""
Step 2 (fixed v4): Download NIFTY 50 and India VIX using yfinance and save raw CSVs.
This version handles MultiIndex columns by flattening them and robustly selecting Close/Adj Close.
Run: python src/data_download.py
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- folders ---
RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

# --- choose tickers ---
nifty_ticker = "^NSEI"
vix_ticker = "^INDIAVIX"

# --- date range (adjust as needed) ---
start = "2015-01-01"
end = datetime.today().strftime("%Y-%m-%d")

def download_and_save(ticker, name):
    print(f"\nDownloading {name} ({ticker}) from Yahoo Finance...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}. Check ticker or internet.")
    raw_path = os.path.join(RAW_DIR, f"{name}_raw.csv")
    df.to_csv(raw_path)
    print(f"Saved raw CSV -> {raw_path}")
    print(f"Original columns for {name}: {list(df.columns)}")
    return df

def flatten_columns(df):
    """If df has MultiIndex columns, flatten to single-level string names."""
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            # join levels with a single space; convert to string to be safe
            new_cols.append(' '.join([str(c) for c in col]).strip())
        df = df.copy()
        df.columns = new_cols
    else:
        # ensure columns are strings
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
    return df

def pick_price_column(df, prefer_adj=True):
    """
    Return best candidate column name for price series.
    prefer_adj=True prefers 'Adj Close' over 'Close' if both exist.
    """
    cols = [c.lower() for c in df.columns]
    # find exact/contains matches
    adj_candidates = [df.columns[i] for i, c in enumerate(cols) if 'adj close' in c or 'adj_close' in c or 'adjusted close' in c]
    close_candidates = [df.columns[i] for i, c in enumerate(cols) if ('close' in c and 'adj' not in c)]
    if prefer_adj and adj_candidates:
        return adj_candidates[0]
    if close_candidates:
        return close_candidates[0]
    if adj_candidates:
        return adj_candidates[0]
    # fallback: if only one column exists, return it
    if df.shape[1] == 1:
        return df.columns[0]
    raise KeyError(f"No suitable price column found. Available columns: {list(df.columns)}")

def build_price_df(df, label, prefer_adj=True):
    """
    Flatten columns, pick a price column, return DataFrame with a 1D numeric series named 'label'.
    """
    df_flat = flatten_columns(df)
    chosen_col = pick_price_column(df_flat, prefer_adj=prefer_adj)
    print(f"Chosen price column for {label}: '{chosen_col}'")
    series = pd.to_numeric(df_flat[chosen_col], errors='coerce')
    series.index = pd.to_datetime(df_flat.index)
    df_out = pd.DataFrame({label: series.values}, index=series.index)
    return df_out

if __name__ == "__main__":
    # Download raw files
    df_nifty_raw = download_and_save(nifty_ticker, "NIFTY50")
    df_vix_raw = download_and_save(vix_ticker, "IndiaVIX")

    # Debug: show first few rows of raw downloads (flattened preview)
    print("\nNIFTY raw head (flattened preview):")
    print(flatten_columns(df_nifty_raw).head().to_string())
    print("\nIndiaVIX raw head (flattened preview):")
    print(flatten_columns(df_vix_raw).head().to_string())

    # Build NIFTY cleaned DF with returns
    nifty_price_df = build_price_df(df_nifty_raw, label='adj_close', prefer_adj=True)
    nifty_price_df['return'] = nifty_price_df['adj_close'].pct_change()
    nifty_clean_path = os.path.join(CLEAN_DIR, "NIFTY50_clean.csv")
    nifty_price_df.to_csv(nifty_clean_path, index=True)
    print("\nSaved cleaned NIFTY (returns) ->", nifty_clean_path)

    # Build VIX cleaned DF (rename column to 'india_vix')
    vix_price_df = build_price_df(df_vix_raw, label='india_vix', prefer_adj=True)
    vix_clean_path = os.path.join(CLEAN_DIR, "IndiaVIX_clean.csv")
    vix_price_df.to_csv(vix_clean_path, index=True)
    print("Saved cleaned India VIX ->", vix_clean_path)