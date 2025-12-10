# src/data_download.py
"""
Step 2: Download NIFTY 50 and India VIX using yfinance and save raw CSVs.
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
# Yahoo tickers:
# NIFTY 50  -> ^NSEI
# NIFTY BANK -> ^NSEBANK
# India VIX  -> ^INDIAVIX
nifty_ticker = "^NSEI"
vix_ticker = "^INDIAVIX"

# --- date range (adjust as needed) ---
start = "2015-01-01"   # beginner-friendly: start from 2015
end = datetime.today().strftime("%Y-%m-%d")

def download_and_save(ticker, name):
    print(f"Downloading {name} ({ticker}) from Yahoo Finance...")
    # explicit: keep auto_adjust=False so 'Adj Close' is present
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}. Check ticker or internet.")
    # save raw
    raw_path = os.path.join(RAW_DIR, f"{name}_raw.csv")
    df.to_csv(raw_path)
    print(f"Saved raw CSV -> {raw_path}")
    return df

def get_adj_close_column(df):
    # Prefer 'Adj Close' if present; otherwise fallback to 'Close'
    if 'Adj Close' in df.columns:
        return df['Adj Close'].rename('adj_close')
    elif 'Close' in df.columns:
        return df['Close'].rename('adj_close')
    else:
        raise KeyError("Neither 'Adj Close' nor 'Close' found in the downloaded dataframe.")

if __name__ == "__main__":
    df_nifty = download_and_save(nifty_ticker, "NIFTY50")
    df_vix = download_and_save(vix_ticker, "IndiaVIX")

    # --- basic preprocessing example: daily returns for NIFTY ---
    nifty_adj = get_adj_close_column(df_nifty).to_frame()
    nifty_adj.index = pd.to_datetime(nifty_adj.index)
    nifty_adj['return'] = nifty_adj['adj_close'].pct_change()  # simple daily returns
    nifty_clean_path = os.path.join(CLEAN_DIR, "NIFTY50_clean.csv")
    nifty_adj.to_csv(nifty_clean_path, index=True)
    print("Saved cleaned NIFTY (returns) ->", nifty_clean_path)

    # India VIX: prefer 'Close' column (VIX is an index, uses Close)
    if 'Close' in df_vix.columns:
        vix_close = df_vix['Close'].rename('india_vix').to_frame()
    elif 'Adj Close' in df_vix.columns:
        vix_close = df_vix['Adj Close'].rename('india_vix').to_frame()
    else:
        raise KeyError("IndiaVIX download lacks 'Close' or 'Adj Close' column.")
    vix_close.index = pd.to_datetime(vix_close.index)
    vix_clean_path = os.path.join(CLEAN_DIR, "IndiaVIX_clean.csv")
    vix_close.to_csv(vix_clean_path, index=True)
    print("Saved cleaned India VIX ->", vix_clean_path)