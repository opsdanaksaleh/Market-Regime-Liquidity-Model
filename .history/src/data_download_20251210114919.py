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
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}. Check ticker or internet.")
    # save raw
    raw_path = f"{RAW_DIR}/{name}_raw.csv"
    df.to_csv(raw_path)
    print(f"Saved raw CSV -> {raw_path}")
    return df

if __name__ == "__main__":
    df_nifty = download_and_save(nifty_ticker, "NIFTY50")
    df_vix = download_and_save(vix_ticker, "IndiaVIX")

    # --- basic preprocessing example: daily returns for NIFTY ---
    df_nifty_clean = df_nifty[['Adj Close']].rename(columns={'Adj Close':'adj_close'})
    df_nifty_clean['return'] = df_nifty_clean['adj_close'].pct_change()  # simple daily returns
    df_nifty_clean.to_csv(f"{CLEAN_DIR}/NIFTY50_clean.csv", index=True)
    print("Saved cleaned NIFTY (returns) -> data/clean/NIFTY50_clean.csv")

    # India VIX: keep close column
    df_vix_clean = df_vix[['Close']].rename(columns={'Close':'india_vix'})
    df_vix_clean.to_csv(f"{CLEAN_DIR}/IndiaVIX_clean.csv", index=True)
    print("Saved cleaned India VIX -> data/clean/IndiaVIX_clean.csv")