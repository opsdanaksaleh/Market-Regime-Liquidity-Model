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
nifty_ticker = "^NSEI"
vix_ticker = "^INDIAVIX"

# --- date range (adjust as needed) ---
start = "2015-01-01"
end = datetime.today().strftime("%Y-%m-%d")

def download_and_save(ticker, name):
    print(f"Downloading {name} ({ticker}) from Yahoo Finance...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(
            f"No data downloaded for {ticker}. Possible causes: incorrect ticker, internet issues, API limits, or market holidays."
        )
    raw_path = os.path.join(RAW_DIR, f"{name}_raw.csv")
    df.to_csv(raw_path)
    print(f"Saved raw CSV -> {raw_path}")
    return df

def build_adj_close_df(df):
    """
    Robustly extract adjusted/close prices and return a DataFrame with column 'adj_close'.
    """
    # Try common column names
    if 'Adj Close' in df.columns:
        series = df['Adj Close']
    elif 'Adj_Close' in df.columns:  # rare variant
        series = df['Adj_Close']
    elif 'Close' in df.columns:
        series = df['Close']
    else:
        # As a final fallback, if dataframe has single column, use it
        if df.shape[1] == 1:
            series = df.iloc[:, 0]
        else:
            raise KeyError("No Close/Adj Close column found in the dataframe.")
    # Ensure datetime index and construct dataframe explicitly
    series.index = pd.to_datetime(series.index)
    df_out = series.rename('adj_close').to_frame()
    return df_out
    df_out = pd.DataFrame({'adj_close': series.values}, index=series.index)
    return df_out

if __name__ == "__main__":
    df_nifty = download_and_save(nifty_ticker, "NIFTY50")
    df_vix = download_and_save(vix_ticker, "IndiaVIX")

    # Build NIFTY cleaned DF with returns
    nifty_df = build_adj_close_df(df_nifty)
    # Rename column to nifty_close for consistency
    nifty_df = nifty_df.rename(columns={'adj_close': 'nifty_close'})
    nifty_df['return'] = nifty_df['nifty_close'].pct_change()
    nifty_clean_path = os.path.join(CLEAN_DIR, "NIFTY50_clean.csv")
    nifty_df.to_csv(nifty_clean_path, index=True)
    print("Saved cleaned NIFTY (returns) ->", nifty_clean_path)

    # Build VIX cleaned DF (use Close/Adj Close as available)
    vix_df = build_adj_close_df(df_vix)
    # Rename column to india_vix to be explicit
    vix_df = vix_df.rename(columns={'adj_close': 'india_vix'})
    vix_clean_path = os.path.join(CLEAN_DIR, "IndiaVIX_clean.csv")
    vix_df.to_csv(vix_clean_path, index=True)
    print("Saved cleaned India VIX ->", vix_clean_path)