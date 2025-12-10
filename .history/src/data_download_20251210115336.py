# src/data_download.py
"""
Step 2 (fixed v3): Download NIFTY 50 and India VIX using yfinance and save raw CSVs.
This version avoids any .rename() on Series and prints debug info about downloaded columns.
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
    print(f"Columns for {name}: {list(df.columns)}")
    return df

def build_adj_close_df(df, debug_name="series"):
    """
    Robustly extract adjusted/close prices and return a DataFrame with column 'adj_close'.
    NO .rename() used anywhere to avoid pandas 'str is not callable' issues.
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
            # Debug info: show available columns and raise informative error
            raise KeyError(f"No Close/Adj Close column found in the dataframe for {debug_name}. Columns: {list(df.columns)}")
    # Ensure datetime index and construct dataframe explicitly WITHOUT rename()
    series_indexed = pd.Series(data=series.values, index=pd.to_datetime(series.index))
    df_out = pd.DataFrame({'adj_close': series_indexed})
    return df_out

if __name__ == "__main__":
    # Download raw files
    df_nifty = download_and_save(nifty_ticker, "NIFTY50")
    df_vix = download_and_save(vix_ticker, "IndiaVIX")

    # Debug: show first few rows of raw downloads
    print("\nNIFTY raw head:")
    print(df_nifty.head().to_string())
    print("\nIndiaVIX raw head:")
    print(df_vix.head().to_string())

    # Build NIFTY cleaned DF with returns
    nifty_df = build_adj_close_df(df_nifty, debug_name="NIFTY50")
    nifty_df['return'] = nifty_df['adj_close'].pct_change()
    nifty_clean_path = os.path.join(CLEAN_DIR, "NIFTY50_clean.csv")
    nifty_df.to_csv(nifty_clean_path, index=True)
    print("\nSaved cleaned NIFTY (returns) ->", nifty_clean_path)

    # Build VIX cleaned DF (use Close/Adj Close as available)
    vix_df = build_adj_close_df(df_vix, debug_name="IndiaVIX")
    # create explicit column name 'india_vix' by building new DataFrame
    vix_df2 = pd.DataFrame({'india_vix': vix_df['adj_close']}, index=vix_df.index)
    vix_clean_path = os.path.join(CLEAN_DIR, "IndiaVIX_clean.csv")
    vix_df2.to_csv(vix_clean_path, index=True)
    print("Saved cleaned India VIX ->", vix_clean_path)

    print("\nAll done. If you still see an error, paste the entire terminal output here.")