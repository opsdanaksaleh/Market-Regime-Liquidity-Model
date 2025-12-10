# src/gsec_load.py
"""
Load and clean G-Sec tenor yields.
Place your downloaded CSV at: data/raw/gsec_tenor.csv
Outputs cleaned CSV: data/clean/gsec_tenor_clean.csv
Run: python src/gsec_load.py
"""

import os
import pandas as pd
from datetime import datetime

RAW = "data/raw/gsec_tenor.csv"
CLEAN = "data/clean/gsec_tenor_clean.csv"
os.makedirs(os.path.dirname(RAW), exist_ok=True)
os.makedirs(os.path.dirname(CLEAN), exist_ok=True)

# -- which tenors we want (string labels expected in CSV) --
WANTED = ['1Y', '3Y', '5Y', '10Y']

def try_parse_percent(x):
    """
    Convert strings like '6.45%' or '6.45' or 6.45 to float 6.45.
    """
    if pd.isna(x):
        return pd.NA
    if isinstance(x, str):
        x = x.strip().replace('%','').replace(',', '')
        if x == '':
            return pd.NA
    try:
        return float(x)
    except Exception:
        return pd.NA

def load_and_normalize():
    if not os.path.exists(RAW):
        raise FileNotFoundError(f"Please download/save the tenor CSV as: {RAW}")

    # Try to read CSV (let pandas infer delimiter)
    df_raw = pd.read_csv(RAW, parse_dates=['Date'] if 'Date' in pd.read_csv(RAW, nrows=1).columns else None)
    df = df_raw.copy()

    # If the file is in long format (Date, Tenor, Yield) convert to wide
    cols_lower = [c.lower() for c in df.columns]
    if 'tenor' in cols_lower and ('yield' in cols_lower or 'rate' in cols_lower or 'value' in cols_lower):
        # standardize column names
        df.columns = [c.strip() for c in df.columns]
        date_col = next(c for c in df.columns if c.lower() == 'date')
        tenor_col = next(c for c in df.columns if c.lower() == 'tenor')
        yield_col = next(c for c in df.columns if c.lower() in ('yield','rate','value'))
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.pivot(index=date_col, columns=tenor_col, values=yield_col).reset_index()
    
    # attempt to find date column
    date_col = None
    for c in df.columns:
        if c.lower() == 'date':
            date_col = c
            break
    if date_col is None:
        # attempt first column
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.set_index(date_col).sort_index()

    # normalize column names (strip spaces)
    df.columns = [str(c).strip() for c in df.columns]

    # pick wanted tenors; try multiple label variants
    final = pd.DataFrame(index=df.index)
    for w in WANTED:
        candidates = [c for c in df.columns if c.replace(' ', '').replace('Y','Y').lower().startswith(w.lower()) or c.lower()==w.lower()]
        if not candidates:
            # fallback: look for '1 year' etc.
            alt = {'1Y':'1 year', '3Y':'3 year', '5Y':'5 year', '10Y':'10 year'}
            candidates = [c for c in df.columns if alt[w].lower() in c.lower()]
        if candidates:
            # take first candidate
            final[w] = df[candidates[0]].apply(try_parse_percent)
        else:
            final[w] = pd.NA
            print(f"Warning: tenor {w} not found in raw CSV. Column candidates: {df.columns[:10]}")

    # convert to numeric and drop rows with all NaNs
    final = final.apply(pd.to_numeric, errors='coerce')
    final = final.dropna(how='all')

    # Resample / reindex to business days and forward-fill short gaps
    idx = pd.date_range(start=final.index.min(), end=final.index.max(), freq='B')
    final = final.reindex(idx)
    final = final.ffill(limit=5)  # forward-fill up to 5 business days
    final = final.dropna(how='all')  # keep only rows with at least one tenor

    # Save cleaned file
    final.index.name = 'Date'
    final.to_csv(CLEAN, float_format='%.4f')
    print(f"Saved cleaned GSec tenor CSV -> {CLEAN}")
    return final

if __name__ == "__main__":
    df_clean = load_and_normalize()
    print(df_clean.head().to_string())