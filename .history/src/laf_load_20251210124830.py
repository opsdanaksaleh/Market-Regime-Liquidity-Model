# src/laf_load.py
"""
Load & clean RBI LAF / liquidity CSV saved at data/raw/laf_liquidity.csv
Outputs: data/clean/laf_liquidity_clean.csv
Also computes:
 - net_laf (repo - reverse_repo OR provided net figure)
 - liquidity_z_30d (30-business-day rolling z-score)
Run: python src/laf_load.py
"""

import os
import pandas as pd
import numpy as np

RAW = "data/raw/laf_liquidity.csv"
CLEAN = "data/clean/laf_liquidity_clean.csv"
os.makedirs(os.path.dirname(RAW), exist_ok=True)
os.makedirs(os.path.dirname(CLEAN), exist_ok=True)

def parse_possible_files():
    """
    Read the raw CSV flexibly: RBI might provide columns named:
    - Date, Repo Outstanding, Reverse Repo Outstanding, Net LAF, Surplus/Deficit
    Or a single 'Surplus(-)/Deficit' column.
    The function tries to find best available columns and returns a DataFrame.
    """
    if not os.path.exists(RAW):
        raise FileNotFoundError(f"Please place the RBI LAF CSV at: {RAW}")

    df = pd.read_csv(RAW, nrows=5)
    # quick peek at columns to debug if needed
    print("Raw file columns preview:", list(df.columns))

    # Read full file with date parsing (try common date header names)
    candidates = [c for c in df.columns if c.lower() == 'date']
    if not candidates:
        # try first column as date
        date_col = df.columns[0]
    else:
        date_col = candidates[0]

    df_full = pd.read_csv(RAW, parse_dates=[date_col], dayfirst=False)
    df_full = df_full.rename(columns={date_col: 'Date'})
    df_full['Date'] = pd.to_datetime(df_full['Date'], errors='coerce')
    df_full = df_full.set_index('Date').sort_index()

    # Normalize column names lowercased -> stripped
    col_map = {c: c.strip() for c in df_full.columns}
    df_full = df_full.rename(columns=col_map)

    # Try to identify key columns
    cols = [c.lower() for c in df_full.columns]

    # Common tokens we might find
    repo_candidates = [c for c in df_full.columns if 'repo' in c.lower() and ('outstand' in c.lower() or 'outstanding' in c.lower() or 'amt' in c.lower())]
    reverse_candidates = [c for c in df_full.columns if 'reverse' in c.lower() and ('repo' in c.lower() or 'outstand' in c.lower())]
    net_candidates = [c for c in df_full.columns if any(x in c.lower() for x in ['net laf','net','surplus','deficit','surplus(-)/deficit'])]
    amount_candidates = [c for c in df_full.columns if any(x in c.lower() for x in ['amount','amt','value'])]

    # detection rules (best-effort)
    repo_col = repo_candidates[0] if repo_candidates else None
    reverse_col = reverse_candidates[0] if reverse_candidates else None
    net_col = None
    if net_candidates:
        net_col = net_candidates[0]
    else:
        # try to find a column with 'surplus' or 'deficit'
        sd = [c for c in df_full.columns if ('surplus' in c.lower() or 'deficit' in c.lower())]
        if sd:
            net_col = sd[0]

    # Build a working DataFrame with as many columns as we can identify
    work = pd.DataFrame(index=df_full.index)
    if repo_col:
        work['repo'] = pd.to_numeric(df_full[repo_col], errors='coerce')
    if reverse_col:
        work['reverse_repo'] = pd.to_numeric(df_full[reverse_col], errors='coerce')
    if net_col:
        work['net_laf_raw'] = pd.to_numeric(df_full[net_col], errors='coerce')

    # If net_laf_raw not present but repo & reverse present, compute net = repo - reverse
    if 'net_laf_raw' not in work.columns and 'repo' in work.columns and 'reverse_repo' in work.columns:
        work['net_laf_raw'] = work['repo'] - work['reverse_repo']

    # If none found, try to find any numeric column that could represent surplus/deficit
    if 'net_laf_raw' not in work.columns:
        numeric_cols = [c for c in df_full.columns if pd.api.types.is_numeric_dtype(df_full[c])]
        if numeric_cols:
            # choose the first numeric column (fallback; user should check)
            col0 = numeric_cols[0]
            print(f"Fallback: using first numeric column '{col0}' as liquidity measure")
            work['net_laf_raw'] = pd.to_numeric(df_full[col0], errors='coerce')

    # Standardize units:
    # RBI usually reports amounts in crores. If values appear huge (e.g., >1e6), we won't touch them;
    # otherwise we keep as-is. The loader doesn't change units automatically — user should confirm.
    # We'll provide net_laf in the same units as the source (document in README later).
    work['net_laf'] = work['net_laf_raw']

    # Compute rolling z-score of net_laf to capture liquidity stress (30 business days)
    # Use business-day rolling window; since index is dates, ensure freq
    work = work.sort_index()
    # reindex to business days and forward fill a small amount to handle missing days
    idx = pd.date_range(start=work.index.min(), end=work.index.max(), freq='B')
    work = work.reindex(idx)
    work = work.ffill(limit=7)

    # Compute rolling statistics (30 business days)
    window = 30
    work['net_laf_ma30'] = work['net_laf'].rolling(window, min_periods=5).mean()
    work['net_laf_sd30'] = work['net_laf'].rolling(window, min_periods=5).std()
    work['liquidity_z_30d'] = (work['net_laf'] - work['net_laf_ma30']) / work['net_laf_sd30']

    # Save cleaned file
    out_cols = ['net_laf', 'net_laf_ma30', 'net_laf_sd30', 'liquidity_z_30d']
    work[out_cols].to_csv(CLEAN, float_format="%.4f")
    print(f"Saved cleaned LAF liquidity -> {CLEAN}")
    print("\nPreview (tail):\n", work[out_cols].tail(8).to_string())

    return work[out_cols]

if __name__ == "__main__":
    df_clean = parse_possible_files()