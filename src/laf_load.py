# src/laf_load.py
"""
Robust LAF loader for RBI / TradingEconomics files.

- Accepts data/raw/laf_liquidity.csv OR data/raw/laf_liquidity.xlsx (.xls)
- Auto-detects header row if the file has multi-row headers.
- Detects repo, reverse repo, net, surplus/deficit columns (many name variants).
- Falls back to first numeric column if nothing explicit is found (with warnings).
- Produces data/clean/laf_liquidity_clean.csv with:
    net_laf, net_laf_ma30, net_laf_sd30, liquidity_z_30d

Run: python src/laf_load.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

RAW_CSV = Path("data/raw/laf_liquidity.csv")
RAW_XLSX = Path("data/raw/laf_liquidity.xlsx")
RAW_XLS = Path("data/raw/laf_liquidity.xls")
OUT = Path("data/clean/laf_liquidity_clean.csv")
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/clean", exist_ok=True)

def read_table_flex(path: Path):
    """Read CSV or Excel, auto-detect header row if needed."""
    print(f"Reading file: {path}")
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        try:
            # try to read header normally first
            df = pd.read_excel(path, engine="openpyxl", header=0)
        except Exception:
            # fallback: read without header and detect
            df0 = pd.read_excel(path, header=None, engine="openpyxl")
            df = detect_and_set_header(df0)
    else:
        # CSV path
        try:
            df = pd.read_csv(path, header=0)
            # if columns are unnamed (happens when CSV is actually an xlsx), detect header
            if all(str(c).startswith("Unnamed") for c in df.columns[:min(6, len(df.columns))]):
                df0 = pd.read_csv(path, header=None)
                df = detect_and_set_header(df0)
        except Exception:
            # fallback to reading without header and detect
            df0 = pd.read_csv(path, header=None, engine="python", encoding='utf-8', errors='ignore')
            df = detect_and_set_header(df0)
    return df

def detect_and_set_header(df_nohead: pd.DataFrame):
    """
    Given a DataFrame read with header=None, find the header row index by searching
    for a cell containing 'date' (case-insensitive) within the first 12 rows.
    Return a dataframe with proper header row set.
    """
    max_scan = min(12, len(df_nohead))
    header_row = None
    for i in range(max_scan):
        row_vals = [str(x).strip().lower() for x in df_nohead.iloc[i].astype(str).values]
        if any("date" in v for v in row_vals):
            header_row = i
            break
    if header_row is None:
        # if we didn't find an explicit header, try to find a row where many entries look like dates
        for i in range(max_scan):
            try:
                parsed = pd.to_datetime(df_nohead.iloc[i].astype(str), errors='coerce')
                if parsed.notna().sum() >= 1:
                    header_row = i
                    break
            except Exception:
                pass
    if header_row is None:
        # as absolute fallback, use row 0 as header
        header_row = 0
        print("Warning: Could not find header row automatically; using row 0 as header. Inspect the output and re-run if incorrect.")
    # Build header
    header = df_nohead.iloc[header_row].astype(str).apply(lambda x: x.strip())
    df = df_nohead.iloc[header_row+1:].copy()
    df.columns = [c if c and not str(c).startswith("Unnamed") else f"col_{i}" for i, c in enumerate(header)]
    df = df.reset_index(drop=True)
    return df

def choose_date_col(cols):
    candidates = [c for c in cols if 'date' in c.lower()]
    if candidates:
        return candidates[0]
    # look for common alternate names
    for c in cols:
        if any(k in c.lower() for k in ['day','period','week','week ended']):
            return c
    # fallback to first column
    return cols[0]

def try_parse_percent_or_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip().replace('%','').replace(',','')
        if x=='':
            return np.nan
    try:
        return float(x)
    except Exception:
        # try to parse commas as thousand separators
        try:
            return float(str(x).replace(',',''))
        except Exception:
            return np.nan

def parse_possible_files():
    # Detect which raw exists
    if RAW_CSV.exists():
        path = RAW_CSV
    elif RAW_XLSX.exists():
        path = RAW_XLSX
    elif RAW_XLS.exists():
        path = RAW_XLS
    else:
        raise FileNotFoundError(f"Please place LAF file at one of: {RAW_CSV}, {RAW_XLSX}, or {RAW_XLS}")

    df_raw = read_table_flex(path)
    print("Detected columns (preview):", list(df_raw.columns[:20]))
    # standardize columns
    cols = [str(c).strip() for c in df_raw.columns]
    df_raw.columns = cols

    # find date column
    date_col = choose_date_col(cols)
    print("Using date column:", date_col)
    # parse dates in that column, coerce errors
    df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors='coerce', dayfirst=False)
    df_full = df_raw.set_index(date_col).sort_index()
    # drop rows where index is NaT
    df_full = df_full[~df_full.index.isna()]

    # Now try to detect repo, reverse repo, net_laf, surplus/deficit columns
    lower_cols = [c.lower() for c in df_full.columns]

    repo_candidates = [c for c in df_full.columns if 'repo' in c.lower() and 'reverse' not in c.lower()]
    reverse_candidates = [c for c in df_full.columns if 'reverse' in c.lower() and 'repo' in c.lower()]
    net_candidates = [c for c in df_full.columns if any(k in c.lower() for k in ['net laf','net','surplus','deficit','surplus(-)/deficit','surplus/deficit','injection','absorption'])]
    # Also look for 'surplus' or 'deficit'
    sd_candidates = [c for c in df_full.columns if ('surplus' in c.lower() or 'deficit' in c.lower())]

    print("Detected candidate columns counts -> repo:", len(repo_candidates),
          "reverse:", len(reverse_candidates), "net-like:", len(net_candidates), "surplus/deficit:", len(sd_candidates))

    work = pd.DataFrame(index=df_full.index)
    if repo_candidates:
        col = repo_candidates[0]
        work['repo'] = pd.to_numeric(df_full[col].apply(try_parse_percent_or_number), errors='coerce')
        print("Using repo column:", col)
    if reverse_candidates:
        col = reverse_candidates[0]
        work['reverse_repo'] = pd.to_numeric(df_full[col].apply(try_parse_percent_or_number), errors='coerce')
        print("Using reverse repo column:", col)
    if net_candidates:
        col = net_candidates[0]
        work['net_laf_raw'] = pd.to_numeric(df_full[col].apply(try_parse_percent_or_number), errors='coerce')
        print("Using net-like column:", col)
    elif sd_candidates:
        col = sd_candidates[0]
        work['net_laf_raw'] = pd.to_numeric(df_full[col].apply(try_parse_percent_or_number), errors='coerce')
        print("Using surplus/deficit column:", col)

    # If no net column but repo & reverse available, compute net
    if 'net_laf_raw' not in work.columns and set(['repo','reverse_repo']).issubset(work.columns):
        work['net_laf_raw'] = work['repo'] - work['reverse_repo']
        print("Computed net_laf_raw as repo - reverse_repo")

    # Fallback: pick first numeric column from df_full
    if 'net_laf_raw' not in work.columns:
        numeric_cols = [c for c in df_full.columns if pd.to_numeric(df_full[c].astype(str).str.replace(',','').str.replace('%',''), errors='coerce').notna().any()]
        if numeric_cols:
            use = numeric_cols[0]
            work['net_laf_raw'] = pd.to_numeric(df_full[use].apply(try_parse_percent_or_number), errors='coerce')
            print(f"Fallback: using first numeric column '{use}' as liquidity measure. Please verify this is correct.")
        else:
            raise KeyError(f"Could not detect any numeric liquidity column in the file. Available columns: {list(df_full.columns)}")

    # Prepare final work frame
    work = work.sort_index()
    # Reindex to business days and forward-fill small gaps
    idx = pd.date_range(start=work.index.min(), end=work.index.max(), freq='B')
    work = work.reindex(idx)
    work = work.ffill(limit=7)

    # Compute rolling stats (30 business days)
    window = 30
    work['net_laf'] = work['net_laf_raw']
    work['net_laf_ma30'] = work['net_laf'].rolling(window, min_periods=5).mean()
    work['net_laf_sd30'] = work['net_laf'].rolling(window, min_periods=5).std()
    work['liquidity_z_30d'] = (work['net_laf'] - work['net_laf_ma30']) / work['net_laf_sd30']

    out_cols = ['net_laf', 'net_laf_ma30', 'net_laf_sd30', 'liquidity_z_30d']
    work[out_cols].to_csv(OUT, float_format="%.4f")
    print(f"Saved cleaned LAF liquidity -> {OUT}")
    print("\nPreview (tail):\n", work[out_cols].tail(8).to_string())

    return work[out_cols]

if __name__ == "__main__":
    df_clean = parse_possible_files()