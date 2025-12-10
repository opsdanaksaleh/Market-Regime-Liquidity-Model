# src/data_prep.py
"""
STEP 3: Volatility & Liquidity Analysis - data alignment, indicators, and plots.

Outputs:
 - data/clean/market_dataset.csv
 - plots/ (several PNGs):
     - ts_prices_vol_vix.png
     - vol_liquidity_timeseries.png
     - yield_curve_metrics.png

Run:
    python src/data_prep.py

Notes (beginner-friendly):
 - This script expects these cleaned inputs (created in Step 2):
    data/clean/NIFTY50_clean.csv        # columns: adj_close, return
    data/clean/IndiaVIX_clean.csv       # column: india_vix
    data/clean/gsec_tenor_clean.csv     # columns: 1Y,3Y,5Y,10Y
    data/clean/laf_liquidity_clean.csv  # columns: net_laf, net_laf_ma30, net_laf_sd30, liquidity_z_30d
 - It aligns everything to business days and forward-fills yields/liquidity small gaps.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- paths ---
CLEAN_DIR = Path("data/clean")
OUT_FILE = CLEAN_DIR / "market_dataset.csv"
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- helper functions ---
def load_csv(path, index_col=0, parse_dates=True):
    df = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
    df.index = pd.to_datetime(df.index)
    return df

def business_index_union(dfs):
    # union of date ranges -> business days between min and max
    min_date = min(df.index.min() for df in dfs if df is not None)
    max_date = max(df.index.max() for df in dfs if df is not None)
    return pd.date_range(start=min_date, end=max_date, freq="B")

def annualize_vol(std_series, trading_days=252):
    return std_series * np.sqrt(trading_days)

# --- load cleaned datasets (assumes prior steps succeeded) ---
print("Loading cleaned inputs...")
nifty = load_csv(CLEAN_DIR / "NIFTY50_clean.csv")
vix = load_csv(CLEAN_DIR / "IndiaVIX_clean.csv")
gsec = load_csv(CLEAN_DIR / "gsec_tenor_clean.csv")
laf = load_csv(CLEAN_DIR / "laf_liquidity_clean.csv")

print("Top rows NIFTY:", nifty.head(2).to_string())
print("Top rows VIX:", vix.head(2).to_string())
print("Top rows GSec:", gsec.head(2).to_string())
print("Top rows LAF:", laf.head(2).to_string())

# --- build unified business-day index ---
print("Aligning to business-day index across datasets...")
full_index = business_index_union([nifty, vix, gsec, laf])

# --- create base DataFrame and join ---
df = pd.DataFrame(index=full_index)
df = df.join(nifty[['adj_close','return']], how='left')
df = df.join(vix, how='left')
df = df.join(gsec, how='left')
df = df.join(laf, how='left')

# forward-fill yields and liquidity small gaps only (not returns)
df[['1Y','3Y','5Y','10Y','net_laf','net_laf_ma30','net_laf_sd30','liquidity_z_30d']] = \
    df[['1Y','3Y','5Y','10Y','net_laf','net_laf_ma30','net_laf_sd30','liquidity_z_30d']].ffill(limit=7)

# --- Derived metrics: realized vol, rolling vol, vix metrics ---
print("Computing volatility metrics...")
# Realized volatility: rolling std of daily returns (window 30 business days), annualized
df['rv_30d'] = df['return'].rolling(window=30, min_periods=10).std()
df['rv_30d_ann'] = annualize_vol(df['rv_30d'])

# VIX: 30-day rolling mean (to remove day noise)
df['vix_30d'] = df['india_vix'].rolling(window=30, min_periods=5).mean()

# Volatility gap: VIX - realized vol (both on approx same annualized scale)
# Convert VIX (percent) to decimal same scale as rv_30d_ann if needed: VIX from index typically in percent points (e.g., 12 = 12%)
# Our rv_30d_ann is in decimal (e.g., 0.12). If vix is in percent, divide by 100.
# Guessing vix is in percent (12 = 12%), so convert:
df['vix_pct'] = df['india_vix']  # keep original descriptor
# If vix values are > 5 (likely percent), convert to decimal for comparison:
df['vix_ann_est'] = np.where(df['vix_pct'] > 5, df['vix_pct']/100.0, df['vix_pct'])  # crude
# ensure rv_30d_ann is not NaN before subtracting
df['rv_30d_ann_filled'] = df['rv_30d_ann'].fillna(method='ffill', limit=5)
df['vix_minus_rv'] = df['vix_ann_est'] - df['rv_30d_ann_filled']

# --- Yield curve metrics ---
print("Computing yield-curve metrics (slope, curvature)...")
# slope = 10Y - 1Y (in percentage points)
df['yield_slope_10y_1y'] = df['10Y'] - df['1Y']
# curvature proxy: 2*5Y - 10Y - 1Y (common measure)
df['yield_curvature'] = 2*df['5Y'] - df['10Y'] - df['1Y']
# alternative mid-short diff
df['yield_5y_3y'] = df['5Y'] - df['3Y']

# --- Liquidity stress markers (use z-score computed earlier) ---
# Already have liquidity_z_30d in laf file; create interaction terms
print("Creating liquidity-volatility interaction signals...")
df['vol_liq_interaction'] = df['rv_30d_ann'] * df['liquidity_z_30d']  # high vol * positive z => stress

# regime seed flags: simple thresholding as visualization seeds
df['high_vol_flag'] = (df['rv_30d_ann'] > df['rv_30d_ann'].quantile(0.75)).astype(int)
df['liquidity_stress_flag'] = (df['liquidity_z_30d'].abs() > 1.0).astype(int)  # |z|>1 indicates unusual

# volatility clustering measure: 30-day vol divided by 90-day vol
df['rv_90d'] = df['return'].rolling(window=90, min_periods=15).std()
df['rv_90d_ann'] = annualize_vol(df['rv_90d'])
df['vol_cluster_ratio'] = df['rv_30d_ann'] / df['rv_90d_ann']

# --- housekeeping: trim an initial warmup period where rolling stats are NaN ---
df = df.dropna(subset=['adj_close']).copy()
# Optionally drop starting rows where rv_30d_ann is NaN (insufficient history)
df = df[df.index >= (df.index.min() + pd.Timedelta(days=60))]

# --- save combined dataset ---
print("Saving combined dataset ->", OUT_FILE)
df.to_csv(OUT_FILE, float_format="%.6f")

# --- quick plots for visualization seeds ---
print("Creating diagnostic plots in", PLOTS_DIR)

# 1) Price (NIFTY) + Realized Vol (ann) + VIX (pct)
plt.figure(figsize=(12,6))
ax1 = plt.gca()
ax1.plot(df.index, df['adj_close'], label='NIFTY Adj Close')
ax1.set_ylabel('NIFTY', fontsize=10)
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.plot(df.index, df['rv_30d_ann'], label='Realized Vol (30d, ann)', linestyle='--')
ax2.plot(df.index, df['vix_ann_est'], label='VIX (est ann)', linestyle=':')
ax2.set_ylabel('Volatility', fontsize=10)
ax2.legend(loc='upper right')
plt.title('NIFTY Price vs Realized Vol & VIX')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ts_prices_vol_vix.png", dpi=150)
plt.close()

# 2) Volatility and Liquidity series
plt.figure(figsize=(12,5))
plt.plot(df.index, df['rv_30d_ann'], label='Realized Vol (30d, ann)')
plt.plot(df.index, df['liquidity_z_30d'], label='Liquidity z-score (30d)')
plt.title('Realized Volatility vs Liquidity z-score')
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "vol_liquidity_timeseries.png", dpi=150)
plt.close()

# 3) Yield curve metrics
plt.figure(figsize=(10,4))
plt.plot(df.index, df['yield_slope_10y_1y'], label='Yield slope (10Y-1Y)')
plt.plot(df.index, df['yield_curvature'], label='Yield curvature (2*5Y-10Y-1Y)')
plt.title('Yield Curve Metrics')
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "yield_curve_metrics.png", dpi=150)
plt.close()

print("Done. Files created:")
print(" -", OUT_FILE)
for p in PLOTS_DIR.iterdir():
    print(" -", p)