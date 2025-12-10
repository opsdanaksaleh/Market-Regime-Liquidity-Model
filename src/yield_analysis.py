# src/yield_analysis.py
"""
STEP 4: Yield Curve Analysis

Produces:
 - plots/yield_curve_snapshots.png      (several curves on same figure for comparison)
 - plots/yield_curve_heatmap.png        (tenor x time heatmap)
 - plots/yield_slope_curvature.png     (time series of slope & curvature)
 - data/clean/yield_metrics.csv        (date, slope, curvature, slope_30d_ma, curvature_30d_ma)

Run:
    python src/yield_analysis.py
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# paths
CLEAN_DIR = Path("data/clean")
OUT_METRICS = CLEAN_DIR / "yield_metrics.csv"
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# helper: ensure file exists
gsec_file = CLEAN_DIR / "gsec_tenor_clean.csv"
if not gsec_file.exists():
    raise FileNotFoundError(f"Missing G-Sec cleaned file: {gsec_file}. Run Step 2.D first.")

# load
gsec = pd.read_csv(gsec_file, index_col=0, parse_dates=True)
gsec = gsec.sort_index()
# keep only common tenor columns and convert to numeric
tenors = ['1Y','3Y','5Y','10Y']
for t in tenors:
    if t not in gsec.columns:
        raise KeyError(f"Tenor column '{t}' not found in {gsec_file}. Columns: {list(gsec.columns)}")
gsec = gsec[tenors].apply(pd.to_numeric, errors='coerce')

# compute slope and curvature
gsec['yield_slope_10y_1y'] = gsec['10Y'] - gsec['1Y']
gsec['yield_curvature'] = 2*gsec['5Y'] - gsec['10Y'] - gsec['1Y']

# rolling summaries
gsec['slope_ma30'] = gsec['yield_slope_10y_1y'].rolling(30, min_periods=5).mean()
gsec['curv_ma30'] = gsec['yield_curvature'].rolling(30, min_periods=5).mean()

# save metrics
metrics = gsec[['yield_slope_10y_1y','yield_curvature','slope_ma30','curv_ma30']].dropna(how='all')
metrics.to_csv(OUT_METRICS, float_format="%.6f")
print("Saved yield metrics ->", OUT_METRICS)

# ---- Plot 1: yield curve snapshots (select a few dates) ----
# choose snapshot dates: earliest, median, recent, and a stress date (highest 10Y)
dates = []
dates.append(gsec.index.min())
dates.append(gsec.index[int(len(gsec.index)/2)])
dates.append(gsec.index.max())

# find a date with highest 10Y (stress snapshot)
if gsec['10Y'].dropna().size:
    max10 = gsec['10Y'].idxmax()
    if max10 not in dates:
        dates.insert(2, max10)

plt.figure(figsize=(10,6))
for d in dates:
    row = gsec.loc[d, tenors]
    if row.isna().all():
        continue
    plt.plot([1,3,5,10], row.values, marker='o', label=f"{d.date()}")
plt.xlabel("Maturity (Years)")
plt.ylabel("Yield (%)")
plt.title("Yield Curve Snapshots (selected dates)")
plt.xticks([1,3,5,10])
plt.grid(axis='y', linestyle=':', linewidth=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "yield_curve_snapshots.png", dpi=150)
plt.close()
print("Saved plot ->", PLOTS_DIR / "yield_curve_snapshots.png")

# ---- Plot 2: heatmap of yields (tenor x time) ----
# prepare matrix: time x tenor
heat = gsec[tenors].dropna(how='all')
# reduce density for plotting if very long (sample weekly to keep image manageable)
if len(heat) > 1500:
    heat_plot = heat.resample('W').last()
else:
    heat_plot = heat

# build 2D array: rows=time, columns=tenors
arr = heat_plot.values.T  # shape (len(tenors), len(time))
fig, ax = plt.subplots(figsize=(12,4))
# use imshow with aspect='auto' and origin='lower' to show maturities on y-axis
im = ax.imshow(arr, aspect='auto', origin='lower', interpolation='nearest')
ax.set_yticks(range(len(tenors)))
ax.set_yticklabels(tenors)
# x-axis date labels
xticks_idx = np.linspace(0, arr.shape[1]-1, 8, dtype=int)
xticks = [heat_plot.index[i].date() for i in xticks_idx]
ax.set_xticks(xticks_idx)
ax.set_xticklabels([str(x) for x in xticks], rotation=30)
ax.set_title("Yield heatmap (tenors on y-axis, time on x-axis)")
fig.colorbar(im, ax=ax, label='Yield (%)')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "yield_curve_heatmap.png", dpi=150)
plt.close()
print("Saved plot ->", PLOTS_DIR / "yield_curve_heatmap.png")

# ---- Plot 3: slope & curvature time series ----
plt.figure(figsize=(12,4))
plt.plot(gsec.index, gsec['yield_slope_10y_1y'], label='Slope (10Y - 1Y)')
plt.plot(gsec.index, gsec['yield_curvature'], label='Curvature (2*5Y - 10Y - 1Y)', alpha=0.8)
plt.plot(gsec.index, gsec['slope_ma30'], label='Slope MA30', linestyle='--', alpha=0.6)
plt.plot(gsec.index, gsec['curv_ma30'], label='Curv MA30', linestyle='--', alpha=0.6)
plt.axhline(0, color='k', linewidth=0.5)
plt.legend()
plt.title("Yield Curve Slope and Curvature (time series)")
plt.xlabel("Date")
plt.ylabel("Percentage points")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "yield_slope_curvature.png", dpi=150)
plt.close()
print("Saved plot ->", PLOTS_DIR / "yield_slope_curvature.png")

# ---- Additional: distribution summary for slope & curvature ----
desc = gsec[['yield_slope_10y_1y','yield_curvature']].describe().T
print("\nYield slope & curvature summary:\n", desc.to_string())

print("\nSTEP 4 complete. Plots & metrics saved in 'plots/' and 'data/clean/'.")