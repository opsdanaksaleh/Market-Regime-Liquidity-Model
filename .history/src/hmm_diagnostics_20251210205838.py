# src/hmm_diagnostics.py
"""
Compute regime diagnostics from market_dataset_with_regimes.csv
Run: python src/hmm_diagnostics.py
"""

import pandas as pd
import numpy as np

PATH = "data/clean/market_dataset_with_regimes.csv"
df = pd.read_csv(PATH, parse_dates=[0], index_col=0)
print("Loaded:", PATH, "shape:", df.shape)

# column names:
print("Columns available:", list(df.columns))

# required columns
cols_needed = ['regime_ensemble','adj_close','return','rv_30d_ann','vix_ann_est','liquidity_z_30d']
for c in cols_needed:
    if c not in df.columns:
        print(f"WARNING: column {c} missing in dataset. Diagnostics limited.")

# keep rows with regime label
rdf = df.dropna(subset=['regime_ensemble']).copy()
rdf['regime'] = rdf['regime_ensemble'].astype(int)

# Basic per-regime stats
group = rdf.groupby('regime')
stats = group.agg(
    count = ('regime', 'size'),
    freq = ('regime', lambda s: len(s)/len(rdf)),
    mean_return = ('return', 'mean'),
    std_return = ('return', 'std'),
    mean_rv = ('rv_30d_ann', 'mean'),
    mean_vix = ('vix_ann_est', 'mean'),
    mean_liq_z = ('liquidity_z_30d', 'mean'),
    mean_price = ('adj_close', 'mean')
).reset_index().sort_values('regime')
print("\nPer-regime summary:\n", stats.to_string(index=False))

# Regime run-lengths and expected duration
runs = []
current = None
current_start = None
for i, (idx, row) in enumerate(rdf.iterrows()):
    r = row['regime']
    if current is None:
        current = r
        current_start = idx
        length = 1
    else:
        if r == current:
            length += 1
        else:
            runs.append({'regime': int(current), 'length': length})
            current = r
            length = 1
# append last
if current is not None:
    runs.append({'regime': int(current), 'length': length})
runs_df = pd.DataFrame(runs)
dur_stats = runs_df.groupby('regime')['length'].agg(['count','mean','median','max']).rename(columns={'mean':'avg_duration_days'})
print("\nRun-length / duration summary (business days):\n", dur_stats.to_string())

# Transition matrix (empirical)
# Build transition counts
regimes = sorted(rdf['regime'].unique())
trans_counts = pd.DataFrame(0, index=regimes, columns=regimes)
prev = None
for idx, r in rdf['regime'].items():
    if prev is not None:
        trans_counts.loc[prev, r] += 1
    prev = r
# convert to probabilities (row-normalized)
trans_prob = trans_counts.div(trans_counts.sum(axis=1).replace(0,1), axis=0)
print("\nTransition probability matrix (rows = from, columns = to):\n", trans_prob.fillna(0).to_string())

# Save diagnostics
out_diag = "data/clean/hmm_regime_diagnostics.csv"
with pd.ExcelWriter("data/clean/hmm_regime_diagnostics.xlsx") as w:
    stats.to_excel(w, sheet_name="per_regime", index=False)
    dur_stats.to_excel(w, sheet_name="durations")
    trans_prob.to_excel(w, sheet_name="transitions")
print("\nSaved diagnostics to: data/clean/hmm_regime_diagnostics.xlsx")