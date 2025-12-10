# src/hmm_regime_detection.py
"""
STEP 5: Hidden Markov Model (HMM) Regime Detection (Vol + Liquidity, ensemble)
FINAL corrected version: ensures all state arrays are converted to Series indexed by training dates
and reindexed into the full output dataframe to avoid length-mismatch errors.

Outputs:
 - data/clean/market_dataset_with_regimes.csv
 - data/clean/hmm_model_stats.csv
 - plots/hmm_regimes_2_states.png
 - plots/hmm_regimes_3_states.png
 - plots/hmm_regimes_4_states.png
 - plots/hmm_regime_ensemble.png

Run:
    python src/hmm_regime_detection.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings("ignore")

# --- paths ---
CLEAN_DIR = Path("data/clean")
OUT_DATA = CLEAN_DIR / "market_dataset_with_regimes.csv"
MODEL_STATS = CLEAN_DIR / "hmm_model_stats.csv"
PLOTS = Path("plots")
PLOTS.mkdir(parents=True, exist_ok=True)

# --- Parameters ---
RANDOM_RESTARTS = 5
N_STATES_LIST = [2, 3, 4]
MAX_ITER = 200

# --- Load dataset ---
df = pd.read_csv(CLEAN_DIR / "market_dataset.csv", parse_dates=[0], index_col=0)
print("Loaded dataset:", df.shape, "Date range:", df.index.min(), "to", df.index.max())

# --- Feature set: Option B (Vol + Liquidity) ---
feat_names = ['rv_30d_ann', 'vix_ann_est', 'liquidity_z_30d', 'vol_liq_interaction']
missing = [f for f in feat_names if f not in df.columns]
if missing:
    raise KeyError(f"Missing required features: {missing}. Run STEP 3 first.")

# Prepare training data (drop rows with NaNs in features)
data = df[feat_names].copy()
data = data.dropna()
print("Training on features shape:", data.shape)

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(data.values)
dates = data.index

# AIC/BIC helper
def compute_aic_bic(model, X):
    n_samples, n_features = X.shape
    n_states = model.n_components
    means_params = n_states * n_features
    covar_params = n_states * (n_features * (n_features + 1) / 2.0)
    trans_params = n_states * (n_states - 1)
    start_params = n_states - 1
    k = int(means_params + covar_params + trans_params + start_params)
    logL = model.score(X)
    aic = 2 * k - 2 * logL
    bic = np.log(n_samples) * k - 2 * logL
    return float(aic), float(bic), float(logL)

# Fit HMMs
results = []
models_by_state = {n: [] for n in N_STATES_LIST}
np.random.seed(42)
print("Fitting HMMs...")
for n_states in N_STATES_LIST:
    for restart in range(RANDOM_RESTARTS):
        model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=MAX_ITER, random_state=restart, verbose=False)
        try:
            model.fit(X)
            aic, bic, logL = compute_aic_bic(model, X)
            states = model.predict(X)
            models_by_state[n_states].append({'model': model, 'aic': aic, 'bic': bic, 'logL': logL, 'states': states, 'restart': restart})
            results.append({'n_states': n_states, 'restart': restart, 'aic': aic, 'bic': bic, 'logL': logL})
            print(f" Fitted n={n_states} restart={restart}  logL={logL:.1f}  AIC={aic:.1f}  BIC={bic:.1f}")
        except Exception as e:
            print(f"  Fit failed for n={n_states} restart={restart}: {e}")

pd.DataFrame(results).to_csv(MODEL_STATS, index=False)
print("Saved model stats ->", MODEL_STATS)

# Choose best per n_states by BIC
best_models = {}
for n_states in N_STATES_LIST:
    candidates = models_by_state.get(n_states, [])
    if not candidates:
        continue
    best = sorted(candidates, key=lambda x: x['bic'])[0]
    best_models[n_states] = best
    print(f"Best model for {n_states} states: restart={best['restart']} BIC={best['bic']:.1f} logL={best['logL']:.1f}")

# Build per-model prediction Series (index=training dates), relabel states by volatility ranking
pred_series = {}     # key: 'pred_{n_states}' -> pd.Series indexed by dates
model_info = []
for n_states, best in best_models.items():
    model = best['model']
    states = best['states']
    # relabel states by rv mean: compute means in original scale
    means_scaled = model.means_
    means_orig = scaler.inverse_transform(means_scaled)
    rv_means = means_orig[:, 0]
    order = np.argsort(rv_means)
    mapping = {old: new for new, old in enumerate(order)}
    relabeled = np.array([mapping[s] for s in states])
    s_series = pd.Series(relabeled, index=dates, name=f"pred_{n_states}")
    pred_series[f"pred_{n_states}"] = s_series
    model_info.append({'n_states': n_states, 'bic': best['bic'], 'logL': best['logL']})
    print(f"Prepared pred Series for n={n_states}")

# Build DataFrame of predictions (indexed by training dates)
if pred_series:
    pred_df = pd.concat(pred_series.values(), axis=1)
    pred_df.columns = list(pred_series.keys())
else:
    raise RuntimeError("No models successfully fitted.")

# Ensemble: majority vote with tie-break by lowest BIC preference
bic_order = sorted(model_info, key=lambda x: x['bic'])
ensemble_list = []
for idx, row in pred_df.iterrows():
    counts = row.value_counts()
    top_count = counts.max()
    top_labels = counts[counts == top_count].index.tolist()
    if len(top_labels) == 1:
        ensemble_list.append(int(top_labels[0]))
    else:
        chosen = None
        for m in bic_order:
            col = f"pred_{m['n_states']}"
            if col in pred_df.columns:
                val = row[col]
                if val in top_labels:
                    chosen = int(val); break
        if chosen is None:
            chosen = int(top_labels[0])
        ensemble_list.append(chosen)
ensemble_series = pd.Series(ensemble_list, index=pred_df.index, name='regime_ensemble')

# Merge back into full df with alignment (reindex predictions to full df index)
outdf = df.copy()
# Add per-model predictions (reindexed)
for col in pred_df.columns:
    outdf[col] = pred_df[col].reindex(outdf.index)
# Add ensemble (reindexed)
outdf['regime_ensemble'] = ensemble_series.reindex(outdf.index)
# Add named per-model regime columns
for n_states in best_models:
    series = pred_df[f"pred_{n_states}"]
    outdf[f"regime_hmm_{n_states}"] = series.reindex(outdf.index)

# Save
outdf.to_csv(OUT_DATA, float_format="%.6f")
print("Saved dataset with regimes ->", OUT_DATA)

# Plotting: price vs regimes
orig = pd.read_csv(CLEAN_DIR / "market_dataset.csv", parse_dates=[0], index_col=0)
price = orig['adj_close'].reindex(outdf.index)

def plot_regimes(price_series, regime_series, title, outpng):
    plt.figure(figsize=(12,4))
    unique = sorted(pd.Series(regime_series).dropna().unique())
    cmap = plt.get_cmap('tab10')
    for r in unique:
        mask = regime_series == r
        plt.plot(price_series.index[mask], price_series[mask], '.', ms=3, label=f"regime {r}", color=cmap(int(r)%10))
    plt.plot(price_series.index, price_series, alpha=0.3, linewidth=0.5)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()
    print("Saved plot ->", outpng)

for n_states in best_models:
    plot_regimes(price, outdf[f"regime_hmm_{n_states}"], f"HMM regimes (n={n_states})", PLOTS / f"hmm_regimes_{n_states}_states.png")
plot_regimes(price, outdf['regime_ensemble'], "HMM regime ensemble (majority vote)", PLOTS / "hmm_regime_ensemble.png")

print("STEP 5 complete. Models trained and ensemble created.")
print("Model summary:\n", pd.DataFrame(model_info))