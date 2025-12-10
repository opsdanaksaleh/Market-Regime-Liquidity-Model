# src/hmm_regime_detection.py
"""
STEP 5: Hidden Markov Model (HMM) Regime Detection (Vol + Liquidity, ensemble)

Outputs:
 - data/clean/market_dataset_with_regimes.csv
 - data/clean/hmm_model_stats.csv
 - plots/hmm_regimes_2_states.png
 - plots/hmm_regimes_3_states.png
 - plots/hmm_regimes_4_states.png
 - plots/hmm_regime_ensemble.png

Run:
    python src/hmm_regime_detection.py

Notes:
 - Requires: numpy, pandas, matplotlib, scikit-learn, hmmlearn
   pip install numpy pandas matplotlib scikit-learn hmmlearn
 - This script fits multiple Gaussian HMMs, computes AIC/BIC, then forms an ensemble by majority vote.
 - Beginner-friendly: prints step-by-step explanations and saves all outputs.
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
RANDOM_RESTARTS = 5    # number of random initializations per n_states
N_STATES_LIST = [2,3,4]
MAX_ITER = 200

# --- Load dataset ---
df = pd.read_csv(CLEAN_DIR / "market_dataset.csv", parse_dates=[0], index_col=0)
print("Loaded dataset:", df.shape, "Date range:", df.index.min(), "to", df.index.max())

# --- Feature set: Option B (Vol + Liquidity) ---
feat_names = ['rv_30d_ann', 'vix_ann_est', 'liquidity_z_30d', 'vol_liq_interaction']
for f in feat_names:
    if f not in df.columns:
        raise KeyError(f"Required feature '{f}' not found in dataset. Run STEP 3 first.")

# keep only rows with non-null required features (HMM needs no NaNs)
data = df[feat_names].copy()
data = data.dropna()
print("Features used (rows kept):", data.shape)

# --- Scaling ---
scaler = StandardScaler()
X = scaler.fit_transform(data.values)
dates = data.index

# --- Helper: AIC/BIC for HMM (Gaussian) ---
def compute_aic_bic(model, X):
    """
    Compute AIC and BIC for a fitted Gaussian HMM.
    k = number of parameters:
      - for GaussianHMM: means (n_states * n_features) + covars (n_states * n_features*(n_features+1)/2) +
        transition matrix (n_states*(n_states-1)) + initial state probs (n_states-1)
    """
    n_samples, n_features = X.shape
    n_states = model.n_components
    # emission params: means + covars (we assume full covars)
    means_params = n_states * n_features
    covar_params = n_states * (n_features * (n_features + 1) / 2.0)
    trans_params = n_states * (n_states - 1)
    start_params = n_states - 1
    k = int(means_params + covar_params + trans_params + start_params)
    logL = model.score(X)
    aic = 2 * k - 2 * logL
    bic = np.log(n_samples) * k - 2 * logL
    return float(aic), float(bic), float(logL)

# --- Fit multiple HMMs and collect stats ---
results = []
models_by_state = {n: [] for n in N_STATES_LIST}
np.random.seed(42)

print("Fitting HMMs (this may take a short while)...")
for n_states in N_STATES_LIST:
    for restart in range(RANDOM_RESTARTS):
        # instantiate Gaussian HMM with full covariance
        model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=MAX_ITER, random_state=restart, verbose=False)
        try:
            model.fit(X)
            aic, bic, logL = compute_aic_bic(model, X)
            # decode hidden states with Viterbi
            states = model.predict(X)
            models_by_state[n_states].append({'model': model, 'aic': aic, 'bic': bic, 'logL': logL, 'states': states, 'restart': restart})
            results.append({'n_states': n_states, 'restart': restart, 'aic': aic, 'bic': bic, 'logL': logL})
            print(f" Fitted n={n_states} restart={restart}  logL={logL:.1f}  AIC={aic:.1f}  BIC={bic:.1f}")
        except Exception as e:
            print(f"  Fit failed for n={n_states} restart={restart}: {e}")

# Save raw model stats
pd.DataFrame(results).to_csv(MODEL_STATS, index=False)
print("Saved model stats ->", MODEL_STATS)

# --- For each n_states, pick best model by BIC ---
best_models = {}
for n_states in N_STATES_LIST:
    candidates = models_by_state.get(n_states, [])
    if not candidates:
        continue
    best = sorted(candidates, key=lambda x: x['bic'])[0]
    best_models[n_states] = best
    print(f"Best model for {n_states} states: restart={best['restart']} BIC={best['bic']:.1f} logL={best['logL']:.1f}")

# --- Build ensemble: majority vote among best models for each n_states (if available) ---
# We'll use predictions from best 2-state, 3-state, and 4-state models (if present)
predictions = []
model_info = []
for n_states in N_STATES_LIST:
    if n_states in best_models:
        s = best_models[n_states]['states']
        # Normalize state labels by sorting states by mean volatility to give interpretability
        # compute mean rv_30d_ann per state (use model.means_ on scaled space -> transform back)
        model = best_models[n_states]['model']
        # compute state means in original feature space
        means_scaled = model.means_  # shape n_states x n_features
        # convert to original scale
        means_orig = scaler.inverse_transform(means_scaled)
        # take rv index = 0
        rv_means = means_orig[:, 0]
        # rank states by rv_means and relabel from 0 (lowest vol) to n_states-1 (highest vol)
        order = np.argsort(rv_means)
        ranking = {old: new for new, old in enumerate(order)}
        relabeled = np.array([ranking[x] for x in s])
        predictions.append(relabeled)
        model_info.append({'n_states': n_states, 'bic': best_models[n_states]['bic'], 'logL': best_models[n_states]['logL']})
    else:
        print(f"No best model for {n_states} states - skipping in ensemble.")

# combine predictions into DataFrame aligned with dates
pred_df = pd.DataFrame(index=dates, data=np.column_stack(predictions) if predictions else np.empty((len(dates),0)),
                       columns=[f"pred_{n}" for n in best_models.keys()])

# majority vote per date (axis=1)
if pred_df.shape[1] == 0:
    raise RuntimeError("No HMM models fitted successfully. Aborting.")
vote_series = pred_df.mode(axis=1).iloc[:,0]  # mode() returns the most frequent label
# if tie (multiple modes) mode() chooses smallest; to break ties by BIC preference, we'll handle ties explicitly:
ties = pred_df.apply(lambda row: row.value_counts().nlargest(2).iloc[1] if row.value_counts().nlargest(2).shape[0] > 1 and row.value_counts().nlargest(2).iloc[0] == row.value_counts().nlargest(2).iloc[1] else np.nan, axis=1)
# For rows where there is a tie (ties not NaN), pick model whose BIC is smallest among competing values
# Simpler approach: if tie present, choose the label predicted by model with lowest BIC (prefer simpler ones)
# Build vote with custom tie-breaker
final_votes = []
bic_order = sorted(model_info, key=lambda x: x['bic'])  # ascending bic (lower better)
for idx, row in pred_df.iterrows():
    counts = row.value_counts()
    top = counts.index.tolist()
    if len(top) == 1:
        final_votes.append(top[0])
    else:
        # tie among labels; pick label from model with lowest BIC
        chosen = None
        for m in bic_order:
            colname = f"pred_{m['n_states']}"
            val = row[colname]
            if val in top:
                chosen = int(val); break
        if chosen is None:
            chosen = int(top[0])
        final_votes.append(chosen)
ensemble_series = pd.Series(final_votes, index=dates, name='regime_ensemble')

# --- Add regime series to original data frame (aligned) ---
outdf = df.copy()
outdf = outdf.join(pd.DataFrame(pred_df, index=dates))
outdf['regime_ensemble'] = ensemble_series

# Also add best-model regimes for reference
for n_states in best_models:
    outdf[f"regime_hmm_{n_states}"] = best_models[n_states]['states']

# Save results
outdf.to_csv(OUT_DATA, float_format="%.6f")
print("Saved dataset with regimes ->", OUT_DATA)

# --- Plot regimes over time (price + regime colors) ---
# We'll plot NIFTY price (from original market_dataset) against ensemble regimes
orig = pd.read_csv(CLEAN_DIR / "market_dataset.csv", parse_dates=[0], index_col=0)
price = orig['adj_close'].reindex(outdf.index)

def plot_regimes(series_price, regime_series, title, outpng):
    plt.figure(figsize=(12,4))
    unique_regimes = sorted(regime_series.dropna().unique())
    cmap = plt.get_cmap('tab10')
    for r in unique_regimes:
        mask = regime_series == r
        plt.plot(series_price.index[mask], series_price[mask], '.', ms=3, label=f"regime {r}", color=cmap(int(r) % 10))
    plt.plot(series_price.index, series_price, alpha=0.3, linewidth=0.5)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()
    print("Saved regime plot ->", outpng)

# plots per-model and ensemble
for n_states in best_models:
    plot_regimes(price, pd.Series(best_models[n_states]['states'], index=dates), f"HMM regimes (n={n_states})", PLOTS / f"hmm_regimes_{n_states}_states.png")

plot_regimes(price, outdf['regime_ensemble'], "HMM regime ensemble (majority vote)", PLOTS / "hmm_regime_ensemble.png")

print("STEP 5 complete. Models trained, ensemble created, outputs saved.")
print("Model summary:")
print(pd.DataFrame(model_info))