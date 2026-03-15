"""
Benchmark: Causal elasticity (CausalForestDML) vs naive OLS price coefficient.

The problem: insurance renewal data contains strong selection bias. High-risk
customers get higher prices (set by the technical model) AND tend to have lower
renewal rates regardless of price (fewer alternatives, less shopping behaviour).
A naive OLS regression of log(renewal_rate) on log(price_change) produces a
biased elasticity because risk-driven price variation is not random — it's
correlated with unobserved customer propensity to renew.

The RenewalElasticityEstimator uses CausalForestDML with CatBoost nuisance
models and cross-fitting to isolate the causal price effect. It leverages the
exogenous component of price variation (e.g., A/B test overrides, quarterly rate
index changes) to identify the true elasticity.

Setup:
- Synthetic UK motor renewal data with known heterogeneous elasticities by NCD
- True overall ATE: approximately -2.0 (semi-elasticity, log-log specification)
- True GATEs vary by NCD band: 0 NCD = -3.5, 5 NCD = -1.0
- Naive OLS: regress renewed on log_price_change (no controls)
- Biased OLS: regress with partial controls (excludes key confounders)
- Causal: RenewalElasticityEstimator (CausalForestDML, CatBoost, 3-fold)

Expected output:
- OLS overestimates |elasticity| due to selection bias
- GATE estimates by NCD band: naive is flat, causal recovers heterogeneity
- Causal estimate is closer to true ATE with valid confidence interval

Run:
    python benchmarks/benchmark.py
"""

from __future__ import annotations

import sys
import time
import warnings

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: CausalForestDML vs naive OLS elasticity (insurance-elasticity)")
print("=" * 70)
print()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    from insurance_elasticity import RenewalElasticityEstimator
    from insurance_elasticity.data import (
        make_renewal_data,
        true_gate_by_ncd,
        true_gate_by_age,
    )
    print("insurance-elasticity imported OK")
except ImportError as e:
    print(f"ERROR: Could not import insurance-elasticity: {e}")
    print("Install with: pip install insurance-elasticity")
    sys.exit(1)

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Generate data
# ---------------------------------------------------------------------------

N = 6_000    # Smaller for speed; production runs use 50k+
SEED = 42

print(f"\nGenerating {N:,} synthetic UK motor renewal records...")
df = make_renewal_data(n=N, seed=SEED, price_variation_sd=0.08)

print(f"  Overall renewal rate:    {df['renewed'].mean():.1%}")
print(f"  Mean log price change:   {df['log_price_change'].mean():.4f}")
print(f"  Std log price change:    {df['log_price_change'].std():.4f}")
print()

# True GATEs for comparison
true_ncd = true_gate_by_ncd(df)
true_age = true_gate_by_age(df)

print("True elasticity by NCD band (from DGP):")
print(f"  {'NCD':>5} {'True elast':>12} {'N':>7}")
for row in true_ncd.iter_rows(named=True):
    print(f"  {row['ncd_years']:>5} {row['true_elasticity_mean']:>12.3f} {row['n']:>7,}")
print()

# ---------------------------------------------------------------------------
# Naive approach 1: OLS, no controls
# ---------------------------------------------------------------------------

print("NAIVE APPROACH 1: OLS regression (no controls)")
print("-" * 55)

y = df["renewed"].to_numpy().astype(float)
d = df["log_price_change"].to_numpy()
X_const = np.column_stack([np.ones(N), d])
# OLS: (X'X)^{-1} X'y
coef_ols, _, _, _ = np.linalg.lstsq(X_const, y, rcond=None)
ols_elasticity = float(coef_ols[1])

print(f"  OLS elasticity (no controls): {ols_elasticity:.4f}")
print(f"  Selection bias direction: OLS conflates risk-driven price with demand response")
print()

# ---------------------------------------------------------------------------
# Naive approach 2: OLS with partial controls
# ---------------------------------------------------------------------------

print("NAIVE APPROACH 2: OLS regression (partial controls: age, ncd_years)")
print("-" * 55)

age_arr = df["age"].to_numpy().astype(float) / 80.0
ncd_arr = df["ncd_years"].to_numpy().astype(float) / 5.0
X_partial = np.column_stack([np.ones(N), d, age_arr, ncd_arr])
coef_partial, _, _, _ = np.linalg.lstsq(X_partial, y, rcond=None)
partial_elasticity = float(coef_partial[1])

print(f"  OLS elasticity (partial controls): {partial_elasticity:.4f}")
print(f"  Missing: channel, region, vehicle_group — still biased")
print()

# ---------------------------------------------------------------------------
# OLS with full controls
# ---------------------------------------------------------------------------

print("NAIVE APPROACH 3: OLS regression (full controls, one-hot encoded)")
print("-" * 55)

import pandas as pd

df_pd = df.to_pandas()
confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]
X_full = pd.get_dummies(df_pd[confounders], drop_first=True).astype(float)
X_full_arr = np.column_stack([np.ones(N), d, X_full.values])
coef_full, _, _, _ = np.linalg.lstsq(X_full_arr, y, rcond=None)
full_ols_elasticity = float(coef_full[1])

print(f"  OLS elasticity (full OHE controls): {full_ols_elasticity:.4f}")
print(f"  OLS assumes: linear treatment effect, no treatment-covariate interaction")
print()

# ---------------------------------------------------------------------------
# Causal approach: RenewalElasticityEstimator
# ---------------------------------------------------------------------------

print("CAUSAL APPROACH: RenewalElasticityEstimator (CausalForestDML + CatBoost)")
print("-" * 55)
print()
print("  CausalForestDML with CatBoost nuisance models, 3-fold cross-fitting")
print("  (production: use 5 folds and n_estimators >= 200)")
print()

est = RenewalElasticityEstimator(
    cate_model="causal_forest",
    binary_outcome=True,
    n_folds=3,
    n_estimators=60,   # divisible by n_folds*2=6; min for speed
    catboost_iterations=100,
    random_state=SEED,
)

t0 = time.time()
est.fit(
    df,
    outcome="renewed",
    treatment="log_price_change",
    confounders=confounders,
)
fit_time = time.time() - t0

ate, lb, ub = est.ate()

print(f"  Causal ATE:    {ate:.4f}  (true mean ≈ -2.0)")
print(f"  95% CI:        [{lb:.4f}, {ub:.4f}]")
ci_covers_true = lb <= -2.0 <= ub
print(f"  CI covers -2.0: {ci_covers_true}")
print(f"  Fit time:      {fit_time:.1f}s")
print()

# GATE by NCD band
gate_ncd = est.gate(df, by="ncd_years")
print("  GATE by NCD band vs true elasticity:")
print(f"  {'NCD':>5} {'True elast':>12} {'Causal GATE':>14} {'CI lower':>10} {'CI upper':>10}")
print(f"  {'-'*5} {'-'*12} {'-'*14} {'-'*10} {'-'*10}")

# Merge with true values
true_ncd_dict = {row['ncd_years']: row['true_elasticity_mean'] for row in true_ncd.iter_rows(named=True)}
for row in gate_ncd.iter_rows(named=True):
    ncd = row['ncd_years']
    true_val = true_ncd_dict.get(ncd, float("nan"))
    print(
        f"  {ncd:>5} {true_val:>12.3f} {row['elasticity']:>14.4f}"
        f" {row['ci_lower']:>10.4f} {row['ci_upper']:>10.4f}"
    )
print()

# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------

# True ATE from DGP
true_ate = float(df["true_elasticity"].mean())

print("COMPARISON SUMMARY")
print("=" * 70)
print(f"{'Estimator':<35} {'Elasticity':>12} {'Bias':>10} {'Has CI':>8} {'GATE':>8}")
print("-" * 70)
rows = [
    ("OLS (no controls)",          ols_elasticity,     False, False),
    ("OLS (partial controls)",     partial_elasticity, False, False),
    ("OLS (full OHE controls)",    full_ols_elasticity,False, False),
    ("CausalForestDML",            ate,                True,  True),
]
for label, elast, has_ci, has_gate in rows:
    bias = elast - true_ate
    print(f"  {label:<33} {elast:>12.4f} {bias:>+10.4f} {'Yes' if has_ci else 'No':>8} {'Yes' if has_gate else 'No':>8}")
print(f"  {'True ATE (from DGP)':<33} {true_ate:>12.4f} {'0.0000':>10} {'—':>8} {'—':>8}")
print()

print("KEY FINDINGS")
print(f"  True ATE: {true_ate:.4f}")
print(f"  OLS (no controls) bias:    {ols_elasticity - true_ate:+.4f}")
print(f"  OLS (full controls) bias:  {full_ols_elasticity - true_ate:+.4f}")
print(f"  Causal estimate bias:      {ate - true_ate:+.4f}")
print(f"  Bias reduction vs OLS:     {(1 - abs(ate - true_ate) / abs(ols_elasticity - true_ate)) * 100:.0f}%")
print()
print("  OLS conflates two things:")
print("  1. The causal effect of price on renewal (what we want)")
print("  2. The correlation between risk-driven price and renewal propensity")
print()
print("  CausalForestDML also recovers heterogeneous effects by NCD band,")
print("  which OLS cannot do without pre-specified interactions. These")
print("  per-segment elasticities are the input to portfolio optimisation.")
print()

elapsed = time.time() - BENCHMARK_START
print(f"Benchmark completed in {elapsed:.1f}s")
