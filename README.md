# insurance-elasticity

Causal price elasticity estimation and FCA PS21/5-compliant renewal pricing
optimisation for UK personal lines insurance.

---

## The problem

UK motor and home insurance pricing teams want to know one thing: if we increase
this customer's renewal price by 10%, how much does their probability of renewing
fall?

The naive answer — run a logistic regression of renewal flag on price, read off
the coefficient — is wrong. Risk factors drive both the price (because we re-rate
them into the premium) and the renewal decision (because higher-risk customers
may also have fewer alternatives). Ordinary regression conflates the two.

Double Machine Learning (DML) separates them. It residualises both the outcome
and the treatment on the same set of observable confounders, then estimates the
causal effect from what's left. Applied to renewal data, it gives a
semi-elasticity: the expected change in renewal probability per unit change in
log price, controlling for everything in your rating factors.

This library wraps EconML's `CausalForestDML` and `LinearDML` to do exactly
that, with insurance-specific defaults and an FCA-compliant pricing optimiser
built in.

---

## What you get

- **Heterogeneous elasticity estimates**: per-customer CATE and segment-level
  GATE (group average treatment effects by NCD band, age, channel, etc.)
- **Treatment variation diagnostics**: flags the near-deterministic price
  problem before you fit — if your pricing grid leaves no residual variation,
  the results are meaningless
- **Elasticity surface**: heatmap and bar chart of elasticity across two
  dimensions simultaneously
- **FCA PS21/5-compliant optimiser**: maximises profit subject to the ENBP
  constraint (offer price ≤ equivalent new business price)
- **ENBP audit**: per-policy FCA ICOBS 6B.2 compliance flag
- **Portfolio demand curve**: renewal rate and expected profit across a sweep
  of price changes

---

## Install

```bash
pip install insurance-elasticity[all]
```

Core dependencies: `polars`, `numpy`, `scipy`, `scikit-learn`.
Optional (for fitting): `econml>=0.15`, `catboost>=1.2`.
Optional (for plotting): `matplotlib>=3.7`.

---

## Quick start

```python
from insurance_elasticity.data import make_renewal_data
from insurance_elasticity.fit import RenewalElasticityEstimator
from insurance_elasticity.surface import ElasticitySurface
from insurance_elasticity.optimise import RenewalPricingOptimiser
from insurance_elasticity.diagnostics import ElasticityDiagnostics
from insurance_elasticity.demand import demand_curve

# 1. Load data (or use the synthetic generator for testing)
df = make_renewal_data(n=50_000)

# 2. Check treatment variation before fitting
diag = ElasticityDiagnostics()
report = diag.treatment_variation_report(
    df,
    treatment="log_price_change",
    confounders=["age", "ncd_years", "vehicle_group", "region", "channel"],
)
print(report.summary())
# If report.weak_treatment is True, read the suggestions before proceeding.

# 3. Fit the elasticity model
confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]
est = RenewalElasticityEstimator(
    cate_model="causal_forest",   # non-parametric CATE surface
    n_estimators=200,
    catboost_iterations=500,
    n_folds=5,
)
est.fit(df, outcome="renewed", treatment="log_price_change", confounders=confounders)

# 4. Average treatment effect
ate, lb, ub = est.ate()
print(f"ATE: {ate:.3f}  95% CI: [{lb:.3f}, {ub:.3f}]")
# A 1-unit increase in log price change reduces renewal by |ATE| percentage points.
# For a 10% price increase (log change ≈ 0.095), effect ≈ ATE * 0.095.

# 5. Segment-level elasticity
gate = est.gate(df, by="ncd_years")
print(gate)

# 6. Elasticity surface and plots
surface = ElasticitySurface(est)
fig = surface.plot_surface(df, dims=["ncd_years", "age_band"])
fig.savefig("elasticity_surface.png", dpi=150, bbox_inches="tight")

fig2 = surface.plot_gate(df, by="channel")
fig2.savefig("gate_by_channel.png", dpi=150, bbox_inches="tight")

# 7. FCA-compliant pricing optimisation
opt = RenewalPricingOptimiser(
    est,
    technical_premium_col="tech_prem",
    enbp_col="enbp",
    floor_loading=1.0,
)
priced_df = opt.optimise(df, objective="profit")

# 8. Compliance audit
audit = opt.enbp_audit(priced_df)
print(f"Breaches: {(audit['compliant'] == False).sum()} / {len(audit)}")

# 9. Portfolio demand curve
demand_df = demand_curve(est, df, price_range=(-0.25, 0.25, 50))
```

---

## The near-deterministic price problem

Insurance re-rating makes the offered price nearly a deterministic function of
the observable risk factors. When `Var(D̃) / Var(D) < 10%` — that is, less than
10% of price variation remains after conditioning on X — DML has almost nothing
to work with. The confidence intervals blow up and the point estimate is noise.

Always run `ElasticityDiagnostics.treatment_variation_report()` first. If
`weak_treatment` is True, do not proceed to fitting without addressing it.

The report's suggestions cover the main remedies: A/B price tests, panel data
with within-customer variation, quasi-experiments from bulk re-rates, and the
PS21/5 regression discontinuity.

---

## FCA PS21/5 and ENBP

Since January 2022, UK GI firms must not quote a renewing customer a price above
the equivalent new business price (ENBP). The `RenewalPricingOptimiser`
enforces this as a hard per-policy constraint. The `enbp_audit()` method returns
a per-row compliance flag for reporting to the compliance function.

---

## Treatment variable

The standard treatment is `log(offer_price / last_year_price)`. This gives a
semi-elasticity directly: a 1-unit change in D (100% price increase) changes
renewal probability by theta percentage points. For the typical 5–20% renewal
re-rates in UK personal lines, interpret as: a 10% increase changes renewal
probability by approximately `ATE * log(1.1) ≈ ATE * 0.095`.

---

## Model choices

**CausalForestDML** (default): non-parametric, requires no pre-specified feature
interactions, provides valid pointwise confidence intervals via honest splitting.
Right for the elasticity surface. Computationally heavier.

**LinearDML**: assumes constant elasticity (or heterogeneity only through
explicitly interacted features). Much faster. Right for quick portfolio-level
ATE estimation.

**CatBoost nuisance models**: UK insurance data is full of categoricals (region,
vehicle group, occupation, payment method). CatBoost handles them natively. The
alternative is to one-hot encode everything and use gradient boosting, which
works but requires more care.

---

## References

- Chernozhukov et al. (2018). Double/debiased machine learning for treatment and
  structural parameters. *Econometrics Journal*, 21(1).
- Athey & Wager (2019). Estimating treatment effects with causal forests.
  *Annals of Statistics*, 47(2).
- Guelman & Guillén (2014). A causal inference approach to measure price
  elasticity in automobile insurance. *Expert Systems with Applications*, 41(2).
- FCA PS21/5 (2021). General Insurance Pricing Practices Policy Statement.

---

## Licence

MIT. Built by [Burning Cost](https://burningcost.github.io).
