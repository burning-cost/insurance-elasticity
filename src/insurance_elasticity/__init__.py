"""
insurance-elasticity: Causal price elasticity estimation and FCA-compliant
renewal pricing optimisation for UK personal lines insurance.

Quick start
-----------
>>> from insurance_elasticity.data import make_renewal_data
>>> from insurance_elasticity.fit import RenewalElasticityEstimator
>>> from insurance_elasticity.optimise import RenewalPricingOptimiser
>>> from insurance_elasticity.surface import ElasticitySurface
>>> from insurance_elasticity.demand import demand_curve, plot_demand_curve
>>> from insurance_elasticity.diagnostics import ElasticityDiagnostics

>>> df = make_renewal_data(n=10_000)
>>> confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]
>>> est = RenewalElasticityEstimator()
>>> est.fit(df, confounders=confounders)
>>> ate, lb, ub = est.ate()
>>> print(f"ATE: {ate:.3f}  95% CI: [{lb:.3f}, {ub:.3f}]")
"""

from insurance_elasticity.fit import RenewalElasticityEstimator
from insurance_elasticity.surface import ElasticitySurface
from insurance_elasticity.optimise import RenewalPricingOptimiser
from insurance_elasticity.diagnostics import ElasticityDiagnostics, TreatmentVariationReport
from insurance_elasticity.demand import demand_curve, plot_demand_curve
from insurance_elasticity.data import make_renewal_data, true_gate_by_ncd, true_gate_by_age

__version__ = "0.1.0"

__all__ = [
    "RenewalElasticityEstimator",
    "ElasticitySurface",
    "RenewalPricingOptimiser",
    "ElasticityDiagnostics",
    "TreatmentVariationReport",
    "demand_curve",
    "plot_demand_curve",
    "make_renewal_data",
    "true_gate_by_ncd",
    "true_gate_by_age",
]
