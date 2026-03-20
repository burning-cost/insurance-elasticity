import warnings

warnings.warn(
    "insurance-elasticity is deprecated. Use insurance-causal instead:\n"
    "  pip install insurance-causal\n"
    "  from insurance_causal.elasticity import RenewalElasticityEstimator\n"
    "  from insurance_causal.elasticity import RenewalPricingOptimiser, ElasticitySurface\n"
    "This package will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location for backwards compatibility
from insurance_causal.elasticity import *  # noqa: F401,F403
