# insurance-elasticity — Deprecated

This package has been superseded by [insurance-causal](https://github.com/burning-cost/insurance-causal).

All functionality — `RenewalElasticityEstimator`, `RenewalPricingOptimiser`, `ElasticitySurface`, `ElasticityDiagnostics`, `TreatmentVariationReport`, `demand_curve`, and `make_renewal_data` — is now part of insurance-causal under the `insurance_causal.elasticity` subpackage.

## Migration

```bash
pip install insurance-causal
```

```python
# Before
from insurance_elasticity import RenewalElasticityEstimator, RenewalPricingOptimiser

# After
from insurance_causal.elasticity import RenewalElasticityEstimator, RenewalPricingOptimiser
```

This repository is archived and will not receive further updates.
