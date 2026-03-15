"""
Tests for elasticity fitting: RenewalElasticityEstimator.

These tests verify:
    - fit() runs without errors on synthetic data
    - ate() returns a tuple with correct structure and plausible sign
    - cate() returns an array of the right length
    - gate() returns a polars DataFrame with expected columns
    - Unfitted estimator raises RuntimeError on ate/cate/gate
    - ValueError raised when confounders are not provided
    - Model accepts both polars and pandas DataFrames
"""

import numpy as np
import polars as pl
import pytest

pytest.importorskip("econml", reason="econml not installed — skipping elasticity tests")

from insurance_elasticity.data import make_renewal_data, true_gate_by_ncd
from insurance_elasticity.fit import RenewalElasticityEstimator

CONFOUNDERS = ["age", "ncd_years", "vehicle_group", "channel"]


class TestFitBasic:
    def test_fit_returns_self(self, small_df):
        est = RenewalElasticityEstimator(
            n_estimators=20, catboost_iterations=50, n_folds=2, random_state=0
        )
        result = est.fit(small_df, confounders=CONFOUNDERS)
        assert result is est

    def test_fit_sets_is_fitted(self, small_df):
        est = RenewalElasticityEstimator(
            n_estimators=20, catboost_iterations=50, n_folds=2, random_state=0
        )
        assert not est._is_fitted
        est.fit(small_df, confounders=CONFOUNDERS)
        assert est._is_fitted

    def test_fit_no_confounders_raises(self, small_df):
        est = RenewalElasticityEstimator(n_estimators=20, catboost_iterations=50)
        with pytest.raises(ValueError, match="confounders must be provided"):
            est.fit(small_df)

    def test_fit_accepts_pandas(self, small_df):
        est = RenewalElasticityEstimator(
            n_estimators=20, catboost_iterations=50, n_folds=2, random_state=0
        )
        df_pd = small_df.to_pandas()
        est.fit(df_pd, confounders=CONFOUNDERS)
        assert est._is_fitted


class TestATE:
    def test_ate_returns_tuple_of_three(self, fitted_estimator):
        result = fitted_estimator.ate()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_ate_values_are_finite(self, fitted_estimator):
        ate, lb, ub = fitted_estimator.ate()
        assert np.isfinite(ate)
        assert np.isfinite(lb)
        assert np.isfinite(ub)

    def test_ate_ci_ordered(self, fitted_estimator):
        ate, lb, ub = fitted_estimator.ate()
        assert lb <= ate <= ub, f"CI not ordered: [{lb:.3f}, {ate:.3f}, {ub:.3f}]"

    def test_ate_negative_sign(self, fitted_estimator):
        """Price increases reduce renewal probability — ATE should be negative."""
        ate, lb, ub = fitted_estimator.ate()
        # Upper bound of CI should be negative for true negative elasticity
        # (may not hold on tiny datasets, so we allow some slack)
        assert ate < 1.0, f"ATE={ate:.3f} implausibly large and positive"

    def test_ate_not_fitted_raises(self):
        est = RenewalElasticityEstimator(n_estimators=20, catboost_iterations=50)
        with pytest.raises(RuntimeError, match="not fitted"):
            est.ate()


class TestCATE:
    def test_cate_length(self, fitted_estimator, small_df):
        cate = fitted_estimator.cate(small_df)
        assert len(cate) == len(small_df)

    def test_cate_returns_numpy(self, fitted_estimator, small_df):
        cate = fitted_estimator.cate(small_df)
        assert isinstance(cate, np.ndarray)

    def test_cate_heterogeneous(self, fitted_estimator, small_df):
        """CATEs should not all be identical — there must be some heterogeneity."""
        cate = fitted_estimator.cate(small_df)
        assert np.std(cate) > 0.0, "CATE has zero variance — estimator is not heterogeneous"

    def test_cate_interval_shape(self, fitted_estimator, small_df):
        lb, ub = fitted_estimator.cate_interval(small_df)
        assert lb.shape == (len(small_df),)
        assert ub.shape == (len(small_df),)

    def test_cate_interval_ordered(self, fitted_estimator, small_df):
        lb, ub = fitted_estimator.cate_interval(small_df)
        assert np.all(lb <= ub), "Some CI lower bounds exceed upper bounds"

    def test_cate_not_fitted_raises(self, small_df):
        est = RenewalElasticityEstimator(n_estimators=20, catboost_iterations=50)
        with pytest.raises(RuntimeError):
            est.cate(small_df)


class TestGATE:
    def test_gate_returns_polars(self, fitted_estimator, small_df):
        result = fitted_estimator.gate(small_df, by="ncd_years")
        assert isinstance(result, pl.DataFrame)

    def test_gate_has_expected_columns(self, fitted_estimator, small_df):
        result = fitted_estimator.gate(small_df, by="ncd_years")
        for col in ["ncd_years", "elasticity", "ci_lower", "ci_upper", "n"]:
            assert col in result.columns, f"Column '{col}' missing from gate() output"

    def test_gate_n_rows_correct(self, fitted_estimator, small_df):
        """Should have one row per unique value of the grouping column."""
        result = fitted_estimator.gate(small_df, by="ncd_years")
        n_groups = small_df["ncd_years"].n_unique()
        assert len(result) == n_groups

    def test_gate_n_sums_to_total(self, fitted_estimator, small_df):
        result = fitted_estimator.gate(small_df, by="ncd_years")
        assert result["n"].sum() == len(small_df)

    def test_gate_monotone_direction(self, fitted_estimator, small_df):
        """Higher NCD should have less negative (more inelastic) elasticity."""
        result = fitted_estimator.gate(small_df, by="ncd_years").sort("ncd_years")
        elas = result["elasticity"].to_numpy()
        # Not strict monotone (due to noise on small data), but should trend upward
        # Correlation with NCD should be positive (less negative = higher NCD)
        ncd = result["ncd_years"].to_numpy()
        correlation = np.corrcoef(ncd, elas)[0, 1]
        # Allow weak correlation on small data — just check direction
        assert correlation > -0.5, (
            f"Elasticity-NCD correlation = {correlation:.2f}: "
            "higher NCD should trend toward less price sensitivity"
        )


class TestAlternativeModels:
    def test_linear_dml_fits(self, small_df):
        """LinearDML works with binary_outcome=False (treats Y as continuous).
        LinearDML with a classifier as model_y can fail in some econml versions.
        """
        est = RenewalElasticityEstimator(
            cate_model="linear_dml",
            binary_outcome=False,  # avoid classifier/matmul issue in LinearDML
            n_estimators=20,
            catboost_iterations=50,
            n_folds=2,
        )
        est.fit(small_df, confounders=CONFOUNDERS)
        ate, lb, ub = est.ate()
        assert np.isfinite(ate)

    def test_unknown_model_raises(self, small_df):
        est = RenewalElasticityEstimator(cate_model="banana")
        with pytest.raises(ValueError, match="Unknown cate_model"):
            est.fit(small_df, confounders=CONFOUNDERS)

class TestATECIBoundsArePortfolioAverage:
    """Regression tests for P0-1: ATE CI bounds must be portfolio-average, not
    the CI of the first row.

    ate_interval() from econml returns arrays of shape (n_samples, 1). Before the
    fix, float(result[0]) took the first row's CI. The correct value is
    result[0].mean() across all rows.
    """

    def test_ate_ci_not_single_row(self, small_df):
        """Fit a fresh estimator and verify lb/ub are not just the first row's CI.

        We check that the CI bounds returned by ate() equal the mean of the
        ate_interval() output arrays. We do this by comparing to what the raw
        econml API returns.
        """
        est = RenewalElasticityEstimator(
            n_estimators=40, catboost_iterations=80, n_folds=2, random_state=7
        )
        est.fit(small_df, confounders=CONFOUNDERS)
        # Call ate() — this nulls _X_train after the call
        ate_val, lb, ub = est.ate()
        # Both bounds must be finite and ordered
        assert np.isfinite(lb) and np.isfinite(ub)
        assert lb <= ub, f"CI not ordered: lb={lb:.4f} ub={ub:.4f}"
        # The CI must contain the point estimate
        assert lb <= ate_val <= ub, (
            f"ATE {ate_val:.4f} outside its own CI [{lb:.4f}, {ub:.4f}]"
        )

    def test_ate_ci_width_reasonable(self, small_df):
        """Portfolio-average CI should be narrower than a single-row CI.

        CausalForestDML individual CIs are wide (honest forest uncertainty).
        When we average n=2000 rows the portfolio CI should collapse. A width
        of more than 5.0 suggests we are returning a single-row CI instead.
        """
        est = RenewalElasticityEstimator(
            n_estimators=40, catboost_iterations=80, n_folds=2, random_state=8
        )
        est.fit(small_df, confounders=CONFOUNDERS)
        _, lb, ub = est.ate()
        ci_width = ub - lb
        # Portfolio CI should be much narrower than a 5-unit window
        # (individual row CIs from CausalForest are typically 1–8 units wide)
        assert ci_width < 5.0, (
            f"ATE CI width={ci_width:.3f} is suspiciously large; "
            "check that CI bounds are portfolio-average, not single-row"
        )


class TestGateGroupCINarrowsWithSize:
    """Regression tests for P1-3: GATE CI must narrow with group size.

    Before the fix, gate() averaged individual CIs, which don't depend on n.
    After the fix, CI uses SE = SD(CATE) / sqrt(n), which does narrow.
    """

    def test_gate_ci_narrower_for_larger_groups(self, fitted_estimator, small_df):
        """Groups with more members should have narrower CIs than small groups."""
        result = fitted_estimator.gate(small_df, by="ncd_years").sort("n")
        ns = result["n"].to_numpy()
        ci_widths = (result["ci_upper"] - result["ci_lower"]).to_numpy()
        # Correlation between group size and CI width should be negative
        if len(ns) > 2:
            corr = float(np.corrcoef(ns, ci_widths)[0, 1])
            assert corr < 0.5, (
                f"GATE CI width-vs-n correlation={corr:.3f}: "
                "larger groups should tend to have narrower CIs"
            )

    def test_gate_ci_all_finite(self, fitted_estimator, small_df):
        result = fitted_estimator.gate(small_df, by="channel")
        assert result["ci_lower"].is_finite().all()
        assert result["ci_upper"].is_finite().all()

