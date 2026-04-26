"""Tests for portfolio-level ensemble CPCV."""
import numpy as np
import pandas as pd
import pytest


def test_ensemble_cpcv_returns_sharpe_distribution():
    from apex.validation.ensemble_cpcv import evaluate_ensemble_cpcv
    rng = np.random.default_rng(42)
    n = 1000
    # Synthetic portfolio NAV: positive drift + noise
    portfolio_returns = pd.Series(rng.normal(0.0005, 0.01, n))
    result = evaluate_ensemble_cpcv(portfolio_returns,
                                     n_blocks=4, n_test_blocks=1)
    assert result["n_folds"] > 0
    assert "sharpe_median" in result
    assert "sharpe_iqr" in result
    assert isinstance(result["oos_sharpes"], list)


def test_ensemble_cpcv_handles_empty_returns():
    from apex.validation.ensemble_cpcv import evaluate_ensemble_cpcv
    result = evaluate_ensemble_cpcv(pd.Series([]),
                                     n_blocks=4, n_test_blocks=1)
    assert result.get("n_folds", 0) == 0


def test_ensemble_cpcv_positive_drift_yields_positive_median():
    """Strong positive drift → median Sharpe should be positive."""
    from apex.validation.ensemble_cpcv import evaluate_ensemble_cpcv
    rng = np.random.default_rng(0)
    # Strong drift, small noise
    returns = pd.Series(rng.normal(0.005, 0.005, 800))
    result = evaluate_ensemble_cpcv(returns, n_blocks=4, n_test_blocks=1)
    assert result["sharpe_median"] > 0
