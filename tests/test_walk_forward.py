"""Tests for walk-forward weight validation."""
import numpy as np
import pandas as pd
import pytest


def test_walk_forward_compares_dynamic_vs_static():
    from apex.validation.walk_forward import compare_dynamic_vs_static_weights
    rng = np.random.default_rng(42)
    n_months = 24
    # Strategy A: stable positive Sharpe; Strategy B: noisy
    months = pd.date_range("2023-01-01", periods=n_months, freq="ME")
    monthly_returns = {
        "vrp_gex_fade": pd.Series(rng.normal(0.01, 0.02, n_months), index=months),
        "opex_gravity": pd.Series(rng.normal(0.005, 0.05, n_months), index=months),
    }
    result = compare_dynamic_vs_static_weights(monthly_returns)
    assert "static_sharpe" in result
    assert "dynamic_sharpe" in result
    assert "uplift" in result
    assert isinstance(result["uplift"], float)


def test_walk_forward_handles_short_history():
    from apex.validation.walk_forward import compare_dynamic_vs_static_weights
    monthly_returns = {
        "vrp_gex_fade": pd.Series([0.01, -0.02]),
        "opex_gravity": pd.Series([0.005, 0.001]),
    }
    result = compare_dynamic_vs_static_weights(monthly_returns)
    assert result.get("n_months", 0) <= 2 or "error" in result
