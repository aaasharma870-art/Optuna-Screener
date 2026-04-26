"""Tests for risk-parity weight computation."""
import numpy as np
import pandas as pd
import pytest


def test_equal_vol_strategies_get_equal_weights():
    from apex.ensemble.risk_parity import compute_risk_parity_weights
    # Three strategies with identical vol → equal weights.
    # Use N=240 (≈ 1 year of daily) so realized vol is close to true vol and
    # inverse-vol weights converge to ~1/3 each within 5%.
    rng = np.random.default_rng(42)
    returns = {
        "s1": pd.Series(rng.normal(0, 0.01, 240)),
        "s2": pd.Series(rng.normal(0, 0.01, 240)),
        "s3": pd.Series(rng.normal(0, 0.01, 240)),
    }
    weights = compute_risk_parity_weights(returns, lookback_days=240)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)
    for w in weights.values():
        assert abs(w - 1/3) < 0.05  # within 5% of equal


def test_higher_vol_strategy_gets_lower_weight():
    from apex.ensemble.risk_parity import compute_risk_parity_weights
    rng = np.random.default_rng(0)
    returns = {
        "low_vol":  pd.Series(rng.normal(0, 0.005, 60)),
        "high_vol": pd.Series(rng.normal(0, 0.020, 60)),
    }
    weights = compute_risk_parity_weights(returns)
    assert weights["low_vol"] > weights["high_vol"]
    # Low-vol weight ≈ 4x high-vol weight (vol ratio is 4)
    assert weights["low_vol"] / weights["high_vol"] == pytest.approx(4.0, rel=0.20)


def test_zero_vol_strategy_gets_zero_weight():
    from apex.ensemble.risk_parity import compute_risk_parity_weights
    returns = {
        "active": pd.Series(np.random.default_rng(1).normal(0, 0.01, 60)),
        "dead": pd.Series([0.0] * 60),  # zero vol
    }
    weights = compute_risk_parity_weights(returns)
    assert weights["dead"] == 0.0
    assert weights["active"] == pytest.approx(1.0)


def test_weights_sum_to_one():
    from apex.ensemble.risk_parity import compute_risk_parity_weights
    rng = np.random.default_rng(7)
    returns = {f"s{i}": pd.Series(rng.normal(0, 0.005 + i*0.003, 60))
               for i in range(5)}
    weights = compute_risk_parity_weights(returns)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)


def test_max_weight_cap_enforced():
    """When one strategy has very low vol relative to others, it should hit
    the max_weight cap — no single strategy may exceed it."""
    from apex.ensemble.risk_parity import compute_risk_parity_weights
    rng = np.random.default_rng(2)
    # 5 strategies so the cap (0.30 * 5 = 1.50) is achievable. The lowest-vol
    # strategy would naturally exceed the cap and must be clamped.
    returns = {
        "tiny_vol":   pd.Series(rng.normal(0, 0.001, 60)),
        "low_vol":    pd.Series(rng.normal(0, 0.005, 60)),
        "med_vol":    pd.Series(rng.normal(0, 0.010, 60)),
        "high_vol_a": pd.Series(rng.normal(0, 0.050, 60)),
        "high_vol_b": pd.Series(rng.normal(0, 0.050, 60)),
    }
    weights = compute_risk_parity_weights(returns, max_weight=0.30)
    for w in weights.values():
        assert w <= 0.30 + 1e-9
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
