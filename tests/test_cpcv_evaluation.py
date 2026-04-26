"""Tests for CPCV-based parameter evaluation."""
import numpy as np
import pandas as pd


def _make_synthetic_df(n=500):
    """Make a deterministic OHLCV df for CPCV testing."""
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "datetime": pd.date_range("2022-01-01", periods=n, freq="h"),
        "open": close - 0.05,
        "high": close + 0.20,
        "low": close - 0.20,
        "close": close,
        "volume": rng.integers(10000, 50000, n).astype(float),
        "vix": np.full(n, 15.0),
        "vxv": np.full(n, 18.0),
        "vrp_pct": np.full(n, 50.0),
    })


def test_cpcv_evaluation_returns_distribution():
    from apex.validation.cpcv import evaluate_params_via_cpcv
    df = _make_synthetic_df(800)
    arch = {"regime_model": "ema", "indicators": [], "direction": "long",
            "min_score": 0, "exits": ["max_bars"], "aggregation": "majority",
            "score_aggregation": "weighted"}
    params = {"atr_stop_mult": 1.5, "atr_target_mult": 3.0, "max_hold_bars": 10,
              "commission_pct": 0.05}
    result = evaluate_params_via_cpcv(
        "TEST", df, daily_df=None, architecture=arch, best_params=params,
        n_blocks=4, n_test_blocks=1
    )
    assert result.get("n_folds", 0) > 0
    assert "sharpe_median" in result
    assert isinstance(result["oos_sharpes"], list)
    assert len(result["oos_sharpes"]) == result["n_folds"]


def test_cpcv_handles_insufficient_bars():
    from apex.validation.cpcv import evaluate_params_via_cpcv
    df = _make_synthetic_df(50)  # too small
    arch = {"regime_model": "ema", "indicators": [], "direction": "long",
            "min_score": 0, "exits": ["max_bars"], "aggregation": "majority",
            "score_aggregation": "weighted"}
    params = {"max_hold_bars": 5, "commission_pct": 0.05}
    result = evaluate_params_via_cpcv(
        "TEST", df, daily_df=None, architecture=arch, best_params=params
    )
    # Should return n_folds=0 or error gracefully
    assert result.get("n_folds", 0) == 0 or "error" in result
