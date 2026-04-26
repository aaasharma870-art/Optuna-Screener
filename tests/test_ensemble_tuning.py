"""Tests for per-strategy Optuna tuning."""
import numpy as np
import pandas as pd
import pytest


def _make_synthetic_data(n=500):
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0.001, 0.01, n))
    return {
        "exec_df_1H": pd.DataFrame({
            "datetime": pd.date_range("2025-01-02", periods=n, freq="h"),
            "open": close - 0.05, "high": close + 0.20,
            "low": close - 0.20, "close": close,
            "volume": rng.integers(10000, 50000, n).astype(float),
            "vix": np.full(n, 15.0),
            "vxv": np.full(n, 18.0),
            "vrp_pct": np.full(n, 75.0),
            "call_wall": close + 5,
            "put_wall": close - 5,
            "gamma_flip": close,
        }),
        "regime_state": pd.Series(["R1"] * n),
        "symbol": "SPY",
    }


def test_tune_strategy_returns_best_params():
    """Tuning should return a dict with best_params, best_sharpe, n_trials."""
    from apex.optimize.ensemble_tuning import tune_strategy
    from apex.strategies.vrp_gex_fade import VRPGEXFadeStrategy
    data = _make_synthetic_data()
    result = tune_strategy(VRPGEXFadeStrategy, data, n_trials=10, n_blocks=4, n_test_blocks=1)
    assert "best_params" in result
    assert "best_sharpe" in result
    assert "n_trials" in result
    assert result["n_trials"] == 10


def test_tune_strategy_with_no_tunable_params():
    """Strategies with empty get_tunable_params() should return empty best_params."""
    from apex.optimize.ensemble_tuning import tune_strategy
    from apex.strategies.cross_asset_vol_overlay import CrossAssetVolOverlayStrategy
    data = _make_synthetic_data()
    # Add the columns the overlay needs
    n = len(data["exec_df_1H"])
    data["exec_df_1H"]["vix_pct"] = 50.0
    data["exec_df_1H"]["move_pct"] = 50.0
    data["exec_df_1H"]["ovx_pct"] = 50.0
    result = tune_strategy(CrossAssetVolOverlayStrategy, data, n_trials=5)
    assert result["best_params"] == {}
    assert result["n_trials"] == 0


def test_tune_ensemble_strategies_runs_all():
    """tune_ensemble_strategies should produce results for every strategy passed."""
    from apex.optimize.ensemble_tuning import tune_ensemble_strategies
    from apex.strategies.vrp_gex_fade import VRPGEXFadeStrategy
    from apex.strategies.vix_term_structure import VIXTermStructureStrategy
    data_per_symbol = {"SPY": _make_synthetic_data()}
    results = tune_ensemble_strategies(
        [VRPGEXFadeStrategy, VIXTermStructureStrategy],
        data_per_symbol,
        n_trials_per_strategy=5,
    )
    assert "vrp_gex_fade" in results
    assert "vix_term_structure" in results
    for name, r in results.items():
        assert "best_params" in r
        assert "best_sharpe" in r
        assert "primary_symbol" in r
