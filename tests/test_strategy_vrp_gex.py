"""Tests for Strategy 1: VRP+GEX Fade."""
import numpy as np
import pandas as pd
import pytest


def _make_test_data(n=200, regime="R1"):
    """Build minimal data dict with VRP regime + gamma walls."""
    rng = np.random.default_rng(42)
    close = 400 + np.cumsum(rng.normal(0, 0.5, n))
    return {
        "exec_df_1H": pd.DataFrame({
            "datetime": pd.date_range("2025-01-02 09:30", periods=n, freq="h"),
            "open": close - 0.05, "high": close + 0.20,
            "low": close - 0.20, "close": close,
            "volume": rng.integers(10000, 50000, n).astype(float),
            "vix": np.full(n, 14.0),
            "vxv": np.full(n, 17.0),
            "vrp_pct": np.full(n, 85.0),
            # Strategy 1 needs gamma wall columns
            "call_wall": np.full(n, 410.0),
            "put_wall": np.full(n, 390.0),
            "gamma_flip": np.full(n, 400.0),
        }),
        "regime_state": pd.Series([regime] * n),
        "symbol": "SPY",
    }


def test_strategy_registers():
    """Strategy 1 should auto-register in STRATEGY_REGISTRY."""
    from apex.strategies import STRATEGY_REGISTRY
    from apex.strategies import vrp_gex_fade  # noqa: F401  triggers @register_strategy
    assert "vrp_gex_fade" in STRATEGY_REGISTRY


def test_compute_signals_returns_correct_columns():
    from apex.strategies.vrp_gex_fade import VRPGEXFadeStrategy
    s = VRPGEXFadeStrategy()
    data = _make_test_data()
    signals = s.compute_signals(data)
    for c in ("entry_long", "entry_short", "exit_long", "exit_short"):
        assert c in signals.columns
    assert len(signals) == len(data["exec_df_1H"])


def test_no_entries_in_r4_regime():
    from apex.strategies.vrp_gex_fade import VRPGEXFadeStrategy
    s = VRPGEXFadeStrategy()
    data = _make_test_data(regime="R4")
    signals = s.compute_signals(data)
    assert not signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_tunable_params_match_spec():
    """Spec lists 7 tunable params for Strategy 1."""
    from apex.strategies.vrp_gex_fade import VRPGEXFadeStrategy
    s = VRPGEXFadeStrategy()
    params = s.get_tunable_params()
    expected = {"vrp_pct_threshold", "gamma_wall_proximity_atr",
                "rsi2_oversold", "rsi2_overbought",
                "vpin_pct_max", "stop_atr_mult", "max_hold_bars"}
    assert set(params.keys()) == expected
