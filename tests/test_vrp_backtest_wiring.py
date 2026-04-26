"""Verify run_backtest uses VRP regime-gated entries when regime_model='vrp'."""
import numpy as np
import pandas as pd
import pytest


def _vrp_test_df(n=200):
    """Synthetic OHLCV with VIX/VXV/VRP columns to drive R1 regime."""
    rng = np.random.default_rng(42)
    close = 400 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=n, freq="h"),
        "open": close - 0.05,
        "high": close + 0.20,
        "low": close - 0.20,
        "close": close,
        "volume": rng.integers(10000, 50000, n).astype(float),
        # Force R1 regime: contango (vix<vxv*0.95), high VRP pct, low VIX
        "vix": np.full(n, 15.0),
        "vxv": np.full(n, 18.0),
        "vrp_pct": np.full(n, 85.0),
    })
    return df


def test_vrp_mode_calls_determine_entry_direction(monkeypatch):
    """In vrp mode, run_backtest delegates entry to determine_entry_direction."""
    from apex.engine.backtest import run_backtest, compute_indicator_signals

    df = _vrp_test_df()
    architecture = {
        "regime_model": "vrp",
        "indicators": ["VPIN", "VWCLV", "VWAP_Bands"],
        "direction": "neutral",
        "min_score": 0,
        "exits": ["max_bars"],
        "aggregation": "majority",
        "score_aggregation": "weighted",
    }
    params = {"stop_pct": 0.02, "target_pct": 0.05, "max_bars": 20,
              "rsi2_oversold": 10, "rsi2_overbought": 90,
              "vpin_threshold_pct": 60, "vwclv_divergence_threshold": 1.3,
              "aggregation": "majority"}

    # Spy on determine_entry_direction
    call_count = {"n": 0}
    from apex.engine import backtest as bt_mod
    orig = bt_mod.determine_entry_direction
    def _spy(*args, **kwargs):
        call_count["n"] += 1
        return orig(*args, **kwargs)
    monkeypatch.setattr(bt_mod, "determine_entry_direction", _spy)

    signals_data = compute_indicator_signals(df, architecture, params)
    trades, stats = run_backtest(df, signals_data, architecture, params)

    # In VRP mode, determine_entry_direction must be called at least once
    assert call_count["n"] > 0, "VRP mode did not call determine_entry_direction"


def test_legacy_mode_does_not_call_determine_entry_direction(monkeypatch):
    """In legacy (non-vrp) mode, determine_entry_direction is never called."""
    from apex.engine.backtest import (
        run_backtest, compute_indicator_signals,
        DEFAULT_ARCHITECTURE, DEFAULT_PARAMS,
    )
    from apex.engine import backtest as bt_mod

    # Use a non-vrp architecture
    architecture = dict(DEFAULT_ARCHITECTURE)
    architecture["regime_model"] = "ema"  # legacy

    df = _vrp_test_df()

    call_count = {"n": 0}
    orig = bt_mod.determine_entry_direction
    def _spy(*args, **kwargs):
        call_count["n"] += 1
        return orig(*args, **kwargs)
    monkeypatch.setattr(bt_mod, "determine_entry_direction", _spy)

    signals_data = compute_indicator_signals(df, architecture, DEFAULT_PARAMS)
    trades, stats = run_backtest(df, signals_data, architecture, DEFAULT_PARAMS)

    assert call_count["n"] == 0, "Legacy mode unexpectedly called determine_entry_direction"


def test_vrp_extras_populated():
    """When regime_model='vrp', signals_data['extras'] must contain VRP signal columns."""
    from apex.engine.backtest import compute_indicator_signals

    df = _vrp_test_df()
    architecture = {
        "regime_model": "vrp",
        "indicators": ["VPIN", "VWCLV", "VWAP_Bands"],
        "direction": "neutral",
        "min_score": 0,
        "exits": ["max_bars"],
        "score_aggregation": "weighted",
    }
    params = {"aggregation": "majority"}
    signals_data = compute_indicator_signals(df, architecture, params)
    extras = signals_data.get("extras", {})
    expected_keys = {"vpin_pct", "cum_vwclv", "breakout_reversal_long",
                     "breakout_reversal_short", "sweep_proxy_long", "sweep_proxy_short",
                     "in_deviation_zone_long", "in_deviation_zone_short", "in_pullback_zone"}
    missing = expected_keys - set(extras.keys())
    assert not missing, f"Missing VRP extras: {missing}"


def test_vrp_r4_produces_no_trades():
    """Force R4 (no trade) regime -- should produce zero trades."""
    from apex.engine.backtest import run_backtest, compute_indicator_signals

    df = _vrp_test_df()
    df["vix"] = 35.0  # high VIX -> R4
    df["vxv"] = 18.0
    df["vrp_pct"] = 50.0

    architecture = {
        "regime_model": "vrp",
        "indicators": ["VPIN", "VWCLV", "VWAP_Bands"],
        "direction": "neutral",
        "min_score": 0,
        "exits": ["max_bars"],
        "score_aggregation": "weighted",
    }
    params = {"stop_pct": 0.02, "target_pct": 0.05, "max_bars": 20,
              "aggregation": "majority"}
    signals_data = compute_indicator_signals(df, architecture, params)
    trades, stats = run_backtest(df, signals_data, architecture, params)

    assert len(trades) == 0, f"R4 regime fired {len(trades)} trades, expected 0"
