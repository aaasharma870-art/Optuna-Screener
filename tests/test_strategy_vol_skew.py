"""Tests for Strategy 4: Vol Skew Arbitrage."""
import numpy as np
import pandas as pd


def _make_data(n=100, skew_ratio=1.10):
    rng = np.random.default_rng(13)
    close = 400 + np.cumsum(rng.normal(0, 0.5, n))
    if isinstance(skew_ratio, (int, float)):
        skew_ratio = np.full(n, float(skew_ratio))
    return {
        "exec_df_1H": pd.DataFrame({
            "datetime": pd.date_range("2025-01-02 09:30", periods=n, freq="h"),
            "open": close - 0.05, "high": close + 0.20,
            "low": close - 0.20, "close": close,
            "volume": rng.integers(10000, 50000, n).astype(float),
            "skew_ratio": skew_ratio,
        }),
        "symbol": "SPY",
    }


def test_vol_skew_registers():
    from apex.strategies import STRATEGY_REGISTRY
    from apex.strategies import vol_skew_arb  # noqa: F401
    assert "vol_skew_arb" in STRATEGY_REGISTRY


def test_no_entries_when_skew_in_normal_band():
    """skew = 1.10 (inside 1.05-1.20) -> no entries, only exit flags."""
    from apex.strategies.vol_skew_arb import VolSkewArbStrategy
    s = VolSkewArbStrategy()
    data = _make_data(n=100, skew_ratio=1.10)
    signals = s.compute_signals(data)
    assert not signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_long_on_extreme_put_skew():
    """skew > 1.30 -> LONG (mean-reversion)."""
    from apex.strategies.vol_skew_arb import VolSkewArbStrategy
    s = VolSkewArbStrategy()
    data = _make_data(n=100, skew_ratio=1.40)
    signals = s.compute_signals(data)
    assert signals["entry_long"].all()
    assert not signals["entry_short"].any()


def test_short_on_extreme_call_skew():
    """skew < 0.95 -> SHORT."""
    from apex.strategies.vol_skew_arb import VolSkewArbStrategy
    s = VolSkewArbStrategy()
    data = _make_data(n=100, skew_ratio=0.90)
    signals = s.compute_signals(data)
    assert signals["entry_short"].all()
    assert not signals["entry_long"].any()


def test_exit_signals_in_normal_band():
    from apex.strategies.vol_skew_arb import VolSkewArbStrategy
    s = VolSkewArbStrategy()
    data = _make_data(n=50, skew_ratio=1.10)
    signals = s.compute_signals(data)
    assert signals["exit_long"].all()
    assert signals["exit_short"].all()


def test_graceful_skip_when_skew_column_missing():
    """If skew_ratio column not pre-merged, return all-False signals (no raise)."""
    from apex.strategies.vol_skew_arb import VolSkewArbStrategy
    s = VolSkewArbStrategy()
    data = _make_data(n=50, skew_ratio=1.10)
    data["exec_df_1H"] = data["exec_df_1H"].drop(columns=["skew_ratio"])
    signals = s.compute_signals(data)
    assert not signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_tunable_params_match_spec():
    from apex.strategies.vol_skew_arb import VolSkewArbStrategy
    s = VolSkewArbStrategy()
    params = s.get_tunable_params()
    expected = {"put_skew_extreme", "call_skew_extreme",
                "normal_low", "normal_high",
                "dte_target", "stop_atr_mult", "max_hold_days"}
    assert set(params.keys()) == expected
    assert len(params) == 7
