"""Tests for Strategy 3: VIX Term Structure."""
import numpy as np
import pandas as pd


def _make_data(n=100, vix=14.0, vxv=17.0):
    """Constant vix/vxv data — caller may override per-test."""
    rng = np.random.default_rng(11)
    close = 400 + np.cumsum(rng.normal(0, 0.5, n))
    return {
        "exec_df_1H": pd.DataFrame({
            "datetime": pd.date_range("2025-01-02 09:30", periods=n, freq="h"),
            "open": close - 0.05, "high": close + 0.20,
            "low": close - 0.20, "close": close,
            "volume": rng.integers(10000, 50000, n).astype(float),
            "vix": np.full(n, vix) if not isinstance(vix, np.ndarray) else vix,
            "vxv": np.full(n, vxv) if not isinstance(vxv, np.ndarray) else vxv,
        }),
        "symbol": "SPY",
    }


def test_vix_term_registers():
    from apex.strategies import STRATEGY_REGISTRY
    from apex.strategies import vix_term_structure  # noqa: F401
    assert "vix_term_structure" in STRATEGY_REGISTRY


def test_no_entries_when_ratio_in_neutral():
    """ts_ratio = 1.0 (vix=15, vxv=15) is squarely inside neutral band -> no entry."""
    from apex.strategies.vix_term_structure import VIXTermStructureStrategy
    s = VIXTermStructureStrategy()
    data = _make_data(n=100, vix=15.0, vxv=15.0)
    signals = s.compute_signals(data)
    assert not signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_long_on_extreme_contango_with_rsi_confirm():
    """Crashing ts_ratio from neutral down to deep contango should fire LONG.

    Uses a noisy downward drift so RSI(5) has both gains and losses to
    produce non-NaN values; final segment crashes the ratio well below 0.85.
    """
    from apex.strategies.vix_term_structure import VIXTermStructureStrategy
    rng = np.random.default_rng(101)
    n = 80
    vxv = np.full(n, 20.0)
    # Warmup: noisy random walk hovering around 20; final stretch drops VIX
    # sharply so the ratio collapses to ~0.70 with periodic up-ticks for RSI.
    warmup = 20.0 + rng.normal(0, 0.4, 30)
    crash = np.concatenate([
        np.linspace(20.0, 14.0, 40) + rng.normal(0, 0.3, 40),
        [14.5, 14.0] * 5,  # tail with mixed gains/losses for non-NaN RSI
    ])
    vix = np.concatenate([warmup, crash])
    data = _make_data(n=n, vix=vix, vxv=vxv)
    s = VIXTermStructureStrategy()
    signals = s.compute_signals(data)
    assert signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_short_on_extreme_backwardation_with_rsi_confirm():
    """Spiking ts_ratio from neutral up to deep backwardation should fire SHORT.

    Uses noisy upward drift so RSI(5) yields non-NaN; final segment pushes
    ratio well above 1.10.
    """
    from apex.strategies.vix_term_structure import VIXTermStructureStrategy
    rng = np.random.default_rng(202)
    n = 80
    vxv = np.full(n, 20.0)
    warmup = 20.0 + rng.normal(0, 0.4, 30)
    spike = np.concatenate([
        np.linspace(20.0, 26.0, 40) + rng.normal(0, 0.3, 40),
        [25.5, 26.0] * 5,
    ])
    vix = np.concatenate([warmup, spike])
    data = _make_data(n=n, vix=vix, vxv=vxv)
    s = VIXTermStructureStrategy()
    signals = s.compute_signals(data)
    assert signals["entry_short"].any()
    assert not signals["entry_long"].any()


def test_exit_signals_in_neutral_band():
    """Exit flags must be set when ts_ratio is inside neutral_low..neutral_high."""
    from apex.strategies.vix_term_structure import VIXTermStructureStrategy
    s = VIXTermStructureStrategy()
    data = _make_data(n=50, vix=15.0, vxv=15.0)  # ratio = 1.0 (in band)
    signals = s.compute_signals(data)
    assert signals["exit_long"].all()
    assert signals["exit_short"].all()


def test_tunable_params_match_spec():
    from apex.strategies.vix_term_structure import VIXTermStructureStrategy
    s = VIXTermStructureStrategy()
    params = s.get_tunable_params()
    expected = {"contango_extreme_threshold", "backwardation_extreme_threshold",
                "neutral_low", "neutral_high", "stop_atr_mult", "max_hold_bars"}
    assert set(params.keys()) == expected
    assert len(params) == 6
