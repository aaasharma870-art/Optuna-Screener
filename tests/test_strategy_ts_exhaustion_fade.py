"""Tests for term-structure gated VWAP exhaustion fade."""
import numpy as np
import pandas as pd


def _make_data(close, vix=14.0, vxv=16.0, vrp_pct=50.0):
    close = np.asarray(close, dtype=float)
    n = len(close)
    return {
        "exec_df_1H": pd.DataFrame({
            "datetime": pd.date_range("2025-01-02 09:30", periods=n, freq="h"),
            "open": close,
            "high": close + 0.25,
            "low": close - 0.25,
            "close": close,
            "volume": np.full(n, 100000.0),
            "vix": np.full(n, vix),
            "vxv": np.full(n, vxv),
            "vrp_pct": np.full(n, vrp_pct),
        }),
        "symbol": "SPY",
    }


def test_ts_exhaustion_fade_registers():
    from apex.strategies import STRATEGY_REGISTRY
    from apex.strategies import ts_exhaustion_fade  # noqa: F401

    assert "ts_exhaustion_fade" in STRATEGY_REGISTRY


def test_long_on_contango_vwap_oversold_exhaustion():
    from apex.strategies.ts_exhaustion_fade import TermStructureExhaustionFadeStrategy

    close = [100, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0, 100.1,
             100.0, 100.1, 100.0, 99.8, 99.2, 98.6, 98.0, 97.4]
    data = _make_data(close)
    strat = TermStructureExhaustionFadeStrategy(params={
        "deviation_sigma": 0.8,
        "vwap_slope_atr_max": 10.0,
        "min_session_bars": 0,
        "max_entry_hour": 23.0,
        "rsi2_oversold": 35,
    })
    signals = strat.compute_signals(data)

    assert signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_short_on_contango_vwap_overbought_exhaustion():
    from apex.strategies.ts_exhaustion_fade import TermStructureExhaustionFadeStrategy

    close = [100, 99.9, 100.0, 99.9, 100.0, 99.9, 100.0, 99.9,
             100.0, 99.9, 100.0, 100.2, 100.8, 101.4, 102.0, 102.6]
    data = _make_data(close)
    strat = TermStructureExhaustionFadeStrategy(params={
        "deviation_sigma": 0.8,
        "vwap_slope_atr_max": 10.0,
        "min_session_bars": 5,
        "max_entry_hour": 23.0,
        "rsi2_oversold": 1,
        "rsi2_overbought": 65,
    })
    signals = strat.compute_signals(data)

    assert signals["entry_short"].any()
    assert not signals["entry_long"].any()


def test_no_entry_when_term_structure_not_contango():
    from apex.strategies.ts_exhaustion_fade import TermStructureExhaustionFadeStrategy

    close = [100, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0, 100.1,
             100.0, 100.1, 100.0, 99.8, 99.2, 98.6, 98.0, 97.4]
    data = _make_data(close, vix=16.0, vxv=16.0)
    strat = TermStructureExhaustionFadeStrategy(params={
        "deviation_sigma": 0.8,
        "vwap_slope_atr_max": 10.0,
        "min_session_bars": 0,
        "max_entry_hour": 23.0,
        "rsi2_oversold": 35,
    })
    signals = strat.compute_signals(data)

    assert not signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_tunable_params_match_research_surface():
    from apex.strategies.ts_exhaustion_fade import TermStructureExhaustionFadeStrategy

    params = TermStructureExhaustionFadeStrategy().get_tunable_params()
    expected = {
        "contango_max", "vix_max", "vrp_calm_low", "vrp_calm_high",
        "use_vrp_calm_filter", "enable_long", "enable_short",
        "deviation_sigma", "vwap_slope_atr_max", "rsi2_oversold",
        "rsi2_overbought", "exit_long_rsi", "exit_short_rsi",
        "min_session_bars", "max_entry_hour", "exit_at_session_end",
        "stop_atr_mult", "max_hold_bars",
    }
    assert set(params) == expected


def test_long_side_can_be_disabled():
    from apex.strategies.ts_exhaustion_fade import TermStructureExhaustionFadeStrategy

    close = [100, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0, 100.1,
             100.0, 100.1, 100.0, 99.8, 99.2, 98.6, 98.0, 97.4]
    data = _make_data(close)
    strat = TermStructureExhaustionFadeStrategy(params={
        "enable_long": False,
        "deviation_sigma": 0.8,
        "vwap_slope_atr_max": 10.0,
        "min_session_bars": 0,
        "max_entry_hour": 23.0,
        "rsi2_oversold": 35,
    })
    signals = strat.compute_signals(data)

    assert not signals["entry_long"].any()


def test_default_exits_by_session_end():
    from apex.strategies.ts_exhaustion_fade import TermStructureExhaustionFadeStrategy

    close = [100, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0, 100.1,
             100.0, 100.1, 100.0, 99.8, 99.2, 98.6, 98.0, 97.4]
    data = _make_data(close)
    strat = TermStructureExhaustionFadeStrategy(params={
        "deviation_sigma": 0.8,
        "vwap_slope_atr_max": 10.0,
        "min_session_bars": 0,
        "max_entry_hour": 23.0,
        "rsi2_oversold": 35,
    })
    signals = strat.compute_signals(data)
    pos = strat.compute_position_size(data, signals)

    dates = pd.to_datetime(data["exec_df_1H"]["datetime"]).dt.date
    last_bar_idx = int(dates.ne(dates.shift(-1)).to_numpy().nonzero()[0][0])
    assert pos.iloc[last_bar_idx] == 0.0
