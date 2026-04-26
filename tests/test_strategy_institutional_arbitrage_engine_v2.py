"""Tests for Institutional Arbitrage Engine v2 research prototype."""
import numpy as np
import pandas as pd


def _make_data(n=120, start="2026-04-13 09:30", close=None,
               vix=14.0, vxv=16.0, vrp_pct=50.0, pin_strike=None):
    if close is None:
        close = np.full(n, 100.0)
    close = np.asarray(close, dtype=float)
    n = len(close)
    data = {
        "datetime": pd.date_range(start, periods=n, freq="h"),
        "open": close,
        "high": close + 0.25,
        "low": close - 0.25,
        "close": close,
        "volume": np.full(n, 100000.0),
        "vix": np.full(n, vix),
        "vxv": np.full(n, vxv),
        "vrp_pct": np.full(n, vrp_pct),
    }
    if pin_strike is not None:
        data["pin_strike"] = np.full(n, float(pin_strike))
    return {"exec_df_1H": pd.DataFrame(data), "symbol": "SPY"}


def test_institutional_engine_registers():
    from apex.strategies import STRATEGY_REGISTRY
    from apex.strategies import institutional_arbitrage_engine_v2  # noqa: F401

    assert "institutional_arbitrage_engine_v2" in STRATEGY_REGISTRY


def test_engine1_fade_long_in_calm_contango():
    from apex.strategies.institutional_arbitrage_engine_v2 import (
        InstitutionalArbitrageEngineV2Strategy,
    )

    close = [100, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0, 100.1,
             100.0, 100.1, 100.0, 99.8, 99.2, 98.6, 98.0, 97.4]
    data = _make_data(close=close)
    strat = InstitutionalArbitrageEngineV2Strategy(params={
        "enable_momentum": False,
        "enable_opex_pin": False,
        "fade_deviation_sigma": 0.8,
        "fade_vwap_slope_atr_max": 10.0,
        "fade_rsi2_oversold": 35,
        "min_session_bars": 0,
    })
    signals = strat.compute_signals(data)

    assert signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_engine2_momentum_long_in_contango():
    from apex.strategies.institutional_arbitrage_engine_v2 import (
        InstitutionalArbitrageEngineV2Strategy,
    )

    close = np.linspace(100.0, 130.0, 520)
    data = _make_data(close=close)
    strat = InstitutionalArbitrageEngineV2Strategy(params={
        "enable_fade": False,
        "enable_opex_pin": False,
        "momentum_threshold_pct": 0.1,
        "momentum_short_bars": 10,
        "momentum_long_bars": 20,
        "momentum_crisis_bars": 30,
    })
    signals = strat.compute_signals(data)

    assert signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_engine2_momentum_uses_crisis_lookback_in_backwardation():
    from apex.strategies.institutional_arbitrage_engine_v2 import (
        InstitutionalArbitrageEngineV2Strategy,
    )

    close = np.concatenate([np.linspace(130.0, 120.0, 60),
                            np.linspace(120.0, 100.0, 60)])
    data = _make_data(close=close, vix=18.0, vxv=16.0)
    strat = InstitutionalArbitrageEngineV2Strategy(params={
        "enable_fade": False,
        "enable_opex_pin": False,
        "momentum_threshold_pct": 0.1,
        "momentum_short_bars": 5,
        "momentum_long_bars": 10,
        "momentum_crisis_bars": 20,
    })
    signals = strat.compute_signals(data)

    assert signals["entry_short"].any()


def test_engine3_opex_pin_long_below_pin():
    from apex.strategies.institutional_arbitrage_engine_v2 import (
        InstitutionalArbitrageEngineV2Strategy,
    )

    data = _make_data(n=80, start="2026-04-13 09:30",
                      close=np.full(80, 395.0), pin_strike=400.0)
    strat = InstitutionalArbitrageEngineV2Strategy(params={
        "enable_fade": False,
        "enable_momentum": False,
        "pin_min_distance_pct": 0.005,
    })
    signals = strat.compute_signals(data)

    assert signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_no_opex_pin_entries_without_pin_strike():
    from apex.strategies.institutional_arbitrage_engine_v2 import (
        InstitutionalArbitrageEngineV2Strategy,
    )

    data = _make_data(n=80, start="2026-04-13 09:30",
                      close=np.full(80, 395.0))
    strat = InstitutionalArbitrageEngineV2Strategy(params={
        "enable_fade": False,
        "enable_momentum": False,
    })
    signals = strat.compute_signals(data)

    assert not signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_tunable_params_include_each_engine_surface():
    from apex.strategies.institutional_arbitrage_engine_v2 import (
        InstitutionalArbitrageEngineV2Strategy,
    )

    params = InstitutionalArbitrageEngineV2Strategy().get_tunable_params()
    expected = {
        "contango_max", "backwardation_min", "vix_max_fade",
        "fade_deviation_sigma", "fade_vwap_slope_atr_max",
        "fade_rsi2_oversold", "fade_rsi2_overbought",
        "momentum_threshold_pct", "pin_min_distance_pct",
        "stop_atr_mult", "max_hold_bars",
    }
    assert set(params) == expected
