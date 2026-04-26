"""Tests for VIX-term liquidity reclaim strategy."""
import numpy as np
import pandas as pd


def _make_data(close, low=None, high=None, start="2025-01-02 09:30",
               vix=14.0, vxv=16.0, datetimes=None):
    close = np.asarray(close, dtype=float)
    n = len(close)
    low = np.asarray(low if low is not None else close - 0.2, dtype=float)
    high = np.asarray(high if high is not None else close + 0.2, dtype=float)
    return {
        "exec_df_1H": pd.DataFrame({
            "datetime": datetimes if datetimes is not None else pd.date_range(start, periods=n, freq="h"),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 100000.0),
            "vix": np.full(n, vix) if not isinstance(vix, (list, np.ndarray)) else np.asarray(vix, dtype=float),
            "vxv": np.full(n, vxv) if not isinstance(vxv, (list, np.ndarray)) else np.asarray(vxv, dtype=float),
        }),
        "symbol": "SPY",
    }


def test_vix_liquidity_reclaim_registers():
    from apex.strategies import STRATEGY_REGISTRY
    from apex.strategies import vix_liquidity_reclaim  # noqa: F401

    assert "vix_liquidity_reclaim" in STRATEGY_REGISTRY


def test_prior_day_low_sweep_reclaim_long():
    from apex.strategies.vix_liquidity_reclaim import VIXLiquidityReclaimStrategy

    dt1 = pd.date_range("2025-01-02 09:30", periods=7, freq="h")
    dt2 = pd.date_range("2025-01-03 09:30", periods=9, freq="h")
    close = [100, 100, 100, 100, 100, 100, 100,
             101, 100.5, 100.2, 100.0, 99.8, 99.6, 99.4, 100.2, 100.5]
    low = [99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5,
           100.5, 100.0, 99.9, 99.7, 99.5, 99.3, 99.2, 99.0, 100.1]
    data = _make_data(close, low=low, datetimes=dt1.append(dt2))
    strat = VIXLiquidityReclaimStrategy(params={
        "use_opening_range": False,
        "sweep_atr_frac": 0.0,
        "reclaim_buffer_atr": 0.0,
        "min_session_bars": 1,
        "max_entry_hour": 23.0,
    })
    signals = strat.compute_signals(data)

    assert signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_opening_range_low_sweep_reclaim_long():
    from apex.strategies.vix_liquidity_reclaim import VIXLiquidityReclaimStrategy

    close = [100, 100.2, 100.1, 100.0, 100.2, 100.1,
             99.8, 99.7, 99.6, 99.5, 99.5, 99.5, 99.4, 99.3, 100.1, 100.4]
    low = [99.8, 100.0, 99.9, 99.8, 100.0, 99.9,
           99.6, 99.5, 99.4, 99.4, 99.4, 99.4, 99.3, 99.2, 99.1, 100.0]
    data = _make_data(close, low=low)
    strat = VIXLiquidityReclaimStrategy(params={
        "use_prior_day": False,
        "opening_range_bars": 3,
        "sweep_atr_frac": 0.0,
        "reclaim_buffer_atr": 0.0,
        "min_session_bars": 3,
        "max_entry_hour": 24.0,
    })
    signals = strat.compute_signals(data)

    assert signals["entry_long"].any()


def test_no_entry_when_term_structure_too_high():
    from apex.strategies.vix_liquidity_reclaim import VIXLiquidityReclaimStrategy

    close = [100, 100, 100, 100, 100, 100, 100,
             101, 100, 99.9, 99.7, 100.2, 100.5, 100.8]
    low = [99.5] * 7 + [100.5, 99.7, 99.3, 99.0, 99.6, 100.1, 100.4]
    data = _make_data(close, low=low, vix=20.0, vxv=16.0)
    strat = VIXLiquidityReclaimStrategy(params={
        "use_opening_range": False,
        "sweep_atr_frac": 0.0,
        "reclaim_buffer_atr": 0.0,
        "min_session_bars": 1,
        "max_entry_hour": 23.0,
    })
    signals = strat.compute_signals(data)

    assert not signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_vix_rollover_filter_blocks_rising_vix():
    from apex.strategies.vix_liquidity_reclaim import VIXLiquidityReclaimStrategy

    dt1 = pd.date_range("2025-01-02 09:30", periods=7, freq="h")
    dt2 = pd.date_range("2025-01-03 09:30", periods=9, freq="h")
    close = [100, 100, 100, 100, 100, 100, 100,
             101, 100.5, 100.2, 100.0, 99.8, 99.6, 99.4, 100.2, 100.5]
    low = [99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5,
           100.5, 100.0, 99.9, 99.7, 99.5, 99.3, 99.2, 99.0, 100.1]
    rising_vix = np.linspace(12.0, 16.0, len(close))
    data = _make_data(close, low=low, vix=rising_vix, datetimes=dt1.append(dt2))
    strat = VIXLiquidityReclaimStrategy(params={
        "use_opening_range": False,
        "sweep_atr_frac": 0.0,
        "reclaim_buffer_atr": 0.0,
        "min_session_bars": 1,
        "max_entry_hour": 23.0,
        "require_vix_rollover": True,
        "vix_rollover_min_drop": 0.01,
    })
    signals = strat.compute_signals(data)

    assert not signals["entry_long"].any()


def test_tunable_params_include_structure_surface():
    from apex.strategies.vix_liquidity_reclaim import VIXLiquidityReclaimStrategy

    params = VIXLiquidityReclaimStrategy().get_tunable_params()
    expected = {
        "enable_long", "enable_short", "contango_max", "backwardation_max",
        "vix_max", "require_vix_rollover", "vix_rollover_lookback",
        "vix_rollover_min_drop", "require_ts_ratio_rollover",
        "ts_ratio_rollover_lookback", "require_ts_rsi_turn", "ts_rsi_long_max",
        "ts_rsi_short_min", "use_prior_day", "use_opening_range",
        "opening_range_bars", "sweep_atr_frac", "reclaim_buffer_atr",
        "max_entry_hour", "min_session_bars", "stop_atr_mult",
        "max_hold_bars", "exit_at_session_end",
    }
    assert set(params) == expected
