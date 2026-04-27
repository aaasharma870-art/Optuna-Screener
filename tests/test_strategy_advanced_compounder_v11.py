"""Tests for Advanced Compounder v11 Pine port."""
import numpy as np
import pandas as pd


def _make_data(close):
    close = np.asarray(close, dtype=float)
    return {
        "exec_df_1H": pd.DataFrame({
            "datetime": pd.date_range("2025-01-02 09:30", periods=len(close), freq="h"),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(close), 100000.0),
        }),
        "symbol": "SPY",
    }


def test_supertrend_returns_line_and_direction():
    from apex.indicators.basics import compute_supertrend

    df = _make_data(np.linspace(100, 120, 60))["exec_df_1H"]
    line, direction = compute_supertrend(df, period=7, factor=2.0)

    assert len(line) == len(df)
    assert len(direction) == len(df)
    assert direction.dropna().isin([-1.0, 1.0]).all()


def test_advanced_compounder_registers():
    from apex.strategies import STRATEGY_REGISTRY
    from apex.strategies import advanced_compounder_v11  # noqa: F401

    assert "advanced_compounder_v11" in STRATEGY_REGISTRY


def test_trending_data_produces_positions():
    from apex.strategies.advanced_compounder_v11 import AdvancedCompounderV11Strategy

    close = np.concatenate([
        np.linspace(100, 105, 30),
        np.linspace(104, 118, 50),
        np.linspace(117, 130, 40),
    ])
    strat = AdvancedCompounderV11Strategy(params={
        "st_slow_factor": 2.0,
        "st_slow_period": 7,
        "st_fast_factor": 0.8,
        "st_fast_period": 5,
        "allow_short": False,
    })
    data = _make_data(close)
    signals = strat.compute_signals(data)
    pos = strat.compute_position_size(data, signals)

    assert signals["entry_long"].any()
    assert pos.max() > 0


def test_pyramiding_caps_at_one():
    from apex.strategies.advanced_compounder_v11 import AdvancedCompounderV11Strategy

    strat = AdvancedCompounderV11Strategy(params={"max_pyramids": 5, "unit_size": 0.25})
    data = _make_data(np.linspace(100, 120, 10))
    signals = pd.DataFrame({
        "entry_long": [True] * 10,
        "entry_short": [False] * 10,
        "exit_long": [False] * 10,
        "exit_short": [False] * 10,
    })
    pos = strat.compute_position_size(data, signals)

    assert pos.max() == 1.0
    assert pos.iloc[0] == 0.25


def test_tunable_params_match_pine_surface():
    from apex.strategies.advanced_compounder_v11 import AdvancedCompounderV11Strategy

    params = AdvancedCompounderV11Strategy().get_tunable_params()
    expected = {
        "st_slow_factor", "st_slow_period", "st_fast_factor", "st_fast_period",
        "use_trail", "max_pyramids", "unit_size", "allow_short",
    }
    assert set(params) == expected
