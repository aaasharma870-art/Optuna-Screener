"""Tests for apex.ensemble.pnl.compute_pnl_stats (Phase 15)."""
import math

import numpy as np
import pandas as pd

from apex.ensemble.pnl import compute_pnl_stats


def test_constant_long_on_rising_prices_positive_return():
    """Constant long position on a steadily rising price -> positive total return + sharpe>0."""
    n = 500
    prices = pd.Series(100.0 * (1.0005 ** np.arange(n)))  # +0.05%/bar
    positions = pd.Series([1.0] * n)
    res = compute_pnl_stats(positions, prices, periods_per_year=252)
    assert res["total_return_pct"] > 10.0
    assert res["sharpe_annualized"] > 0.0
    # No mid-series position change after the initial fill -> 0 trades counted
    # (diff's first NaN is filled to 0 by spec; we don't synthesize an entry)
    assert res["n_trades"] == 0
    # Equity curve should be increasing overall
    eq = res["equity_curve"]
    assert eq[-1] > eq[0]
    assert len(eq) == n


def test_all_flat_positions_zero_return_zero_trades():
    """All-flat positions on noisy prices -> 0% return, 0 trades, 0 sharpe."""
    n = 200
    rng = np.random.default_rng(42)
    prices = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.5, n)))
    positions = pd.Series([0.0] * n)
    res = compute_pnl_stats(positions, prices, periods_per_year=252)
    assert res["n_trades"] == 0
    assert abs(res["total_return_pct"]) < 1e-9
    assert res["sharpe_annualized"] == 0.0
    assert res["win_rate_pct"] == 0.0
    assert res["n_bars_in_position"] == 0


def test_drawdown_calculation_on_v_shaped_equity():
    """Long position on V-shaped price -> max drawdown matches the drop."""
    # Price drops 20% then recovers fully
    down = np.linspace(100.0, 80.0, 50)
    up = np.linspace(80.0, 100.0, 50)
    prices = pd.Series(np.concatenate([down, up]))
    positions = pd.Series([1.0] * len(prices))
    res = compute_pnl_stats(positions, prices, periods_per_year=252,
                            commission_pct=0.0)  # disable cost so DD isn't biased
    # Max drawdown should be approximately -20%
    assert res["max_dd_pct"] < -15.0
    assert res["max_dd_pct"] > -25.0
    # Total return ~0% (commission disabled, V-shape recovers)
    assert abs(res["total_return_pct"]) < 1.0


def test_win_rate_counts_correctly():
    """Long position on alternating up/down bars -> ~50% win rate."""
    n = 100
    # alternating +1% and -1% returns (chained)
    rets = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(n)])
    prices = pd.Series(100.0 * np.cumprod(1.0 + rets))
    positions = pd.Series([1.0] * n)
    res = compute_pnl_stats(positions, prices, periods_per_year=252,
                            commission_pct=0.0)
    # First bar has no prior position, so it's not a winner. Of remaining 99,
    # we expect roughly half to be winners.
    assert 35.0 <= res["win_rate_pct"] <= 65.0


def test_trade_count_matches_position_changes():
    """Position changes should be counted as trades (above 0.01 threshold)."""
    # Position pattern: 0,0,1,1,1,0,0,-1,-1,0  -> 4 transitions
    pos = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0])
    prices = pd.Series([100.0 + i * 0.1 for i in range(len(pos))])
    res = compute_pnl_stats(pos, prices, periods_per_year=252, commission_pct=0.0)
    assert res["n_trades"] == 4


def test_calmar_ratio_finite_when_drawdown_present():
    """Calmar should be finite when there is a non-zero drawdown."""
    # Up then down then up (some drawdown)
    prices = pd.Series([100, 110, 120, 100, 90, 110, 130], dtype=float)
    positions = pd.Series([1.0] * len(prices))
    res = compute_pnl_stats(positions, prices, periods_per_year=252,
                            commission_pct=0.0)
    assert math.isfinite(res["calmar"])
    assert abs(res["max_dd_pct"]) > 0.5  # there is real drawdown


def test_insufficient_data_returns_error():
    """A single bar should return error key without crashing."""
    res = compute_pnl_stats(pd.Series([1.0]), pd.Series([100.0]))
    assert "error" in res
    assert res["n_bars"] == 1


def test_equity_curve_length_matches_input():
    """Equity curve length should match the (truncated) input length."""
    n = 50
    prices = pd.Series([100.0 + i for i in range(n)])
    positions = pd.Series([1.0] * n)
    res = compute_pnl_stats(positions, prices, periods_per_year=252)
    assert len(res["equity_curve"]) == n


def test_handles_short_position_negative_return_on_rising_prices():
    """Short position on rising prices -> negative total return."""
    n = 200
    prices = pd.Series(100.0 * (1.001 ** np.arange(n)))
    positions = pd.Series([-1.0] * n)
    res = compute_pnl_stats(positions, prices, periods_per_year=252,
                            commission_pct=0.0)
    assert res["total_return_pct"] < 0.0
    assert res["sharpe_annualized"] < 0.0
