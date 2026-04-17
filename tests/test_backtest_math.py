"""Tests for direction-aware backtest execution (long/short/neutral)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from apex.engine.backtest import (
    run_backtest, compute_stats, DEFAULT_ARCHITECTURE, DEFAULT_PARAMS,
)


def _make_df(prices, highs=None, lows=None, volumes=None):
    """Build a minimal OHLCV DataFrame from a close price list."""
    n = len(prices)
    close = np.array(prices, dtype=float)
    if highs is None:
        highs = close * 1.005
    else:
        highs = np.array(highs, dtype=float)
    if lows is None:
        lows = close * 0.995
    else:
        lows = np.array(lows, dtype=float)
    open_ = close.copy()
    if volumes is None:
        volumes = np.full(n, 1_000_000.0)
    df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": open_,
        "high": highs,
        "low": lows,
        "close": close,
        "volume": volumes,
    })
    return df


def _make_signals_data(df, score_vals, regime_vals=None, atr_val=1.0):
    """Build the signals_data dict expected by run_backtest."""
    n = len(df)
    if regime_vals is None:
        regime_vals = ["R1"] * n
    return {
        "regime": pd.Series(regime_vals, index=df.index),
        "score": pd.Series(score_vals, index=df.index),
        "atr": pd.Series(atr_val, index=df.index),
    }


class TestLongProfitsOnUptrend:
    def test_long_profits_on_uptrend(self):
        """Long trade profits when price trends up."""
        # Steady uptrend: 100, 102, 104, 106, 108, 110
        prices = [100, 102, 104, 106, 108, 110]
        df = _make_df(prices)
        # Score triggers entry at bar 0 -> fill at bar 1
        scores = [10, 0, 0, 0, 0, 0]
        arch = {"direction": "long", "min_score": 5,
                "exit_methods": ["time_exit"]}
        params = {"max_hold_bars": 3, "commission_pct": 0.0, "atr_stop_mult": 100,
                  "atr_target_mult": 100}
        sd = _make_signals_data(df, scores)
        trades, stats = run_backtest(df, sd, arch, params)
        assert len(trades) >= 1
        assert trades[0]["direction"] == "long"
        assert trades[0]["pnl_pct"] > 0


class TestShortProfitsOnDowntrend:
    def test_short_profits_on_downtrend(self):
        """Short trade profits when price trends down."""
        prices = [110, 108, 106, 104, 102, 100]
        df = _make_df(prices)
        scores = [10, 0, 0, 0, 0, 0]
        arch = {"direction": "short", "min_score": 5,
                "exit_methods": ["time_exit"]}
        params = {"max_hold_bars": 3, "commission_pct": 0.0,
                  "borrow_rate": 0.0, "bars_per_day": 7,
                  "atr_stop_mult": 100, "atr_target_mult": 100}
        sd = _make_signals_data(df, scores)
        trades, stats = run_backtest(df, sd, arch, params)
        assert len(trades) >= 1
        assert trades[0]["direction"] == "short"
        assert trades[0]["pnl_pct"] > 0


class TestShortPnLIncludesBorrowFee:
    def test_short_pnl_includes_borrow_fee(self):
        """Borrow fee reduces short PnL."""
        prices = [110, 108, 106, 104, 102, 100]
        df = _make_df(prices)
        scores = [10, 0, 0, 0, 0, 0]
        arch = {"direction": "short", "min_score": 5,
                "exit_methods": ["time_exit"]}

        # Without borrow fee
        params_no_fee = {"max_hold_bars": 3, "commission_pct": 0.0,
                         "borrow_rate": 0.0, "bars_per_day": 7,
                         "atr_stop_mult": 100, "atr_target_mult": 100}
        sd = _make_signals_data(df, scores)
        trades_no, _ = run_backtest(df, sd, arch, params_no_fee)

        # With borrow fee
        params_fee = {"max_hold_bars": 3, "commission_pct": 0.0,
                      "borrow_rate": 0.10, "bars_per_day": 7,
                      "atr_stop_mult": 100, "atr_target_mult": 100}
        trades_fee, _ = run_backtest(df, sd, arch, params_fee)

        assert len(trades_no) >= 1 and len(trades_fee) >= 1
        # Fee reduces PnL
        assert trades_fee[0]["pnl_pct"] < trades_no[0]["pnl_pct"]


class TestStopSymmetricForShorts:
    def test_stop_symmetric_for_shorts(self):
        """Short stop is ABOVE entry (entry * (1 + stop_pct))."""
        prices = [100, 100, 105, 110, 115, 120]  # Price goes up = bad for short
        highs = [101, 101, 106, 111, 116, 121]
        lows = [99, 99, 104, 109, 114, 119]
        df = _make_df(prices, highs=highs, lows=lows)
        scores = [10, 0, 0, 0, 0, 0]
        arch = {"direction": "short", "min_score": 5,
                "exit_methods": ["fixed_stop"]}
        # ATR = 1.0, stop_mult = 2.0 -> stop at entry + 2.0
        params = {"atr_stop_mult": 2.0, "atr_target_mult": 100,
                  "commission_pct": 0.0, "borrow_rate": 0.0,
                  "bars_per_day": 7}
        sd = _make_signals_data(df, scores, atr_val=1.0)
        trades, _ = run_backtest(df, sd, arch, params)
        assert len(trades) >= 1
        t = trades[0]
        assert t["direction"] == "short"
        assert t["exit_reason"] == "fixed_stop"
        assert t["pnl_pct"] < 0  # Stopped out at a loss


class TestLegacyDirectionDefaultIsLong:
    def test_legacy_direction_default_is_long(self):
        assert DEFAULT_ARCHITECTURE["direction"] == "long"


class TestTradeRecordHasDirectionField:
    def test_trade_record_has_direction_field(self):
        prices = [100, 102, 104, 106, 108, 110]
        df = _make_df(prices)
        scores = [10, 0, 0, 0, 0, 0]
        arch = {"direction": "long", "min_score": 5,
                "exit_methods": ["time_exit"]}
        params = {"max_hold_bars": 2, "commission_pct": 0.0,
                  "atr_stop_mult": 100, "atr_target_mult": 100}
        sd = _make_signals_data(df, scores)
        trades, _ = run_backtest(df, sd, arch, params)
        assert len(trades) >= 1
        assert "direction" in trades[0]
        assert trades[0]["direction"] in ("long", "short")
