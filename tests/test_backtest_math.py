"""Tests for direction-aware backtest execution (long/short/neutral)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from apex.engine.backtest import (
    run_backtest, compute_stats, determine_entry_direction,
    DEFAULT_ARCHITECTURE, DEFAULT_PARAMS,
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


# ---------------------------------------------------------------------------
# Phase 6 Gap 5: VRP-scaled position sizing
# ---------------------------------------------------------------------------

def _signals_data_with_extras(n, vrp_pct, vix, **series):
    """Build a signals_data dict carrying scalar VRP/VIX inputs as Series."""
    idx = pd.RangeIndex(n)
    extras = {
        "vrp_pct": pd.Series([vrp_pct] * n, index=idx),
        "vix": pd.Series([vix] * n, index=idx),
    }
    for k, v in series.items():
        extras[k] = pd.Series([v] * n, index=idx)
    return {
        "regime": pd.Series(["R1"] * n, index=idx),
        "score": pd.Series([0] * n, index=idx),
        "atr": pd.Series([1.0] * n, index=idx),
        "extras": extras,
    }


class TestDetermineEntryDirectionReturnShape:
    def test_determine_entry_direction_returns_tuple(self):
        """determine_entry_direction must always return a 2-tuple."""
        sd = _signals_data_with_extras(5, vrp_pct=85, vix=15)
        result = determine_entry_direction("R1", 0, sd, 0, {})
        assert isinstance(result, tuple)
        assert len(result) == 2
        direction, size_mult = result
        assert direction in ("long", "short", None)
        assert isinstance(size_mult, float)
        assert 0.0 <= size_mult <= 1.0


class TestVRPScaledSizing:
    def test_size_mult_for_r1_high_conviction(self):
        """R1 with VRP=85 (>80) AND VIX=15 (<18) -> full size 1.0."""
        sd = _signals_data_with_extras(5, vrp_pct=85, vix=15)
        _, size_mult = determine_entry_direction("R1", 0, sd, 0, {})
        assert size_mult == 1.0

    def test_size_mult_for_r1_marginal(self):
        """R1 with VRP=75 ([70,80]) and VIX=20 ([18,22]) -> half size 0.5."""
        sd = _signals_data_with_extras(5, vrp_pct=75, vix=20)
        _, size_mult = determine_entry_direction("R1", 0, sd, 0, {})
        assert size_mult == 0.5

    def test_size_mult_for_r2(self):
        """R2 always -> 0.5."""
        sd = _signals_data_with_extras(5, vrp_pct=50, vix=25)
        _, size_mult = determine_entry_direction("R2", 0, sd, 0, {})
        assert size_mult == 0.5

    def test_size_mult_for_r3(self):
        """R3 always -> 1.0."""
        sd = _signals_data_with_extras(5, vrp_pct=20, vix=30)
        _, size_mult = determine_entry_direction("R3", 0, sd, 0, {})
        assert size_mult == 1.0

    def test_size_mult_for_r4(self):
        """R4 -> direction None and size_mult 0.0."""
        sd = _signals_data_with_extras(5, vrp_pct=10, vix=40)
        direction, size_mult = determine_entry_direction("R4", 0, sd, 0, {})
        assert direction is None
        assert size_mult == 0.0


# ---------------------------------------------------------------------------
# Phase 6 Gap 6: VWAP-anchored profit target
# ---------------------------------------------------------------------------

class TestVWAPTarget:
    def test_default_target_type_preserves_legacy(self):
        """Without target_type set, behavior matches the legacy fixed_target."""
        prices = [100, 102, 104, 106, 108, 110]
        df = _make_df(prices)
        scores = [10, 0, 0, 0, 0, 0]
        arch = {"direction": "long", "min_score": 5,
                "exit_methods": ["fixed_target", "time_exit"]}
        params = {"max_hold_bars": 4, "commission_pct": 0.0,
                  "atr_stop_mult": 100, "atr_target_mult": 1.0}
        sd = _make_signals_data(df, scores, atr_val=1.0)
        trades, _ = run_backtest(df, sd, arch, params)
        assert len(trades) >= 1
        # entry at bar 1 open = 102, target = 102 + 1.0*ATR = 103
        assert trades[0]["exit_reason"] == "fixed_target"

    def test_vwap_target_computes_inline_when_missing(self):
        """Phase 10: target_type='vwap' without df['vwap'] now computes it
        inline (used to raise ValueError). Allows Optuna to sweep target_type
        as a categorical without requiring upstream pre-compute.
        """
        prices = [100, 102, 104, 106]
        df = _make_df(prices)
        scores = [10, 0, 0, 0]
        arch = {"direction": "long", "min_score": 5,
                "exit_methods": ["fixed_target", "time_exit"]}
        params = {"target_type": "vwap", "max_hold_bars": 3,
                  "commission_pct": 0.0, "atr_stop_mult": 100,
                  "atr_target_mult": 100}
        sd = _make_signals_data(df, scores)
        # Should not raise — VWAP computed inline from df's OHLCV.
        trades, stats = run_backtest(df, sd, arch, params)
        # Either fired a trade or didn't (depends on inline-VWAP target levels);
        # the contract is just "no error".
        assert isinstance(trades, list)

    def test_vwap_target_exits_when_price_crosses_vwap(self):
        """Long entered far below VWAP, then price rises through VWAP -> fixed_target hit."""
        # Entry bar 1 open = 95, prices climb so that high crosses vwap.
        prices = [95, 95, 96, 99, 101, 103]
        df = _make_df(prices)
        # Inject a constant VWAP at 100 so the target is unambiguous
        df["vwap"] = 100.0
        scores = [10, 0, 0, 0, 0, 0]
        arch = {"direction": "long", "min_score": 5,
                "exit_methods": ["fixed_target", "time_exit"]}
        params = {"target_type": "vwap", "max_hold_bars": 5,
                  "commission_pct": 0.0, "atr_stop_mult": 100,
                  "atr_target_mult": 100}
        sd = _make_signals_data(df, scores, atr_val=1.0)
        trades, _ = run_backtest(df, sd, arch, params)
        assert len(trades) >= 1
        t = trades[0]
        assert t["exit_reason"] == "fixed_target"
        # Exit is clamped to bar range; price reaches 100 -> profitable long
        assert t["pnl_pct"] > 0
