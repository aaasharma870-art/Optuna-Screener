"""Tests for FVG-anchored dynamic trailing stops."""

import numpy as np
import pandas as pd
import pytest

from apex.engine.stops import compute_dynamic_stop
from apex.indicators.fvg import detect_fvgs, unfilled_fvgs_at


def _make_df(highs, lows, closes, opens=None):
    """Build a minimal OHLCV DataFrame."""
    n = len(highs)
    if opens is None:
        opens = closes
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.full(n, 1000.0),
    })


# ---- Unit tests for compute_dynamic_stop ----

class TestDynamicStop:
    def test_long_stop_selects_nearest_unfilled_bullish_below(self):
        """Long stop anchors to the nearest un-filled bullish FVG below price."""
        # Bullish FVG at bars 0-2: high[0]=100 < low[2]=102 => gap [100, 102]
        # This FVG gets filled at bar 3 (close=99 <= 100)
        # Bullish FVG at bars 3-5: high[3]=90 < low[5]=92 => gap [90, 92]
        # No fill for bars after that in our short data
        df = _make_df(
            highs= [100, 105, 108, 90, 95, 98],
            lows=  [ 95, 100, 102, 85, 90, 92],
            closes=[  98, 103, 106, 88, 93, 96],
        )
        fvgs = detect_fvgs(df)

        # At idx 5, first FVG is filled (filled_at_idx=3), second just formed
        unfilled = unfilled_fvgs_at(fvgs, 5)
        bullish_unfilled = [f for f in unfilled if f["direction"] == "bullish"]
        assert len(bullish_unfilled) >= 1

        # Price at 96; nearest bullish FVG below price has low=90
        atr = 2.0
        stop = compute_dynamic_stop("long", 96.0, fvgs, 5, atr)
        expected = 90.0 - 0.05 * atr  # low of nearest bullish FVG - buffer
        assert stop == pytest.approx(expected), f"stop={stop}, expected={expected}"

    def test_short_stop_selects_nearest_unfilled_bearish_above(self):
        """Short stop anchors to the nearest un-filled bearish FVG above price."""
        # Single bearish FVG: use data where only one forms
        # bar0: low=108 > high[2]=98 won't work if bar1 also creates one
        # Use 3 bars only so exactly 1 bearish FVG forms at bars 0-2
        df = _make_df(
            highs= [110, 105, 98],
            lows=  [105, 100, 93],
            closes=[108, 102, 95],
        )
        fvgs = detect_fvgs(df)
        bearish = [f for f in fvgs if f["direction"] == "bearish"]
        assert len(bearish) == 1
        # Bearish FVG zone: low=high[2]=98, high=low[0]=105

        # Price at 92 (below the gap); bearish FVG low=98 > 92 => qualifies
        atr = 2.0
        stop = compute_dynamic_stop("short", 92.0, fvgs, 2, atr)
        expected = 105.0 + 0.05 * atr  # high of bearish FVG + buffer
        assert stop == pytest.approx(expected)

    def test_fallback_to_atr_when_no_fvg(self):
        """Long falls back to ATR-based stop when no qualifying FVG exists."""
        fvgs = []  # no FVGs at all
        atr = 3.0
        price = 100.0
        stop = compute_dynamic_stop("long", price, fvgs, 5, atr, atr_fallback_mult=2.0)
        assert stop == pytest.approx(100.0 - 2.0 * 3.0)

    def test_short_fallback_to_atr(self):
        """Short falls back to ATR-based stop when no qualifying FVG exists."""
        fvgs = []
        atr = 3.0
        price = 100.0
        stop = compute_dynamic_stop("short", price, fvgs, 5, atr, atr_fallback_mult=2.0)
        assert stop == pytest.approx(100.0 + 2.0 * 3.0)

    def test_skip_filled_fvg(self):
        """Filled FVGs are excluded from stop anchoring."""
        # Bullish FVG at bars 0-2 (gap [100, 102]), filled at bar 3
        df = _make_df(
            highs= [100, 105, 108, 103, 110],
            lows=  [ 95, 100, 102,  97, 105],
            closes=[ 98, 103, 106,  99, 108],
        )
        fvgs = detect_fvgs(df)
        bullish = [f for f in fvgs if f["direction"] == "bullish"]
        assert len(bullish) == 1
        assert bullish[0]["filled_at_idx"] == 3

        # At idx=4, the FVG is filled => no qualifying FVG => fallback
        atr = 2.0
        stop = compute_dynamic_stop("long", 108.0, fvgs, 4, atr, atr_fallback_mult=2.0)
        expected = 108.0 - 2.0 * 2.0
        assert stop == pytest.approx(expected)

    def test_dynamic_stop_engaged_in_backtest(self):
        """Smoke test: run_backtest with dynamic_stop=True produces trades."""
        from apex.engine.backtest import run_backtest
        from apex.indicators.basics import compute_atr

        np.random.seed(42)
        n = 200
        base = 100.0
        noise = np.random.randn(n).cumsum() * 0.5
        closes = base + noise
        highs = closes + np.abs(np.random.randn(n)) * 0.5
        lows = closes - np.abs(np.random.randn(n)) * 0.5
        opens = closes + np.random.randn(n) * 0.1

        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(n, 10000.0),
        })

        atr = compute_atr(df, 14)
        regime = pd.Series("R1", index=df.index)
        # Make entries easy: high score everywhere
        score = pd.Series(5, index=df.index)

        signals_data = {"regime": regime, "score": score, "atr": atr}
        architecture = {
            "direction": "long",
            "min_score": 5,
            "exit_methods": ["fixed_stop", "trailing_stop", "time_exit"],
        }
        params = {
            "atr_stop_mult": 1.5,
            "atr_target_mult": 3.0,
            "atr_trail_mult": 1.0,
            "trail_activate_atr": 1.0,
            "max_hold_bars": 20,
            "commission_pct": 0.05,
            "borrow_rate": 0.0,
            "bars_per_day": 7,
            "dynamic_stop": True,
            "dynamic_stop_atr_fallback": 2.0,
            "dynamic_stop_fvg_buffer": 0.05,
        }

        trades, stats = run_backtest(df, signals_data, architecture, params)
        assert stats["trades"] > 0, "Expected at least one trade with dynamic_stop=True"
        # Every trade should have a direction
        for t in trades:
            assert t["direction"] == "long"
