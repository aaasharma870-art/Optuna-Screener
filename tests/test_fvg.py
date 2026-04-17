"""Tests for Fair Value Gap detector."""

import numpy as np
import pandas as pd
import pytest

from apex.indicators.fvg import detect_fvgs, unfilled_fvgs_at


def _make_df(highs, lows, closes):
    """Build a minimal OHLCV DataFrame from arrays."""
    n = len(highs)
    return pd.DataFrame({
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.full(n, 1000.0),
    })


class TestFVG:
    def test_bullish_fvg_detected(self):
        """Bullish FVG: high[0] < low[2] → gap."""
        #        bar0    bar1    bar2
        # high:  100     105     108
        # low:    95     100     102
        # Gap: high[0]=100 < low[2]=102
        df = _make_df(
            highs=[100, 105, 108],
            lows=[95, 100, 102],
            closes=[98, 103, 106],
        )
        fvgs = detect_fvgs(df)
        bullish = [f for f in fvgs if f["direction"] == "bullish"]
        assert len(bullish) == 1
        assert bullish[0]["start_idx"] == 0
        assert bullish[0]["end_idx"] == 2

    def test_bearish_fvg_detected(self):
        """Bearish FVG: low[0] > high[2] → gap."""
        #        bar0    bar1    bar2
        # high:  110     105     98
        # low:   105     100     93
        # Gap: low[0]=105 > high[2]=98
        df = _make_df(
            highs=[110, 105, 98],
            lows=[105, 100, 93],
            closes=[108, 102, 95],
        )
        fvgs = detect_fvgs(df)
        bearish = [f for f in fvgs if f["direction"] == "bearish"]
        assert len(bearish) == 1
        assert bearish[0]["start_idx"] == 0

    def test_fill_tracking(self):
        """Bullish FVG should be filled when close drops to gap low."""
        # bar0: high=100, bar2: low=102 → bullish gap [100, 102]
        # bar3: close drops to 99 → fills
        df = _make_df(
            highs=[100, 105, 108, 103],
            lows=[95, 100, 102, 97],
            closes=[98, 103, 106, 99],
        )
        fvgs = detect_fvgs(df)
        bullish = [f for f in fvgs if f["direction"] == "bullish"]
        assert len(bullish) == 1
        assert bullish[0]["filled_at_idx"] == 3

    def test_no_fvg_in_continuous_bars(self):
        """Overlapping bars produce no FVGs."""
        df = _make_df(
            highs=[102, 103, 104, 105],
            lows=[99, 100, 101, 102],
            closes=[101, 102, 103, 104],
        )
        fvgs = detect_fvgs(df)
        assert len(fvgs) == 0

    def test_chronological_order(self):
        """FVGs should be in chronological order by start_idx."""
        # Two gaps
        df = _make_df(
            highs=[100, 105, 108, 100, 105, 108],
            lows=[95, 100, 102, 95, 100, 102],
            closes=[98, 103, 106, 98, 103, 106],
        )
        fvgs = detect_fvgs(df)
        if len(fvgs) > 1:
            for i in range(len(fvgs) - 1):
                assert fvgs[i]["start_idx"] <= fvgs[i + 1]["start_idx"]

    def test_unfilled_fvgs_at(self):
        """unfilled_fvgs_at filters correctly."""
        df = _make_df(
            highs=[100, 105, 108, 103],
            lows=[95, 100, 102, 97],
            closes=[98, 103, 106, 99],
        )
        fvgs = detect_fvgs(df)
        # At idx 2, the FVG just formed but not yet filled
        unfilled = unfilled_fvgs_at(fvgs, 2)
        assert len(unfilled) == 1
        # At idx 3, the FVG is filled
        unfilled = unfilled_fvgs_at(fvgs, 3)
        assert len(unfilled) == 0
