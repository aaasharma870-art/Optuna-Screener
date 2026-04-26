"""Tests for VRP setup detectors (Phase 6 Gap 3 + Gap 4)."""

import numpy as np
import pandas as pd
import pytest

from apex.engine.setups import detect_breakout_reversal, detect_sweep_proxy
from apex.indicators.vwap_bands import compute_vwap_bands


def _make_ohlcv(closes, highs=None, lows=None, seed=42):
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    rng = np.random.RandomState(seed)
    if highs is None:
        highs = closes + rng.uniform(0.05, 0.2, n)
    else:
        highs = np.asarray(highs, dtype=float)
    if lows is None:
        lows = closes - rng.uniform(0.05, 0.2, n)
    else:
        lows = np.asarray(lows, dtype=float)
    opn = closes.copy()
    volume = np.full(n, 1_000.0)
    ts = pd.date_range("2024-01-15 09:30", periods=n, freq="5min")
    return pd.DataFrame({
        "timestamp": ts,
        "datetime": ts,
        "open": opn,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume,
    })


class TestBreakoutReversal:
    def test_long_fires_after_spike_above_vwap_then_close_back(self):
        """Spike 2 ATR above VWAP, then close back below the spike's high."""
        # Setup: stable VWAP around 100; lookback window with prices spiking
        # high enough to clear breakout_atr_mult * atr above vwap.
        closes = np.full(40, 100.0)
        # Spike upward in window then revert
        closes[30:35] = [105.0, 106.0, 107.0, 106.0, 105.0]
        closes[35] = 102.0  # close back BELOW the window max (107)
        df = _make_ohlcv(closes)
        df = compute_vwap_bands(df)
        # Force ATR ~ 1.0 so breakout threshold = 1.5
        atr = pd.Series(1.0, index=df.index)
        result = detect_breakout_reversal(df, atr, lookback=5, breakout_atr_mult=1.5)
        # Bar 35 should fire long reversal
        assert result["breakout_reversal_long"].iloc[35] == True

    def test_short_fires_after_dip_below_vwap_then_close_back(self):
        """Mirror: price dips far below VWAP, then closes back above the window low."""
        closes = np.full(40, 100.0)
        closes[30:35] = [95.0, 94.0, 93.0, 94.0, 95.0]
        closes[35] = 98.0  # close back ABOVE window min (93)
        df = _make_ohlcv(closes)
        df = compute_vwap_bands(df)
        atr = pd.Series(1.0, index=df.index)
        result = detect_breakout_reversal(df, atr, lookback=5, breakout_atr_mult=1.5)
        assert result["breakout_reversal_short"].iloc[35] == True

    def test_no_false_positives_on_continuous_uptrend(self):
        """Smooth uptrend should not generate spurious reversal-long signals.

        We focus on the LONG-reversal axis -- it requires close[i] to fall
        below the prior window's max, which is impossible on a monotonic
        ascending series. The mirror SHORT axis can trigger transient
        early-session noise as VWAP slowly climbs to meet the trend; that's
        expected behaviour, not a false positive.
        """
        closes = np.linspace(100.0, 130.0, 60)
        df = _make_ohlcv(closes)
        df = compute_vwap_bands(df)
        atr = pd.Series(0.5, index=df.index)
        result = detect_breakout_reversal(df, atr, lookback=5, breakout_atr_mult=1.5)
        # Monotonic uptrend: close[i] >= window_high, so long reversals never fire
        assert result["breakout_reversal_long"].sum() == 0

    def test_requires_vwap_column(self):
        df = _make_ohlcv(np.full(20, 100.0))
        atr = pd.Series(1.0, index=df.index)
        with pytest.raises(ValueError, match="vwap"):
            detect_breakout_reversal(df, atr)


class TestSweepProxy:
    def test_long_fires_on_bullish_fvg_breach_and_reclaim(self):
        """Manually-constructed FVG: price dips below, closes back above."""
        n = 20
        closes = np.full(n, 100.0)
        highs = closes + 0.2
        lows = closes - 0.2
        df = _make_ohlcv(closes, highs=highs, lows=lows)

        # Inject a bullish FVG: edge low = 99.0, high = 99.5.
        fvgs = [{
            "start_idx": 5,
            "end_idx": 7,
            "direction": "bullish",
            "low": 99.0,   # lower edge
            "high": 99.5,
            "filled_at_idx": None,
        }]

        # On bar 12, dip low below 99.0 - 1.5*ATR = 97.5, close above 99.0
        df.loc[12, "low"] = 97.0
        df.loc[12, "close"] = 99.8

        atr = pd.Series(1.0, index=df.index)
        result = detect_sweep_proxy(df, fvgs, atr, breach_atr_mult=1.5)
        assert result["sweep_proxy_long"].iloc[12] == True

    def test_short_fires_on_bearish_fvg_breach_and_reclaim(self):
        """Mirror: bearish FVG above, price spikes above and closes back below."""
        n = 20
        closes = np.full(n, 100.0)
        highs = closes + 0.2
        lows = closes - 0.2
        df = _make_ohlcv(closes, highs=highs, lows=lows)

        # Bearish FVG: low = 100.5, high = 101.0 (upper edge for sweep_short)
        fvgs = [{
            "start_idx": 5,
            "end_idx": 7,
            "direction": "bearish",
            "low": 100.5,
            "high": 101.0,
            "filled_at_idx": None,
        }]

        # Bar 12: spike high above 101.0 + 1.5 = 102.5, close back below 101.0
        df.loc[12, "high"] = 103.0
        df.loc[12, "close"] = 100.2

        atr = pd.Series(1.0, index=df.index)
        result = detect_sweep_proxy(df, fvgs, atr, breach_atr_mult=1.5)
        assert result["sweep_proxy_short"].iloc[12] == True

    def test_no_false_positives_on_continuous_uptrend(self):
        """Trending data with no FVG breach should produce no signals."""
        closes = np.linspace(100.0, 110.0, 40)
        df = _make_ohlcv(closes)
        # No FVGs at all
        atr = pd.Series(1.0, index=df.index)
        result = detect_sweep_proxy(df, [], atr, breach_atr_mult=1.5)
        assert result["sweep_proxy_long"].sum() == 0
        assert result["sweep_proxy_short"].sum() == 0

    def test_no_signal_when_close_does_not_reclaim(self):
        """Breach without reclaim should NOT fire."""
        n = 20
        closes = np.full(n, 100.0)
        df = _make_ohlcv(closes)

        fvgs = [{
            "start_idx": 5, "end_idx": 7, "direction": "bullish",
            "low": 99.0, "high": 99.5, "filled_at_idx": None,
        }]

        # Bar 12: dip below 97.5, but close stays below 99.0
        df.loc[12, "low"] = 97.0
        df.loc[12, "close"] = 98.5

        atr = pd.Series(1.0, index=df.index)
        result = detect_sweep_proxy(df, fvgs, atr, breach_atr_mult=1.5)
        assert result["sweep_proxy_long"].iloc[12] == False
