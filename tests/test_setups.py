"""Tests for VRP setup detectors (Phase 6 Gap 3 + Gap 4)."""

import numpy as np
import pandas as pd
import pytest

from apex.engine.setups import detect_breakout_reversal
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
        closes = np.full(40, 100.0)
        closes[30:35] = [105.0, 106.0, 107.0, 106.0, 105.0]
        closes[35] = 102.0  # close back BELOW the window max (107)
        df = _make_ohlcv(closes)
        df = compute_vwap_bands(df)
        atr = pd.Series(1.0, index=df.index)
        result = detect_breakout_reversal(df, atr, lookback=5, breakout_atr_mult=1.5)
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

        Reversal-long requires close[i] < max(close[i-lookback..i-1]); on a
        monotonic ascending series close[i] is the max, so it never fires.
        """
        closes = np.linspace(100.0, 130.0, 60)
        df = _make_ohlcv(closes)
        df = compute_vwap_bands(df)
        atr = pd.Series(0.5, index=df.index)
        result = detect_breakout_reversal(df, atr, lookback=5, breakout_atr_mult=1.5)
        assert result["breakout_reversal_long"].sum() == 0

    def test_requires_vwap_column(self):
        df = _make_ohlcv(np.full(20, 100.0))
        atr = pd.Series(1.0, index=df.index)
        with pytest.raises(ValueError, match="vwap"):
            detect_breakout_reversal(df, atr)
