"""Tests for VWAP slope columns added in Phase 6."""

import numpy as np
import pandas as pd
import pytest

from apex.indicators.vwap_bands import compute_vwap_bands


def _make_ohlcv(closes, seed=42):
    """Build a synthetic OHLCV DataFrame from a closing-price sequence."""
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    rng = np.random.RandomState(seed)
    high = closes + rng.uniform(0.05, 0.2, n)
    low = closes - rng.uniform(0.05, 0.2, n)
    opn = closes.copy()
    volume = np.full(n, 1_000.0)
    ts = pd.date_range("2024-01-15 09:30", periods=n, freq="5min")
    return pd.DataFrame({
        "timestamp": ts,
        "open": opn,
        "high": high,
        "low": low,
        "close": closes,
        "volume": volume,
    })


class TestVWAPSlope:
    def test_columns_present(self):
        df = _make_ohlcv(np.full(60, 100.0))
        result = compute_vwap_bands(df, slope_window=5)
        assert "vwap_slope" in result.columns
        assert "vwap_slope_atr" in result.columns

    def test_slope_flat_close_to_zero(self):
        """Constant price -> VWAP slope ~ 0."""
        df = _make_ohlcv(np.full(80, 100.0))
        result = compute_vwap_bands(df, slope_window=5)
        # Skip leading NaNs from diff
        slopes = result["vwap_slope"].dropna()
        assert len(slopes) > 0
        # Allow tiny noise from non-uniform high/low
        assert slopes.abs().max() < 0.5

    def test_slope_positive_on_rising_data(self):
        """Monotonically rising prices -> positive slope on later bars."""
        closes = np.linspace(100.0, 120.0, 80)
        df = _make_ohlcv(closes)
        result = compute_vwap_bands(df, slope_window=5)
        slopes = result["vwap_slope"].dropna()
        # Use later bars (after VWAP has caught up to trend)
        late_slopes = slopes.iloc[20:]
        assert (late_slopes > 0).all(), f"Expected all positive slopes, got {late_slopes.tolist()}"

    def test_slope_negative_on_falling_data(self):
        """Monotonically falling prices -> negative slope on later bars."""
        closes = np.linspace(120.0, 100.0, 80)
        df = _make_ohlcv(closes)
        result = compute_vwap_bands(df, slope_window=5)
        slopes = result["vwap_slope"].dropna()
        late_slopes = slopes.iloc[20:]
        assert (late_slopes < 0).all(), f"Expected all negative slopes, got {late_slopes.tolist()}"

    def test_slope_atr_dimensionless(self):
        """vwap_slope_atr should be a small dimensionless ratio."""
        closes = np.linspace(100.0, 110.0, 80)
        df = _make_ohlcv(closes)
        result = compute_vwap_bands(df, slope_window=5)
        ratios = result["vwap_slope_atr"].dropna()
        assert len(ratios) > 0
        # Order-of-magnitude check: should be within +/- 100 (same scale as ATR)
        assert ratios.abs().max() < 100.0
