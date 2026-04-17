"""Tests for VWCLV indicator."""

import numpy as np
import pandas as pd
import pytest

from apex.indicators.vwclv import compute_vwclv


def _make_df(**overrides):
    """Create a small OHLCV DataFrame with optional overrides."""
    n = overrides.pop("n", 30)
    rng = np.random.RandomState(42)
    base = 100.0 + np.cumsum(rng.randn(n) * 0.3)
    data = {
        "open": base,
        "high": base + 1.0,
        "low": base - 1.0,
        "close": base + 0.5,
        "volume": np.full(n, 1000.0),
    }
    data.update(overrides)
    return pd.DataFrame(data)


class TestVWCLV:
    def test_close_at_high_positive(self):
        """When close == high, CLV should be 1.0, vwclv should be positive."""
        df = _make_df()
        df["close"] = df["high"]  # close at high
        result = compute_vwclv(df)
        # CLV = (high - low) / (high - low) = 1.0
        assert (result["clv"] == 1.0).all()
        # vwclv = (2*1 - 1) * weight = +weight
        assert (result["vwclv"] > 0).all()

    def test_close_at_low_negative(self):
        """When close == low, CLV should be 0.0, vwclv should be negative."""
        df = _make_df()
        df["close"] = df["low"]  # close at low
        result = compute_vwclv(df)
        assert (result["clv"] == 0.0).all()
        # vwclv = (2*0 - 1) * weight = -weight
        assert (result["vwclv"] < 0).all()

    def test_zero_range_neutral(self):
        """When high == low (zero range), CLV should be 0.5 and vwclv ~0."""
        df = _make_df()
        df["high"] = df["close"]
        df["low"] = df["close"]
        result = compute_vwclv(df)
        np.testing.assert_allclose(result["clv"], 0.5, atol=1e-10)
        np.testing.assert_allclose(result["vwclv"], 0.0, atol=1e-10)

    def test_cum_vwclv_is_rolling_sum(self):
        """cum_vwclv should equal rolling sum of vwclv."""
        df = _make_df(n=50)
        window = 5
        result = compute_vwclv(df, cumulative_window=window)
        expected = result["vwclv"].rolling(window, min_periods=1).sum()
        np.testing.assert_allclose(result["cum_vwclv"], expected, atol=1e-10)
