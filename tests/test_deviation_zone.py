"""Tests for compute_deviation_zone added in Phase 6."""

import numpy as np
import pandas as pd
import pytest

from apex.indicators.vwap_bands import compute_vwap_bands, compute_deviation_zone


def _make_ohlcv(closes, seed=42):
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


class TestDeviationZone:
    def test_columns_present(self):
        # Need some price variation so sigma > 0
        rng = np.random.RandomState(42)
        closes = 100.0 + np.cumsum(rng.randn(80) * 0.5)
        df = _make_ohlcv(closes)
        df = compute_vwap_bands(df)
        result = compute_deviation_zone(df, deviation_sigma=1.5, pullback_sigma=0.5)
        assert "in_deviation_zone_long" in result.columns
        assert "in_deviation_zone_short" in result.columns
        assert "in_pullback_zone" in result.columns

    def test_requires_vwap_bands(self):
        """Calling without compute_vwap_bands first should error clearly."""
        rng = np.random.RandomState(42)
        closes = 100.0 + np.cumsum(rng.randn(40) * 0.5)
        df = _make_ohlcv(closes)
        with pytest.raises(ValueError, match="compute_vwap_bands"):
            compute_deviation_zone(df)

    def test_long_zone_when_close_below_dev_band(self):
        """When close < vwap - 1.5*sigma, in_deviation_zone_long fires."""
        rng = np.random.RandomState(7)
        closes = 100.0 + np.cumsum(rng.randn(80) * 0.5)
        df = _make_ohlcv(closes)
        df = compute_vwap_bands(df)
        result = compute_deviation_zone(df, deviation_sigma=1.5, pullback_sigma=0.5)

        sigma = result["vwap_1s_upper"] - result["vwap"]
        manual_long = result["close"] < (result["vwap"] - 1.5 * sigma)
        # boolean equality (manual definition matches)
        pd.testing.assert_series_equal(
            result["in_deviation_zone_long"].rename(None),
            manual_long.rename(None),
            check_names=False,
        )

    def test_short_zone_when_close_above_dev_band(self):
        """When close > vwap + 1.5*sigma, in_deviation_zone_short fires."""
        rng = np.random.RandomState(7)
        closes = 100.0 + np.cumsum(rng.randn(80) * 0.5)
        df = _make_ohlcv(closes)
        df = compute_vwap_bands(df)
        result = compute_deviation_zone(df, deviation_sigma=1.5, pullback_sigma=0.5)

        sigma = result["vwap_1s_upper"] - result["vwap"]
        manual_short = result["close"] > (result["vwap"] + 1.5 * sigma)
        pd.testing.assert_series_equal(
            result["in_deviation_zone_short"].rename(None),
            manual_short.rename(None),
            check_names=False,
        )

    def test_pullback_zone_when_close_within_half_sigma(self):
        """|close - vwap| <= 0.5*sigma -> in_pullback_zone."""
        rng = np.random.RandomState(7)
        closes = 100.0 + np.cumsum(rng.randn(80) * 0.5)
        df = _make_ohlcv(closes)
        df = compute_vwap_bands(df)
        result = compute_deviation_zone(df, deviation_sigma=1.5, pullback_sigma=0.5)

        sigma = result["vwap_1s_upper"] - result["vwap"]
        manual_pullback = (result["close"] - result["vwap"]).abs() <= (0.5 * sigma)
        pd.testing.assert_series_equal(
            result["in_pullback_zone"].rename(None),
            manual_pullback.rename(None),
            check_names=False,
        )

    def test_extreme_below_triggers_long_zone(self):
        """Inject a deep downward spike and confirm in_deviation_zone_long fires."""
        rng = np.random.RandomState(11)
        closes = 100.0 + np.cumsum(rng.randn(60) * 0.3)
        # Inject a sharp downward bar near the end
        closes = closes.copy()
        closes[55] -= 5.0
        df = _make_ohlcv(closes)
        df = compute_vwap_bands(df)
        result = compute_deviation_zone(df, deviation_sigma=1.5)
        # The injected bar should land in long zone
        assert result["in_deviation_zone_long"].iloc[55] == True
