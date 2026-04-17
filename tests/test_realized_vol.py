"""Tests for apex.regime.realized_vol."""

import numpy as np
import pandas as pd
import pytest

from apex.regime.realized_vol import compute_realized_vol_20d


class TestRealizedVol:
    def test_constant_close_gives_zero_vol(self):
        """Constant prices should produce zero (or near-zero) realized vol."""
        close = pd.Series([100.0] * 50)
        rv = compute_realized_vol_20d(close, window=20)
        # First 20 values NaN (window=20 + 1 for log return)
        valid = rv.dropna()
        assert len(valid) > 0
        assert all(v == 0.0 or np.isclose(v, 0.0, atol=1e-12) for v in valid)

    def test_known_stdev_series(self):
        """A series with known daily moves should produce annualized vol in expected range."""
        np.random.seed(42)
        # Generate 300 prices with ~1% daily moves (annualized ~15.87%)
        daily_returns = np.random.normal(0, 0.01, 300)
        prices = [100.0]
        for r in daily_returns:
            prices.append(prices[-1] * np.exp(r))
        close = pd.Series(prices)
        rv = compute_realized_vol_20d(close, window=20)
        valid = rv.dropna()
        assert len(valid) > 0
        # Mean annualized vol should be roughly 0.01 * sqrt(252) ~ 0.159
        mean_rv = valid.mean()
        assert 0.08 < mean_rv < 0.30, f"Mean RV {mean_rv} outside expected range"

    def test_first_window_values_are_nan(self):
        """First `window` values should be NaN."""
        close = pd.Series(np.linspace(100, 120, 50))
        rv = compute_realized_vol_20d(close, window=20)
        # Index 0: NaN from log return shift
        # Indices 1..19: rolling(20) needs 20 points, so indices 1-19 are NaN
        # Index 20: first valid value (log returns 1..20 = 20 points)
        assert pd.isna(rv.iloc[0])
        assert pd.isna(rv.iloc[19])
        assert pd.notna(rv.iloc[20])  # first valid

    def test_output_length_matches_input(self):
        """Output series should have the same length as input."""
        close = pd.Series(np.linspace(100, 150, 100))
        rv = compute_realized_vol_20d(close, window=20)
        assert len(rv) == len(close)
