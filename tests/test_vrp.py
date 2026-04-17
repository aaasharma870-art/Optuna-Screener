"""Tests for apex.regime.vrp."""

import numpy as np
import pandas as pd
import pytest

from apex.regime.vrp import compute_vrp, compute_vrp_percentile


class TestVRPPercentile:
    def test_percentile_in_range(self):
        """VRP percentile values should be in [0, 100]."""
        np.random.seed(42)
        vrp = pd.Series(np.random.randn(400))
        pct = compute_vrp_percentile(vrp, window=252)
        valid = pct.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0

    def test_warmup_nan(self):
        """Values during the warmup period should be NaN."""
        vrp = pd.Series(np.random.randn(300))
        pct = compute_vrp_percentile(vrp, window=252)
        # First window+1 values should be NaN
        assert all(pd.isna(pct.iloc[i]) for i in range(253))

    def test_monotonic_series_high_percentile(self):
        """A monotonically increasing series should end near 100th percentile."""
        vrp = pd.Series(np.linspace(0, 100, 400))
        pct = compute_vrp_percentile(vrp, window=252)
        valid = pct.dropna()
        # Last value should be near 100 (it's the largest in its lookback)
        assert valid.iloc[-1] >= 95.0

    def test_excludes_current_bar(self):
        """Percentile should exclude the current bar from ranking."""
        # All same values except last is much higher
        data = [5.0] * 300 + [1000.0]
        vrp = pd.Series(data)
        pct = compute_vrp_percentile(vrp, window=252)
        # Last bar is 1000, lookback is all 5.0 -> should rank at 100
        last_valid = pct.iloc[-1]
        assert pd.notna(last_valid)
        assert last_valid == 100.0


class TestComputeVRP:
    def test_vrp_columns(self):
        """compute_vrp should return DataFrame with expected columns."""
        np.random.seed(42)
        close = pd.Series(np.cumsum(np.random.randn(400)) + 200)
        iv = pd.Series(np.random.uniform(15, 25, 400))
        result = compute_vrp(iv, close, rv_window=20, pct_window=252)
        assert list(result.columns) == ["iv", "rv", "vrp_raw", "vrp_pct"]
        assert len(result) == 400

    def test_vrp_pct_bounded(self):
        """VRP percentile should be in [0, 100] where not NaN."""
        np.random.seed(42)
        close = pd.Series(np.cumsum(np.random.randn(400)) + 200)
        iv = pd.Series(np.random.uniform(15, 25, 400))
        result = compute_vrp(iv, close, rv_window=20, pct_window=252)
        valid_pct = result["vrp_pct"].dropna()
        if len(valid_pct) > 0:
            assert valid_pct.min() >= 0.0
            assert valid_pct.max() <= 100.0
