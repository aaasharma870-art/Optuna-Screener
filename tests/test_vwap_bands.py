"""Tests for VWAP sigma bands."""

import numpy as np
import pandas as pd
import pytest

from apex.indicators.vwap_bands import compute_vwap_bands


def _make_ohlcv(n=100, seed=42):
    """Generate synthetic OHLCV data spanning 2 days."""
    rng = np.random.RandomState(seed)
    base = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    high = base + rng.uniform(0.2, 1.0, n)
    low = base - rng.uniform(0.2, 1.0, n)
    close = base + rng.uniform(-0.3, 0.3, n)
    opn = base + rng.uniform(-0.3, 0.3, n)
    volume = rng.randint(100, 10000, n).astype(float)
    # 2 days: first half day 1, second half day 2
    ts = pd.date_range("2024-01-15 09:30", periods=n, freq="5min")
    return pd.DataFrame({
        "timestamp": ts,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestVWAPBands:
    def test_columns_present(self):
        df = _make_ohlcv()
        result = compute_vwap_bands(df)
        expected_cols = [
            "vwap", "vwap_1s_upper", "vwap_1s_lower",
            "vwap_2s_upper", "vwap_2s_lower",
            "vwap_3s_upper", "vwap_3s_lower",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_bands_nest_correctly(self):
        """1-sigma bands inside 2-sigma inside 3-sigma."""
        df = _make_ohlcv(200)
        result = compute_vwap_bands(df)
        # Skip first bar of each session (sigma=0 there)
        mask = result["vwap_1s_upper"] != result["vwap"]
        r = result[mask]
        if len(r) > 0:
            assert (r["vwap_1s_upper"] <= r["vwap_2s_upper"] + 1e-10).all()
            assert (r["vwap_2s_upper"] <= r["vwap_3s_upper"] + 1e-10).all()
            assert (r["vwap_1s_lower"] >= r["vwap_2s_lower"] - 1e-10).all()
            assert (r["vwap_2s_lower"] >= r["vwap_3s_lower"] - 1e-10).all()

    def test_bands_center_on_vwap(self):
        """Upper and lower bands should be symmetric around VWAP."""
        df = _make_ohlcv()
        result = compute_vwap_bands(df)
        for n in (1, 2, 3):
            upper_dist = result[f"vwap_{n}s_upper"] - result["vwap"]
            lower_dist = result["vwap"] - result[f"vwap_{n}s_lower"]
            np.testing.assert_allclose(upper_dist, lower_dist, atol=1e-10)

    def test_session_reset_at_day_boundary(self):
        """VWAP should reset when the calendar day changes."""
        # Create data spanning exactly 2 days
        n_per_day = 50
        ts_day1 = pd.date_range("2024-01-15 09:30", periods=n_per_day, freq="5min")
        ts_day2 = pd.date_range("2024-01-16 09:30", periods=n_per_day, freq="5min")
        ts = ts_day1.append(ts_day2)

        rng = np.random.RandomState(99)
        n = len(ts)
        base = 100.0 + np.cumsum(rng.randn(n) * 0.5)
        df = pd.DataFrame({
            "timestamp": ts,
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base,
            "volume": rng.randint(100, 5000, n).astype(float),
        })

        result = compute_vwap_bands(df)

        # First bar of day2: VWAP should equal typical price of that bar
        day2_start = n_per_day
        tp_day2_first = (
            df["high"].iloc[day2_start]
            + df["low"].iloc[day2_start]
            + df["close"].iloc[day2_start]
        ) / 3.0
        assert abs(result["vwap"].iloc[day2_start] - tp_day2_first) < 1e-10
