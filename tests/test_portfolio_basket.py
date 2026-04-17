"""Tests for cross-asset basket momentum alignment."""

import pandas as pd
import numpy as np
import pytest

from apex.engine.portfolio import compute_basket_alignment


def _make_basket_df(prices, start="2024-01-01"):
    """Build a daily DataFrame from a list of close prices."""
    dates = pd.date_range(start, periods=len(prices), freq="B")
    return pd.DataFrame({
        "datetime": dates,
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": [1_000_000] * len(prices),
    })


def _rising_prices(n=100, start=100.0, daily_ret=0.002):
    """Generate n rising prices."""
    return [start * (1 + daily_ret) ** i for i in range(n)]


def _falling_prices(n=100, start=100.0, daily_ret=-0.002):
    """Generate n falling prices."""
    return [start * (1 + daily_ret) ** i for i in range(n)]


def _flat_prices(n=100, start=100.0):
    """Generate n flat prices."""
    return [start] * n


class TestBasketAlignment:
    """Tests for compute_basket_alignment."""

    def test_all_positive_drift_triggers_boost(self):
        """All 5 assets rising -> alignment >= 3 -> boost = 1.25."""
        basket = {
            "SPY": _make_basket_df(_rising_prices()),
            "QQQ": _make_basket_df(_rising_prices()),
            "GLD": _make_basket_df(_rising_prices()),
            "USO": _make_basket_df(_rising_prices()),
            "IEF": _make_basket_df(_rising_prices()),
        }
        as_of = pd.Timestamp("2024-06-15")
        result = compute_basket_alignment(basket, as_of)
        assert result == 1.25

    def test_split_drift_no_boost(self):
        """2 rising, 2 falling, 1 flat -> max(2,2) < 3 -> no boost."""
        basket = {
            "SPY": _make_basket_df(_rising_prices()),
            "QQQ": _make_basket_df(_rising_prices()),
            "GLD": _make_basket_df(_falling_prices()),
            "USO": _make_basket_df(_falling_prices()),
            "IEF": _make_basket_df(_flat_prices()),
        }
        as_of = pd.Timestamp("2024-06-15")
        result = compute_basket_alignment(basket, as_of)
        assert result == 1.0

    def test_three_aligned_triggers_boost(self):
        """3 rising, 2 falling -> max(3,2) >= 3 -> boost."""
        basket = {
            "SPY": _make_basket_df(_rising_prices()),
            "QQQ": _make_basket_df(_rising_prices()),
            "GLD": _make_basket_df(_rising_prices()),
            "USO": _make_basket_df(_falling_prices()),
            "IEF": _make_basket_df(_falling_prices()),
        }
        as_of = pd.Timestamp("2024-06-15")
        result = compute_basket_alignment(basket, as_of)
        assert result == 1.25

    def test_uses_shift_minus_one(self):
        """Data on as_of date itself must NOT be used (look-ahead safe)."""
        prices = _rising_prices(100)
        basket = {
            "SPY": _make_basket_df(prices),
            "QQQ": _make_basket_df(prices),
            "GLD": _make_basket_df(prices),
        }
        # Set as_of to exactly the first date -> no data strictly before
        first_date = basket["SPY"]["datetime"].iloc[0]
        result = compute_basket_alignment(basket, first_date, alignment_threshold=2)
        assert result == 1.0  # Not enough historical data

    def test_custom_threshold_and_multiplier(self):
        """Custom alignment_threshold=2, size_multiplier=1.5."""
        basket = {
            "SPY": _make_basket_df(_rising_prices()),
            "QQQ": _make_basket_df(_rising_prices()),
            "GLD": _make_basket_df(_falling_prices()),
        }
        as_of = pd.Timestamp("2024-06-15")
        result = compute_basket_alignment(
            basket, as_of, alignment_threshold=2, size_multiplier=1.5
        )
        assert result == 1.5

    def test_empty_basket(self):
        """Empty basket -> no alignment -> 1.0."""
        result = compute_basket_alignment({}, pd.Timestamp("2024-06-15"))
        assert result == 1.0

    def test_all_negative_drift_triggers_boost(self):
        """All negative -> negative_count >= threshold -> boost."""
        basket = {
            "SPY": _make_basket_df(_falling_prices()),
            "QQQ": _make_basket_df(_falling_prices()),
            "GLD": _make_basket_df(_falling_prices()),
            "USO": _make_basket_df(_falling_prices()),
            "IEF": _make_basket_df(_falling_prices()),
        }
        as_of = pd.Timestamp("2024-06-15")
        result = compute_basket_alignment(basket, as_of)
        assert result == 1.25
