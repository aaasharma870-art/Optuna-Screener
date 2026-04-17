"""Tests for the borrow-fee model."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from apex.engine.fees import borrow_fee, borrow_fee_from_bars, lookup_borrow_rate


class TestBorrowFee:
    def test_zero_days_returns_zero(self):
        assert borrow_fee(100.0, 0.05, 0.0) == 0.0

    def test_linear_in_days(self):
        fee1 = borrow_fee(100.0, 0.02, 10)
        fee2 = borrow_fee(100.0, 0.02, 20)
        assert abs(fee2 - 2 * fee1) < 1e-10

    def test_linear_in_price(self):
        fee1 = borrow_fee(100.0, 0.02, 10)
        fee2 = borrow_fee(200.0, 0.02, 10)
        assert abs(fee2 - 2 * fee1) < 1e-10

    def test_known_value(self):
        # 100 * 0.05 * 252 / 252 = 5.0
        assert abs(borrow_fee(100.0, 0.05, 252) - 5.0) < 1e-10


class TestBorrowFeeFromBars:
    def test_whole_day(self):
        # 7 bars = 1 day
        fee = borrow_fee_from_bars(100.0, 0.0252, 7, bars_per_day=7)
        expected = borrow_fee(100.0, 0.0252, 1.0)
        assert abs(fee - expected) < 1e-10

    def test_fractional_bars(self):
        # 3 bars at 7 bars/day = 3/7 day
        fee = borrow_fee_from_bars(100.0, 0.05, 3, bars_per_day=7)
        expected = borrow_fee(100.0, 0.05, 3.0 / 7.0)
        assert abs(fee - expected) < 1e-10

    def test_zero_bars(self):
        assert borrow_fee_from_bars(100.0, 0.05, 0, bars_per_day=7) == 0.0


class TestLookupBorrowRate:
    def test_specific_symbol(self):
        rates = {"default": 0.02, "TSLA": 0.05}
        assert lookup_borrow_rate("TSLA", rates) == 0.05

    def test_fallback_to_default(self):
        rates = {"default": 0.02, "TSLA": 0.05}
        assert lookup_borrow_rate("AAPL", rates) == 0.02

    def test_no_default_returns_zero(self):
        rates = {"TSLA": 0.05}
        assert lookup_borrow_rate("AAPL", rates) == 0.0
