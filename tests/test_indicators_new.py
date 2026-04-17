"""Tests for RSI2 and other new indicator verifications."""

import numpy as np
import pandas as pd
import pytest

from apex.indicators.basics import compute_rsi


class TestRSI2:
    def test_rsi2_extreme_values(self):
        """RSI with period=2 should produce extreme readings on strongly trending series."""
        # Strongly rising close with tiny dips so avg_loss is nonzero
        vals = []
        v = 100.0
        for i in range(50):
            if i % 10 == 5:
                v -= 0.01  # tiny dip
            else:
                v += 1.0
            vals.append(v)
        close = pd.Series(vals)
        rsi2 = compute_rsi(close, period=2)
        assert rsi2.iloc[-1] > 90

        # Strongly falling close with tiny bounces
        vals_down = []
        v = 200.0
        for i in range(50):
            if i % 10 == 5:
                v += 0.01  # tiny bounce
            else:
                v -= 1.0
            vals_down.append(v)
        close_down = pd.Series(vals_down)
        rsi2_down = compute_rsi(close_down, period=2)
        assert rsi2_down.iloc[-1] < 10
