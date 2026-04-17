"""Tests for regime-specific fitness functions."""

import math
import pytest

from apex.optimize.fitness import (
    suppressed_fitness,
    amplified_fitness,
    compute_regime_fitness,
    SUPPRESSED_REGIMES,
    AMPLIFIED_REGIMES,
    MIN_DD_CAP,
    MIN_LOSS_CAP,
)


class TestSuppressedFitness:
    """Tests for suppressed_fitness formula."""

    def test_known_value(self):
        """60^2 * 1.8 = 6480."""
        result = suppressed_fitness(60.0, 1.8)
        assert result == pytest.approx(6480.0)

    def test_zero_pf_returns_zero(self):
        result = suppressed_fitness(50.0, 0.0)
        assert result == 0.0

    def test_negative_pf_returns_zero(self):
        result = suppressed_fitness(50.0, -1.5)
        assert result == 0.0

    def test_high_win_rate(self):
        """90^2 * 2.0 = 16200."""
        result = suppressed_fitness(90.0, 2.0)
        assert result == pytest.approx(16200.0)


class TestAmplifiedFitness:
    """Tests for amplified_fitness formula."""

    def test_known_value(self):
        """(30/10) * (5/2.5) = 3.0 * 2.0 = 6.0."""
        result = amplified_fitness(30.0, 10.0, 5.0, 2.5)
        assert result == pytest.approx(6.0)

    def test_zero_dd_uses_cap(self):
        """dd=0 -> denom = MIN_DD_CAP = 0.5."""
        result = amplified_fitness(10.0, 0.0, 2.0, 1.0)
        expected = (10.0 / MIN_DD_CAP) * (2.0 / 1.0)
        assert result == pytest.approx(expected)

    def test_zero_avg_loss_uses_cap(self):
        """avg_loss=0 -> denom = MIN_LOSS_CAP = 0.1."""
        result = amplified_fitness(10.0, 5.0, 2.0, 0.0)
        expected = (10.0 / 5.0) * (2.0 / MIN_LOSS_CAP)
        assert result == pytest.approx(expected)

    def test_negative_dd_uses_absolute(self):
        """Negative dd -> abs taken."""
        result = amplified_fitness(20.0, -8.0, 4.0, 2.0)
        expected = (20.0 / 8.0) * (4.0 / 2.0)
        assert result == pytest.approx(expected)


class TestComputeRegimeFitness:
    """Tests for regime dispatch."""

    def test_r1_dispatches_to_suppressed(self):
        stats = {"wr_pct": 60.0, "pf": 1.8}
        result = compute_regime_fitness("R1", stats)
        assert result == pytest.approx(6480.0)

    def test_r2_dispatches_to_suppressed(self):
        stats = {"wr_pct": 50.0, "pf": 1.5}
        result = compute_regime_fitness("R2", stats)
        assert result == pytest.approx(50.0 ** 2 * 1.5)

    def test_r3_dispatches_to_amplified(self):
        stats = {
            "total_return_pct": 30.0,
            "max_dd_pct": 10.0,
            "avg_win": 5.0,
            "avg_loss": 2.5,
        }
        result = compute_regime_fitness("R3", stats)
        assert result == pytest.approx(6.0)

    def test_contango_calm_is_suppressed(self):
        assert "Contango_Calm" in SUPPRESSED_REGIMES

    def test_backwardation_elevated_is_amplified(self):
        assert "Backwardation_Elevated" in AMPLIFIED_REGIMES

    def test_unknown_regime_uses_legacy(self):
        """Unknown regime -> PF * sqrt(trades) * (1 - dd/100)."""
        stats = {"pf": 2.0, "trades": 16, "max_dd_pct": 10.0}
        result = compute_regime_fitness("UNKNOWN_REGIME", stats)
        expected = 2.0 * math.sqrt(16) * (1.0 - 10.0 / 100.0)
        assert result == pytest.approx(expected)

    def test_unknown_regime_zero_trades(self):
        stats = {"pf": 2.0, "trades": 0, "max_dd_pct": 5.0}
        result = compute_regime_fitness("UNKNOWN_REGIME", stats)
        assert result == 0.0

    def test_unknown_regime_zero_pf(self):
        stats = {"pf": 0.0, "trades": 10, "max_dd_pct": 5.0}
        result = compute_regime_fitness("UNKNOWN_REGIME", stats)
        assert result == 0.0
