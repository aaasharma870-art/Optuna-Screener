"""Tests for Deflated Sharpe Ratio."""

import pytest

from apex.validation.dsr import deflated_sharpe_ratio, _expected_max_sr


class TestExpectedMaxSR:

    def test_increases_with_trials(self):
        """Expected max SR should increase with more trials."""
        e10 = _expected_max_sr(1.0, 10)
        e100 = _expected_max_sr(1.0, 100)
        e1000 = _expected_max_sr(1.0, 1000)
        assert e10 < e100 < e1000

    def test_single_trial(self):
        assert _expected_max_sr(1.0, 1) == 0.0

    def test_positive(self):
        assert _expected_max_sr(0.5, 50) > 0.0


class TestDeflatedSharpeRatio:

    def test_result_in_0_1(self):
        dsr = deflated_sharpe_ratio(
            observed_sr=1.5, n_trials=100, sr_variance=0.5,
            skew=0.0, kurtosis=3.0, n_samples=252
        )
        assert 0.0 < dsr < 1.0

    def test_monotone_in_sr(self):
        """Higher observed SR should give higher DSR, all else equal."""
        common = dict(n_trials=50, sr_variance=0.3, skew=0.0, kurtosis=3.0, n_samples=252)
        d1 = deflated_sharpe_ratio(observed_sr=0.5, **common)
        d2 = deflated_sharpe_ratio(observed_sr=1.0, **common)
        d3 = deflated_sharpe_ratio(observed_sr=2.0, **common)
        assert d1 < d2 < d3

    def test_decreases_with_trials(self):
        """More trials (more multiple-testing) should deflate the ratio."""
        common = dict(observed_sr=1.5, sr_variance=0.5, skew=0.0, kurtosis=3.0, n_samples=252)
        d10 = deflated_sharpe_ratio(n_trials=10, **common)
        d100 = deflated_sharpe_ratio(n_trials=100, **common)
        d1000 = deflated_sharpe_ratio(n_trials=1000, **common)
        assert d10 > d100 > d1000

    def test_high_sr_high_dsr(self):
        """A very high Sharpe with few trials should give DSR close to 1."""
        dsr = deflated_sharpe_ratio(
            observed_sr=5.0, n_trials=5, sr_variance=0.1,
            skew=0.0, kurtosis=3.0, n_samples=1000
        )
        assert dsr > 0.95

    def test_low_sr_many_trials_low_dsr(self):
        """A mediocre Sharpe with many trials should give low DSR."""
        dsr = deflated_sharpe_ratio(
            observed_sr=0.3, n_trials=5000, sr_variance=0.5,
            skew=0.0, kurtosis=3.0, n_samples=252
        )
        assert dsr < 0.1
