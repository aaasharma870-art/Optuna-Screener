"""Tests for block-bootstrap synthetic price-path Monte Carlo."""

import numpy as np
import pandas as pd
import pytest

from apex.validation.synthetic_mc import synthetic_price_mc, passes_synthetic_gate


class TestSyntheticPriceMC:
    """Tests for synthetic_price_mc."""

    def test_output_shape(self):
        close = pd.Series(np.linspace(100, 120, 200))
        paths = synthetic_price_mc(close, n_paths=50, block_size=5, seed=42)
        assert paths.shape == (50, 200)

    def test_all_paths_start_at_close0(self):
        close = pd.Series(np.linspace(100, 130, 150))
        paths = synthetic_price_mc(close, n_paths=100, block_size=5, seed=42)
        np.testing.assert_allclose(paths[:, 0], close.iloc[0])

    def test_positive_prices(self):
        """All generated prices should be strictly positive."""
        close = pd.Series(np.linspace(50, 80, 300))
        paths = synthetic_price_mc(close, n_paths=200, block_size=5, seed=42)
        assert np.all(paths > 0)

    def test_determinism_with_seed(self):
        close = pd.Series(np.linspace(100, 110, 100))
        p1 = synthetic_price_mc(close, n_paths=20, block_size=5, seed=99)
        p2 = synthetic_price_mc(close, n_paths=20, block_size=5, seed=99)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self):
        close = pd.Series(np.linspace(100, 110, 100))
        p1 = synthetic_price_mc(close, n_paths=20, block_size=5, seed=1)
        p2 = synthetic_price_mc(close, n_paths=20, block_size=5, seed=2)
        assert not np.array_equal(p1, p2)

    def test_drift_preservation(self):
        """An upward-trending series should produce paths with positive median drift."""
        rng = np.random.RandomState(42)
        # Strong upward trend
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.01, 500))))
        paths = synthetic_price_mc(close, n_paths=500, block_size=10, seed=42)
        final_prices = paths[:, -1]
        # Median final price should exceed start
        assert np.median(final_prices) > close.iloc[0]


class TestPassesSyntheticGate:

    def test_passes_at_threshold(self):
        assert passes_synthetic_gate(0.20, min_pass_pct=20.0) is True

    def test_fails_below_threshold(self):
        assert passes_synthetic_gate(0.10, min_pass_pct=20.0) is False

    def test_passes_above_threshold(self):
        assert passes_synthetic_gate(0.50, min_pass_pct=20.0) is True
