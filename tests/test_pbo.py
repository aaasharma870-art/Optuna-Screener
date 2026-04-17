"""Tests for Probability of Backtest Overfitting."""

import numpy as np
import pytest

from apex.validation.pbo import probability_of_backtest_overfitting


class TestPBO:

    def test_random_scores_pbo_near_half(self):
        """Random IS/OOS should yield PBO near 0.5."""
        rng = np.random.RandomState(42)
        n_trials, n_folds = 50, 20
        is_scores = rng.randn(n_trials, n_folds)
        oos_scores = rng.randn(n_trials, n_folds)
        pbo = probability_of_backtest_overfitting(is_scores, oos_scores)
        # Should be roughly 0.5 +/- 0.3
        assert 0.2 <= pbo <= 0.8

    def test_monotonic_pbo_near_zero(self):
        """When IS ranking perfectly predicts OOS, PBO should be near 0."""
        rng = np.random.RandomState(42)
        n_trials, n_folds = 30, 10
        # IS and OOS are the same (perfect alignment)
        scores = rng.randn(n_trials, n_folds)
        pbo = probability_of_backtest_overfitting(scores, scores)
        assert pbo <= 0.1

    def test_inverted_pbo_high(self):
        """When IS-best is OOS-worst, PBO should be high."""
        rng = np.random.RandomState(42)
        n_trials, n_folds = 30, 10
        is_scores = rng.randn(n_trials, n_folds)
        # Invert: best IS -> worst OOS
        oos_scores = -is_scores
        pbo = probability_of_backtest_overfitting(is_scores, oos_scores)
        assert pbo > 0.5

    def test_result_in_0_1(self):
        rng = np.random.RandomState(123)
        is_scores = rng.randn(20, 8)
        oos_scores = rng.randn(20, 8)
        pbo = probability_of_backtest_overfitting(is_scores, oos_scores)
        assert 0.0 <= pbo <= 1.0

    def test_single_trial(self):
        """Edge case: only one trial, no overfitting possible."""
        is_scores = np.array([[1.0, 2.0, 3.0]])
        oos_scores = np.array([[0.5, 1.5, 2.5]])
        pbo = probability_of_backtest_overfitting(is_scores, oos_scores)
        assert pbo == 0.0
