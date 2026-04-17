"""Tests for Combinatorial Purged Cross-Validation."""

import math

import numpy as np
import pytest

from apex.validation.cpcv import cpcv_split


class TestCPCV:
    """Tests for cpcv_split."""

    def test_fold_count_8_2(self):
        """C(8, 2) = 28 folds."""
        folds = list(cpcv_split(1000, n_blocks=8, n_test_blocks=2, purge_bars=10))
        assert len(folds) == math.comb(8, 2)  # 28

    def test_fold_count_6_2(self):
        folds = list(cpcv_split(600, n_blocks=6, n_test_blocks=2, purge_bars=5))
        assert len(folds) == math.comb(6, 2)  # 15

    def test_no_overlap_train_test(self):
        """Train and test indices must not overlap in any fold."""
        for train, test in cpcv_split(500, n_blocks=8, n_test_blocks=2, purge_bars=10):
            overlap = np.intersect1d(train, test)
            assert len(overlap) == 0, f"Overlap found: {overlap[:5]}..."

    def test_purge_enforced(self):
        """Bars within purge_bars of test block boundaries must be absent from train."""
        n_bars = 800
        purge = 10
        for train, test in cpcv_split(n_bars, n_blocks=8, n_test_blocks=2, purge_bars=purge):
            train_set = set(train)
            test_sorted = np.sort(test)
            test_min, test_max = test_sorted[0], test_sorted[-1]
            # Check bars just before test start are purged
            for offset in range(1, purge + 1):
                idx = test_min - offset
                if 0 <= idx < n_bars:
                    assert idx not in train_set, f"Bar {idx} should be purged (before test start {test_min})"

    def test_test_indices_within_bounds(self):
        n_bars = 400
        for train, test in cpcv_split(n_bars, n_blocks=8, n_test_blocks=2, purge_bars=5):
            assert np.all(test >= 0)
            assert np.all(test < n_bars)
            assert np.all(train >= 0)
            assert np.all(train < n_bars)

    def test_small_dataset(self):
        """Should not crash on a very small dataset."""
        folds = list(cpcv_split(5, n_blocks=8, n_test_blocks=2, purge_bars=2))
        assert len(folds) >= 1  # degenerate case yields at least 1 fold

    def test_all_bars_covered_across_folds(self):
        """Each bar should appear in at least one test set across all folds."""
        n_bars = 800
        seen = set()
        for _, test in cpcv_split(n_bars, n_blocks=8, n_test_blocks=2, purge_bars=5):
            seen.update(test.tolist())
        assert seen == set(range(n_bars))
