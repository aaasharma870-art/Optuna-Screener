"""Combinatorial Purged Cross-Validation (CPCV)."""

from itertools import combinations
from typing import Iterator, Tuple

import numpy as np


def cpcv_split(n_bars: int, n_blocks: int = 8, n_test_blocks: int = 2,
               purge_bars: int = 10) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` for each C(n_blocks, n_test_blocks) fold.

    Parameters
    ----------
    n_bars : int
        Total number of bars (rows) in the dataset.
    n_blocks : int
        Number of contiguous blocks to partition the data into.
    n_test_blocks : int
        Number of blocks held out for testing in each fold.
    purge_bars : int
        Number of bars to remove from the training set at the boundary
        of each test block (both before and after) to prevent leakage.

    Yields
    ------
    (train_idx, test_idx) : tuple of np.ndarray
        Integer index arrays into ``range(n_bars)``.
    """
    if n_bars < n_blocks:
        # Degenerate case: fewer bars than blocks
        all_idx = np.arange(n_bars)
        yield all_idx, all_idx
        return

    # Build block boundaries
    edges = np.linspace(0, n_bars, n_blocks + 1, dtype=int)
    block_ranges = [(edges[i], edges[i + 1]) for i in range(n_blocks)]

    for test_combo in combinations(range(n_blocks), n_test_blocks):
        test_set = set()
        purge_set = set()

        for blk in test_combo:
            start, end = block_ranges[blk]
            test_set.update(range(start, end))

            # Purge zone: bars just before the test block
            purge_start = max(0, start - purge_bars)
            purge_set.update(range(purge_start, start))

            # Purge zone: bars just after the test block
            purge_end = min(n_bars, end + purge_bars)
            purge_set.update(range(end, purge_end))

        # Remove purge bars that are inside the test set (they stay in test)
        purge_set -= test_set

        train_set = set(range(n_bars)) - test_set - purge_set
        yield np.sort(np.array(list(train_set))), np.sort(np.array(list(test_set)))
