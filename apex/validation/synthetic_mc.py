"""Block-bootstrap price-path Monte Carlo simulation."""

import math

import numpy as np
import pandas as pd


def synthetic_price_mc(close: pd.Series, n_paths: int = 1000,
                       block_size: int = 5, seed: int = 42) -> np.ndarray:
    """Generate *n_paths* synthetic close-price histories via block bootstrap.

    Algorithm
    ---------
    1. Compute log-returns: ``log(close).diff().dropna()``.
    2. For each path, sample ``ceil(len / block_size)`` overlapping blocks
       (with replacement) from the log-returns, concatenate, trim to length,
       then reconstruct prices via ``exp(cumsum) * close[0]``.
    3. Every path starts at ``close.iloc[0]``.

    Returns
    -------
    np.ndarray of shape ``(n_paths, len(close))``.
    """
    close_arr = np.asarray(close, dtype=np.float64)
    n = len(close_arr)
    if n < 2:
        return np.full((n_paths, max(n, 1)), close_arr[0] if n else 0.0)

    log_ret = np.diff(np.log(close_arr))  # length n-1
    T = len(log_ret)
    n_blocks = math.ceil(T / block_size)

    rng = np.random.RandomState(seed)
    paths = np.empty((n_paths, n), dtype=np.float64)
    paths[:, 0] = close_arr[0]

    for i in range(n_paths):
        # sample block start indices
        starts = rng.randint(0, T - block_size + 1, size=n_blocks) if T >= block_size else rng.randint(0, max(T, 1), size=n_blocks)
        blocks = []
        for s in starts:
            end = min(s + block_size, T)
            blocks.append(log_ret[s:end])
        sampled = np.concatenate(blocks)[:T]
        cum = np.cumsum(sampled)
        paths[i, 1:] = close_arr[0] * np.exp(cum)

    return paths


def passes_synthetic_gate(profitable_fraction: float, min_pass_pct: float = 20.0) -> bool:
    """Return True if ``profitable_fraction * 100 >= min_pass_pct``."""
    return profitable_fraction * 100.0 >= min_pass_pct
