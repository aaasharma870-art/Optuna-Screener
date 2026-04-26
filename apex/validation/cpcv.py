"""Combinatorial Purged Cross-Validation (CPCV)."""

from itertools import combinations
from typing import Iterator, Tuple

import numpy as np
import pandas as pd


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


def evaluate_params_via_cpcv(symbol: str, df: pd.DataFrame, daily_df,
                             architecture: dict, best_params: dict,
                             n_blocks: int = 8, n_test_blocks: int = 2,
                             purge_bars: int = 10) -> dict:
    """Evaluate ``best_params`` across all C(n_blocks, n_test_blocks) folds.

    For each fold:
      * ``test_idx`` is the held-out portion (~25% of bars by default)
      * ``full_backtest`` is run on ``df.iloc[test_idx]`` with ``best_params``
      * OOS Sharpe, total-return, profit-factor, and trade count are recorded

    Returns
    -------
    dict
        ``{
            'n_folds': int,
            'oos_sharpes': list[float],
            'oos_returns': list[float],
            'oos_trades': list[int],
            'sharpe_median': float,
            'sharpe_iqr': (q25, q75),
            'sharpe_pct_positive': float,
            'mean_oos_pf': float,
            'mean_oos_return': float,
        }``

        On insufficient data or all-fold failure, returns
        ``{'n_folds': 0, 'error': <reason>}``.
    """
    # Local import to avoid a circular dependency at module load time
    # (apex.engine.backtest imports from many submodules).
    from apex.engine.backtest import full_backtest

    if df is None:
        return {"n_folds": 0, "error": "df is None"}
    n = len(df)
    if n < 100:
        return {"n_folds": 0, "error": "insufficient bars"}

    oos_sharpes = []
    oos_returns = []
    oos_trades = []
    oos_pfs = []

    # Inject symbol context so the VRP short-whitelist (and any other
    # symbol-aware gates) behaves identically to the production backtest.
    eval_params = dict(best_params)
    eval_params["symbol"] = symbol

    has_dt = "datetime" in df.columns
    daily_has_dt = daily_df is not None and "datetime" in getattr(daily_df, "columns", [])

    for train_idx, test_idx in cpcv_split(n, n_blocks=n_blocks,
                                          n_test_blocks=n_test_blocks,
                                          purge_bars=purge_bars):
        if len(test_idx) < 50:
            continue

        df_test = df.iloc[test_idx].reset_index(drop=True)

        # Subset daily_df to dates spanning the test window when possible.
        if daily_df is not None and has_dt and daily_has_dt and len(df_test) > 0:
            test_start = df_test["datetime"].iloc[0]
            test_end = df_test["datetime"].iloc[-1]
            daily_test = daily_df[(daily_df["datetime"] >= test_start) &
                                  (daily_df["datetime"] <= test_end)].reset_index(drop=True)
        else:
            daily_test = daily_df

        try:
            _, stats = full_backtest(df_test, daily_test, architecture, eval_params)
            oos_sharpes.append(float(stats.get("sharpe", 0.0)))
            oos_returns.append(float(stats.get("total_return_pct", 0.0)))
            oos_trades.append(int(stats.get("trades", 0)))
            oos_pfs.append(float(stats.get("pf", 0.0)))
        except Exception:
            continue

    if not oos_sharpes:
        return {"n_folds": 0, "error": "no successful folds"}

    arr = np.array(oos_sharpes, dtype=float)
    return {
        "n_folds": len(oos_sharpes),
        "oos_sharpes": oos_sharpes,
        "oos_returns": oos_returns,
        "oos_trades": oos_trades,
        "sharpe_median": float(np.median(arr)),
        "sharpe_iqr": (float(np.percentile(arr, 25)), float(np.percentile(arr, 75))),
        "sharpe_pct_positive": float((arr > 0).mean()),
        "mean_oos_pf": float(np.mean(oos_pfs)),
        "mean_oos_return": float(np.mean(oos_returns)),
    }
