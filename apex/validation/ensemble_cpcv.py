"""Portfolio-level CPCV evaluation for ensemble NAV."""
from typing import Any, Dict

import numpy as np
import pandas as pd

from apex.validation.cpcv import cpcv_split


def _annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe assuming returns are per-bar."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    mu = r.mean()
    sigma = r.std(ddof=1)
    if sigma <= 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(periods_per_year))


def evaluate_ensemble_cpcv(portfolio_returns: pd.Series,
                            n_blocks: int = 8,
                            n_test_blocks: int = 2,
                            purge_bars: int = 10,
                            periods_per_year: int = 252) -> Dict[str, Any]:
    """Run CPCV at the portfolio NAV level.

    Args:
        portfolio_returns: Series of per-bar portfolio returns
        n_blocks, n_test_blocks, purge_bars: passed to cpcv_split

    Returns:
        {n_folds, oos_sharpes, sharpe_median, sharpe_iqr, sharpe_pct_positive,
         oos_returns}
    """
    n = len(portfolio_returns)
    if n < 100:
        return {"n_folds": 0, "error": "insufficient bars"}

    sharpes = []
    cum_returns = []
    for train_idx, test_idx in cpcv_split(n, n_blocks=n_blocks,
                                           n_test_blocks=n_test_blocks,
                                           purge_bars=purge_bars):
        if len(test_idx) < 30:
            continue
        test_returns = portfolio_returns.iloc[test_idx]
        s = _annualized_sharpe(test_returns, periods_per_year)
        sharpes.append(s)
        cum_returns.append(float((1 + test_returns).prod() - 1))

    if not sharpes:
        return {"n_folds": 0, "error": "no successful folds"}

    arr = np.array(sharpes)
    return {
        "n_folds": len(sharpes),
        "oos_sharpes": sharpes,
        "oos_returns": cum_returns,
        "sharpe_median": float(np.median(arr)),
        "sharpe_iqr": (float(np.percentile(arr, 25)),
                       float(np.percentile(arr, 75))),
        "sharpe_pct_positive": float((arr > 0).mean()),
    }
