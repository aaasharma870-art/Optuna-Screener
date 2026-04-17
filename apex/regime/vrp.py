"""Volatility Risk Premium (VRP) calculation and percentile ranking."""

import numpy as np
import pandas as pd

from apex.regime.realized_vol import compute_realized_vol_20d


def compute_vrp_percentile(vrp: pd.Series, window: int = 252) -> pd.Series:
    """Rolling percentile rank (0-100).

    Excludes current bar from its own rank: uses bars [i-window, i-1]
    to rank bar i.
    """
    result = pd.Series(np.nan, index=vrp.index)
    for i in range(window + 1, len(vrp)):
        lookback = vrp.iloc[i - window:i]  # excludes bar i
        valid = lookback.dropna()
        if len(valid) == 0:
            continue
        current = vrp.iloc[i]
        if pd.isna(current):
            continue
        rank = (valid < current).sum()
        result.iloc[i] = (rank / len(valid)) * 100.0
    return result


def compute_vrp(iv_series: pd.Series, close_series: pd.Series,
                rv_window: int = 20, pct_window: int = 252) -> pd.DataFrame:
    """Compute VRP and its percentile rank.

    Parameters
    ----------
    iv_series : pd.Series
        Implied volatility in annualized % terms (e.g., VIX=18 means 18%).
    close_series : pd.Series
        Price series for realized vol computation.
    rv_window : int
        Window for realized vol (default 20).
    pct_window : int
        Window for percentile ranking (default 252).

    Returns
    -------
    pd.DataFrame
        Columns: iv, rv, vrp_raw, vrp_pct
    """
    rv_decimal = compute_realized_vol_20d(close_series, window=rv_window)
    rv_pct = rv_decimal * 100.0  # convert to % terms to match IV

    vrp_raw = iv_series - rv_pct
    vrp_pct = compute_vrp_percentile(vrp_raw, window=pct_window)

    return pd.DataFrame({
        "iv": iv_series,
        "rv": rv_pct,
        "vrp_raw": vrp_raw,
        "vrp_pct": vrp_pct,
    }, index=close_series.index)
