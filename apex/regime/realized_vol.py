"""20-day annualized realized volatility from close prices."""

import numpy as np
import pandas as pd


def compute_realized_vol_20d(close: pd.Series, window: int = 20) -> pd.Series:
    """Compute annualized realized volatility from log returns.

    log_returns = log(close / close.shift(1))
    rv = log_returns.rolling(window).std(ddof=1) * sqrt(252)

    Returns annualized vol as a decimal (e.g., 0.15 = 15%).
    First *window* values are NaN.
    """
    log_returns = np.log(close / close.shift(1))
    rv = log_returns.rolling(window).std(ddof=1) * np.sqrt(252)
    return rv
