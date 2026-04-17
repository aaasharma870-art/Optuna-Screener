"""Volume-Weighted Close Location Value."""

import numpy as np
import pandas as pd


def compute_vwclv(
    df: pd.DataFrame,
    cumulative_window: int = 5,
    vol_ma_window: int = 20,
) -> pd.DataFrame:
    """
    CLV = (close - low) / (high - low)  -- [0,1], guard H==L -> 0.5
    weight = volume / volume.rolling(vol_ma_window).mean()
    vwclv = (2*CLV - 1) * weight  -- shifted to [-w, +w]
    cum_vwclv = vwclv.rolling(cumulative_window).sum()

    Returns DataFrame with columns: clv, vwclv, cum_vwclv
    """
    result = df.copy()

    hl_range = df["high"] - df["low"]
    # Guard zero-range bars
    clv = np.where(
        hl_range > 0,
        (df["close"] - df["low"]) / hl_range,
        0.5,
    )
    clv = pd.Series(clv, index=df.index)

    vol_ma = df["volume"].rolling(vol_ma_window, min_periods=1).mean()
    weight = df["volume"] / vol_ma.replace(0, np.nan)

    vwclv = (2.0 * clv - 1.0) * weight
    cum_vwclv = vwclv.rolling(cumulative_window, min_periods=1).sum()

    result["clv"] = clv.values
    result["vwclv"] = vwclv.values
    result["cum_vwclv"] = cum_vwclv.values

    return result
