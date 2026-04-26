"""VWAP with volume-weighted sigma bands."""

import numpy as np
import pandas as pd

from apex.indicators.basics import compute_atr


def compute_vwap_bands(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    slope_window: int = 5,
) -> pd.DataFrame:
    """Return df with columns: vwap, vwap_1s_upper, vwap_1s_lower,
    vwap_2s_upper, vwap_2s_lower, vwap_3s_upper, vwap_3s_lower,
    vwap_slope, vwap_slope_atr.

    Session reset: VWAP cumulative sums reset at each calendar-day boundary.
    typical_price = (high + low + close) / 3
    vwap = cumsum(tp * volume) / cumsum(volume)  per session
    variance = cumsum(volume * (tp - vwap)^2) / cumsum(volume)
    sigma = sqrt(variance)
    bands = vwap +/- N*sigma for N in {1, 2, 3}

    Slope columns:
      vwap_slope = vwap.diff(slope_window)
      vwap_slope_atr = vwap_slope / atr  (ATR computed inline if not present)
    """
    # Determine date boundaries
    ts = pd.to_datetime(df[timestamp_col])
    dates = ts.dt.date
    new_day = (dates != dates.shift(1)).values

    tp = ((df["high"] + df["low"] + df["close"]) / 3.0).values
    vol = df["volume"].values.astype(np.float64)

    n = len(df)
    vwap = np.empty(n, dtype=np.float64)
    sigma = np.empty(n, dtype=np.float64)

    cum_tpv = 0.0
    cum_vol = 0.0
    cum_var = 0.0

    for i in range(n):
        if new_day[i]:
            cum_tpv = tp[i] * vol[i]
            cum_vol = vol[i]
            vwap[i] = tp[i] if cum_vol > 0 else 0.0
            cum_var = 0.0
            sigma[i] = 0.0
        else:
            cum_tpv += tp[i] * vol[i]
            cum_vol += vol[i]
            if cum_vol > 0:
                vwap[i] = cum_tpv / cum_vol
            else:
                vwap[i] = tp[i]
            cum_var += vol[i] * (tp[i] - vwap[i]) ** 2
            if cum_vol > 0:
                sigma[i] = np.sqrt(cum_var / cum_vol)
            else:
                sigma[i] = 0.0

    result = df.copy()
    result["vwap"] = vwap
    for n_sigma in (1, 2, 3):
        result[f"vwap_{n_sigma}s_upper"] = vwap + n_sigma * sigma
        result[f"vwap_{n_sigma}s_lower"] = vwap - n_sigma * sigma

    # VWAP slope (absolute) and slope-to-ATR ratio (dimensionless)
    vwap_series = pd.Series(vwap, index=df.index)
    result["vwap_slope"] = vwap_series.diff(slope_window)

    if "atr" in df.columns:
        atr = df["atr"]
    else:
        atr = compute_atr(df, period=14)
    atr_safe = atr.replace(0, np.nan)
    result["vwap_slope_atr"] = result["vwap_slope"] / atr_safe

    return result
