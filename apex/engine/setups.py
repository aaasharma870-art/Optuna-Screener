"""Setup detectors for VRP-regime entry triggers.

These functions emit boolean entry-trigger columns. They are pure utility
functions used by VRP-mode wiring; the legacy long-only engine path does
not call them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_breakout_reversal(
    df: pd.DataFrame,
    atr: pd.Series,
    lookback: int = 5,
    breakout_atr_mult: float = 1.5,
) -> pd.DataFrame:
    """Suppressed-regime entry trigger: failed breakout reversal.

    Adds two boolean columns to a copy of df:

      - breakout_reversal_long:  on bar i, true if
            max(close[i-lookback..i-1]) - vwap[i-1] > breakout_atr_mult * atr[i-1]
            AND close[i] < max(close[i-lookback..i-1])
        (price stretched far above VWAP then closed back inside the range)

      - breakout_reversal_short: mirror -- min low far below VWAP, then close
        back up above the prior min.

    Requires df["vwap"] to be present (use compute_vwap_bands first).
    """
    if "vwap" not in df.columns:
        raise ValueError("detect_breakout_reversal requires df['vwap']; run compute_vwap_bands first.")

    n = len(df)
    result = df.copy()
    long_trigger = np.zeros(n, dtype=bool)
    short_trigger = np.zeros(n, dtype=bool)

    close = df["close"].values
    vwap = df["vwap"].values
    atr_vals = atr.values if isinstance(atr, pd.Series) else np.asarray(atr)

    for i in range(lookback, n):
        prev_atr = atr_vals[i - 1]
        prev_vwap = vwap[i - 1]
        if not np.isfinite(prev_atr) or prev_atr <= 0 or not np.isfinite(prev_vwap):
            continue

        window_high = close[i - lookback : i].max()
        window_low = close[i - lookback : i].min()

        # Long trigger: price was stretched ABOVE vwap, now reverting
        if (window_high - prev_vwap) > (breakout_atr_mult * prev_atr) and close[i] < window_high:
            long_trigger[i] = True

        # Short trigger: price was stretched BELOW vwap, now reverting
        if (prev_vwap - window_low) > (breakout_atr_mult * prev_atr) and close[i] > window_low:
            short_trigger[i] = True

    result["breakout_reversal_long"] = long_trigger
    result["breakout_reversal_short"] = short_trigger
    return result
