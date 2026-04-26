"""Setup detectors for VRP-regime entry triggers.

These functions emit boolean entry-trigger columns. They are pure utility
functions used by VRP-mode wiring; the legacy long-only engine path does
not call them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apex.indicators.fvg import unfilled_fvgs_at


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


def detect_sweep_proxy(
    df: pd.DataFrame,
    fvgs: list,
    atr: pd.Series,
    breach_atr_mult: float = 1.5,
) -> pd.DataFrame:
    """Amplified-regime entry trigger: liquidity sweep proxy via FVG breach + reclaim.

    Adds two boolean columns to a copy of df:

      - sweep_proxy_long:  true on bar i if price briefly exceeded the LOWER
        edge of the nearest un-filled BULLISH FVG by breach_atr_mult * atr,
        then reclaimed (closed back above that edge).

      - sweep_proxy_short: mirror -- price exceeded UPPER edge of nearest
        un-filled BEARISH FVG by breach_atr_mult * atr, then reclaimed
        (closed back below that edge).

    For a bullish FVG the relevant edge is fvg["low"] (the lower boundary).
    For a bearish FVG the relevant edge is fvg["high"] (the upper boundary).
    """
    n = len(df)
    result = df.copy()
    long_trigger = np.zeros(n, dtype=bool)
    short_trigger = np.zeros(n, dtype=bool)

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    atr_vals = atr.values if isinstance(atr, pd.Series) else np.asarray(atr)

    for i in range(n):
        a = atr_vals[i]
        if not np.isfinite(a) or a <= 0:
            continue
        breach = breach_atr_mult * a

        # Find un-filled FVGs available at bar i
        unfilled = unfilled_fvgs_at(fvgs, i)
        if not unfilled:
            continue

        # Nearest bullish FVG: smallest distance from current close to fvg["low"]
        bullish = [f for f in unfilled if f["direction"] == "bullish"]
        if bullish:
            nearest_b = min(bullish, key=lambda f: abs(close[i] - f["low"]))
            edge = nearest_b["low"]
            # Sweep: bar dipped breach below edge, then closed back above
            if low[i] < (edge - breach) and close[i] > edge:
                long_trigger[i] = True

        # Nearest bearish FVG: smallest distance from close to fvg["high"]
        bearish = [f for f in unfilled if f["direction"] == "bearish"]
        if bearish:
            nearest_s = min(bearish, key=lambda f: abs(close[i] - f["high"]))
            edge = nearest_s["high"]
            # Sweep: bar spiked breach above edge, then closed back below
            if high[i] > (edge + breach) and close[i] < edge:
                short_trigger[i] = True

    result["sweep_proxy_long"] = long_trigger
    result["sweep_proxy_short"] = short_trigger
    return result
