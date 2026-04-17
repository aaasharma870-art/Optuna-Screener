"""Fair Value Gap (3-bar imbalance) detector."""

from __future__ import annotations

import pandas as pd


def detect_fvgs(df: pd.DataFrame) -> list[dict]:
    """3-bar imbalance detector.

    Bullish FVG: high[i] < low[i+2] -- gap between bar[i].high and bar[i+2].low
    Bearish FVG: low[i] > high[i+2]

    Fill: bullish filled when close returns to high[i] or below;
          bearish when close to low[i] or above.

    Returns list of dicts: {start_idx, end_idx, direction, low, high, filled_at_idx}
    NEVER used as entry signals -- only for trailing-stop anchors.
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    fvgs: list[dict] = []

    for i in range(n - 2):
        i2 = i + 2

        # Bullish FVG: gap above bar[i].high up to bar[i+2].low
        if highs[i] < lows[i2]:
            fvg = {
                "start_idx": i,
                "end_idx": i2,
                "direction": "bullish",
                "low": float(highs[i]),
                "high": float(lows[i2]),
                "filled_at_idx": None,
            }
            # Check fill going forward
            for j in range(i2 + 1, n):
                if closes[j] <= highs[i]:
                    fvg["filled_at_idx"] = j
                    break
            fvgs.append(fvg)

        # Bearish FVG: gap below bar[i].low down to bar[i+2].high
        if lows[i] > highs[i2]:
            fvg = {
                "start_idx": i,
                "end_idx": i2,
                "direction": "bearish",
                "low": float(highs[i2]),
                "high": float(lows[i]),
                "filled_at_idx": None,
            }
            # Check fill going forward
            for j in range(i2 + 1, n):
                if closes[j] >= lows[i]:
                    fvg["filled_at_idx"] = j
                    break
            fvgs.append(fvg)

    return fvgs


def unfilled_fvgs_at(fvgs: list[dict], idx: int) -> list[dict]:
    """Return FVGs present at time idx and not yet filled."""
    result = []
    for fvg in fvgs:
        # FVG must have been formed (end_idx <= idx)
        if fvg["end_idx"] > idx:
            continue
        # Not yet filled, or filled after idx
        if fvg["filled_at_idx"] is None or fvg["filled_at_idx"] > idx:
            result.append(fvg)
    return result
