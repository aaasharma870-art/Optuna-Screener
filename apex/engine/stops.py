"""Dynamic trailing-stop selection anchored to Fair Value Gaps."""

from __future__ import annotations

from apex.indicators.fvg import unfilled_fvgs_at


def compute_dynamic_stop(
    position_side: str,
    price: float,
    fvgs: list,
    current_idx: int,
    atr: float,
    atr_fallback_mult: float = 2.0,
    fvg_buffer_atr_mult: float = 0.05,
) -> float:
    """Return a stop-loss price.

    For long: nearest un-filled bullish FVG BELOW price; stop at its lower edge
    minus a small buffer (fvg_buffer_atr_mult * atr). Falls back to
    price - atr_fallback_mult * atr when no qualifying FVG exists.

    For short: mirror -- nearest un-filled bearish FVG ABOVE price; stop at its
    upper edge plus buffer. Falls back to price + atr_fallback_mult * atr.
    """
    unfilled = unfilled_fvgs_at(fvgs, current_idx)
    buffer = fvg_buffer_atr_mult * atr

    if position_side == "long":
        # Find nearest bullish FVG with upper edge (high) below price
        best = None
        for fvg in unfilled:
            if fvg["direction"] != "bullish":
                continue
            if fvg["high"] >= price:
                continue
            if best is None or fvg["low"] > best["low"]:
                best = fvg
        if best is not None:
            return best["low"] - buffer
        return price - atr_fallback_mult * atr

    else:  # short
        # Find nearest bearish FVG with lower edge (low) above price
        best = None
        for fvg in unfilled:
            if fvg["direction"] != "bearish":
                continue
            if fvg["low"] <= price:
                continue
            if best is None or fvg["high"] < best["high"]:
                best = fvg
        if best is not None:
            return best["high"] + buffer
        return price + atr_fallback_mult * atr
