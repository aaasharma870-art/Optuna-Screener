"""OPEX calendar + pin-strike helpers for Strategy 2."""
from typing import Optional

import pandas as pd


def is_opex_week(date) -> bool:
    """Return True if `date` is in the trading week containing the third Friday."""
    ts = pd.Timestamp(date)
    first_day = ts.replace(day=1)
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + pd.Timedelta(days=days_until_friday)
    third_friday = first_friday + pd.Timedelta(days=14)
    week_start = third_friday - pd.Timedelta(days=third_friday.weekday())
    week_end = week_start + pd.Timedelta(days=4)
    return week_start.normalize() <= ts.normalize() <= week_end.normalize()


def find_pin_strike(chain: dict, spot: float,
                     window_pct: float = 0.05) -> Optional[float]:
    """Return strike with highest combined call+put OI within +/- window_pct of spot."""
    if "strikes" not in chain:
        return None
    lo = spot * (1 - window_pct)
    hi = spot * (1 + window_pct)
    eligible = [s for s in chain["strikes"] if lo <= s["strike"] <= hi]
    if not eligible:
        return None
    best = max(eligible, key=lambda s: s.get("call_oi", 0) + s.get("put_oi", 0))
    return float(best["strike"])
