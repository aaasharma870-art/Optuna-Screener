"""MOVE + OVX fetchers via FRED for cross-asset vol regime."""
from pathlib import Path
from typing import Optional

import pandas as pd

from apex.data.fred_client import fetch_fred_series


def fetch_move_index(start: str, end: str, cache_dir: Optional[Path]) -> pd.DataFrame:
    """ICE BofA MOVE Index proxy via FRED. Series 'BAMLH0A0HYM2EY' is used as
    a high-yield credit spread proxy; for true MOVE substitute with vendor data
    if available. Returns DataFrame indexed by date with 'value' column."""
    return fetch_fred_series("BAMLH0A0HYM2EY", start, end, cache_dir=cache_dir)


def fetch_ovx(start: str, end: str, cache_dir: Optional[Path]) -> pd.DataFrame:
    """CBOE Crude Oil Volatility Index via FRED 'OVXCLS'."""
    return fetch_fred_series("OVXCLS", start, end, cache_dir=cache_dir)


def compute_vol_percentiles(vix: pd.Series, move: pd.Series,
                             ovx: pd.Series, window: int = 252) -> pd.DataFrame:
    """Return DataFrame with vix_pct, move_pct, ovx_pct columns (rolling percentile)."""
    def _pct(series, w):
        return series.rolling(w, min_periods=w // 2).rank(pct=True) * 100

    df = pd.DataFrame({
        "vix": vix, "move": move, "ovx": ovx,
        "vix_pct": _pct(vix, window),
        "move_pct": _pct(move, window),
        "ovx_pct": _pct(ovx, window),
    })
    return df
