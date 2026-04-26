"""Merge dealer-level columns (gamma walls) onto intraday exec df."""
from pathlib import Path
from typing import Optional

import pandas as pd

from apex.data.options_gex import compute_gex_proxy


def ingest_flux_points(exec_df: pd.DataFrame, symbol: str,
                       cache_dir: Optional[Path]) -> pd.DataFrame:
    """Add call_wall, put_wall, gamma_flip, vol_trigger, abs_gamma_strike columns
    to exec_df by computing per-day GEX proxy from cached options chains.

    Uses previous-day GEX (shift-1) to prevent look-ahead.
    """
    df = exec_df.copy()
    if "datetime" not in df.columns:
        return df

    # Group by date - fetch one GEX proxy per day
    df["_date"] = pd.to_datetime(df["datetime"]).dt.normalize()
    unique_dates = sorted(df["_date"].unique())

    nan_levels = {
        "call_wall": float("nan"), "put_wall": float("nan"),
        "gamma_flip": float("nan"), "vol_trigger": float("nan"),
        "abs_gamma_strike": float("nan"),
    }

    daily_levels = {}
    for d in unique_dates:
        prev_day = (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            levels = compute_gex_proxy(symbol, prev_day, cache_dir)
            daily_levels[d] = levels if isinstance(levels, dict) else dict(nan_levels)
        except Exception:
            daily_levels[d] = dict(nan_levels)

    # Forward-fill onto bars
    df["call_wall"] = df["_date"].map(lambda d: daily_levels[d]["call_wall"])
    df["put_wall"] = df["_date"].map(lambda d: daily_levels[d]["put_wall"])
    df["gamma_flip"] = df["_date"].map(lambda d: daily_levels[d]["gamma_flip"])
    df["vol_trigger"] = df["_date"].map(lambda d: daily_levels[d]["vol_trigger"])
    df["abs_gamma_strike"] = df["_date"].map(lambda d: daily_levels[d]["abs_gamma_strike"])

    return df.drop(columns=["_date"])
