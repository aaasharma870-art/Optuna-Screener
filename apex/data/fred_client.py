"""FRED REST API client with caching."""

import os
import time

import pandas as pd
import requests

from apex.config import CFG, CACHE_DIR
from apex.logging_util import log

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Series IDs supported
FRED_SERIES = {
    "VIXCLS": "VIX (SPY implied vol)",
    "VXVCLS": "VXV (3-month VIX, term structure)",
    "VXNCLS": "VXN (QQQ implied vol)",
    "GVZCLS": "GVZ (GLD implied vol)",
}

# Map ticker -> implied vol FRED series
IV_MAP = {
    "SPY": "VIXCLS",
    "QQQ": "VXNCLS",
    "GLD": "GVZCLS",
}


def _get_fred_api_key() -> str:
    """Resolve FRED API key from env or config."""
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    key = CFG.get("fred_api_key")
    if key:
        return key
    raise RuntimeError(
        "FRED API key not found. Set FRED_API_KEY env var or "
        "fred_api_key in apex_config.json."
    )


def fetch_fred_series(series_id: str, start_date: str, end_date: str,
                      cache_dir=None) -> pd.DataFrame:
    """Fetch a FRED series and return DataFrame with DatetimeIndex and 'value' column.

    Cached to ``{cache_dir}/fred_{series_id}.parquet``.
    FRED rate limit: 120 req/min. Sleeps 1s between calls.

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g. "VIXCLS").
    start_date : str
        Start date as "YYYY-MM-DD".
    end_date : str
        End date as "YYYY-MM-DD".
    cache_dir : Path or None
        Override cache directory; defaults to CACHE_DIR.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, single column 'value' (float). Missing values dropped.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR

    cache_file = cache_dir / f"fred_{series_id}.parquet"
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            if len(df) > 0:
                log(f"FRED {series_id}: cached ({len(df)} rows)")
                return df
        except Exception:
            pass

    api_key = _get_fred_api_key()

    params = {
        "series_id": series_id,
        "observation_start": start_date,
        "observation_end": end_date,
        "api_key": api_key,
        "file_type": "json",
    }

    log(f"FRED {series_id}: fetching {start_date} to {end_date}")
    try:
        r = requests.get(FRED_BASE, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException as e:
        log(f"FRED {series_id}: request failed: {e}", "ERROR")
        return pd.DataFrame(columns=["value"])

    observations = data.get("observations", [])
    if not observations:
        log(f"FRED {series_id}: no observations returned", "WARN")
        return pd.DataFrame(columns=["value"])

    rows = []
    for obs in observations:
        date_str = obs.get("date", "")
        val_str = obs.get("value", ".")
        if val_str == "." or val_str == "":
            continue  # missing value
        try:
            rows.append({"date": pd.to_datetime(date_str), "value": float(val_str)})
        except (ValueError, TypeError):
            continue

    if not rows:
        log(f"FRED {series_id}: all values missing", "WARN")
        return pd.DataFrame(columns=["value"])

    df = pd.DataFrame(rows).set_index("date")
    df.index.name = None
    df = df.sort_index()

    # Cache
    try:
        df.to_parquet(cache_file)
    except Exception as e:
        log(f"FRED {series_id}: cache write failed: {e}", "WARN")

    # Rate-limit courtesy sleep
    time.sleep(1)

    log(f"FRED {series_id}: fetched {len(df)} observations")
    return df
