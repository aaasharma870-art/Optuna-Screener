"""Polygon.io REST client with caching and retry logic."""

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from apex.config import (
    CFG, POLYGON_KEY, POLYGON_BASE, POLYGON_SLEEP,
    MAX_RETRIES, RETRY_WAIT, CACHE_DIR,
)
from apex.logging_util import log


def polygon_request(endpoint, params=None, retries=None):
    """
    GET a Polygon REST endpoint with automatic retry on 429 / 5xx.

    *endpoint* may be a relative path (``v2/aggs/...``) which is appended to
    POLYGON_BASE, or a full URL (used for ``next_url`` pagination).
    """
    if retries is None:
        retries = MAX_RETRIES
    if params is None:
        params = {}
    params = dict(params)
    params["apiKey"] = POLYGON_KEY

    if endpoint.startswith("http"):
        url = endpoint
    else:
        url = f"{POLYGON_BASE}/{endpoint.lstrip('/')}"

    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                wait = RETRY_WAIT * (attempt + 1)
                log(f"Polygon 429, waiting {wait}s...", "WARN")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                wait = RETRY_WAIT * (attempt + 1)
                log(f"Polygon {r.status_code}, waiting {wait}s...", "WARN")
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            time.sleep(POLYGON_SLEEP)
            return r.json()
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                log(f"Polygon request error: {e}. Retry {attempt + 1}/{retries}", "WARN")
                time.sleep(RETRY_WAIT)
            else:
                log(f"Polygon request failed after {retries} attempts: {e}", "ERROR")
                return None
    return None


def fetch_daily(symbol):
    """
    Fetch daily OHLCV for *symbol* from Polygon and cache to CSV.

    Returns (symbol, DataFrame | None, status_str).
    """
    cache_file = CACHE_DIR / f"{symbol}_daily.csv"
    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=["datetime"])
            if len(df) >= 50:
                return symbol, df, "CACHED"
        except Exception:
            pass

    bars_needed = CFG.get("universe", {}).get("min_daily_bars", 252)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=int(bars_needed * 1.6))).strftime("%Y-%m-%d")

    data = polygon_request(
        f"v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}",
        {"adjusted": "true", "sort": "asc", "limit": 50000},
    )
    if data is None or data.get("status") == "ERROR" or "results" not in data:
        return symbol, None, "NO_DATA"

    rows = data["results"]
    if not rows:
        return symbol, None, "NO_DATA"

    df = pd.DataFrame([
        {
            "datetime": pd.to_datetime(r["t"], unit="ms"),
            "open": r["o"],
            "high": r["h"],
            "low": r["l"],
            "close": r["c"],
            "volume": r.get("v", 0),
        }
        for r in rows
    ])
    df = df.sort_values("datetime").reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    return symbol, df, "FETCHED"


def fetch_bars(symbol, timeframe="1H", start_date=None, end_date=None):
    """
    Fetch intraday OHLCV bars for *symbol* at *timeframe* resolution.

    Supported timeframes: ``"1H"``, ``"30min"``, ``"15min"``, ``"5min"``.
    Returns (symbol, DataFrame | None, status_str).
    """
    tf_map = {
        "1H": ("1", "hour"),
        "30min": ("30", "minute"),
        "15min": ("15", "minute"),
        "5min": ("5", "minute"),
    }
    mult, span = tf_map.get(timeframe, ("1", "hour"))
    safe_tf = timeframe.replace("/", "").replace(":", "")
    cache_file = CACHE_DIR / f"{symbol}_{safe_tf}.csv"

    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=["datetime"])
            if len(df) >= 100:
                return symbol, df, "CACHED"
        except Exception:
            pass

    p3 = CFG.get("phase3_params", {})
    if start_date is None:
        start_date = p3.get("start_date", (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"))
    if end_date is None:
        end_date = p3.get("end_date", datetime.now().strftime("%Y-%m-%d"))

    data = polygon_request(
        f"v2/aggs/ticker/{symbol}/range/{mult}/{span}/{start_date}/{end_date}",
        {"adjusted": "true", "sort": "asc", "limit": 50000},
    )
    if data is None or "results" not in data or not data["results"]:
        return symbol, None, "NO_DATA"

    rows = data["results"]
    df = pd.DataFrame([
        {
            "datetime": pd.to_datetime(r["t"], unit="ms"),
            "open": r["o"],
            "high": r["h"],
            "low": r["l"],
            "close": r["c"],
            "volume": r.get("v", 0),
        }
        for r in rows
    ])
    # Keep regular-hours bars only (9:30-16:00 ET approximated as hour 9-16)
    df["hour"] = df["datetime"].dt.hour
    df = df[(df["hour"] >= 9) & (df["hour"] <= 16)].drop(columns=["hour"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    return symbol, df, "FETCHED"
