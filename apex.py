"""
Optuna Screener Pipeline
========================
A generic, Optuna-driven research pipeline for systematic trading strategies.

The pipeline discovers, tunes, and validates indicator-based trading strategies
with a layered defense against overfitting:

  * Layer 1  - Architecture search (which indicators, which exits, which regime)
  * Layer 2  - Per-symbol deep parameter tuning with walk-forward IS/OOS split
  * Layer 3  - Robustness gauntlet (Monte Carlo, noise, regime stress, sensitivity)
  * Correlation filter + sector caps
  * Final backtest on the FULL tune universe AND a truly held-out 25% window
  * Self-contained HTML report + trade CSV + parameters JSON + AmiBroker AFL

All market data is pulled from the Polygon.io REST API.
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import json
import os
import sys
import time
import math
import warnings
import argparse
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

def load_config(path="apex_config.json"):
    """Load pipeline configuration from JSON file."""
    script_dir = Path(__file__).resolve().parent
    full_path = script_dir / path
    if not full_path.exists():
        full_path = Path(path)
    if not full_path.exists():
        print(f"[ERROR] Config file not found: {path}")
        sys.exit(1)
    with open(full_path, "r") as f:
        cfg = json.load(f)
    # Environment variable overrides
    env_polygon = os.environ.get("POLYGON_API_KEY")
    if env_polygon:
        cfg["polygon_api_key"] = env_polygon
    env_fred = os.environ.get("FRED_API_KEY")
    if env_fred:
        cfg["fred_api_key"] = env_fred
    return cfg


CFG = load_config()
POLYGON_KEY = CFG["polygon_api_key"]
CACHE_DIR = Path(CFG.get("cache_dir", "apex_cache"))
OUTPUT_DIR = Path(CFG.get("output_dir", "apex_results"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RL = CFG.get("polygon_rate_limit", {})
POLYGON_SLEEP = RL.get("sleep_between_calls", 0.12)
MAX_RETRIES = RL.get("max_retries", 3)
RETRY_WAIT = RL.get("retry_wait", 10)

POLYGON_BASE = "https://api.polygon.io"


# ============================================================
# 3. UTILITY FUNCTIONS
# ============================================================

def log(msg, level="INFO"):
    """Print a timestamped log line."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def eta_str(remaining, rate_per_sec):
    """Return a human-readable ETA string given items remaining and rate."""
    if rate_per_sec <= 0:
        return "???"
    secs = remaining / rate_per_sec
    if secs < 60:
        return f"{secs:.0f}s"
    elif secs < 3600:
        return f"{secs / 60:.1f}min"
    else:
        return f"{secs / 3600:.1f}hr"


# ============================================================
# 4. POLYGON REST CLIENT
# ============================================================

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


# ============================================================
# 5. TECHNICAL INDICATOR LIBRARY
# ============================================================

def compute_ema(series, span):
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_atr(df, period=14):
    """Average True Range over *period* bars."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def compute_vwap(df):
    """Compute VWAP with intraday reset on each new calendar day."""
    dates = df["datetime"].dt.date
    new_day = dates != dates.shift(1)
    typ_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_tpv = 0.0
    cum_vol = 0.0
    vwap = np.empty(len(df), dtype=np.float64)
    for i in range(len(df)):
        if new_day.iloc[i]:
            cum_tpv = typ_price.iloc[i] * df["volume"].iloc[i]
            cum_vol = float(df["volume"].iloc[i])
        else:
            cum_tpv += typ_price.iloc[i] * df["volume"].iloc[i]
            cum_vol += float(df["volume"].iloc[i])
        vwap[i] = cum_tpv / cum_vol if cum_vol > 0 else df["close"].iloc[i]
    return pd.Series(vwap, index=df.index, name="vwap")


def compute_rsi(series, period=14):
    """Wilder-smoothed RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(close, fast=12, slow=26, signal=9):
    """
    MACD with configurable periods.

    Returns (macd_line, signal_line, histogram) as three pd.Series.
    """
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands. Returns (upper, mid, lower)."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def compute_stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator. Returns (%K, %D)."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    denom = highest_high - lowest_low
    denom = denom.replace(0, np.nan)
    k_raw = 100.0 * (close - lowest_low) / denom
    k_smooth = k_raw.rolling(d_period).mean()
    d_smooth = k_smooth.rolling(d_period).mean()
    return k_smooth, d_smooth


def compute_obv(close, volume, ma_period=20):
    """On-Balance Volume with moving-average signal."""
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    obv_ma = obv.rolling(ma_period).mean()
    return obv, obv_ma


def compute_adx(high, low, close, period=14):
    """Average Directional Index (ADX)."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = high - prev_high
    minus_dm = prev_low - low
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr_smooth = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    plus_di = 100.0 * plus_dm_smooth / atr_smooth.replace(0, np.nan)
    minus_di = 100.0 * minus_dm_smooth / atr_smooth.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return adx


def compute_cci(high, low, close, period=20):
    """Commodity Channel Index (CCI)."""
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(period).mean()
    mean_dev = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    mean_dev = mean_dev.replace(0, np.nan)
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    return cci


def compute_williams_r(high, low, close, period=14):
    """Williams %R oscillator (range -100 to 0)."""
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    denom = highest_high - lowest_low
    denom = denom.replace(0, np.nan)
    wr = -100.0 * (highest_high - close) / denom
    return wr


def compute_keltner(close, atr, period=20, mult=2.0):
    """Keltner Channel using EMA mid-line and ATR bands."""
    mid = compute_ema(close, period)
    upper = mid + mult * atr
    lower = mid - mult * atr
    return upper, mid, lower


def compute_volume_surge(volume, ma_period=20, mult=1.5):
    """Detect volume surges (True where volume > ma * mult)."""
    vol_ma = volume.rolling(ma_period).mean()
    return volume > (vol_ma * mult)


def parkinson_iv_proxy(df, window=20):
    """
    Parkinson volatility estimator (annualised) from daily high/low range.

    sigma^2 = (1 / (4 ln 2)) * mean( ln(H/L)^2 )
    """
    hl = np.log(df["high"] / df["low"].replace(0, np.nan))
    hl2 = hl ** 2
    factor = 1.0 / (4.0 * math.log(2.0))
    rv = (hl2.rolling(window).mean() * factor).pow(0.5)
    return rv * math.sqrt(252)


# ============================================================
# 6. CONCEPT PARSER
# ============================================================

INDICATOR_REGISTRY = {
    "RSI": {
        "compute": "compute_rsi",
        "params": {"rsi_period": (5, 30), "rsi_oversold": (20, 40), "rsi_overbought": (60, 85)},
        "signal_type": "oscillator",
    },
    "MACD": {
        "compute": "compute_macd",
        "params": {"macd_fast": (8, 16), "macd_slow": (20, 35), "macd_signal": (5, 12)},
        "signal_type": "crossover",
    },
    "Bollinger": {
        "compute": "compute_bollinger",
        "params": {"boll_period": (14, 30), "boll_std": (1.5, 3.0)},
        "signal_type": "band",
    },
    "Stochastic": {
        "compute": "compute_stochastic",
        "params": {"stoch_k": (5, 21), "stoch_d": (3, 7)},
        "signal_type": "oscillator",
    },
    "OBV": {
        "compute": "compute_obv",
        "params": {"obv_ma_period": (10, 30)},
        "signal_type": "volume",
    },
    "ADX": {
        "compute": "compute_adx",
        "params": {"adx_period": (10, 25), "adx_threshold": (20, 35)},
        "signal_type": "trend_strength",
    },
    "CCI": {
        "compute": "compute_cci",
        "params": {"cci_period": (14, 30), "cci_oversold": (-150, -80), "cci_overbought": (80, 150)},
        "signal_type": "oscillator",
    },
    "WilliamsR": {
        "compute": "compute_williams_r",
        "params": {"willr_period": (7, 21), "willr_oversold": (-90, -70), "willr_overbought": (-30, -10)},
        "signal_type": "oscillator",
    },
    "Keltner": {
        "compute": "compute_keltner",
        "params": {"keltner_period": (14, 30), "keltner_mult": (1.0, 3.0)},
        "signal_type": "band",
    },
    "VolumeSurge": {
        "compute": "compute_volume_surge",
        "params": {"volume_surge_ma": (10, 30), "volume_surge_mult": (1.2, 3.0)},
        "signal_type": "volume",
    },
    "VWAP": {
        "compute": "compute_vwap",
        "params": {},
        "signal_type": "level",
    },
    "EMA_Cross": {
        "compute": "compute_ema",
        "params": {"ema_fast": (5, 15), "ema_slow": (18, 50)},
        "signal_type": "crossover",
    },
}


def parse_concept(concept_str):
    """
    Parse a human-readable strategy concept string into indicator bias weights.

    Examples:
      ``"mean reversion with volume confirmation"``
      ``"trend following momentum breakout"``

    Returns a dict mapping indicator names to float weights (0.0 to 2.0).
    """
    concept = concept_str.lower().strip()
    weights = {name: 1.0 for name in INDICATOR_REGISTRY}

    # Mean-reversion keywords boost oscillators, suppress trend
    mean_rev_kw = ["mean reversion", "revert", "bounce", "oversold", "dip", "pullback", "range"]
    if any(kw in concept for kw in mean_rev_kw):
        weights["RSI"] = 2.0
        weights["Bollinger"] = 2.0
        weights["Stochastic"] = 1.8
        weights["CCI"] = 1.5
        weights["WilliamsR"] = 1.5
        weights["ADX"] = 0.5
        weights["MACD"] = 0.6
        weights["EMA_Cross"] = 0.5

    # Trend-following keywords boost trend indicators
    trend_kw = ["trend", "momentum", "breakout", "follow", "directional"]
    if any(kw in concept for kw in trend_kw):
        weights["MACD"] = 2.0
        weights["ADX"] = 2.0
        weights["EMA_Cross"] = 2.0
        weights["Keltner"] = 1.5
        weights["RSI"] = 0.7
        weights["Stochastic"] = 0.5
        weights["Bollinger"] = 0.8

    # Volume keywords
    vol_kw = ["volume", "surge", "liquidity", "accumulation"]
    if any(kw in concept for kw in vol_kw):
        weights["VolumeSurge"] = 2.0
        weights["OBV"] = 2.0
        weights["VWAP"] = 1.5

    # Volatility keywords
    volat_kw = ["volatility", "squeeze", "expansion", "compress"]
    if any(kw in concept for kw in volat_kw):
        weights["Bollinger"] = 2.0
        weights["Keltner"] = 2.0
        weights["ADX"] = 1.5
        weights["CCI"] = 1.3

    # Scalp / intraday keywords
    scalp_kw = ["scalp", "intraday", "quick", "fast"]
    if any(kw in concept for kw in scalp_kw):
        weights["VWAP"] = 2.0
        weights["RSI"] = 1.5
        weights["Stochastic"] = 1.5
        weights["VolumeSurge"] = 1.8
        weights["EMA_Cross"] = 1.3

    # Swing keywords
    swing_kw = ["swing", "multi-day", "position", "hold"]
    if any(kw in concept for kw in swing_kw):
        weights["MACD"] = 1.8
        weights["ADX"] = 1.8
        weights["Bollinger"] = 1.3
        weights["EMA_Cross"] = 1.5
        weights["OBV"] = 1.5

    return weights


# ============================================================
# 7. SECTOR MAP
# ============================================================

SECTOR_MAP = {
    # Broad market ETFs
    "SPY": "Index", "SPX": "Index", "QQQ": "Index", "IWM": "Index", "DIA": "Index",
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "META": "Technology", "NVDA": "Technology", "AMD": "Technology",
    "NFLX": "Technology", "CRM": "Technology", "ADBE": "Technology",
    "INTC": "Technology", "AVGO": "Technology", "MU": "Technology",
    "QCOM": "Technology", "ORCL": "Technology",
    # Semiconductors
    "SMH": "Semiconductors", "SOXX": "Semiconductors",
    "TSM": "Semiconductors", "ASML": "Semiconductors",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "C": "Financials", "WFC": "Financials",
    "XLF": "Financials", "V": "Financials", "MA": "Financials",
    # Energy
    "XLE": "Energy", "XOM": "Energy", "CVX": "Energy",
    "COP": "Energy", "SLB": "Energy",
    # Healthcare
    "XLV": "Healthcare", "JNJ": "Healthcare", "UNH": "Healthcare",
    "PFE": "Healthcare", "LLY": "Healthcare",
    # Consumer
    "AMZN": "Consumer", "TSLA": "Consumer", "WMT": "Consumer",
    "HD": "Consumer", "COST": "Consumer", "MCD": "Consumer",
    "DIS": "Consumer",
    # Industrials
    "BA": "Industrials", "CAT": "Industrials", "HON": "Industrials",
    # Materials / Metals
    "GDX": "Materials", "GLD": "Materials", "SLV": "Materials",
    # Communication
    "GOOG": "Communication", "T": "Communication", "VZ": "Communication",
}


# ============================================================
# 8. BACKTEST ENGINE
# ============================================================

def compute_indicator_signals(df, architecture, params):
    """
    Compute all indicators specified in *architecture['indicators']* and
    return a dict of per-bar signal Series.

    Each signal is an integer Series: +1 = bullish, -1 = bearish, 0 = neutral.
    """
    active = architecture.get("indicators", [])
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    signals = {}

    atr_period = params.get("atr_period", 14)
    atr = compute_atr(df, atr_period)

    if "RSI" in active:
        rsi_period = params.get("rsi_period", 14)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi = compute_rsi(close, rsi_period)
        sig = pd.Series(0, index=df.index)
        sig[rsi < rsi_oversold] = 1
        sig[rsi > rsi_overbought] = -1
        signals["RSI"] = sig

    if "MACD" in active:
        macd_fast = params.get("macd_fast", 12)
        macd_slow = params.get("macd_slow", 26)
        macd_signal_p = params.get("macd_signal", 9)
        macd_line, signal_line, histogram = compute_macd(close, macd_fast, macd_slow, macd_signal_p)
        sig = pd.Series(0, index=df.index)
        sig[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1
        sig[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1
        sig[(sig == 0) & (histogram > 0) & (histogram > histogram.shift(1))] = 1
        sig[(sig == 0) & (histogram < 0) & (histogram < histogram.shift(1))] = -1
        signals["MACD"] = sig

    if "Bollinger" in active:
        boll_period = params.get("boll_period", 20)
        boll_std = params.get("boll_std", 2.0)
        upper, mid, lower = compute_bollinger(close, boll_period, boll_std)
        sig = pd.Series(0, index=df.index)
        sig[close <= lower] = 1
        sig[close >= upper] = -1
        signals["Bollinger"] = sig

    if "Stochastic" in active:
        stoch_k = params.get("stoch_k", 14)
        stoch_d = params.get("stoch_d", 3)
        k_line, d_line = compute_stochastic(high, low, close, stoch_k, stoch_d)
        sig = pd.Series(0, index=df.index)
        sig[(k_line < 20) & (k_line > d_line)] = 1
        sig[(k_line > 80) & (k_line < d_line)] = -1
        signals["Stochastic"] = sig

    if "OBV" in active:
        obv_ma_period = params.get("obv_ma_period", 20)
        obv, obv_ma = compute_obv(close, volume, obv_ma_period)
        sig = pd.Series(0, index=df.index)
        sig[obv > obv_ma] = 1
        sig[obv < obv_ma] = -1
        signals["OBV"] = sig

    if "ADX" in active:
        adx_period = params.get("adx_period", 14)
        adx_threshold = params.get("adx_threshold", 25)
        adx = compute_adx(high, low, close, adx_period)
        ema_short = compute_ema(close, 9)
        ema_long = compute_ema(close, 21)
        sig = pd.Series(0, index=df.index)
        trending = adx > adx_threshold
        sig[(trending) & (ema_short > ema_long)] = 1
        sig[(trending) & (ema_short < ema_long)] = -1
        signals["ADX"] = sig

    if "CCI" in active:
        cci_period = params.get("cci_period", 20)
        cci_oversold = params.get("cci_oversold", -100)
        cci_overbought = params.get("cci_overbought", 100)
        cci = compute_cci(high, low, close, cci_period)
        sig = pd.Series(0, index=df.index)
        sig[cci < cci_oversold] = 1
        sig[cci > cci_overbought] = -1
        signals["CCI"] = sig

    if "WilliamsR" in active:
        willr_period = params.get("willr_period", 14)
        willr_oversold = params.get("willr_oversold", -80)
        willr_overbought = params.get("willr_overbought", -20)
        wr = compute_williams_r(high, low, close, willr_period)
        sig = pd.Series(0, index=df.index)
        sig[wr < willr_oversold] = 1
        sig[wr > willr_overbought] = -1
        signals["WilliamsR"] = sig

    if "Keltner" in active:
        keltner_period = params.get("keltner_period", 20)
        keltner_mult = params.get("keltner_mult", 2.0)
        k_upper, k_mid, k_lower = compute_keltner(close, atr, keltner_period, keltner_mult)
        sig = pd.Series(0, index=df.index)
        sig[close <= k_lower] = 1
        sig[close >= k_upper] = -1
        signals["Keltner"] = sig

    if "VolumeSurge" in active:
        vs_ma = params.get("volume_surge_ma", 20)
        vs_mult = params.get("volume_surge_mult", 1.5)
        surge = compute_volume_surge(volume, vs_ma, vs_mult)
        sig = pd.Series(0, index=df.index)
        sig[(surge) & (close > df["open"])] = 1
        sig[(surge) & (close < df["open"])] = -1
        signals["VolumeSurge"] = sig

    if "VWAP" in active:
        vwap = compute_vwap(df)
        sig = pd.Series(0, index=df.index)
        sig[close > vwap] = 1
        sig[close < vwap] = -1
        signals["VWAP"] = sig

    if "EMA_Cross" in active:
        ema_fast_p = params.get("ema_fast", 9)
        ema_slow_p = params.get("ema_slow", 21)
        ema_f = compute_ema(close, ema_fast_p)
        ema_s = compute_ema(close, ema_slow_p)
        sig = pd.Series(0, index=df.index)
        sig[(ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))] = 1
        sig[(ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))] = -1
        sig[(sig == 0) & (ema_f > ema_s)] = 1
        sig[(sig == 0) & (ema_f < ema_s)] = -1
        signals["EMA_Cross"] = sig

    signals["_atr"] = atr
    return signals


def compute_regime(df, daily_df, regime_model, params):
    """
    Compute a per-bar regime label using a simple price-vs-EMA model.

    Regime codes (R1=best, R4=worst) are a compact way for the backtester
    to accept or reject entries based on broad market conditions.

      * ``"trend"``       ADX + EMA cross
      * ``"volatility"``  ATR percentile bucket + price vs EMA20
      * ``"ema"``         simple EMA20 vs EMA50 classification (default)

    Custom regime models can be plugged in by extending this function.
    """
    n = len(df)
    regime = pd.Series("R1", index=df.index)

    if regime_model == "volatility":
        atr = compute_atr(df, params.get("atr_period", 14))
        atr_pct = (atr / df["close"]) * 100.0
        atr_med = atr_pct.rolling(100, min_periods=20).median()
        ema20 = compute_ema(df["close"], 20)
        for i in range(n):
            am = atr_med.iloc[i]
            ap = atr_pct.iloc[i]
            above_ema = df["close"].iloc[i] > ema20.iloc[i]
            if pd.isna(am) or pd.isna(ap):
                regime.iloc[i] = "R1"
            elif ap < am and above_ema:
                regime.iloc[i] = "R1"   # Low vol, bullish
            elif ap < am:
                regime.iloc[i] = "R2"   # Low vol, bearish
            elif above_ema:
                regime.iloc[i] = "R3"   # High vol, bullish
            else:
                regime.iloc[i] = "R4"   # High vol, bearish

    elif regime_model == "trend":
        adx = compute_adx(df["high"], df["low"], df["close"], params.get("adx_period", 14))
        ema_f = compute_ema(df["close"], 9)
        ema_s = compute_ema(df["close"], 21)
        for i in range(n):
            adx_val = adx.iloc[i]
            bullish = ema_f.iloc[i] > ema_s.iloc[i]
            if pd.isna(adx_val):
                regime.iloc[i] = "R1"
            elif adx_val > 25 and bullish:
                regime.iloc[i] = "R1"
            elif adx_val > 25:
                regime.iloc[i] = "R4"
            elif bullish:
                regime.iloc[i] = "R2"
            else:
                regime.iloc[i] = "R3"

    else:
        # Default simple "ema" regime: EMA20 vs EMA50 on the execution timeframe
        ema20 = compute_ema(df["close"], 20)
        ema50 = compute_ema(df["close"], 50)
        close = df["close"]
        for i in range(n):
            if pd.isna(ema50.iloc[i]):
                regime.iloc[i] = "R1"
                continue
            above20 = close.iloc[i] > ema20.iloc[i]
            above50 = close.iloc[i] > ema50.iloc[i]
            bull_stack = ema20.iloc[i] > ema50.iloc[i]
            if above20 and above50 and bull_stack:
                regime.iloc[i] = "R1"
            elif above50:
                regime.iloc[i] = "R2"
            elif bull_stack:
                regime.iloc[i] = "R3"
            else:
                regime.iloc[i] = "R4"

    return regime


def compute_entry_score(signals, regime, architecture, params):
    """
    Aggregate individual indicator signals into a composite entry score.

    Aggregation modes:
      - ``"additive"``:  sum of bullish signals (+1 each)
      - ``"weighted"``:  weighted sum using concept weights
      - ``"unanimous"``: all active indicators must agree

    Returns a pd.Series of integer scores.
    """
    active = architecture.get("indicators", [])
    aggregation = architecture.get("score_aggregation", "additive")
    concept_weights = architecture.get("concept_weights", {})
    score = pd.Series(0.0, index=regime.index)

    if aggregation == "additive":
        for name in active:
            if name in signals:
                bullish = (signals[name] == 1).astype(float)
                score = score + bullish

    elif aggregation == "weighted":
        total_weight = 0.0
        for name in active:
            if name in signals:
                w = concept_weights.get(name, 1.0)
                bullish = (signals[name] == 1).astype(float)
                score = score + bullish * w
                total_weight += w
        if total_weight > 0:
            score = score * (len(active) / total_weight)

    elif aggregation == "unanimous":
        all_bull = pd.Series(True, index=regime.index)
        for name in active:
            if name in signals:
                all_bull = all_bull & (signals[name] == 1)
        score = all_bull.astype(float) * len(active)

    # Regime bonus/penalty
    regime_bonus = params.get("regime_bonus", 0)
    if regime_bonus > 0:
        score = score + (regime == "R1").astype(float) * regime_bonus
        score = score + (regime == "R2").astype(float) * (regime_bonus * 0.5)
        score = score - (regime == "R3").astype(float) * (regime_bonus * 0.5)
        score = score - (regime == "R4").astype(float) * regime_bonus

    return score.astype(int)


def run_backtest(df, signals_data, architecture, params):
    """
    Bar-by-bar long-only backtest engine with multiple simultaneous exit methods.

    Iterates through the execution-timeframe DataFrame.  Enters long when the
    composite score >= min_score AND regime != R4.  Tracks MFE/MAE and handles
    the following exit methods simultaneously (first trigger wins):

      - ``"fixed_target"``:  exit at entry + atr_target_mult * ATR
      - ``"fixed_stop"``:    exit at entry - atr_stop_mult * ATR
      - ``"trailing_stop"``: chandelier trail, activates after trail_activate_atr
      - ``"regime_exit"``:   forced exit when regime transitions to R4
      - ``"time_exit"``:     forced exit after max_hold_bars

    Look-ahead safety: the signal at bar i-1 fills at the OPEN of bar i.

    Returns (trades_list, stats_dict).
    """
    min_score = architecture.get("min_score", 5)
    exit_methods = architecture.get("exit_methods", ["trailing_stop", "regime_exit", "time_exit"])

    atr_stop_mult = params.get("atr_stop_mult", 1.5)
    atr_target_mult = params.get("atr_target_mult", 2.5)
    atr_trail_mult = params.get("atr_trail_mult", 1.0)
    trail_activate_atr = params.get("trail_activate_atr", 1.0)
    max_hold_bars = params.get("max_hold_bars", 35)
    commission_pct = params.get("commission_pct", 0.05)

    regime = signals_data["regime"]
    score = signals_data["score"]
    atr = signals_data["atr"]

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    dt = df["datetime"].values

    regime_vals = regime.values
    score_vals = score.values
    atr_vals = atr.values

    # Entry condition: score >= min_score AND regime is not R4
    entry_ok = np.array(
        [(score_vals[i] >= min_score and regime_vals[i] != "R4" and
          not np.isnan(atr_vals[i]) and atr_vals[i] > 0)
         for i in range(len(close))],
        dtype=bool,
    )

    use_fixed_target = "fixed_target" in exit_methods
    use_fixed_stop = "fixed_stop" in exit_methods or "trailing_stop" in exit_methods
    use_trailing = "trailing_stop" in exit_methods
    use_regime_exit = "regime_exit" in exit_methods
    use_time_exit = "time_exit" in exit_methods

    trades = []
    in_pos = False
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    target_price = 0.0
    trail_active = False
    trail_stop = 0.0
    high_since = 0.0
    low_since = 0.0
    bars_held = 0
    entry_idx = 0
    entry_regime = ""
    entry_dt = None
    mfe = 0.0
    mae = 0.0

    for i in range(1, len(close)):
        if not in_pos:
            # Signal from prior bar fills at current bar's open (no look-ahead)
            if entry_ok[i - 1]:
                in_pos = True
                entry_price = open_[i]
                entry_atr = atr_vals[i - 1]
                stop_price = entry_price - atr_stop_mult * entry_atr
                target_price = entry_price + atr_target_mult * entry_atr
                trail_active = False
                trail_stop = stop_price
                high_since = high[i]
                low_since = low[i]
                bars_held = 0
                entry_idx = i
                entry_regime = regime_vals[i - 1]
                entry_dt = dt[i]
                mfe = 0.0
                mae = 0.0
        else:
            bars_held += 1

            # Track MFE / MAE
            if high[i] > high_since:
                high_since = high[i]
            if low[i] < low_since:
                low_since = low[i]

            fav_pnl_pct = (high_since - entry_price) / entry_price * 100.0
            adv_pnl_pct = (low_since - entry_price) / entry_price * 100.0

            if fav_pnl_pct > mfe:
                mfe = fav_pnl_pct
            if adv_pnl_pct < mae:
                mae = adv_pnl_pct

            exit_reason = None

            # 1) Fixed stop
            if use_fixed_stop and exit_reason is None:
                if low[i] <= stop_price:
                    exit_reason = "fixed_stop"

            # 2) Fixed target
            if use_fixed_target and exit_reason is None:
                if high[i] >= target_price:
                    exit_reason = "fixed_target"

            # 3) Trailing stop (chandelier)
            if use_trailing and exit_reason is None:
                gain_in_atr = (close[i] - entry_price) / entry_atr if entry_atr > 0 else 0.0
                if gain_in_atr >= trail_activate_atr:
                    trail_active = True
                if trail_active:
                    new_trail = high_since - atr_trail_mult * entry_atr
                    if new_trail > trail_stop:
                        trail_stop = new_trail
                    if low[i] <= trail_stop:
                        exit_reason = "trailing_stop"

            # 4) Regime exit
            if use_regime_exit and exit_reason is None:
                if regime_vals[i] == "R4":
                    exit_reason = "regime_exit"

            # 5) Time exit
            if use_time_exit and exit_reason is None:
                if bars_held >= max_hold_bars:
                    exit_reason = "time_exit"

            if exit_reason is not None:
                if exit_reason == "fixed_target":
                    exit_price = target_price
                elif exit_reason == "fixed_stop":
                    exit_price = stop_price
                elif exit_reason == "trailing_stop":
                    exit_price = trail_stop
                else:
                    exit_price = close[i]

                # Clamp exit price to bar range
                exit_price = max(low[i], min(high[i], exit_price))

                pnl_pct = (exit_price - entry_price) / entry_price * 100.0
                net_pnl_pct = pnl_pct - 2.0 * commission_pct

                trades.append({
                    "entry_datetime": str(entry_dt),
                    "exit_datetime": str(dt[i]),
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "pnl_pct": round(net_pnl_pct, 4),
                    "gross_pnl_pct": round(pnl_pct, 4),
                    "mfe": round(mfe, 4),
                    "mae": round(mae, 4),
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                    "entry_regime": entry_regime,
                    "entry_atr": round(entry_atr, 4),
                    "entry_score": int(score_vals[entry_idx]),
                    "direction": "long",
                })
                in_pos = False

    stats = compute_stats(trades)
    return trades, stats


def compute_stats(trades):
    """
    Compute comprehensive performance statistics from a list of trade dicts.

    Returns dict with: trades, pf, wr_pct, total_return_pct, max_dd_pct,
    sharpe, sortino, edge_ratio, avg_bars_held, avg_pnl, avg_win, avg_loss,
    largest_win, largest_loss, exit_reason_counts, per-regime trade counts.
    """
    if not trades:
        return {
            "trades": 0, "pf": 0.0, "wr_pct": 0.0,
            "total_return_pct": 0.0, "max_dd_pct": 0.0,
            "sharpe": 0.0, "sortino": 0.0, "edge_ratio": 0.0,
            "avg_bars_held": 0.0, "avg_pnl": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0,
            "largest_win": 0.0, "largest_loss": 0.0,
            "regime_exit_count": 0, "exit_reason_counts": {},
            "r1_trades": 0, "r2_trades": 0, "r3_trades": 0, "r4_trades": 0,
        }

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    if gross_loss < 0.5 or len(losses) < 3:
        pf = min(gross_profit / max(gross_loss, 0.5), 10.0)
    else:
        pf = gross_profit / gross_loss
    pf = min(pf, 10.0)
    wr = len(wins) / len(pnls) * 100.0

    # Equity curve and drawdown
    equity = 10000.0
    peak_eq = equity
    max_dd = 0.0
    for p in pnls:
        equity *= (1.0 + p / 100.0)
        if equity > peak_eq:
            peak_eq = equity
        dd = (peak_eq - equity) / peak_eq * 100.0
        if dd > max_dd:
            max_dd = dd

    total_return = (equity / 10000.0 - 1.0) * 100.0

    pnl_arr = np.array(pnls)
    mean_pnl = float(np.mean(pnl_arr))
    std_pnl = float(np.std(pnl_arr)) if len(pnl_arr) > 1 else 0.001
    if std_pnl < 0.001:
        std_pnl = 0.001
    sharpe = (mean_pnl / std_pnl) * math.sqrt(min(len(trades), 250))

    downside = pnl_arr[pnl_arr < 0]
    if len(downside) > 1:
        downside_std = float(np.std(downside))
        if downside_std < 0.001:
            downside_std = 0.001
        sortino = (mean_pnl / downside_std) * math.sqrt(min(len(trades), 250))
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0.0

    mfes = [t["mfe"] for t in trades]
    maes = [abs(t["mae"]) for t in trades]
    mean_mae = float(np.mean(maes)) if maes else 0.001
    if mean_mae < 0.001:
        mean_mae = 0.001
    edge_ratio = float(np.mean(mfes)) / mean_mae

    avg_bars = float(np.mean([t["bars_held"] for t in trades]))
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    largest_win = max(pnls) if pnls else 0.0
    largest_loss = min(pnls) if pnls else 0.0

    exit_counts = {}
    for t in trades:
        reason = t.get("exit_reason", "unknown")
        exit_counts[reason] = exit_counts.get(reason, 0) + 1

    regime_exit_count = exit_counts.get("regime_exit", 0)

    r1 = sum(1 for t in trades if t.get("entry_regime") == "R1")
    r2 = sum(1 for t in trades if t.get("entry_regime") == "R2")
    r3 = sum(1 for t in trades if t.get("entry_regime") == "R3")
    r4 = sum(1 for t in trades if t.get("entry_regime") == "R4")

    return {
        "trades": len(trades),
        "pf": round(pf, 3),
        "wr_pct": round(wr, 2),
        "total_return_pct": round(total_return, 2),
        "max_dd_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "edge_ratio": round(edge_ratio, 3),
        "avg_bars_held": round(avg_bars, 1),
        "avg_pnl": round(mean_pnl, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "largest_win": round(largest_win, 4),
        "largest_loss": round(largest_loss, 4),
        "regime_exit_count": regime_exit_count,
        "exit_reason_counts": exit_counts,
        "r1_trades": r1,
        "r2_trades": r2,
        "r3_trades": r3,
        "r4_trades": r4,
    }


# ============================================================
# Single-pass backtest wrapper
# ============================================================

def full_backtest(df, daily_df, architecture, params):
    """
    End-to-end single-pass backtest.

    1. Compute indicator signals
    2. Compute regime
    3. Compute entry score
    4. Run bar-by-bar backtest

    Returns (trades, stats).
    """
    signals = compute_indicator_signals(df, architecture, params)
    atr = signals.pop("_atr")

    regime_model = architecture.get("regime_model", "ema")
    regime = compute_regime(df, daily_df, regime_model, params)

    score = compute_entry_score(signals, regime, architecture, params)

    signals_data = {
        "signals": signals,
        "regime": regime,
        "score": score,
        "atr": atr,
    }

    return run_backtest(df, signals_data, architecture, params)


# ============================================================
# DEFAULT ARCHITECTURE AND PARAMS
# ============================================================

DEFAULT_ARCHITECTURE = {
    "indicators": ["RSI", "Keltner", "VolumeSurge", "MACD", "EMA_Cross", "VWAP"],
    "min_score": 4,
    "exit_methods": ["fixed_target", "fixed_stop", "trailing_stop", "regime_exit", "time_exit"],
    "regime_model": "ema",
    "position_sizing": "equal",
    "exec_timeframe": "1H",
    "score_aggregation": "additive",
    "concept_weights": {},
}

DEFAULT_PARAMS = {
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "boll_period": 20,
    "boll_std": 2.0,
    "stoch_k": 14,
    "stoch_d": 3,
    "obv_ma_period": 20,
    "adx_period": 14,
    "adx_threshold": 25,
    "cci_period": 20,
    "cci_oversold": -100,
    "cci_overbought": 100,
    "willr_period": 14,
    "willr_oversold": -80,
    "willr_overbought": -20,
    "keltner_period": 20,
    "keltner_mult": 2.0,
    "volume_surge_ma": 20,
    "volume_surge_mult": 1.5,
    "ema_fast": 9,
    "ema_slow": 21,
    "atr_period": 14,
    "atr_stop_mult": 1.5,
    "atr_target_mult": 2.5,
    "atr_trail_mult": 1.0,
    "trail_activate_atr": 1.0,
    "max_hold_bars": 35,
    "commission_pct": 0.05,
    "regime_bonus": 0,
}


# ============================================================
# 9. CHECKPOINT HELPERS
# ============================================================

def save_checkpoint(name, data, output_dir=None):
    """Save pipeline checkpoint as JSON to *output_dir*."""
    od = Path(output_dir) if output_dir else OUTPUT_DIR
    od.mkdir(parents=True, exist_ok=True)
    cp_dir = od / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    path = cp_dir / f"{name}.json"

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp, datetime)):
            return str(o)
        if isinstance(o, Path):
            return str(o)
        return str(o)

    with open(path, "w") as f:
        json.dump(data, f, default=_default, indent=2)
    log(f"Checkpoint saved: {path}")


def load_checkpoint(name, output_dir=None):
    """Load a checkpoint by name. Returns dict or None if not found."""
    od = Path(output_dir) if output_dir else OUTPUT_DIR
    path = od / "checkpoints" / f"{name}.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        data = json.load(f)
    log(f"Checkpoint loaded: {path}")
    return data


# ============================================================
# 10. LAYER 1 - ARCHITECTURE SEARCH
# ============================================================

def _compute_fitness(stats):
    """Fitness = PF * sqrt(trades) * (1 - max_dd/100). Penalises small samples."""
    pf = stats.get("pf", 0.0)
    trades = stats.get("trades", 0)
    max_dd = stats.get("max_dd_pct", 100.0)
    if trades < 5 or pf <= 0:
        return -999.0
    return pf * math.sqrt(trades) * (1.0 - max_dd / 100.0)


def _mini_monte_carlo(trade_pnls, n_sims=200, threshold=0.7):
    """
    Quick Monte Carlo: shuffle trades *n_sims* times, return a penalty
    multiplier in [0, 1] based on fraction of net-profitable runs.
    """
    if len(trade_pnls) < 5:
        return 0.0
    rng = np.random.RandomState(42)
    arr = np.array(trade_pnls)
    profit_count = 0
    for _ in range(n_sims):
        rng.shuffle(arr)
        equity = 10000.0
        for p in arr:
            equity *= (1.0 + p / 100.0)
        if equity > 10000.0:
            profit_count += 1
    prob_profit = profit_count / n_sims
    if prob_profit >= threshold:
        return 1.0
    return prob_profit / threshold


def _select_indicators_biased(trial, concept_bias, min_count=3, max_count=8):
    """Select indicator subset biased by concept weights."""
    all_names = list(INDICATOR_REGISTRY.keys())
    n_indicators = trial.suggest_int("n_indicators", min_count, min(max_count, len(all_names)))
    selected = []
    available = list(all_names)
    for idx in range(n_indicators):
        weights = np.array([concept_bias.get(nm, 1.0) for nm in available], dtype=np.float64)
        weights = np.clip(weights, 0.1, 10.0)
        weights /= weights.sum()
        cumulative = np.cumsum(weights)
        pick_val = trial.suggest_float(f"ind_pick_{idx}", 0.0, 1.0)
        pick_idx = int(np.searchsorted(cumulative, pick_val))
        pick_idx = min(pick_idx, len(available) - 1)
        selected.append(available[pick_idx])
        available.pop(pick_idx)
        if not available:
            break
    return list(set(selected)) if selected else all_names[:min_count]


def architecture_trial(trial, data_dict, concept_bias, cfg):
    """
    Optuna objective for Layer 1: search over architecture space.

    Runs a quick inner tune on a subset of symbols to evaluate architectural
    fitness.
    """
    indicators = _select_indicators_biased(trial, concept_bias)

    exit_combos = [
        ["fixed_target", "fixed_stop", "trailing_stop", "regime_exit", "time_exit"],
        ["trailing_stop", "regime_exit", "time_exit"],
        ["fixed_target", "fixed_stop", "regime_exit"],
        ["trailing_stop", "time_exit"],
        ["fixed_target", "trailing_stop", "regime_exit", "time_exit"],
    ]
    exit_idx = trial.suggest_int("exit_combo", 0, len(exit_combos) - 1)
    exit_methods = exit_combos[exit_idx]

    regime_model = trial.suggest_categorical("regime_model", ["ema", "volatility", "trend"])
    position_sizing = trial.suggest_categorical("position_sizing", ["equal", "volatility_scaled"])
    score_aggregation = trial.suggest_categorical("score_aggregation", ["additive", "weighted", "unanimous"])

    min_score = trial.suggest_int("min_score", max(2, len(indicators) // 2), max(3, len(indicators) - 1))

    architecture = {
        "indicators": indicators,
        "min_score": min_score,
        "exit_methods": exit_methods,
        "regime_model": regime_model,
        "position_sizing": position_sizing,
        "exec_timeframe": cfg.get("phase3_params", {}).get("exec_timeframe", "1H"),
        "score_aggregation": score_aggregation,
        "concept_weights": concept_bias,
    }

    inner_trials = cfg.get("optimization", {}).get("inner_trials", 30)
    symbols = list(data_dict.keys())[:5]
    if not symbols:
        return -999.0

    arch_fitness_values = []
    inner_study = None
    for sym in symbols:
        sym_data = data_dict[sym]
        df = sym_data.get("exec_df")
        daily_df = sym_data.get("daily_df")
        if df is None or len(df) < 100:
            continue

        def inner_objective(inner_trial):
            params = dict(DEFAULT_PARAMS)
            for ind_name in indicators:
                reg = INDICATOR_REGISTRY.get(ind_name, {})
                for pname, (lo, hi) in reg.get("params", {}).items():
                    if isinstance(lo, float) or isinstance(hi, float):
                        params[pname] = inner_trial.suggest_float(pname, float(lo), float(hi))
                    else:
                        params[pname] = inner_trial.suggest_int(pname, int(lo), int(hi))
            params["atr_stop_mult"] = inner_trial.suggest_float("atr_stop_mult", 0.8, 3.0)
            params["atr_target_mult"] = inner_trial.suggest_float("atr_target_mult", 1.5, 5.0)
            params["atr_trail_mult"] = inner_trial.suggest_float("atr_trail_mult", 0.5, 2.5)
            params["trail_activate_atr"] = inner_trial.suggest_float("trail_activate_atr", 0.5, 2.0)
            params["max_hold_bars"] = inner_trial.suggest_int("max_hold_bars", 10, 60)
            params["regime_bonus"] = inner_trial.suggest_int("regime_bonus", 0, 2)

            _, stats = full_backtest(df, daily_df, architecture, params)
            return _compute_fitness(stats)

        inner_study = optuna.create_study(direction="maximize",
                                          sampler=optuna.samplers.TPESampler(seed=42))
        inner_study.optimize(inner_objective, n_trials=inner_trials, show_progress_bar=False)
        best_val = inner_study.best_value if inner_study.best_trial else -999.0
        arch_fitness_values.append(best_val)

    if not arch_fitness_values or all(v <= -999 for v in arch_fitness_values):
        return -999.0

    valid = [v for v in arch_fitness_values if v > -999]
    if not valid:
        return -999.0

    mean_fitness = float(np.mean(valid))

    # Mini Monte Carlo penalty using best inner params on first symbol
    first_sym = symbols[0]
    first_data = data_dict[first_sym]
    df0 = first_data.get("exec_df")
    daily0 = first_data.get("daily_df")
    if df0 is not None and len(df0) >= 100 and inner_study is not None:
        best_params = dict(DEFAULT_PARAMS)
        if inner_study.best_trial:
            best_params.update(inner_study.best_params)
        trades0, _ = full_backtest(df0, daily0, architecture, best_params)
        pnls0 = [t["pnl_pct"] for t in trades0]
        mc_mult = _mini_monte_carlo(pnls0)
        mean_fitness *= mc_mult

    return mean_fitness


def layer1_architecture_search(data_dict, concept_bias, cfg):
    """
    Layer 1: Optuna search over the architecture space. Returns the best
    architecture dict.
    """
    arch_trials = cfg.get("optimization", {}).get("arch_trials", 20)
    log(f"=== LAYER 1: Architecture Search ({arch_trials} trials) ===")

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))

    def objective(trial):
        return architecture_trial(trial, data_dict, concept_bias, cfg)

    study.optimize(objective, n_trials=arch_trials, show_progress_bar=True)

    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -9999, reverse=True)
    log("Top 5 architectures:")
    for rank, t in enumerate(sorted_trials[:5], 1):
        log(f"  #{rank}: fitness={t.value:.4f} params={t.params}")

    best = study.best_trial
    best_params = best.params

    # Reconstruct indicators from the best trial's biased picks
    n_ind = best_params.get("n_indicators", 5)
    all_names = list(INDICATOR_REGISTRY.keys())
    available = list(all_names)
    selected = []
    for idx in range(n_ind):
        pick_val = best_params.get(f"ind_pick_{idx}", 0.5)
        weights = np.array([concept_bias.get(nm, 1.0) for nm in available], dtype=np.float64)
        weights = np.clip(weights, 0.1, 10.0)
        weights /= weights.sum()
        cumulative = np.cumsum(weights)
        pick_idx = int(np.searchsorted(cumulative, pick_val))
        pick_idx = min(pick_idx, len(available) - 1)
        selected.append(available[pick_idx])
        available.pop(pick_idx)
        if not available:
            break
    selected = list(set(selected)) if selected else all_names[:3]

    exit_combos = [
        ["fixed_target", "fixed_stop", "trailing_stop", "regime_exit", "time_exit"],
        ["trailing_stop", "regime_exit", "time_exit"],
        ["fixed_target", "fixed_stop", "regime_exit"],
        ["trailing_stop", "time_exit"],
        ["fixed_target", "trailing_stop", "regime_exit", "time_exit"],
    ]
    exit_idx = best_params.get("exit_combo", 0)
    exit_methods = exit_combos[exit_idx]

    architecture = {
        "indicators": selected,
        "min_score": best_params.get("min_score", 4),
        "exit_methods": exit_methods,
        "regime_model": best_params.get("regime_model", "ema"),
        "position_sizing": best_params.get("position_sizing", "equal"),
        "exec_timeframe": cfg.get("phase3_params", {}).get("exec_timeframe", "1H"),
        "score_aggregation": best_params.get("score_aggregation", "additive"),
        "concept_weights": concept_bias,
    }

    log(f"Best architecture: {architecture}")
    save_checkpoint("layer1_architecture", {"architecture": architecture, "fitness": best.value})
    return architecture


# ============================================================
# 11. LAYER 2 - DEEP PARAMETER OPTIMIZATION
# ============================================================

def deep_tune_objective(trial, sym, df_dict, architecture, cfg):
    """
    Per-symbol deep parameter tuning objective with walk-forward validation.

    Splits data 70/30 (IS/OOS), runs backtest on both, returns blended fitness.
    """
    sym_data = df_dict[sym]
    df = sym_data.get("exec_df")
    daily_df = sym_data.get("daily_df")

    if df is None or len(df) < 100:
        return -999.0

    # Suggest all numerical params
    params = dict(DEFAULT_PARAMS)
    active_indicators = architecture.get("indicators", [])
    for ind_name in active_indicators:
        reg = INDICATOR_REGISTRY.get(ind_name, {})
        for pname, (lo, hi) in reg.get("params", {}).items():
            if isinstance(lo, float) or isinstance(hi, float):
                params[pname] = trial.suggest_float(pname, float(lo), float(hi))
            else:
                params[pname] = trial.suggest_int(pname, int(lo), int(hi))

    params["atr_period"] = trial.suggest_int("atr_period", 10, 21)
    params["atr_stop_mult"] = trial.suggest_float("atr_stop_mult", 0.8, 3.0)
    params["atr_target_mult"] = trial.suggest_float("atr_target_mult", 1.5, 5.0)
    params["atr_trail_mult"] = trial.suggest_float("atr_trail_mult", 0.5, 2.5)
    params["trail_activate_atr"] = trial.suggest_float("trail_activate_atr", 0.3, 2.5)
    params["max_hold_bars"] = trial.suggest_int("max_hold_bars", 10, 60)
    params["regime_bonus"] = trial.suggest_int("regime_bonus", 0, 3)
    params["commission_pct"] = trial.suggest_float("commission_pct", 0.03, 0.10)
    params["min_score"] = trial.suggest_int("min_score_tune", 2, max(2, len(active_indicators)))
    architecture = dict(architecture)
    architecture["min_score"] = params["min_score"]

    # Walk-forward split: 70% IS, 30% OOS
    split_idx = int(len(df) * 0.7)
    df_is = df.iloc[:split_idx].reset_index(drop=True)
    df_oos = df.iloc[split_idx:].reset_index(drop=True)

    if daily_df is not None and len(daily_df) > 0:
        split_date = df["datetime"].iloc[split_idx]
        daily_is = daily_df[daily_df["datetime"] <= split_date].reset_index(drop=True)
        daily_oos = daily_df[daily_df["datetime"] > split_date].reset_index(drop=True)
        if len(daily_oos) < 20:
            daily_oos = daily_df.copy()
    else:
        daily_is = daily_df
        daily_oos = daily_df

    if len(df_is) < 80 or len(df_oos) < 30:
        return -999.0

    _, stats_is = full_backtest(df_is, daily_is, architecture, params)
    _, stats_oos = full_backtest(df_oos, daily_oos, architecture, params)

    fitness_is = _compute_fitness(stats_is)
    fitness_oos = _compute_fitness(stats_oos)

    if fitness_is <= -999 or fitness_oos <= -999:
        return -999.0

    # Blended fitness: favour OOS
    is_w = cfg.get("optimization", {}).get("fitness_is_weight", 0.4)
    oos_w = cfg.get("optimization", {}).get("fitness_oos_weight", 0.6)
    fitness = is_w * fitness_is + oos_w * fitness_oos

    # Require a minimum number of trades on both slices
    if stats_is.get("trades", 0) < 6 or stats_oos.get("trades", 0) < 3:
        return -999.0

    # Reject curve-fit PF artifacts
    if stats_is.get("pf", 0) > 12.0:
        return -999.0

    # Reject severe IS/OOS divergence (memorization signature)
    if abs(fitness_is) > 1e-6:
        divergence = abs(fitness_is - fitness_oos) / abs(fitness_is)
        if divergence > 0.8:
            return -999.0

    return fitness


def layer2_deep_tune(data_dict, architecture, survivors, cfg):
    """
    Layer 2: per-symbol deep parameter optimization.

    Runs an Optuna study per symbol with TPE sampler.
    Returns dict of {sym: {"params": best_params, "stats": stats, ...}}.
    """
    deep_trials = cfg.get("optimization", {}).get("deep_trials", 100)
    log(f"=== LAYER 2: Deep Parameter Optimization ({deep_trials} trials/symbol) ===")

    results = {}
    for idx, sym in enumerate(survivors, 1):
        if sym not in data_dict:
            log(f"  [{idx}/{len(survivors)}] {sym} - no data, skipping", "WARN")
            continue

        log(f"  [{idx}/{len(survivors)}] Tuning {sym}...")
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))

        def objective(trial, _sym=sym):
            return deep_tune_objective(trial, _sym, data_dict, architecture, cfg)

        study.optimize(objective, n_trials=deep_trials, show_progress_bar=False)

        if study.best_trial is None or study.best_value <= -999:
            log(f"    {sym}: no valid solution found", "WARN")
            continue

        best_params = dict(DEFAULT_PARAMS)
        best_params.update(study.best_params)

        sym_data = data_dict[sym]
        df = sym_data.get("exec_df")
        daily_df = sym_data.get("daily_df")
        trades, stats = full_backtest(df, daily_df, architecture, best_params)

        trade_pnls = [t["pnl_pct"] for t in trades]

        results[sym] = {
            "params": best_params,
            "stats": stats,
            "trade_pnls": trade_pnls,
            "trades": trades,
            "fitness": study.best_value,
        }
        log(f"    {sym}: PF={stats['pf']:.2f}, WR={stats['wr_pct']:.1f}%, "
            f"trades={stats['trades']}, fitness={study.best_value:.4f}")

    save_checkpoint("layer2_tuned", {sym: {k: v for k, v in r.items() if k != "trades"}
                                     for sym, r in results.items()})
    log(f"Layer 2 complete: {len(results)} symbols tuned")
    return results


# ============================================================
# 12. LAYER 3 - ROBUSTNESS GAUNTLET
# ============================================================

def monte_carlo_validate(trade_pnls, n_sims=3000, initial_equity=10000):
    """
    Monte Carlo validation: shuffle trade returns *n_sims* times and compute
    probability and percentile statistics on the resulting equity curves.
    """
    if len(trade_pnls) < 5:
        return {
            "prob_profit": 0.0, "p5_equity": initial_equity,
            "p50_equity": initial_equity, "p95_equity": initial_equity,
            "p95_dd": 100.0,
        }

    rng = np.random.RandomState(42)
    arr = np.array(trade_pnls, dtype=np.float64)
    final_equities = []
    max_drawdowns = []

    for _ in range(n_sims):
        shuffled = rng.permutation(arr)
        equity = float(initial_equity)
        peak = equity
        worst_dd = 0.0
        for p in shuffled:
            equity *= (1.0 + p / 100.0)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
            if dd > worst_dd:
                worst_dd = dd
        final_equities.append(equity)
        max_drawdowns.append(worst_dd)

    final_equities = np.array(final_equities)
    max_drawdowns = np.array(max_drawdowns)

    return {
        "prob_profit": float(np.mean(final_equities > initial_equity)),
        "p5_equity": float(np.percentile(final_equities, 5)),
        "p50_equity": float(np.percentile(final_equities, 50)),
        "p95_equity": float(np.percentile(final_equities, 95)),
        "p95_dd": float(np.percentile(max_drawdowns, 95)),
    }


def noise_injection_test(df_dict, architecture, params, cfg):
    """
    Inject noise into data and measure backtest degradation.

      * Add +/-5% random noise to close prices
      * Shift close by 1 bar to simulate timing jitter

    Returns dict with clean_pf, noisy_pf, pf_retention.
    """
    df = df_dict.get("exec_df")
    daily_df = df_dict.get("daily_df")
    if df is None or len(df) < 100:
        return {"clean_pf": 0.0, "noisy_pf": 0.0, "pf_retention": 0.0}

    _, stats_clean = full_backtest(df, daily_df, architecture, params)
    clean_pf = stats_clean.get("pf", 0.0)

    rng = np.random.RandomState(123)
    df_noisy = df.copy()
    noise = rng.uniform(-0.05, 0.05, size=len(df_noisy))
    df_noisy["close"] = df_noisy["close"] * (1.0 + noise)
    df_noisy["high"] = np.maximum(df_noisy["high"], df_noisy["close"])
    df_noisy["low"] = np.minimum(df_noisy["low"], df_noisy["close"])
    df_noisy["close"] = df_noisy["close"].shift(1).bfill()

    _, stats_noisy = full_backtest(df_noisy, daily_df, architecture, params)
    noisy_pf = stats_noisy.get("pf", 0.0)

    pf_retention = noisy_pf / clean_pf if clean_pf > 0 else 0.0

    return {
        "clean_pf": round(clean_pf, 3),
        "noisy_pf": round(noisy_pf, 3),
        "pf_retention": round(pf_retention, 4),
    }


def regime_stress_test(df_dict, architecture, params, cfg):
    """
    Flip the regime model to measure regime sensitivity.

    The stressed run re-evaluates the strategy with an alternate regime model
    (``trend`` if the baseline was ``ema``, else ``ema``).
    """
    df = df_dict.get("exec_df")
    daily_df = df_dict.get("daily_df")
    if df is None or len(df) < 100:
        return {"normal_pf": 0.0, "stressed_pf": 0.0, "pf_retention": 0.0}

    _, stats_normal = full_backtest(df, daily_df, architecture, params)
    normal_pf = stats_normal.get("pf", 0.0)

    arch_stressed = dict(architecture)
    baseline = architecture.get("regime_model", "ema")
    arch_stressed["regime_model"] = "trend" if baseline != "trend" else "ema"

    _, stats_stressed = full_backtest(df, daily_df, arch_stressed, params)
    stressed_pf = stats_stressed.get("pf", 0.0)

    pf_retention = stressed_pf / normal_pf if normal_pf > 0 else 0.0

    return {
        "normal_pf": round(normal_pf, 3),
        "stressed_pf": round(stressed_pf, 3),
        "pf_retention": round(pf_retention, 4),
    }


def param_sensitivity_test(df_dict, architecture, params, cfg):
    """
    Jitter each numerical parameter by +/-10% and measure PF stability.

    Returns dict of {param_name: {"stable": bool, "pf_range": [min, max]}}.
    """
    df = df_dict.get("exec_df")
    daily_df = df_dict.get("daily_df")
    if df is None or len(df) < 100:
        return {}

    _, stats_base = full_backtest(df, daily_df, architecture, params)
    base_pf = stats_base.get("pf", 0.0)

    sensitivity = {}
    numeric_params = {k: v for k, v in params.items()
                      if isinstance(v, (int, float)) and k not in ("commission_pct",)}

    for pname, pval in numeric_params.items():
        if pval == 0:
            continue
        pf_values = []
        for jitter in [-0.10, 0.10]:
            test_params = dict(params)
            jittered = pval * (1.0 + jitter)
            if isinstance(pval, int):
                jittered = max(1, int(round(jittered)))
            else:
                jittered = round(jittered, 6)
            test_params[pname] = jittered
            _, test_stats = full_backtest(df, daily_df, architecture, test_params)
            pf_values.append(test_stats.get("pf", 0.0))

        pf_min = min(pf_values)
        pf_max = max(pf_values)
        stable = True
        if base_pf > 0:
            if pf_min < base_pf * 0.7 or pf_max > base_pf * 1.3:
                stable = False
        sensitivity[pname] = {
            "stable": stable,
            "pf_range": [round(pf_min, 3), round(pf_max, 3)],
            "base_pf": round(base_pf, 3),
        }

    return sensitivity


def layer3_robustness_gauntlet(data_dict, architecture, tuned_results, cfg):
    """
    Layer 3: comprehensive robustness testing.

    Monte Carlo, noise injection, regime stress, and parameter sensitivity
    tests are combined into a composite score per symbol.
    """
    threshold = cfg.get("optimization", {}).get("robustness_threshold", 0.4)
    log(f"=== LAYER 3: Robustness Gauntlet (threshold={threshold}) ===")

    validated = {}
    robustness_data = {}

    for idx, (sym, result) in enumerate(tuned_results.items(), 1):
        log(f"  [{idx}/{len(tuned_results)}] Testing {sym}...")
        params = result["params"]
        trade_pnls = result["trade_pnls"]
        sym_data = data_dict.get(sym, {})

        mc = monte_carlo_validate(trade_pnls, n_sims=cfg.get("robustness", {}).get("monte_carlo_sims", 3000))
        mc_score = mc["prob_profit"]

        noise = noise_injection_test(sym_data, architecture, params, cfg)
        noise_score = min(1.0, max(0.0, noise["pf_retention"]))

        stress = regime_stress_test(sym_data, architecture, params, cfg)
        stress_score = min(1.0, max(0.0, stress["pf_retention"]))

        sensitivity = param_sensitivity_test(sym_data, architecture, params, cfg)
        if sensitivity:
            stable_count = sum(1 for v in sensitivity.values() if v["stable"])
            sensitivity_score = stable_count / len(sensitivity)
        else:
            sensitivity_score = 0.5

        scores = [max(0.001, mc_score), max(0.001, noise_score),
                  max(0.001, stress_score), max(0.001, sensitivity_score)]
        composite = float(np.prod(scores) ** (1.0 / len(scores)))

        robustness_data[sym] = {
            "mc": mc,
            "noise": noise,
            "stress": stress,
            "sensitivity": sensitivity,
            "mc_score": round(mc_score, 4),
            "noise_score": round(noise_score, 4),
            "stress_score": round(stress_score, 4),
            "sensitivity_score": round(sensitivity_score, 4),
            "composite": round(composite, 4),
        }

        log(f"    {sym}: MC={mc_score:.3f} Noise={noise_score:.3f} "
            f"Stress={stress_score:.3f} Sens={sensitivity_score:.3f} "
            f"Composite={composite:.3f}")

        if composite >= threshold:
            validated[sym] = result
            validated[sym]["robustness"] = robustness_data[sym]
            log(f"    {sym}: PASSED")
        else:
            log(f"    {sym}: REJECTED (composite {composite:.3f} < {threshold})")

    save_checkpoint("layer3_robustness", robustness_data)
    log(f"Layer 3 complete: {len(validated)}/{len(tuned_results)} passed")
    return validated, robustness_data


# ============================================================
# 13. CORRELATION FILTER
# ============================================================

def correlation_filter(validated_results, cfg):
    """
    Filter validated symbols by pairwise return correlation and sector caps.

    - Reject one of any pair with trade-return correlation > max_corr
    - Max N symbols per sector (using SECTOR_MAP)
    """
    max_corr = cfg.get("optimization", {}).get("max_correlation", 0.70)
    max_per_sector = cfg.get("optimization", {}).get("max_per_sector", 3)
    log(f"=== CORRELATION FILTER (max_corr={max_corr}, max_sector={max_per_sector}) ===")

    if len(validated_results) <= 1:
        return validated_results

    syms = list(validated_results.keys())
    syms.sort(key=lambda s: validated_results[s].get("fitness", 0), reverse=True)

    return_series = {}
    for sym in syms:
        pnls = validated_results[sym].get("trade_pnls", [])
        if pnls:
            return_series[sym] = pd.Series(pnls)

    rejected = set()
    sym_list = [s for s in syms if s in return_series]
    for i in range(len(sym_list)):
        if sym_list[i] in rejected:
            continue
        for j in range(i + 1, len(sym_list)):
            if sym_list[j] in rejected:
                continue
            s1 = return_series[sym_list[i]]
            s2 = return_series[sym_list[j]]
            min_len = min(len(s1), len(s2))
            if min_len < 10:
                continue
            corr = float(s1.iloc[:min_len].corr(s2.iloc[:min_len]))
            if abs(corr) > max_corr:
                loser = sym_list[j]
                if loser in FORCED_SYMBOLS:
                    loser = sym_list[i]
                    if loser in FORCED_SYMBOLS:
                        continue
                rejected.add(loser)
                log(f"  Correlation filter: {sym_list[i]} vs {loser} = {corr:.3f} -> reject {loser}")

    sector_counts = {}
    sector_rejected = set()
    for sym in syms:
        if sym in rejected:
            continue
        sector = SECTOR_MAP.get(sym, "Unknown")
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if sector_counts[sector] > max_per_sector and sym not in FORCED_SYMBOLS:
            sector_rejected.add(sym)
            log(f"  Sector cap: {sym} ({sector}) -> rejected (already {max_per_sector} in sector)")

    all_rejected = rejected | sector_rejected
    final = {s: v for s, v in validated_results.items() if s not in all_rejected}
    log(f"Correlation filter: {len(final)}/{len(validated_results)} survived")
    return final


# ============================================================
# 14. FULL FINAL BACKTEST (TUNE + TRUE HOLDOUT)
# ============================================================

def phase_full_backtest(data_dict, architecture, final_results, cfg, tuned_results=None):
    """
    Re-run backtest with final params on the FULL tune window AND the held-out
    final window that no optimization layer has ever touched.

    Backtests EVERY tuned symbol (not just the survivors of the correlation
    filter) so headline stats reflect the full optimized universe and aren't
    inflated by post-selection bias.  The survivor subset is reported
    separately.  The TRUE HOLDOUT block is the honest performance number.
    """
    log("=== PHASE: Full Final Backtest ===")
    all_trades = []
    per_symbol = {}
    universe_source = tuned_results if tuned_results else final_results
    survivor_set = set(final_results.keys())

    sorted_syms = sorted(universe_source.keys(),
                         key=lambda s: universe_source[s].get("fitness", 0), reverse=True)

    for sym in sorted_syms:
        sym_data = data_dict.get(sym, {})
        df = sym_data.get("exec_df")
        daily_df = sym_data.get("daily_df")
        params = universe_source[sym]["params"]

        if df is None or len(df) < 100:
            continue

        trades, stats = full_backtest(df, daily_df, architecture, params)

        survived = sym in survivor_set
        for t in trades:
            t["symbol"] = sym
            t["survived"] = survived
            t["phase"] = "tune"
        all_trades.extend(trades)

        # True OOS backtest on the never-seen final holdout window
        holdout_df = sym_data.get("exec_df_holdout")
        holdout_daily = sym_data.get("daily_df_holdout")
        holdout_trades, holdout_stats = [], {}
        if holdout_df is not None and len(holdout_df) >= 50:
            holdout_trades, holdout_stats = full_backtest(
                holdout_df, holdout_daily, architecture, params
            )
            for t in holdout_trades:
                t["symbol"] = sym
                t["survived"] = survived
                t["phase"] = "holdout"

        per_symbol[sym] = {"trades": trades, "stats": stats, "params": params,
                           "survived": survived,
                           "holdout_trades": holdout_trades,
                           "holdout_stats": holdout_stats}
        log(f"  {sym}{' [SURVIVOR]' if survived else ''}: tune={stats['trades']}tr "
            f"PF={stats['pf']:.2f} Ret={stats['total_return_pct']:.1f}% | "
            f"holdout={holdout_stats.get('trades', 0)}tr "
            f"PF={holdout_stats.get('pf', 0):.2f} Ret={holdout_stats.get('total_return_pct', 0):.1f}%")

    all_trades.sort(key=lambda t: t["entry_datetime"])

    # Build portfolio equity curve (equal weight per trade)
    n_symbols = max(1, len(sorted_syms))
    equity = 10000.0
    peak = equity
    max_dd = 0.0
    equity_dates = []
    equity_values = []
    for t in all_trades:
        weight = 1.0 / n_symbols
        pnl_pct = t["pnl_pct"] * weight
        equity *= (1.0 + pnl_pct / 100.0)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100.0
        if dd > max_dd:
            max_dd = dd
        equity_dates.append(t["exit_datetime"])
        equity_values.append(equity)

    # Fetch SPY for benchmark comparison
    spy_sym, spy_df, spy_status = fetch_daily("SPY")
    benchmark = None
    if spy_df is not None and len(spy_df) > 50:
        spy_returns = spy_df["close"].pct_change().dropna()
        spy_equity = 10000.0
        spy_curve = [spy_equity]
        for r in spy_returns:
            spy_equity *= (1.0 + r)
            spy_curve.append(spy_equity)
        benchmark = {
            "dates": spy_df["datetime"].tolist(),
            "equity": spy_curve,
            "total_return_pct": round((spy_equity / 10000.0 - 1.0) * 100.0, 2),
        }

    portfolio_stats = compute_stats(all_trades)
    if equity_values:
        port_return = (equity_values[-1] / 10000.0 - 1.0) * 100.0
        portfolio_stats["total_return_pct"] = round(port_return, 2)
        portfolio_stats["max_dd_pct"] = round(max_dd, 2)
    survivor_trades = [t for t in all_trades if t.get("survived")]
    survivor_stats = compute_stats(survivor_trades) if survivor_trades else {}

    holdout_all_trades = []
    for sym in per_symbol:
        holdout_all_trades.extend(per_symbol[sym].get("holdout_trades", []))
    holdout_universe_stats = compute_stats(holdout_all_trades) if holdout_all_trades else {}
    holdout_survivor_trades = [t for t in holdout_all_trades if t.get("survived")]
    holdout_survivor_stats = compute_stats(holdout_survivor_trades) if holdout_survivor_trades else {}

    results = {
        "all_trades": all_trades,
        "survivor_trades": survivor_trades,
        "holdout_all_trades": holdout_all_trades,
        "holdout_survivor_trades": holdout_survivor_trades,
        "per_symbol": per_symbol,
        "sorted_syms": sorted_syms,
        "survivor_syms": sorted([s for s in sorted_syms if per_symbol.get(s, {}).get("survived")]),
        "portfolio_stats": portfolio_stats,
        "survivor_stats": survivor_stats,
        "holdout_universe_stats": holdout_universe_stats,
        "holdout_survivor_stats": holdout_survivor_stats,
        "equity_dates": equity_dates,
        "equity_values": equity_values,
        "max_dd": round(max_dd, 2),
        "benchmark": benchmark,
    }
    log(f"TUNE Universe ({len(sorted_syms)} syms): {len(all_trades)} trades, "
        f"PF={portfolio_stats['pf']:.2f}, MaxDD={max_dd:.1f}%")
    if survivor_stats:
        log(f"TUNE Survivors ({len(survivor_set)} syms): {len(survivor_trades)} trades, "
            f"PF={survivor_stats.get('pf', 0):.2f} [post-selection, biased]")
    if holdout_universe_stats:
        log(f"HOLDOUT Universe: {len(holdout_all_trades)} trades, "
            f"PF={holdout_universe_stats.get('pf', 0):.2f}, "
            f"WR={holdout_universe_stats.get('wr_pct', 0):.1f}%, "
            f"Ret={holdout_universe_stats.get('total_return_pct', 0):.1f}% "
            f"[TRUE OOS - never seen by optimizer]")
    if holdout_survivor_stats:
        log(f"HOLDOUT Survivors: {len(holdout_survivor_trades)} trades, "
            f"PF={holdout_survivor_stats.get('pf', 0):.2f}, "
            f"Ret={holdout_survivor_stats.get('total_return_pct', 0):.1f}% "
            f"[TRUE OOS - survivors only]")
    return results


# ============================================================
# 15. HTML REPORT GENERATION
# ============================================================

def generate_html_report(results, architecture, robustness_data, run_info, output_dir):
    """
    Generate a complete self-contained HTML report with embedded Plotly charts.

    The headline number is the TRUE HOLDOUT result (the window the optimizer
    never saw) whenever holdout data exists, with a TUNE-vs-HOLDOUT diagnostic
    table so the reader can see performance decay.
    """
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)

    holdout_stats = results.get("holdout_universe_stats", {}) or {}
    tune_stats = results.get("portfolio_stats", {}) or {}
    use_holdout = holdout_stats.get("trades", 0) > 0
    stats = holdout_stats if use_holdout else tune_stats
    headline_label = "TRUE OOS HOLDOUT" if use_holdout else "TUNE WINDOW (biased)"
    all_trades = results.get("all_trades", [])

    # Exit reason breakdown
    exit_counts = {}
    for t in all_trades:
        r = t.get("exit_reason", "unknown")
        exit_counts[r] = exit_counts.get(r, 0) + 1
    _target_n = exit_counts.get("fixed_target", 0)
    _atr_n = exit_counts.get("fixed_stop", 0)
    _trail_n = exit_counts.get("trailing_stop", 0)
    _regime_n = exit_counts.get("regime_exit", 0)
    _time_n = exit_counts.get("time_exit", 0)
    _exit_total = max(1, sum(exit_counts.values()))
    exit_html = f"""
    <div style="margin:20px;padding:15px;border:1px solid #30363d;background:#161b22;">
      <h2>Exit Reason Breakdown</h2>
      <ul>
        <li>fixed_target: {_target_n} ({_target_n/_exit_total*100:.1f}%)</li>
        <li>fixed_stop: {_atr_n} ({_atr_n/_exit_total*100:.1f}%)</li>
        <li>trailing_stop: {_trail_n} ({_trail_n/_exit_total*100:.1f}%)</li>
        <li>regime_exit: {_regime_n} ({_regime_n/_exit_total*100:.1f}%)</li>
        <li>time_exit: {_time_n} ({_time_n/_exit_total*100:.1f}%)</li>
      </ul>
    </div>
    """

    header_banner = f"""
    <div style="background:#1a3a52;color:white;padding:20px;text-align:center;font-size:24px;">
      Optuna Screener Results &mdash; {headline_label}
    </div>
    """

    if holdout_stats.get("trades", 0) > 0 and tune_stats.get("trades", 0) > 0:
        diag_html = f"""
        <div style="margin:20px;">
        <h2>Diagnostic: TUNE vs HOLDOUT</h2>
        <table border="1" style="border-collapse:collapse;width:100%;">
          <tr><th>Metric</th><th>TUNE (biased)</th><th>HOLDOUT (true OOS)</th><th>Decay</th></tr>
          <tr><td>Trades</td><td>{tune_stats.get('trades', 0)}</td>
              <td>{holdout_stats.get('trades', 0)}</td><td>&mdash;</td></tr>
          <tr><td>PF</td><td>{tune_stats.get('pf', 0):.2f}</td>
              <td>{holdout_stats.get('pf', 0):.2f}</td>
              <td>{(1 - holdout_stats.get('pf', 0) / max(tune_stats.get('pf', 1), 0.01)) * 100:.0f}%</td></tr>
          <tr><td>Sharpe</td><td>{tune_stats.get('sharpe', 0):.2f}</td>
              <td>{holdout_stats.get('sharpe', 0):.2f}</td>
              <td>{(1 - holdout_stats.get('sharpe', 0) / max(tune_stats.get('sharpe', 1), 0.01)) * 100:.0f}%</td></tr>
          <tr><td>Win %</td><td>{tune_stats.get('wr_pct', 0):.1f}%</td>
              <td>{holdout_stats.get('wr_pct', 0):.1f}%</td><td>&mdash;</td></tr>
          <tr><td>Return</td><td>{tune_stats.get('total_return_pct', 0):.1f}%</td>
              <td>{holdout_stats.get('total_return_pct', 0):.1f}%</td><td>&mdash;</td></tr>
        </table>
        </div>
        """
    else:
        diag_html = ""
    per_symbol = results.get("per_symbol", {})
    sorted_syms = results.get("sorted_syms", [])
    equity_dates = results.get("equity_dates", [])
    equity_values = results.get("equity_values", [])
    benchmark = results.get("benchmark")

    eq_dates_json = json.dumps(equity_dates)
    eq_vals_json = json.dumps([round(v, 2) for v in equity_values])

    bench_trace = ""
    if benchmark:
        bench_dates = json.dumps([str(d) for d in benchmark["dates"]])
        bench_vals = json.dumps([round(v, 2) for v in benchmark["equity"]])
        bench_trace = f"""
        {{
            x: {bench_dates},
            y: {bench_vals},
            type: 'scatter',
            mode: 'lines',
            name: 'SPY Benchmark',
            line: {{color: '#888888', dash: 'dot'}}
        }},"""

    dd_values = []
    peak_eq = 10000.0
    for v in equity_values:
        if v > peak_eq:
            peak_eq = v
        dd = (peak_eq - v) / peak_eq * 100.0
        dd_values.append(round(-dd, 2))
    dd_json = json.dumps(dd_values)

    # Monthly returns heatmap
    monthly_data = {}
    for t in all_trades:
        try:
            dt_str = t.get("exit_datetime", "")[:10]
            dt_obj = datetime.strptime(dt_str, "%Y-%m-%d")
            key = (dt_obj.year, dt_obj.month)
            monthly_data[key] = monthly_data.get(key, 0.0) + t["pnl_pct"]
        except (ValueError, TypeError):
            continue

    years = sorted(set(k[0] for k in monthly_data.keys())) if monthly_data else [2024]
    months = list(range(1, 13))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    heatmap_z = []
    for y in years:
        row = []
        for m in months:
            row.append(round(monthly_data.get((y, m), 0.0), 2))
        heatmap_z.append(row)
    heatmap_z_json = json.dumps(heatmap_z)
    heatmap_y_json = json.dumps([str(y) for y in years])
    heatmap_x_json = json.dumps(month_names)

    sym_rows = ""
    for sym in sorted_syms:
        if sym not in per_symbol:
            continue
        s = per_symbol[sym]["stats"]
        rob = robustness_data.get(sym, {})
        sym_rows += f"""<tr>
            <td>{sym}</td><td>{SECTOR_MAP.get(sym, 'Unknown')}</td>
            <td>{s.get('trades', 0)}</td><td>{s.get('pf', 0):.2f}</td>
            <td>{s.get('wr_pct', 0):.1f}%</td><td>{s.get('total_return_pct', 0):.1f}%</td>
            <td>{s.get('max_dd_pct', 0):.1f}%</td><td>{s.get('sharpe', 0):.2f}</td>
            <td>{rob.get('composite', 0):.3f}</td>
        </tr>\n"""

    trade_rows = ""
    display_trades = all_trades[-200:] if len(all_trades) > 200 else all_trades
    for t in display_trades:
        pnl_class = "positive" if t["pnl_pct"] > 0 else "negative"
        trade_rows += f"""<tr class="{pnl_class}">
            <td>{t.get('symbol', '')}</td>
            <td>{t['entry_datetime'][:19]}</td><td>{t['exit_datetime'][:19]}</td>
            <td>{t['entry_price']:.2f}</td><td>{t['exit_price']:.2f}</td>
            <td>{t['pnl_pct']:.2f}%</td><td>{t['bars_held']}</td>
            <td>{t['exit_reason']}</td><td>{t['entry_regime']}</td>
        </tr>\n"""

    rob_rows = ""
    for sym in sorted_syms:
        rd = robustness_data.get(sym, {})
        if not rd:
            continue
        mc = rd.get("mc", {})
        rob_rows += f"""<tr>
            <td>{sym}</td>
            <td>{rd.get('mc_score', 0):.3f}</td>
            <td>{rd.get('noise_score', 0):.3f}</td>
            <td>{rd.get('stress_score', 0):.3f}</td>
            <td>{rd.get('sensitivity_score', 0):.3f}</td>
            <td>{rd.get('composite', 0):.3f}</td>
            <td>{mc.get('prob_profit', 0):.1%}</td>
            <td>{mc.get('p95_dd', 0):.1f}%</td>
        </tr>\n"""

    arch_desc = f"""
    <strong>Indicators:</strong> {', '.join(architecture.get('indicators', []))}<br>
    <strong>Exit Methods:</strong> {', '.join(architecture.get('exit_methods', []))}<br>
    <strong>Regime Model:</strong> {architecture.get('regime_model', 'N/A')}<br>
    <strong>Score Aggregation:</strong> {architecture.get('score_aggregation', 'N/A')}<br>
    <strong>Position Sizing:</strong> {architecture.get('position_sizing', 'N/A')}<br>
    <strong>Min Score:</strong> {architecture.get('min_score', 'N/A')}<br>
    <strong>Timeframe:</strong> {architecture.get('exec_timeframe', 'N/A')}
    """

    param_imp_rows = ""
    all_sens = {}
    for sym in sorted_syms:
        rd = robustness_data.get(sym, {})
        sens = rd.get("sensitivity", {})
        for pname, pdata in sens.items():
            if pname not in all_sens:
                all_sens[pname] = {"stable_count": 0, "total": 0}
            all_sens[pname]["total"] += 1
            if pdata.get("stable", False):
                all_sens[pname]["stable_count"] += 1
    for pname, pdata in sorted(all_sens.items(), key=lambda x: x[1]["stable_count"] / max(1, x[1]["total"])):
        stability_pct = pdata["stable_count"] / max(1, pdata["total"]) * 100
        param_imp_rows += f"""<tr>
            <td>{pname}</td>
            <td>{stability_pct:.0f}%</td>
            <td>{pdata['total']}</td>
        </tr>\n"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Optuna Screener Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', Tahoma, sans-serif; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: #58a6ff; font-size: 28px; margin-bottom: 10px; }}
  h2 {{ color: #58a6ff; font-size: 22px; margin: 30px 0 15px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
  h3 {{ color: #8b949e; font-size: 16px; margin: 15px 0 10px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }}
  .stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center; }}
  .stat-card .value {{ font-size: 24px; font-weight: bold; color: #58a6ff; }}
  .stat-card .label {{ font-size: 12px; color: #8b949e; margin-top: 5px; }}
  .stat-card.positive .value {{ color: #3fb950; }}
  .stat-card.negative .value {{ color: #f85149; }}
  .arch-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 15px 0; line-height: 1.8; }}
  table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: #161b22; border-radius: 8px; overflow: hidden; }}
  th {{ background: #21262d; color: #58a6ff; padding: 10px 12px; text-align: left; font-size: 13px; cursor: pointer; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; font-size: 13px; }}
  tr:hover {{ background: #1c2128; }}
  tr.positive td:nth-child(6) {{ color: #3fb950; }}
  tr.negative td:nth-child(6) {{ color: #f85149; }}
  .chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin: 15px 0; }}
  .tab-container {{ display: flex; gap: 5px; margin: 20px 0 0; flex-wrap: wrap; }}
  .tab {{ padding: 10px 20px; background: #21262d; border: 1px solid #30363d; border-bottom: none;
           border-radius: 8px 8px 0 0; cursor: pointer; color: #8b949e; font-size: 14px; }}
  .tab.active {{ background: #161b22; color: #58a6ff; border-color: #58a6ff; }}
  .tab-page {{ display: none; border: 1px solid #30363d; border-radius: 0 8px 8px 8px; padding: 20px; background: #161b22; }}
  .tab-page.active {{ display: block; }}
  .footer {{ text-align: center; color: #484f58; margin-top: 40px; padding: 20px; font-size: 12px; }}
</style>
</head>
<body>
{header_banner}
{diag_html}
{exit_html}
<div class="container">
<h1>Optuna Screener Report</h1>
<p style="color:#8b949e;">Generated: {run_info.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
| Concept: {run_info.get('concept', 'N/A')} | Symbols: {len(sorted_syms)}</p>

<div class="tab-container">
  <div class="tab active" onclick="showTab(0)">Executive Summary</div>
  <div class="tab" onclick="showTab(1)">Equity &amp; Returns</div>
  <div class="tab" onclick="showTab(2)">Per-Symbol</div>
  <div class="tab" onclick="showTab(3)">Trade Journal</div>
  <div class="tab" onclick="showTab(4)">Optimization</div>
  <div class="tab" onclick="showTab(5)">Robustness</div>
</div>

<!-- PAGE 1: Executive Summary -->
<div class="tab-page active" id="page0">
<h2>Executive Summary</h2>
<div class="stats-grid">
  <div class="stat-card {'positive' if stats.get('total_return_pct', 0) > 0 else 'negative'}">
    <div class="value">{stats.get('total_return_pct', 0):.1f}%</div><div class="label">Total Return</div>
  </div>
  <div class="stat-card"><div class="value">{stats.get('pf', 0):.2f}</div><div class="label">Profit Factor</div></div>
  <div class="stat-card"><div class="value">{stats.get('wr_pct', 0):.1f}%</div><div class="label">Win Rate</div></div>
  <div class="stat-card"><div class="value">{stats.get('trades', 0)}</div><div class="label">Total Trades</div></div>
  <div class="stat-card negative"><div class="value">{stats.get('max_dd_pct', 0):.1f}%</div><div class="label">Max Drawdown</div></div>
  <div class="stat-card"><div class="value">{stats.get('sharpe', 0):.2f}</div><div class="label">Sharpe Ratio</div></div>
  <div class="stat-card"><div class="value">{stats.get('sortino', 0):.2f}</div><div class="label">Sortino Ratio</div></div>
  <div class="stat-card"><div class="value">{stats.get('edge_ratio', 0):.2f}</div><div class="label">Edge Ratio</div></div>
  <div class="stat-card"><div class="value">{stats.get('avg_bars_held', 0):.1f}</div><div class="label">Avg Bars Held</div></div>
  <div class="stat-card positive"><div class="value">{stats.get('avg_win', 0):.2f}%</div><div class="label">Avg Win</div></div>
  <div class="stat-card negative"><div class="value">{stats.get('avg_loss', 0):.2f}%</div><div class="label">Avg Loss</div></div>
  <div class="stat-card"><div class="value">{len(sorted_syms)}</div><div class="label">Symbols</div></div>
</div>
<h3>Architecture</h3>
<div class="arch-box">{arch_desc}</div>
</div>

<!-- PAGE 2: Equity & Returns -->
<div class="tab-page" id="page1">
<h2>Equity Curve</h2>
<div class="chart-container" id="equity-chart" style="height:450px;"></div>
<h2>Drawdown</h2>
<div class="chart-container" id="dd-chart" style="height:300px;"></div>
<h2>Monthly Returns Heatmap</h2>
<div class="chart-container" id="heatmap-chart" style="height:350px;"></div>
</div>

<!-- PAGE 3: Per-Symbol -->
<div class="tab-page" id="page2">
<h2>Per-Symbol Results</h2>
<table id="sym-table">
<thead><tr>
  <th onclick="sortTable('sym-table',0)">Symbol</th>
  <th onclick="sortTable('sym-table',1)">Sector</th>
  <th onclick="sortTable('sym-table',2)">Trades</th>
  <th onclick="sortTable('sym-table',3)">PF</th>
  <th onclick="sortTable('sym-table',4)">Win Rate</th>
  <th onclick="sortTable('sym-table',5)">Return</th>
  <th onclick="sortTable('sym-table',6)">Max DD</th>
  <th onclick="sortTable('sym-table',7)">Sharpe</th>
  <th onclick="sortTable('sym-table',8)">Robustness</th>
</tr></thead>
<tbody>{sym_rows}</tbody>
</table>
</div>

<!-- PAGE 4: Trade Journal -->
<div class="tab-page" id="page3">
<h2>Trade Journal (last 200)</h2>
<table id="trade-table">
<thead><tr>
  <th>Symbol</th><th>Entry</th><th>Exit</th><th>Entry$</th><th>Exit$</th>
  <th>PnL%</th><th>Bars</th><th>Exit Reason</th><th>Regime</th>
</tr></thead>
<tbody>{trade_rows}</tbody>
</table>
</div>

<!-- PAGE 5: Optimization -->
<div class="tab-page" id="page4">
<h2>Parameter Importance</h2>
<table>
<thead><tr><th>Parameter</th><th>Stability</th><th>Symbols Tested</th></tr></thead>
<tbody>{param_imp_rows}</tbody>
</table>
</div>

<!-- PAGE 6: Robustness -->
<div class="tab-page" id="page5">
<h2>Robustness Gauntlet Results</h2>
<table>
<thead><tr>
  <th>Symbol</th><th>MC Score</th><th>Noise Score</th><th>Stress Score</th>
  <th>Sensitivity</th><th>Composite</th><th>MC Prob Profit</th><th>MC P95 DD</th>
</tr></thead>
<tbody>{rob_rows}</tbody>
</table>
</div>

</div><!-- end container -->

<div class="footer">
Optuna Screener Pipeline | Report generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>

<script>
function showTab(idx) {{
  document.querySelectorAll('.tab-page').forEach((p, i) => {{
    p.classList.toggle('active', i === idx);
  }});
  document.querySelectorAll('.tab').forEach((t, i) => {{
    t.classList.toggle('active', i === idx);
  }});
  if (idx === 1) {{
    setTimeout(() => {{
      Plotly.Plots.resize('equity-chart');
      Plotly.Plots.resize('dd-chart');
      Plotly.Plots.resize('heatmap-chart');
    }}, 100);
  }}
}}

function sortTable(tableId, col) {{
  var table = document.getElementById(tableId);
  var tbody = table.querySelector('tbody');
  var rows = Array.from(tbody.querySelectorAll('tr'));
  var dir = table.dataset.sortDir === 'asc' ? 'desc' : 'asc';
  table.dataset.sortDir = dir;
  rows.sort(function(a, b) {{
    var va = a.cells[col].textContent.replace('%','').trim();
    var vb = b.cells[col].textContent.replace('%','').trim();
    var na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) {{
      return dir === 'asc' ? na - nb : nb - na;
    }}
    return dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

var darkLayout = {{
  paper_bgcolor: '#161b22',
  plot_bgcolor: '#161b22',
  font: {{ color: '#c9d1d9' }},
  xaxis: {{ gridcolor: '#21262d', linecolor: '#30363d' }},
  yaxis: {{ gridcolor: '#21262d', linecolor: '#30363d' }},
  margin: {{ l: 60, r: 30, t: 40, b: 50 }},
}};

Plotly.newPlot('equity-chart', [
  {bench_trace}
  {{
    x: {eq_dates_json},
    y: {eq_vals_json},
    type: 'scatter',
    mode: 'lines',
    name: 'Portfolio',
    line: {{ color: '#58a6ff', width: 2 }}
  }}
], Object.assign({{}}, darkLayout, {{
  title: 'Portfolio Equity Curve',
  yaxis: {{ title: 'Equity ($)', gridcolor: '#21262d' }}
}}), {{ responsive: true }});

Plotly.newPlot('dd-chart', [{{
  x: {eq_dates_json},
  y: {dd_json},
  type: 'scatter',
  mode: 'lines',
  fill: 'tozeroy',
  name: 'Drawdown',
  line: {{ color: '#f85149' }},
  fillcolor: 'rgba(248,81,73,0.2)'
}}], Object.assign({{}}, darkLayout, {{
  title: 'Drawdown',
  yaxis: {{ title: 'Drawdown (%)', gridcolor: '#21262d' }}
}}), {{ responsive: true }});

Plotly.newPlot('heatmap-chart', [{{
  z: {heatmap_z_json},
  x: {heatmap_x_json},
  y: {heatmap_y_json},
  type: 'heatmap',
  colorscale: [
    [0, '#f85149'],
    [0.5, '#161b22'],
    [1, '#3fb950']
  ],
  showscale: true,
  colorbar: {{ title: 'Return %', tickfont: {{ color: '#c9d1d9' }} }}
}}], Object.assign({{}}, darkLayout, {{
  title: 'Monthly Returns',
  yaxis: {{ autorange: 'reversed' }}
}}), {{ responsive: true }});
</script>
</body>
</html>"""

    report_path = od / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"HTML report saved: {report_path}")
    return str(report_path)


def generate_trades_csv(all_trades, output_dir):
    """Export all trades to CSV."""
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    path = od / "trades.csv"
    if not all_trades:
        log("No trades to export")
        return str(path)
    df = pd.DataFrame(all_trades)
    df.to_csv(path, index=False)
    log(f"Trades CSV saved: {path}")
    return str(path)


def generate_summary_csv(results, output_dir):
    """Export per-symbol summary to CSV."""
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    path = od / "summary.csv"
    per_symbol = results.get("per_symbol", {})
    rows = []
    for sym, data in per_symbol.items():
        s = data.get("stats", {})
        row = {"symbol": sym, "sector": SECTOR_MAP.get(sym, "Unknown")}
        row.update(s)
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
    log(f"Summary CSV saved: {path}")
    return str(path)


def generate_parameters_json(results, architecture, output_dir):
    """Export optimized parameters and architecture to JSON."""
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    path = od / "parameters.json"

    per_symbol = results.get("per_symbol", {})
    params_out = {}
    for sym, data in per_symbol.items():
        params_out[sym] = data.get("params", {})

    output = {
        "architecture": architecture,
        "per_symbol_params": params_out,
        "portfolio_stats": results.get("portfolio_stats", {}),
    }

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w") as f:
        json.dump(output, f, default=_default, indent=2)
    log(f"Parameters JSON saved: {path}")
    return str(path)


# ============================================================
# 16. AMIBROKER INTEGRATION
# ============================================================

def generate_apex_afl(sorted_syms, results, architecture):
    """
    Generate an AmiBroker AFL formula string with per-symbol optimized
    parameters, indicator computations, regime shading, buy/sell arrows, and
    exploration columns. Pure generic indicator logic - no extra gates.
    """
    per_symbol = results.get("per_symbol", {})
    indicators = architecture.get("indicators", [])

    sym_blocks = []
    for sym in sorted_syms:
        if sym not in per_symbol:
            continue
        p = per_symbol[sym].get("params", {})
        block_lines = [f'    if (Name() == "{sym}")']
        block_lines.append("    {")
        for k, v in sorted(p.items()):
            if isinstance(v, float):
                block_lines.append(f'        {k} = {v:.4f};')
            elif isinstance(v, int):
                block_lines.append(f'        {k} = {v};')
        block_lines.append("    }")
        sym_blocks.append("\n".join(block_lines))

    sym_param_code = "\n    else ".join(sym_blocks)

    ind_code_lines = []
    if "RSI" in indicators:
        ind_code_lines.append("rsi_val = RSI(rsi_period);")
        ind_code_lines.append("rsi_bull = rsi_val < rsi_oversold;")
        ind_code_lines.append("rsi_bear = rsi_val > rsi_overbought;")
    if "MACD" in indicators:
        ind_code_lines.append("macd_line = MACD(macd_fast, macd_slow);")
        ind_code_lines.append("signal_line = Signal(macd_fast, macd_slow, macd_signal);")
        ind_code_lines.append("macd_bull = Cross(macd_line, signal_line);")
        ind_code_lines.append("macd_bear = Cross(signal_line, macd_line);")
    if "Bollinger" in indicators:
        ind_code_lines.append("bb_top = BBandTop(C, boll_period, boll_std);")
        ind_code_lines.append("bb_bot = BBandBot(C, boll_period, boll_std);")
        ind_code_lines.append("boll_bull = C <= bb_bot;")
        ind_code_lines.append("boll_bear = C >= bb_top;")
    if "EMA_Cross" in indicators:
        ind_code_lines.append("ema_f = EMA(C, ema_fast);")
        ind_code_lines.append("ema_s = EMA(C, ema_slow);")
        ind_code_lines.append("ema_bull = Cross(ema_f, ema_s);")
        ind_code_lines.append("ema_bear = Cross(ema_s, ema_f);")
    if "Stochastic" in indicators:
        ind_code_lines.append("stoch_k_val = StochK(stoch_k, stoch_d);")
        ind_code_lines.append("stoch_d_val = StochD(stoch_k, stoch_d);")
        ind_code_lines.append("stoch_bull = stoch_k_val < 20 AND stoch_k_val > stoch_d_val;")
        ind_code_lines.append("stoch_bear = stoch_k_val > 80 AND stoch_k_val < stoch_d_val;")
    if "ADX" in indicators:
        ind_code_lines.append("adx_val = ADX(adx_period);")
        ind_code_lines.append("adx_trending = adx_val > adx_threshold;")
    if "CCI" in indicators:
        ind_code_lines.append("cci_val = CCI(cci_period);")
        ind_code_lines.append("cci_bull = cci_val < cci_oversold;")
        ind_code_lines.append("cci_bear = cci_val > cci_overbought;")
    if "Keltner" in indicators:
        ind_code_lines.append("kelt_mid = EMA(C, keltner_period);")
        ind_code_lines.append("kelt_atr = ATR(14);")
        ind_code_lines.append("kelt_upper = kelt_mid + keltner_mult * kelt_atr;")
        ind_code_lines.append("kelt_lower = kelt_mid - keltner_mult * kelt_atr;")
        ind_code_lines.append("kelt_bull = C <= kelt_lower;")
        ind_code_lines.append("kelt_bear = C >= kelt_upper;")
    if "VolumeSurge" in indicators:
        ind_code_lines.append("vol_ma = MA(V, volume_surge_ma);")
        ind_code_lines.append("vol_surge_sig = V > vol_ma * volume_surge_mult;")
        ind_code_lines.append("vol_bull = vol_surge_sig AND C > O;")
        ind_code_lines.append("vol_bear = vol_surge_sig AND C < O;")
    if "VWAP" in indicators:
        ind_code_lines.append("tp = (H + L + C) / 3;")
        ind_code_lines.append("newday = Day() != Ref(Day(), -1);")
        ind_code_lines.append("cum_tpv = SumSince(newday, tp * V);")
        ind_code_lines.append("cum_v = SumSince(newday, V);")
        ind_code_lines.append("vwap_val = IIf(cum_v > 0, cum_tpv / cum_v, C);")
        ind_code_lines.append("vwap_bull = C > vwap_val;")
        ind_code_lines.append("vwap_bear = C < vwap_val;")
    if "OBV" in indicators:
        ind_code_lines.append("obv_val = OBV();")
        ind_code_lines.append("obv_ma_val = MA(obv_val, obv_ma_period);")
        ind_code_lines.append("obv_bull = obv_val > obv_ma_val;")
        ind_code_lines.append("obv_bear = obv_val < obv_ma_val;")
    if "WilliamsR" in indicators:
        ind_code_lines.append("wr_val = (HHV(H, willr_period) - C) / (HHV(H, willr_period) - LLV(L, willr_period)) * -100;")
        ind_code_lines.append("wr_bull = wr_val < willr_oversold;")
        ind_code_lines.append("wr_bear = wr_val > willr_overbought;")

    ind_code = "\n    ".join(ind_code_lines)

    score_parts = []
    for ind in indicators:
        lname = ind.lower()
        if ind == "EMA_Cross":
            lname = "ema"
        elif ind == "VolumeSurge":
            lname = "vol"
        score_parts.append(f"{lname}_bull")
    score_formula = " + ".join(score_parts) if score_parts else "0"

    afl = f"""// ============================================================
// Optuna Screener Optimized Strategy - Auto-generated AFL
// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// Indicators: {', '.join(indicators)}
// ============================================================

// --- Default Params (overridden per-symbol below) ---
rsi_period = 14; rsi_oversold = 30; rsi_overbought = 70;
macd_fast = 12; macd_slow = 26; macd_signal = 9;
boll_period = 20; boll_std = 2.0;
stoch_k = 14; stoch_d = 3;
obv_ma_period = 20;
adx_period = 14; adx_threshold = 25;
cci_period = 20; cci_oversold = -100; cci_overbought = 100;
willr_period = 14; willr_oversold = -80; willr_overbought = -20;
keltner_period = 20; keltner_mult = 2.0;
volume_surge_ma = 20; volume_surge_mult = 1.5;
ema_fast = 9; ema_slow = 21;
atr_stop_mult = 1.5; atr_target_mult = 2.5;
atr_trail_mult = 1.0; trail_activate_atr = 1.0;
max_hold_bars = 35;
min_score = {architecture.get('min_score', 4)};

// --- Per-Symbol Optimized Params ---
{sym_param_code}

// --- Indicator Computation ---
{ind_code}

// --- Composite Score ---
score = {score_formula};

// --- Regime (simplified: EMA trend) ---
regime_ema20 = EMA(C, 20);
regime_ema50 = EMA(C, 50);
regime_bull = regime_ema20 > regime_ema50;

// --- Entry / Exit ---
entry_atr = ATR(14);
Buy = score >= min_score AND regime_bull;
BuyPrice = Close;

trail_stop = Highest(H, BarsSince(Buy)) - atr_trail_mult * entry_atr;
target_price = ValueWhen(Buy, BuyPrice) + atr_target_mult * ValueWhen(Buy, entry_atr);
stop_price = ValueWhen(Buy, BuyPrice) - atr_stop_mult * ValueWhen(Buy, entry_atr);

Sell = L <= trail_stop OR H >= target_price OR L <= stop_price OR
       BarsSince(Buy) >= max_hold_bars OR NOT regime_bull;
SellPrice = Close;

Buy = ExRem(Buy, Sell);
Sell = ExRem(Sell, Buy);

// --- Charting ---
SetChartOptions(0, chartShowArrows | chartShowDates);
_SECTION_BEGIN("Optuna Screener Price");
Plot(C, "Close", colorDefault, styleCandle);
Plot(regime_ema20, "EMA20", colorYellow, styleLine);
Plot(regime_ema50, "EMA50", colorOrange, styleLine);

clr = IIf(regime_bull, ColorBlend(colorGreen, colorBlack, 0.85),
                        ColorBlend(colorRed, colorBlack, 0.85));
Plot(1, "", clr, styleArea | styleOwnScale | styleNoLabel, 0, 1);

PlotShapes(IIf(Buy, shapeUpArrow, shapeNone), colorBrightGreen, 0, L, -20);
PlotShapes(IIf(Sell, shapeDownArrow, shapeNone), colorRed, 0, H, -20);
_SECTION_END();

// --- Exploration Columns ---
Filter = Buy OR Sell;
AddTextColumn(WriteIf(Buy, "BUY", "SELL"), "Signal");
AddColumn(C, "Price", 1.2);
AddColumn(score, "Score", 1.0);
AddColumn(entry_atr, "ATR", 1.4);
AddColumn(target_price, "Target", 1.2);
AddColumn(stop_price, "Stop", 1.2);
AddTextColumn(WriteIf(regime_bull, "BULLISH", "BEARISH"), "Regime");
"""

    return afl


def push_to_amibroker(results, afl_str, output_dir, cfg):
    """
    Push results to AmiBroker via COM automation. Falls back gracefully to
    file-based output if pywin32 or AmiBroker are not available.
    """
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)

    sorted_syms = results.get("sorted_syms", [])

    afl_path = od / "OptunaScreener_Strategy.afl"
    with open(afl_path, "w", encoding="utf-8") as f:
        f.write(afl_str)
    log(f"AFL saved: {afl_path}")

    tls_path = od / "OptunaScreener_Watchlist.tls"
    with open(tls_path, "w") as f:
        for sym in sorted_syms:
            f.write(f"{sym}\n")
    log(f"Watchlist saved: {tls_path}")

    com_success = False
    try:
        import win32com.client
        ab = win32com.client.Dispatch("Broker.Application")
        log("Connected to AmiBroker via COM")

        ab_path = None
        try:
            ab_path = ab.DatabasePath
        except Exception:
            pass
        if ab_path:
            formulas_dir = Path(ab_path).parent / "Formulas" / "Custom"
            formulas_dir.mkdir(parents=True, exist_ok=True)
            dest_afl = formulas_dir / "OptunaScreener_Strategy.afl"
            with open(dest_afl, "w", encoding="utf-8") as f:
                f.write(afl_str)
            log(f"AFL copied to AmiBroker: {dest_afl}")

        try:
            wl_idx = ab.AddWatchList("OptunaScreener_Picks")
            for sym in sorted_syms:
                ab.AddToWatchList(wl_idx, sym)
            log(f"Watchlist 'OptunaScreener_Picks' created with {len(sorted_syms)} symbols")
        except Exception as e:
            log(f"Watchlist creation via COM failed: {e}", "WARN")

        ab.RefreshAll()
        com_success = True
        log("AmiBroker COM push complete")

    except ImportError:
        log("win32com not available - COM push skipped. AFL saved to disk.", "WARN")
    except Exception as e:
        log(f"AmiBroker COM error: {e}. AFL saved to disk.", "WARN")

    return com_success


# ============================================================
# 17. MAIN PIPELINE
# ============================================================

FORCED_SYMBOLS = ["SPY", "QQQ"]


def phase1_universe(cfg):
    """Return candidate symbol list from config plus forced tickers."""
    target = cfg.get("target_symbols", [])
    if not target:
        target = list(SECTOR_MAP.keys())[:30]
    for sym in FORCED_SYMBOLS:
        if sym not in target:
            target.append(sym)
    log(f"Phase 1: Universe of {len(target)} candidates (includes forced: {FORCED_SYMBOLS})")
    return target


def phase2_quick_screen(candidates, cfg):
    """
    Quick screen: fetch daily data, filter by liquidity / price / volume.

    Returns (list of surviving symbols, dict of {sym: daily_df}).
    """
    uni = cfg.get("universe", {})
    min_price = uni.get("min_price", 10)
    max_price = uni.get("max_price", 5000)
    min_volume = uni.get("min_avg_volume", 500000)
    min_bars = uni.get("min_daily_bars", 252)

    log(f"Phase 2: Quick screen on {len(candidates)} symbols "
        f"(price ${min_price}-${max_price}, vol>{min_volume/1e6:.1f}M)")

    survivors = []
    daily_data = {}

    for idx, sym in enumerate(candidates, 1):
        forced = sym in FORCED_SYMBOLS
        sym_name, df, status = fetch_daily(sym)
        if df is None or len(df) < min(min_bars, 50):
            log(f"  [{idx}/{len(candidates)}] {sym}: SKIP ({status}, bars={len(df) if df is not None else 0})"
                + (" [FORCED but no data]" if forced else ""))
            continue

        avg_price = df["close"].iloc[-20:].mean() if len(df) >= 20 else df["close"].mean()
        avg_volume = df["volume"].iloc[-20:].mean() if len(df) >= 20 else df["volume"].mean()

        if not forced:
            if avg_price < min_price or avg_price > max_price:
                log(f"  [{idx}/{len(candidates)}] {sym}: SKIP (price ${avg_price:.2f})")
                continue
            if avg_volume < min_volume:
                log(f"  [{idx}/{len(candidates)}] {sym}: SKIP (volume {avg_volume/1e6:.1f}M)")
                continue

        survivors.append(sym)
        daily_data[sym] = df
        tag = " [FORCED]" if forced else ""
        log(f"  [{idx}/{len(candidates)}] {sym}: PASS{tag} (${avg_price:.0f}, "
            f"vol={avg_volume/1e6:.1f}M, bars={len(df)})")

    log(f"Phase 2 complete: {len(survivors)}/{len(candidates)} survived")
    return survivors, daily_data


def phase3_fetch_data(survivors, daily_data, cfg):
    """
    Fetch execution-timeframe data for each survivor and split off a
    final holdout window that no optimizer will ever see.
    """
    exec_tf = cfg.get("phase3_params", {}).get("exec_timeframe", "1H")
    log(f"Phase 3: Fetching {exec_tf} data for {len(survivors)} symbols")

    data_dict = {}
    for idx, sym in enumerate(survivors, 1):
        log(f"  [{idx}/{len(survivors)}] Fetching {sym}...")

        _, exec_df_full, status = fetch_bars(sym, timeframe=exec_tf)
        if exec_df_full is None or len(exec_df_full) < 100:
            log(f"    {sym}: SKIP (exec bars: {len(exec_df_full) if exec_df_full is not None else 0})")
            continue

        daily_df_full = daily_data.get(sym)

        # Reserve the final N% as a true holdout that NO optimizer ever sees.
        # Layer 1 / Layer 2 / robustness all run on the tune window only.
        holdout_pct = cfg.get("optimization", {}).get("final_holdout_pct", 0.25)
        cut = int(len(exec_df_full) * (1.0 - holdout_pct))
        exec_df = exec_df_full.iloc[:cut].reset_index(drop=True)
        exec_df_holdout = exec_df_full.iloc[cut:].reset_index(drop=True)

        if daily_df_full is not None and len(exec_df) > 0:
            split_dt = exec_df["datetime"].iloc[-1]
            daily_df = daily_df_full[daily_df_full["datetime"] <= split_dt].reset_index(drop=True)
            daily_df_holdout = daily_df_full[daily_df_full["datetime"] > split_dt].reset_index(drop=True)
        else:
            daily_df = daily_df_full
            daily_df_holdout = None

        data_dict[sym] = {
            "exec_df": exec_df,
            "daily_df": daily_df,
            "exec_df_holdout": exec_df_holdout,
            "daily_df_holdout": daily_df_holdout,
        }
        log(f"    {sym}: OK (tune={len(exec_df)} bars, holdout={len(exec_df_holdout)} bars)")

    log(f"Phase 3 complete: {len(data_dict)} symbols with full data")
    return data_dict


def main():
    """
    Pipeline entry point.

      1. Universe selection
      2. Quick screen (daily liquidity)
      3. Multi-TF data fetch + holdout split
      4. Layer 1: Architecture search
      5. Layer 2: Deep parameter optimization
      6. Layer 3: Robustness gauntlet
      7. Correlation filter
      8. Full final backtest (tune + true holdout)
      9. Report generation (HTML + CSV + JSON)
     10. AmiBroker push
     11. Open report in browser
    """
    parser = argparse.ArgumentParser(description="Optuna Screener Pipeline")
    parser.add_argument("--config", default="apex_config.json", help="Config JSON path")
    parser.add_argument("--test", action="store_true", help="Test mode: 3 symbols, light budget")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--no-amibroker", action="store_true", help="Skip AmiBroker push")
    parser.add_argument("--concept", type=str, default="", help="Strategy concept string")
    parser.add_argument("--budget", type=str, default="medium",
                        choices=["light", "medium", "heavy"], help="Compute budget")
    parser.add_argument("--output", type=str, default="", help="Output directory override")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply budget profile
    budget = cfg.get("budget_profiles", {}).get(args.budget, {})
    opt = cfg.get("optimization", {})
    rob = cfg.get("robustness", {})
    if budget:
        opt["arch_trials"] = budget.get("arch_trials", 200)
        opt["inner_trials"] = budget.get("arch_trials", 200) // 5
        opt["deep_trials"] = budget.get("deep_trials", 800)
        rob["monte_carlo_sims"] = budget.get("mc_sims", 3000)
    else:
        opt.setdefault("arch_trials", 200)
        opt.setdefault("inner_trials", 40)
        opt.setdefault("deep_trials", 800)
    opt["robustness_threshold"] = rob.get("min_robustness_score", 0.5)
    cfg["optimization"] = opt
    cfg["robustness"] = rob
    log(f"Budget: {args.budget} - arch={opt['arch_trials']} trials, "
        f"deep={opt['deep_trials']}/symbol, MC={rob.get('monte_carlo_sims', 3000)} sims, "
        f"robustness>={opt['robustness_threshold']}")

    opt = cfg.get("optimization", {})
    if args.test:
        log("*** TEST MODE: light budget, 3 symbols ***")
        opt["arch_trials"] = 5
        opt["inner_trials"] = 10
        opt["deep_trials"] = 20
        opt["robustness_threshold"] = 0.2
        cfg["optimization"] = opt
        target = cfg.get("target_symbols", list(SECTOR_MAP.keys())[:3])
        cfg["target_symbols"] = target[:3]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        run_output = Path(args.output)
    else:
        run_output = OUTPUT_DIR / f"run_{timestamp}"
    run_output.mkdir(parents=True, exist_ok=True)

    run_info = {
        "timestamp": timestamp,
        "concept": args.concept or cfg.get("concept", "adaptive multi-indicator"),
        "test_mode": args.test,
        "config": args.config,
    }

    log("=" * 60)
    log("Optuna Screener Pipeline")
    log("=" * 60)
    log(f"Output: {run_output}")
    log(f"Concept: {run_info['concept']}")

    concept_bias = parse_concept(run_info["concept"])
    log(f"Concept bias: { {k: v for k, v in concept_bias.items() if v != 1.0} }")

    resume_stage = None
    if args.resume:
        for stage in ["layer3_robustness", "layer2_tuned", "layer1_architecture"]:
            cp = load_checkpoint(stage, str(run_output))
            if cp is not None:
                resume_stage = stage
                break
        if resume_stage:
            log(f"Resuming from checkpoint: {resume_stage}")
        else:
            log("No checkpoint found, starting fresh")

    # ---- Phase 1 ----
    candidates = phase1_universe(cfg)

    # ---- Phase 2 ----
    survivors, daily_data = phase2_quick_screen(candidates, cfg)
    if not survivors:
        log("No symbols survived Phase 2. Exiting.", "ERROR")
        sys.exit(1)

    # ---- Phase 3 ----
    data_dict = phase3_fetch_data(survivors, daily_data, cfg)
    if not data_dict:
        log("No symbols have sufficient data. Exiting.", "ERROR")
        sys.exit(1)
    survivors = list(data_dict.keys())

    # ---- Layer 1 ----
    if resume_stage in ("layer2_tuned", "layer3_robustness"):
        cp = load_checkpoint("layer1_architecture", str(run_output))
        architecture = cp["architecture"] if cp else DEFAULT_ARCHITECTURE
    else:
        architecture = layer1_architecture_search(data_dict, concept_bias, cfg)

    # ---- Layer 2 ----
    if resume_stage == "layer3_robustness":
        cp = load_checkpoint("layer2_tuned", str(run_output))
        tuned_results = cp if cp else {}
        for sym, sym_result in tuned_results.items():
            if "trades" not in sym_result and sym in data_dict:
                sd = data_dict[sym]
                params = sym_result.get("params", DEFAULT_PARAMS)
                trades, stats = full_backtest(sd["exec_df"], sd["daily_df"],
                                              architecture, params)
                sym_result["trades"] = trades
                sym_result["trade_pnls"] = [t["pnl_pct"] for t in trades]
                sym_result["stats"] = stats
    else:
        tuned_results = layer2_deep_tune(data_dict, architecture, survivors, cfg)

    if not tuned_results:
        log("No symbols survived Layer 2. Exiting.", "ERROR")
        sys.exit(1)

    # ---- Layer 3 ----
    validated_results, robustness_data = layer3_robustness_gauntlet(
        data_dict, architecture, tuned_results, cfg
    )

    if not validated_results:
        log("No symbols passed robustness gauntlet. Relaxing threshold...", "WARN")
        sorted_by_composite = sorted(
            robustness_data.items(),
            key=lambda x: x[1].get("composite", 0),
            reverse=True
        )
        for sym, rd in sorted_by_composite[:5]:
            if sym in tuned_results:
                validated_results[sym] = tuned_results[sym]
                validated_results[sym]["robustness"] = rd
        if not validated_results:
            log("Still no symbols. Using best tuned results.", "WARN")
            best_sym = max(tuned_results, key=lambda s: tuned_results[s].get("fitness", 0))
            validated_results = {best_sym: tuned_results[best_sym]}
            validated_results[best_sym]["robustness"] = robustness_data.get(best_sym, {})

    # ---- Correlation Filter ----
    final_results = correlation_filter(validated_results, cfg)

    if not final_results:
        log("All symbols filtered out. Using validated results.", "WARN")
        final_results = validated_results

    # ---- Final Backtest ----
    backtest_results = phase_full_backtest(data_dict, architecture, final_results, cfg,
                                           tuned_results=tuned_results)

    # ---- Report Generation ----
    log("=== GENERATING REPORTS ===")
    report_path = generate_html_report(
        backtest_results, architecture, robustness_data, run_info, str(run_output)
    )
    generate_trades_csv(backtest_results.get("all_trades", []), str(run_output))
    generate_summary_csv(backtest_results, str(run_output))
    generate_parameters_json(backtest_results, architecture, str(run_output))

    # ---- AmiBroker Push ----
    if not args.no_amibroker:
        log("=== AMIBROKER INTEGRATION ===")
        sorted_syms = backtest_results.get("sorted_syms", [])
        afl_str = generate_apex_afl(sorted_syms, backtest_results, architecture)
        push_to_amibroker(backtest_results, afl_str, str(run_output), cfg)
    else:
        log("AmiBroker push skipped (--no-amibroker)")

    # ---- Summary ----
    log("=== PIPELINE COMPLETE ===")
    pstats = backtest_results.get("portfolio_stats", {})
    log("Final Results:")
    log(f"  Symbols: {len(backtest_results.get('sorted_syms', []))}")
    log(f"  Trades:  {pstats.get('trades', 0)}")
    log(f"  PF:      {pstats.get('pf', 0):.2f}")
    log(f"  Win%:    {pstats.get('wr_pct', 0):.1f}%")
    log(f"  Return:  {pstats.get('total_return_pct', 0):.1f}%")
    log(f"  MaxDD:   {pstats.get('max_dd_pct', 0):.1f}%")
    log(f"  Sharpe:  {pstats.get('sharpe', 0):.2f}")
    hstats = backtest_results.get("holdout_universe_stats", {})
    if hstats:
        log("  --- TRUE HOLDOUT (never seen by optimizer) ---")
        log(f"  Holdout Trades:  {hstats.get('trades', 0)}")
        log(f"  Holdout PF:      {hstats.get('pf', 0):.2f}")
        log(f"  Holdout Win%:    {hstats.get('wr_pct', 0):.1f}%")
        log(f"  Holdout Return:  {hstats.get('total_return_pct', 0):.1f}%")
        log(f"  Holdout Sharpe:  {hstats.get('sharpe', 0):.2f}")
    log(f"  Report:  {report_path}")

    try:
        abs_report = str(Path(report_path).resolve()).replace("\\", "/")
        webbrowser.open(f"file:///{abs_report}")
        log(f"Report opened in browser: file:///{abs_report}")
    except Exception as e:
        log(f"Could not open browser: {e}", "WARN")
        log(f"Open manually: {Path(report_path).resolve()}")

    return backtest_results


if __name__ == "__main__":
    main()
