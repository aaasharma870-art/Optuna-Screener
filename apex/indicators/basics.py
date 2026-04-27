"""Technical indicator library."""

import math

import numpy as np
import pandas as pd


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


def compute_supertrend(df, period=10, factor=3.0):
    """TradingView-style Supertrend.

    Returns (supertrend_line, direction), where direction matches Pine's
    convention used by ta.supertrend: -1 is bullish/uptrend, +1 is bearish.
    """
    high = df["high"].astype(float).reset_index(drop=True)
    low = df["low"].astype(float).reset_index(drop=True)
    close = df["close"].astype(float).reset_index(drop=True)

    atr = compute_atr(
        pd.DataFrame({"high": high, "low": low, "close": close}),
        period=period,
    ).reset_index(drop=True)
    hl2 = (high + low) / 2.0
    upper_basic = hl2 + factor * atr
    lower_basic = hl2 - factor * atr

    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    st = np.full(n, np.nan)
    direction = np.full(n, np.nan)

    for i in range(n):
        if pd.isna(atr.iloc[i]):
            continue
        if i == 0 or pd.isna(st[i - 1]):
            upper[i] = upper_basic.iloc[i]
            lower[i] = lower_basic.iloc[i]
            if close.iloc[i] >= hl2.iloc[i]:
                direction[i] = -1.0
                st[i] = lower[i]
            else:
                direction[i] = 1.0
                st[i] = upper[i]
            continue

        prev_upper = upper[i - 1]
        prev_lower = lower[i - 1]
        prev_close = close.iloc[i - 1]

        upper[i] = (
            upper_basic.iloc[i]
            if upper_basic.iloc[i] < prev_upper or prev_close > prev_upper
            else prev_upper
        )
        lower[i] = (
            lower_basic.iloc[i]
            if lower_basic.iloc[i] > prev_lower or prev_close < prev_lower
            else prev_lower
        )

        if st[i - 1] == prev_upper:
            if close.iloc[i] <= upper[i]:
                st[i] = upper[i]
                direction[i] = 1.0
            else:
                st[i] = lower[i]
                direction[i] = -1.0
        else:
            if close.iloc[i] >= lower[i]:
                st[i] = lower[i]
                direction[i] = -1.0
            else:
                st[i] = upper[i]
                direction[i] = 1.0

    return (
        pd.Series(st, index=df.index, name="supertrend"),
        pd.Series(direction, index=df.index, name="supertrend_direction"),
    )


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
