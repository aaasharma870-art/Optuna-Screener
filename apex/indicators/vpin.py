"""Volume-Bucketed VPIN per Easley / López de Prado / O'Hara."""

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_vpin(
    df: pd.DataFrame,
    buckets_per_day: int = 50,
    window_buckets: int = 50,
    sigma_window: int = 60,
) -> pd.DataFrame:
    """Bulk Volume Classification VPIN.

    Algorithm:
    1. bar returns: ret = close - open
    2. rolling stdev of bar returns: sigma_R = ret.rolling(sigma_window).std()
    3. BVC: buy_vol_i = volume_i * norm.cdf(ret_i / sigma_R_i)
            sell_vol_i = volume_i - buy_vol_i
    4. Aggregate into equal-volume buckets:
       bucket_size = total_daily_volume / buckets_per_day
       Walk forward summing volume; when cumulative >= bucket_size, close bucket.
       Record bucket_buy_vol and bucket_sell_vol per bucket.
    5. VPIN_bucket = |bucket_buy - bucket_sell| / (bucket_buy + bucket_sell)
    6. Rolling mean of VPIN over window_buckets -> smoothed VPIN
    7. Back-fill to bar-level by forward-filling most recent completed bucket's VPIN
    8. vpin_pct = vpin.rolling(252*buckets_per_day).rank(pct=True) * 100

    Returns DataFrame with columns: vpin, vpin_pct
    """
    result = df.copy()
    n = len(df)

    # Step 1-2: bar returns and rolling sigma
    ret = (df["close"] - df["open"]).values.astype(np.float64)
    ret_series = pd.Series(ret)
    sigma_r = ret_series.rolling(sigma_window).std().values

    # Step 3: BVC
    volume = df["volume"].values.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma_r > 0, ret / sigma_r, 0.0)
    buy_vol = volume * norm.cdf(z)
    sell_vol = volume - buy_vol

    # Step 4: Determine bucket size from total volume
    total_volume = volume.sum()
    # Estimate number of trading days from data length or use total_volume / buckets_per_day
    # Use median daily volume approach: group by date if timestamp available,
    # otherwise use total / estimated days
    if "timestamp" in df.columns:
        dates = pd.to_datetime(df["timestamp"]).dt.date
    elif "datetime" in df.columns:
        dates = pd.to_datetime(df["datetime"]).dt.date
    else:
        # Fallback: estimate from index
        dates = pd.Series(np.arange(n) // 24)  # rough day estimate for hourly

    unique_dates = dates.nunique()
    if unique_dates < 1:
        unique_dates = 1
    avg_daily_vol = total_volume / unique_dates
    bucket_size = avg_daily_vol / buckets_per_day

    if bucket_size <= 0:
        result["vpin"] = np.nan
        result["vpin_pct"] = np.nan
        return result

    # Walk forward to build volume buckets
    bucket_buy_vols = []
    bucket_sell_vols = []
    bucket_last_bar_idx = []  # bar index when bucket closed

    cum_buy = 0.0
    cum_sell = 0.0
    cum_vol = 0.0

    for i in range(n):
        cum_buy += buy_vol[i]
        cum_sell += sell_vol[i]
        cum_vol += volume[i]

        while cum_vol >= bucket_size and bucket_size > 0:
            # Close this bucket
            # Fraction of current bar that fills the bucket
            overflow = cum_vol - bucket_size
            frac = 1.0 - (overflow / volume[i]) if volume[i] > 0 else 1.0
            frac = np.clip(frac, 0.0, 1.0)

            # For simplicity and vectorization, assign full bars to buckets
            bucket_buy_vols.append(cum_buy - overflow * (buy_vol[i] / volume[i] if volume[i] > 0 else 0.5))
            bucket_sell_vols.append(cum_sell - overflow * (sell_vol[i] / volume[i] if volume[i] > 0 else 0.5))
            bucket_last_bar_idx.append(i)

            # Carry over the overflow to next bucket
            cum_buy = overflow * (buy_vol[i] / volume[i] if volume[i] > 0 else 0.5)
            cum_sell = overflow * (sell_vol[i] / volume[i] if volume[i] > 0 else 0.5)
            cum_vol = overflow

    if len(bucket_buy_vols) == 0:
        result["vpin"] = np.nan
        result["vpin_pct"] = np.nan
        return result

    # Step 5: VPIN per bucket
    bucket_buy = np.array(bucket_buy_vols)
    bucket_sell = np.array(bucket_sell_vols)
    bucket_total = bucket_buy + bucket_sell
    with np.errstate(divide="ignore", invalid="ignore"):
        bucket_vpin = np.where(
            bucket_total > 0,
            np.abs(bucket_buy - bucket_sell) / bucket_total,
            np.nan,
        )

    # Step 6: Rolling mean over window_buckets
    bucket_vpin_series = pd.Series(bucket_vpin)
    smoothed_vpin = bucket_vpin_series.rolling(window_buckets, min_periods=1).mean().values

    # Step 7: Back-fill to bar-level
    bar_vpin = np.full(n, np.nan, dtype=np.float64)
    bucket_indices = np.array(bucket_last_bar_idx)

    for b_idx in range(len(bucket_indices)):
        bar_i = bucket_indices[b_idx]
        bar_vpin[bar_i] = smoothed_vpin[b_idx]

    # Forward-fill
    bar_vpin_series = pd.Series(bar_vpin)
    bar_vpin_series = bar_vpin_series.ffill()

    result["vpin"] = bar_vpin_series.values

    # Step 8: percentile rank
    rank_window = 252 * buckets_per_day
    result["vpin_pct"] = (
        result["vpin"].rolling(rank_window, min_periods=1).rank(pct=True) * 100
    )

    return result
