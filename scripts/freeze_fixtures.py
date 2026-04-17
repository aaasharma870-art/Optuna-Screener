"""
Freeze SPY/QQQ parquet fixtures for testing.

Uses synthetic data with reproducible random walk so tests are
deterministic and don't require a Polygon API key.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def make_ohlcv(n_bars: int, start_close: float, freq: str, start_dt: datetime,
               rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame with random walk."""
    drift = 0.001   # 0.1 % per bar
    sigma = 0.005   # 0.5 % stdev

    returns = rng.normal(drift, sigma, size=n_bars)
    closes = start_close * np.cumprod(1.0 + returns)

    # Build OHLCV
    highs = closes * (1.0 + rng.uniform(0.001, 0.008, n_bars))
    lows = closes * (1.0 - rng.uniform(0.001, 0.008, n_bars))
    opens = lows + (highs - lows) * rng.uniform(0.2, 0.8, n_bars)
    volumes = rng.integers(500_000, 5_000_000, size=n_bars).astype(float)

    if freq == "1H":
        # Business-day hours: 6.5 bars per day, skip weekends
        timestamps = []
        dt = start_dt
        bar_count = 0
        while bar_count < n_bars:
            if dt.weekday() < 5:  # Mon-Fri
                for hour_offset in range(10, 17):  # 10:00-16:00
                    if bar_count >= n_bars:
                        break
                    timestamps.append(dt.replace(hour=hour_offset, minute=30, second=0))
                    bar_count += 1
            dt += timedelta(days=1)
    else:
        # Daily
        timestamps = []
        dt = start_dt
        bar_count = 0
        while bar_count < n_bars:
            if dt.weekday() < 5:
                timestamps.append(dt.replace(hour=16, minute=0, second=0))
                bar_count += 1
            dt += timedelta(days=1)

    timestamps = timestamps[:n_bars]

    df = pd.DataFrame({
        "datetime": pd.to_datetime(timestamps),
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(closes, 2),
        "volume": volumes.astype(int),
    })
    return df


def main():
    rng = np.random.default_rng(42)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        ("SPY", 500.0, 180, "1H", datetime(2025, 9, 1)),
        ("QQQ", 400.0, 180, "1H", datetime(2025, 9, 1)),
        ("SPY", 500.0, 400, "daily", datetime(2024, 4, 1)),
        ("QQQ", 400.0, 400, "daily", datetime(2024, 4, 1)),
    ]

    for symbol, start_price, n_bars, freq, start_dt in configs:
        df = make_ohlcv(n_bars, start_price, freq, start_dt, rng)
        fname = f"{symbol}_{freq}.parquet"
        out = FIXTURES_DIR / fname
        df.to_parquet(out, index=False)
        print(f"Wrote {out}  ({len(df)} bars)")


if __name__ == "__main__":
    main()
