"""One-shot script: verify Strategy 1 produces some signals on real cached data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ.setdefault("POLYGON_API_KEY", "i2G48KgZOdi2QJdxXIGrkwiUyzDeQg8_")
os.environ.setdefault("FRED_API_KEY", "aedfa3c4c33c527fd4b4cbcf4c25f258")

import pandas as pd
import numpy as np

# Load cached SPY 1H data and synthesize required columns
spy_1h = pd.read_csv("apex_cache/SPY_1H.csv", parse_dates=["datetime"])
spy_1h = spy_1h.tail(2000).reset_index(drop=True)

# Inject realistic VRP/regime/gamma columns
n = len(spy_1h)
np.random.seed(42)
spy_1h["vix"] = 16.0
spy_1h["vxv"] = 18.0  # contango
spy_1h["vrp_pct"] = 80.0  # suppressed regime
spy_1h["call_wall"] = spy_1h["close"].rolling(50).mean() + 5
spy_1h["put_wall"] = spy_1h["close"].rolling(50).mean() - 5
spy_1h["gamma_flip"] = spy_1h["close"].rolling(50).mean()
spy_1h = spy_1h.fillna(method="bfill")

regime_state = pd.Series(["R1"] * n)

from apex.strategies.vrp_gex_fade import VRPGEXFadeStrategy
strat = VRPGEXFadeStrategy()
data = {"exec_df_1H": spy_1h, "regime_state": regime_state, "symbol": "SPY"}
signals = strat.compute_signals(data)
positions = strat.compute_position_size(data, signals)

n_long = int(signals["entry_long"].sum())
n_short = int(signals["entry_short"].sum())
print(f"Strategy 1 verification on {n} bars of cached SPY:")
print(f"  Long entries: {n_long}")
print(f"  Short entries: {n_short}")
print(f"  Position bars: {int((positions != 0).sum())}")
if n_long + n_short == 0:
    print("  WARNING: zero entries -- check filters")
    sys.exit(1)
print("  OK")
