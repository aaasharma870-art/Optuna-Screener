"""Strategy 4: Volatility Skew Arbitrage.

Structural primitive: 25-delta put IV / 25-delta call IV ratio extremes.
Trades asymmetric fear pricing in the options surface (mean-reversion).

Spec: docs/superpowers/specs/2026-04-26-institutional-ensemble-design.md sec 5.4
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from apex.strategies import register_strategy
from apex.strategies.base import StrategyBase


@register_strategy
class VolSkewArbStrategy(StrategyBase):
    name = "vol_skew_arb"
    data_requirements = ["exec_df_1H"]

    # Strategy 4 expects a `skew_ratio` column on exec_df_1H pre-merged
    # by the data layer (apex.data.vol_skew.compute_skew_ratio aggregated daily).

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        defaults = {
            "put_skew_extreme": 1.30,
            "call_skew_extreme": 0.95,
            "normal_low": 1.05,
            "normal_high": 1.20,
            "dte_target": 30,
            "stop_atr_mult": 1.0,
            "max_hold_days": 5,
        }
        if params:
            defaults.update(params)
        self.params = defaults

    def compute_signals(self, data: Dict[str, Any]) -> pd.DataFrame:
        df = data["exec_df_1H"]
        n = len(df)
        entry_long = np.zeros(n, dtype=bool)
        entry_short = np.zeros(n, dtype=bool)
        exit_long = np.zeros(n, dtype=bool)
        exit_short = np.zeros(n, dtype=bool)

        skew = df.get("skew_ratio")
        if skew is None:
            return pd.DataFrame({
                "entry_long": entry_long, "entry_short": entry_short,
                "exit_long": exit_long, "exit_short": exit_short,
            })

        put_extreme = self.params["put_skew_extreme"]
        call_extreme = self.params["call_skew_extreme"]
        normal_lo = self.params["normal_low"]
        normal_hi = self.params["normal_high"]

        for i in range(n):
            r = skew.iloc[i]
            if pd.isna(r):
                continue

            # Exit when skew reverts to normal band
            if normal_lo <= r <= normal_hi:
                exit_long[i] = True
                exit_short[i] = True
                continue

            # Extreme put fear -> mean-rev LONG
            if r > put_extreme:
                entry_long[i] = True
            # Extreme call greed -> mean-rev SHORT
            elif r < call_extreme:
                entry_short[i] = True

        return pd.DataFrame({
            "entry_long": entry_long, "entry_short": entry_short,
            "exit_long": exit_long, "exit_short": exit_short,
        })

    def compute_position_size(self, data: Dict[str, Any],
                              signals: pd.DataFrame) -> pd.Series:
        n = len(signals)
        pos = np.zeros(n, dtype=float)
        side = 0
        bars_in_pos = 0
        # max_hold_days at 1H bars: assume ~7 bars per trading day (RTH)
        # Use 7 bars/day as a conservative cap; param is in days.
        max_hold_bars = int(self.params["max_hold_days"] * 7)

        for i in range(n):
            if side == 0:
                if signals["entry_long"].iloc[i]:
                    side = 1
                    bars_in_pos = 0
                elif signals["entry_short"].iloc[i]:
                    side = -1
                    bars_in_pos = 0
            else:
                if (side == 1 and signals["exit_long"].iloc[i]) or \
                   (side == -1 and signals["exit_short"].iloc[i]):
                    side = 0
                    bars_in_pos = 0

            if side != 0:
                pos[i] = float(side)
                bars_in_pos += 1
                if bars_in_pos >= max_hold_bars:
                    side = 0
                    bars_in_pos = 0

        return pd.Series(pos)

    def get_tunable_params(self) -> Dict[str, tuple]:
        return {
            "put_skew_extreme":  (1.20, 1.50, "float"),
            "call_skew_extreme": (0.85, 1.00, "float"),
            "normal_low":        (1.00, 1.10, "float"),
            "normal_high":       (1.15, 1.30, "float"),
            "dte_target":        (20,   45,   "int"),
            "stop_atr_mult":     (0.7,  1.5,  "float"),
            "max_hold_days":     (2,    10,   "int"),
        }
