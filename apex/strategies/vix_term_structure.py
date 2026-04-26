"""Strategy 3: VIX Term Structure mean-reversion.

Structural primitive: VIX/VIX3M ratio mean-reversion. Trades the curve, not the
level. Extreme contango -> LONG SPY; extreme backwardation -> SHORT SPY.

Spec: docs/superpowers/specs/2026-04-26-institutional-ensemble-design.md sec 5.3
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from apex.indicators.basics import compute_rsi
from apex.strategies import register_strategy
from apex.strategies.base import StrategyBase


@register_strategy
class VIXTermStructureStrategy(StrategyBase):
    name = "vix_term_structure"
    data_requirements = ["exec_df_1H"]

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        defaults = {
            "contango_extreme_threshold": 0.85,
            "backwardation_extreme_threshold": 1.10,
            "neutral_low": 0.95,
            "neutral_high": 1.02,
            "stop_atr_mult": 1.5,
            "max_hold_bars": 10,
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

        vix = df.get("vix")
        vxv = df.get("vxv")
        if vix is None or vxv is None:
            return pd.DataFrame({
                "entry_long": entry_long, "entry_short": entry_short,
                "exit_long": exit_long, "exit_short": exit_short,
            })

        # ts_ratio per bar (NaN where inputs missing)
        ts_ratio = vix / vxv.replace(0, np.nan)
        ts_ratio_rsi = compute_rsi(ts_ratio, period=5)

        contango_thr = self.params["contango_extreme_threshold"]
        backwardation_thr = self.params["backwardation_extreme_threshold"]
        neutral_lo = self.params["neutral_low"]
        neutral_hi = self.params["neutral_high"]

        for i in range(n):
            r = ts_ratio.iloc[i]
            rsi = ts_ratio_rsi.iloc[i]
            if pd.isna(r):
                continue

            # Exit signals when ratio is back inside neutral band
            if neutral_lo <= r <= neutral_hi:
                exit_long[i] = True
                exit_short[i] = True
                continue

            if pd.isna(rsi):
                continue

            # Extreme contango (low ratio) + RSI confirm -> LONG SPY
            if r < contango_thr and rsi < 30:
                entry_long[i] = True
            # Extreme backwardation (high ratio) + RSI confirm -> SHORT SPY
            elif r > backwardation_thr and rsi > 70:
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
        max_hold = self.params["max_hold_bars"]

        for i in range(n):
            # Open new position
            if side == 0:
                if signals["entry_long"].iloc[i]:
                    side = 1
                    bars_in_pos = 0
                elif signals["entry_short"].iloc[i]:
                    side = -1
                    bars_in_pos = 0
            else:
                # Explicit exit signals or max-hold
                if (side == 1 and signals["exit_long"].iloc[i]) or \
                   (side == -1 and signals["exit_short"].iloc[i]):
                    side = 0
                    bars_in_pos = 0

            if side != 0:
                pos[i] = float(side)
                bars_in_pos += 1
                if bars_in_pos >= max_hold:
                    side = 0
                    bars_in_pos = 0

        return pd.Series(pos)

    def get_tunable_params(self) -> Dict[str, tuple]:
        return {
            "contango_extreme_threshold":      (0.80, 0.92, "float"),
            "backwardation_extreme_threshold": (1.05, 1.20, "float"),
            "neutral_low":                     (0.93, 0.97, "float"),
            "neutral_high":                    (1.00, 1.04, "float"),
            "stop_atr_mult":                   (0.8,  2.0,  "float"),
            "max_hold_bars":                   (5,    30,   "int"),
        }
