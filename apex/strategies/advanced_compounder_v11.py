"""Advanced Compounder v11.3 Pine port.

Literal research port of the TradingView idea:
macro Supertrend is the regime/stop line, micro Supertrend flips trigger adds,
and macro flips close exposure. Pyramiding is represented as fractional exposure
up to 1.0 so the strategy remains compatible with StrategyBase.
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from apex.indicators.basics import compute_supertrend
from apex.strategies import register_strategy
from apex.strategies.base import StrategyBase


@register_strategy
class AdvancedCompounderV11Strategy(StrategyBase):
    name = "advanced_compounder_v11"
    data_requirements = ["exec_df_1H"]

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        defaults = {
            "st_slow_factor": 3.0,
            "st_slow_period": 10,
            "st_fast_factor": 1.5,
            "st_fast_period": 10,
            "use_trail": True,
            "max_pyramids": 5,
            "unit_size": 0.20,
            "allow_short": True,
        }
        if params:
            defaults.update(params)
        self.params = defaults

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = df.copy()
        macro, dir_macro = compute_supertrend(
            feat,
            period=int(self.params["st_slow_period"]),
            factor=float(self.params["st_slow_factor"]),
        )
        micro, dir_micro = compute_supertrend(
            feat,
            period=int(self.params["st_fast_period"]),
            factor=float(self.params["st_fast_factor"]),
        )
        feat["st_macro"] = macro
        feat["dir_macro"] = dir_macro
        feat["st_micro"] = micro
        feat["dir_micro"] = dir_micro
        return feat

    def compute_signals(self, data: Dict[str, Any]) -> pd.DataFrame:
        df = data["exec_df_1H"]
        n = len(df)
        entry_long = np.zeros(n, dtype=bool)
        entry_short = np.zeros(n, dtype=bool)
        exit_long = np.zeros(n, dtype=bool)
        exit_short = np.zeros(n, dtype=bool)
        if n == 0:
            return pd.DataFrame({
                "entry_long": entry_long, "entry_short": entry_short,
                "exit_long": exit_long, "exit_short": exit_short,
            })

        feat = self._features(df)
        close = feat["close"]

        for i in range(1, n):
            st_micro = feat["st_micro"].iloc[i]
            st_micro_prev = feat["st_micro"].iloc[i - 1]
            if pd.isna(st_micro) or pd.isna(st_micro_prev):
                continue

            macro_dir = feat["dir_macro"].iloc[i]
            macro_prev = feat["dir_macro"].iloc[i - 1]
            bullish_macro = macro_dir == -1
            bearish_macro = macro_dir == 1

            bull_trigger = close.iloc[i - 1] <= st_micro_prev and close.iloc[i] > st_micro
            bear_trigger = close.iloc[i - 1] >= st_micro_prev and close.iloc[i] < st_micro

            if bullish_macro and bull_trigger:
                entry_long[i] = True
            if self.params["allow_short"] and bearish_macro and bear_trigger:
                entry_short[i] = True

            if pd.notna(macro_prev) and macro_dir != macro_prev:
                if macro_dir == 1:
                    exit_long[i] = True
                elif macro_dir == -1:
                    exit_short[i] = True

            macro_line = feat["st_macro"].iloc[i]
            if self.params["use_trail"] and pd.notna(macro_line):
                if close.iloc[i] < macro_line:
                    exit_long[i] = True
                if close.iloc[i] > macro_line:
                    exit_short[i] = True

        return pd.DataFrame({
            "entry_long": entry_long, "entry_short": entry_short,
            "exit_long": exit_long, "exit_short": exit_short,
        })

    def compute_position_size(self, data: Dict[str, Any],
                              signals: pd.DataFrame) -> pd.Series:
        n = len(signals)
        pos = np.zeros(n, dtype=float)
        side = 0
        units = 0
        max_pyramids = int(self.params["max_pyramids"])
        unit = float(self.params["unit_size"])

        for i in range(n):
            if side == 1 and signals["exit_long"].iloc[i]:
                side = 0
                units = 0
            elif side == -1 and signals["exit_short"].iloc[i]:
                side = 0
                units = 0

            if signals["entry_long"].iloc[i]:
                if side < 0:
                    side = 0
                    units = 0
                side = 1
                units = min(max_pyramids, units + 1)
            elif signals["entry_short"].iloc[i]:
                if side > 0:
                    side = 0
                    units = 0
                side = -1
                units = min(max_pyramids, units + 1)

            if side != 0 and units > 0:
                pos[i] = float(side) * min(1.0, units * unit)

        return pd.Series(pos)

    def get_tunable_params(self) -> Dict[str, tuple]:
        return {
            "st_slow_factor": (1.0, 6.0, "float"),
            "st_slow_period": (5, 30, "int"),
            "st_fast_factor": (0.5, 4.0, "float"),
            "st_fast_period": (3, 20, "int"),
            "use_trail": (None, None, "categorical", [True, False]),
            "max_pyramids": (1, 5, "int"),
            "unit_size": (0.10, 0.50, "float"),
            "allow_short": (None, None, "categorical", [True, False]),
        }
