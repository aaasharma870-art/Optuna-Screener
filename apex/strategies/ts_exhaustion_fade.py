"""Term-structure gated VWAP exhaustion fade.

Research hypothesis from the April 2026 strategy docs:
VIX/VIX3M term structure does the regime heavy lifting. In contango/calm
conditions, intraday price moves beyond VWAP sigma bands should mean-revert.

This is intentionally not the production VIX term-structure preset. That
strategy trades curve extremes directly. This one trades price exhaustion only
when the curve says dealer/vol conditions are likely damped.
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from apex.indicators.basics import compute_atr, compute_rsi
from apex.indicators.vwap_bands import compute_vwap_bands
from apex.strategies import register_strategy
from apex.strategies.base import StrategyBase


@register_strategy
class TermStructureExhaustionFadeStrategy(StrategyBase):
    name = "ts_exhaustion_fade"
    data_requirements = ["exec_df_1H"]

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        defaults = {
            "contango_max": 0.95,
            "vix_max": 25.0,
            "vrp_calm_low": 30.0,
            "vrp_calm_high": 70.0,
            "use_vrp_calm_filter": True,
            "enable_long": True,
            "enable_short": True,
            "deviation_sigma": 2.0,
            "vwap_slope_atr_max": 0.10,
            "rsi2_oversold": 10,
            "rsi2_overbought": 90,
            "exit_long_rsi": 60,
            "exit_short_rsi": 40,
            "min_session_bars": 5,
            "max_entry_hour": 15.5,
            "exit_at_session_end": True,
            "stop_atr_mult": 0.8,
            "max_hold_bars": 8,
        }
        if params:
            defaults.update(params)
        self.params = defaults

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        timestamp_col = "datetime" if "datetime" in df.columns else "timestamp"
        feat = compute_vwap_bands(df, timestamp_col=timestamp_col)
        feat["atr"] = compute_atr(feat, period=14)
        feat["rsi2"] = compute_rsi(feat["close"], period=2)
        return feat

    def compute_signals(self, data: Dict[str, Any]) -> pd.DataFrame:
        df = data["exec_df_1H"]
        n = len(df)
        entry_long = np.zeros(n, dtype=bool)
        entry_short = np.zeros(n, dtype=bool)
        exit_long = np.zeros(n, dtype=bool)
        exit_short = np.zeros(n, dtype=bool)

        vix = df.get("vix")
        vxv = df.get("vxv")
        if vix is None or vxv is None or n == 0:
            return pd.DataFrame({
                "entry_long": entry_long, "entry_short": entry_short,
                "exit_long": exit_long, "exit_short": exit_short,
            })

        feat = self._features(df)
        sigma = feat["vwap_1s_upper"] - feat["vwap"]
        dev = float(self.params["deviation_sigma"]) * sigma
        ts_ratio = vix / vxv.replace(0, np.nan)
        vrp_pct = df.get("vrp_pct")
        timestamp_col = "datetime" if "datetime" in feat.columns else "timestamp"
        dt = pd.to_datetime(feat[timestamp_col])
        session_dates = dt.dt.date
        session_bar = feat.groupby(session_dates).cumcount()
        hour_float = dt.dt.hour + dt.dt.minute / 60.0

        for i in range(n):
            if pd.isna(ts_ratio.iloc[i]) or pd.isna(feat["rsi2"].iloc[i]):
                continue
            if session_bar.iloc[i] < int(self.params["min_session_bars"]):
                continue
            if hour_float.iloc[i] > float(self.params["max_entry_hour"]):
                continue
            if ts_ratio.iloc[i] >= self.params["contango_max"]:
                continue
            if pd.isna(vix.iloc[i]) or vix.iloc[i] >= self.params["vix_max"]:
                continue
            if abs(feat["vwap_slope_atr"].iloc[i]) > self.params["vwap_slope_atr_max"]:
                continue

            if self.params["use_vrp_calm_filter"] and vrp_pct is not None:
                vrp_i = vrp_pct.iloc[i]
                if pd.isna(vrp_i):
                    continue
                if not (self.params["vrp_calm_low"] <= vrp_i <= self.params["vrp_calm_high"]):
                    continue

            close_i = feat["close"].iloc[i]
            vwap_i = feat["vwap"].iloc[i]
            dev_i = dev.iloc[i]
            if pd.isna(dev_i) or dev_i <= 0:
                continue

            if (self.params["enable_long"]
                    and close_i < vwap_i - dev_i
                    and feat["rsi2"].iloc[i] <= self.params["rsi2_oversold"]):
                entry_long[i] = True
            elif (self.params["enable_short"]
                  and close_i > vwap_i + dev_i
                  and feat["rsi2"].iloc[i] >= self.params["rsi2_overbought"]):
                entry_short[i] = True

            # Mean-reversion target: session VWAP. RSI unwind is a backup exit.
            if close_i >= vwap_i or feat["rsi2"].iloc[i] >= self.params["exit_long_rsi"]:
                exit_long[i] = True
            if close_i <= vwap_i or feat["rsi2"].iloc[i] <= self.params["exit_short_rsi"]:
                exit_short[i] = True

        return pd.DataFrame({
            "entry_long": entry_long, "entry_short": entry_short,
            "exit_long": exit_long, "exit_short": exit_short,
        })

    def compute_position_size(self, data: Dict[str, Any],
                              signals: pd.DataFrame) -> pd.Series:
        df = data["exec_df_1H"]
        atr = compute_atr(df, period=14)
        n = len(signals)
        pos = np.zeros(n, dtype=float)
        side = 0
        bars_in_pos = 0
        entry_price = 0.0
        entry_atr = 0.0
        timestamp_col = "datetime" if "datetime" in df.columns else "timestamp"
        if timestamp_col in df.columns:
            dates = pd.to_datetime(df[timestamp_col]).dt.date
            is_last_bar_of_day = dates.ne(dates.shift(-1))
        else:
            is_last_bar_of_day = pd.Series(False, index=df.index)
        max_hold = int(self.params["max_hold_bars"])
        stop_mult = float(self.params["stop_atr_mult"])

        for i in range(n):
            close_i = float(df["close"].iloc[i])
            atr_i = atr.iloc[i]
            if pd.isna(atr_i) or atr_i <= 0:
                atr_i = entry_atr if entry_atr > 0 else 0.0

            if side == 0:
                if signals["entry_long"].iloc[i]:
                    side = 1
                    bars_in_pos = 0
                    entry_price = close_i
                    entry_atr = float(atr_i)
                elif signals["entry_short"].iloc[i]:
                    side = -1
                    bars_in_pos = 0
                    entry_price = close_i
                    entry_atr = float(atr_i)
            else:
                stop_hit = False
                if entry_atr > 0:
                    if side == 1 and close_i <= entry_price - stop_mult * entry_atr:
                        stop_hit = True
                    elif side == -1 and close_i >= entry_price + stop_mult * entry_atr:
                        stop_hit = True

                exit_hit = (
                    (side == 1 and signals["exit_long"].iloc[i]) or
                    (side == -1 and signals["exit_short"].iloc[i]) or
                    stop_hit or
                    bars_in_pos >= max_hold or
                    (self.params["exit_at_session_end"] and is_last_bar_of_day.iloc[i])
                )
                if exit_hit:
                    side = 0
                    bars_in_pos = 0
                    entry_price = 0.0
                    entry_atr = 0.0

            if side != 0:
                pos[i] = float(side)
                bars_in_pos += 1

        return pd.Series(pos)

    def get_tunable_params(self) -> Dict[str, tuple]:
        return {
            "contango_max":            (0.88, 1.02, "float"),
            "vix_max":                (16.0, 35.0, "float"),
            "vrp_calm_low":           (10.0, 50.0, "float"),
            "vrp_calm_high":          (50.0, 90.0, "float"),
            "use_vrp_calm_filter":    (None, None, "categorical", [True, False]),
            "enable_long":            (None, None, "categorical", [True, False]),
            "enable_short":           (None, None, "categorical", [True, False]),
            "deviation_sigma":        (0.6, 3.2, "float"),
            "vwap_slope_atr_max":     (0.05, 1.50, "float"),
            "rsi2_oversold":          (5, 20, "int"),
            "rsi2_overbought":        (80, 95, "int"),
            "exit_long_rsi":          (45, 80, "int"),
            "exit_short_rsi":         (20, 55, "int"),
            "min_session_bars":       (0, 12, "int"),
            "max_entry_hour":         (12.0, 15.75, "float"),
            "exit_at_session_end":    (None, None, "categorical", [True, False]),
            "stop_atr_mult":          (0.3, 2.5, "float"),
            "max_hold_bars":          (2, 40, "int"),
        }
