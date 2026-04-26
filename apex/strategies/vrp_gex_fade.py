"""Strategy 1: VRP + GEX Fade.

Structural primitive: Real options-derived gamma walls (Call Wall / Put Wall)
combined with VRP percentile filter. Trades mean-reversion in suppressed
volatility regimes when price is near a gamma wall.

Spec: docs/superpowers/specs/2026-04-26-institutional-ensemble-design.md sec 5.1
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from apex.indicators.basics import compute_atr, compute_rsi
from apex.indicators.vpin import compute_vpin
from apex.strategies import register_strategy
from apex.strategies.base import StrategyBase


@register_strategy
class VRPGEXFadeStrategy(StrategyBase):
    name = "vrp_gex_fade"
    data_requirements = ["exec_df_1H", "regime_state"]

    # Strategy 1 also needs gamma walls (call_wall, put_wall columns) on
    # exec_df_1H. The ensemble pre-merges these via apex.data.dealer_levels.

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        defaults = {
            "vrp_pct_threshold": 70,
            "gamma_wall_proximity_atr": 0.5,
            "rsi2_oversold": 15,
            "rsi2_overbought": 85,
            "vpin_pct_max": 50,
            "stop_atr_mult": 1.0,
            "max_hold_bars": 21,
        }
        if params:
            defaults.update(params)
        self.params = defaults

    def compute_signals(self, data: Dict[str, Any]) -> pd.DataFrame:
        df = data["exec_df_1H"]
        regime = data.get("regime_state", pd.Series(["R4"] * len(df)))

        n = len(df)
        entry_long = np.zeros(n, dtype=bool)
        entry_short = np.zeros(n, dtype=bool)
        exit_long = np.zeros(n, dtype=bool)
        exit_short = np.zeros(n, dtype=bool)

        # Pre-compute features
        atr = compute_atr(df, period=14).values
        rsi2 = compute_rsi(df["close"], period=2).values
        vpin_df = compute_vpin(df)
        vpin_pct = vpin_df["vpin_pct"].values

        vrp_pct = df.get("vrp_pct")
        vix = df.get("vix")
        vxv = df.get("vxv")
        call_wall = df.get("call_wall")
        put_wall = df.get("put_wall")

        if any(s is None for s in (vrp_pct, vix, vxv, call_wall, put_wall)):
            return pd.DataFrame({
                "entry_long": entry_long, "entry_short": entry_short,
                "exit_long": exit_long, "exit_short": exit_short,
            })

        for i in range(n):
            if regime.iloc[i] == "R4":
                continue
            if pd.isna(vrp_pct.iloc[i]) or pd.isna(vix.iloc[i]) or pd.isna(vxv.iloc[i]):
                continue
            ts_ratio = vix.iloc[i] / vxv.iloc[i] if vxv.iloc[i] > 0 else float("inf")

            # Filters: suppressed regime + contango + low vol
            if vrp_pct.iloc[i] < self.params["vrp_pct_threshold"]:
                continue
            if ts_ratio >= 0.95:
                continue
            if vix.iloc[i] >= 25:
                continue

            # VPIN gate (low VPIN = noise, no informed flow)
            if pd.isna(vpin_pct[i]) or vpin_pct[i] >= self.params["vpin_pct_max"]:
                continue

            atr_i = atr[i] if i < len(atr) and not np.isnan(atr[i]) else 1.0
            proximity = self.params["gamma_wall_proximity_atr"] * atr_i

            close_i = df["close"].iloc[i]
            put_wall_i = put_wall.iloc[i]
            call_wall_i = call_wall.iloc[i]

            if pd.isna(put_wall_i) or pd.isna(call_wall_i):
                continue

            # LONG: near put wall + RSI2 oversold
            if (abs(close_i - put_wall_i) <= proximity
                    and not pd.isna(rsi2[i])
                    and rsi2[i] < self.params["rsi2_oversold"]):
                entry_long[i] = True
                continue

            # SHORT: near call wall + RSI2 overbought
            if (abs(close_i - call_wall_i) <= proximity
                    and not pd.isna(rsi2[i])
                    and rsi2[i] > self.params["rsi2_overbought"]):
                entry_short[i] = True

        return pd.DataFrame({
            "entry_long": entry_long, "entry_short": entry_short,
            "exit_long": exit_long, "exit_short": exit_short,
        })

    def compute_position_size(self, data: Dict[str, Any],
                              signals: pd.DataFrame) -> pd.Series:
        # Per-bar position: +1 when long, -1 when short, decay over max_hold_bars
        n = len(signals)
        pos = np.zeros(n, dtype=float)
        bars_in_pos = 0
        side = 0
        max_hold = self.params["max_hold_bars"]

        for i in range(n):
            if signals["entry_long"].iloc[i] and side == 0:
                side = 1
                bars_in_pos = 0
            elif signals["entry_short"].iloc[i] and side == 0:
                side = -1
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
            "vrp_pct_threshold":         (60,   90,   "int"),
            "gamma_wall_proximity_atr":  (0.2,  1.0,  "float"),
            "rsi2_oversold":             (5,    25,   "int"),
            "rsi2_overbought":           (75,   95,   "int"),
            "vpin_pct_max":              (40,   60,   "int"),
            "stop_atr_mult":             (0.6,  1.6,  "float"),
            "max_hold_bars":             (7,    35,   "int"),
        }
