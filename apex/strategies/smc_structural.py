"""Strategy 5: SMC Structural — FVG + Order Block confluence.

Structural primitive: pure price-structure entries. Trades retests of unfilled
bullish/bearish FVGs and order blocks gated by VIX < 25 and VPIN percentile
< 50 (avoid noisy / informed-flow regimes).

Spec: docs/superpowers/specs/2026-04-26-institutional-ensemble-design.md sec 5.5
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from apex.indicators.fvg import detect_fvgs, unfilled_fvgs_at
from apex.indicators.order_blocks import detect_order_blocks
from apex.indicators.vpin import compute_vpin
from apex.strategies import register_strategy
from apex.strategies.base import StrategyBase


@register_strategy
class SMCStructuralStrategy(StrategyBase):
    name = "smc_structural"
    data_requirements = ["exec_df_1H"]

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        defaults = {
            "vix_filter_max": 25,
            "vpin_pct_max": 50,
            "ob_min_body_ratio": 0.5,
            "max_hold_bars": 16,
            "dynamic_stop": False,
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

        if n < 3:
            return pd.DataFrame({
                "entry_long": entry_long, "entry_short": entry_short,
                "exit_long": exit_long, "exit_short": exit_short,
            })

        # Pre-compute FVGs and order blocks for the whole frame
        fvgs = detect_fvgs(df)
        obs = detect_order_blocks(df, min_body_ratio=self.params["ob_min_body_ratio"])

        # VPIN percentile per bar
        try:
            vpin_df = compute_vpin(df)
            vpin_pct = vpin_df["vpin_pct"].values
        except Exception:
            vpin_pct = np.full(n, np.nan)

        vix = df.get("vix")
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        vix_max = self.params["vix_filter_max"]
        vpin_max = self.params["vpin_pct_max"]

        for i in range(n):
            # Filter: VIX < threshold (skip if missing)
            if vix is not None:
                v = vix.iloc[i]
                if pd.isna(v) or v >= vix_max:
                    continue

            # Filter: VPIN percentile < threshold (skip if missing/high)
            vp = vpin_pct[i] if i < len(vpin_pct) else np.nan
            if pd.isna(vp) or vp >= vpin_max:
                # Even if filters fail, check FVG fill -> exit
                pass
            else:
                # Check entry conditions
                close_i = closes[i]
                low_i = lows[i]
                high_i = highs[i]

                # Unfilled bullish FVG retest: price within FVG zone
                bullish_fvgs = [
                    f for f in unfilled_fvgs_at(fvgs, i)
                    if f["direction"] == "bullish"
                ]
                if any(f["low"] <= close_i <= f["high"] or
                        f["low"] <= low_i <= f["high"]
                        for f in bullish_fvgs):
                    entry_long[i] = True

                # Bullish OB retest: price re-entering OB zone after formation
                bullish_obs = [
                    ob for ob in obs
                    if ob["direction"] == "bullish"
                    and ob["end_idx"] < i
                    and (ob["mitigated_at_idx"] is None
                         or ob["mitigated_at_idx"] >= i)
                ]
                if not entry_long[i] and any(
                        ob["low"] <= low_i <= ob["high"]
                        or ob["low"] <= close_i <= ob["high"]
                        for ob in bullish_obs):
                    entry_long[i] = True

                # Bearish FVG retest
                bearish_fvgs = [
                    f for f in unfilled_fvgs_at(fvgs, i)
                    if f["direction"] == "bearish"
                ]
                if any(f["low"] <= close_i <= f["high"] or
                        f["low"] <= high_i <= f["high"]
                        for f in bearish_fvgs):
                    entry_short[i] = True

                # Bearish OB retest
                bearish_obs = [
                    ob for ob in obs
                    if ob["direction"] == "bearish"
                    and ob["end_idx"] < i
                    and (ob["mitigated_at_idx"] is None
                         or ob["mitigated_at_idx"] >= i)
                ]
                if not entry_short[i] and any(
                        ob["low"] <= high_i <= ob["high"]
                        or ob["low"] <= close_i <= ob["high"]
                        for ob in bearish_obs):
                    entry_short[i] = True

            # Exit logic — independent of filters: FVG fully filled or
            # opposite-direction FVG forms.
            close_i = closes[i]
            for f in fvgs:
                if f["filled_at_idx"] == i:
                    if f["direction"] == "bullish":
                        exit_long[i] = True
                    else:
                        exit_short[i] = True
                # Opposite-direction FVG forming at this bar
                if f["end_idx"] == i:
                    if f["direction"] == "bearish":
                        exit_long[i] = True
                    elif f["direction"] == "bullish":
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
        bars_in_pos = 0
        max_hold = self.params["max_hold_bars"]

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
                if bars_in_pos >= max_hold:
                    side = 0
                    bars_in_pos = 0

        return pd.Series(pos)

    def get_tunable_params(self) -> Dict[str, tuple]:
        return {
            "vix_filter_max":     (15,   30,   "int"),
            "vpin_pct_max":       (30,   60,   "int"),
            "ob_min_body_ratio":  (0.3,  0.8,  "float"),
            "max_hold_bars":      (8,    24,   "int"),
            "dynamic_stop":       (None, None, "categorical", [True, False]),
        }
