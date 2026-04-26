"""Strategy 6: Cross-Asset Vol Regime Overlay.

NOT a standalone strategy. Multiplies position sizes of strategies 1-5 based
on macro vol regime (VIX/MOVE/OVX percentiles).

Spec: docs/superpowers/specs/2026-04-26-institutional-ensemble-design.md sec 5.6
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from apex.strategies import register_strategy
from apex.strategies.base import StrategyBase


@register_strategy
class CrossAssetVolOverlayStrategy(StrategyBase):
    name = "cross_asset_vol_overlay"
    data_requirements = ["exec_df_1H"]

    # Expected columns on exec_df_1H: vix_pct, move_pct, ovx_pct
    # (rolling 252-day percentiles of VIX, MOVE proxy, OVX).

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        # No tunable params per spec; accept overrides for thresholds for
        # future flexibility but default to the spec values.
        defaults = {
            "high_pct_threshold": 80.0,
            "low_pct_threshold": 20.0,
            "size_mult_high": 0.5,
            "size_mult_low": 1.2,
            "size_mult_neutral": 1.0,
        }
        if params:
            defaults.update(params)
        self.params = defaults

    def compute_signals(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Overlay never emits trade signals — return all-False columns."""
        df = data["exec_df_1H"]
        n = len(df)
        false_col = np.zeros(n, dtype=bool)
        return pd.DataFrame({
            "entry_long": false_col, "entry_short": false_col,
            "exit_long": false_col, "exit_short": false_col,
        })

    def compute_position_size(self, data: Dict[str, Any],
                              signals: pd.DataFrame) -> pd.Series:
        """Return per-bar SIZE MULTIPLIER (not a directional position).

        - All three pcts > high_pct_threshold (80) -> size_mult_high (0.5)
        - All three pcts < low_pct_threshold (20)  -> size_mult_low (1.2)
        - Otherwise (divergent / neutral / missing) -> size_mult_neutral (1.0)
        """
        df = data["exec_df_1H"]
        n = len(df)
        mult = np.full(n, self.params["size_mult_neutral"], dtype=float)

        vix_pct = df.get("vix_pct")
        move_pct = df.get("move_pct")
        ovx_pct = df.get("ovx_pct")

        if vix_pct is None or move_pct is None or ovx_pct is None:
            return pd.Series(mult)

        hi = self.params["high_pct_threshold"]
        lo = self.params["low_pct_threshold"]
        m_high = self.params["size_mult_high"]
        m_low = self.params["size_mult_low"]

        for i in range(n):
            v = vix_pct.iloc[i]
            m = move_pct.iloc[i]
            o = ovx_pct.iloc[i]
            if pd.isna(v) or pd.isna(m) or pd.isna(o):
                continue
            if v > hi and m > hi and o > hi:
                mult[i] = m_high
            elif v < lo and m < lo and o < lo:
                mult[i] = m_low
            # else: leave at neutral (1.0)

        return pd.Series(mult)

    def get_tunable_params(self) -> Dict[str, tuple]:
        """No tunable params — overlay thresholds are hardcoded per spec."""
        return {}
