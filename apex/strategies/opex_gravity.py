"""Strategy 2: OPEX Gravity.

Structural primitive: Max-pain magnetism. Predictable gamma-induced pinning
around monthly options expiration (3rd Friday). Trades on Tue/Wed during OPEX
week, exits forced on Friday.

Spec: docs/superpowers/specs/2026-04-26-institutional-ensemble-design.md sec 5.2
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from apex.data.opex_calendar import is_opex_week
from apex.strategies import register_strategy
from apex.strategies.base import StrategyBase


# Mapping for entry-day windows
_ENTRY_DOW_MAP = {
    "Mon-Tue": (0, 1),
    "Tue-Wed": (1, 2),
    "Wed-Thu": (2, 3),
}

# Mapping for forced-exit timing — represented by latest weekday at which
# we hold (Friday = 4). For 'Thu' we exit at Thursday close (weekday 3).
# 'Fri-mid' exits during Friday session; 'Fri-close' lets us hold to Friday EOD.
# We coalesce by treating each as a Friday-or-earlier close cutoff.
_FORCED_EXIT_MAP = {
    "Thu": 3,
    "Fri-mid": 4,
    "Fri-close": 4,
}


@register_strategy
class OPEXGravityStrategy(StrategyBase):
    name = "opex_gravity"
    data_requirements = ["exec_df_1H"]

    # OPEX Gravity also needs `opex_chain` in data with weekly pin_strike map
    # (pre-merged in Phase 12H by ingest pipeline).

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        defaults = {
            "min_pin_distance_pct": 0.005,
            "pin_strike_window_pct": 0.05,
            "entry_dow": "Tue-Wed",
            "forced_exit_dow": "Fri-mid",
        }
        if params:
            defaults.update(params)
        self.params = defaults

    def _resolve_pin_strike(self, data: Dict[str, Any], ts: pd.Timestamp,
                             spot: float) -> Optional[float]:
        """Resolve the pin strike for the OPEX week containing ``ts``.

        Lookup priority:
          1) data['opex_chain'][week_key] -> dict with 'pin_strike' float
          2) data['exec_df_1H']['pin_strike'] column at this row
          3) None (skip)
        """
        chain_map = data.get("opex_chain")
        if isinstance(chain_map, dict):
            # Normalize the key to the Monday of the OPEX week (date string)
            week_monday = (ts - pd.Timedelta(days=ts.weekday())).normalize()
            key_str = week_monday.strftime("%Y-%m-%d")
            entry = chain_map.get(key_str) or chain_map.get(week_monday)
            if entry is not None:
                pin = entry.get("pin_strike") if isinstance(entry, dict) else entry
                if pin is not None and not pd.isna(pin):
                    return float(pin)
        return None

    def compute_signals(self, data: Dict[str, Any]) -> pd.DataFrame:
        df = data["exec_df_1H"]
        n = len(df)
        entry_long = np.zeros(n, dtype=bool)
        entry_short = np.zeros(n, dtype=bool)
        exit_long = np.zeros(n, dtype=bool)
        exit_short = np.zeros(n, dtype=bool)

        if "datetime" not in df.columns or "close" not in df.columns:
            return pd.DataFrame({
                "entry_long": entry_long, "entry_short": entry_short,
                "exit_long": exit_long, "exit_short": exit_short,
            })

        dt_series = pd.to_datetime(df["datetime"])
        close_series = df["close"]

        # Optional column-based pin_strike fallback
        pin_col = df.get("pin_strike")

        entry_dow_lo, entry_dow_hi = _ENTRY_DOW_MAP.get(
            self.params["entry_dow"], (1, 2))
        min_dist = self.params["min_pin_distance_pct"]

        for i in range(n):
            ts = dt_series.iloc[i]
            if pd.isna(ts):
                continue
            if not is_opex_week(ts):
                continue
            wd = ts.weekday()
            if wd < entry_dow_lo or wd > entry_dow_hi:
                continue

            spot = close_series.iloc[i]
            if pd.isna(spot) or spot <= 0:
                continue

            pin = None
            if pin_col is not None:
                v = pin_col.iloc[i]
                if not pd.isna(v):
                    pin = float(v)
            if pin is None:
                pin = self._resolve_pin_strike(data, ts, float(spot))
            if pin is None:
                continue

            distance = (pin - spot) / spot
            if distance > min_dist:
                # price below pin -> upward gravity -> LONG
                entry_long[i] = True
            elif distance < -min_dist:
                # price above pin -> downward gravity -> SHORT
                entry_short[i] = True

        return pd.DataFrame({
            "entry_long": entry_long, "entry_short": entry_short,
            "exit_long": exit_long, "exit_short": exit_short,
        })

    def compute_position_size(self, data: Dict[str, Any],
                              signals: pd.DataFrame) -> pd.Series:
        df = data["exec_df_1H"]
        n = len(signals)
        pos = np.zeros(n, dtype=float)

        if "datetime" not in df.columns:
            return pd.Series(pos)

        dt_series = pd.to_datetime(df["datetime"])
        close_series = df.get("close")

        # Forced exit weekday cutoff
        forced_exit_wd = _FORCED_EXIT_MAP.get(self.params["forced_exit_dow"], 4)
        forced_exit_dow_label = self.params["forced_exit_dow"]
        # 'Fri-mid' = exit at noon (hour < 14), 'Fri-close' = end of day.
        side = 0
        entry_pin = None

        # Pre-extract pin per bar (column or per-week resolution) for exit checks.
        pin_col = df.get("pin_strike")

        for i in range(n):
            ts = dt_series.iloc[i]
            if pd.isna(ts):
                continue

            # Open new position
            if side == 0:
                if signals["entry_long"].iloc[i]:
                    side = 1
                elif signals["entry_short"].iloc[i]:
                    side = -1
                if side != 0:
                    # Capture this bar's pin for exit-on-touch checks
                    if pin_col is not None and not pd.isna(pin_col.iloc[i]):
                        entry_pin = float(pin_col.iloc[i])
                    else:
                        entry_pin = self._resolve_pin_strike(
                            data, ts, float(close_series.iloc[i])
                            if close_series is not None else 0.0)

            if side != 0:
                pos[i] = float(side)

                # Pin-touch exit (within +/- 0.2%)
                if entry_pin is not None and close_series is not None:
                    spot = close_series.iloc[i]
                    if not pd.isna(spot) and entry_pin > 0:
                        if abs(spot - entry_pin) / entry_pin <= 0.002:
                            side = 0
                            entry_pin = None
                            continue

                # Forced Friday close exit
                wd = ts.weekday()
                hour = ts.hour
                exit_now = False
                if wd > forced_exit_wd:
                    exit_now = True
                elif wd == forced_exit_wd:
                    if forced_exit_dow_label == "Thu":
                        # Thursday: exit at end of day (>= 15:00)
                        exit_now = hour >= 15
                    elif forced_exit_dow_label == "Fri-mid":
                        exit_now = hour >= 12
                    elif forced_exit_dow_label == "Fri-close":
                        exit_now = hour >= 15
                if exit_now:
                    side = 0
                    entry_pin = None

        return pd.Series(pos)

    def get_tunable_params(self) -> Dict[str, tuple]:
        return {
            "min_pin_distance_pct":   (0.003, 0.015, "float"),
            "pin_strike_window_pct":  (0.03,  0.08,  "float"),
            "entry_dow":              (None,  None,  "categorical",
                                       ["Mon-Tue", "Tue-Wed", "Wed-Thu"]),
            "forced_exit_dow":        (None,  None,  "categorical",
                                       ["Thu", "Fri-mid", "Fri-close"]),
        }
