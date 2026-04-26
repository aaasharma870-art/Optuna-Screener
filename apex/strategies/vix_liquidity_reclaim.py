"""VIX-term liquidity reclaim strategy.

Structural hypothesis:
Price should not be faded just because it is far from VWAP. A better intraday
mean-reversion setup is a failed liquidity break: sweep a known reference level
(prior-day high/low or opening range), reclaim it, and only trade when VIX term
structure is not in an adverse panic-continuation regime.
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from apex.indicators.basics import compute_atr, compute_rsi
from apex.indicators.vwap_bands import compute_vwap_bands
from apex.strategies import register_strategy
from apex.strategies.base import StrategyBase


@register_strategy
class VIXLiquidityReclaimStrategy(StrategyBase):
    name = "vix_liquidity_reclaim"
    data_requirements = ["exec_df_1H"]

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        defaults = {
            "enable_long": True,
            "enable_short": False,
            "contango_max": 1.02,
            "backwardation_max": 1.12,
            "vix_max": 35.0,
            "require_vix_rollover": True,
            "vix_rollover_lookback": 3,
            "vix_rollover_min_drop": 0.0,
            "require_ts_ratio_rollover": False,
            "ts_ratio_rollover_lookback": 3,
            "require_ts_rsi_turn": False,
            "ts_rsi_long_max": 55,
            "ts_rsi_short_min": 45,
            "use_prior_day": True,
            "use_opening_range": True,
            "opening_range_bars": 6,
            "sweep_atr_frac": 0.05,
            "reclaim_buffer_atr": 0.02,
            "max_entry_hour": 15.5,
            "min_session_bars": 1,
            "target": "vwap",
            "stop_atr_mult": 1.2,
            "max_hold_bars": 12,
            "exit_at_session_end": True,
        }
        if params:
            defaults.update(params)
        self.params = defaults

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        timestamp_col = "datetime" if "datetime" in df.columns else "timestamp"
        feat = compute_vwap_bands(df, timestamp_col=timestamp_col)
        feat["atr"] = compute_atr(feat, period=14)
        vix = feat.get("vix")
        vxv = feat.get("vxv")
        if vix is not None and vxv is not None:
            feat["ts_ratio"] = vix / vxv.replace(0, np.nan)
            feat["ts_rsi"] = compute_rsi(feat["ts_ratio"], period=5)
            vix_roll_n = int(self.params["vix_rollover_lookback"])
            ts_roll_n = int(self.params["ts_ratio_rollover_lookback"])
            feat["vix_recent_high"] = vix.shift(1).rolling(
                vix_roll_n, min_periods=1).max()
            feat["vix_recent_low"] = vix.shift(1).rolling(
                vix_roll_n, min_periods=1).min()
            feat["ts_recent_high"] = feat["ts_ratio"].rolling(
                ts_roll_n, min_periods=1).max().shift(1)
            feat["ts_recent_low"] = feat["ts_ratio"].rolling(
                ts_roll_n, min_periods=1).min().shift(1)
        else:
            feat["ts_ratio"] = np.nan
            feat["ts_rsi"] = np.nan
            feat["vix_recent_high"] = np.nan
            feat["vix_recent_low"] = np.nan
            feat["ts_recent_high"] = np.nan
            feat["ts_recent_low"] = np.nan

        dt = pd.to_datetime(feat[timestamp_col])
        feat["_date"] = dt.dt.date
        feat["_hour"] = dt.dt.hour + dt.dt.minute / 60.0
        feat["_session_bar"] = feat.groupby("_date").cumcount()

        daily = feat.groupby("_date").agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
        )
        feat["prior_day_high"] = feat["_date"].map(daily["day_high"].shift(1))
        feat["prior_day_low"] = feat["_date"].map(daily["day_low"].shift(1))
        feat["prior_day_close"] = feat["_date"].map(daily["day_close"].shift(1))

        orb_n = int(self.params["opening_range_bars"])
        feat["opening_range_high"] = (
            feat.groupby("_date")["high"]
            .transform(lambda s: s.iloc[:orb_n].max() if len(s) >= orb_n else np.nan)
        )
        feat["opening_range_low"] = (
            feat.groupby("_date")["low"]
            .transform(lambda s: s.iloc[:orb_n].min() if len(s) >= orb_n else np.nan)
        )
        feat["_last_bar_of_day"] = feat["_date"].ne(feat["_date"].shift(-1))
        return feat

    def compute_signals(self, data: Dict[str, Any]) -> pd.DataFrame:
        df = data["exec_df_1H"]
        n = len(df)
        entry_long = np.zeros(n, dtype=bool)
        entry_short = np.zeros(n, dtype=bool)
        exit_long = np.zeros(n, dtype=bool)
        exit_short = np.zeros(n, dtype=bool)

        if n == 0 or "datetime" not in df.columns:
            return pd.DataFrame({
                "entry_long": entry_long, "entry_short": entry_short,
                "exit_long": exit_long, "exit_short": exit_short,
            })

        feat = self._features(df)

        for i in range(1, n):
            r = feat.iloc[i]
            prev = feat.iloc[i - 1]
            atr = r["atr"]
            if pd.isna(atr) or atr <= 0:
                continue
            if r["_session_bar"] < int(self.params["min_session_bars"]):
                continue
            if r["_hour"] > float(self.params["max_entry_hour"]):
                continue
            if pd.isna(r["ts_ratio"]) or r["ts_ratio"] > self.params["backwardation_max"]:
                continue
            if pd.isna(r["vix"]) or r["vix"] > self.params["vix_max"]:
                continue

            ts_ok_long = r["ts_ratio"] <= self.params["contango_max"]
            ts_ok_short = r["ts_ratio"] <= self.params["contango_max"]
            if self.params["require_vix_rollover"]:
                min_drop = float(self.params["vix_rollover_min_drop"])
                vix_roll_long = (
                    pd.notna(r["vix_recent_high"]) and
                    r["vix"] <= r["vix_recent_high"] - min_drop
                )
                vix_roll_short = (
                    pd.notna(r["vix_recent_low"]) and
                    r["vix"] >= r["vix_recent_low"] + min_drop
                )
                ts_ok_long = ts_ok_long and vix_roll_long
                ts_ok_short = ts_ok_short and vix_roll_short
            if self.params["require_ts_ratio_rollover"]:
                ts_ok_long = (
                    ts_ok_long and pd.notna(r["ts_recent_high"]) and
                    r["ts_ratio"] <= r["ts_recent_high"]
                )
                ts_ok_short = (
                    ts_ok_short and pd.notna(r["ts_recent_low"]) and
                    r["ts_ratio"] >= r["ts_recent_low"]
                )
            if self.params["require_ts_rsi_turn"]:
                ts_ok_long = ts_ok_long and pd.notna(r["ts_rsi"]) and r["ts_rsi"] <= self.params["ts_rsi_long_max"]
                ts_ok_short = ts_ok_short and pd.notna(r["ts_rsi"]) and r["ts_rsi"] >= self.params["ts_rsi_short_min"]

            sweep = float(self.params["sweep_atr_frac"]) * atr
            reclaim = float(self.params["reclaim_buffer_atr"]) * atr

            long_reclaim = False
            short_reclaim = False

            if self.params["use_prior_day"]:
                pdl = r["prior_day_low"]
                pdh = r["prior_day_high"]
                if pd.notna(pdl):
                    long_reclaim |= (
                        r["low"] < pdl - sweep and
                        r["close"] > pdl + reclaim and
                        prev["close"] <= pdl + reclaim
                    )
                if pd.notna(pdh):
                    short_reclaim |= (
                        r["high"] > pdh + sweep and
                        r["close"] < pdh - reclaim and
                        prev["close"] >= pdh - reclaim
                    )

            if self.params["use_opening_range"] and r["_session_bar"] >= self.params["opening_range_bars"]:
                orl = r["opening_range_low"]
                orh = r["opening_range_high"]
                if pd.notna(orl):
                    long_reclaim |= (
                        r["low"] < orl - sweep and
                        r["close"] > orl + reclaim and
                        prev["close"] <= orl + reclaim
                    )
                if pd.notna(orh):
                    short_reclaim |= (
                        r["high"] > orh + sweep and
                        r["close"] < orh - reclaim and
                        prev["close"] >= orh - reclaim
                    )

            if self.params["enable_long"] and ts_ok_long and long_reclaim:
                entry_long[i] = True
            if self.params["enable_short"] and ts_ok_short and short_reclaim:
                entry_short[i] = True

            if r["close"] >= r["vwap"]:
                exit_long[i] = True
            if r["close"] <= r["vwap"]:
                exit_short[i] = True

        return pd.DataFrame({
            "entry_long": entry_long, "entry_short": entry_short,
            "exit_long": exit_long, "exit_short": exit_short,
        })

    def compute_position_size(self, data: Dict[str, Any],
                              signals: pd.DataFrame) -> pd.Series:
        df = data["exec_df_1H"]
        feat = self._features(df)
        n = len(signals)
        pos = np.zeros(n, dtype=float)
        side = 0
        entry_price = 0.0
        entry_atr = 0.0
        bars = 0
        max_hold = int(self.params["max_hold_bars"])
        stop_mult = float(self.params["stop_atr_mult"])

        for i in range(n):
            close = float(feat["close"].iloc[i])
            atr = feat["atr"].iloc[i]
            if pd.isna(atr) or atr <= 0:
                atr = entry_atr if entry_atr > 0 else 0.0

            if side == 0:
                if signals["entry_long"].iloc[i] and not signals["entry_short"].iloc[i]:
                    side = 1
                    entry_price = close
                    entry_atr = float(atr)
                    bars = 0
                elif signals["entry_short"].iloc[i] and not signals["entry_long"].iloc[i]:
                    side = -1
                    entry_price = close
                    entry_atr = float(atr)
                    bars = 0
            else:
                stop_hit = False
                if entry_atr > 0:
                    if side == 1 and close <= entry_price - stop_mult * entry_atr:
                        stop_hit = True
                    elif side == -1 and close >= entry_price + stop_mult * entry_atr:
                        stop_hit = True

                eod = bool(self.params["exit_at_session_end"] and feat["_last_bar_of_day"].iloc[i])
                exit_hit = (
                    (side == 1 and signals["exit_long"].iloc[i]) or
                    (side == -1 and signals["exit_short"].iloc[i]) or
                    stop_hit or eod or bars >= max_hold
                )
                if exit_hit:
                    side = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    bars = 0

            if side != 0:
                pos[i] = float(side)
                bars += 1

        return pd.Series(pos)

    def get_tunable_params(self) -> Dict[str, tuple]:
        return {
            "enable_long":          (None, None, "categorical", [True, False]),
            "enable_short":         (None, None, "categorical", [True, False]),
            "contango_max":         (0.90, 1.04, "float"),
            "backwardation_max":    (1.02, 1.20, "float"),
            "vix_max":              (18.0, 45.0, "float"),
            "require_vix_rollover": (None, None, "categorical", [True, False]),
            "vix_rollover_lookback": (2, 8, "int"),
            "vix_rollover_min_drop": (0.0, 2.0, "float"),
            "require_ts_ratio_rollover": (None, None, "categorical", [True, False]),
            "ts_ratio_rollover_lookback": (2, 8, "int"),
            "require_ts_rsi_turn":  (None, None, "categorical", [True, False]),
            "ts_rsi_long_max":      (30, 70, "int"),
            "ts_rsi_short_min":     (30, 70, "int"),
            "use_prior_day":        (None, None, "categorical", [True, False]),
            "use_opening_range":    (None, None, "categorical", [True, False]),
            "opening_range_bars":   (3, 12, "int"),
            "sweep_atr_frac":       (0.00, 0.50, "float"),
            "reclaim_buffer_atr":   (0.00, 0.20, "float"),
            "max_entry_hour":       (11.0, 15.75, "float"),
            "min_session_bars":     (1, 20, "int"),
            "stop_atr_mult":        (0.5, 3.0, "float"),
            "max_hold_bars":        (2, 40, "int"),
            "exit_at_session_end":  (None, None, "categorical", [True, False]),
        }
