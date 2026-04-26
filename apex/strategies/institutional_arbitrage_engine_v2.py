"""Institutional Arbitrage Engine v2 research prototype.

This is the backtestable-underlying subset of the multi-engine report:

* Engine 1: term-structure gated VWAP/RSI(2) fade in calm contango.
* Engine 2: regime-adaptive momentum using 21/63-day returns in contango and
  126-day returns in backwardation.
* Engine 3: OPEX pin-gravity direction when a `pin_strike` column is present.

It does NOT model iron-condor option PnL. Until the engine supports option legs,
premium-selling claims stay out of this strategy.
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from apex.data.opex_calendar import is_opex_week
from apex.indicators.basics import compute_atr, compute_rsi
from apex.indicators.vwap_bands import compute_vwap_bands
from apex.strategies import register_strategy
from apex.strategies.base import StrategyBase


@register_strategy
class InstitutionalArbitrageEngineV2Strategy(StrategyBase):
    name = "institutional_arbitrage_engine_v2"
    data_requirements = ["exec_df_1H"]

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        defaults = {
            "contango_max": 0.95,
            "backwardation_min": 1.02,
            "vix_max_fade": 25.0,
            "vrp_calm_low": 30.0,
            "vrp_calm_high": 70.0,
            "vrp_elevated_low": 20.0,
            "vrp_elevated_high": 80.0,
            "fade_deviation_sigma": 2.0,
            "fade_vwap_slope_atr_max": 0.10,
            "fade_rsi2_oversold": 10,
            "fade_rsi2_overbought": 90,
            "min_session_bars": 5,
            "momentum_short_bars": 21 * 7,
            "momentum_long_bars": 63 * 7,
            "momentum_crisis_bars": 126 * 7,
            "momentum_threshold_pct": 0.0,
            "pin_min_distance_pct": 0.005,
            "pin_touch_exit_pct": 0.002,
            "stop_atr_mult": 1.0,
            "max_hold_bars": 21,
            "enable_fade": True,
            "enable_momentum": True,
            "enable_opex_pin": True,
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

    def _regime_inputs(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        vix = df.get("vix", pd.Series(np.nan, index=df.index))
        vxv = df.get("vxv", pd.Series(np.nan, index=df.index))
        vrp = df.get("vrp_pct", pd.Series(np.nan, index=df.index))
        ts_ratio = vix / vxv.replace(0, np.nan)
        return ts_ratio, vix, vrp

    def compute_signals(self, data: Dict[str, Any]) -> pd.DataFrame:
        df = data["exec_df_1H"]
        n = len(df)
        entry_long = np.zeros(n, dtype=bool)
        entry_short = np.zeros(n, dtype=bool)
        exit_long = np.zeros(n, dtype=bool)
        exit_short = np.zeros(n, dtype=bool)

        if n == 0 or "close" not in df.columns or "datetime" not in df.columns:
            return pd.DataFrame({
                "entry_long": entry_long, "entry_short": entry_short,
                "exit_long": exit_long, "exit_short": exit_short,
            })

        feat = self._features(df)
        ts_ratio, vix, vrp = self._regime_inputs(feat)
        dt = pd.to_datetime(feat["datetime"])
        session_bar = feat.groupby(dt.dt.date).cumcount()
        close = feat["close"]

        sigma = feat["vwap_1s_upper"] - feat["vwap"]
        dev = float(self.params["fade_deviation_sigma"]) * sigma

        short_ret = close.pct_change(int(self.params["momentum_short_bars"]))
        long_ret = close.pct_change(int(self.params["momentum_long_bars"]))
        crisis_ret = close.pct_change(int(self.params["momentum_crisis_bars"]))
        blended_mom = 0.30 * short_ret + 0.70 * long_ret

        pin_strike = feat.get("pin_strike")
        mom_thr = float(self.params["momentum_threshold_pct"]) / 100.0

        for i in range(n):
            r = ts_ratio.iloc[i]
            if pd.isna(r):
                continue
            contango = r < self.params["contango_max"]
            backwardation = r > self.params["backwardation_min"]
            calm_vrp = (
                not pd.isna(vrp.iloc[i]) and
                self.params["vrp_calm_low"] <= vrp.iloc[i] <= self.params["vrp_calm_high"]
            )
            elevated_vrp = (
                not pd.isna(vrp.iloc[i]) and
                (vrp.iloc[i] <= self.params["vrp_elevated_low"] or
                 vrp.iloc[i] >= self.params["vrp_elevated_high"])
            )

            # Engine 1: calm-contango VWAP exhaustion fade.
            if self.params["enable_fade"] and contango and calm_vrp:
                if (
                    not pd.isna(vix.iloc[i]) and vix.iloc[i] < self.params["vix_max_fade"] and
                    session_bar.iloc[i] >= int(self.params["min_session_bars"]) and
                    abs(feat["vwap_slope_atr"].iloc[i]) <= self.params["fade_vwap_slope_atr_max"] and
                    not pd.isna(dev.iloc[i]) and dev.iloc[i] > 0
                ):
                    if (
                        close.iloc[i] < feat["vwap"].iloc[i] - dev.iloc[i] and
                        feat["rsi2"].iloc[i] <= self.params["fade_rsi2_oversold"]
                    ):
                        entry_long[i] = True
                    elif (
                        close.iloc[i] > feat["vwap"].iloc[i] + dev.iloc[i] and
                        feat["rsi2"].iloc[i] >= self.params["fade_rsi2_overbought"]
                    ):
                        entry_short[i] = True

            # Fade exits: VWAP mean reversion or RSI unwind. These are only
            # valid inside the fade regime; applying them globally turns the
            # momentum engine into bar-by-bar churn.
            if self.params["enable_fade"] and contango and calm_vrp:
                if close.iloc[i] >= feat["vwap"].iloc[i] or feat["rsi2"].iloc[i] >= 60:
                    exit_long[i] = True
                if close.iloc[i] <= feat["vwap"].iloc[i] or feat["rsi2"].iloc[i] <= 40:
                    exit_short[i] = True

            # Engine 2: regime-adaptive momentum.
            if self.params["enable_momentum"]:
                mom = crisis_ret.iloc[i] if backwardation else blended_mom.iloc[i]
                # VRP extremes are a risk scaler, not a direction veto. We still
                # allow momentum in elevated regimes, but position sizing reduces.
                if not pd.isna(mom):
                    if contango or backwardation or elevated_vrp:
                        if mom > mom_thr:
                            entry_long[i] = True
                            exit_short[i] = True
                        elif mom < -mom_thr:
                            entry_short[i] = True
                            exit_long[i] = True

            # Engine 3: OPEX pin-gravity when a pin_strike column exists.
            if self.params["enable_opex_pin"] and pin_strike is not None:
                pin = pin_strike.iloc[i]
                if (
                    contango and calm_vrp and not pd.isna(pin) and pin > 0 and
                    is_opex_week(dt.iloc[i]) and dt.iloc[i].weekday() in (1, 2)
                ):
                    distance = (pin - close.iloc[i]) / close.iloc[i]
                    if distance > self.params["pin_min_distance_pct"]:
                        entry_long[i] = True
                    elif distance < -self.params["pin_min_distance_pct"]:
                        entry_short[i] = True

                if not pd.isna(pin) and pin > 0:
                    if abs(close.iloc[i] - pin) / pin <= self.params["pin_touch_exit_pct"]:
                        exit_long[i] = True
                        exit_short[i] = True
                if dt.iloc[i].weekday() == 4 and dt.iloc[i].hour >= 12:
                    exit_long[i] = True
                    exit_short[i] = True

            # Same-direction entry means at least one active engine still wants
            # the exposure. Do not let another engine's exit flatten it.
            if entry_long[i]:
                exit_long[i] = False
            if entry_short[i]:
                exit_short[i] = False

        return pd.DataFrame({
            "entry_long": entry_long, "entry_short": entry_short,
            "exit_long": exit_long, "exit_short": exit_short,
        })

    def compute_position_size(self, data: Dict[str, Any],
                              signals: pd.DataFrame) -> pd.Series:
        df = data["exec_df_1H"]
        n = len(signals)
        pos = np.zeros(n, dtype=float)
        atr = compute_atr(df, period=14)
        _, _, vrp = self._regime_inputs(df)

        side = 0
        bars_in_pos = 0
        entry_price = 0.0
        entry_atr = 0.0

        for i in range(n):
            close_i = float(df["close"].iloc[i])
            atr_i = atr.iloc[i]
            if pd.isna(atr_i) or atr_i <= 0:
                atr_i = entry_atr if entry_atr > 0 else 0.0

            if side == 0:
                if signals["entry_long"].iloc[i] and not signals["entry_short"].iloc[i]:
                    side = 1
                elif signals["entry_short"].iloc[i] and not signals["entry_long"].iloc[i]:
                    side = -1
                if side != 0:
                    bars_in_pos = 0
                    entry_price = close_i
                    entry_atr = float(atr_i)
            else:
                stop_hit = False
                if entry_atr > 0:
                    if side == 1 and close_i <= entry_price - self.params["stop_atr_mult"] * entry_atr:
                        stop_hit = True
                    elif side == -1 and close_i >= entry_price + self.params["stop_atr_mult"] * entry_atr:
                        stop_hit = True
                exit_hit = (
                    (side == 1 and signals["exit_long"].iloc[i]) or
                    (side == -1 and signals["exit_short"].iloc[i]) or
                    stop_hit or
                    bars_in_pos >= int(self.params["max_hold_bars"])
                )
                if exit_hit:
                    side = 0
                    bars_in_pos = 0
                    entry_price = 0.0
                    entry_atr = 0.0

            if side != 0:
                size = 1.0
                vrp_i = vrp.iloc[i]
                if (
                    not pd.isna(vrp_i) and
                    (vrp_i <= self.params["vrp_elevated_low"] or
                     vrp_i >= self.params["vrp_elevated_high"])
                ):
                    size = 0.6
                pos[i] = float(side) * size
                bars_in_pos += 1

        return pd.Series(pos)

    def get_tunable_params(self) -> Dict[str, tuple]:
        return {
            "contango_max":              (0.90, 0.99, "float"),
            "backwardation_min":         (1.00, 1.12, "float"),
            "vix_max_fade":              (18.0, 30.0, "float"),
            "fade_deviation_sigma":      (1.2, 2.8, "float"),
            "fade_vwap_slope_atr_max":   (0.05, 0.30, "float"),
            "fade_rsi2_oversold":        (5, 20, "int"),
            "fade_rsi2_overbought":      (80, 95, "int"),
            "momentum_threshold_pct":     (0.0, 2.0, "float"),
            "pin_min_distance_pct":       (0.003, 0.015, "float"),
            "stop_atr_mult":             (0.5, 2.0, "float"),
            "max_hold_bars":             (5, 60, "int"),
        }
