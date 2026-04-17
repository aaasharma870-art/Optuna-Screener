"""Backtest engine: indicator signals, regime, entry scoring, bar-by-bar simulation."""

import math

import numpy as np
import pandas as pd

from apex.indicators.basics import (
    compute_atr, compute_ema, compute_rsi, compute_macd, compute_bollinger,
    compute_stochastic, compute_obv, compute_adx, compute_cci,
    compute_williams_r, compute_keltner, compute_volume_surge, compute_vwap,
)


def compute_indicator_signals(df, architecture, params):
    """
    Compute all indicators specified in *architecture['indicators']* and
    return a dict of per-bar signal Series.

    Each signal is an integer Series: +1 = bullish, -1 = bearish, 0 = neutral.
    """
    active = architecture.get("indicators", [])
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    signals = {}

    atr_period = params.get("atr_period", 14)
    atr = compute_atr(df, atr_period)

    if "RSI" in active:
        rsi_period = params.get("rsi_period", 14)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi = compute_rsi(close, rsi_period)
        sig = pd.Series(0, index=df.index)
        sig[rsi < rsi_oversold] = 1
        sig[rsi > rsi_overbought] = -1
        signals["RSI"] = sig

    if "MACD" in active:
        macd_fast = params.get("macd_fast", 12)
        macd_slow = params.get("macd_slow", 26)
        macd_signal_p = params.get("macd_signal", 9)
        macd_line, signal_line, histogram = compute_macd(close, macd_fast, macd_slow, macd_signal_p)
        sig = pd.Series(0, index=df.index)
        sig[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1
        sig[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1
        sig[(sig == 0) & (histogram > 0) & (histogram > histogram.shift(1))] = 1
        sig[(sig == 0) & (histogram < 0) & (histogram < histogram.shift(1))] = -1
        signals["MACD"] = sig

    if "Bollinger" in active:
        boll_period = params.get("boll_period", 20)
        boll_std = params.get("boll_std", 2.0)
        upper, mid, lower = compute_bollinger(close, boll_period, boll_std)
        sig = pd.Series(0, index=df.index)
        sig[close <= lower] = 1
        sig[close >= upper] = -1
        signals["Bollinger"] = sig

    if "Stochastic" in active:
        stoch_k = params.get("stoch_k", 14)
        stoch_d = params.get("stoch_d", 3)
        k_line, d_line = compute_stochastic(high, low, close, stoch_k, stoch_d)
        sig = pd.Series(0, index=df.index)
        sig[(k_line < 20) & (k_line > d_line)] = 1
        sig[(k_line > 80) & (k_line < d_line)] = -1
        signals["Stochastic"] = sig

    if "OBV" in active:
        obv_ma_period = params.get("obv_ma_period", 20)
        obv, obv_ma = compute_obv(close, volume, obv_ma_period)
        sig = pd.Series(0, index=df.index)
        sig[obv > obv_ma] = 1
        sig[obv < obv_ma] = -1
        signals["OBV"] = sig

    if "ADX" in active:
        adx_period = params.get("adx_period", 14)
        adx_threshold = params.get("adx_threshold", 25)
        adx = compute_adx(high, low, close, adx_period)
        ema_short = compute_ema(close, 9)
        ema_long = compute_ema(close, 21)
        sig = pd.Series(0, index=df.index)
        trending = adx > adx_threshold
        sig[(trending) & (ema_short > ema_long)] = 1
        sig[(trending) & (ema_short < ema_long)] = -1
        signals["ADX"] = sig

    if "CCI" in active:
        cci_period = params.get("cci_period", 20)
        cci_oversold = params.get("cci_oversold", -100)
        cci_overbought = params.get("cci_overbought", 100)
        cci = compute_cci(high, low, close, cci_period)
        sig = pd.Series(0, index=df.index)
        sig[cci < cci_oversold] = 1
        sig[cci > cci_overbought] = -1
        signals["CCI"] = sig

    if "WilliamsR" in active:
        willr_period = params.get("willr_period", 14)
        willr_oversold = params.get("willr_oversold", -80)
        willr_overbought = params.get("willr_overbought", -20)
        wr = compute_williams_r(high, low, close, willr_period)
        sig = pd.Series(0, index=df.index)
        sig[wr < willr_oversold] = 1
        sig[wr > willr_overbought] = -1
        signals["WilliamsR"] = sig

    if "Keltner" in active:
        keltner_period = params.get("keltner_period", 20)
        keltner_mult = params.get("keltner_mult", 2.0)
        k_upper, k_mid, k_lower = compute_keltner(close, atr, keltner_period, keltner_mult)
        sig = pd.Series(0, index=df.index)
        sig[close <= k_lower] = 1
        sig[close >= k_upper] = -1
        signals["Keltner"] = sig

    if "VolumeSurge" in active:
        vs_ma = params.get("volume_surge_ma", 20)
        vs_mult = params.get("volume_surge_mult", 1.5)
        surge = compute_volume_surge(volume, vs_ma, vs_mult)
        sig = pd.Series(0, index=df.index)
        sig[(surge) & (close > df["open"])] = 1
        sig[(surge) & (close < df["open"])] = -1
        signals["VolumeSurge"] = sig

    if "VWAP" in active:
        vwap = compute_vwap(df)
        sig = pd.Series(0, index=df.index)
        sig[close > vwap] = 1
        sig[close < vwap] = -1
        signals["VWAP"] = sig

    if "EMA_Cross" in active:
        ema_fast_p = params.get("ema_fast", 9)
        ema_slow_p = params.get("ema_slow", 21)
        ema_f = compute_ema(close, ema_fast_p)
        ema_s = compute_ema(close, ema_slow_p)
        sig = pd.Series(0, index=df.index)
        sig[(ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))] = 1
        sig[(ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))] = -1
        sig[(sig == 0) & (ema_f > ema_s)] = 1
        sig[(sig == 0) & (ema_f < ema_s)] = -1
        signals["EMA_Cross"] = sig

    signals["_atr"] = atr
    return signals


def compute_regime(df, daily_df, regime_model, params):
    """
    Compute a per-bar regime label using a simple price-vs-EMA model.

    Regime codes (R1=best, R4=worst) are a compact way for the backtester
    to accept or reject entries based on broad market conditions.

      * ``"trend"``       ADX + EMA cross
      * ``"volatility"``  ATR percentile bucket + price vs EMA20
      * ``"ema"``         simple EMA20 vs EMA50 classification (default)

    Custom regime models can be plugged in by extending this function.
    """
    n = len(df)
    regime = pd.Series("R1", index=df.index)

    if regime_model == "volatility":
        atr = compute_atr(df, params.get("atr_period", 14))
        atr_pct = (atr / df["close"]) * 100.0
        atr_med = atr_pct.rolling(100, min_periods=20).median()
        ema20 = compute_ema(df["close"], 20)
        for i in range(n):
            am = atr_med.iloc[i]
            ap = atr_pct.iloc[i]
            above_ema = df["close"].iloc[i] > ema20.iloc[i]
            if pd.isna(am) or pd.isna(ap):
                regime.iloc[i] = "R1"
            elif ap < am and above_ema:
                regime.iloc[i] = "R1"   # Low vol, bullish
            elif ap < am:
                regime.iloc[i] = "R2"   # Low vol, bearish
            elif above_ema:
                regime.iloc[i] = "R3"   # High vol, bullish
            else:
                regime.iloc[i] = "R4"   # High vol, bearish

    elif regime_model == "trend":
        adx = compute_adx(df["high"], df["low"], df["close"], params.get("adx_period", 14))
        ema_f = compute_ema(df["close"], 9)
        ema_s = compute_ema(df["close"], 21)
        for i in range(n):
            adx_val = adx.iloc[i]
            bullish = ema_f.iloc[i] > ema_s.iloc[i]
            if pd.isna(adx_val):
                regime.iloc[i] = "R1"
            elif adx_val > 25 and bullish:
                regime.iloc[i] = "R1"
            elif adx_val > 25:
                regime.iloc[i] = "R4"
            elif bullish:
                regime.iloc[i] = "R2"
            else:
                regime.iloc[i] = "R3"

    elif regime_model == "vrp":
        from apex.regime.vrp_regime import compute_vrp_regime
        # df must already have vix, vxv, vrp_pct columns (merged upstream)
        regime = compute_vrp_regime(df, df["vix"], df["vxv"], df["vrp_pct"])

    else:
        # Default simple "ema" regime: EMA20 vs EMA50 on the execution timeframe
        ema20 = compute_ema(df["close"], 20)
        ema50 = compute_ema(df["close"], 50)
        close = df["close"]
        for i in range(n):
            if pd.isna(ema50.iloc[i]):
                regime.iloc[i] = "R1"
                continue
            above20 = close.iloc[i] > ema20.iloc[i]
            above50 = close.iloc[i] > ema50.iloc[i]
            bull_stack = ema20.iloc[i] > ema50.iloc[i]
            if above20 and above50 and bull_stack:
                regime.iloc[i] = "R1"
            elif above50:
                regime.iloc[i] = "R2"
            elif bull_stack:
                regime.iloc[i] = "R3"
            else:
                regime.iloc[i] = "R4"

    return regime


def compute_entry_score(signals, regime, architecture, params):
    """
    Aggregate individual indicator signals into a composite entry score.

    Aggregation modes:
      - ``"additive"``:  sum of bullish signals (+1 each)
      - ``"weighted"``:  weighted sum using concept weights
      - ``"unanimous"``: all active indicators must agree

    Returns a pd.Series of integer scores.
    """
    active = architecture.get("indicators", [])
    aggregation = architecture.get("score_aggregation", "additive")
    concept_weights = architecture.get("concept_weights", {})
    score = pd.Series(0.0, index=regime.index)

    if aggregation == "additive":
        for name in active:
            if name in signals:
                bullish = (signals[name] == 1).astype(float)
                score = score + bullish

    elif aggregation == "weighted":
        total_weight = 0.0
        for name in active:
            if name in signals:
                w = concept_weights.get(name, 1.0)
                bullish = (signals[name] == 1).astype(float)
                score = score + bullish * w
                total_weight += w
        if total_weight > 0:
            score = score * (len(active) / total_weight)

    elif aggregation == "unanimous":
        all_bull = pd.Series(True, index=regime.index)
        for name in active:
            if name in signals:
                all_bull = all_bull & (signals[name] == 1)
        score = all_bull.astype(float) * len(active)

    # Regime bonus/penalty
    regime_bonus = params.get("regime_bonus", 0)
    if regime_bonus > 0:
        score = score + (regime == "R1").astype(float) * regime_bonus
        score = score + (regime == "R2").astype(float) * (regime_bonus * 0.5)
        score = score - (regime == "R3").astype(float) * (regime_bonus * 0.5)
        score = score - (regime == "R4").astype(float) * regime_bonus

    return score.astype(int)


def run_backtest(df, signals_data, architecture, params):
    """
    Bar-by-bar long-only backtest engine with multiple simultaneous exit methods.

    Iterates through the execution-timeframe DataFrame.  Enters long when the
    composite score >= min_score AND regime != R4.  Tracks MFE/MAE and handles
    the following exit methods simultaneously (first trigger wins):

      - ``"fixed_target"``:  exit at entry + atr_target_mult * ATR
      - ``"fixed_stop"``:    exit at entry - atr_stop_mult * ATR
      - ``"trailing_stop"``: chandelier trail, activates after trail_activate_atr
      - ``"regime_exit"``:   forced exit when regime transitions to R4
      - ``"time_exit"``:     forced exit after max_hold_bars

    Look-ahead safety: the signal at bar i-1 fills at the OPEN of bar i.

    Returns (trades_list, stats_dict).
    """
    min_score = architecture.get("min_score", 5)
    exit_methods = architecture.get("exit_methods", ["trailing_stop", "regime_exit", "time_exit"])

    atr_stop_mult = params.get("atr_stop_mult", 1.5)
    atr_target_mult = params.get("atr_target_mult", 2.5)
    atr_trail_mult = params.get("atr_trail_mult", 1.0)
    trail_activate_atr = params.get("trail_activate_atr", 1.0)
    max_hold_bars = params.get("max_hold_bars", 35)
    commission_pct = params.get("commission_pct", 0.05)

    regime = signals_data["regime"]
    score = signals_data["score"]
    atr = signals_data["atr"]

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    dt = df["datetime"].values

    regime_vals = regime.values
    score_vals = score.values
    atr_vals = atr.values

    # Entry condition: score >= min_score AND regime is not R4
    entry_ok = np.array(
        [(score_vals[i] >= min_score and regime_vals[i] != "R4" and
          not np.isnan(atr_vals[i]) and atr_vals[i] > 0)
         for i in range(len(close))],
        dtype=bool,
    )

    use_fixed_target = "fixed_target" in exit_methods
    use_fixed_stop = "fixed_stop" in exit_methods or "trailing_stop" in exit_methods
    use_trailing = "trailing_stop" in exit_methods
    use_regime_exit = "regime_exit" in exit_methods
    use_time_exit = "time_exit" in exit_methods

    trades = []
    in_pos = False
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    target_price = 0.0
    trail_active = False
    trail_stop = 0.0
    high_since = 0.0
    low_since = 0.0
    bars_held = 0
    entry_idx = 0
    entry_regime = ""
    entry_dt = None
    mfe = 0.0
    mae = 0.0

    for i in range(1, len(close)):
        if not in_pos:
            # Signal from prior bar fills at current bar's open (no look-ahead)
            if entry_ok[i - 1]:
                in_pos = True
                entry_price = open_[i]
                entry_atr = atr_vals[i - 1]
                stop_price = entry_price - atr_stop_mult * entry_atr
                target_price = entry_price + atr_target_mult * entry_atr
                trail_active = False
                trail_stop = stop_price
                high_since = high[i]
                low_since = low[i]
                bars_held = 0
                entry_idx = i
                entry_regime = regime_vals[i - 1]
                entry_dt = dt[i]
                mfe = 0.0
                mae = 0.0
        else:
            bars_held += 1

            # Track MFE / MAE
            if high[i] > high_since:
                high_since = high[i]
            if low[i] < low_since:
                low_since = low[i]

            fav_pnl_pct = (high_since - entry_price) / entry_price * 100.0
            adv_pnl_pct = (low_since - entry_price) / entry_price * 100.0

            if fav_pnl_pct > mfe:
                mfe = fav_pnl_pct
            if adv_pnl_pct < mae:
                mae = adv_pnl_pct

            exit_reason = None

            # 1) Fixed stop
            if use_fixed_stop and exit_reason is None:
                if low[i] <= stop_price:
                    exit_reason = "fixed_stop"

            # 2) Fixed target
            if use_fixed_target and exit_reason is None:
                if high[i] >= target_price:
                    exit_reason = "fixed_target"

            # 3) Trailing stop (chandelier)
            if use_trailing and exit_reason is None:
                gain_in_atr = (close[i] - entry_price) / entry_atr if entry_atr > 0 else 0.0
                if gain_in_atr >= trail_activate_atr:
                    trail_active = True
                if trail_active:
                    new_trail = high_since - atr_trail_mult * entry_atr
                    if new_trail > trail_stop:
                        trail_stop = new_trail
                    if low[i] <= trail_stop:
                        exit_reason = "trailing_stop"

            # 4) Regime exit
            if use_regime_exit and exit_reason is None:
                if regime_vals[i] == "R4":
                    exit_reason = "regime_exit"

            # 5) Time exit
            if use_time_exit and exit_reason is None:
                if bars_held >= max_hold_bars:
                    exit_reason = "time_exit"

            if exit_reason is not None:
                if exit_reason == "fixed_target":
                    exit_price = target_price
                elif exit_reason == "fixed_stop":
                    exit_price = stop_price
                elif exit_reason == "trailing_stop":
                    exit_price = trail_stop
                else:
                    exit_price = close[i]

                # Clamp exit price to bar range
                exit_price = max(low[i], min(high[i], exit_price))

                pnl_pct = (exit_price - entry_price) / entry_price * 100.0
                net_pnl_pct = pnl_pct - 2.0 * commission_pct

                trades.append({
                    "entry_datetime": str(entry_dt),
                    "exit_datetime": str(dt[i]),
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "pnl_pct": round(net_pnl_pct, 4),
                    "gross_pnl_pct": round(pnl_pct, 4),
                    "mfe": round(mfe, 4),
                    "mae": round(mae, 4),
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                    "entry_regime": entry_regime,
                    "entry_atr": round(entry_atr, 4),
                    "entry_score": int(score_vals[entry_idx]),
                    "direction": "long",
                })
                in_pos = False

    stats = compute_stats(trades)
    return trades, stats


def compute_stats(trades):
    """
    Compute comprehensive performance statistics from a list of trade dicts.

    Returns dict with: trades, pf, wr_pct, total_return_pct, max_dd_pct,
    sharpe, sortino, edge_ratio, avg_bars_held, avg_pnl, avg_win, avg_loss,
    largest_win, largest_loss, exit_reason_counts, per-regime trade counts.
    """
    if not trades:
        return {
            "trades": 0, "pf": 0.0, "wr_pct": 0.0,
            "total_return_pct": 0.0, "max_dd_pct": 0.0,
            "sharpe": 0.0, "sortino": 0.0, "edge_ratio": 0.0,
            "avg_bars_held": 0.0, "avg_pnl": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0,
            "largest_win": 0.0, "largest_loss": 0.0,
            "regime_exit_count": 0, "exit_reason_counts": {},
            "r1_trades": 0, "r2_trades": 0, "r3_trades": 0, "r4_trades": 0,
        }

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    if gross_loss < 0.5 or len(losses) < 3:
        pf = min(gross_profit / max(gross_loss, 0.5), 10.0)
    else:
        pf = gross_profit / gross_loss
    pf = min(pf, 10.0)
    wr = len(wins) / len(pnls) * 100.0

    # Equity curve and drawdown
    equity = 10000.0
    peak_eq = equity
    max_dd = 0.0
    for p in pnls:
        equity *= (1.0 + p / 100.0)
        if equity > peak_eq:
            peak_eq = equity
        dd = (peak_eq - equity) / peak_eq * 100.0
        if dd > max_dd:
            max_dd = dd

    total_return = (equity / 10000.0 - 1.0) * 100.0

    pnl_arr = np.array(pnls)
    mean_pnl = float(np.mean(pnl_arr))
    std_pnl = float(np.std(pnl_arr)) if len(pnl_arr) > 1 else 0.001
    if std_pnl < 0.001:
        std_pnl = 0.001
    sharpe = (mean_pnl / std_pnl) * math.sqrt(min(len(trades), 250))

    downside = pnl_arr[pnl_arr < 0]
    if len(downside) > 1:
        downside_std = float(np.std(downside))
        if downside_std < 0.001:
            downside_std = 0.001
        sortino = (mean_pnl / downside_std) * math.sqrt(min(len(trades), 250))
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0.0

    mfes = [t["mfe"] for t in trades]
    maes = [abs(t["mae"]) for t in trades]
    mean_mae = float(np.mean(maes)) if maes else 0.001
    if mean_mae < 0.001:
        mean_mae = 0.001
    edge_ratio = float(np.mean(mfes)) / mean_mae

    avg_bars = float(np.mean([t["bars_held"] for t in trades]))
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    largest_win = max(pnls) if pnls else 0.0
    largest_loss = min(pnls) if pnls else 0.0

    exit_counts = {}
    for t in trades:
        reason = t.get("exit_reason", "unknown")
        exit_counts[reason] = exit_counts.get(reason, 0) + 1

    regime_exit_count = exit_counts.get("regime_exit", 0)

    r1 = sum(1 for t in trades if t.get("entry_regime") == "R1")
    r2 = sum(1 for t in trades if t.get("entry_regime") == "R2")
    r3 = sum(1 for t in trades if t.get("entry_regime") == "R3")
    r4 = sum(1 for t in trades if t.get("entry_regime") == "R4")

    return {
        "trades": len(trades),
        "pf": round(pf, 3),
        "wr_pct": round(wr, 2),
        "total_return_pct": round(total_return, 2),
        "max_dd_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "edge_ratio": round(edge_ratio, 3),
        "avg_bars_held": round(avg_bars, 1),
        "avg_pnl": round(mean_pnl, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "largest_win": round(largest_win, 4),
        "largest_loss": round(largest_loss, 4),
        "regime_exit_count": regime_exit_count,
        "exit_reason_counts": exit_counts,
        "r1_trades": r1,
        "r2_trades": r2,
        "r3_trades": r3,
        "r4_trades": r4,
    }


def full_backtest(df, daily_df, architecture, params):
    """
    End-to-end single-pass backtest.

    1. Compute indicator signals
    2. Compute regime
    3. Compute entry score
    4. Run bar-by-bar backtest

    Returns (trades, stats).
    """
    signals = compute_indicator_signals(df, architecture, params)
    atr = signals.pop("_atr")

    regime_model = architecture.get("regime_model", "ema")
    regime = compute_regime(df, daily_df, regime_model, params)

    score = compute_entry_score(signals, regime, architecture, params)

    signals_data = {
        "signals": signals,
        "regime": regime,
        "score": score,
        "atr": atr,
    }

    return run_backtest(df, signals_data, architecture, params)


DEFAULT_ARCHITECTURE = {
    "indicators": ["RSI", "Keltner", "VolumeSurge", "MACD", "EMA_Cross", "VWAP"],
    "min_score": 4,
    "exit_methods": ["fixed_target", "fixed_stop", "trailing_stop", "regime_exit", "time_exit"],
    "regime_model": "ema",
    "position_sizing": "equal",
    "exec_timeframe": "1H",
    "score_aggregation": "additive",
    "concept_weights": {},
}

DEFAULT_PARAMS = {
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "boll_period": 20,
    "boll_std": 2.0,
    "stoch_k": 14,
    "stoch_d": 3,
    "obv_ma_period": 20,
    "adx_period": 14,
    "adx_threshold": 25,
    "cci_period": 20,
    "cci_oversold": -100,
    "cci_overbought": 100,
    "willr_period": 14,
    "willr_oversold": -80,
    "willr_overbought": -20,
    "keltner_period": 20,
    "keltner_mult": 2.0,
    "volume_surge_ma": 20,
    "volume_surge_mult": 1.5,
    "ema_fast": 9,
    "ema_slow": 21,
    "atr_period": 14,
    "atr_stop_mult": 1.5,
    "atr_target_mult": 2.5,
    "atr_trail_mult": 1.0,
    "trail_activate_atr": 1.0,
    "max_hold_bars": 35,
    "commission_pct": 0.05,
    "regime_bonus": 0,
}
