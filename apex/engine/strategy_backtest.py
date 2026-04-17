"""Bar-by-bar backtest engine that runs a user strategy's exact entry/exit logic."""

import numpy as np
import pandas as pd

from apex.engine.backtest import compute_stats


def run_strategy_backtest(adapter, prepared_df, sym, commission_pct=0.05):
    """
    Run a user strategy's entry_fn/exit_fn bar-by-bar on a prepared DataFrame.

    Returns (trades, stats) in the same format as apex.engine.backtest.run_backtest()
    so all downstream pipeline components (Layer 3, reports, etc.) work unchanged.

    Parameters
    ----------
    adapter : StrategyAdapter
        The loaded strategy adapter with entry_fn and exit_fn.
    prepared_df : pd.DataFrame
        DataFrame with user-convention columns (Close, High, Low, ATR, etc.)
        as produced by prepare_strategy_dataframe().
    sym : str
        Ticker symbol.
    commission_pct : float
        Round-trip commission in percent.

    Returns
    -------
    trades : list[dict]
        Each trade has: entry_datetime, exit_datetime, entry_price, exit_price,
        pnl_pct, gross_pnl_pct, mfe, mae, bars_held, exit_reason,
        entry_regime, entry_atr, entry_score, direction.
    stats : dict
        Performance statistics from compute_stats().
    """
    df = prepared_df
    if len(df) < 3:
        return [], compute_stats([])

    trades = []
    position = None

    for idx in range(2, len(df)):
        r = df.iloc[idx]
        prev = df.iloc[idx - 1]
        prev2 = df.iloc[idx - 2]

        atr_val = r.get("ATR", 0)
        if pd.isna(atr_val) or atr_val == 0:
            continue

        # --- Check exit first ---
        if position is not None:
            try:
                should_exit, exit_price, reason = adapter.exit_fn(r, prev, position, df, idx)
            except Exception:
                should_exit, exit_price, reason = False, 0, ""

            if should_exit:
                entry_price = position["ep"]
                bars_held = idx - position["entry_idx"]
                if position.get("dir", "L") == "L":
                    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100.0
                net_pnl_pct = pnl_pct - 2.0 * commission_pct

                # MFE / MAE from tracked extremes
                highest = position.get("highest", entry_price)
                lowest = position.get("lowest", entry_price)
                if position.get("dir", "L") == "L":
                    mfe = (highest - entry_price) / entry_price * 100.0
                    mae = (lowest - entry_price) / entry_price * 100.0
                else:
                    mfe = (entry_price - lowest) / entry_price * 100.0
                    mae = (entry_price - highest) / entry_price * 100.0

                trades.append({
                    "entry_datetime": str(position.get("entry_date", "")),
                    "exit_datetime": str(r.get("Date", "")),
                    "entry_idx": position["entry_idx"],
                    "exit_idx": idx,
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "pnl_pct": round(net_pnl_pct, 4),
                    "gross_pnl_pct": round(pnl_pct, 4),
                    "mfe": round(mfe, 4),
                    "mae": round(mae, 4),
                    "bars_held": bars_held,
                    "exit_reason": reason,
                    "entry_regime": "R1",
                    "entry_atr": round(position.get("entry_atr", 0), 4),
                    "entry_score": int(position.get("score", 0)),
                    "direction": "long" if position.get("dir", "L") == "L" else "short",
                })
                position = None
                continue

            # Update trailing highs/lows
            if position.get("dir", "L") == "L":
                if r["High"] > position.get("highest", position["ep"]):
                    position["highest"] = r["High"]
            else:
                if r["Low"] < position.get("lowest", position["ep"]):
                    position["lowest"] = r["Low"]

        # --- Check entry ---
        if position is None:
            try:
                sig = adapter.entry_fn(r, prev, prev2, sym, df, idx)
            except Exception:
                sig = None

            if sig is not None:
                position = {
                    "dir": sig.get("dir", "L"),
                    "ep": sig["price"],
                    "shares": 1,
                    "orig_shares": 1,
                    "stop": sig.get("stop", sig["price"] * 0.95),
                    "highest": sig["price"],
                    "lowest": sig["price"],
                    "entry_atr": sig.get("atr", atr_val),
                    "entry_date": sig.get("date", r.get("Date")),
                    "entry_idx": idx,
                    "score": sig.get("score", 0),
                    "pyramided": False,
                    "be_hit": False,
                    "tp_hit": False,
                }

    # Close open position at end of data
    if position is not None and len(df) > 0:
        last = df.iloc[-1]
        exit_price = last["Close"]
        entry_price = position["ep"]
        bars_held = len(df) - 1 - position["entry_idx"]
        if position.get("dir", "L") == "L":
            pnl_pct = (exit_price - entry_price) / entry_price * 100.0
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100.0
        net_pnl_pct = pnl_pct - 2.0 * commission_pct

        highest = position.get("highest", entry_price)
        lowest = position.get("lowest", entry_price)
        if position.get("dir", "L") == "L":
            mfe = (highest - entry_price) / entry_price * 100.0
            mae = (lowest - entry_price) / entry_price * 100.0
        else:
            mfe = (entry_price - lowest) / entry_price * 100.0
            mae = (entry_price - highest) / entry_price * 100.0

        trades.append({
            "entry_datetime": str(position.get("entry_date", "")),
            "exit_datetime": str(last.get("Date", "")),
            "entry_idx": position["entry_idx"],
            "exit_idx": len(df) - 1,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "pnl_pct": round(net_pnl_pct, 4),
            "gross_pnl_pct": round(pnl_pct, 4),
            "mfe": round(mfe, 4),
            "mae": round(mae, 4),
            "bars_held": bars_held,
            "exit_reason": "end_of_data",
            "entry_regime": "R1",
            "entry_atr": round(position.get("entry_atr", 0), 4),
            "entry_score": int(position.get("score", 0)),
            "direction": "long" if position.get("dir", "L") == "L" else "short",
        })

    stats = compute_stats(trades)
    return trades, stats


def strategy_full_backtest(adapter, polygon_df, spy_df, sym, commission_pct=0.05):
    """
    End-to-end: prepare DataFrame + run strategy backtest.

    This is the strategy-mode equivalent of backtest.full_backtest().
    """
    prepared = adapter.prepare_df(polygon_df, spy_df=spy_df, sym=sym)
    return run_strategy_backtest(adapter, prepared, sym, commission_pct=commission_pct)
