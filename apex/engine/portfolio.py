"""Correlation filter, basket alignment, and full final backtest (tune + true holdout)."""

import pandas as pd

from apex.logging_util import log
from apex.engine.backtest import full_backtest, compute_stats
from apex.engine.strategy_backtest import strategy_full_backtest
from apex.util.sector_map import SECTOR_MAP
from apex.config import FORCED_SYMBOLS
from apex.data.polygon_client import fetch_daily


def compute_basket_alignment(basket, as_of, short_days=21,
                              long_days=63, alignment_threshold=3,
                              size_multiplier=1.25):
    """
    Blended momentum = 0.5 * ret_short + 0.5 * ret_long per symbol.

    Uses data STRICTLY before as_of (look-ahead safe).
    If max(positive_count, negative_count) >= alignment_threshold -> return size_multiplier.
    Else -> return 1.0.
    """
    positive_count = 0
    negative_count = 0

    for sym, df in basket.items():
        # Use only data strictly before as_of
        mask = df["datetime"] < as_of
        hist = df.loc[mask]

        if len(hist) < long_days + 1:
            continue

        closes = hist["close"].values

        # Short-term return: last short_days
        if len(closes) >= short_days + 1:
            ret_short = (closes[-1] / closes[-(short_days + 1)]) - 1.0
        else:
            ret_short = 0.0

        # Long-term return: last long_days
        if len(closes) >= long_days + 1:
            ret_long = (closes[-1] / closes[-(long_days + 1)]) - 1.0
        else:
            ret_long = 0.0

        blended = 0.5 * ret_short + 0.5 * ret_long

        if blended > 0:
            positive_count += 1
        elif blended < 0:
            negative_count += 1

    if max(positive_count, negative_count) >= alignment_threshold:
        return size_multiplier
    return 1.0


def correlation_filter(validated_results, cfg):
    """
    Filter validated symbols by pairwise return correlation and sector caps.

    - Reject one of any pair with trade-return correlation > max_corr
    - Max N symbols per sector (using SECTOR_MAP)
    """
    max_corr = cfg.get("optimization", {}).get("max_correlation", 0.70)
    max_per_sector = cfg.get("optimization", {}).get("max_per_sector", 3)
    log(f"=== CORRELATION FILTER (max_corr={max_corr}, max_sector={max_per_sector}) ===")

    if len(validated_results) <= 1:
        return validated_results

    syms = list(validated_results.keys())
    syms.sort(key=lambda s: validated_results[s].get("fitness", 0), reverse=True)

    return_series = {}
    for sym in syms:
        pnls = validated_results[sym].get("trade_pnls", [])
        if pnls:
            return_series[sym] = pd.Series(pnls)

    rejected = set()
    sym_list = [s for s in syms if s in return_series]
    for i in range(len(sym_list)):
        if sym_list[i] in rejected:
            continue
        for j in range(i + 1, len(sym_list)):
            if sym_list[j] in rejected:
                continue
            s1 = return_series[sym_list[i]]
            s2 = return_series[sym_list[j]]
            min_len = min(len(s1), len(s2))
            if min_len < 10:
                continue
            corr = float(s1.iloc[:min_len].corr(s2.iloc[:min_len]))
            if abs(corr) > max_corr:
                loser = sym_list[j]
                if loser in FORCED_SYMBOLS:
                    loser = sym_list[i]
                    if loser in FORCED_SYMBOLS:
                        continue
                rejected.add(loser)
                log(f"  Correlation filter: {sym_list[i]} vs {loser} = {corr:.3f} -> reject {loser}")

    sector_counts = {}
    sector_rejected = set()
    for sym in syms:
        if sym in rejected:
            continue
        sector = SECTOR_MAP.get(sym, "Unknown")
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if sector_counts[sector] > max_per_sector and sym not in FORCED_SYMBOLS:
            sector_rejected.add(sym)
            log(f"  Sector cap: {sym} ({sector}) -> rejected (already {max_per_sector} in sector)")

    all_rejected = rejected | sector_rejected
    final = {s: v for s, v in validated_results.items() if s not in all_rejected}
    log(f"Correlation filter: {len(final)}/{len(validated_results)} survived")
    return final


def phase_full_backtest(data_dict, architecture, final_results, cfg, tuned_results=None, strategy_adapter=None):
    """
    Re-run backtest with final params on the FULL tune window AND the held-out
    final window that no optimization layer has ever touched.

    Backtests EVERY tuned symbol (not just the survivors of the correlation
    filter) so headline stats reflect the full optimized universe and aren't
    inflated by post-selection bias.  The survivor subset is reported
    separately.  The TRUE HOLDOUT block is the honest performance number.
    """
    log("=== PHASE: Full Final Backtest ===")
    all_trades = []
    per_symbol = {}
    universe_source = tuned_results if tuned_results else final_results
    survivor_set = set(final_results.keys())

    sorted_syms = sorted(universe_source.keys(),
                         key=lambda s: universe_source[s].get("fitness", 0), reverse=True)

    for sym in sorted_syms:
        sym_data = data_dict.get(sym, {})
        df = sym_data.get("exec_df")
        daily_df = sym_data.get("daily_df")
        params = universe_source[sym]["params"]

        if df is None or len(df) < 100:
            continue

        if strategy_adapter is not None:
            # Re-apply per-symbol tuned params before backtesting
            if params:
                strategy_adapter.set_params(params)
            spy_data = data_dict.get("SPY", data_dict.get("_spy_data", {}))
            spy_df = spy_data.get("exec_df") if isinstance(spy_data, dict) else None
            trades, stats = strategy_full_backtest(strategy_adapter, df, spy_df, sym)
            if params:
                strategy_adapter.reset_params()
        else:
            trades, stats = full_backtest(df, daily_df, architecture, params)

        survived = sym in survivor_set
        for t in trades:
            t["symbol"] = sym
            t["survived"] = survived
            t["phase"] = "tune"
        all_trades.extend(trades)

        # True OOS backtest on the never-seen final holdout window
        holdout_df = sym_data.get("exec_df_holdout")
        holdout_daily = sym_data.get("daily_df_holdout")
        holdout_trades, holdout_stats = [], {}
        if holdout_df is not None and len(holdout_df) >= 50:
            if strategy_adapter is not None:
                if params:
                    strategy_adapter.set_params(params)
                spy_data = data_dict.get("SPY", data_dict.get("_spy_data", {}))
                spy_df = spy_data.get("exec_df_holdout") if isinstance(spy_data, dict) else None
                holdout_trades, holdout_stats = strategy_full_backtest(
                    strategy_adapter, holdout_df, spy_df, sym
                )
                if params:
                    strategy_adapter.reset_params()
            else:
                holdout_trades, holdout_stats = full_backtest(
                    holdout_df, holdout_daily, architecture, params
                )
            for t in holdout_trades:
                t["symbol"] = sym
                t["survived"] = survived
                t["phase"] = "holdout"

        per_symbol[sym] = {"trades": trades, "stats": stats, "params": params,
                           "survived": survived,
                           "holdout_trades": holdout_trades,
                           "holdout_stats": holdout_stats}
        log(f"  {sym}{' [SURVIVOR]' if survived else ''}: tune={stats['trades']}tr "
            f"PF={stats['pf']:.2f} Ret={stats['total_return_pct']:.1f}% | "
            f"holdout={holdout_stats.get('trades', 0)}tr "
            f"PF={holdout_stats.get('pf', 0):.2f} Ret={holdout_stats.get('total_return_pct', 0):.1f}%")

    all_trades.sort(key=lambda t: t["entry_datetime"])

    # Build portfolio equity curve (equal weight per trade)
    n_symbols = max(1, len(sorted_syms))
    equity = 10000.0
    peak = equity
    max_dd = 0.0
    equity_dates = []
    equity_values = []
    for t in all_trades:
        weight = 1.0 / n_symbols
        pnl_pct = t["pnl_pct"] * weight
        equity *= (1.0 + pnl_pct / 100.0)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100.0
        if dd > max_dd:
            max_dd = dd
        equity_dates.append(t["exit_datetime"])
        equity_values.append(equity)

    # Fetch SPY for benchmark comparison
    spy_sym, spy_df, spy_status = fetch_daily("SPY")
    benchmark = None
    if spy_df is not None and len(spy_df) > 50:
        spy_returns = spy_df["close"].pct_change().dropna()
        spy_equity = 10000.0
        spy_curve = [spy_equity]
        for r in spy_returns:
            spy_equity *= (1.0 + r)
            spy_curve.append(spy_equity)
        benchmark = {
            "dates": spy_df["datetime"].tolist(),
            "equity": spy_curve,
            "total_return_pct": round((spy_equity / 10000.0 - 1.0) * 100.0, 2),
        }

    portfolio_stats = compute_stats(all_trades)
    if equity_values:
        port_return = (equity_values[-1] / 10000.0 - 1.0) * 100.0
        portfolio_stats["total_return_pct"] = round(port_return, 2)
        portfolio_stats["max_dd_pct"] = round(max_dd, 2)
    survivor_trades = [t for t in all_trades if t.get("survived")]
    survivor_stats = compute_stats(survivor_trades) if survivor_trades else {}

    holdout_all_trades = []
    for sym in per_symbol:
        holdout_all_trades.extend(per_symbol[sym].get("holdout_trades", []))
    holdout_universe_stats = compute_stats(holdout_all_trades) if holdout_all_trades else {}
    holdout_survivor_trades = [t for t in holdout_all_trades if t.get("survived")]
    holdout_survivor_stats = compute_stats(holdout_survivor_trades) if holdout_survivor_trades else {}

    results = {
        "all_trades": all_trades,
        "survivor_trades": survivor_trades,
        "holdout_all_trades": holdout_all_trades,
        "holdout_survivor_trades": holdout_survivor_trades,
        "per_symbol": per_symbol,
        "sorted_syms": sorted_syms,
        "survivor_syms": sorted([s for s in sorted_syms if per_symbol.get(s, {}).get("survived")]),
        "portfolio_stats": portfolio_stats,
        "survivor_stats": survivor_stats,
        "holdout_universe_stats": holdout_universe_stats,
        "holdout_survivor_stats": holdout_survivor_stats,
        "equity_dates": equity_dates,
        "equity_values": equity_values,
        "max_dd": round(max_dd, 2),
        "benchmark": benchmark,
    }
    log(f"TUNE Universe ({len(sorted_syms)} syms): {len(all_trades)} trades, "
        f"PF={portfolio_stats['pf']:.2f}, MaxDD={max_dd:.1f}%")
    if survivor_stats:
        log(f"TUNE Survivors ({len(survivor_set)} syms): {len(survivor_trades)} trades, "
            f"PF={survivor_stats.get('pf', 0):.2f} [post-selection, biased]")
    if holdout_universe_stats:
        log(f"HOLDOUT Universe: {len(holdout_all_trades)} trades, "
            f"PF={holdout_universe_stats.get('pf', 0):.2f}, "
            f"WR={holdout_universe_stats.get('wr_pct', 0):.1f}%, "
            f"Ret={holdout_universe_stats.get('total_return_pct', 0):.1f}% "
            f"[TRUE OOS - never seen by optimizer]")
    if holdout_survivor_stats:
        log(f"HOLDOUT Survivors: {len(holdout_survivor_trades)} trades, "
            f"PF={holdout_survivor_stats.get('pf', 0):.2f}, "
            f"Ret={holdout_survivor_stats.get('total_return_pct', 0):.1f}% "
            f"[TRUE OOS - survivors only]")
    return results
