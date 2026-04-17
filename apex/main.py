"""Main pipeline entry point."""

import argparse
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from apex.config import CFG, OUTPUT_DIR, FORCED_SYMBOLS, load_config
from apex.logging_util import log
from apex.data.polygon_client import fetch_daily, fetch_bars
from apex.engine.backtest import full_backtest, DEFAULT_ARCHITECTURE, DEFAULT_PARAMS
from apex.engine.portfolio import correlation_filter, phase_full_backtest
from apex.optimize.layer1 import layer1_architecture_search
from apex.optimize.layer2 import layer2_deep_tune
from apex.optimize.layer3 import layer3_robustness_gauntlet
from apex.report.html_report import generate_html_report
from apex.report.csv_json import generate_trades_csv, generate_summary_csv, generate_parameters_json
from apex.report.amibroker import generate_apex_afl, push_to_amibroker
from apex.util.concept_parser import parse_concept
from apex.util.sector_map import SECTOR_MAP
from apex.util.checkpoints import save_checkpoint, load_checkpoint


def phase1_universe(cfg):
    """Return candidate symbol list from config plus forced tickers."""
    target = cfg.get("target_symbols", [])
    if not target:
        target = list(SECTOR_MAP.keys())[:30]
    for sym in FORCED_SYMBOLS:
        if sym not in target:
            target.append(sym)
    log(f"Phase 1: Universe of {len(target)} candidates (includes forced: {FORCED_SYMBOLS})")
    return target


def phase2_quick_screen(candidates, cfg):
    """
    Quick screen: fetch daily data, filter by liquidity / price / volume.

    Returns (list of surviving symbols, dict of {sym: daily_df}).
    """
    uni = cfg.get("universe", {})
    min_price = uni.get("min_price", 10)
    max_price = uni.get("max_price", 5000)
    min_volume = uni.get("min_avg_volume", 500000)
    min_bars = uni.get("min_daily_bars", 252)

    log(f"Phase 2: Quick screen on {len(candidates)} symbols "
        f"(price ${min_price}-${max_price}, vol>{min_volume/1e6:.1f}M)")

    survivors = []
    daily_data = {}

    for idx, sym in enumerate(candidates, 1):
        forced = sym in FORCED_SYMBOLS
        sym_name, df, status = fetch_daily(sym)
        if df is None or len(df) < min(min_bars, 50):
            log(f"  [{idx}/{len(candidates)}] {sym}: SKIP ({status}, bars={len(df) if df is not None else 0})"
                + (" [FORCED but no data]" if forced else ""))
            continue

        avg_price = df["close"].iloc[-20:].mean() if len(df) >= 20 else df["close"].mean()
        avg_volume = df["volume"].iloc[-20:].mean() if len(df) >= 20 else df["volume"].mean()

        if not forced:
            if avg_price < min_price or avg_price > max_price:
                log(f"  [{idx}/{len(candidates)}] {sym}: SKIP (price ${avg_price:.2f})")
                continue
            if avg_volume < min_volume:
                log(f"  [{idx}/{len(candidates)}] {sym}: SKIP (volume {avg_volume/1e6:.1f}M)")
                continue

        survivors.append(sym)
        daily_data[sym] = df
        tag = " [FORCED]" if forced else ""
        log(f"  [{idx}/{len(candidates)}] {sym}: PASS{tag} (${avg_price:.0f}, "
            f"vol={avg_volume/1e6:.1f}M, bars={len(df)})")

    log(f"Phase 2 complete: {len(survivors)}/{len(candidates)} survived")
    return survivors, daily_data


def phase3_fetch_data(survivors, daily_data, cfg):
    """
    Fetch execution-timeframe data for each survivor and split off a
    final holdout window that no optimizer will ever see.
    """
    exec_tf = cfg.get("phase3_params", {}).get("exec_timeframe", "1H")
    log(f"Phase 3: Fetching {exec_tf} data for {len(survivors)} symbols")

    data_dict = {}
    for idx, sym in enumerate(survivors, 1):
        log(f"  [{idx}/{len(survivors)}] Fetching {sym}...")

        _, exec_df_full, status = fetch_bars(sym, timeframe=exec_tf)
        if exec_df_full is None or len(exec_df_full) < 100:
            log(f"    {sym}: SKIP (exec bars: {len(exec_df_full) if exec_df_full is not None else 0})")
            continue

        daily_df_full = daily_data.get(sym)

        # Reserve the final N% as a true holdout that NO optimizer ever sees.
        # Layer 1 / Layer 2 / robustness all run on the tune window only.
        holdout_pct = cfg.get("optimization", {}).get("final_holdout_pct", 0.25)
        cut = int(len(exec_df_full) * (1.0 - holdout_pct))
        exec_df = exec_df_full.iloc[:cut].reset_index(drop=True)
        exec_df_holdout = exec_df_full.iloc[cut:].reset_index(drop=True)

        if daily_df_full is not None and len(exec_df) > 0:
            split_dt = exec_df["datetime"].iloc[-1]
            daily_df = daily_df_full[daily_df_full["datetime"] <= split_dt].reset_index(drop=True)
            daily_df_holdout = daily_df_full[daily_df_full["datetime"] > split_dt].reset_index(drop=True)
        else:
            daily_df = daily_df_full
            daily_df_holdout = None

        data_dict[sym] = {
            "exec_df": exec_df,
            "daily_df": daily_df,
            "exec_df_holdout": exec_df_holdout,
            "daily_df_holdout": daily_df_holdout,
        }
        log(f"    {sym}: OK (tune={len(exec_df)} bars, holdout={len(exec_df_holdout)} bars)")

    log(f"Phase 3 complete: {len(data_dict)} symbols with full data")
    return data_dict


def main():
    """
    Pipeline entry point.

      1. Universe selection
      2. Quick screen (daily liquidity)
      3. Multi-TF data fetch + holdout split
      4. Layer 1: Architecture search
      5. Layer 2: Deep parameter optimization
      6. Layer 3: Robustness gauntlet
      7. Correlation filter
      8. Full final backtest (tune + true holdout)
      9. Report generation (HTML + CSV + JSON)
     10. AmiBroker push
     11. Open report in browser
    """
    parser = argparse.ArgumentParser(description="Optuna Screener Pipeline")
    parser.add_argument("--config", default="apex_config.json", help="Config JSON path")
    parser.add_argument("--test", action="store_true", help="Test mode: 3 symbols, light budget")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--no-amibroker", action="store_true", help="Skip AmiBroker push")
    parser.add_argument("--concept", type=str, default="", help="Strategy concept string")
    parser.add_argument("--budget", type=str, default="medium",
                        choices=["light", "medium", "heavy"], help="Compute budget")
    parser.add_argument("--output", type=str, default="", help="Output directory override")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply budget profile
    budget = cfg.get("budget_profiles", {}).get(args.budget, {})
    opt = cfg.get("optimization", {})
    rob = cfg.get("robustness", {})
    if budget:
        opt["arch_trials"] = budget.get("arch_trials", 200)
        opt["inner_trials"] = budget.get("arch_trials", 200) // 5
        opt["deep_trials"] = budget.get("deep_trials", 800)
        rob["monte_carlo_sims"] = budget.get("mc_sims", 3000)
    else:
        opt.setdefault("arch_trials", 200)
        opt.setdefault("inner_trials", 40)
        opt.setdefault("deep_trials", 800)
    opt["robustness_threshold"] = rob.get("min_robustness_score", 0.5)
    cfg["optimization"] = opt
    cfg["robustness"] = rob
    log(f"Budget: {args.budget} - arch={opt['arch_trials']} trials, "
        f"deep={opt['deep_trials']}/symbol, MC={rob.get('monte_carlo_sims', 3000)} sims, "
        f"robustness>={opt['robustness_threshold']}")

    opt = cfg.get("optimization", {})
    if args.test:
        log("*** TEST MODE: light budget, 3 symbols ***")
        opt["arch_trials"] = 5
        opt["inner_trials"] = 10
        opt["deep_trials"] = 20
        opt["robustness_threshold"] = 0.2
        cfg["optimization"] = opt
        target = cfg.get("target_symbols", list(SECTOR_MAP.keys())[:3])
        cfg["target_symbols"] = target[:3]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        run_output = Path(args.output)
    else:
        run_output = OUTPUT_DIR / f"run_{timestamp}"
    run_output.mkdir(parents=True, exist_ok=True)

    run_info = {
        "timestamp": timestamp,
        "concept": args.concept or cfg.get("concept", "adaptive multi-indicator"),
        "test_mode": args.test,
        "config": args.config,
    }

    log("=" * 60)
    log("Optuna Screener Pipeline")
    log("=" * 60)
    log(f"Output: {run_output}")
    log(f"Concept: {run_info['concept']}")

    concept_bias = parse_concept(run_info["concept"])
    log(f"Concept bias: { {k: v for k, v in concept_bias.items() if v != 1.0} }")

    resume_stage = None
    if args.resume:
        for stage in ["layer3_robustness", "layer2_tuned", "layer1_architecture"]:
            cp = load_checkpoint(stage, str(run_output))
            if cp is not None:
                resume_stage = stage
                break
        if resume_stage:
            log(f"Resuming from checkpoint: {resume_stage}")
        else:
            log("No checkpoint found, starting fresh")

    # ---- Phase 1 ----
    candidates = phase1_universe(cfg)

    # ---- Phase 2 ----
    survivors, daily_data = phase2_quick_screen(candidates, cfg)
    if not survivors:
        log("No symbols survived Phase 2. Exiting.", "ERROR")
        sys.exit(1)

    # ---- Phase 3 ----
    data_dict = phase3_fetch_data(survivors, daily_data, cfg)
    if not data_dict:
        log("No symbols have sufficient data. Exiting.", "ERROR")
        sys.exit(1)
    survivors = list(data_dict.keys())

    # ---- Layer 1 ----
    if resume_stage in ("layer2_tuned", "layer3_robustness"):
        cp = load_checkpoint("layer1_architecture", str(run_output))
        architecture = cp["architecture"] if cp else DEFAULT_ARCHITECTURE
    else:
        architecture = layer1_architecture_search(data_dict, concept_bias, cfg)

    # ---- Layer 2 ----
    if resume_stage == "layer3_robustness":
        cp = load_checkpoint("layer2_tuned", str(run_output))
        tuned_results = cp if cp else {}
        for sym, sym_result in tuned_results.items():
            if "trades" not in sym_result and sym in data_dict:
                sd = data_dict[sym]
                params = sym_result.get("params", DEFAULT_PARAMS)
                trades, stats = full_backtest(sd["exec_df"], sd["daily_df"],
                                              architecture, params)
                sym_result["trades"] = trades
                sym_result["trade_pnls"] = [t["pnl_pct"] for t in trades]
                sym_result["stats"] = stats
    else:
        tuned_results = layer2_deep_tune(data_dict, architecture, survivors, cfg)

    if not tuned_results:
        log("No symbols survived Layer 2. Exiting.", "ERROR")
        sys.exit(1)

    # ---- Layer 3 ----
    validated_results, robustness_data = layer3_robustness_gauntlet(
        data_dict, architecture, tuned_results, cfg
    )

    if not validated_results:
        log("No symbols passed robustness gauntlet. Relaxing threshold...", "WARN")
        sorted_by_composite = sorted(
            robustness_data.items(),
            key=lambda x: x[1].get("composite", 0),
            reverse=True
        )
        for sym, rd in sorted_by_composite[:5]:
            if sym in tuned_results:
                validated_results[sym] = tuned_results[sym]
                validated_results[sym]["robustness"] = rd
        if not validated_results:
            log("Still no symbols. Using best tuned results.", "WARN")
            best_sym = max(tuned_results, key=lambda s: tuned_results[s].get("fitness", 0))
            validated_results = {best_sym: tuned_results[best_sym]}
            validated_results[best_sym]["robustness"] = robustness_data.get(best_sym, {})

    # ---- Correlation Filter ----
    final_results = correlation_filter(validated_results, cfg)

    if not final_results:
        log("All symbols filtered out. Using validated results.", "WARN")
        final_results = validated_results

    # ---- Final Backtest ----
    backtest_results = phase_full_backtest(data_dict, architecture, final_results, cfg,
                                           tuned_results=tuned_results)

    # ---- Report Generation ----
    log("=== GENERATING REPORTS ===")
    report_path = generate_html_report(
        backtest_results, architecture, robustness_data, run_info, str(run_output)
    )
    generate_trades_csv(backtest_results.get("all_trades", []), str(run_output))
    generate_summary_csv(backtest_results, str(run_output))
    generate_parameters_json(backtest_results, architecture, str(run_output))

    # ---- AmiBroker Push ----
    if not args.no_amibroker:
        log("=== AMIBROKER INTEGRATION ===")
        sorted_syms = backtest_results.get("sorted_syms", [])
        afl_str = generate_apex_afl(sorted_syms, backtest_results, architecture)
        push_to_amibroker(backtest_results, afl_str, str(run_output), cfg)
    else:
        log("AmiBroker push skipped (--no-amibroker)")

    # ---- Summary ----
    log("=== PIPELINE COMPLETE ===")
    pstats = backtest_results.get("portfolio_stats", {})
    log("Final Results:")
    log(f"  Symbols: {len(backtest_results.get('sorted_syms', []))}")
    log(f"  Trades:  {pstats.get('trades', 0)}")
    log(f"  PF:      {pstats.get('pf', 0):.2f}")
    log(f"  Win%:    {pstats.get('wr_pct', 0):.1f}%")
    log(f"  Return:  {pstats.get('total_return_pct', 0):.1f}%")
    log(f"  MaxDD:   {pstats.get('max_dd_pct', 0):.1f}%")
    log(f"  Sharpe:  {pstats.get('sharpe', 0):.2f}")
    hstats = backtest_results.get("holdout_universe_stats", {})
    if hstats:
        log("  --- TRUE HOLDOUT (never seen by optimizer) ---")
        log(f"  Holdout Trades:  {hstats.get('trades', 0)}")
        log(f"  Holdout PF:      {hstats.get('pf', 0):.2f}")
        log(f"  Holdout Win%:    {hstats.get('wr_pct', 0):.1f}%")
        log(f"  Holdout Return:  {hstats.get('total_return_pct', 0):.1f}%")
        log(f"  Holdout Sharpe:  {hstats.get('sharpe', 0):.2f}")
    log(f"  Report:  {report_path}")

    try:
        abs_report = str(Path(report_path).resolve()).replace("\\", "/")
        webbrowser.open(f"file:///{abs_report}")
        log(f"Report opened in browser: file:///{abs_report}")
    except Exception as e:
        log(f"Could not open browser: {e}", "WARN")
        log(f"Open manually: {Path(report_path).resolve()}")

    return backtest_results
